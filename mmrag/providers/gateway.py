"""
mmrag/providers/gateway.py
=============================

The resilience layer: timeout, bounded retry, and ordered fallback across
candidate models for every provider call (Week 8 part 3, sections 7-9).

Notable decisions:
- The EMBEDDER gets retries but NO fallback candidates (see factory.py):
  query vectors must come from the same model that embedded the corpus, or
  similarity scores become meaningless (the Week 2 invariant).
- Counters (calls, retries, fallbacks, failures) are exposed on the dashboard
  — a fallback or retry nobody can see is indistinguishable from one that is
  silently broken (Week 8 part 3, section 20).
"""

import asyncio
from dataclasses import dataclass

from mmrag.providers.base import ChatResult, EmbedResult, ProviderUnavailableError, TranscriptResult


@dataclass
class ReliabilityCounters:
    """Process-wide counters of resilience events, surfaced on the dashboard."""

    calls: int = 0
    retries: int = 0
    fallbacks: int = 0
    failures: int = 0

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "fallbacks": self.fallbacks,
            "failures": self.failures,
        }


class Resilient:
    """
    Calls `method` on each candidate in order, with a per-attempt timeout and
    bounded retries per candidate, falling through to the next candidate on exhaustion.
    """

    def __init__(
        self,
        candidates: list,
        *,
        timeout: float,
        retries: int,
        backoff: float,
        counters: ReliabilityCounters,
    ) -> None:
        if not candidates:
            raise ValueError("Resilient requires at least one candidate.")
        self._candidates = list(candidates)
        self._timeout = timeout
        self._retries = retries
        self._backoff = backoff
        self._counters = counters

    @property
    def model(self) -> str:
        """The PRIMARY candidate's model id."""
        return self._candidates[0].model

    async def call(self, method: str, *args, **kwargs):
        """
        Invoke `method` with resilience.

        Raises:
            ProviderUnavailableError: Every candidate failed (after retries) or timed out.
        """
        self._counters.calls += 1
        last_error: Exception | None = None

        for index, candidate in enumerate(self._candidates):
            if index > 0:
                self._counters.fallbacks += 1
            for attempt in range(self._retries + 1):
                try:
                    return await asyncio.wait_for(
                        getattr(candidate, method)(*args, **kwargs), timeout=self._timeout
                    )
                except Exception as exc:  # noqa: BLE001 - any failure triggers retry/fallback
                    last_error = exc
                    if attempt < self._retries:
                        self._counters.retries += 1
                        await asyncio.sleep(self._backoff)

        self._counters.failures += 1
        raise ProviderUnavailableError(
            f"All {len(self._candidates)} candidate(s) failed for '{method}': {last_error!r}"
        ) from last_error


class ResilientChat(Resilient):
    """Resilient text completion."""

    async def complete(self, system: str, user: str) -> ChatResult:
        return await self.call("complete", system, user)


class ResilientVision(Resilient):
    """Resilient image understanding."""

    async def describe(self, image_data_url: str, instruction: str) -> ChatResult:
        return await self.call("describe", image_data_url, instruction)


class ResilientEmbedder(Resilient):
    """Resilient embedding — construct with a SINGLE candidate (no cross-model fallback)."""

    async def embed(self, texts: list[str]) -> EmbedResult:
        return await self.call("embed", texts)


class ResilientSTT(Resilient):
    """Resilient speech-to-text."""

    async def transcribe(self, audio: bytes, filename: str, mime: str) -> TranscriptResult:
        return await self.call("transcribe", audio, filename, mime)
