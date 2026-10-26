"""
mmrag/pipeline.py
====================

THE ORCHESTRATOR: one ticket in -> one TicketResult out, with every channel,
retrieval, extraction, and cost event accounted for.

Request lifecycle:

    validate input
      -> (text) injection scan + PII redaction
      -> (image, voice) analyze CONCURRENTLY       <- independent channels overlap:
                                                       latency = max, cost = sum
      -> assemble evidence + retrieval query
      -> hybrid retrieval over the multimodal knowledge index
      -> grounded structured extraction (validate, repair, budget-checked)
      -> deterministic review gating
      -> ledger summary

EXPLICIT PARTIAL-FAILURE POLICY (Week 8 part 2, section 13 — decided in
advance, not discovered in production):

- A channel failing does NOT fail the ticket if other evidence exists; the
  ticket is marked DEGRADED and forced to human review (its evidence is incomplete).
- No usable evidence at all -> "failed" (provider outage / empty input) or
  "rejected" (every input was invalid; nothing was spent).
- Retrieval unavailable -> proceed ungrounded, flagged, forced to review.
- Extraction cannot produce a valid record -> "needs_human_review" with NO
  record (never a silently wrong one).

REVIEW GATING is deterministic and does not trust the model's self-report:
low confidence, any injection flag, or a degraded ticket always routes to a human.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass

from mmrag.accounting import record_paid_call
from mmrag.budget import TicketBudget
from mmrag.config import Settings
from mmrag.extraction import StructuredExtractor
from mmrag.ingest.errors import MediaError
from mmrag.ingest.image import ImageChannel
from mmrag.ingest.voice import VoiceChannel
from mmrag.ledger import CostLedger, TicketSummary
from mmrag.pricing import PricingTable
from mmrag.providers.base import ProviderUnavailableError
from mmrag.providers.gateway import ReliabilityCounters
from mmrag.rag.index import KnowledgeIndex
from mmrag.schemas import (
    CHANNEL_IMAGE,
    CHANNEL_SHARED,
    CHANNEL_TEXT,
    CHANNEL_VOICE,
    RetrievedRef,
    TicketResult,
)
from mmrag.security import redact_pii, scan_for_injection

_MAX_QUERY_CHARS = 1500


@dataclass
class TicketInput:
    """
    One incoming ticket. At least one of text/image/audio must be present.

    Attributes:
        text: Typed ticket text.
        image: Raw image bytes (screenshot/photo).
        audio: Raw audio bytes (a voicemail/call segment).
        audio_filename / audio_mime: Passed to the STT service.
        ticket_id: Optional caller-supplied id (generated if omitted).
    """

    text: str | None = None
    image: bytes | None = None
    audio: bytes | None = None
    audio_filename: str = "audio.wav"
    audio_mime: str = "audio/wav"
    ticket_id: str | None = None


def _merge(into: dict, other: dict) -> None:
    for key, value in other.items():
        into[key] = into.get(key, 0) + value


class TicketPipeline:
    """Processes tickets end to end. Holds no per-ticket state, so it is safe to share across concurrent requests."""

    def __init__(
        self,
        *,
        settings: Settings,
        pricing: PricingTable,
        ledger: CostLedger,
        index: KnowledgeIndex,
        image_channel: ImageChannel,
        voice_channel: VoiceChannel,
        extractor: StructuredExtractor,
        counters: ReliabilityCounters,
    ) -> None:
        self._settings = settings
        self._pricing = pricing
        self._ledger = ledger
        self._index = index
        self._image_channel = image_channel
        self._voice_channel = voice_channel
        self._extractor = extractor
        self._counters = counters
        self.rejected_total = 0  # invalid-input tickets; they consume nothing so they are not in the ledger

    async def _run_image(self, ticket_id: str, data: bytes, budget: TicketBudget):
        analysis = await self._image_channel.analyze(data)
        record_paid_call(
            ledger=self._ledger, pricing=self._pricing, budget=budget, ticket_id=ticket_id, stage="vision",
            channel=CHANNEL_IMAGE, model=analysis.model, usage=analysis.usage, latency_ms=analysis.latency_ms,
        )
        return analysis

    async def _run_voice(self, ticket_id: str, ticket: TicketInput, budget: TicketBudget):
        transcript = await self._voice_channel.transcribe(ticket.audio, ticket.audio_filename, ticket.audio_mime)
        record_paid_call(
            ledger=self._ledger, pricing=self._pricing, budget=budget, ticket_id=ticket_id, stage="stt",
            channel=CHANNEL_VOICE, model=transcript.model, usage=transcript.usage, latency_ms=transcript.latency_ms,
        )
        return transcript

    async def process(self, ticket: TicketInput) -> TicketResult:
        """
        Process one ticket.

        Returns:
            A TicketResult. This method does not raise for input, provider,
            or model problems — each is mapped to a status and flags.
        """
        started = time.perf_counter()
        ticket_id = ticket.ticket_id or f"T-{uuid.uuid4().hex[:10]}"
        has_text = bool(ticket.text and ticket.text.strip())

        if not (has_text or ticket.image or ticket.audio):
            self.rejected_total += 1
            return TicketResult(
                ticket_id=ticket_id, status="rejected",
                problems=["No input provided: send text, an image, audio, or a combination."],
            )

        budget = TicketBudget(self._settings.max_ticket_cost_usd)
        flags: list[str] = []
        problems: list[str] = []
        pii_total: dict = {}
        channels_used: list[str] = []
        injection_flagged = False
        degraded = False

        # --- text channel (cheap, synchronous) --------------------------------
        safe_text = ""
        if has_text:
            if scan_for_injection(ticket.text):
                injection_flagged = True
            safe_text, counts = redact_pii(ticket.text.strip())
            _merge(pii_total, counts)
            channels_used.append(CHANNEL_TEXT)

        # --- media channels, concurrently ---------------------------------------
        jobs: list[tuple[str, object]] = []
        if ticket.image:
            jobs.append((CHANNEL_IMAGE, self._run_image(ticket_id, ticket.image, budget)))
        if ticket.audio:
            jobs.append((CHANNEL_VOICE, self._run_voice(ticket_id, ticket, budget)))
        outcomes = await asyncio.gather(*(job for _label, job in jobs), return_exceptions=True) if jobs else []

        transcript_text = ""
        image_text = ""
        image_visible = ""
        media_failures = 0
        media_client_errors = 0
        for (label, _job), outcome in zip(jobs, outcomes):
            if isinstance(outcome, MediaError):
                problems.append(f"{label} input rejected: {outcome}")
                media_failures += 1
                media_client_errors += 1
                degraded = True
            elif isinstance(outcome, ProviderUnavailableError):
                problems.append(f"{label} channel unavailable: {outcome}")
                media_failures += 1
                degraded = True
            elif isinstance(outcome, Exception):
                problems.append(f"{label} channel failed unexpectedly: {outcome!r}")
                media_failures += 1
                degraded = True
            elif label == CHANNEL_VOICE:
                if not outcome.text:
                    problems.append("voice channel produced an empty transcript (no speech recognized)")
                    media_failures += 1
                    degraded = True
                else:
                    transcript_text = outcome.text
                    injection_flagged = injection_flagged or outcome.injection_flagged
                    _merge(pii_total, outcome.pii_redactions)
                    channels_used.append(CHANNEL_VOICE)
            else:  # image
                image_visible = outcome.visible_text
                entities = ", ".join(f"{k}={v}" for k, v in outcome.entities.items())
                image_text = (
                    f"{outcome.description}\nVisible text: {outcome.visible_text}"
                    + (f"\nEntities: {entities}" if entities else "")
                )
                injection_flagged = injection_flagged or outcome.injection_flagged
                _merge(pii_total, outcome.pii_redactions)
                channels_used.append(CHANNEL_IMAGE)

        # --- no evidence at all ---------------------------------------------------
        if not (safe_text or transcript_text or image_text):
            all_client_errors = media_failures > 0 and media_failures == media_client_errors and not has_text
            if all_client_errors:
                self.rejected_total += 1
                return TicketResult(ticket_id=ticket_id, status="rejected", problems=problems)
            return self._finish(
                ticket_id, started, "failed", None, channels_used, [], flags + ["no_usable_evidence"], problems,
                budget_ok=True,
            )

        if degraded:
            flags.append("degraded")
        if injection_flagged:
            flags.append("injection_flagged")
        if pii_total:
            flags.append("pii_redacted")

        # --- retrieval --------------------------------------------------------------
        query = " ".join(p for p in (safe_text, transcript_text, image_visible, image_text) if p)[:_MAX_QUERY_CHARS]
        retrieved = []
        try:
            search = await self._index.search(query, self._settings.top_k)
            record_paid_call(
                ledger=self._ledger, pricing=self._pricing, budget=budget, ticket_id=ticket_id, stage="embedding",
                channel=CHANNEL_SHARED, model=search.model, usage=search.usage, latency_ms=search.latency_ms,
            )
            retrieved = search.results
        except ProviderUnavailableError as exc:
            problems.append(f"retrieval unavailable, proceeding ungrounded: {exc}")
            flags.append("retrieval_unavailable")
            degraded = True

        # --- extraction ----------------------------------------------------------------
        outcome = await self._extractor.extract(
            ticket_id=ticket_id, text=safe_text, transcript=transcript_text, image_text=image_text,
            retrieved=retrieved, provided_channels=list(channels_used), budget=budget,
        )
        refs = [
            RetrievedRef(r.chunk.chunk_id, r.chunk.source, r.chunk.section, r.chunk.modality, round(r.score, 5))
            for r in retrieved
        ]

        if outcome.status != "ok":
            flags.append(outcome.status)
            problems.extend(outcome.problems)
            status = "failed" if outcome.status == "provider_unavailable" else "needs_human_review"
            return self._finish(ticket_id, started, status, None, channels_used, refs, flags, problems, budget_ok=True)

        record = outcome.record
        if record.confidence < self._settings.review_confidence_threshold:
            flags.append("low_confidence")
        force_review = "low_confidence" in flags or injection_flagged or degraded
        record.needs_human_review = record.needs_human_review or force_review
        status = "needs_human_review" if record.needs_human_review else "ok"
        return self._finish(ticket_id, started, status, record, channels_used, refs, flags, problems, budget_ok=True)

    def _finish(
        self, ticket_id, started, status, record, channels_used, refs, flags, problems, *, budget_ok: bool
    ) -> TicketResult:
        """Total up the ticket's cost from the ledger, write its summary, and build the result."""
        events = self._ledger.events_for(ticket_id)
        breakdown: dict = {}
        for event in events:
            breakdown[event.stage] = round(breakdown.get(event.stage, 0.0) + event.cost_usd, 8)
        cost = round(sum(e.cost_usd for e in events), 8)
        estimated = any(e.estimated_price for e in events)
        latency_ms = (time.perf_counter() - started) * 1000

        self._ledger.record_ticket(
            TicketSummary(
                ticket_id=ticket_id, status=status, channels=list(channels_used), cost_usd=cost,
                latency_ms=latency_ms, category=record.category if record else None,
                urgency=record.urgency if record else None,
                needs_human_review=(record.needs_human_review if record else True),
                flags=list(flags), estimated_price=estimated,
            )
        )
        return TicketResult(
            ticket_id=ticket_id, status=status, record=record, channels_used=list(channels_used),
            retrieved=list(refs), cost_usd=cost, latency_ms=latency_ms, cost_breakdown=breakdown,
            flags=list(flags), problems=list(problems), estimated_price=estimated,
        )
