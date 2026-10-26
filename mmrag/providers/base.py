"""
mmrag/providers/base.py
==========================

Provider interfaces and result types.

Each result carries `usage` (metered consumption) and `model` (the model that
ACTUALLY served the call) so the cost ledger prices the call correctly even
when a fallback model answered.
"""

from dataclasses import dataclass
from typing import Protocol

from mmrag.schemas import Usage


class ProviderUnavailableError(Exception):
    """Raised by the resilience layer when every candidate failed or timed out."""


@dataclass
class ChatResult:
    """Output of a text or vision completion."""

    text: str
    usage: Usage
    model: str


@dataclass
class EmbedResult:
    """Output of an embedding call: one vector per input text."""

    vectors: list[list[float]]
    usage: Usage
    model: str


@dataclass
class TranscriptResult:
    """Output of speech-to-text."""

    text: str
    usage: Usage
    model: str


class ChatModel(Protocol):
    """Text completion: (system, user) -> text."""

    model: str

    async def complete(self, system: str, user: str) -> ChatResult: ...


class VisionModel(Protocol):
    """Image understanding: (image data URL, instruction) -> text."""

    model: str

    async def describe(self, image_data_url: str, instruction: str) -> ChatResult: ...


class Embedder(Protocol):
    """Text embedding."""

    model: str

    async def embed(self, texts: list[str]) -> EmbedResult: ...


class SpeechToText(Protocol):
    """Speech-to-text over a complete audio payload."""

    model: str

    async def transcribe(self, audio: bytes, filename: str, mime: str) -> TranscriptResult: ...
