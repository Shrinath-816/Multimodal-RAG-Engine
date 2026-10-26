"""
mmrag/providers/openai_compat.py
===================================

Adapters for OpenAI-compatible APIs (OpenRouter for chat/vision/embeddings,
OpenAI for Whisper speech-to-text). These are the ONLY classes in the core
that know the wire format — everything else depends on the Protocols in base.py.

Each adapter takes an already-constructed async client (the `openai`
package's AsyncOpenAI) so tests can inject a fake client and verify request
construction and response parsing without any network.

Vision requests use the standard OpenAI-compatible multimodal message shape:
a `content` list with a text part and an `image_url` part whose url is a
base64 data URL.
"""

from mmrag.audio_utils import audio_duration_seconds
from mmrag.providers.base import ChatResult, EmbedResult, TranscriptResult
from mmrag.schemas import Usage


def _usage_from(response) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


class OpenAICompatChat:
    """Text completion via /chat/completions."""

    def __init__(self, client, model: str) -> None:
        self._client = client
        self.model = model

    async def complete(self, system: str, user: str) -> ChatResult:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        return ChatResult(
            text=response.choices[0].message.content or "", usage=_usage_from(response), model=self.model
        )


class OpenAICompatVision:
    """Image understanding via /chat/completions with an image_url content part."""

    def __init__(self, client, model: str) -> None:
        self._client = client
        self.model = model

    async def describe(self, image_data_url: str, instruction: str) -> ChatResult:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            temperature=0,
        )
        return ChatResult(
            text=response.choices[0].message.content or "", usage=_usage_from(response), model=self.model
        )


class OpenAICompatEmbedder:
    """Embeddings via /embeddings (vectors re-ordered by index for safety)."""

    def __init__(self, client, model: str) -> None:
        self._client = client
        self.model = model

    async def embed(self, texts: list[str]) -> EmbedResult:
        response = await self._client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        usage = getattr(response, "usage", None)
        tokens = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        return EmbedResult(
            vectors=[list(item.embedding) for item in ordered],
            usage=Usage(input_tokens=tokens),
            model=self.model,
        )


class OpenAIWhisperSTT:
    """
    Speech-to-text via /audio/transcriptions. Whisper-family models are
    batch-oriented (they transcribe a complete payload); true streaming STT
    would be a different adapter behind the same Protocol.
    """

    def __init__(self, client, model: str) -> None:
        self._client = client
        self.model = model

    async def transcribe(self, audio: bytes, filename: str, mime: str) -> TranscriptResult:
        response = await self._client.audio.transcriptions.create(
            model=self.model, file=(filename, audio, mime)
        )
        seconds, _exact = audio_duration_seconds(audio)
        return TranscriptResult(
            text=getattr(response, "text", "") or "", usage=Usage(audio_seconds=seconds), model=self.model
        )
