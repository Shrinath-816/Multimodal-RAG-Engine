"""
test_providers.py
====================

Tests for the resilience layer (timeout, retry, fallback, counters) and for the
OpenAI-compatible adapters, using fake SDK clients that verify the exact
request shapes sent over the wire (image_url content part, whisper file tuple, ...).
"""

import asyncio
from types import SimpleNamespace

import pytest

from mmrag.providers.base import ChatResult, ProviderUnavailableError
from mmrag.providers.gateway import ReliabilityCounters, ResilientChat
from mmrag.providers.openai_compat import (
    OpenAICompatChat,
    OpenAICompatEmbedder,
    OpenAICompatVision,
    OpenAIWhisperSTT,
)
from mmrag.schemas import Usage
from tests.helpers import FailingChat, run


class _OkChat:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0

    async def complete(self, system: str, user: str) -> ChatResult:
        self.calls += 1
        return ChatResult("ok", Usage(1, 1), self.model)


class _FlakyChat:
    """Fails the first N calls, then succeeds."""

    model = "flaky"

    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.calls = 0

    async def complete(self, system: str, user: str) -> ChatResult:
        self.calls += 1
        if self.calls <= self._failures:
            raise RuntimeError("transient")
        return ChatResult("ok", Usage(1, 1), self.model)


class _SlowChat:
    model = "slow"

    async def complete(self, system: str, user: str) -> ChatResult:
        await asyncio.sleep(1.0)
        return ChatResult("late", Usage(), self.model)


def _resilient(candidates, retries=0, timeout=1.0):
    counters = ReliabilityCounters()
    return ResilientChat(candidates, timeout=timeout, retries=retries, backoff=0.0, counters=counters), counters


def test_primary_success_uses_no_fallback() -> None:
    primary, backup = _OkChat("primary"), _OkChat("backup")
    chat, counters = _resilient([primary, backup])
    result = run(chat.complete("s", "u"))
    assert result.model == "primary" and backup.calls == 0
    assert counters.fallbacks == 0 and counters.calls == 1


def test_falls_back_when_primary_fails_and_reports_actual_model() -> None:
    chat, counters = _resilient([FailingChat(), _OkChat("backup")])
    result = run(chat.complete("s", "u"))
    assert result.model == "backup"  # the ledger must price the model that actually answered
    assert counters.fallbacks == 1


def test_retries_transient_failures_before_falling_back() -> None:
    flaky = _FlakyChat(failures=1)
    chat, counters = _resilient([flaky, _OkChat("backup")], retries=1)
    result = run(chat.complete("s", "u"))
    assert result.model == "flaky" and flaky.calls == 2
    assert counters.retries == 1 and counters.fallbacks == 0


def test_raises_when_every_candidate_fails() -> None:
    chat, counters = _resilient([FailingChat(), FailingChat()])
    with pytest.raises(ProviderUnavailableError):
        run(chat.complete("s", "u"))
    assert counters.failures == 1


def test_timeout_triggers_fallback() -> None:
    chat, counters = _resilient([_SlowChat(), _OkChat("backup")], timeout=0.05)
    assert run(chat.complete("s", "u")).model == "backup"


# ---- OpenAI-compatible adapters with fake SDK clients ----

def _completion(text: str, prompt_tokens=12, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _FakeChatClient:
    def __init__(self, response) -> None:
        self.captured: dict = {}

        async def create(**kwargs):
            self.captured = kwargs
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_chat_adapter_builds_messages_and_parses_usage() -> None:
    client = _FakeChatClient(_completion("hello"))
    result = run(OpenAICompatChat(client, "m1").complete("SYS", "USER"))
    assert client.captured["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert client.captured["model"] == "m1"
    assert (result.text, result.usage.input_tokens, result.usage.output_tokens) == ("hello", 12, 5)


def test_vision_adapter_sends_image_url_content_part() -> None:
    client = _FakeChatClient(_completion("{}"))
    run(OpenAICompatVision(client, "vlm").describe("data:image/png;base64,AAAA", "describe it"))
    content = client.captured["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe it"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_embedder_adapter_reorders_by_index_and_reads_usage() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=1, embedding=[2.0]), SimpleNamespace(index=0, embedding=[1.0])],
        usage=SimpleNamespace(prompt_tokens=9),
    )

    async def create(**kwargs):
        return response

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    result = run(OpenAICompatEmbedder(client, "emb").embed(["a", "b"]))
    assert result.vectors == [[1.0], [2.0]] and result.usage.input_tokens == 9


def test_whisper_adapter_uploads_file_tuple_and_reports_audio_seconds() -> None:
    captured: dict = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="hello world")

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    from mmrag.audio_utils import pcm_to_wav
    from mmrag.demo_media import make_pcm_silence

    wav = pcm_to_wav(make_pcm_silence(2000))
    result = run(OpenAIWhisperSTT(client, "whisper-1").transcribe(wav, "call.wav", "audio/wav"))
    assert captured["file"] == ("call.wav", wav, "audio/wav") and captured["model"] == "whisper-1"
    assert result.text == "hello world"
    assert result.usage.audio_seconds == pytest.approx(2.0)
