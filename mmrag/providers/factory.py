"""
mmrag/providers/factory.py
=============================

Builds the provider set from Settings: deterministic fakes in DEMO_MODE,
OpenAI-compatible adapters otherwise. The `openai` package is imported
lazily so demo mode and the test suite never require it.
"""

from dataclasses import dataclass

from mmrag.config import Settings
from mmrag.providers.fakes import FakeEmbedder, FakeSTT, FakeVision, RuleBasedChat


@dataclass
class Providers:
    """
    The raw (un-wrapped) provider candidates.

    Attributes:
        embedder: ONE embedder (no fallback — vector-space consistency).
        chat: Priority-ordered chat candidates.
        vision: Priority-ordered vision candidates.
        stt: Priority-ordered speech-to-text candidates.
    """

    embedder: object
    chat: list
    vision: list
    stt: list


def build_providers(settings: Settings) -> Providers:
    """Construct providers for the given settings (fakes if demo_mode)."""
    if settings.demo_mode:
        return Providers(
            embedder=FakeEmbedder(), chat=[RuleBasedChat()], vision=[FakeVision()], stt=[FakeSTT()]
        )

    from openai import AsyncOpenAI  # lazy: not needed in demo mode / tests

    from mmrag.providers.openai_compat import (
        OpenAICompatChat,
        OpenAICompatEmbedder,
        OpenAICompatVision,
        OpenAIWhisperSTT,
    )

    router = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)
    speech = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
    return Providers(
        embedder=OpenAICompatEmbedder(router, settings.embedding_model),
        chat=[OpenAICompatChat(router, model) for model in settings.chat_models],
        vision=[OpenAICompatVision(router, model) for model in settings.vision_models],
        stt=[OpenAIWhisperSTT(speech, settings.stt_model)],
    )
