"""
tests/helpers.py
===================

Shared test utilities: paths, settings factory, an async runner (so tests are
plain sync functions and need no pytest plugin), and scripted/failing provider doubles.
"""

import asyncio
import dataclasses
import os

from mmrag.config import Settings
from mmrag.providers.base import ChatResult, TranscriptResult
from mmrag.schemas import Usage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "data", "knowledge_base")
PRICING_PATH = os.path.join(ROOT, "config", "pricing.example.json")
GOLDEN_PATH = os.path.join(ROOT, "data", "golden_tickets.json")


def run(coro):
    """Run a coroutine to completion (each test gets a fresh event loop)."""
    return asyncio.run(coro)


def make_settings(**overrides) -> Settings:
    """Demo-mode settings pointing at the repo's data/config, with zero retry backoff for speed."""
    base = Settings(
        demo_mode=True, kb_dir=KB_DIR, pricing_path=PRICING_PATH, retry_backoff_seconds=0.0, model_retries=0
    )
    return dataclasses.replace(base, **overrides)


class ScriptedChat:
    """A chat double returning pre-scripted responses in order (records every prompt it receives)."""

    model = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, system: str, user: str) -> ChatResult:
        self.prompts.append(user)
        return ChatResult(text=self._responses.pop(0), usage=Usage(input_tokens=100, output_tokens=50), model=self.model)


class FailingSTT:
    """A speech-to-text double that always fails — simulates a provider outage."""

    model = "failing-stt"

    async def transcribe(self, audio: bytes, filename: str, mime: str) -> TranscriptResult:
        raise RuntimeError("stt provider down")


class FailingChat:
    """A chat double that always fails."""

    model = "failing-chat"

    async def complete(self, system: str, user: str) -> ChatResult:
        raise RuntimeError("chat provider down")


def valid_record_json(**overrides) -> dict:
    """A schema-valid extraction payload; override fields to build invalid variants."""
    payload = {
        "category": "product_defect",
        "urgency": "high",
        "customer_sentiment": "negative",
        "requested_resolution": "refund",
        "issue_summary": "Cracked screen on arrival.",
        "order_id": "ORD-1042",
        "amount_mentioned": None,
        "evidence_modalities": ["text"],
        "citations": [],
        "confidence": 0.9,
        "needs_human_review": False,
    }
    payload.update(overrides)
    return payload
