"""
test_extraction.py
=====================

Tests for schema + grounding validation and the extract -> validate -> repair loop.
"""

import json

import pytest

from mmrag.budget import TicketBudget
from mmrag.extraction import StructuredExtractor, build_user_prompt, validate_ticket_json
from mmrag.ledger import CostLedger
from mmrag.pricing import ModelPrice, PricingTable
from mmrag.providers.base import ProviderUnavailableError
from mmrag.rag.index import KBChunk, RetrievedChunk
from mmrag.schemas import ExtractionValidationError
from tests.helpers import FailingChat, ScriptedChat, run, valid_record_json

PATTERN = r"^ORD-\d{3,}$"


def _validate(data, *, citations=("kb:a.md#1",), channels=("text",), evidence="order ORD-1042 arrived cracked"):
    return validate_ticket_json(
        data, allowed_citation_ids=set(citations), provided_channels=list(channels),
        evidence_text=evidence, order_id_pattern=PATTERN,
    )


def _problems(data, **kwargs) -> list[str]:
    with pytest.raises(ExtractionValidationError) as caught:
        _validate(data, **kwargs)
    return caught.value.problems


def test_valid_payload_becomes_a_record() -> None:
    record = _validate(valid_record_json())
    assert record.category == "product_defect" and record.order_id == "ORD-1042"


def test_missing_keys_are_reported() -> None:
    data = valid_record_json()
    del data["category"]
    assert any("missing required key 'category'" in p for p in _problems(data))


def test_enum_and_range_violations_are_all_reported_at_once() -> None:
    problems = _problems(valid_record_json(category="weird", urgency="now", confidence=1.7, amount_mentioned=-5))
    text = " | ".join(problems)
    assert "category" in text and "urgency" in text and "confidence" in text and "amount_mentioned" in text


def test_fabricated_order_id_is_rejected_even_if_well_formed() -> None:
    problems = _problems(valid_record_json(order_id="ORD-9999"))  # well-formed but absent from the evidence
    assert any("does not appear in the ticket evidence" in p for p in problems)


def test_malformed_order_id_is_rejected() -> None:
    assert any("does not match pattern" in p for p in _problems(valid_record_json(order_id="1042")))


def test_fabricated_citations_are_rejected() -> None:
    problems = _problems(valid_record_json(citations=["kb:invented.md#9"]))
    assert any("fabricated" in p for p in problems)
    assert _validate(valid_record_json(citations=["kb:a.md#1"])).citations == ["kb:a.md#1"]


def test_evidence_modalities_must_be_provided_channels() -> None:
    problems = _problems(valid_record_json(evidence_modalities=["voice"]), channels=("text",))
    assert any("was not provided" in p for p in problems)
    assert any("unknown channel" in p for p in _problems(valid_record_json(evidence_modalities=["smell"])))


def test_bool_is_not_accepted_as_a_number() -> None:
    assert _problems(valid_record_json(confidence=True)) and _problems(valid_record_json(amount_mentioned=True))


def test_summary_length_limit() -> None:
    assert any("300" in p for p in _problems(valid_record_json(issue_summary="x" * 301)))


# ---- the extract/repair loop ----

def _extractor(chat, max_repairs=1):
    pricing = PricingTable({"scripted": ModelPrice(input_per_million=1.0, output_per_million=2.0)})
    ledger = CostLedger()
    return StructuredExtractor(chat, pricing, ledger, PATTERN, max_repairs), ledger


def _extract(extractor, budget=None):
    chunk = KBChunk("kb:a.md#1", "Damaged items get a refund.", "a.md", "Damaged items", "text")
    return run(
        extractor.extract(
            ticket_id="t1", text="order ORD-1042 arrived cracked", transcript="", image_text="",
            retrieved=[RetrievedChunk(chunk, 0.5)], provided_channels=["text"],
            budget=budget or TicketBudget(1.0),
        )
    )


def test_first_attempt_success_records_one_extraction_event() -> None:
    extractor, ledger = _extractor(ScriptedChat([json.dumps(valid_record_json())]))
    outcome = _extract(extractor)
    assert outcome.status == "ok" and outcome.attempts == 1
    assert [e.stage for e in ledger.events_for("t1")] == ["extraction"]


def test_invalid_output_triggers_one_targeted_repair() -> None:
    bad = json.dumps(valid_record_json(category="weird"))
    chat = ScriptedChat([bad, json.dumps(valid_record_json())])
    extractor, ledger = _extractor(chat)
    outcome = _extract(extractor)
    assert outcome.status == "ok" and outcome.attempts == 2
    assert [e.stage for e in ledger.events_for("t1")] == ["extraction", "repair"]  # the retry is paid for and visible
    assert "failed validation" in chat.prompts[1] and "category" in chat.prompts[1]  # problems fed back verbatim


def test_non_json_output_is_repaired_too() -> None:
    chat = ScriptedChat(["Sorry, I cannot do that.", json.dumps(valid_record_json())])
    extractor, _ = _extractor(chat)
    assert _extract(extractor).status == "ok"


def test_gives_up_after_max_repairs_without_a_record() -> None:
    bad = json.dumps(valid_record_json(category="weird"))
    extractor, ledger = _extractor(ScriptedChat([bad, bad]), max_repairs=1)
    outcome = _extract(extractor)
    assert outcome.status == "extraction_failed" and outcome.record is None and outcome.attempts == 2
    assert outcome.problems


def test_budget_exceeded_stops_before_spending() -> None:
    extractor, ledger = _extractor(ScriptedChat([json.dumps(valid_record_json())]))
    outcome = _extract(extractor, budget=TicketBudget(max_cost_usd=0.0))
    assert outcome.status == "budget_exceeded" and outcome.attempts == 0
    assert ledger.events_for("t1") == []  # nothing was spent


def test_provider_outage_returns_status_instead_of_raising() -> None:
    class _Down:
        model = "scripted"

        async def complete(self, system, user):
            raise ProviderUnavailableError("all down")

    extractor, _ = _extractor(_Down())
    assert _extract(extractor).status == "provider_unavailable"


def test_prompt_wraps_all_channels_as_untrusted_and_lists_context() -> None:
    chunk = KBChunk("kb:a.md#1", "Damaged items get a refund.", "a.md", "Damaged items", "text")
    prompt = build_user_prompt(
        text="typed", transcript="spoken", image_text="pixels",
        retrieved=[RetrievedChunk(chunk, 0.5)], provided_channels=["text", "voice", "image"],
    )
    for tag in ("untrusted_text", "untrusted_voice_transcript", "untrusted_image_analysis"):
        assert f"<{tag}>" in prompt and f"</{tag}>" in prompt
    assert "[kb:a.md#1]" in prompt and "Provided channels: text, voice, image" in prompt
