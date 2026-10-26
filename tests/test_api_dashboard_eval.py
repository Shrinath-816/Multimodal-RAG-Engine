"""
test_api_dashboard_eval.py
=============================

Tests for the framework-agnostic API handlers, the dashboard page's safety
properties, and the evaluation harness. (The FastAPI adapter in mmrag/main.py is a thin
shell over TicketAPI; it is compile-checked here but needs fastapi to actually run.)
"""

import json
import py_compile
import os

import pytest

from mmrag.audio_utils import DEFAULT_SAMPLE_RATE
from mmrag.bootstrap import build_engine
from mmrag.dashboard import DASHBOARD_HTML
from mmrag.demo_media import make_pcm_silence, make_pcm_tone
from mmrag.evaluation import aggregate_scores, score_record
from mmrag.ingest.voice import VoiceStreamSession
from mmrag.schemas import TicketRecord
from tests.helpers import GOLDEN_PATH, ROOT, make_settings, run


def _engine():
    return run(build_engine(make_settings()))


def test_submit_shapes_response_and_maps_status_to_http() -> None:
    api = _engine().api
    ok = run(api.submit(text="ORD-1042 arrived cracked, refund please."))
    assert ok["http_status"] == 200 and ok["status"] in ("ok", "needs_human_review")
    assert ok["record"]["order_id"] == "ORD-1042" and ok["cost_usd"] > 0
    assert {"chunk_id", "source", "modality", "score"} <= set(ok["retrieved"][0])

    rejected = run(api.submit())
    assert rejected["http_status"] == 422 and rejected["record"] is None


def test_metrics_reflect_submitted_tickets_and_expose_mode_and_reliability() -> None:
    api = _engine().api
    run(api.submit(text="ORD-1042 arrived cracked, refund please."))
    run(api.submit(text="I was charged twice for ORD-2201."))
    metrics = api.metrics()
    assert metrics["tickets_total"] == 2 and metrics["demo_mode"] is True
    assert metrics["avg_cost_per_ticket_usd"] > 0
    assert metrics["reliability"]["calls"] > 0 and metrics["knowledge_base"]["by_modality"]["image"] == 1
    assert len(api.recent_tickets(limit=1)) == 1


def test_a_live_voice_stream_becomes_a_voice_ticket() -> None:
    api = _engine().api
    stream = make_pcm_tone(600) + make_pcm_silence(900)
    utterance = VoiceStreamSession().feed(stream)[0]
    result = run(api.submit_voice_pcm(utterance, DEFAULT_SAMPLE_RATE))
    assert "voice" in result["channels_used"] and result["cost_breakdown"].get("stt", 0) >= 0
    assert "stt" in result["cost_breakdown"]


def test_dashboard_never_uses_innerhtml_and_renders_via_textcontent() -> None:
    """Ticket content is customer/model controlled; the page must render it as text, never as HTML."""
    assert "innerHTML" not in DASHBOARD_HTML and "outerHTML" not in DASHBOARD_HTML
    assert "insertAdjacentHTML" not in DASHBOARD_HTML and "document.write" not in DASHBOARD_HTML
    assert "textContent" in DASHBOARD_HTML
    assert "/api/metrics" in DASHBOARD_HTML and "setInterval" in DASHBOARD_HTML  # it is actually live


def test_web_adapter_compiles() -> None:
    py_compile.compile(os.path.join(ROOT, "mmrag", "main.py"), doraise=True)


def test_score_record_is_field_level_and_failure_counts_as_wrong() -> None:
    record = TicketRecord("billing", "normal", "neutral", "refund", "s", "ORD-1001", None, ["text"], [], 0.9, False)
    expected = {"category": "billing", "urgency": "high", "order_id": "ORD-1001", "requested_resolution": "refund"}
    assert score_record(record, expected) == {
        "category": True, "urgency": False, "order_id": True, "requested_resolution": True,
    }
    assert score_record(None, expected) == {name: False for name in expected}


def test_aggregate_scores() -> None:
    summary = aggregate_scores([{"category": True, "urgency": False}, {"category": True, "urgency": True}])
    assert summary["per_field"] == {"category": 1.0, "urgency": 0.5}
    assert summary["overall"] == 0.75 and summary["tickets"] == 2
    assert aggregate_scores([]) == {"per_field": {}, "overall": 0.0, "tickets": 0}


def test_golden_set_runs_through_the_pipeline_and_scores_above_floor() -> None:
    from mmrag.demo_media import demo_png_for_scenario
    from mmrag.pipeline import TicketInput

    engine = _engine()
    with open(GOLDEN_PATH, encoding="utf-8") as handle:
        golden = json.load(handle)
    all_scores = []
    for entry in golden:
        ticket = TicketInput(
            text=entry.get("text"),
            audio=entry["voice_transcript"].encode() if "voice_transcript" in entry else None,
            image=demo_png_for_scenario(entry["image_scenario"]) if "image_scenario" in entry else None,
        )
        all_scores.append(score_record(run(engine.pipeline.process(ticket)).record, entry["expected"]))
    summary = aggregate_scores(all_scores)
    assert summary["tickets"] == len(golden)
    assert summary["overall"] >= 0.5  # demo-mode floor: validates plumbing, not model quality
