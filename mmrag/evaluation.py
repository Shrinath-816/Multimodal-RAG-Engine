"""
mmrag/evaluation.py
======================

A small evaluation harness (the Week 5 discipline, applied to the extraction
task): score extracted records against a golden set, field by field, and
report accuracy next to cost per ticket. Seed for the capstone's eval suite.

Field-level scoring (not one blended number) shows WHICH field regressed —
"urgency got worse after the prompt change" is actionable; "accuracy fell 3%"
is not. Exact-match fields only: closed enums and ids are exactly checkable.
"""

from mmrag.schemas import TicketRecord

SCORED_FIELDS = ("category", "urgency", "order_id", "requested_resolution")


def score_record(record: TicketRecord | None, expected: dict) -> dict:
    """
    Compare one extracted record to its golden expectation.

    Args:
        record: The extracted record, or None if extraction failed
            (every expected field then counts as wrong — a failure is not a pass).
        expected: Golden values; only keys present are scored.

    Returns:
        field -> bool for each scored field present in `expected`.
    """
    scores = {}
    for name in SCORED_FIELDS:
        if name in expected:
            scores[name] = record is not None and getattr(record, name) == expected[name]
    return scores


def aggregate_scores(per_ticket: list[dict]) -> dict:
    """
    Aggregate per-ticket field scores.

    Returns:
        {"per_field": {field: accuracy}, "overall": accuracy over all scored
        fields, "tickets": n}. Empty input -> zeros.
    """
    totals: dict[str, list[int]] = {}
    for scores in per_ticket:
        for name, correct in scores.items():
            bucket = totals.setdefault(name, [0, 0])
            bucket[0] += 1 if correct else 0
            bucket[1] += 1
    per_field = {name: (c / n if n else 0.0) for name, (c, n) in sorted(totals.items())}
    all_correct = sum(c for c, _n in totals.values())
    all_total = sum(n for _c, n in totals.values())
    return {
        "per_field": {k: round(v, 4) for k, v in per_field.items()},
        "overall": round(all_correct / all_total, 4) if all_total else 0.0,
        "tickets": len(per_ticket),
    }
