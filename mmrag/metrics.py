"""
mmrag/metrics.py
===================

Aggregations behind the live dashboard: cost per ticket, latency
percentiles, cost by stage and by channel, token/audio totals, and the
share of cost that was priced with a default (estimated) price.

Pure functions over a CostLedger — no I/O, trivially testable.
"""

from mmrag.ledger import CostLedger


def percentile(values: list[float], pct: float) -> float:
    """
    Linear-interpolated percentile (pct in [0, 100]). Returns 0.0 for empty input.

    Tail percentiles (p95) matter more than averages for latency (Week 8 part 1, section 8).
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def compute_metrics(ledger: CostLedger, recent: int = 15, series_points: int = 50) -> dict:
    """
    Compute every dashboard figure from the ledger.

    Args:
        ledger: The cost ledger.
        recent: How many recent tickets to include in the table.
        series_points: How many recent per-ticket costs to include in the cost series.

    Returns:
        A JSON-serializable dict (see keys below). Safe on an empty ledger.
    """
    summaries = ledger.summaries()
    events = ledger.all_events()
    total_cost = sum(s.cost_usd for s in summaries)
    count = len(summaries)

    status_counts: dict[str, int] = {}
    for summary in summaries:
        status_counts[summary.status] = status_counts.get(summary.status, 0) + 1

    cost_by_stage: dict[str, float] = {}
    cost_by_channel: dict[str, float] = {}
    input_tokens = output_tokens = 0
    audio_seconds = 0.0
    estimated_cost = 0.0
    for event in events:
        cost_by_stage[event.stage] = cost_by_stage.get(event.stage, 0.0) + event.cost_usd
        cost_by_channel[event.channel] = cost_by_channel.get(event.channel, 0.0) + event.cost_usd
        input_tokens += event.usage.input_tokens
        output_tokens += event.usage.output_tokens
        audio_seconds += event.usage.audio_seconds
        if event.estimated_price:
            estimated_cost += event.cost_usd

    review_count = sum(1 for s in summaries if s.needs_human_review)
    costs = [s.cost_usd for s in summaries]
    latencies = [s.latency_ms for s in summaries]

    recent_rows = [
        {
            "ticket_id": s.ticket_id,
            "status": s.status,
            "channels": list(s.channels),
            "cost_usd": round(s.cost_usd, 6),
            "latency_ms": round(s.latency_ms, 1),
            "category": s.category,
            "urgency": s.urgency,
            "needs_human_review": s.needs_human_review,
            "flags": list(s.flags),
            "created_at": s.created_at,
        }
        for s in reversed(summaries[-recent:])
    ]

    return {
        "tickets_total": count,
        "status_counts": status_counts,
        "needs_review_rate": round(review_count / count, 4) if count else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_ticket_usd": round(total_cost / count, 6) if count else 0.0,
        "p50_cost_usd": round(percentile(costs, 50), 6),
        "p95_cost_usd": round(percentile(costs, 95), 6),
        "p50_latency_ms": round(percentile(latencies, 50), 1),
        "p95_latency_ms": round(percentile(latencies, 95), 1),
        "cost_by_stage": {k: round(v, 6) for k, v in sorted(cost_by_stage.items())},
        "cost_by_channel": {k: round(v, 6) for k, v in sorted(cost_by_channel.items())},
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "audio_seconds": round(audio_seconds, 2),
        },
        "estimated_price_share": round(estimated_cost / total_cost, 4) if total_cost > 0 else 0.0,
        "cost_series": [round(c, 6) for c in costs[-series_points:]],
        "recent_tickets": recent_rows,
    }
