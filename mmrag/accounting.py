"""
mmrag/accounting.py
======================

One function that every paid call goes through: price it, commit it to the
ticket's budget, and append it to the ledger. Centralizing this guarantees no
stage can spend money without being counted — the cost-per-ticket figure is
only trustworthy if there is no side door.
"""

from mmrag.budget import TicketBudget
from mmrag.ledger import CostEvent, CostLedger
from mmrag.pricing import PricingError, PricingTable
from mmrag.schemas import Usage


def record_paid_call(
    *,
    ledger: CostLedger,
    pricing: PricingTable,
    budget: TicketBudget,
    ticket_id: str,
    stage: str,
    channel: str,
    model: str,
    usage: Usage,
    latency_ms: float,
) -> CostEvent:
    """
    Price one completed provider call and record it.

    If the model has no price (and the table has no default), the call is still
    recorded — at $0 but flagged `estimated_price=True` — because dropping the
    event would hide real usage. The dashboard's "estimated share" surfaces it.

    Returns:
        The recorded CostEvent.
    """
    try:
        cost, estimated = pricing.cost_for(model, usage)
    except PricingError:
        cost, estimated = 0.0, True

    budget.commit(cost)
    event = CostEvent(
        ticket_id=ticket_id,
        stage=stage,
        channel=channel,
        model=model,
        usage=usage,
        cost_usd=cost,
        latency_ms=latency_ms,
        estimated_price=estimated,
    )
    ledger.record_event(event)
    return event
