"""
mmrag/ledger.py
==================

The cost ledger: an append-only record of every paid provider call
(CostEvent) plus one TicketSummary per processed ticket. This is the data
behind the COST-PER-TICKET dashboard.

Cost is recorded at the granularity failures and optimizations happen at —
per STAGE (stt, vision, embedding, extraction, repair) and per CHANNEL —
so "why did this ticket cost 4x the average?" is answerable from data
(Week 8 part 3, section 20: attribute cost at the granularity you isolate failures).

In-memory with a retention bound (oldest tickets are evicted) — fine for a
single process / demo. A production deployment would write events to a
durable store (and/or emit them as OpenTelemetry spans, Week 5).
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field

from mmrag.schemas import Usage


@dataclass
class CostEvent:
    """
    One paid provider call.

    Attributes:
        ticket_id: The ticket this call served.
        stage: "stt" | "vision" | "embedding" | "extraction" | "repair".
        channel: "voice" | "image" | "text" | "shared" (see mmrag.schemas).
        model: The model id that ACTUALLY served the call (after fallback).
        usage: Metered usage reported by the provider.
        cost_usd: Computed cost.
        latency_ms: Wall-clock latency of the call.
        estimated_price: True if priced with the default (unknown-model) price.
        timestamp: Unix time the event was recorded.
    """

    ticket_id: str
    stage: str
    channel: str
    model: str
    usage: Usage
    cost_usd: float
    latency_ms: float
    estimated_price: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class TicketSummary:
    """
    One row per processed ticket — what the dashboard's table and
    aggregates are built from.

    Attributes:
        ticket_id: Unique id.
        status: "ok" | "needs_human_review" | "failed".
        channels: Channels that contributed evidence.
        cost_usd: Total ticket cost.
        latency_ms: End-to-end latency.
        category / urgency: From the extracted record (None if extraction failed).
        needs_human_review: Final review flag.
        flags: Reasons (e.g. "injection_flagged").
        estimated_price: Any stage priced with the default price.
        created_at: Unix time the ticket finished.
    """

    ticket_id: str
    status: str
    channels: list
    cost_usd: float
    latency_ms: float
    category: str | None = None
    urgency: str | None = None
    needs_human_review: bool = False
    flags: list = field(default_factory=list)
    estimated_price: bool = False
    created_at: float = field(default_factory=time.time)


class CostLedger:
    """Append-only event log + per-ticket summaries, with a retention bound."""

    def __init__(self, max_tickets: int = 5000) -> None:
        """
        Args:
            max_tickets: Summaries kept before the OLDEST ticket (and its
                events) is evicted. Bounds memory for long-running processes.
        """
        self._max_tickets = max_tickets
        self._events: dict[str, list[CostEvent]] = {}
        self._summaries: "OrderedDict[str, TicketSummary]" = OrderedDict()

    def record_event(self, event: CostEvent) -> None:
        """Append one paid-call event."""
        self._events.setdefault(event.ticket_id, []).append(event)

    def record_ticket(self, summary: TicketSummary) -> None:
        """Record a finished ticket's summary, evicting the oldest if over the bound."""
        self._summaries[summary.ticket_id] = summary
        while len(self._summaries) > self._max_tickets:
            oldest_id, _ = self._summaries.popitem(last=False)
            self._events.pop(oldest_id, None)

    def events_for(self, ticket_id: str) -> list[CostEvent]:
        """Every event recorded for one ticket (empty list if none)."""
        return list(self._events.get(ticket_id, []))

    def summaries(self) -> list[TicketSummary]:
        """All retained ticket summaries, oldest first."""
        return list(self._summaries.values())

    def all_events(self) -> list[CostEvent]:
        """Events for every retained (summarized) ticket."""
        events: list[CostEvent] = []
        for ticket_id in self._summaries:
            events.extend(self._events.get(ticket_id, []))
        return events

    def ticket_count(self) -> int:
        """Number of retained ticket summaries."""
        return len(self._summaries)
