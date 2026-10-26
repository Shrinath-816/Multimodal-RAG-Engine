"""
mmrag/schemas.py
===================

Shared data shapes: usage accounting, the STRUCTURED TICKET RECORD the
system extracts, and the per-ticket result. Plain dataclasses with explicit
validation (see mmrag/extraction.py) instead of a validation framework, so
the core has no third-party dependency.
"""

from dataclasses import dataclass, field

# Channels a piece of evidence (or a cost) can belong to.
CHANNEL_VOICE = "voice"
CHANNEL_IMAGE = "image"
CHANNEL_TEXT = "text"
CHANNEL_SHARED = "shared"  # embedding/extraction costs that serve all channels
EVIDENCE_CHANNELS = (CHANNEL_VOICE, CHANNEL_IMAGE, CHANNEL_TEXT)

# Closed vocabularies for the structured record. A closed enum is something
# CODE can check exactly — better than asking another LLM whether a label is valid.
CATEGORIES = ("billing", "shipping", "product_defect", "technical_issue", "account_access", "other")
URGENCIES = ("low", "normal", "high", "critical")
SENTIMENTS = ("positive", "neutral", "negative")
RESOLUTIONS = ("refund", "replacement", "repair", "information", "escalation", "none")


@dataclass(frozen=True)
class Usage:
    """
    Metered resource consumption of one provider call.

    Attributes:
        input_tokens: Prompt tokens (for vision calls this INCLUDES image tokens,
            as reported by the provider).
        output_tokens: Completion tokens.
        audio_seconds: Seconds of audio processed (speech-to-text is priced by duration).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    audio_seconds: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.audio_seconds + other.audio_seconds,
        )


class ExtractionValidationError(Exception):
    """
    Raised when model output fails schema/grounding validation.

    Attributes:
        problems: Human-readable list of every violation found — fed back to
            the model verbatim in the repair prompt, so the retry is targeted.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class TicketRecord:
    """
    The structured data extracted from one support ticket.

    Attributes:
        category: One of CATEGORIES.
        urgency: One of URGENCIES.
        customer_sentiment: One of SENTIMENTS.
        requested_resolution: One of RESOLUTIONS ("none" if unspecified).
        issue_summary: Short PII-redacted summary (<= 300 chars).
        order_id: Matches the configured pattern AND must literally appear in
            the ticket evidence (a deterministic anti-hallucination check), or None.
        amount_mentioned: A non-negative monetary amount, or None.
        evidence_modalities: Which provided channels the facts came from.
        citations: Knowledge-base chunk ids used; each MUST be an id that was
            actually retrieved (fabricated citations fail validation).
        confidence: Model-reported confidence in [0, 1].
        needs_human_review: Final review flag. The pipeline can force this to
            True (low confidence, injection flag, degraded channels) regardless
            of what the model said about itself.
    """

    category: str
    urgency: str
    customer_sentiment: str
    requested_resolution: str
    issue_summary: str
    order_id: str | None
    amount_mentioned: float | None
    evidence_modalities: list[str]
    citations: list[str]
    confidence: float
    needs_human_review: bool

    def to_dict(self) -> dict:
        """JSON-serializable form."""
        return {
            "category": self.category,
            "urgency": self.urgency,
            "customer_sentiment": self.customer_sentiment,
            "requested_resolution": self.requested_resolution,
            "issue_summary": self.issue_summary,
            "order_id": self.order_id,
            "amount_mentioned": self.amount_mentioned,
            "evidence_modalities": list(self.evidence_modalities),
            "citations": list(self.citations),
            "confidence": self.confidence,
            "needs_human_review": self.needs_human_review,
        }


@dataclass(frozen=True)
class RetrievedRef:
    """A compact, serializable reference to one retrieved knowledge-base chunk."""

    chunk_id: str
    source: str
    section: str | None
    modality: str
    score: float


@dataclass
class TicketResult:
    """
    The full outcome of processing one ticket.

    Attributes:
        ticket_id: Unique id.
        status: "ok" | "needs_human_review" | "failed" | "rejected".
            "rejected" = invalid input (never consumed resources); "failed" =
            no usable evidence could be produced.
        record: The extracted TicketRecord, or None when extraction failed.
        channels_used: Channels that contributed evidence.
        retrieved: References to the knowledge chunks retrieved.
        cost_usd: Total cost of this ticket (sum of its ledger events).
        latency_ms: End-to-end wall-clock latency.
        cost_breakdown: stage -> USD.
        flags: Machine-readable reasons (e.g. "injection_flagged", "degraded").
        problems: Validation/channel problems, for debugging and review.
        estimated_price: True if any stage was priced with the default (unknown-model) price.
    """

    ticket_id: str
    status: str
    record: TicketRecord | None = None
    channels_used: list[str] = field(default_factory=list)
    retrieved: list[RetrievedRef] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cost_breakdown: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    estimated_price: bool = False
