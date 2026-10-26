"""
mmrag/api.py
===============

Framework-agnostic request handlers. The web layer (main.py) is a thin
adapter over these methods, so everything with logic in it — response
shaping, status mapping, metrics assembly — is unit-testable without FastAPI.
"""

from mmrag.audio_utils import DEFAULT_SAMPLE_RATE
from mmrag.config import Settings
from mmrag.ledger import CostLedger
from mmrag.metrics import compute_metrics
from mmrag.pipeline import TicketInput, TicketPipeline
from mmrag.providers.gateway import ReliabilityCounters
from mmrag.rag.index import KnowledgeIndex
from mmrag.schemas import TicketResult

STATUS_TO_HTTP = {"ok": 200, "needs_human_review": 200, "failed": 502, "rejected": 422}


def result_to_dict(result: TicketResult) -> dict:
    """Serialize a TicketResult for JSON responses (includes an `http_status` hint for the web layer)."""
    return {
        "ticket_id": result.ticket_id,
        "status": result.status,
        "http_status": STATUS_TO_HTTP.get(result.status, 500),
        "record": result.record.to_dict() if result.record else None,
        "channels_used": list(result.channels_used),
        "retrieved": [
            {
                "chunk_id": r.chunk_id, "source": r.source, "section": r.section,
                "modality": r.modality, "score": r.score,
            }
            for r in result.retrieved
        ],
        "cost_usd": result.cost_usd,
        "cost_breakdown": dict(result.cost_breakdown),
        "latency_ms": round(result.latency_ms, 1),
        "flags": list(result.flags),
        "problems": list(result.problems),
        "estimated_price": result.estimated_price,
    }


class TicketAPI:
    """Handlers for submitting tickets and reading dashboard data."""

    def __init__(
        self,
        pipeline: TicketPipeline,
        ledger: CostLedger,
        counters: ReliabilityCounters,
        settings: Settings,
        index: KnowledgeIndex,
    ) -> None:
        self._pipeline = pipeline
        self._ledger = ledger
        self._counters = counters
        self._settings = settings
        self._index = index

    async def submit(
        self,
        text: str | None = None,
        image: bytes | None = None,
        audio: bytes | None = None,
        audio_filename: str = "audio.wav",
        audio_mime: str = "audio/wav",
    ) -> dict:
        """Process a ticket made of any combination of text, image, and audio."""
        result = await self._pipeline.process(
            TicketInput(text=text, image=image, audio=audio, audio_filename=audio_filename, audio_mime=audio_mime)
        )
        return result_to_dict(result)

    async def submit_voice_pcm(self, pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict:
        """Process one finished utterance from a live stream (raw PCM16) as a voice ticket."""
        from mmrag.audio_utils import pcm_to_wav

        return await self.submit(audio=pcm_to_wav(pcm, sample_rate), audio_filename="utterance.wav")

    def metrics(self) -> dict:
        """Everything the live dashboard renders."""
        data = compute_metrics(self._ledger)
        data["demo_mode"] = self._settings.demo_mode
        data["reliability"] = self._counters.to_dict()
        data["knowledge_base"] = self._index.stats()
        data["rejected_total"] = self._pipeline.rejected_total
        data["budget_per_ticket_usd"] = self._settings.max_ticket_cost_usd
        return data

    def recent_tickets(self, limit: int = 20) -> list[dict]:
        """Most recent ticket summaries, newest first."""
        return compute_metrics(self._ledger, recent=limit)["recent_tickets"]

    def health(self) -> dict:
        """Liveness plus a few facts useful for debugging a deployment."""
        return {"status": "ok", "demo_mode": self._settings.demo_mode, "kb_chunks": self._index.stats()["chunks"]}
