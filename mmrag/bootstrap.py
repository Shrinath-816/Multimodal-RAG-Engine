"""
mmrag/bootstrap.py
=====================

The composition root: wires settings -> providers -> resilience wrappers ->
channels -> index -> extractor -> pipeline -> API. Everything that needs a
running engine (the web app, the eval script, the traffic simulator, tests)
builds it through `build_engine`, so there is exactly one wiring to keep correct.
"""

from dataclasses import dataclass

from mmrag.api import TicketAPI
from mmrag.config import Settings
from mmrag.extraction import StructuredExtractor
from mmrag.ingest.image import ImageChannel
from mmrag.ingest.voice import VoiceChannel
from mmrag.ledger import CostLedger
from mmrag.pipeline import TicketPipeline
from mmrag.pricing import PricingTable
from mmrag.providers.factory import build_providers
from mmrag.providers.gateway import (
    ReliabilityCounters,
    ResilientChat,
    ResilientEmbedder,
    ResilientSTT,
    ResilientVision,
)
from mmrag.rag.index import KnowledgeIndex
from mmrag.rag.kb_loader import LoadStats, load_knowledge_base


@dataclass
class Engine:
    """Everything a caller needs from a built engine."""

    settings: Settings
    api: TicketAPI
    pipeline: TicketPipeline
    index: KnowledgeIndex
    ledger: CostLedger
    counters: ReliabilityCounters
    kb_stats: LoadStats


async def build_engine(settings: Settings, *, load_kb: bool = True, pricing: PricingTable | None = None) -> Engine:
    """
    Build a fully wired engine.

    Args:
        settings: Configuration.
        load_kb: Index the knowledge base directory (disable for tests that add their own docs).
        pricing: Override the price table (tests); otherwise loaded from settings.pricing_path.
    """
    providers = build_providers(settings)
    counters = ReliabilityCounters()
    tuning = dict(
        timeout=settings.model_timeout_seconds,
        retries=settings.model_retries,
        backoff=settings.retry_backoff_seconds,
        counters=counters,
    )

    chat = ResilientChat(providers.chat, **tuning)
    vision = ResilientVision(providers.vision, **tuning)
    stt = ResilientSTT(providers.stt, **tuning)
    # ONE candidate only: no cross-model embedding fallback (vector-space consistency).
    embedder = ResilientEmbedder([providers.embedder], **tuning)

    price_table = pricing or PricingTable.from_file(settings.pricing_path)
    ledger = CostLedger(max_tickets=settings.ledger_max_tickets)
    index = KnowledgeIndex(embedder)

    kb_stats = LoadStats()
    if load_kb:
        kb_stats = await load_knowledge_base(
            index, settings.kb_dir, vision=vision,
            max_image_side=settings.max_image_side, max_image_bytes=settings.max_image_bytes,
        )

    extractor = StructuredExtractor(chat, price_table, ledger, settings.order_id_pattern)
    pipeline = TicketPipeline(
        settings=settings,
        pricing=price_table,
        ledger=ledger,
        index=index,
        image_channel=ImageChannel(vision, settings.max_image_side, settings.max_image_bytes),
        voice_channel=VoiceChannel(stt, settings.max_audio_bytes),
        extractor=extractor,
        counters=counters,
    )
    api = TicketAPI(pipeline, ledger, counters, settings, index)
    return Engine(settings, api, pipeline, index, ledger, counters, kb_stats)
