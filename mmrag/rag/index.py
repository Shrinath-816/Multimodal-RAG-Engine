"""
mmrag/rag/index.py
=====================

The knowledge index behind multimodal RAG.

- Every chunk has a MODALITY ("text" or "image"). Image documents are
  indexed by their caption (caption-and-index), so one retrieval path serves
  both modalities and results can be filtered/labelled by modality.
- Retrieval is HYBRID: dense (embedding cosine) + lexical (BM25), fused with
  Reciprocal Rank Fusion (Week 3). Lexical matters here because tickets
  quote exact tokens — order ids, error codes — that dense embeddings blur.
- The embedder is a single model with NO fallback: query vectors must come
  from the same model that embedded the corpus (the Week 2 invariant).

In-memory and brute-force: appropriate for a demo/capstone scale knowledge
base (hundreds to low thousands of chunks). Swap `_dense` for Qdrant/pgvector
(Week 2) when the corpus outgrows it; the KnowledgeIndex interface stays.
"""

import math
import re
import time
from dataclasses import dataclass, field

from mmrag.rag.chunking import chunk_markdown
from mmrag.schemas import Usage

MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
EMBED_BATCH = 64


@dataclass
class KBChunk:
    """
    One retrievable unit of knowledge.

    Attributes:
        chunk_id: Stable id, e.g. "kb:returns_policy.md#2" — what the model must cite.
        text: The retrievable text (for image documents, the caption).
        source: Originating file name.
        section: Markdown heading the chunk fell under, if any.
        modality: "text" or "image".
    """

    chunk_id: str
    text: str
    source: str
    section: str | None
    modality: str

    def embed_text(self) -> str:
        """Text actually embedded: section title + body gives the vector more context."""
        return f"{self.section}\n{self.text}" if self.section else self.text


@dataclass
class RetrievedChunk:
    """A chunk returned by search, with its fused score and which rankers found it."""

    chunk: KBChunk
    score: float
    via: list[str] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Search results plus the cost-relevant facts of the query embedding."""

    results: list[RetrievedChunk]
    usage: Usage
    model: str
    latency_ms: float


@dataclass
class IngestStats:
    """What an add_* call did (and what embedding it cost)."""

    chunks_added: int
    usage: Usage


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LexicalIndex:
    """A small BM25 index (k1=1.5, b=0.75) over chunk texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._doc_tokens: dict[str, list[str]] = {}
        self._df: dict[str, int] = {}
        self._total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        """Index one document."""
        tokens = _tokenize(text)
        self._doc_tokens[doc_id] = tokens
        self._total_len += len(tokens)
        for token in set(tokens):
            self._df[token] = self._df.get(token, 0) + 1

    def search(self, query: str, limit: int) -> list[str]:
        """Return up to `limit` doc ids with positive BM25 score, best first."""
        n_docs = len(self._doc_tokens)
        if n_docs == 0:
            return []
        avg_len = self._total_len / n_docs or 1.0
        query_tokens = set(_tokenize(query))
        scored: list[tuple[float, str]] = []
        for doc_id, tokens in self._doc_tokens.items():
            if not tokens:
                continue
            length_norm = 1 - self._b + self._b * len(tokens) / avg_len
            score = 0.0
            for term in query_tokens:
                tf = tokens.count(term)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                score += idf * (tf * (self._k1 + 1)) / (tf + self._k1 * length_norm)
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for _score, doc_id in scored[:limit]]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion: score(id) = sum over rankings of 1 / (k + rank).

    Rank-based on purpose — dense cosine scores and BM25 scores live on
    incomparable scales, so blending raw scores is meaningless (Week 3).

    Returns:
        (id, fused_score) sorted best-first (ties keep first-seen order).
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            order.setdefault(doc_id, len(order))
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


class KnowledgeIndex:
    """Hybrid dense + lexical index over modality-tagged chunks."""

    def __init__(self, embedder, max_words: int = 120, overlap_words: int = 20) -> None:
        """
        Args:
            embedder: Object with `async embed(texts) -> EmbedResult` (normally a
                ResilientEmbedder built with a SINGLE candidate).
            max_words / overlap_words: Chunking parameters for text documents.
        """
        self._embedder = embedder
        self._max_words = max_words
        self._overlap_words = overlap_words
        self._chunks: dict[str, KBChunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._lexical = LexicalIndex()

    async def _embed_and_store(self, chunks: list[KBChunk]) -> IngestStats:
        usage = Usage()
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            result = await self._embedder.embed([c.embed_text() for c in batch])
            usage = usage + result.usage
            for chunk, vector in zip(batch, result.vectors):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = vector
                self._lexical.add(chunk.chunk_id, chunk.embed_text())
        return IngestStats(chunks_added=len(chunks), usage=usage)

    async def add_text_document(self, source: str, text: str) -> IngestStats:
        """Chunk, embed, and index a text/markdown document."""
        chunks = [
            KBChunk(f"kb:{source}#{i}", c.text, source, c.section, MODALITY_TEXT)
            for i, c in enumerate(chunk_markdown(text, self._max_words, self._overlap_words), start=1)
        ]
        return await self._embed_and_store(chunks)

    async def add_image_document(self, source: str, caption: str, section: str | None = None) -> IngestStats:
        """Index an image by its caption (caption-and-index): ONE chunk, modality 'image'."""
        chunk = KBChunk(f"kb:{source}#1", caption.strip(), source, section, MODALITY_IMAGE)
        return await self._embed_and_store([chunk])

    async def search(self, query: str, top_k: int, modality: str | None = None) -> SearchResponse:
        """
        Retrieve the top_k chunks for `query` (dense + lexical, RRF-fused).

        Args:
            query: Search text (already PII-redacted by the pipeline).
            top_k: Number of results.
            modality: Restrict to "text" or "image" chunks, or None for both.
        """
        started = time.perf_counter()
        embedded = await self._embedder.embed([query])
        query_vec = embedded.vectors[0]
        pool = top_k * 3

        def allowed(chunk_id: str) -> bool:
            return modality is None or self._chunks[chunk_id].modality == modality

        dense_scored = [
            (_cosine(query_vec, vec), chunk_id) for chunk_id, vec in self._vectors.items() if allowed(chunk_id)
        ]
        dense_scored = [pair for pair in dense_scored if pair[0] > 0]
        dense_scored.sort(key=lambda pair: (-pair[0], pair[1]))
        dense_ids = [chunk_id for _s, chunk_id in dense_scored[:pool]]

        lexical_ids = [cid for cid in self._lexical.search(query, pool * 2) if allowed(cid)][:pool]

        fused = rrf_fuse([dense_ids, lexical_ids])[:top_k]
        results = [
            RetrievedChunk(
                chunk=self._chunks[chunk_id],
                score=score,
                via=[name for name, ids in (("dense", dense_ids), ("lexical", lexical_ids)) if chunk_id in ids],
            )
            for chunk_id, score in fused
        ]
        return SearchResponse(
            results=results,
            usage=embedded.usage,
            model=embedded.model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def stats(self) -> dict:
        """Counts for the dashboard: chunks by modality and number of distinct sources."""
        by_modality: dict[str, int] = {}
        for chunk in self._chunks.values():
            by_modality[chunk.modality] = by_modality.get(chunk.modality, 0) + 1
        return {
            "chunks": len(self._chunks),
            "by_modality": by_modality,
            "sources": len({c.source for c in self._chunks.values()}),
        }
