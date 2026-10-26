"""
test_rag.py
==============

Tests for chunking, BM25, RRF, the hybrid multimodal index, and the knowledge-base loader.
"""

import pytest

from mmrag.providers.fakes import FakeEmbedder
from mmrag.rag.chunking import chunk_markdown
from mmrag.rag.index import KnowledgeIndex, LexicalIndex, rrf_fuse
from mmrag.rag.kb_loader import load_knowledge_base
from tests.helpers import KB_DIR, run


def test_chunking_tracks_sections_and_never_crosses_headings() -> None:
    text = "# Alpha\n\nfirst body words here.\n\n# Beta\n\nsecond body words here."
    chunks = chunk_markdown(text, max_words=50, overlap_words=5)
    assert [c.section for c in chunks] == ["Alpha", "Beta"]
    assert "second" not in chunks[0].text and "first" not in chunks[1].text


def test_chunking_respects_max_words_and_overlaps_within_a_section() -> None:
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_markdown(f"# S\n\n{words}", max_words=30, overlap_words=5)
    assert all(len(c.text.split()) <= 30 for c in chunks)
    assert len(chunks) >= 4
    first_tail = chunks[0].text.split()[-5:]
    assert chunks[1].text.split()[:5] == first_tail  # overlap carried forward


def test_chunking_edge_cases() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []
    with pytest.raises(ValueError):
        chunk_markdown("x", max_words=10, overlap_words=10)


def test_bm25_finds_exact_identifier_tokens() -> None:
    index = LexicalIndex()
    index.add("a", "Customer asked about order ORD-1042 refund status")
    index.add("b", "General shipping policy and delivery times")
    index.add("c", "Troubleshooting wifi router lights")
    assert index.search("ORD-1042", limit=3) == ["a"]
    assert index.search("nonexistent-token", limit=3) == []


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = rrf_fuse([["a", "b"], ["b", "c"]])
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["b", "a", "c"]  # b appears in both rankings; a outranks c by ranking position
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)


def _index_with_docs() -> KnowledgeIndex:
    index = KnowledgeIndex(FakeEmbedder())
    run(index.add_text_document("returns.md", "# Damaged items\n\nCracked or damaged items get a refund or replacement."))
    run(index.add_text_document("wifi.md", "# Wifi\n\nRestart the router and check the status lights for wifi drops."))
    run(index.add_image_document("dialog.png", "Screenshot of error 0x80070005 access denied activation dialog."))
    return index


def test_hybrid_search_returns_relevant_chunk_first() -> None:
    index = _index_with_docs()
    response = run(index.search("my phone screen is cracked and damaged, I want a refund", top_k=2))
    assert response.results[0].chunk.source == "returns.md"
    assert response.usage.input_tokens > 0 and response.model == "demo-embed"


def test_image_documents_are_retrievable_and_filterable_by_modality() -> None:
    index = _index_with_docs()
    both = run(index.search("error 0x80070005 access denied", top_k=3))
    assert both.results[0].chunk.modality == "image"
    assert "lexical" in both.results[0].via

    text_only = run(index.search("error 0x80070005 access denied", top_k=3, modality="text"))
    assert all(r.chunk.modality == "text" for r in text_only.results)
    images_only = run(index.search("wifi router", top_k=3, modality="image"))
    assert all(r.chunk.modality == "image" for r in images_only.results)


def test_index_stats() -> None:
    stats = _index_with_docs().stats()
    assert stats["by_modality"] == {"text": 2, "image": 1}
    assert stats["sources"] == 3


def test_kb_loader_indexes_repo_knowledge_base_using_sidecar_captions() -> None:
    index = KnowledgeIndex(FakeEmbedder())
    stats = run(load_knowledge_base(index, KB_DIR, vision=None))
    assert stats.text_docs == 5
    assert stats.image_docs == 1  # captioned via sidecar, no vision model needed
    assert stats.skipped == []
    assert stats.usage.input_tokens > 0  # embedding cost is reported, not hidden


def test_kb_loader_reports_missing_directory_and_uncaptioned_images(tmp_path) -> None:
    index = KnowledgeIndex(FakeEmbedder())
    missing = run(load_knowledge_base(index, str(tmp_path / "nope")))
    assert missing.skipped and "not found" in missing.skipped[0][1]

    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # no sidecar, no vision model
    (tmp_path / "data.bin").write_bytes(b"\x00")
    stats = run(load_knowledge_base(index, str(tmp_path), vision=None))
    reasons = dict(stats.skipped)
    assert "no sidecar caption" in reasons["photo.png"] and "unsupported" in reasons["data.bin"]
