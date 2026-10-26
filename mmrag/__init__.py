"""
The Multimodal RAG Engine
===========================

Week 9 project: a multimodal RAG system with VISION and VOICE data
channels that extracts STRUCTURED ticket data and tracks COST PER
TICKET on a live dashboard — and that establishes the architecture you
will carry into the Week 10 capstone.

How the pieces map to the Week 9 notes (week-9-notes.md):

- mmrag/ingest/image.py   Vision channel: preprocess (resize, orient, strip
                          metadata) -> VLM -> description + visible text,
                          treated as UNTRUSTED content (Section 3).
- mmrag/ingest/voice.py   Voice channel: energy VAD + endpointing over a PCM
                          stream -> streaming-style STT (Section 4).
- mmrag/rag/              Multimodal RAG: caption-and-index for image
                          documents, hybrid dense + lexical retrieval fused with
                          RRF, modality-aware metadata (Sections 6-7).
- mmrag/extraction.py     Grounded structured extraction: schema validation,
                          deterministic grounding checks, one repair retry.
- mmrag/pipeline.py       Orchestration: parallel channels, PII redaction,
                          retrieval, extraction, explicit partial-failure policy.
- mmrag/ledger.py,        Per-ticket cost accounting (tokens, audio minutes,
  pricing.py, metrics.py  stage + channel breakdown) feeding the live dashboard.
- mmrag/providers/        Provider abstraction: OpenAI-compatible adapters for
                          real use, deterministic fakes for DEMO_MODE and tests,
                          and a resilience layer (timeout, retry, fallback).

Design rule for the core: it is PURE STANDARD LIBRARY (plus optional Pillow
for image preprocessing). Third-party packages (openai, fastapi) live only at
the edges — providers/openai_compat.py and main.py — so the logic that
matters (validation, cost accounting, VAD, retrieval, orchestration) is
testable without network access or API keys.
"""
