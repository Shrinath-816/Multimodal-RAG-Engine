# The Multimodal RAG Engine (Week 9)

A multimodal RAG system for support tickets. Text, **image** (screenshots, photos) and
**voice** (audio / PCM stream) channels are ingested, grounded against a hybrid-retrieval
knowledge base (text + captioned images), turned into **validated structured JSON**, and
the **cost of every ticket** is tracked and shown on a live dashboard.

## Quick start (no API keys, no network)

```bash
pip install -r requirements.txt          # Pillow recommended; fastapi/uvicorn for the server
cp .env.example .env                     # DEMO_MODE=1 by default
python scripts/run_eval.py               # golden set through the real pipeline
python -m pytest -q                      # 100 tests (core needs only the stdlib + pytest)
uvicorn mmrag.main:create_app --factory --port 8000
# open http://localhost:8000/dashboard
python scripts/simulate_traffic.py       # in another shell: fills the dashboard
```

## DEMO_MODE and honesty notes
- `DEMO_MODE=1` uses deterministic fakes (hashed bag-of-words embedder, rule-based "LLM",
  scenario-based "vision", fake STT). Costs are **synthetic**; eval accuracy measures
  plumbing, **not** model quality.
- Real providers: set `DEMO_MODE=0`, provide OpenAI-compatible keys/base URL (see `.env.example`)
  and a pricing file via `PRICING_PATH` (start from `config/pricing.example.json`;
  its prices are **fake placeholders** — put in your provider's real prices).
  Models missing from the table are recorded at $0 and flagged *estimated*.

## Architecture
```
text ─────────────┐
image → preprocess → vision model ─┤ (concurrent, partial failure tolerated)
voice → VAD/STT ───────────────────┘
        │ evidence (PII-redacted, injection-scanned, <untrusted_*> delimited)
        ▼
hybrid retrieval (dense + BM25 → RRF; text + image-caption chunks)
        ▼
structured extraction (closed enums, grounding checks, 1 repair retry, pre-call budget)
        ▼
review gating (low confidence / injection / degraded channel → human review)
        ▼
cost ledger (stage + channel attribution) → /api/metrics → live dashboard
```
Statuses: `ok`, `needs_human_review`, `failed` (provider outage, HTTP 502),
`rejected` (invalid input, HTTP 422, nothing spent).

## Layout
`mmrag/` core (config, schemas, pricing, budget, ledger, metrics, security, providers/, ingest/, rag/,
extraction, pipeline, api, dashboard, main) · `scripts/` eval + traffic simulator ·
`data/` knowledge base + golden set · `tests/`.

## Week 9 concept map
Encoder→projector→LLM and image-token cost → `ingest/image.py` (downscale caps) ·
caption-and-index multimodal RAG → `rag/kb_loader.py` · VAD/endpointing → `ingest/voice.py` ·
per-stage latency/cost budgets → `budget.py`, `ledger.py` · safety → `security.py`.

## Known limitations
- The FastAPI layer (`main.py`: multipart, WebSocket `/ws/voice`, dashboard route) and the
  real OpenAI-compatible adapters were **not executed** in the build sandbox (no packages
  available); adapters are tested against fake SDK clients for request shape only.
- Energy-based VAD is a teaching baseline; use a neural VAD in production.
- Ledger/index are in-memory; the capstone should move them to Postgres/pgvector or Qdrant.
