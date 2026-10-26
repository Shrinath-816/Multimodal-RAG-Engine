"""
rag package
=============

Multimodal retrieval-augmented generation components:

- chunking.py   paragraph/heading-aware text chunking with overlap
- index.py      KnowledgeIndex: dense + lexical (BM25) retrieval fused with RRF,
                with modality-tagged chunks (text and image documents)
- kb_loader.py  loads a knowledge-base directory (text docs + captioned images)

Image documents use CAPTION-AND-INDEX: each image is turned into a text
caption once at ingest time (or read from a sidecar .caption.txt) and indexed
like any other text. See week-9-notes.md section 7 for the trade-offs against
unified embeddings and page-as-image late interaction.
"""
