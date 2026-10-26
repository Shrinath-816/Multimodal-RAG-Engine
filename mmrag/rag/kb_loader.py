"""
mmrag/rag/kb_loader.py
=========================

Loads a knowledge-base directory into a KnowledgeIndex:

- *.md / *.txt          -> chunked text documents (modality "text")
- *.png/.jpg/.jpeg/.webp -> image documents (modality "image"), captioned by:
    1. a sidecar file "<image>.caption.txt" if present (free, deterministic,
       human-reviewed captions are usually BETTER than model captions), else
    2. the vision model, if one is supplied (costs money, once, at ingest), else
    3. skipped and reported — never silently dropped.
"""

import os
from dataclasses import dataclass, field

from mmrag.ingest.image import preprocess_image, to_data_url
from mmrag.rag.index import KnowledgeIndex
from mmrag.schemas import Usage

KB_CAPTION_INSTRUCTION = (
    "Write a 2-3 sentence factual caption of this knowledge-base image for search indexing. "
    "Include any readable labels, error codes, part names, or LED/indicator states. "
    "Plain text only."
)

_TEXT_EXT = (".md", ".txt")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class LoadStats:
    """
    Result of loading a knowledge base.

    Attributes:
        text_docs / image_docs: Documents indexed.
        chunks: Total chunks added.
        skipped: (file, reason) pairs for anything not indexed.
        usage: Embedding (and any captioning) usage incurred at ingest time.
    """

    text_docs: int = 0
    image_docs: int = 0
    chunks: int = 0
    skipped: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


async def load_knowledge_base(
    index: KnowledgeIndex, kb_dir: str, vision=None, max_image_side: int = 1024, max_image_bytes: int = 5_000_000
) -> LoadStats:
    """
    Index every supported file in `kb_dir` (non-recursive, sorted for determinism).

    Args:
        index: Target index.
        kb_dir: Directory path. A missing directory yields an empty LoadStats with a skip note.
        vision: Optional vision model for captioning images that lack a sidecar caption.
        max_image_side / max_image_bytes: Preprocessing limits for captioning.
    """
    stats = LoadStats()
    if not os.path.isdir(kb_dir):
        stats.skipped.append((kb_dir, "knowledge base directory not found"))
        return stats

    for name in sorted(os.listdir(kb_dir)):
        path = os.path.join(kb_dir, name)
        lower = name.lower()
        if not os.path.isfile(path) or lower.endswith(".caption.txt"):
            continue

        if lower.endswith(_TEXT_EXT):
            with open(path, encoding="utf-8") as handle:
                added = await index.add_text_document(name, handle.read())
            stats.text_docs += 1
            stats.chunks += added.chunks_added
            stats.usage = stats.usage + added.usage

        elif lower.endswith(_IMAGE_EXT):
            sidecar = path + ".caption.txt"
            caption = None
            if os.path.isfile(sidecar):
                with open(sidecar, encoding="utf-8") as handle:
                    caption = handle.read().strip()
            elif vision is not None:
                with open(path, "rb") as handle:
                    raw = handle.read()
                prepared = preprocess_image(raw, max_image_side, max_image_bytes)
                described = await vision.describe(to_data_url(prepared.data, prepared.mime), KB_CAPTION_INSTRUCTION)
                caption = described.text.strip()
                stats.usage = stats.usage + described.usage
            if not caption:
                stats.skipped.append((name, "no sidecar caption and no vision model available"))
                continue
            added = await index.add_image_document(name, caption)
            stats.image_docs += 1
            stats.chunks += added.chunks_added
            stats.usage = stats.usage + added.usage

        else:
            stats.skipped.append((name, "unsupported file type"))

    return stats
