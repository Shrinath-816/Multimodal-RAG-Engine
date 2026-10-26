"""
test_image.py
================

Tests for image validation/preprocessing and the vision channel. Tests that
need Pillow are skipped when it isn't installed (the no-Pillow path is tested too).
"""

import json

import pytest

from mmrag.demo_media import demo_png_for_scenario, make_png
from mmrag.ingest.errors import MediaTooLargeError, UnsupportedMediaError
from mmrag.ingest.image import ImageChannel, preprocess_image, sniff_image_mime, to_data_url
from mmrag.providers.base import ChatResult
from mmrag.providers.fakes import FakeVision
from mmrag.schemas import Usage
from tests.helpers import run

try:
    import PIL  # noqa: F401

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


class _StubVision:
    """Vision double returning fixed text (for testing parse/redact/flag behavior)."""

    model = "stub-vision"

    def __init__(self, text: str) -> None:
        self._text = text

    async def describe(self, image_data_url: str, instruction: str) -> ChatResult:
        return ChatResult(self._text, Usage(input_tokens=700, output_tokens=40), self.model)


def test_sniff_detects_formats_by_magic_bytes() -> None:
    assert sniff_image_mime(make_png()) == "image/png"
    assert sniff_image_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff_image_mime(b"GIF89a....") == "image/gif"
    assert sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image_mime(b"not an image") is None


def test_preprocess_rejects_oversize_and_non_images() -> None:
    with pytest.raises(MediaTooLargeError):
        preprocess_image(make_png(), max_side=1024, max_bytes=10)
    with pytest.raises(UnsupportedMediaError):
        preprocess_image(b"just text, not an image", max_side=1024, max_bytes=1_000_000)


def test_preprocess_downscales_large_images() -> None:
    if not HAS_PILLOW:
        return
    big = make_png(width=400, height=200)
    result = preprocess_image(big, max_side=100, max_bytes=1_000_000)
    assert result.resized is True
    assert (result.width, result.height) == (100, 50)
    assert "metadata stripped" in result.note


def test_preprocess_leaves_small_images_unresized_but_reencodes() -> None:
    if not HAS_PILLOW:
        return
    result = preprocess_image(make_png(width=32, height=32), max_side=1024, max_bytes=1_000_000)
    assert result.resized is False and (result.width, result.height) == (32, 32)


def test_preprocess_rejects_corrupt_image_with_valid_magic() -> None:
    if not HAS_PILLOW:
        return
    corrupt = b"\x89PNG\r\n\x1a\n" + b"garbage" * 10
    with pytest.raises(UnsupportedMediaError):
        preprocess_image(corrupt, max_side=1024, max_bytes=1_000_000)


def test_to_data_url_shape() -> None:
    assert to_data_url(b"abc", "image/png") == "data:image/png;base64,YWJj"


def test_vision_channel_flags_injection_in_image_text() -> None:
    channel = ImageChannel(FakeVision(), max_side=1024, max_bytes=1_000_000)
    analysis = run(channel.analyze(demo_png_for_scenario(3)))  # scenario 3 carries an injected instruction
    assert analysis.injection_flagged is True
    assert "ORD-4488" in analysis.visible_text


def test_vision_channel_picks_scenario_by_width_and_extracts_entities() -> None:
    channel = ImageChannel(FakeVision(), max_side=1024, max_bytes=1_000_000)
    analysis = run(channel.analyze(demo_png_for_scenario(1)))
    assert analysis.entities.get("order_id") == "ORD-2201"
    assert analysis.injection_flagged is False
    assert analysis.usage.input_tokens > 0


def test_vision_channel_redacts_pii_found_in_images() -> None:
    payload = json.dumps({"description": "A letter", "visible_text": "Contact jane@example.com ORD-1001", "entities": {}})
    channel = ImageChannel(_StubVision(payload), max_side=1024, max_bytes=1_000_000)
    analysis = run(channel.analyze(make_png()))
    assert "jane@example.com" not in analysis.visible_text
    assert analysis.pii_redactions == {"email": 1}


def test_vision_channel_survives_non_json_model_output() -> None:
    channel = ImageChannel(_StubVision("A photo of a broken phone."), max_side=1024, max_bytes=1_000_000)
    analysis = run(channel.analyze(make_png()))
    assert analysis.description == "A photo of a broken phone."
    assert analysis.visible_text == "" and analysis.entities == {}
