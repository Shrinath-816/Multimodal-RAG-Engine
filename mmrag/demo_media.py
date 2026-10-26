"""
mmrag/demo_media.py
======================

Tiny, dependency-free media generators for demos and tests: a valid PNG
and synthetic 16-bit PCM audio (tone bursts / silence) for exercising the
voice activity detector.

Demo-image convention (used by FakeVision in demo mode only): a PNG's WIDTH
selects the canned scenario — width 32 + 8*n -> scenario n. Width survives
re-encoding by the preprocessing step, unlike trailing bytes or metadata.
"""

import array
import math
import struct
import sys
import zlib


def make_png(width: int = 32, height: int = 32, rgb: tuple = (200, 60, 60), trailer: bytes = b"") -> bytes:
    """Build a valid solid-color RGB PNG (optionally with trailing bytes after IEND)."""
    row = b"\x00" + bytes(rgb) * width  # filter byte 0 + pixels
    raw = row * height

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
        + trailer
    )


def demo_png_for_scenario(scenario: int) -> bytes:
    """A PNG that FakeVision maps to canned scenario `scenario` (0-3) — see module docstring."""
    return make_png(width=32 + 8 * (scenario % 4), height=32)


def make_pcm_tone(duration_ms: int, amplitude: int = 3000, freq: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Little-endian signed 16-bit mono PCM sine tone — stands in for 'speech' energy in VAD tests."""
    count = int(sample_rate * duration_ms / 1000)
    samples = array.array("h", (int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(count)))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def make_pcm_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """All-zero PCM of the given duration."""
    return b"\x00\x00" * int(sample_rate * duration_ms / 1000)
