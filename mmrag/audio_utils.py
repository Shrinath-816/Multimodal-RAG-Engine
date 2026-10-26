"""
mmrag/audio_utils.py
=======================

Small, dependency-free audio helpers (stdlib `wave` + `array` only — the
`audioop` module was removed in Python 3.13, so it is deliberately avoided).

All PCM in this project is signed 16-bit little-endian mono.
"""

import array
import io
import math
import sys
import wave

DEFAULT_SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def pcm16_rms(frame: bytes) -> float:
    """
    Root-mean-square energy of a 16-bit PCM frame — the signal the energy VAD thresholds.

    Returns 0.0 for an empty frame.
    """
    usable = len(frame) - (len(frame) % BYTES_PER_SAMPLE)
    if usable <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frame[:usable])
    if sys.byteorder == "big":
        samples.byteswap()
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples))


def pcm_to_wav(pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container so STT services can ingest it."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(BYTES_PER_SAMPLE)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


def audio_duration_seconds(audio: bytes) -> tuple[float, bool]:
    """
    Duration of an audio payload, for per-minute STT pricing.

    Returns:
        (seconds, exact). WAV files are measured exactly from the header.
        Anything else is ESTIMATED assuming 16 kHz 16-bit mono PCM
        (32,000 bytes/second) and exact=False — compressed formats (MP3/Opus)
        will be mis-estimated, so prefer WAV/PCM where cost accuracy matters.
    """
    try:
        with wave.open(io.BytesIO(audio), "rb") as reader:
            frames = reader.getnframes()
            rate = reader.getframerate()
            if rate > 0:
                return frames / rate, True
    except (wave.Error, EOFError):
        pass
    return len(audio) / (DEFAULT_SAMPLE_RATE * BYTES_PER_SAMPLE), False
