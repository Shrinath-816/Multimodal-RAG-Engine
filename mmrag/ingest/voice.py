"""
mmrag/ingest/voice.py
========================

The VOICE channel (week-9-notes.md, section 4).

Two pieces:

1. VoiceStreamSession — an energy-based VAD with ENDPOINTING over a live PCM
   stream. Feed it arbitrary-sized chunks (e.g. from a WebSocket); it returns
   each completed utterance as PCM. This is the "continuous, interruptible
   loop" front end: voice activity detection -> turn (end-of-utterance)
   detection. Includes a short pre-roll so the first syllable isn't clipped,
   a minimum-speech filter so noise blips don't become turns, and a maximum
   utterance length so a stuck-open mic can't buffer forever.

2. VoiceChannel — runs speech-to-text on a finished utterance (or an uploaded
   audio file) and treats the transcript as UNTRUSTED text: injection-scanned
   and PII-redacted before anything else sees it.

Honest limits: a fixed-silence energy VAD is the simplest endpointing there
is. Notes section 4.5 explains why production systems prefer model-based VAD
and semantic turn detection; the VoiceStreamSession interface is what you'd
swap those into. There is no barge-in here because this engine does not
SPEAK — it ingests voice for ticket extraction.
"""

import time
from collections import deque
from dataclasses import dataclass

from mmrag.audio_utils import (
    BYTES_PER_SAMPLE,
    DEFAULT_SAMPLE_RATE,
    audio_duration_seconds,
    pcm16_rms,
    pcm_to_wav,
)
from mmrag.ingest.errors import MediaTooLargeError, UnsupportedMediaError
from mmrag.schemas import Usage
from mmrag.security import redact_pii, scan_for_injection


@dataclass(frozen=True)
class VADConfig:
    """
    Tunables for the energy VAD / endpointing.

    Attributes:
        sample_rate: PCM sample rate (Hz).
        frame_ms: Analysis frame length.
        speech_rms_threshold: A frame with RMS >= this counts as speech.
        min_speech_ms: Utterances with less total speech than this are discarded as noise.
        end_silence_ms: Trailing silence that ends an utterance (the endpointing hold-off —
            too short cuts people off mid-thought, too long adds dead air).
        preroll_ms: Audio kept from BEFORE speech onset so the first syllable isn't clipped.
        max_utterance_ms: Hard cap; an utterance is force-closed at this length.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_ms: int = 20
    speech_rms_threshold: float = 500.0
    min_speech_ms: int = 200
    end_silence_ms: int = 700
    preroll_ms: int = 200
    max_utterance_ms: int = 30_000


class VoiceStreamSession:
    """Stateful VAD + endpointing over a PCM16 mono stream."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self._cfg = config or VADConfig()
        self._frame_bytes = int(self._cfg.sample_rate * self._cfg.frame_ms / 1000) * BYTES_PER_SAMPLE
        self._preroll_frames = max(0, self._cfg.preroll_ms // self._cfg.frame_ms)
        self._leftover = b""
        self._reset()

    def _reset(self) -> None:
        self._in_speech = False
        self._preroll: deque = deque(maxlen=self._preroll_frames or None)
        if self._preroll_frames == 0:
            self._preroll = deque(maxlen=0)
        self._utterance = bytearray()
        self._speech_frames = 0
        self._silence_frames = 0
        self._utterance_frames = 0

    def _finalize(self) -> bytes | None:
        """Close the current utterance; return it only if it contained enough speech."""
        utterance = bytes(self._utterance)
        enough = self._speech_frames * self._cfg.frame_ms >= self._cfg.min_speech_ms
        self._reset()
        return utterance if enough and utterance else None

    def feed(self, chunk: bytes) -> list[bytes]:
        """
        Consume a chunk of PCM16 audio.

        Args:
            chunk: Any number of bytes (need not align to frame boundaries).

        Returns:
            Zero or more completed utterances (raw PCM16), in order.
        """
        data = self._leftover + chunk
        usable = len(data) - (len(data) % self._frame_bytes)
        self._leftover = data[usable:]
        utterances: list[bytes] = []

        for offset in range(0, usable, self._frame_bytes):
            frame = data[offset : offset + self._frame_bytes]
            is_speech = pcm16_rms(frame) >= self._cfg.speech_rms_threshold

            if not self._in_speech:
                if is_speech:
                    self._in_speech = True
                    for buffered in self._preroll:
                        self._utterance.extend(buffered)
                    self._preroll.clear()
                    self._utterance.extend(frame)
                    self._speech_frames = 1
                    self._silence_frames = 0
                    self._utterance_frames = len(self._utterance) // self._frame_bytes
                else:
                    self._preroll.append(frame)
                continue

            self._utterance.extend(frame)
            self._utterance_frames += 1
            if is_speech:
                self._speech_frames += 1
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            ended = self._silence_frames * self._cfg.frame_ms >= self._cfg.end_silence_ms
            too_long = self._utterance_frames * self._cfg.frame_ms >= self._cfg.max_utterance_ms
            if ended or too_long:
                finished = self._finalize()
                if finished is not None:
                    utterances.append(finished)

        return utterances

    def flush(self) -> bytes | None:
        """End of stream: return any in-progress utterance (if it had enough speech)."""
        self._leftover = b""
        if self._in_speech:
            return self._finalize()
        self._reset()
        return None


@dataclass
class VoiceTranscript:
    """
    What the voice channel produced for one utterance/file.

    Attributes:
        text: PII-redacted transcript.
        injection_flagged: Injection-like text found in the transcript.
        pii_redactions: PII type -> count removed.
        duration_seconds: Audio duration used for pricing.
        duration_exact: False when duration was estimated (non-WAV payload).
        usage / model / latency_ms: For cost accounting.
    """

    text: str
    injection_flagged: bool
    pii_redactions: dict
    duration_seconds: float
    duration_exact: bool
    usage: Usage
    model: str
    latency_ms: float


class VoiceChannel:
    """Speech-to-text for a finished utterance or uploaded audio file."""

    def __init__(self, stt, max_audio_bytes: int) -> None:
        """
        Args:
            stt: Object with `async transcribe(audio, filename, mime) -> TranscriptResult`
                (normally a ResilientSTT wrapper).
            max_audio_bytes: Upload cap.
        """
        self._stt = stt
        self._max_audio_bytes = max_audio_bytes

    async def transcribe(self, audio: bytes, filename: str = "audio.wav", mime: str = "audio/wav") -> VoiceTranscript:
        """
        Transcribe one audio payload.

        Raises:
            UnsupportedMediaError: Empty payload.
            MediaTooLargeError: Over the size cap.
            ProviderUnavailableError: STT providers all failed.
        """
        if not audio:
            raise UnsupportedMediaError("Audio payload is empty.")
        if len(audio) > self._max_audio_bytes:
            raise MediaTooLargeError(f"Audio is {len(audio)} bytes; limit is {self._max_audio_bytes}.")

        _seconds, exact = audio_duration_seconds(audio)
        started = time.perf_counter()
        result = await self._stt.transcribe(audio, filename, mime)
        latency_ms = (time.perf_counter() - started) * 1000

        flagged = bool(scan_for_injection(result.text))
        redacted, counts = redact_pii(result.text)
        return VoiceTranscript(
            text=redacted.strip(),
            injection_flagged=flagged,
            pii_redactions=counts,
            duration_seconds=result.usage.audio_seconds,
            duration_exact=exact,
            usage=result.usage,
            model=result.model,
            latency_ms=latency_ms,
        )

    async def transcribe_pcm(self, pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> VoiceTranscript:
        """Transcribe raw PCM16 (e.g. a VoiceStreamSession utterance) by wrapping it as WAV."""
        return await self.transcribe(pcm_to_wav(pcm, sample_rate), "utterance.wav", "audio/wav")
