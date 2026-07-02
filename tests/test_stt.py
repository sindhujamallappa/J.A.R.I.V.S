"""Tests for utterance capture endpointing (fake microphone, no audio HW)."""

from __future__ import annotations

import numpy as np

from jarvis.config import AudioConfig, STTConfig
from jarvis.stt.capture import UtteranceCapturer, _rms

FRAME = 1600  # 0.1 s at 16 kHz — keeps the frame arithmetic simple
AUDIO = AudioConfig(sample_rate=16000, frame_length=FRAME)


def _silence() -> np.ndarray:
    return np.zeros(FRAME, dtype=np.int16)


def _speech(level: float = 5000.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(FRAME) * level).clip(-32768, 32767).astype(np.int16)


class FakeMic:
    """Replays a fixed frame sequence, then silence forever."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = iter(frames)

    def read(self) -> np.ndarray:
        return next(self._frames, _silence())


def _cfg(**overrides) -> STTConfig:
    base = dict(
        min_rms_threshold=300.0,
        speech_multiplier=2.5,
        calibration_sec=0.2,       # 2 frames
        silence_duration_sec=0.3,  # 3 frames
        max_duration_sec=5.0,
        onset_timeout_sec=0.5,     # 5 frames
        preroll_sec=0.2,           # 2 frames
    )
    base.update(overrides)
    return STTConfig(**base)


def test_rms_of_silence_is_zero():
    assert _rms(_silence()) == 0.0
    assert _rms(_speech()) > 300.0


def test_returns_none_when_no_speech():
    got = UtteranceCapturer(_cfg(), AUDIO).capture(FakeMic([_silence()] * 50))
    assert got is None


def test_captures_speech_with_preroll_and_stops_on_silence():
    # 2 calibration + 2 pre-onset silence, 5 speech, then silence.
    frames = [_silence()] * 4 + [_speech()] * 5 + [_silence()] * 10
    got = UtteranceCapturer(_cfg(), AUDIO).capture(FakeMic(frames))
    assert got is not None
    # preroll(2) + onset(1) + speech(4) + trailing silence(3) = 10 frames
    assert got.size == 10 * FRAME
    assert got.dtype == np.int16


def test_respects_max_duration_cap():
    frames = [_silence()] * 2 + [_speech()] * 100  # speaker never stops
    got = UtteranceCapturer(_cfg(max_duration_sec=1.0), AUDIO).capture(FakeMic(frames))
    assert got is not None
    assert got.size == int(1.0 * AUDIO.sample_rate)  # capped at 10 frames


def test_onset_threshold_scales_with_ambient_noise():
    # Loud ambient (rms ~2000) -> onset threshold ~5000. The steady noise
    # never crosses it, so capture must time out instead of false-triggering.
    noisy = [_speech(2000.0)] * 50
    got = UtteranceCapturer(_cfg(), AUDIO).capture(FakeMic(noisy))
    assert got is None
