"""Utterance capture with silence endpointing for J.A.R.I.V.S.

Records the spoken command that follows a wake-word trigger: calibrates the
ambient noise floor, waits for speech onset (keeping a short preroll so the
first syllable isn't clipped), then stops after trailing silence or a hard
duration cap. Operates in the int16 RMS domain on frames from
:class:`~jarvis.utils.audio.MicrophoneStream`.

Classification: input only — microphone reads and arithmetic. Non-destructive;
no confirmation gate applies.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional, Protocol

import numpy as np

from ..config import AudioConfig, STTConfig

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    """Anything that yields fixed-length int16 frames (a MicrophoneStream)."""

    def read(self) -> np.ndarray: ...


def _rms(frame: np.ndarray) -> float:
    """Root-mean-square level of an int16 frame."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))


class UtteranceCapturer:
    """Captures a single utterance (int16 mono) from an open frame source.

    Endpointing parameters come from :class:`STTConfig`; frame geometry from
    :class:`AudioConfig`. The capturer does not own the microphone — pass an
    open stream so the same mic can be shared with the wake-word stage.
    """

    def __init__(self, cfg: STTConfig, audio: AudioConfig) -> None:
        self._cfg = cfg
        self._sample_rate = audio.sample_rate
        self._frame_length = audio.frame_length

    def _frames_for(self, seconds: float) -> int:
        # Integer frame counts (via sample counts) — accumulating float
        # seconds drifts and yields off-by-one frame budgets.
        samples = round(seconds * self._sample_rate)
        return max(1, math.ceil(samples / self._frame_length))

    def capture(self, mic: FrameSource) -> Optional[np.ndarray]:
        """Record one utterance, ended by trailing silence.

        Args:
            mic: An open frame source (e.g. ``MicrophoneStream``).

        Returns:
            1-D int16 array with the utterance (including preroll), or
            ``None`` if speech never started within ``onset_timeout_sec``.
        """
        cfg = self._cfg

        # 1) Ambient-noise calibration.
        calib = [mic.read() for _ in range(self._frames_for(cfg.calibration_sec))]
        ambient = float(np.mean([_rms(f) for f in calib]))
        threshold = max(cfg.min_rms_threshold, ambient * cfg.speech_multiplier)
        log.debug(
            "Capture calibrated: ambient_rms=%.1f onset_threshold=%.1f", ambient, threshold
        )

        # 2) Wait for speech onset, keeping a preroll ring buffer.
        preroll: deque[np.ndarray] = deque(calib, maxlen=self._frames_for(cfg.preroll_sec))
        onset: Optional[np.ndarray] = None
        for _ in range(self._frames_for(cfg.onset_timeout_sec)):
            frame = mic.read()
            if _rms(frame) >= threshold:
                onset = frame
                break
            preroll.append(frame)
        if onset is None:
            log.info("No speech within %.1fs — abandoning capture", cfg.onset_timeout_sec)
            return None

        # 3) Record until trailing silence or the hard duration cap.
        max_frames = self._frames_for(cfg.max_duration_sec)
        silence_needed = self._frames_for(cfg.silence_duration_sec)
        frames: list[np.ndarray] = [*preroll, onset]
        silence_run = 0
        while len(frames) < max_frames:
            frame = mic.read()
            frames.append(frame)
            if _rms(frame) < threshold:
                silence_run += 1
                if silence_run >= silence_needed:
                    break
            else:
                silence_run = 0

        utterance = np.concatenate(frames)
        log.info(
            "Captured utterance: %.2fs (%d frames)",
            utterance.size / self._sample_rate,
            len(frames),
        )
        return utterance
