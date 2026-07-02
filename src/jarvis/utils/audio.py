"""Shared audio helpers for J.A.R.I.V.S.

Currently provides microphone capture as a context manager that yields fixed
-length int16 frames — the format both openWakeWord and faster-whisper expect
(16 kHz, mono, signed 16-bit PCM). Playback helpers for TTS land in a later
stage.

Classification: microphone capture is *input only*. It performs no filesystem
or process operations and is non-destructive; no confirmation gate applies.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Optional

import numpy as np
import sounddevice as sd

from ..config import AudioConfig

log = logging.getLogger(__name__)


def play_tone(
    frequency_hz: float = 880.0,
    duration_sec: float = 0.12,
    volume: float = 0.2,
    sample_rate: int = 16000,
) -> None:
    """Play a short sine 'earcon' (listening cue); blocks until done.

    Classification: audio output only — non-destructive.
    """
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = np.sin(2.0 * np.pi * frequency_hz * t) * volume
    # Fade edges to avoid clicks.
    fade = max(1, min(len(tone) // 10, int(0.01 * sample_rate)))
    envelope = np.ones_like(tone)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    sd.play((tone * envelope * 32767).astype(np.int16), samplerate=sample_rate, blocking=True)


def resolve_input_device(device: Optional[str]) -> Optional[int]:
    """Resolve a device spec to a sounddevice input-device index.

    Args:
        device: ``None`` for the system default, an int-like string for a
            device index, or a substring to match against device names.

    Returns:
        The resolved device index, or ``None`` to use the system default.

    Raises:
        ValueError: If a name substring matches no input device.
    """
    if device is None or device == "":
        return None

    # Accept an explicit integer index (possibly arriving as a string).
    try:
        return int(device)
    except (TypeError, ValueError):
        pass

    needle = str(device).lower()
    for idx, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0 and needle in info["name"].lower():
            log.info("Resolved input device %r -> [%d] %s", device, idx, info["name"])
            return idx
    raise ValueError(f"No input device matching {device!r}")


class MicrophoneStream:
    """Blocking microphone reader yielding fixed-length int16 frames.

    Use as a context manager so the underlying stream is always closed::

        with MicrophoneStream(cfg.audio) as mic:
            frame = mic.read()   # np.ndarray, shape (frame_length,), int16
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self._sample_rate = cfg.sample_rate
        self._frame_length = cfg.frame_length
        self._channels = cfg.channels
        self._device = resolve_input_device(cfg.input_device)
        self._stream: Optional[sd.InputStream] = None

    @property
    def frame_length(self) -> int:
        """Samples returned per :meth:`read` call."""
        return self._frame_length

    def __enter__(self) -> "MicrophoneStream":
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._frame_length,
            channels=self._channels,
            dtype="int16",
            device=self._device,
        )
        self._stream.start()
        log.info(
            "Microphone open (rate=%d Hz, frame=%d, channels=%d, device=%s)",
            self._sample_rate,
            self._frame_length,
            self._channels,
            self._device if self._device is not None else "default",
        )
        return self

    def read(self) -> np.ndarray:
        """Read one frame of audio.

        Returns:
            A 1-D ``int16`` array of length ``frame_length`` (mono). For stereo
            input the two channels are averaged down to mono.

        Raises:
            RuntimeError: If called outside the context manager.
        """
        if self._stream is None:
            raise RuntimeError("MicrophoneStream must be used as a context manager")

        data, overflowed = self._stream.read(self._frame_length)
        if overflowed:
            log.warning("Audio input overflow — frame(s) may have been dropped")

        if self._channels == 1:
            return data.reshape(-1)
        # Downmix to mono without overflowing int16.
        return data.mean(axis=1).astype(np.int16)

    def flush(self) -> None:
        """Discard everything currently buffered so the next read is live.

        Call after TTS playback (the mic hears the speakers) or any long
        pause in reading — otherwise stale audio, including the assistant's
        own voice, is what gets captured next.

        Raises:
            RuntimeError: If called outside the context manager.
        """
        if self._stream is None:
            raise RuntimeError("MicrophoneStream must be used as a context manager")
        dropped = 0
        while self._stream.read_available >= self._frame_length:
            self._stream.read(self._frame_length)
            dropped += 1
        if dropped:
            log.debug("Flushed %d stale audio frame(s)", dropped)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Microphone closed")
