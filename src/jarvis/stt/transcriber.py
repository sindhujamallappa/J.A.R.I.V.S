"""Speech-to-text for J.A.R.I.V.S using faster-whisper.

Loads the Whisper model once (eagerly, at startup — model load takes seconds
and must not happen per-utterance) and transcribes captured int16 utterances.
Model files download to ``stt.models_dir`` on first run; subsequent runs are
fully offline.

Classification: local inference only — non-destructive; no confirmation gate
applies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from ..config import STTConfig

log = logging.getLogger(__name__)

_INT16_SCALE = 32768.0


@dataclass(frozen=True)
class Transcription:
    """Result of transcribing one utterance."""

    text: str
    language: str
    duration_sec: float


class Transcriber:
    """One-shot command transcription via faster-whisper."""

    def __init__(self, cfg: STTConfig) -> None:
        self._cfg = cfg
        log.info(
            "Loading Whisper model '%s' (device=%s, compute=%s)",
            cfg.model_size, cfg.device, cfg.compute_type,
        )
        start = time.perf_counter()
        self._model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            download_root=cfg.models_dir,
        )
        log.info("Whisper model ready in %.1fs", time.perf_counter() - start)

    def transcribe(self, audio: np.ndarray) -> Transcription:
        """Transcribe one utterance.

        Args:
            audio: 1-D int16 mono audio at the configured sample rate (16 kHz).

        Returns:
            The recognised text (may be empty if nothing intelligible).
        """
        float_audio = audio.astype(np.float32) / _INT16_SCALE
        start = time.perf_counter()
        segments, info = self._model.transcribe(
            float_audio,
            language=self._cfg.language or None,  # "" -> autodetect
            beam_size=self._cfg.beam_size,
            # One-shot commands: don't seed the decoder with prior text, which
            # otherwise invites hallucinated carry-over between utterances.
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.info(
            "Transcribed %.2fs of audio in %.2fs -> %r",
            info.duration, time.perf_counter() - start, text,
        )
        return Transcription(text=text, language=info.language, duration_sec=info.duration)


def _demo() -> None:
    """Standalone smoke test: python -m src.jarvis.stt.transcriber

    Captures one utterance from the default mic and prints the transcription.
    """
    from ..config import load_config
    from ..utils.audio import MicrophoneStream
    from ..utils.logging_config import configure_logging
    from .capture import UtteranceCapturer

    cfg = load_config()
    configure_logging(cfg.logging)
    transcriber = Transcriber(cfg.stt)
    capturer = UtteranceCapturer(cfg.stt, cfg.audio)
    with MicrophoneStream(cfg.audio) as mic:
        log.info("Speak your command now…")
        utterance = capturer.capture(mic)
    if utterance is None:
        log.warning("No speech detected — nothing to transcribe")
        return
    result = transcriber.transcribe(utterance)
    log.info("→ %r (lang=%s)", result.text, result.language)


if __name__ == "__main__":
    _demo()
