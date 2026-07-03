"""Wake-word detection for J.A.R.I.V.S using openWakeWord.

Listens to the microphone and fires when it hears "Hey Jarvis". Uses the
bundled ``hey_jarvis`` pretrained model by default; a custom model path can be
set in config. Runs fully on-device via ONNX (or tflite).

Classification: this stage is *input only* — it reads the microphone and runs
local inference. It performs no filesystem mutations or process control, so no
destructive-action confirmation gate applies here. The gate lives in the
execution layer.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Iterator, NamedTuple, Optional

import numpy as np
import openwakeword
from openwakeword.model import Model

from ..config import AudioConfig, WakeWordConfig, _looks_like_path
from ..ui.events import BUS
from ..utils.audio import MicrophoneStream

# Publish a HUD meter tick every Nth frame (~3 Hz at 80 ms frames).
_METER_EVERY_N_FRAMES = 4
# int16 RMS that maps to a "full" meter — normal speech peaks around here.
_METER_FULL_SCALE_RMS = 3000.0
# Log (throttled) when a score lands here-to-threshold: helps tune the
# threshold and proves the mic hears *something* wake-word-shaped.
_NEAR_MISS_FLOOR = 0.2
_NEAR_MISS_LOG_INTERVAL_SEC = 2.0

log = logging.getLogger(__name__)


class WakeWordDetection(NamedTuple):
    """A single wake-word trigger."""

    model_name: str
    score: float


def _ensure_models_available(models: tuple[str, ...]) -> None:
    """Ensure openWakeWord's shared feature models (and any bundled wake
    models named) are downloaded locally.

    openWakeWord always needs its shared melspectrogram / embedding / VAD
    models. Bundled wake models (e.g. ``hey_jarvis``) are fetched by name;
    custom model files are the caller's responsibility (validated in config).

    Network is only used on first run; subsequent starts are fully offline.
    """
    bundled = [m for m in models if not _looks_like_path(m)]
    try:
        if bundled:
            try:
                openwakeword.utils.download_models(model_names=bundled)
            except TypeError:
                # Older openWakeWord: no model_names kwarg.
                openwakeword.utils.download_models()
        else:
            # Custom files only: just the shared feature models are needed.
            openwakeword.utils.download_models()
    except Exception as exc:  # network down, host unreachable, etc.
        log.warning(
            "Could not verify/download openWakeWord models (%s). "
            "If this is the first run, connect once to fetch them.",
            exc,
        )


class WakeWordListener:
    """Detects the "Hey Jarvis" wake word from the default microphone.

    Use as a context manager to hold the model and mic open::

        with WakeWordListener(cfg.wake_word, cfg.audio) as ww:
            detection = ww.wait()          # blocks until "Hey Jarvis"
            # ... hand control to STT ...

    or iterate continuously::

        with WakeWordListener(cfg.wake_word, cfg.audio) as ww:
            for detection in ww.stream():
                ...
    """

    def __init__(
        self,
        cfg: WakeWordConfig,
        audio: AudioConfig,
        mic: Optional[MicrophoneStream] = None,
    ) -> None:
        """Args:
            cfg: Wake-word configuration.
            audio: Audio/frame configuration.
            mic: An already-open microphone stream to share (the orchestrator
                passes one so STT capture reads the same device). ``None``
                means the listener opens and owns its own.
        """
        self._cfg = cfg
        self._audio = audio
        self._threshold = cfg.threshold
        self._cooldown = cfg.trigger_cooldown_sec
        # Score-dict keys for each configured model, in config order.
        self._score_keys = [self._expected_score_key(m) for m in cfg.models]
        self._model: Optional[Model] = None
        self._external_mic = mic
        self._mic: Optional[MicrophoneStream] = None
        self._owns_mic = mic is None
        self._last_trigger = 0.0
        self._last_near_miss = 0.0

    @staticmethod
    def _expected_score_key(model: str) -> str:
        """The key openWakeWord uses for this model in ``predict`` output."""
        if _looks_like_path(model):
            return Path(model).stem
        return model

    def _build_model(self) -> Model:
        _ensure_models_available(self._cfg.models)
        log.info(
            "Loading wake-word model(s) %s (framework=%s, threshold=%.2f)",
            ", ".join(repr(m) for m in self._cfg.models),
            self._cfg.inference_framework,
            self._threshold,
        )
        return Model(
            wakeword_models=list(self._cfg.models),  # names and/or file paths
            inference_framework=self._cfg.inference_framework,
            vad_threshold=self._cfg.vad_threshold,
            enable_speex_noise_suppression=self._cfg.enable_noise_suppression,
        )

    def __enter__(self) -> "WakeWordListener":
        self._model = self._build_model()
        if self._external_mic is not None:
            self._mic = self._external_mic
        else:
            self._mic = MicrophoneStream(self._audio).__enter__()
        log.info("Wake-word listener ready — %d wake phrase(s) armed", len(self._score_keys))
        return self

    def _score_frame(self, frame) -> tuple[str, float]:
        """Run one frame through the models; return the best (name, score)."""
        assert self._model is not None
        scores = self._model.predict(frame)
        best_key, best = "", 0.0
        for key in self._score_keys:
            value = float(scores.get(key, 0.0))
            if value >= best:
                best_key, best = key, value
        if not best_key and scores:
            # Fallback: key mismatch (openWakeWord renamed it) — take the max.
            best_key = max(scores, key=scores.get)  # type: ignore[arg-type]
            best = float(scores[best_key])
        return best_key, best

    def stream(self) -> Iterator[WakeWordDetection]:
        """Yield a :class:`WakeWordDetection` each time the wake word fires.

        Applies a rising-edge cooldown so a single utterance triggers once.

        Yields:
            WakeWordDetection: on each detection above threshold.

        Raises:
            RuntimeError: If used outside the context manager.
        """
        if self._model is None or self._mic is None:
            raise RuntimeError("WakeWordListener must be used as a context manager")

        frame_count = 0
        while True:
            frame = self._mic.read()
            name, score = self._score_frame(frame)
            frame_count += 1
            if frame_count % _METER_EVERY_N_FRAMES == 0:
                rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))
                BUS.publish(
                    "meter",
                    score=round(score, 4),
                    level=round(min(1.0, rms / _METER_FULL_SCALE_RMS), 3),
                )
            now = time.monotonic()
            if score >= self._threshold and (now - self._last_trigger) >= self._cooldown:
                self._last_trigger = now
                self._model.reset()  # clear buffers to prevent immediate re-fire
                log.info("Wake word %r detected (score=%.3f)", name, score)
                yield WakeWordDetection(name, score)
            elif (
                score >= _NEAR_MISS_FLOOR
                and (now - self._last_near_miss) >= _NEAR_MISS_LOG_INTERVAL_SEC
            ):
                self._last_near_miss = now
                log.info(
                    "Near miss: %r scored %.2f below threshold %.2f — "
                    "speak closer/clearer, or lower wake_word.threshold",
                    name, score, self._threshold,
                )

    def wait(self) -> WakeWordDetection:
        """Block until the wake word is detected once, then return.

        Returns:
            The :class:`WakeWordDetection` for the trigger.
        """
        return next(self.stream())

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if self._mic is not None:
            if self._owns_mic:
                self._mic.__exit__(exc_type, exc, tb)
            self._mic = None
        self._model = None
        log.info("Wake-word listener stopped")


def _demo() -> None:
    """Standalone smoke test: python -m src.jarvis.wake_word.listener"""
    from ..config import load_config
    from ..utils.logging_config import configure_logging

    cfg = load_config()
    configure_logging(cfg.logging)
    with WakeWordListener(cfg.wake_word, cfg.audio) as ww:
        try:
            for detection in ww.stream():
                log.info("→ triggered by %s @ %.3f", detection.model_name, detection.score)
        except KeyboardInterrupt:
            log.info("Interrupted — shutting down")


if __name__ == "__main__":
    _demo()
