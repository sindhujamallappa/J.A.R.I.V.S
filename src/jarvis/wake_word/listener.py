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

import openwakeword
from openwakeword.model import Model

from ..config import AudioConfig, WakeWordConfig, _looks_like_path
from ..utils.audio import MicrophoneStream

log = logging.getLogger(__name__)


class WakeWordDetection(NamedTuple):
    """A single wake-word trigger."""

    model_name: str
    score: float


def _ensure_models_available(model: str) -> None:
    """Ensure openWakeWord's shared feature models (and a bundled wake model,
    if named) are downloaded locally.

    openWakeWord always needs its shared melspectrogram / embedding / VAD
    models. A bundled wake model (e.g. ``hey_jarvis``) is fetched by name;
    custom model files are the caller's responsibility (validated in config).

    Network is only used on first run; subsequent starts are fully offline.
    """
    try:
        if _looks_like_path(model):
            # Custom file: only the shared feature models are needed.
            openwakeword.utils.download_models()
        else:
            try:
                openwakeword.utils.download_models(model_names=[model])
            except TypeError:
                # Older openWakeWord: no model_names kwarg.
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
        self._score_key = self._expected_score_key(cfg.model)
        self._model: Optional[Model] = None
        self._external_mic = mic
        self._mic: Optional[MicrophoneStream] = None
        self._owns_mic = mic is None
        self._last_trigger = 0.0

    @staticmethod
    def _expected_score_key(model: str) -> str:
        """The key openWakeWord uses for this model in ``predict`` output."""
        if _looks_like_path(model):
            return Path(model).stem
        return model

    def _build_model(self) -> Model:
        _ensure_models_available(self._cfg.model)
        wakeword_models = [self._cfg.model]  # bundled name or file path
        log.info(
            "Loading wake-word model '%s' (framework=%s, threshold=%.2f)",
            self._cfg.model,
            self._cfg.inference_framework,
            self._threshold,
        )
        return Model(
            wakeword_models=wakeword_models,
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
        log.info("Wake-word listener ready — say 'Hey Jarvis'")
        return self

    def _score_frame(self, frame) -> float:
        """Run one frame through the model and return our model's score."""
        assert self._model is not None
        scores = self._model.predict(frame)
        if self._score_key in scores:
            return float(scores[self._score_key])
        # Fallback: single model loaded, take the max score present.
        return float(max(scores.values())) if scores else 0.0

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

        while True:
            frame = self._mic.read()
            score = self._score_frame(frame)
            now = time.monotonic()
            if score >= self._threshold and (now - self._last_trigger) >= self._cooldown:
                self._last_trigger = now
                self._model.reset()  # clear buffers to prevent immediate re-fire
                log.info("Wake word detected (score=%.3f)", score)
                yield WakeWordDetection(self._score_key, score)

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
