"""Text-to-speech for J.A.R.I.V.S using Piper (fully offline).

Loads a Piper voice once at startup and plays synthesized speech through the
default output device via sounddevice. The voice model downloads to
``tts.models_dir`` on first run; subsequent runs are fully offline.

If Piper can't initialise (missing model and no network, broken install),
:func:`build_speaker` degrades to the built-in Windows SAPI voice instead of
crashing the service — the assistant keeps talking, just less prettily.

Classification: audio synthesis + playback only — non-destructive; no
confirmation gate applies.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd

from ..config import TTSConfig

log = logging.getLogger(__name__)


class Speaker(Protocol):
    """Anything that can speak text aloud (blocking until playback ends)."""

    def speak(self, text: str) -> None: ...


def _resolve_voice_path(cfg: TTSConfig) -> Path:
    """Resolve the configured voice to an .onnx model path.

    ``tts.voice`` may be a bundled voice name (looked up under
    ``tts.models_dir``) or a direct path to a ``.onnx`` file.
    """
    if cfg.voice.endswith(".onnx"):
        return Path(cfg.voice)
    return Path(cfg.models_dir) / f"{cfg.voice}.onnx"


def _ensure_voice_available(cfg: TTSConfig, path: Path) -> None:
    """Download the named voice on first run (no-op if already present)."""
    if path.is_file():
        return
    if cfg.voice.endswith(".onnx"):
        raise FileNotFoundError(f"Configured Piper voice file not found: {path}")
    log.info("Piper voice '%s' not found locally — downloading to %s",
             cfg.voice, cfg.models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices",
         "--download-dir", str(path.parent), cfg.voice],
        check=True,
    )


class PiperSpeaker:
    """Offline TTS via a Piper voice model."""

    def __init__(self, cfg: TTSConfig) -> None:
        from piper import PiperVoice, SynthesisConfig

        voice_path = _resolve_voice_path(cfg)
        _ensure_voice_available(cfg, voice_path)
        log.info("Loading Piper voice %s", voice_path)
        self._voice = PiperVoice.load(str(voice_path))
        self._syn_config = SynthesisConfig(
            speaker_id=cfg.speaker_id if cfg.speaker_id > 0 else None,
            length_scale=1.0 / cfg.speaking_rate,  # rate>1 => shorter frames
            volume=cfg.volume,
        )
        log.info("Piper voice ready (rate=%.2f, volume=%.2f)",
                 cfg.speaking_rate, cfg.volume)

    def speak(self, text: str) -> None:
        """Synthesize ``text`` and play it; blocks until playback finishes."""
        text = text.strip()
        if not text:
            return
        chunks = list(self._voice.synthesize(text, self._syn_config))
        if not chunks:
            return
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        sd.play(audio, samplerate=chunks[0].sample_rate, blocking=True)


class SapiSpeaker:
    """Fallback TTS via the built-in Windows SAPI voice (no extra deps)."""

    def speak(self, text: str) -> None:
        """Speak via System.Speech through PowerShell; blocks until done."""
        text = text.strip()
        if not text:
            return
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Speak([Console]::In.ReadToEnd()); $s.Dispose()"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            input=text, text=True, check=True,
        )


def build_speaker(cfg: TTSConfig) -> Speaker:
    """Build the configured speaker, degrading gracefully on failure.

    Piper init can fail on a fresh machine with no network (voice not yet
    downloaded) or a broken install; a voice assistant that cannot speak is
    dead, so fall back to Windows SAPI rather than crash-looping the service.
    """
    if cfg.engine != "piper":
        raise ValueError(f"Unsupported tts.engine '{cfg.engine}' (expected 'piper')")
    try:
        return PiperSpeaker(cfg)
    except Exception:  # noqa: BLE001 — degrade, don't die
        log.exception("Piper TTS unavailable — falling back to Windows SAPI voice")
        return SapiSpeaker()


def _demo() -> None:
    """Standalone smoke test: python -m src.jarvis.tts.speaker "hello there" """
    from ..config import load_config
    from ..utils.logging_config import configure_logging

    cfg = load_config()
    configure_logging(cfg.logging)
    text = " ".join(sys.argv[1:]) or "Hello. Jarvis text to speech is working."
    build_speaker(cfg.tts).speak(text)


if __name__ == "__main__":
    _demo()
