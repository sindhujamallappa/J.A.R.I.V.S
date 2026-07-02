"""Tests for TTS voice-path resolution (no model load)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import TTSConfig
from jarvis.tts.speaker import _ensure_voice_available, _resolve_voice_path


def test_voice_name_resolves_under_models_dir():
    cfg = TTSConfig(voice="en_US-lessac-medium", models_dir="models/piper")
    assert _resolve_voice_path(cfg) == Path("models/piper/en_US-lessac-medium.onnx")


def test_voice_path_passes_through():
    cfg = TTSConfig(voice=r"C:\voices\custom.onnx")
    assert _resolve_voice_path(cfg) == Path(r"C:\voices\custom.onnx")


def test_missing_voice_file_path_raises(tmp_path: Path):
    missing = tmp_path / "nope.onnx"
    cfg = TTSConfig(voice=str(missing))
    with pytest.raises(FileNotFoundError):
        _ensure_voice_available(cfg, missing)


def test_present_voice_skips_download(tmp_path: Path):
    voice = tmp_path / "v.onnx"
    voice.touch()
    cfg = TTSConfig(voice=str(voice))
    _ensure_voice_available(cfg, voice)  # must not raise or shell out
