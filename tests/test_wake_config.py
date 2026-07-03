"""Tests for multi-model wake-word configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import ConfigError, load_config

_BASE_YAML = """
wake_word:
  models: {models}
"""


def _write(tmp_path: Path, models_yaml: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_BASE_YAML.format(models=models_yaml), encoding="utf-8")
    return cfg


def test_multiple_bundled_models_accepted(tmp_path: Path):
    cfg = load_config(_write(tmp_path, '["hey_jarvis", "hey_mycroft"]'))
    assert cfg.wake_word.models == ["hey_jarvis", "hey_mycroft"]


def test_custom_model_path_must_exist(tmp_path: Path):
    with pytest.raises(ConfigError, match="file not found"):
        load_config(_write(tmp_path, '["hey_jarvis", "models/wake/nope.onnx"]'))


def test_existing_custom_model_path_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    custom = tmp_path / "custom.onnx"
    custom.touch()
    monkeypatch.chdir(tmp_path)  # relative path resolution
    cfg = load_config(_write(tmp_path, f'["hey_jarvis", "{custom.name}"]'))
    assert custom.name in cfg.wake_word.models[1]


def test_empty_models_list_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="at least one model"):
        load_config(_write(tmp_path, "[]"))


def test_old_singular_model_key_rejected(tmp_path: Path):
    # The old key must fail loudly (strict sections), not be silently ignored.
    cfg = tmp_path / "config.yaml"
    cfg.write_text("wake_word:\n  model: hey_jarvis\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(cfg)


def test_env_override_splits_commas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JARVIS_WAKE_MODELS", "hey_jarvis, hey_mycroft")
    cfg = load_config(_write(tmp_path, '["hey_jarvis"]'))
    assert cfg.wake_word.models == ("hey_jarvis", "hey_mycroft")
