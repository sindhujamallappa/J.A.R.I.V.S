"""Tests for input-device resolution (host-API ranking, WDM-KS fallback)."""

from __future__ import annotations

import pytest

import jarvis.utils.audio as audio_mod
from jarvis.utils.audio import resolve_input_device

_HOSTAPIS = (
    {"name": "MME"},
    {"name": "Windows WASAPI"},
    {"name": "Windows WDM-KS"},
)


def _patch_devices(monkeypatch: pytest.MonkeyPatch, devices: list[dict]) -> None:
    monkeypatch.setattr(audio_mod.sd, "query_devices", lambda: devices)
    monkeypatch.setattr(audio_mod.sd, "query_hostapis", lambda: _HOSTAPIS)


def test_none_and_empty_use_default():
    assert resolve_input_device(None) is None
    assert resolve_input_device("") is None


def test_integer_string_is_an_index():
    assert resolve_input_device("3") == 3


def test_prefers_mme_over_wdmks(monkeypatch: pytest.MonkeyPatch):
    _patch_devices(monkeypatch, [
        {"name": "Headset (Jabra) KS", "max_input_channels": 1, "hostapi": 2},
        {"name": "Headset Microphone (Jabra Link)", "max_input_channels": 1, "hostapi": 0},
        {"name": "Headset Microphone (Jabra Link)", "max_input_channels": 2, "hostapi": 1},
    ])
    assert resolve_input_device("jabra") == 1  # the MME entry wins


def test_wdmks_only_falls_back_to_default(monkeypatch: pytest.MonkeyPatch):
    _patch_devices(monkeypatch, [
        {"name": "Headset (Jabra BT Hands-Free)", "max_input_channels": 1, "hostapi": 2},
        {"name": "Speakers (Realtek)", "max_input_channels": 0, "hostapi": 0},
    ])
    assert resolve_input_device("jabra") is None  # loud warning, default mic


def test_no_match_raises(monkeypatch: pytest.MonkeyPatch):
    _patch_devices(monkeypatch, [
        {"name": "Microphone Array (Realtek)", "max_input_channels": 4, "hostapi": 0},
    ])
    with pytest.raises(ValueError, match="No input device"):
        resolve_input_device("jabra")


def test_output_only_devices_ignored(monkeypatch: pytest.MonkeyPatch):
    _patch_devices(monkeypatch, [
        {"name": "Headset Earphone (Jabra)", "max_input_channels": 0, "hostapi": 0},
        {"name": "Headset Microphone (Jabra)", "max_input_channels": 1, "hostapi": 1},
    ])
    assert resolve_input_device("jabra") == 1
