"""Tests for the background-service layer (supervisor, XML, instance guard)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import ServiceConfig
from jarvis.service import (
    SingleInstance,
    _mutex_name,
    _supervise,
    build_task_xml,
)


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #
def test_supervise_returns_zero_on_clean_stop():
    calls = {"n": 0}

    def run_once():
        calls["n"] += 1  # returns normally = clean stop

    assert _supervise(run_once, ServiceConfig(), sleep=lambda _s: None) == 0
    assert calls["n"] == 1


def test_supervise_restarts_after_crash_then_succeeds():
    seq = iter([RuntimeError("boom"), RuntimeError("boom again"), None])
    slept: list[float] = []

    def run_once():
        exc = next(seq)
        if exc:
            raise exc

    code = _supervise(
        run_once,
        ServiceConfig(restart_backoff_sec=1.0, restart_backoff_max_sec=10.0),
        sleep=slept.append,
        max_restarts=5,
    )
    assert code == 0
    assert slept == [1.0, 2.0]  # exponential backoff between the two crashes


def test_supervise_gives_up_when_restart_disabled():
    def run_once():
        raise RuntimeError("crash")

    code = _supervise(
        run_once, ServiceConfig(restart_on_crash=False), sleep=lambda _s: None
    )
    assert code == 1


def test_supervise_respects_max_restarts():
    def run_once():
        raise RuntimeError("always")

    code = _supervise(
        run_once, ServiceConfig(), sleep=lambda _s: None, max_restarts=3
    )
    assert code == 1


# --------------------------------------------------------------------------- #
# Task XML
# --------------------------------------------------------------------------- #
def test_task_xml_contains_key_fields():
    xml = build_task_xml(
        task_name="JARVIS Voice Assistant",
        command=r"C:\venv\Scripts\pythonw.exe",
        arguments="-m src.jarvis.service run",
        working_dir=r"C:\dev\JARVIS",
        user_id="Sindhuja.M",
    )
    assert "<LogonTrigger>" in xml
    assert "pythonw.exe" in xml
    assert "-m src.jarvis.service run" in xml
    assert "IgnoreNew" in xml
    assert "<RestartOnFailure>" in xml
    assert "Sindhuja.M" in xml


def test_task_xml_escapes_values():
    xml = build_task_xml(
        task_name="t", command="c", arguments="a & b",
        working_dir="w", user_id="u",
    )
    assert "a &amp; b" in xml  # ampersand escaped


def test_mutex_name_sanitised():
    assert _mutex_name("JARVIS Voice Assistant") == "Local\\JARVIS_JARVIS_Voice_Assistant"


# --------------------------------------------------------------------------- #
# Single-instance guard (POSIX flock path, exercised on Linux CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    __import__("os").name == "nt", reason="flock path is POSIX-only"
)
def test_single_instance_blocks_second_holder(tmp_path: Path):
    lock = tmp_path / "test.lock"
    first = SingleInstance("Local\\JARVIS_test", lock)
    second = SingleInstance("Local\\JARVIS_test", lock)
    try:
        assert first.acquire() is True
        assert second.acquire() is False  # already held
    finally:
        first.release()
        second.release()

    # After release, a fresh holder can acquire again.
    third = SingleInstance("Local\\JARVIS_test", lock)
    try:
        assert third.acquire() is True
    finally:
        third.release()
