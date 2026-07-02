"""Tests for the executor — file handlers on tmp_path, system calls mocked."""

from __future__ import annotations

from pathlib import Path

import pytest

import jarvis.execution.executor as executor_mod
from jarvis.config import ExecutionConfig
from jarvis.execution.executor import ExecutionResult, Executor, _resolve_path
from jarvis.intent.parser import Intent


def _run(action: str, cfg: ExecutionConfig = ExecutionConfig(), **params: str) -> ExecutionResult:
    return Executor(cfg).execute(Intent(action, dict(params), f"test {action}"))


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def test_resolve_path_home_relative():
    assert _resolve_path("Desktop/x.txt") == Path.home() / "Desktop/x.txt"


def test_resolve_path_absolute_passthrough(tmp_path: Path):
    assert _resolve_path(str(tmp_path)) == tmp_path


# --------------------------------------------------------------------------- #
# File handlers (non-destructive group)
# --------------------------------------------------------------------------- #
def test_read_file_truncates(tmp_path: Path):
    f = tmp_path / "long.txt"
    f.write_text("x" * 5000, encoding="utf-8")
    result = _run("read_file", ExecutionConfig(read_file_max_chars=100), path=str(f))
    assert result.ok
    assert result.message.startswith("x" * 100)
    assert "continues" in result.message


def test_read_file_missing(tmp_path: Path):
    result = _run("read_file", path=str(tmp_path / "nope.txt"))
    assert not result.ok


def test_list_folder(tmp_path: Path):
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    result = _run("list_folder", path=str(tmp_path))
    assert result.ok
    assert "a.txt" in result.message and "b.txt" in result.message


def test_create_folder_refuses_existing(tmp_path: Path):
    target = tmp_path / "new"
    assert _run("create_folder", path=str(target)).ok
    assert target.is_dir()
    assert not _run("create_folder", path=str(target)).ok  # create-only


def test_copy_file_never_overwrites(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("data", encoding="utf-8")
    dst = tmp_path / "dst.txt"
    assert _run("copy_file", source=str(src), destination=str(dst)).ok
    assert dst.read_text(encoding="utf-8") == "data"
    # Second copy onto the existing destination must refuse (no-clobber).
    result = _run("copy_file", source=str(src), destination=str(dst))
    assert not result.ok
    assert "overwrite" in result.message


# --------------------------------------------------------------------------- #
# File handlers (destructive group — exercised directly here; in the real
# pipeline these are only reached after the gate's spoken confirmation)
# --------------------------------------------------------------------------- #
def test_write_file_overwrites(tmp_path: Path):
    f = tmp_path / "out.txt"
    f.write_text("old", encoding="utf-8")
    assert _run("write_file", path=str(f), content="new").ok
    assert f.read_text(encoding="utf-8") == "new"


def test_move_file(tmp_path: Path):
    src = tmp_path / "a.txt"
    src.write_text("data", encoding="utf-8")
    dst = tmp_path / "sub" / "b.txt"
    assert _run("move_file", source=str(src), destination=str(dst)).ok
    assert not src.exists()
    assert dst.read_text(encoding="utf-8") == "data"


def test_delete_file_permanent_when_trash_disabled(tmp_path: Path):
    f = tmp_path / "doomed.txt"
    f.touch()
    result = _run("delete_file", ExecutionConfig(delete_to_trash=False), path=str(f))
    assert result.ok
    assert not f.exists()


def test_delete_missing_file(tmp_path: Path):
    result = _run("delete_file", path=str(tmp_path / "ghost.txt"))
    assert not result.ok


# --------------------------------------------------------------------------- #
# Shell gate
# --------------------------------------------------------------------------- #
def test_run_command_refused_by_default():
    result = _run("run_command", command="echo hi")
    assert not result.ok
    assert "disabled" in result.message


def test_run_command_when_enabled():
    result = _run("run_command", ExecutionConfig(allow_run_command=True), command="echo hi")
    assert result.ok
    assert "hi" in result.message


# --------------------------------------------------------------------------- #
# Web handlers (browser mocked)
# --------------------------------------------------------------------------- #
def test_open_url_adds_scheme(monkeypatch: pytest.MonkeyPatch):
    opened: list[str] = []
    monkeypatch.setattr(executor_mod.webbrowser, "open", opened.append)
    result = _run("open_url", url="example.com")
    assert result.ok
    assert opened == ["https://example.com"]


def test_web_search_quotes_query(monkeypatch: pytest.MonkeyPatch):
    opened: list[str] = []
    monkeypatch.setattr(executor_mod.webbrowser, "open", opened.append)
    result = _run("web_search", query="hello world")
    assert result.ok
    assert opened == ["https://www.google.com/search?q=hello+world"]


# --------------------------------------------------------------------------- #
# Conversation / fallback
# --------------------------------------------------------------------------- #
def test_respond_passes_text_through():
    result = _run("respond", text="Hello there.")
    assert result.ok
    assert result.message == "Hello there."


def test_unknown_action_is_spoken_refusal():
    result = _run("unknown", reason="nonsense")
    assert not result.ok


def test_unhandled_action_name_fails_safely():
    result = _run("warp_drive")
    assert not result.ok


def test_get_time_speaks_a_time():
    result = _run("get_time")
    assert result.ok
    assert "It's" in result.message


def test_handler_exception_is_contained(monkeypatch: pytest.MonkeyPatch):
    ex = Executor(ExecutionConfig())
    monkeypatch.setitem(
        ex._handlers, "get_time",
        lambda p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = ex.execute(Intent("get_time", {}, "time"))
    assert not result.ok
    assert "failed" in result.message
