"""Tests for the action catalog and intent parser (mocked Ollama client)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jarvis.actions import ACTIONS, is_destructive, progress_phrase
from jarvis.config import IntentConfig
from jarvis.intent.parser import IntentError, IntentParser

# These tests exercise the LLM path and its validation; the regex fast path
# (tested in test_fast_path.py) would intercept simple phrases before the
# mocked client is ever consulted.
CFG = IntentConfig(fast_path=False)


class FakeClient:
    """Returns a canned response (or raises) instead of calling Ollama."""

    def __init__(self, content: str | None = None, exc: Exception | None = None) -> None:
        self._content = content
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return {"message": {"content": self._content}}


def _parser(content: str | None = None, exc: Exception | None = None) -> IntentParser:
    return IntentParser(CFG, client=FakeClient(content, exc))


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def test_destructive_classification_of_core_actions():
    assert is_destructive("delete_file")
    assert is_destructive("write_file")
    assert is_destructive("move_file")
    assert is_destructive("shutdown")
    assert is_destructive("restart")
    assert is_destructive("kill_process")
    assert is_destructive("run_command")
    assert not is_destructive("open_app")
    assert not is_destructive("web_search")
    assert not is_destructive("read_file")


def test_unrecognised_action_is_destructive_by_default():
    assert is_destructive("format_disk")  # fail-safe


def test_extra_destructive_actions_override():
    assert not is_destructive("close_app")
    assert is_destructive("close_app", extra_destructive=("close_app",))


def test_progress_phrase_fills_template():
    phrase = progress_phrase("answer_question", {"query": "next f1 race"})
    expected = {t.format(query="next f1 race")
                for t in ACTIONS["answer_question"].progress}
    assert phrase in expected


def test_progress_phrase_empty_for_fast_or_unknown_actions():
    assert progress_phrase("open_app", {"name": "notepad"}) == ""
    assert progress_phrase("warp_drive", {}) == ""
    # A missing template param must yield silence, never a crash.
    assert progress_phrase("answer_question", {}) == ""


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def test_parses_valid_action():
    got = _parser('{"action": "open_app", "params": {"name": "notepad"}}').parse("open notepad")
    assert got.action == "open_app"
    assert got.params == {"name": "notepad"}
    assert got.raw_text == "open notepad"


def test_action_name_is_normalised():
    got = _parser('{"action": " Delete_File ", "params": {"path": "x.txt"}}').parse("delete x")
    assert got.action == "delete_file"


def test_unsupported_action_falls_back_to_unknown():
    got = _parser('{"action": "format_disk", "params": {}}').parse("format the disk")
    assert got.action == "unknown"
    assert "format_disk" in got.params["reason"]


def test_unparseable_output_falls_back_to_unknown():
    got = _parser("I think you want to open notepad!").parse("open notepad")
    assert got.action == "unknown"


def test_json_extracted_from_surrounding_prose():
    content = 'Sure! {"action": "get_time", "params": {}} Hope that helps.'
    got = _parser(content).parse("what time is it")
    assert got.action == "get_time"


def test_missing_required_param_falls_back_to_unknown():
    got = _parser('{"action": "delete_file", "params": {}}').parse("delete it")
    assert got.action == "unknown"
    assert "path" in got.params["reason"]


def test_unspecced_params_are_dropped():
    content = json.dumps({
        "action": "open_app",
        "params": {"name": "notepad", "force": "true", "admin": "yes"},
    })
    got = _parser(content).parse("open notepad")
    assert got.params == {"name": "notepad"}


def test_param_values_coerced_to_str():
    got = _parser('{"action": "set_volume", "params": {"level": 50}}').parse("volume 50")
    assert got.params == {"level": "50"}


def test_empty_transcript_short_circuits_without_llm_call():
    client = FakeClient('{"action": "get_time", "params": {}}')
    got = IntentParser(CFG, client=client).parse("   ")
    assert got.action == "unknown"
    assert client.calls == []


def test_backend_failure_raises_intent_error():
    with pytest.raises(IntentError):
        _parser(exc=ConnectionError("refused")).parse("open notepad")


def test_every_prompt_example_action_exists_in_catalog():
    # Guards against catalog renames silently breaking the few-shot prompt.
    from jarvis.intent.parser import _build_system_prompt

    prompt = _build_system_prompt()
    for name in ("open_app", "delete_file", "web_search", "get_news",
                 "answer_question", "respond", "unknown"):
        assert name in ACTIONS and name in prompt


# --------------------------------------------------------------------------- #
# Spoken answers from web snippets (answer_question support)
# --------------------------------------------------------------------------- #
def test_answer_composes_from_snippets():
    client = FakeClient("The next race is the British Grand Prix on July 5.")
    parser = IntentParser(CFG, client=client)
    answer = parser.answer("when is the next f1 race", ["F1 calendar: July 5 Silverstone"])
    assert answer == "The next race is the British Grand Prix on July 5."
    # Plain-text generation: JSON mode must NOT be requested.
    assert "format" not in client.calls[0]
    assert "July 5 Silverstone" in client.calls[0]["messages"][1]["content"]


def test_answer_returns_none_on_backend_failure():
    parser = _parser(exc=ConnectionError("refused"))
    assert parser.answer("anything", ["snippet"]) is None


def test_answer_returns_none_on_empty_output():
    assert _parser("   ").answer("anything", ["snippet"]) is None
