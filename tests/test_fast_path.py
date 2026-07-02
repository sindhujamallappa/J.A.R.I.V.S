"""Tests for the rule-based intent fast path (must never guess)."""

from __future__ import annotations

import pytest

from jarvis.config import IntentConfig
from jarvis.intent.fast_path import match
from jarvis.intent.parser import IntentParser


# --------------------------------------------------------------------------- #
# Positive matches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,action,params", [
    ("What time is it?", "get_time", {}),
    ("Hey Jarvis, what's the time", "get_time", {}),
    ("lock my screen", "lock_screen", {}),
    ("Lock the computer, please.", "lock_screen", {}),
    ("mute", "mute", {}),
    ("unmute the sound", "unmute", {}),
    ("set volume to 40 percent", "set_volume", {"level": "40"}),
    ("Volume 75", "set_volume", {"level": "75"}),
    ("search for weather in Bangalore", "web_search", {"query": "weather in bangalore"}),
    ("google llama 3", "web_search", {"query": "llama 3"}),
    ("open github.com", "open_url", {"url": "github.com"}),
    ("Open Notepad", "open_app", {"name": "notepad"}),
    ("launch spotify please", "open_app", {"name": "spotify"}),
    ("close notepad", "close_app", {"name": "notepad"}),
])
def test_fast_path_matches(text: str, action: str, params: dict[str, str]):
    intent = match(text)
    assert intent is not None, f"expected a match for {text!r}"
    assert intent.action == action
    assert intent.params == params
    assert intent.raw_text == text


# --------------------------------------------------------------------------- #
# Must fall through to the LLM (None)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "delete report.txt from my desktop",     # destructive — LLM + gate
    "shut down the computer",                # destructive
    "open the file notes.txt",               # file, not an app
    "open my documents folder",              # folder, not an app
    "make me a coffee",                      # nonsense
    "hey how are you doing",                 # chit-chat
    "search",                                # no query
    "volume up a bit",                       # no number
    "",                                      # empty
])
def test_fast_path_declines(text: str):
    assert match(text) is None


# --------------------------------------------------------------------------- #
# Parser integration
# --------------------------------------------------------------------------- #
class _ExplodingClient:
    def chat(self, **kwargs):
        raise AssertionError("LLM must not be called for fast-path commands")


def test_parser_uses_fast_path_without_llm():
    parser = IntentParser(IntentConfig(fast_path=True), client=_ExplodingClient())
    intent = parser.parse("what time is it")
    assert intent.action == "get_time"


def test_parser_skips_fast_path_when_disabled():
    class Canned:
        def chat(self, **kwargs):
            return {"message": {"content": '{"action": "get_time", "params": {}}'}}

    parser = IntentParser(IntentConfig(fast_path=False), client=Canned())
    assert parser.parse("what time is it").action == "get_time"
