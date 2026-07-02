"""Tests for the spoken-confirmation gate — the destructive-action guardrail."""

from __future__ import annotations

from typing import Optional

from jarvis.actions import describe
from jarvis.config import SafetyConfig
from jarvis.intent.parser import Intent
from jarvis.safety.gate import ConfirmationGate, classify_reply

CFG = SafetyConfig()


class FakeAsk:
    """Scripted replies; records every prompt spoken."""

    def __init__(self, *replies: Optional[str]) -> None:
        self._replies = iter(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Optional[str]:
        self.prompts.append(prompt)
        return next(self._replies, None)


def _gate(*replies: Optional[str], cfg: SafetyConfig = CFG) -> tuple[ConfirmationGate, FakeAsk]:
    ask = FakeAsk(*replies)
    return ConfirmationGate(cfg, ask), ask


DELETE = Intent("delete_file", {"path": "report.txt"}, "delete report.txt")
OPEN = Intent("open_app", {"name": "notepad"}, "open notepad")


# --------------------------------------------------------------------------- #
# Reply classification
# --------------------------------------------------------------------------- #
def test_classify_yes_no_unclear():
    assert classify_reply("yes", CFG) == "yes"
    assert classify_reply("Sure, go ahead!", CFG) == "yes"
    assert classify_reply("no", CFG) == "no"
    assert classify_reply("Never mind.", CFG) == "no"
    assert classify_reply("what was that?", CFG) == "unclear"
    assert classify_reply("", CFG) == "unclear"
    assert classify_reply(None, CFG) == "unclear"


def test_negatives_dominate_mixed_signals():
    # Contains "do it" (affirmative) AND "no"/"don't" — must refuse.
    assert classify_reply("no, don't do it", CFG) == "no"


def test_word_boundary_matching():
    # "notepad" must not match the negative "no"; "okay" not double-matched.
    assert classify_reply("notepad", CFG) == "unclear"
    assert classify_reply("okay", CFG) == "yes"


# --------------------------------------------------------------------------- #
# Gate decisions
# --------------------------------------------------------------------------- #
def test_non_destructive_allowed_without_asking():
    gate, ask = _gate("yes")
    result = gate.authorize(OPEN)
    assert result.allowed
    assert ask.prompts == []  # never spoke a confirmation


def test_destructive_confirmed_by_yes():
    gate, ask = _gate("yes")
    result = gate.authorize(DELETE)
    assert result.allowed
    assert result.reason == "confirmed by user"
    assert "report.txt" in ask.prompts[0]  # prompt named the target


def test_destructive_denied_by_no():
    gate, _ = _gate("no")
    result = gate.authorize(DELETE)
    assert not result.allowed
    assert result.reason == "denied by user"


def test_unclear_then_yes_within_retries():
    gate, ask = _gate("ummm", "yes")
    result = gate.authorize(DELETE)
    assert result.allowed
    assert len(ask.prompts) == 2  # initial prompt + one reprompt
    assert ask.prompts[1] == CFG.reprompt


def test_all_unclear_denies_after_retries():
    gate, ask = _gate("what", "huh", "eh")
    result = gate.authorize(DELETE)
    assert not result.allowed
    # initial prompt + confirmation_retries reprompts
    assert len(ask.prompts) == 1 + CFG.confirmation_retries


def test_silence_denies():
    gate, _ = _gate(None, None, None)
    assert not gate.authorize(DELETE).allowed


def test_extra_destructive_actions_are_gated():
    cfg = SafetyConfig(extra_destructive_actions=("close_app",))
    gate, ask = _gate("yes", cfg=cfg)
    result = gate.authorize(Intent("close_app", {"name": "word"}, "close word"))
    assert result.allowed
    assert len(ask.prompts) == 1  # confirmation was demanded


def test_uncataloged_action_confirmed_when_configured():
    gate, ask = _gate("yes")
    result = gate.authorize(Intent("mystery_op", {}, "do the mystery"))
    assert result.allowed
    assert len(ask.prompts) == 1


def test_uncataloged_action_refused_when_confirm_unknown_disabled():
    cfg = SafetyConfig(confirm_unknown_actions=False)
    gate, ask = _gate("yes", cfg=cfg)
    result = gate.authorize(Intent("mystery_op", {}, "do the mystery"))
    assert not result.allowed
    assert ask.prompts == []  # refused outright, no confirmation offered


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def test_describe_uses_template_for_destructive_actions():
    assert describe("delete_file", {"path": "a.txt"}) == "delete the file a.txt"
    assert describe("move_file", {"source": "a", "destination": "b"}) == "move a to b"


def test_describe_falls_back_generically():
    assert describe("open_app", {"name": "notepad"}) == "open app: name notepad"
    assert describe("delete_file", {}) == "delete file"  # template param missing
