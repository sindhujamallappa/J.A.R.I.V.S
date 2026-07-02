"""Spoken-confirmation gate for J.A.R.I.V.S — the destructive-action guardrail.

Every intent must pass through :meth:`ConfirmationGate.authorize` before the
executor may run it. Non-destructive actions are allowed straight through;
destructive ones (per the catalog in :mod:`jarvis.actions`, plus any names in
``safety.extra_destructive_actions``) require an explicit spoken "yes". The
default on anything unclear — mixed signals, silence, STT failure, exhausted
retries — is DENY.

The gate does not talk or listen itself: the orchestrator injects an ``ask``
callable that speaks a prompt (TTS) and returns the user's transcribed reply
(STT). That keeps this module pure decision logic, fully unit-testable.

Classification: decision logic only — performs no filesystem/process
operations itself; it is the in-code enforcement point that gates the ones
the executor performs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from ..actions import ACTIONS, describe, is_destructive
from ..config import SafetyConfig
from ..intent.parser import Intent

log = logging.getLogger(__name__)

# Speaks `prompt`, returns the user's transcribed reply (None = heard nothing).
AskFn = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class GateResult:
    """Outcome of authorizing one intent."""

    allowed: bool
    reason: str
    # What the gate said/asked, for logs and for the orchestrator's reply.
    spoken_summary: str = ""


def _matches_any(reply: str, phrases: Iterable[str]) -> bool:
    """True if any phrase occurs in ``reply`` as whole words."""
    for phrase in phrases:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", reply):
            return True
    return False


def classify_reply(reply: Optional[str], cfg: SafetyConfig) -> str:
    """Classify a confirmation reply as 'yes', 'no', or 'unclear'.

    Negatives dominate: a reply containing both signals ("no, don't do it")
    is a refusal — fail-safe over eager execution.
    """
    if reply is None or not reply.strip():
        return "unclear"
    text = reply.strip().lower()
    if _matches_any(text, cfg.negatives):
        return "no"
    if _matches_any(text, cfg.affirmatives):
        return "yes"
    return "unclear"


class ConfirmationGate:
    """Authorizes intents, demanding spoken confirmation for destructive ones."""

    def __init__(self, cfg: SafetyConfig, ask: AskFn) -> None:
        self._cfg = cfg
        self._ask = ask

    def authorize(self, intent: Intent) -> GateResult:
        """Decide whether ``intent`` may execute.

        Args:
            intent: A validated intent from the parser.

        Returns:
            A :class:`GateResult`; ``allowed`` is True only for
            non-destructive actions or explicitly confirmed destructive ones.
        """
        cfg = self._cfg
        action = intent.action

        if action not in ACTIONS:
            # Parser normally collapses these to 'unknown'; if one slips
            # through, confirm it (fail-safe) or refuse outright.
            if not cfg.confirm_unknown_actions:
                log.warning("Refusing uncataloged action %r outright", action)
                return GateResult(False, "uncataloged action refused")
            return self._confirm(intent)

        if not is_destructive(action, cfg.extra_destructive_actions):
            log.debug("Action %r is non-destructive — allowed", action)
            return GateResult(True, "non-destructive")

        return self._confirm(intent)

    def _confirm(self, intent: Intent) -> GateResult:
        """Run the spoken yes/no loop. DENY is the default outcome."""
        cfg = self._cfg
        summary = describe(intent.action, intent.params)
        prompt = cfg.confirmation_prompt.format(action=summary)

        log.info("Destructive action %r — requesting confirmation: %s",
                 intent.action, summary)
        reply = self._ask(prompt)
        for attempt in range(cfg.confirmation_retries + 1):
            verdict = classify_reply(reply, cfg)
            if verdict == "yes":
                log.info("User CONFIRMED destructive action %r", intent.action)
                return GateResult(True, "confirmed by user", summary)
            if verdict == "no":
                log.info("User DENIED destructive action %r", intent.action)
                return GateResult(False, "denied by user", summary)
            if attempt < cfg.confirmation_retries:
                log.info("Unclear confirmation reply %r — reprompting", reply)
                reply = self._ask(cfg.reprompt)

        log.warning("No clear yes/no after %d attempt(s) — denying %r (fail-safe)",
                    cfg.confirmation_retries + 1, intent.action)
        return GateResult(False, "no clear confirmation — denied", summary)
