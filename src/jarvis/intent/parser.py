"""Intent parsing for J.A.R.I.V.S via a local LLM served by Ollama.

Maps a transcribed command to one action from the catalog in
:mod:`jarvis.actions`, with validated parameters. The LLM is constrained to
JSON output and its answer is *never trusted blindly*: unknown action names,
malformed JSON, or missing required parameters all collapse to the safe
``unknown`` intent instead of propagating garbage to the executor.

Classification: pure computation plus an HTTP call to localhost (the Ollama
server) — non-destructive. Nothing here executes actions; the destructive
gate is applied downstream using the catalog's flags.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import ollama

from ..actions import ACTIONS, catalog_lines
from ..config import IntentConfig

log = logging.getLogger(__name__)


class IntentError(Exception):
    """Raised when the LLM backend cannot be reached or fails outright."""


@dataclass(frozen=True)
class Intent:
    """A validated, executable interpretation of one spoken command."""

    action: str
    params: dict[str, str]
    raw_text: str


class ChatClient(Protocol):
    """The slice of ``ollama.Client`` we use (injectable for tests)."""

    def chat(self, **kwargs: Any) -> Any: ...


def _build_system_prompt() -> str:
    actions = "\n".join(catalog_lines())
    return f"""You translate one spoken command for a Windows voice assistant into exactly one action.

Available actions (parameters ending in ? are optional):
{actions}

Rules:
- Reply with ONLY a JSON object: {{"action": "<name>", "params": {{...}}}}
- Pick exactly one action from the list. Never invent new actions or parameters.
- For greetings, questions, or chit-chat use "respond"; params.text must be a complete spoken sentence.
- If the command cannot be mapped to any action, use "unknown" with a brief params.reason.
- open_app / close_app are ONLY for software applications on this computer — never for
  physical objects, appliances, food, or anything not installable on a PC.
- Keep file paths, app names, and URLs verbatim from the command; do not embellish them.
- The user speaks; transcripts may have odd casing or punctuation. Interpret them charitably.

Examples:
"Open Notepad, please." -> {{"action": "open_app", "params": {{"name": "notepad"}}}}
"Delete the file report.txt from my desktop" -> {{"action": "delete_file", "params": {{"path": "Desktop/report.txt"}}}}
"Search the web for tomorrow's weather" -> {{"action": "web_search", "params": {{"query": "tomorrow's weather"}}}}
"Hey, how are you?" -> {{"action": "respond", "params": {{"text": "I am doing well and ready to help."}}}}
"Make me a sandwich" -> {{"action": "unknown", "params": {{"reason": "no physical-world actions"}}}}"""


class IntentParser:
    """Parses transcribed commands into validated :class:`Intent` objects."""

    def __init__(self, cfg: IntentConfig, client: Optional[ChatClient] = None) -> None:
        self._cfg = cfg
        self._client: ChatClient = client if client is not None else ollama.Client(
            host=cfg.host, timeout=cfg.timeout_sec
        )
        self._system_prompt = _build_system_prompt()

    def warm_up(self) -> bool:
        """Preload the model into server RAM (first cold load can take 30s+).

        Call once at startup so the first spoken command doesn't eat the
        load latency — or time out entirely.

        Returns:
            True if the backend responded; False (logged) otherwise.
        """
        try:
            self._client.chat(
                model=self._cfg.model,
                messages=[{"role": "user", "content": "ping"}],
                options={"num_predict": 1},
                keep_alive=self._cfg.keep_alive,
            )
            log.info("Intent model %r warmed up", self._cfg.model)
            return True
        except Exception as exc:  # noqa: BLE001 — startup should not crash
            log.warning("Intent model warm-up failed: %s", exc)
            return False

    def parse(self, text: str) -> Intent:
        """Map one transcript to a validated intent.

        Args:
            text: The transcribed spoken command.

        Returns:
            A validated :class:`Intent`; falls back to action ``unknown``
            when the model's answer can't be trusted.

        Raises:
            IntentError: If the Ollama server can't be reached or errors.
        """
        if not text.strip():
            return Intent("unknown", {"reason": "empty transcript"}, text)

        try:
            response = self._client.chat(
                model=self._cfg.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": text},
                ],
                format="json",
                options={
                    "temperature": self._cfg.temperature,
                    "num_predict": self._cfg.max_tokens,
                },
                keep_alive=self._cfg.keep_alive,
            )
        except Exception as exc:
            raise IntentError(f"Ollama request failed: {exc}") from exc

        content = response["message"]["content"]
        intent = self._validate(content, text)
        log.info("Intent: %r -> %s %s", text, intent.action, intent.params)
        return intent

    def _validate(self, content: str, raw_text: str) -> Intent:
        """Validate LLM output against the action catalog (fail-safe)."""
        data = _parse_json_object(content)
        if data is None:
            log.warning("Unparseable intent JSON: %r", content[:200])
            return Intent("unknown", {"reason": "unparseable model output"}, raw_text)

        action = str(data.get("action", "")).strip().lower()
        spec = ACTIONS.get(action)
        if spec is None:
            log.warning("Model emitted unsupported action %r", action)
            return Intent("unknown", {"reason": f"unsupported action '{action}'"}, raw_text)

        raw_params = data.get("params") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}
        allowed = set(spec.required) | set(spec.optional)
        params = {
            k: str(v) for k, v in raw_params.items()
            if k in allowed and v is not None
        }

        missing = [p for p in spec.required if not params.get(p, "").strip()]
        if missing:
            log.warning("Action %r missing required param(s) %s", action, missing)
            return Intent(
                "unknown",
                {"reason": f"missing parameter '{missing[0]}' for action '{action}'"},
                raw_text,
            )
        return Intent(action, params, raw_text)


def _parse_json_object(content: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object, tolerating stray text around the braces."""
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(content[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _demo() -> None:
    """Standalone smoke test: python -m src.jarvis.intent.parser "open notepad" """
    import sys

    from ..config import load_config
    from ..utils.logging_config import configure_logging

    cfg = load_config()
    configure_logging(cfg.logging)
    text = " ".join(sys.argv[1:]) or "open notepad"
    intent = IntentParser(cfg.intent).parse(text)
    log.info("→ action=%s params=%s", intent.action, intent.params)


if __name__ == "__main__":
    _demo()
