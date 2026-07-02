"""Rule-based fast path for common spoken commands.

Recognizes unambiguous, frequent phrasings instantly with regexes and skips
the LLM round trip entirely (multi-second savings on CPU). Anything not
matched with high confidence falls through to the LLM parser — this module
must prefer returning ``None`` over guessing.

The output is an ordinary :class:`Intent` that feeds the SAME safety gate and
catalog-validated executor as LLM-parsed intents; the fast path deliberately
only matches non-destructive commands, so nothing here bypasses confirmation.

Classification: pure text parsing — non-destructive.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from .parser import Intent

log = logging.getLogger(__name__)

# Keep "." out of this table: URLs like github.com need their dots. Trailing
# sentence periods are stripped separately in _normalize.
_PUNCT = str.maketrans("", "", ",!?;:'\"")

# Words that mean the target is a file/path/URL — leave those to the LLM.
_APP_NAME_BLOCKLIST = ("file", "folder", "directory", "http", "www", "/", "\\", ".")


def _normalize(text: str) -> str:
    t = text.lower().translate(_PUNCT)
    t = re.sub(r"\.+(\s|$)", r"\1", t)  # sentence periods, not URL dots
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(?:(?:hey |ok |okay )?jarvis |please )+", "", t)
    t = re.sub(r"(?: please| now| for me)+$", "", t)
    return t.strip()


def _app_intent(action: str) -> Callable[[re.Match, str], Optional[Intent]]:
    def build(m: re.Match, raw: str) -> Optional[Intent]:
        name = m.group("name").strip()
        if any(marker in name for marker in _APP_NAME_BLOCKLIST):
            return None  # smells like a file/URL — the LLM decides
        return Intent(action, {"name": name}, raw)
    return build


def _volume(m: re.Match, raw: str) -> Optional[Intent]:
    return Intent("set_volume", {"level": m.group("level")}, raw)


def _search(m: re.Match, raw: str) -> Optional[Intent]:
    return Intent("web_search", {"query": m.group("query").strip()}, raw)


def _url(m: re.Match, raw: str) -> Optional[Intent]:
    return Intent("open_url", {"url": m.group("url").strip()}, raw)


def _fixed(action: str) -> Callable[[re.Match, str], Optional[Intent]]:
    def build(m: re.Match, raw: str) -> Optional[Intent]:
        return Intent(action, {}, raw)
    return build


_RULES: tuple[tuple[re.Pattern[str], Callable[[re.Match, str], Optional[Intent]]], ...] = (
    (re.compile(r"^(whats the time|what is the time|what time is it|tell me the time"
                r"|whats todays date|what day is it|whats the date|what is the date)$"),
     _fixed("get_time")),
    (re.compile(r"^lock(?: (?:the|my))? (?:screen|computer|pc|laptop|workstation)$"),
     _fixed("lock_screen")),
    (re.compile(r"^mute(?: (?:the|my))?(?: (?:audio|sound|volume|speakers))?$"),
     _fixed("mute")),
    (re.compile(r"^unmute(?: (?:the|my))?(?: (?:audio|sound|volume|speakers))?$"),
     _fixed("unmute")),
    (re.compile(r"^(?:set )?(?:the )?volume(?: (?:to|at))? (?P<level>\d{1,3})(?: percent)?$"),
     _volume),
    (re.compile(r"^(?:google|look up|search(?: the web)?(?: for)?) (?P<query>.+)$"),
     _search),
    (re.compile(r"^(?:open|go to|visit) (?P<url>[\w-]+(?:\.[\w-]+)+(?:/\S*)?)$"),
     _url),
    (re.compile(r"^(?:open|launch|start) (?P<name>[a-z0-9 +-]{2,40})$"),
     _app_intent("open_app")),
    (re.compile(r"^(?:close|quit|exit) (?P<name>[a-z0-9 +-]{2,40})$"),
     _app_intent("close_app")),
)


def match(text: str) -> Optional[Intent]:
    """Match one transcript against the fast-path rules.

    Args:
        text: The raw transcript.

    Returns:
        A validated :class:`Intent`, or ``None`` when the command isn't an
        unambiguous match (the LLM parser handles it instead).
    """
    normalized = _normalize(text)
    if not normalized:
        return None
    for pattern, build in _RULES:
        m = pattern.match(normalized)
        if m is not None:
            intent = build(m, text)
            if intent is not None:
                return intent
    return None
