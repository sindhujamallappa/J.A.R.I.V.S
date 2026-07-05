"""Live web answers for J.A.R.I.V.S — spoken news and quick factual lookups.

Instead of only opening a browser tab, the assistant can FETCH public data and
speak the result: top headlines from a Google News RSS feed (``get_news``) and
search-result snippets from DuckDuckGo's HTML endpoint that the local LLM
condenses into a one-or-two-sentence spoken answer (``answer_question``).
Standard library only (urllib + xml/html parsers) — no new dependencies.

PRIVACY: this module is the one deliberate exception to fully-offline
operation. When — and only when — the user asks a live question, that query is
sent over HTTPS to the configured endpoints. Nothing else ever leaves the
device, and ``web_answers.enabled: false`` turns the whole module off.

Classification: network read-only (HTTP GET) — non-destructive. No filesystem
or process side effects.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import quote_plus

from ..config import WebAnswersConfig

log = logging.getLogger(__name__)

# Some endpoints reject clients with a missing or unusual User-Agent.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# Bing serves its classic static-HTML results (li.b_algo) to text-mode
# browsers; graphical UAs get a JavaScript shell with no parseable results.
# DuckDuckGo's html endpoint doesn't care either way, so snippet fetches
# always use this one.
_SNIPPET_UA = "Lynx/2.8.9rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/3.6.13"

# HTML void elements never get a closing tag — must not affect capture depth.
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input",
     "link", "meta", "source", "track", "wbr"}
)


class WebAnswerError(Exception):
    """Raised when a live lookup fails (network, HTTP, or parse trouble)."""


@dataclass(frozen=True)
class Headline:
    """One news headline; ``source`` and ``published`` may be empty."""

    title: str
    source: str
    published: str = ""  # feed's pubDate, verbatim (useful LLM context)


def _http_get(url: str, timeout_sec: float, user_agent: str = _USER_AGENT) -> bytes:
    """GET ``url`` and return the raw body.

    Raises:
        WebAnswerError: On any network, DNS, TLS, or HTTP failure.
    """
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WebAnswerError(f"request to {url} failed: {exc}") from exc


def fetch_news(cfg: WebAnswersConfig, topic: str = "") -> list[Headline]:
    """Fetch up to ``cfg.max_headlines`` headlines, optionally about ``topic``.

    Args:
        cfg: Live-answer settings (URLs, limits, timeout).
        topic: Optional subject; empty means the general top-headlines feed.

    Returns:
        Headlines in feed order, with the feed's " - Source" title suffix
        stripped when the source is also given separately.

    Raises:
        WebAnswerError: If the feed can't be fetched or isn't valid XML.
    """
    topic = topic.strip()
    url = cfg.news_search_url.format(query=quote_plus(topic)) if topic else cfg.news_url
    raw = _http_get(url, cfg.timeout_sec)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise WebAnswerError(f"news feed is not valid XML: {exc}") from exc

    headlines: list[Headline] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        suffix = f" - {source}"
        if source and title.endswith(suffix):
            title = title[: -len(suffix)].rstrip()
        headlines.append(Headline(title, source, published))
        if len(headlines) >= cfg.max_headlines:
            break
    log.debug("Fetched %d headline(s) for topic %r", len(headlines), topic)
    return headlines


class _ResultParser(HTMLParser):
    """Pulls (title, snippet) pairs out of DuckDuckGo's HTML results page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str]] = []
        self._capturing: Optional[str] = None  # "title" | "snippet"
        self._depth = 0
        self._buffer: list[str] = []
        self._pending_title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if self._capturing is not None:
            if tag in _VOID_TAGS:
                self._buffer.append(" ")  # <br> renders as whitespace, not glue
            else:
                self._depth += 1
            return
        classes = dict(attrs).get("class") or ""
        if "result__snippet" in classes:
            self._capturing, self._depth, self._buffer = "snippet", 0, []
        elif "result__a" in classes:
            self._capturing, self._depth, self._buffer = "title", 0, []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if self._capturing is not None:
            self._buffer.append(" ")  # structurally neutral, visually a gap

    def handle_endtag(self, tag: str) -> None:
        if self._capturing is None:
            return
        if self._depth > 0:
            self._depth -= 1
            return
        text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
        if self._capturing == "title":
            self._pending_title = text
        elif text:
            self.results.append((self._pending_title, text))
            self._pending_title = ""
        self._capturing = None

    def handle_data(self, data: str) -> None:
        if self._capturing is not None:
            self._buffer.append(data)


# --- Bing's static results page (regex-friendly, decade-stable markup) ----- #
_BING_RESULT_START = re.compile(r'<li class="b_algo')
_BING_TITLE = re.compile(r"<h2[^>]*>(.*?)</h2>", re.S)
_BING_SNIPPET = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _strip_tags(fragment: str) -> str:
    """Markup fragment -> readable one-line text (tags out, entities decoded)."""
    return re.sub(r"\s+", " ", unescape(_TAG.sub(" ", fragment))).strip()


def _parse_bing(html_text: str, limit: int) -> list[str]:
    """Extract "Title: snippet" strings from Bing's static results page."""
    starts = [m.start() for m in _BING_RESULT_START.finditer(html_text)]
    snippets: list[str] = []
    for start, end in zip(starts, starts[1:] + [len(html_text)]):
        block = html_text[start:end]
        title_match = _BING_TITLE.search(block)
        snippet_match = _BING_SNIPPET.search(block)
        title = _strip_tags(title_match.group(1)) if title_match else ""
        snippet = _strip_tags(snippet_match.group(1)) if snippet_match else ""
        if not snippet:
            continue
        snippets.append(f"{title}: {snippet}" if title else snippet)
        if len(snippets) >= limit:
            break
    return snippets


def fetch_snippets(cfg: WebAnswersConfig, query: str) -> list[str]:
    """Fetch up to ``cfg.max_snippets`` search-result snippets for ``query``.

    The markup dialect is auto-detected: DuckDuckGo's html endpoint
    (``result__`` classes) and Bing's static page (``b_algo`` blocks) are
    both supported, so ``snippets_url`` can point at either.

    Args:
        cfg: Live-answer settings (URLs, limits, timeout).
        query: The user's question, sent verbatim to the search endpoint.

    Returns:
        "Title: snippet" strings (title omitted when absent), best-ranked
        first — raw material for the local LLM to compose a spoken answer.
        Empty when the page had no recognizable results.

    Raises:
        WebAnswerError: If the results page can't be fetched.
    """
    raw = _http_get(
        cfg.snippets_url.format(query=quote_plus(query)),
        cfg.timeout_sec,
        user_agent=_SNIPPET_UA,
    )
    text = raw.decode("utf-8", errors="replace")
    if "result__" in text:  # DuckDuckGo html/lite markup
        parser = _ResultParser()
        parser.feed(text)
        snippets = [
            f"{title}: {snippet}" if title else snippet
            for title, snippet in parser.results[: cfg.max_snippets]
        ]
    else:
        snippets = _parse_bing(text, cfg.max_snippets)
    log.debug("Fetched %d snippet(s) for %r", len(snippets), query)
    return snippets


# --------------------------------------------------------------------------- #
# Combined QA context: news backbone + relevance-filtered engine snippets
# --------------------------------------------------------------------------- #
# Question words carry no topical signal; only content words count.
_QUERY_STOPWORDS = frozenset({
    "what", "whats", "when", "whens", "where", "which", "whose", "there",
    "their", "about", "with", "from", "this", "that", "these", "those",
    "does", "will", "would", "should", "could", "next", "latest", "today",
    "tomorrow", "please", "tell",
})


def _query_terms(query: str) -> set[str]:
    """Topical words of a query (length >= 4, letters only, no stopwords)."""
    words = re.findall(r"[a-z]+", query.lower())
    return {w for w in words if len(w) >= 4 and w not in _QUERY_STOPWORDS}


def _relevant(snippets: list[str], query: str) -> list[str]:
    """Drop snippets sharing no topical word with the query.

    Search engines sometimes serve decoy/cached pages to non-browser
    clients; feeding those to the LLM would produce confidently wrong
    spoken answers, which is worse than no answer.
    """
    terms = _query_terms(query)
    if not terms:
        return snippets
    kept = [s for s in snippets if any(t in s.lower() for t in terms)]
    if len(kept) < len(snippets):
        log.info("Dropped %d off-topic snippet(s) (decoy guard)",
                 len(snippets) - len(kept))
    return kept


def fetch_qa_context(cfg: WebAnswersConfig, query: str) -> list[str]:
    """Context lines for answering one live question.

    Two independent sources, so one failing (or being blocked, as corporate
    networks often do to search engines) doesn't kill the answer:
    dated news headlines from the RSS search feed, then engine snippets
    that pass the decoy guard.

    Returns:
        Possibly empty list of context lines for the local LLM.

    Raises:
        WebAnswerError: Only when BOTH sources are unreachable — the caller
            can then say the web itself is down rather than "no results".
    """
    lines: list[str] = []
    failures = 0

    try:
        for h in fetch_news(cfg, query):
            meta = ", ".join(x for x in (h.source, h.published) if x)
            lines.append(f"News headline ({meta}): {h.title}" if meta
                         else f"News headline: {h.title}")
    except WebAnswerError:
        failures += 1
        log.warning("News context fetch failed", exc_info=True)

    try:
        lines.extend(_relevant(fetch_snippets(cfg, query), query))
    except WebAnswerError:
        failures += 1
        log.warning("Snippet context fetch failed", exc_info=True)

    if failures == 2:
        raise WebAnswerError("all live sources unreachable")
    return lines
