"""Tests for live web answers (all network I/O mocked — no real HTTP here)."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

import jarvis.execution.web_answers as wa
from jarvis.config import ConfigError, WebAnswersConfig, load_config

CFG = WebAnswersConfig(max_headlines=3, max_snippets=2)

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><title>Rain lashes Bengaluru - The Hindu</title>
        <source url="https://thehindu.com">The Hindu</source></item>
  <item><title>Markets rally on earnings</title></item>
  <item><title>Third story - NDTV</title><source>NDTV</source></item>
  <item><title>Fourth story beyond the limit</title></item>
</channel></rss>"""

HTML = b"""
<div class="result">
  <h2 class="result__title"><a class="result__a" href="#">F1 2026 calendar</a></h2>
  <a class="result__snippet" href="#">The next race is the <b>British Grand Prix</b> on July 5.</a>
</div>
<div class="result">
  <a class="result__snippet" href="#">Second snippet<br>with a void tag.</a>
</div>
<div class="result">
  <a class="result__snippet" href="#">Third snippet beyond the limit.</a>
</div>
"""


# --------------------------------------------------------------------------- #
# News feed
# --------------------------------------------------------------------------- #
def test_fetch_news_parses_strips_sources_and_limits(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, str] = {}

    def fake_get(url: str, timeout_sec: float) -> bytes:
        seen["url"] = url
        return RSS

    monkeypatch.setattr(wa, "_http_get", fake_get)
    headlines = wa.fetch_news(CFG)
    assert seen["url"] == CFG.news_url
    assert len(headlines) == 3  # max_headlines respected
    assert headlines[0] == wa.Headline("Rain lashes Bengaluru", "The Hindu")
    assert headlines[1] == wa.Headline("Markets rally on earnings", "")
    assert headlines[2] == wa.Headline("Third story", "NDTV")


def test_fetch_news_topic_uses_search_feed(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, str] = {}

    def fake_get(url: str, timeout_sec: float) -> bytes:
        seen["url"] = url
        return RSS

    monkeypatch.setattr(wa, "_http_get", fake_get)
    wa.fetch_news(CFG, "formula 1")
    assert "formula+1" in seen["url"]


def test_fetch_news_bad_xml_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wa, "_http_get", lambda url, t: b"<html>not a feed")
    with pytest.raises(wa.WebAnswerError, match="XML"):
        wa.fetch_news(CFG)


# --------------------------------------------------------------------------- #
# Search snippets
# --------------------------------------------------------------------------- #
def test_fetch_snippets_pairs_titles_and_limits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wa, "_http_get", lambda url, t, **kw: HTML)
    snippets = wa.fetch_snippets(CFG, "next f1 race")
    assert snippets == [
        "F1 2026 calendar: The next race is the British Grand Prix on July 5.",
        "Second snippet with a void tag.",  # <br> must not break capture
    ]


BING = b"""
<ol id="b_results">
<li class="b_algo"><h2><a href="#">F1 <strong>2026</strong> calendar</a></h2>
<div class="b_caption"><p>The next race is the British Grand Prix on July 5.</p></div></li>
<li class="b_algo"><h2><a href="#">Second title</a></h2>
<p class="b_lineclamp4">Second snippet &amp; more.</p></li>
<li class="b_algo"><h2><a href="#">Third</a></h2><p>Third snippet beyond the limit.</p></li>
</ol>
"""


def test_fetch_snippets_parses_bing_markup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wa, "_http_get", lambda url, t, **kw: BING)
    snippets = wa.fetch_snippets(CFG, "next f1 race")
    assert snippets == [
        "F1 2026 calendar: The next race is the British Grand Prix on July 5.",
        "Second title: Second snippet & more.",
    ]


def test_fetch_snippets_empty_page(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wa, "_http_get", lambda url, t, **kw: b"<html><body>nothing</body></html>")
    assert wa.fetch_snippets(CFG, "anything") == []


# --------------------------------------------------------------------------- #
# QA context: decoy guard + two-source merge
# --------------------------------------------------------------------------- #
def test_relevance_filter_drops_decoy_snippets():
    snippets = [
        "F1 2026 calendar: The next race is at Silverstone.",
        "Download RStudio Desktop: Simple data science tools.",  # decoy page
        "Formula 1 schedule: full season dates.",
    ]
    kept = wa._relevant(snippets, "when is the next formula 1 race")
    assert kept == [snippets[0], snippets[2]]


def test_qa_context_merges_news_and_snippets(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        wa, "fetch_news",
        lambda cfg, q: [wa.Headline("British GP this Sunday", "BBC",
                                    "Sat, 04 Jul 2026 10:00:00 GMT")],
    )
    monkeypatch.setattr(
        wa, "fetch_snippets",
        lambda cfg, q: ["F1 calendar: race on July 5.", "Unrelated decoy page."],
    )
    lines = wa.fetch_qa_context(CFG, "next formula 1 race")
    assert lines == [
        "News headline (BBC, Sat, 04 Jul 2026 10:00:00 GMT): British GP this Sunday",
        "F1 calendar: race on July 5.",
    ]


def test_qa_context_survives_one_source_failing(monkeypatch: pytest.MonkeyPatch):
    def boom(cfg, q):
        raise wa.WebAnswerError("blocked")

    monkeypatch.setattr(wa, "fetch_news", boom)
    monkeypatch.setattr(wa, "fetch_snippets", lambda cfg, q: ["Formula 1: July 5."])
    assert wa.fetch_qa_context(CFG, "formula 1 race") == ["Formula 1: July 5."]


def test_qa_context_raises_when_both_sources_fail(monkeypatch: pytest.MonkeyPatch):
    def boom(cfg, q):
        raise wa.WebAnswerError("offline")

    monkeypatch.setattr(wa, "fetch_news", boom)
    monkeypatch.setattr(wa, "fetch_snippets", boom)
    with pytest.raises(wa.WebAnswerError, match="all live sources"):
        wa.fetch_qa_context(CFG, "anything")


# --------------------------------------------------------------------------- #
# HTTP failures
# --------------------------------------------------------------------------- #
def test_http_get_wraps_network_errors(monkeypatch: pytest.MonkeyPatch):
    def boom(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(wa.urllib.request, "urlopen", boom)
    with pytest.raises(wa.WebAnswerError, match="failed"):
        wa._http_get("https://example.com", 1.0)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def _load(tmp_path: Path, body: str):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"web_answers:\n{body}", encoding="utf-8")
    return load_config(cfg)


def test_config_defaults_load(tmp_path: Path):
    cfg = _load(tmp_path, "  enabled: false\n")
    assert cfg.web_answers.enabled is False
    assert cfg.web_answers.max_headlines == 3


def test_config_rejects_bad_timeout(tmp_path: Path):
    with pytest.raises(ConfigError, match="timeout_sec"):
        _load(tmp_path, "  timeout_sec: 0\n")


def test_config_rejects_missing_query_placeholder(tmp_path: Path):
    with pytest.raises(ConfigError, match="snippets_url"):
        _load(tmp_path, '  snippets_url: "https://example.com/"\n')
