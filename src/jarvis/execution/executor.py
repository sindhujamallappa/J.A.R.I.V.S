"""Action execution for J.A.R.I.V.S — the hands of the assistant.

Implements every action declared in :mod:`jarvis.actions` and nothing else.
Each handler returns an :class:`ExecutionResult` whose ``message`` is written
for the ear (it gets spoken by TTS).

SAFETY CONTRACT: the orchestrator must only call :meth:`Executor.execute`
with intents that passed :class:`jarvis.safety.gate.ConfirmationGate`. The
destructive handlers below (write_file, move_file, delete_file, kill_process,
run_command, shutdown, restart) therefore run only after an explicit spoken
"yes". ``run_command`` is additionally refused unless
``execution.allow_run_command`` is enabled in config, regardless of
confirmation.

Classification per handler is stated in its docstring. Non-destructive
handlers are side-effect-bounded: create-only, no-clobber, read-only, or
trivially reversible (volume, lock, sleep).
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import subprocess
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from ..config import ExecutionConfig, WebAnswersConfig
from ..intent.parser import Intent
from . import web_answers

log = logging.getLogger(__name__)

_RUN_COMMAND_TIMEOUT_SEC = 60

# Spoken list prefixes for get_news ("1." reads badly aloud).
_ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth",
             "Sixth", "Seventh", "Eighth", "Ninth", "Tenth")


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one action; ``message`` is spoken to the user."""

    ok: bool
    message: str


def _resolve_path(raw: str) -> Path:
    """Resolve a spoken path: ~, env vars, and home-relative shorthand.

    "Desktop/report.txt" resolves under the user's home directory; absolute
    paths pass through unchanged.
    """
    expanded = os.path.expandvars(os.path.expanduser(raw.strip()))
    p = Path(expanded)
    return p if p.is_absolute() else Path.home() / p


class Executor:
    """Dispatches validated, authorized intents to action handlers."""

    def __init__(
        self,
        cfg: ExecutionConfig,
        web: Optional[WebAnswersConfig] = None,
        answerer: Optional[Callable[[str, Sequence[str]], Optional[str]]] = None,
    ) -> None:
        """Args:
            cfg: Execution settings.
            web: Live web-answer settings; ``None`` (or ``enabled: false``)
                makes get_news / answer_question fall back to a browser search.
            answerer: Composes a spoken answer from (question, snippets) —
                the orchestrator wires this to ``IntentParser.answer`` so the
                same local LLM is reused. ``None`` disables answer_question
                the same way.
        """
        self._cfg = cfg
        self._web = web
        self._answerer = answerer
        self._handlers: dict[str, Callable[[Mapping[str, str]], ExecutionResult]] = {
            "open_app": self._open_app,
            "close_app": self._close_app,
            "open_url": self._open_url,
            "web_search": self._web_search,
            "get_news": self._get_news,
            "answer_question": self._answer_question,
            "read_file": self._read_file,
            "list_folder": self._list_folder,
            "create_folder": self._create_folder,
            "copy_file": self._copy_file,
            "write_file": self._write_file,
            "move_file": self._move_file,
            "delete_file": self._delete_file,
            "kill_process": self._kill_process,
            "run_command": self._run_command,
            "set_volume": self._set_volume,
            "mute": self._mute,
            "unmute": self._unmute,
            "lock_screen": self._lock_screen,
            "sleep_pc": self._sleep_pc,
            "shutdown": self._shutdown,
            "restart": self._restart,
            "get_time": self._get_time,
            "respond": self._respond,
            "unknown": self._unknown,
        }

    def execute(self, intent: Intent) -> ExecutionResult:
        """Run one gate-authorized intent.

        Args:
            intent: Validated intent that already passed the confirmation gate.

        Returns:
            An :class:`ExecutionResult` with a speakable outcome message.
        """
        handler = self._handlers.get(intent.action)
        if handler is None:
            log.warning("No handler for action %r", intent.action)
            return ExecutionResult(False, "I don't know how to do that yet.")
        try:
            result = handler(intent.params)
        except Exception:  # noqa: BLE001 — one bad action must not kill the loop
            log.exception("Action %r failed", intent.action)
            return ExecutionResult(False, f"Sorry, {intent.action.replace('_', ' ')} failed.")
        log.info("Executed %r ok=%s: %s", intent.action, result.ok, result.message)
        return result

    # ------------------------------------------------------------------ #
    # Apps  (non-destructive: launch / graceful close)
    # ------------------------------------------------------------------ #
    def _open_app(self, p: Mapping[str, str]) -> ExecutionResult:
        """Launch an app. Non-destructive."""
        name = p["name"].strip()
        target = self._cfg.apps.get(name.lower(), name)
        try:
            subprocess.Popen([target])
        except FileNotFoundError:
            try:
                subprocess.Popen([f"{target}.exe"])
            except FileNotFoundError:
                return ExecutionResult(False, f"I couldn't find an app called {name}.")
        return ExecutionResult(True, f"Opening {name}.")

    def _close_app(self, p: Mapping[str, str]) -> ExecutionResult:
        """Gracefully close an app (WM_CLOSE — it may prompt to save).

        Non-destructive: no force flag; unsaved-work protection stays with
        the app. Force-termination is the separate, gated kill_process.
        """
        name = p["name"].strip()
        image = name if name.lower().endswith(".exe") else f"{name}.exe"
        proc = subprocess.run(
            ["taskkill", "/IM", image], capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return ExecutionResult(False, f"I couldn't close {name} — it may not be running.")
        return ExecutionResult(True, f"Closed {name}.")

    # ------------------------------------------------------------------ #
    # Web  (non-destructive)
    # ------------------------------------------------------------------ #
    def _open_url(self, p: Mapping[str, str]) -> ExecutionResult:
        """Open a URL in the default browser. Non-destructive."""
        url = p["url"].strip()
        if "://" not in url:
            url = f"https://{url}"
        webbrowser.open(url)
        return ExecutionResult(True, f"Opening {urllib.parse.urlparse(url).netloc or url}.")

    def _web_search(self, p: Mapping[str, str]) -> ExecutionResult:
        """Web search in the default browser. Non-destructive."""
        query = p["query"].strip()
        webbrowser.open(self._cfg.search_url.format(query=urllib.parse.quote_plus(query)))
        return ExecutionResult(True, f"Searching for {query}.")

    # ------------------------------------------------------------------ #
    # Live web answers  (non-destructive: network read-only)
    # ------------------------------------------------------------------ #
    def _browser_fallback(self, query: str, lead: str) -> ExecutionResult:
        """Plan B for live answers: open a browser search, explain why aloud."""
        webbrowser.open(self._cfg.search_url.format(query=urllib.parse.quote_plus(query)))
        return ExecutionResult(True, f"{lead} I opened a web search for {query} instead.")

    def _get_news(self, p: Mapping[str, str]) -> ExecutionResult:
        """Speak the top news headlines. Non-destructive (network read-only)."""
        topic = p.get("topic", "").strip()
        search_query = f"{topic} news" if topic else "latest news"
        if self._web is None or not self._web.enabled:
            return self._browser_fallback(
                search_query, "Live news is disabled in my configuration."
            )
        try:
            headlines = web_answers.fetch_news(self._web, topic)
        except web_answers.WebAnswerError:
            log.exception("News fetch failed")
            return self._browser_fallback(search_query, "I couldn't reach the news feed.")
        if not headlines:
            label = f"news about {topic}" if topic else "news"
            return ExecutionResult(False, f"I couldn't find any {label} right now.")
        parts = (
            [f"Here are the top {len(headlines)} headlines."]
            if len(headlines) > 1 else ["Here is the top headline."]
        )
        for i, headline in enumerate(headlines):
            prefix = _ORDINALS[i] if i < len(_ORDINALS) else "Next"
            source = f", from {headline.source}" if headline.source else ""
            parts.append(f"{prefix}: {headline.title}{source}.")
        return ExecutionResult(True, " ".join(parts))

    def _answer_question(self, p: Mapping[str, str]) -> ExecutionResult:
        """Answer a live question aloud: web snippets condensed by the LOCAL
        LLM. Non-destructive (network read-only)."""
        query = p["query"].strip()
        if self._web is None or not self._web.enabled or self._answerer is None:
            return self._browser_fallback(
                query, "Live answers are disabled in my configuration."
            )
        try:
            context = web_answers.fetch_qa_context(self._web, query)
        except web_answers.WebAnswerError:
            log.exception("Live lookup failed")
            return self._browser_fallback(query, "I couldn't reach the web just now.")
        if not context:
            return self._browser_fallback(query, "I couldn't find anything useful.")
        answer = self._answerer(query, context)
        if not answer:
            return self._browser_fallback(query, "I couldn't put together an answer.")
        return ExecutionResult(True, answer)

    # ------------------------------------------------------------------ #
    # Files — non-destructive group (read-only / create-only / no-clobber)
    # ------------------------------------------------------------------ #
    def _read_file(self, p: Mapping[str, str]) -> ExecutionResult:
        """Read a text file aloud (truncated). Non-destructive (read-only)."""
        path = _resolve_path(p["path"])
        if not path.is_file():
            return ExecutionResult(False, f"I couldn't find a file at {path}.")
        content = path.read_text(encoding="utf-8", errors="replace")
        limit = self._cfg.read_file_max_chars
        clipped = content[:limit]
        suffix = " …and it continues beyond what I'll read aloud." if len(content) > limit else ""
        return ExecutionResult(True, clipped + suffix)

    def _list_folder(self, p: Mapping[str, str]) -> ExecutionResult:
        """List folder entries. Non-destructive (read-only)."""
        path = _resolve_path(p["path"])
        if not path.is_dir():
            return ExecutionResult(False, f"I couldn't find a folder at {path}.")
        entries = sorted(e.name for e in path.iterdir())
        if not entries:
            return ExecutionResult(True, f"The folder {path.name} is empty.")
        shown = entries[:15]
        more = f", and {len(entries) - len(shown)} more" if len(entries) > len(shown) else ""
        return ExecutionResult(True, f"{path.name} contains: {', '.join(shown)}{more}.")

    def _create_folder(self, p: Mapping[str, str]) -> ExecutionResult:
        """Create a folder; refuses if it exists. Non-destructive (create-only)."""
        path = _resolve_path(p["path"])
        if path.exists():
            return ExecutionResult(False, f"{path.name} already exists.")
        path.mkdir(parents=True)
        return ExecutionResult(True, f"Created the folder {path.name}.")

    def _copy_file(self, p: Mapping[str, str]) -> ExecutionResult:
        """Copy a file; never overwrites. Non-destructive (no-clobber)."""
        import shutil

        source = _resolve_path(p["source"])
        destination = _resolve_path(p["destination"])
        if not source.is_file():
            return ExecutionResult(False, f"I couldn't find a file at {source}.")
        if destination.is_dir():
            destination = destination / source.name
        if destination.exists():
            return ExecutionResult(
                False, f"{destination.name} already exists — I won't overwrite it."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return ExecutionResult(True, f"Copied {source.name} to {destination}.")

    # ------------------------------------------------------------------ #
    # Files — DESTRUCTIVE group (reached only after spoken confirmation)
    # ------------------------------------------------------------------ #
    def _write_file(self, p: Mapping[str, str]) -> ExecutionResult:
        """Write/overwrite a text file. DESTRUCTIVE (gated)."""
        path = _resolve_path(p["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(p["content"], encoding="utf-8")
        return ExecutionResult(True, f"Wrote {path.name}.")

    def _move_file(self, p: Mapping[str, str]) -> ExecutionResult:
        """Move/rename a file. DESTRUCTIVE (gated)."""
        import shutil

        source = _resolve_path(p["source"])
        destination = _resolve_path(p["destination"])
        if not source.exists():
            return ExecutionResult(False, f"I couldn't find {source}.")
        if destination.is_dir():
            destination = destination / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return ExecutionResult(True, f"Moved {source.name} to {destination}.")

    def _delete_file(self, p: Mapping[str, str]) -> ExecutionResult:
        """Delete a file/folder — Recycle Bin by default. DESTRUCTIVE (gated)."""
        path = _resolve_path(p["path"])
        if not path.exists():
            return ExecutionResult(False, f"I couldn't find {path}.")
        if self._cfg.delete_to_trash:
            try:
                from send2trash import send2trash
            except ImportError:
                return ExecutionResult(
                    False,
                    "Recycle Bin deletion needs the send2trash package, "
                    "which isn't installed — I didn't delete anything.",
                )
            send2trash(str(path))
            return ExecutionResult(True, f"Sent {path.name} to the Recycle Bin.")
        if path.is_dir():
            return ExecutionResult(
                False, "I only delete folders to the Recycle Bin, not permanently."
            )
        path.unlink()
        return ExecutionResult(True, f"Permanently deleted {path.name}.")

    # ------------------------------------------------------------------ #
    # Processes / shell — DESTRUCTIVE group
    # ------------------------------------------------------------------ #
    def _kill_process(self, p: Mapping[str, str]) -> ExecutionResult:
        """Force-terminate a process. DESTRUCTIVE (gated)."""
        name = p["name"].strip()
        image = name if name.lower().endswith(".exe") else f"{name}.exe"
        proc = subprocess.run(
            ["taskkill", "/F", "/IM", image], capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return ExecutionResult(False, f"I couldn't terminate {name} — it may not be running.")
        return ExecutionResult(True, f"Terminated {name}.")

    def _run_command(self, p: Mapping[str, str]) -> ExecutionResult:
        """Run an arbitrary shell command. DESTRUCTIVE (gated) AND
        config-gated: refused unless execution.allow_run_command is true."""
        if not self._cfg.allow_run_command:
            return ExecutionResult(
                False, "Running arbitrary commands is disabled in my configuration."
            )
        command = p["command"]
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=_RUN_COMMAND_TIMEOUT_SEC,
        )
        output = (proc.stdout or proc.stderr or "").strip()
        summary = output[:200] if output else "no output"
        if proc.returncode != 0:
            return ExecutionResult(False, f"The command failed: {summary}")
        return ExecutionResult(True, f"Done. {summary}")

    # ------------------------------------------------------------------ #
    # System controls
    # ------------------------------------------------------------------ #
    def _volume_endpoint(self):
        """COM endpoint for the default audio output (needs pycaw)."""
        import comtypes
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        try:
            # COM may be uninitialized (or initialized to a different model
            # by the audio stack) in this thread; without this, pycaw fails
            # with "Volume control isn't available".
            comtypes.CoInitialize()
        except OSError:
            pass  # already initialized in an incompatible mode — still usable

        device = AudioUtilities.GetSpeakers()
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return interface.QueryInterface(IAudioEndpointVolume)

    def _set_volume(self, p: Mapping[str, str]) -> ExecutionResult:
        """Set master volume 0–100. Non-destructive (trivially reversible)."""
        try:
            level = max(0, min(100, int(float(p["level"]))))
        except ValueError:
            return ExecutionResult(False, f"{p['level']} isn't a volume level I understand.")
        try:
            endpoint = self._volume_endpoint()
        except Exception:  # noqa: BLE001 — pycaw missing or COM failure
            log.exception("Volume endpoint unavailable")
            return ExecutionResult(False, "Volume control isn't available.")
        endpoint.SetMasterVolumeLevelScalar(level / 100.0, None)
        return ExecutionResult(True, f"Volume set to {level} percent.")

    def _set_mute(self, muted: bool) -> ExecutionResult:
        try:
            endpoint = self._volume_endpoint()
        except Exception:  # noqa: BLE001
            log.exception("Volume endpoint unavailable")
            return ExecutionResult(False, "Volume control isn't available.")
        endpoint.SetMute(1 if muted else 0, None)
        return ExecutionResult(True, "Muted." if muted else "Unmuted.")

    def _mute(self, p: Mapping[str, str]) -> ExecutionResult:
        """Mute audio. Non-destructive."""
        return self._set_mute(True)

    def _unmute(self, p: Mapping[str, str]) -> ExecutionResult:
        """Unmute audio. Non-destructive."""
        return self._set_mute(False)

    def _lock_screen(self, p: Mapping[str, str]) -> ExecutionResult:
        """Lock the workstation. Non-destructive."""
        import ctypes

        ctypes.windll.user32.LockWorkStation()  # type: ignore[attr-defined]
        return ExecutionResult(True, "Locking your screen.")

    def _sleep_pc(self, p: Mapping[str, str]) -> ExecutionResult:
        """Sleep the PC. Non-destructive (resumable; nothing is lost)."""
        subprocess.Popen(
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]
        )
        return ExecutionResult(True, "Going to sleep.")

    def _shutdown(self, p: Mapping[str, str]) -> ExecutionResult:
        """Shut down (10 s grace). DESTRUCTIVE (gated)."""
        subprocess.run(["shutdown", "/s", "/t", "10"], check=True)
        return ExecutionResult(True, "Shutting down in ten seconds. Goodbye.")

    def _restart(self, p: Mapping[str, str]) -> ExecutionResult:
        """Restart (10 s grace). DESTRUCTIVE (gated)."""
        subprocess.run(["shutdown", "/r", "/t", "10"], check=True)
        return ExecutionResult(True, "Restarting in ten seconds.")

    # ------------------------------------------------------------------ #
    # Conversation  (non-destructive: speech only)
    # ------------------------------------------------------------------ #
    def _get_time(self, p: Mapping[str, str]) -> ExecutionResult:
        """Say the date/time. Non-destructive."""
        now = _dt.datetime.now()
        return ExecutionResult(
            True, now.strftime("It's %I:%M %p on %A, %B %d.").replace(" 0", " ")
        )

    def _respond(self, p: Mapping[str, str]) -> ExecutionResult:
        """Speak the LLM's conversational reply. Non-destructive."""
        return ExecutionResult(True, p["text"])

    def _unknown(self, p: Mapping[str, str]) -> ExecutionResult:
        """Spoken fallback for unmappable requests. Non-destructive."""
        return ExecutionResult(False, "Sorry, I don't know how to do that.")
