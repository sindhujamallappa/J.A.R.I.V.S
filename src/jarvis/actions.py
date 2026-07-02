"""Action catalog for J.A.R.I.V.S — the single source of truth for what the
assistant can do.

Every voice-commanded capability is declared here with its parameter spec and
its safety classification. The intent parser validates LLM output against this
catalog, the safety gate reads ``destructive`` to decide whether a spoken
confirmation is required, and the executor implements exactly these actions.
Add a new capability by adding a spec here first.

Classification: this module is pure data — non-destructive. The flags it
declares are enforced by the safety gate in the execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ActionSpec:
    """Declares one voice-commandable action.

    Attributes:
        name: Canonical action name the LLM must emit.
        required: Parameter names that must be present.
        optional: Parameter names that may be present.
        destructive: True if the action is irreversible or can lose data —
            these always require a spoken yes/no confirmation before running.
        description: One line used to build the intent-parser prompt.
    """

    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    destructive: bool
    description: str
    # Spoken-summary template for confirmation prompts, e.g.
    # "delete the file {path}". Empty -> a generic summary is derived.
    summary: str = ""


_SPECS: tuple[ActionSpec, ...] = (
    # --- apps ---
    ActionSpec("open_app", ("name",), (), False,
               "Launch an application by friendly name or executable"),
    ActionSpec("close_app", ("name",), (), False,
               "Gracefully close an application (it may still prompt to save)"),
    # --- web ---
    ActionSpec("open_url", ("url",), (), False,
               "Open a URL in the default browser"),
    ActionSpec("web_search", ("query",), (), False,
               "Search the web in the default browser"),
    # --- files (read-only / no-clobber are non-destructive) ---
    ActionSpec("read_file", ("path",), (), False,
               "Read a text file's contents aloud"),
    ActionSpec("list_folder", ("path",), (), False,
               "List the entries in a folder"),
    ActionSpec("create_folder", ("path",), (), False,
               "Create a new folder (fails if it already exists)"),
    ActionSpec("copy_file", ("source", "destination"), (), False,
               "Copy a file (never overwrites the destination)"),
    # --- files (data-loss risk => destructive, always confirmed) ---
    ActionSpec("write_file", ("path", "content"), (), True,
               "Write text to a file, overwriting any existing content",
               "write to the file {path}, replacing its contents"),
    ActionSpec("move_file", ("source", "destination"), (), True,
               "Move or rename a file",
               "move {source} to {destination}"),
    ActionSpec("delete_file", ("path",), (), True,
               "Delete a file (Recycle Bin by default)",
               "delete the file {path}"),
    # --- processes / shell (destructive) ---
    ActionSpec("kill_process", ("name",), (), True,
               "Force-terminate a running process",
               "force-terminate the process {name}"),
    ActionSpec("run_command", ("command",), (), True,
               "Run an arbitrary shell command (only if enabled in config)",
               "run the shell command: {command}"),
    # --- system controls ---
    ActionSpec("set_volume", ("level",), (), False,
               "Set the master volume to a level from 0 to 100"),
    ActionSpec("mute", (), (), False, "Mute the system audio"),
    ActionSpec("unmute", (), (), False, "Unmute the system audio"),
    ActionSpec("lock_screen", (), (), False, "Lock the workstation"),
    ActionSpec("sleep_pc", (), (), False, "Put the computer to sleep"),
    ActionSpec("shutdown", (), (), True, "Shut the computer down",
               "shut the computer down"),
    ActionSpec("restart", (), (), True, "Restart the computer",
               "restart the computer"),
    # --- conversation / fallback ---
    ActionSpec("get_time", (), (), False, "Say the current date and time"),
    ActionSpec("respond", ("text",), (), False,
               "Speak a short reply when no OS action is needed (greetings, questions)"),
    ActionSpec("unknown", (), ("reason",), False,
               "Fallback when the request cannot be mapped to any action"),
)

ACTIONS: dict[str, ActionSpec] = {spec.name: spec for spec in _SPECS}


def is_destructive(action: str, extra_destructive: Iterable[str] = ()) -> bool:
    """True if ``action`` requires spoken confirmation before executing.

    Unrecognised action names are treated as destructive — fail-safe: if we
    don't know what it does, we don't run it unconfirmed.

    Args:
        action: Canonical action name.
        extra_destructive: Extra names to always treat as destructive
            (from ``safety.extra_destructive_actions``).
    """
    if action in extra_destructive:
        return True
    spec = ACTIONS.get(action)
    return True if spec is None else spec.destructive


def describe(action: str, params: Mapping[str, str]) -> str:
    """Human/speakable summary of an action, for confirmation prompts.

    Uses the spec's ``summary`` template when available; otherwise derives a
    generic "action name, key value, …" summary. Never raises on missing
    template params — falls back to the generic form instead.
    """
    spec = ACTIONS.get(action)
    if spec is not None and spec.summary:
        try:
            return spec.summary.format(**params)
        except (KeyError, IndexError):
            pass
    readable = action.replace("_", " ")
    if params:
        readable += ": " + ", ".join(f"{k} {v}" for k, v in params.items())
    return readable


def catalog_lines() -> list[str]:
    """Human/LLM-readable one-liners for every action (for prompt building)."""
    lines = []
    for spec in _SPECS:
        params = ", ".join(spec.required + tuple(f"{p}?" for p in spec.optional))
        lines.append(f"- {spec.name}({params}): {spec.description}")
    return lines
