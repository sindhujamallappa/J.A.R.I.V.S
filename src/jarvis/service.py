"""Background-service management for J.A.R.I.V.S on Windows.

Provides a supervised runner (auto-restart with exponential backoff, plus a
single-instance guard so two copies never fight over the microphone) and thin
wrappers around ``schtasks`` to install/uninstall/query a Task Scheduler entry
that launches J.A.R.I.V.S at logon.

CLI::

    python -m src.jarvis.service run          # supervised foreground/background run
    python -m src.jarvis.service install      # register the logon task
    python -m src.jarvis.service uninstall    # remove it (prompts to confirm)
    python -m src.jarvis.service status       # query the task

Classification: ``install``/``uninstall`` create/delete a Task Scheduler object
via ``schtasks`` — developer-invoked setup commands, not part of the runtime
voice pipeline, so they sit outside the destructive-action gate. ``uninstall``
is nonetheless confirmed at the console before deleting. The supervised runner
performs no filesystem/process mutation itself; all voice-commanded actions
still route through the gated executor inside the orchestrator.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional
from xml.sax.saxutils import escape

from .config import Config, ServiceConfig, load_config
from .utils.logging_config import configure_logging

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Paths / environment helpers
# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    """Repo root (the directory containing ``src/`` and ``config/``)."""
    return Path(__file__).resolve().parents[2]


def _lock_path() -> Path:
    return _project_root() / ".jarvis.lock"


def _resolve_python(run_hidden: bool) -> str:
    """Path to the interpreter to launch. Prefers pythonw.exe when hidden."""
    base = Path(sys.executable)
    if run_hidden:
        pythonw = base.with_name("pythonw.exe")
        if pythonw.is_file():
            return str(pythonw)
    return str(base)


def _mutex_name(task_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", task_name)
    return f"Local\\JARVIS_{safe}"


# --------------------------------------------------------------------------- #
# Single-instance guard
# --------------------------------------------------------------------------- #
class SingleInstance:
    """Cross-process guard: a Windows named mutex, or a POSIX file lock.

    ``acquire`` returns ``True`` if this process got the lock, ``False`` if
    another instance already holds it. The lock is released on :meth:`release`
    or process exit.
    """

    def __init__(self, name: str, lock_path: Path) -> None:
        self._name = name
        self._lock_path = lock_path
        self._handle: Optional[object] = None
        self._fp = None
        self._acquired = False

    def acquire(self) -> bool:
        if os.name == "nt":
            return self._acquire_mutex()
        return self._acquire_flock()

    def _acquire_mutex(self) -> bool:
        import ctypes

        error_already_exists = 183
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.CreateMutexW(None, False, self._name)
        last_error = kernel32.GetLastError()
        self._handle = handle
        self._acquired = handle != 0 and last_error != error_already_exists
        return self._acquired

    def _acquire_flock(self) -> bool:
        import fcntl

        self._fp = open(self._lock_path, "w")
        try:
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._acquired = True
        except OSError:
            self._acquired = False
        return self._acquired

    def release(self) -> None:
        if os.name == "nt" and self._handle:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(self._handle)  # type: ignore[attr-defined]
            self._handle = None
        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                self._fp = None

    def __enter__(self) -> "SingleInstance":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Supervised runner
# --------------------------------------------------------------------------- #
def _supervise(
    run_once: Callable[[], None],
    policy: ServiceConfig,
    sleep: Callable[[float], None] = time.sleep,
    max_restarts: Optional[int] = None,
) -> int:
    """Run ``run_once`` until it returns cleanly; restart on crash with backoff.

    Args:
        run_once: Blocking callable (the orchestrator loop). Returns on clean
            stop; raising means a crash to recover from.
        policy: Restart configuration.
        sleep: Injected for testing.
        max_restarts: Optional cap (for testing); ``None`` means unlimited.

    Returns:
        Process exit code (0 on clean stop, 1 if giving up).
    """
    attempts = 0
    backoff = policy.restart_backoff_sec
    while True:
        try:
            run_once()
            return 0
        except KeyboardInterrupt:
            log.info("Interrupted — stopping service")
            return 0
        except Exception:  # noqa: BLE001 — supervisor recovers from any crash
            log.exception("J.A.R.I.V.S crashed")
            if not policy.restart_on_crash:
                return 1
            attempts += 1
            if max_restarts is not None and attempts > max_restarts:
                log.error("Exceeded restart budget (%d) — giving up", max_restarts)
                return 1
            log.info("Restarting in %.1fs (attempt %d)", backoff, attempts)
            sleep(backoff)
            backoff = min(backoff * 2, policy.restart_backoff_max_sec)


def _run_once(config: Config) -> None:
    # Imported here so install/uninstall/status (and the tests) work while the
    # orchestrator/TTS stages are still landing on the roadmap.
    from .orchestrator import Orchestrator
    from .tts.speaker import build_speaker

    Orchestrator(config, speaker=build_speaker(config.tts)).run()


def run_supervised(config: Config) -> int:
    """Acquire the single-instance lock (if enabled) and run supervised."""
    guard = SingleInstance(_mutex_name(config.service.task_name), _lock_path())
    if config.service.single_instance and not guard.acquire():
        log.error("Another J.A.R.I.V.S instance is already running — exiting")
        return 1
    try:
        return _supervise(lambda: _run_once(config), config.service)
    finally:
        guard.release()


# --------------------------------------------------------------------------- #
# Task Scheduler XML + schtasks wrappers (Windows only)
# --------------------------------------------------------------------------- #
def build_task_xml(
    *,
    task_name: str,
    command: str,
    arguments: str,
    working_dir: str,
    user_id: str,
    restart_count: int = 3,
    restart_interval: str = "PT1M",
) -> str:
    """Build a Task Scheduler v1.2 XML definition (logon-triggered, restarting).

    Values are XML-escaped. The app's own supervisor handles fast restarts;
    this Task Scheduler restart is the backstop if the whole process dies.
    """
    c, a = escape(command), escape(arguments)
    wd, u, tn = escape(working_dir), escape(user_id), escape(task_name)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>J.A.R.I.V.S offline voice assistant ({tn})</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{u}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{u}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Enabled>true</Enabled>
    <RestartOnFailure>
      <Interval>{restart_interval}</Interval>
      <Count>{restart_count}</Count>
    </RestartOnFailure>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
    </IdleSettings>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{c}</Command>
      <Arguments>{a}</Arguments>
      <WorkingDirectory>{wd}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _require_windows() -> None:
    if os.name != "nt":
        raise RuntimeError("Task Scheduler commands are only available on Windows")


def install(config: Config) -> None:
    """Register (or replace) the logon task via schtasks."""
    _require_windows()
    import getpass

    root = _project_root()
    command = _resolve_python(config.service.run_hidden)
    xml = build_task_xml(
        task_name=config.service.task_name,
        command=command,
        arguments="-m src.jarvis.service run",
        working_dir=str(root),
        user_id=getpass.getuser(),
    )

    # schtasks expects UTF-16 XML.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".xml", encoding="utf-16", delete=False
    ) as fh:
        fh.write(xml)
        xml_path = fh.name
    try:
        subprocess.run(
            ["schtasks", "/Create", "/TN", config.service.task_name,
             "/XML", xml_path, "/F"],
            check=True,
        )
        log.info("Installed scheduled task %r (runs: %s %s)",
                 config.service.task_name, command, "-m src.jarvis.service run")
    finally:
        os.unlink(xml_path)


def uninstall(config: Config, assume_yes: bool = False) -> None:
    """Delete the logon task (confirmed at the console unless ``assume_yes``)."""
    _require_windows()
    name = config.service.task_name
    if not assume_yes:
        answer = input(f"Delete scheduled task {name!r}? [y/n] ").strip().lower()
        if answer not in {"y", "yes"}:
            log.info("Uninstall cancelled")
            return
    subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], check=True)
    log.info("Removed scheduled task %r", name)


def status(config: Config) -> int:
    """Print the task's current state; return the schtasks exit code."""
    _require_windows()
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", config.service.task_name],
        capture_output=True, text=True,
    )
    print(result.stdout or result.stderr)
    return result.returncode


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="jarvis-service")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="run supervised (used by the scheduled task)")
    sub.add_parser("install", help="register the logon task")
    p_uninstall = sub.add_parser("uninstall", help="remove the logon task")
    p_uninstall.add_argument("--yes", action="store_true", help="skip confirmation")
    sub.add_parser("status", help="query the logon task")
    args = parser.parse_args(argv)

    config = load_config()
    configure_logging(config.logging)

    if args.cmd == "run":
        return run_supervised(config)
    if args.cmd == "install":
        install(config)
        return 0
    if args.cmd == "uninstall":
        uninstall(config, assume_yes=args.yes)
        return 0
    if args.cmd == "status":
        return status(config)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
