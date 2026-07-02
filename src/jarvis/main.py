"""J.A.R.I.V.S entrypoint.

Builds config + logging, then runs the full voice pipeline:
wake word → STT → intent → safety gate → execution → TTS.

Run:
    python -m src.jarvis.main

For the supervised background-service form (auto-restart, single instance),
use ``python -m src.jarvis.service run`` instead.
"""

from __future__ import annotations

import logging
import sys

from .config import ConfigError, load_config
from .orchestrator import Orchestrator
from .tts.speaker import build_speaker
from .utils.logging_config import configure_logging

log = logging.getLogger("jarvis")


def main() -> int:
    """Program entrypoint. Returns a process exit code."""
    try:
        cfg = load_config()
    except ConfigError as exc:
        # Logging isn't configured yet; go straight to stderr.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(cfg.logging)
    log.info("Starting %s", "J.A.R.I.V.S")

    try:
        Orchestrator(cfg, speaker=build_speaker(cfg.tts)).run()
    except KeyboardInterrupt:
        log.info("Shutdown requested — goodbye")
    except Exception:  # noqa: BLE001 — top-level guard for graceful exit
        log.exception("Fatal error in main loop")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
