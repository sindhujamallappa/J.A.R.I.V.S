"""J.A.R.I.V.S entrypoint.

Builds config + logging and runs the pipeline. As of this stage only the
wake-word listener is wired up; STT / intent / safety / execution / TTS get
attached to the orchestrator as they land on the roadmap.

Run:
    python -m src.jarvis.main
"""

from __future__ import annotations

import logging
import sys

from .config import ConfigError, load_config
from .utils.logging_config import configure_logging
from .wake_word.listener import WakeWordListener

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
        with WakeWordListener(cfg.wake_word, cfg.audio) as ww:
            for detection in ww.stream():
                # TODO(orchestrator): hand off to STT -> intent -> safety -> exec -> TTS
                log.info(
                    "Wake word '%s' fired (score=%.3f) — pipeline continues here",
                    detection.model_name,
                    detection.score,
                )
    except KeyboardInterrupt:
        log.info("Shutdown requested — goodbye")
    except Exception:  # noqa: BLE001 — top-level guard for graceful exit
        log.exception("Fatal error in main loop")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
