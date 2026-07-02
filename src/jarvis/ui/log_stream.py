"""Live log streaming into the HUD.

A ``logging.Handler`` that forwards J.A.R.I.V.S log records to the event bus
so the HUD's SYSTEM LOG panel updates in real time. Only records from our own
modules are forwarded — third-party loggers (httpx, urllib3, …) would be
noise and, worse, a server logging its own sends would loop forever.

Classification: log fan-out only — non-destructive.
"""

from __future__ import annotations

import logging
import time

from .events import EventBus


class BusLogHandler(logging.Handler):
    """Forwards jarvis-namespace log records to an :class:`EventBus`."""

    def __init__(self, bus: EventBus, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._bus = bus

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 (base doc)
        if "jarvis" not in record.name.split("."):
            return
        try:
            # Trim the package prefix for display: src.jarvis.stt.capture -> stt.capture
            parts = record.name.split(".")
            short = ".".join(parts[parts.index("jarvis") + 1:]) or "core"
            self._bus.publish(
                "log",
                level=record.levelname,
                logger=short,
                msg=record.getMessage(),
                t=time.strftime("%H:%M:%S", time.localtime(record.created)),
            )
        except Exception:  # noqa: BLE001 — logging must never take down the app
            self.handleError(record)
