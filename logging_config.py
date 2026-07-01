"""Structured logging setup for J.A.R.I.V.S.

Configures the root logger with a console handler and an optional rotating
file handler. Supports a human-readable ``text`` format and a machine-readable
``json`` format (no third-party dependency). Idempotent: calling
:func:`configure_logging` twice replaces handlers rather than stacking them.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

from ..config import LoggingConfig

# Standard LogRecord attributes we don't want to duplicate as JSON "extras".
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
}

_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 (shadow ok)
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Structured extras passed via logger.info(..., extra={...}).
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(cfg: LoggingConfig) -> None:
    """Configure the root logger from a :class:`LoggingConfig`.

    Args:
        cfg: Validated logging configuration.
    """
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    formatter: logging.Formatter = (
        _JsonFormatter() if cfg.format == "json"
        else logging.Formatter(_TEXT_FORMAT, datefmt=_DATE_FORMAT)
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Drop existing handlers so repeated calls stay idempotent.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if cfg.file:
        log_path = Path(cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.getLogger(__name__).debug(
        "Logging configured", extra={"level": cfg.level, "format": cfg.format}
    )
