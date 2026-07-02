"""Local HUD web server for J.A.R.I.V.S — stdlib only, loopback only.

Serves the static HUD page and a Server-Sent Events stream (``/events``)
carrying live pipeline events from the :mod:`jarvis.ui.events` bus. Runs on a
daemon thread so it never blocks the voice loop; binds to loopback by default
so nothing is exposed off-machine (privacy-first).

Classification: read-only telemetry over localhost HTTP — non-destructive.
It renders what the pipeline does; it cannot command anything.
"""

from __future__ import annotations

import logging
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from ..config import UIConfig
from .events import BUS, EventBus
from .log_stream import BusLogHandler

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_SSE_HEARTBEAT_SEC = 15.0


def _make_handler(bus: EventBus, index_html: bytes) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to this server's bus and page."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "JarvisHUD/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            # Default handler prints to stderr per request — silence it; a
            # forwarded access log would echo through the bus forever.
            pass

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path in ("/", "/index.html"):
                self._serve_index()
            elif self.path == "/events":
                self._serve_events()
            else:
                self.send_error(404)

        def _serve_index(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(index_html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(index_html)

        def _serve_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q = bus.subscribe(replay=True)
            try:
                while True:
                    try:
                        event = q.get(timeout=_SSE_HEARTBEAT_SEC)
                        self.wfile.write(f"data: {event}\n\n".encode("utf-8"))
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")  # keep-alive comment
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass  # client closed the tab — normal
            finally:
                bus.unsubscribe(q)

    return Handler


class UIServer:
    """Owns the HTTP server thread and the live-log bridge."""

    def __init__(self, cfg: UIConfig, bus: EventBus = BUS) -> None:
        self._cfg = cfg
        self._bus = bus
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._log_handler: Optional[BusLogHandler] = None

    @property
    def url(self) -> str:
        """The HUD's address (valid after :meth:`start`)."""
        host, port = self._cfg.host, self._cfg.port
        if self._httpd is not None:
            host, port = self._httpd.server_address[:2]  # resolves port 0
        return f"http://{host}:{port}/"

    def start(self) -> None:
        """Bind, start serving on a daemon thread, and bridge live logs."""
        index_html = (_STATIC_DIR / "index.html").read_bytes()
        handler = _make_handler(self._bus, index_html)
        self._httpd = ThreadingHTTPServer((self._cfg.host, self._cfg.port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="jarvis-hud", daemon=True
        )
        self._thread.start()
        self._log_handler = BusLogHandler(self._bus)
        logging.getLogger().addHandler(self._log_handler)
        log.info("HUD available at %s", self.url)

    def stop(self) -> None:
        """Shut the server down and detach the log bridge (idempotent)."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None


def maybe_start_ui(cfg: UIConfig) -> Optional[UIServer]:
    """Start the HUD if enabled; UI failure must never block the voice loop."""
    if not cfg.enabled:
        return None
    server = UIServer(cfg)
    try:
        server.start()
    except Exception:  # noqa: BLE001 — HUD is optional; the assistant is not
        log.exception("HUD failed to start — continuing without UI")
        return None
    if cfg.open_browser:
        try:
            webbrowser.open(server.url)
        except Exception:  # noqa: BLE001
            log.warning("Could not open a browser for the HUD", exc_info=True)
    return server
