"""Tests for the HUD event bus, log bridge, and SSE server."""

from __future__ import annotations

import http.client
import json
import logging
import queue

import pytest

from jarvis.config import UIConfig
from jarvis.ui.events import EventBus
from jarvis.ui.log_stream import BusLogHandler
from jarvis.ui.server import UIServer


# --------------------------------------------------------------------------- #
# EventBus
# --------------------------------------------------------------------------- #
def test_publish_reaches_subscriber():
    bus = EventBus()
    q = bus.subscribe()
    bus.publish("state", state="idle")
    event = json.loads(q.get_nowait())
    assert event["type"] == "state" and event["state"] == "idle"
    assert "ts" in event


def test_history_replayed_to_late_subscriber():
    bus = EventBus()
    bus.publish("transcript", who="user", text="hello")
    q = bus.subscribe(replay=True)
    assert json.loads(q.get_nowait())["text"] == "hello"


def test_meter_events_not_replayed():
    bus = EventBus()
    bus.publish("meter", level=0.5)
    bus.publish("state", state="idle")
    q = bus.subscribe(replay=True)
    types = []
    while True:
        try:
            types.append(json.loads(q.get_nowait())["type"])
        except queue.Empty:
            break
    assert types == ["state"]  # meter tick was transient


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    bus.publish("state", state="idle")
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_full_subscriber_queue_never_blocks_publisher():
    bus = EventBus()
    q = bus.subscribe()
    for i in range(2000):  # far beyond the queue bound
        bus.publish("log", i=i)
    # No exception raised, and the queue holds at most its bound.
    assert q.qsize() <= 500


# --------------------------------------------------------------------------- #
# BusLogHandler
# --------------------------------------------------------------------------- #
def _record(name: str, msg: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, None, None)


def test_log_handler_forwards_jarvis_records():
    bus = EventBus()
    q = bus.subscribe()
    handler = BusLogHandler(bus)
    handler.emit(_record("src.jarvis.stt.capture", "captured 1.9s"))
    event = json.loads(q.get_nowait())
    assert event["type"] == "log"
    assert event["logger"] == "stt.capture"
    assert event["msg"] == "captured 1.9s"
    assert event["level"] == "INFO"


def test_log_handler_ignores_third_party_records():
    bus = EventBus()
    q = bus.subscribe()
    handler = BusLogHandler(bus)
    handler.emit(_record("httpx", "GET /"))
    handler.emit(_record("urllib3.connectionpool", "new connection"))
    with pytest.raises(queue.Empty):
        q.get_nowait()


# --------------------------------------------------------------------------- #
# UIServer (real HTTP on an ephemeral port)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def server():
    bus = EventBus()
    srv = UIServer(UIConfig(port=0), bus)  # port 0 -> OS-assigned
    srv.start()
    yield srv, bus
    srv.stop()


def _port(srv: UIServer) -> int:
    return srv._httpd.server_address[1]


def test_serves_hud_page(server):
    srv, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", _port(srv), timeout=5)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "J.A.R.V.I.S" in body and "EventSource" in body
    conn.close()


def test_unknown_path_404s(server):
    srv, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", _port(srv), timeout=5)
    conn.request("GET", "/nope")
    assert conn.getresponse().status == 404
    conn.close()


def test_sse_stream_delivers_events(server):
    srv, bus = server
    bus.publish("state", state="listening")  # lands in history -> replayed
    conn = http.client.HTTPConnection("127.0.0.1", _port(srv), timeout=5)
    conn.request("GET", "/events")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "text/event-stream"
    line = resp.fp.readline().decode("utf-8").strip()
    assert line.startswith("data: ")
    event = json.loads(line[len("data: "):])
    assert event == {"type": "state", "state": "listening", "ts": event["ts"]}
    conn.close()


def test_log_records_flow_to_sse_while_running(server):
    srv, bus = server
    q = bus.subscribe(replay=False)
    logger = logging.getLogger("jarvis.test_ui")
    logger.setLevel(logging.INFO)  # pytest leaves root at WARNING
    logger.info("hello hud")
    event = json.loads(q.get(timeout=2))
    assert event["type"] == "log" and event["msg"] == "hello hud"
