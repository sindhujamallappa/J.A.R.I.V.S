"""Thread-safe event bus feeding the J.A.R.I.V.S HUD.

Pipeline stages publish small JSON events (state changes, transcripts,
intents, log lines, audio meter ticks); the SSE server fans them out to any
connected browser. Publishing with no subscribers is a cheap no-op, so the
pipeline can emit unconditionally whether or not the UI is enabled.

Classification: in-memory pub/sub only — non-destructive.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from typing import Any

# Recent events replayed to newly connected clients so the HUD isn't blank.
_HISTORY_LIMIT = 200
# Per-client buffer; a stalled browser drops events rather than blocking us.
_QUEUE_LIMIT = 500
# High-frequency ephemeral events that shouldn't clutter the replay history.
_TRANSIENT_TYPES = {"meter"}


class EventBus:
    """Fan-out pub/sub with bounded per-subscriber queues and history replay."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[str]] = []
        self._history: deque[str] = deque(maxlen=_HISTORY_LIMIT)

    def publish(self, event_type: str, **payload: Any) -> None:
        """Broadcast one event to all subscribers (never blocks, never raises)."""
        event = json.dumps(
            {"type": event_type, "ts": time.time(), **payload},
            default=str, ensure_ascii=False,
        )
        with self._lock:
            if event_type not in _TRANSIENT_TYPES:
                self._history.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow client: drop for them, never stall the pipeline

    def subscribe(self, replay: bool = True) -> "queue.Queue[str]":
        """Register a subscriber; optionally preload recent history."""
        q: queue.Queue[str] = queue.Queue(maxsize=_QUEUE_LIMIT)
        with self._lock:
            if replay:
                for event in self._history:
                    q.put_nowait(event)
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[str]") -> None:
        """Remove a subscriber (idempotent)."""
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


#: Process-wide bus shared by the pipeline and the HUD server.
BUS = EventBus()
