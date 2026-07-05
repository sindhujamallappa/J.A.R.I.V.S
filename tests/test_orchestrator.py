"""Tests for the orchestrator's command loop (all stages faked)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

import jarvis.orchestrator as orch_mod
from jarvis.config import (
    AudioConfig, Config, ExecutionConfig, IntentConfig, LoggingConfig,
    STTConfig, SafetyConfig, ServiceConfig, TTSConfig, UIConfig, WakeWordConfig,
)
from jarvis.execution.executor import ExecutionResult
from jarvis.intent.parser import Intent, IntentError
from jarvis.orchestrator import Orchestrator
from jarvis.safety.gate import ConfirmationGate, GateResult
from jarvis.stt.transcriber import Transcription

CONFIG = Config(
    logging=LoggingConfig(), audio=AudioConfig(), wake_word=WakeWordConfig(),
    safety=SafetyConfig(), execution=ExecutionConfig(), stt=STTConfig(),
    intent=IntentConfig(), tts=TTSConfig(), ui=UIConfig(enabled=False),
    service=ServiceConfig(),
)

UTTERANCE = np.zeros(16000, dtype=np.int16)


class FakeMic:
    def __init__(self) -> None:
        self.flushes = 0

    def read(self) -> np.ndarray:
        return np.zeros(1280, dtype=np.int16)

    def flush(self) -> None:
        self.flushes += 1


@dataclass
class FakeSpeaker:
    spoken: list[str] = field(default_factory=list)

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeCapturer:
    def __init__(self, *utterances: Optional[np.ndarray]) -> None:
        self._utterances = iter(utterances)

    def capture(self, mic) -> Optional[np.ndarray]:
        return next(self._utterances, None)


class FakeTranscriber:
    def __init__(self, *texts: str) -> None:
        self._texts = iter(texts)

    def transcribe(self, audio) -> Transcription:
        return Transcription(next(self._texts, ""), "en", 1.0)


class FakeParser:
    def __init__(self, intent: Optional[Intent] = None, exc: Optional[Exception] = None) -> None:
        self._intent, self._exc = intent, exc

    def parse(self, text: str) -> Intent:
        if self._exc is not None:
            raise self._exc
        assert self._intent is not None
        return self._intent


class FakeGate:
    def __init__(self, result: GateResult) -> None:
        self._result = result

    def authorize(self, intent: Intent) -> GateResult:
        return self._result


class FakeExecutor:
    def __init__(self, result: ExecutionResult = ExecutionResult(True, "Done.")) -> None:
        self._result = result
        self.executed: list[Intent] = []

    def execute(self, intent: Intent) -> ExecutionResult:
        self.executed.append(intent)
        return self._result


@pytest.fixture(autouse=True)
def _silence_earcon(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(orch_mod, "play_tone", lambda *a, **k: None)


def _orchestrator(**stages) -> tuple[Orchestrator, FakeSpeaker]:
    speaker = FakeSpeaker()
    defaults = dict(
        capturer=FakeCapturer(UTTERANCE),
        transcriber=FakeTranscriber("open notepad"),
        parser=FakeParser(Intent("open_app", {"name": "notepad"}, "open notepad")),
        gate=FakeGate(GateResult(True, "non-destructive")),
        executor=FakeExecutor(ExecutionResult(True, "Opening notepad.")),
    )
    defaults.update(stages)
    return Orchestrator(CONFIG, speaker, **defaults), speaker


# --------------------------------------------------------------------------- #
# _handle_command paths
# --------------------------------------------------------------------------- #
def test_happy_path_executes_and_speaks_result():
    executor = FakeExecutor(ExecutionResult(True, "Opening notepad."))
    orch, speaker = _orchestrator(executor=executor)
    orch._handle_command(FakeMic())
    assert [i.action for i in executor.executed] == ["open_app"]
    assert speaker.spoken == ["Opening notepad."]


def test_no_speech_apologises_without_executing():
    executor = FakeExecutor()
    orch, speaker = _orchestrator(capturer=FakeCapturer(None), executor=executor)
    orch._handle_command(FakeMic())
    assert executor.executed == []
    assert speaker.spoken == ["I didn't hear a command."]


def test_empty_transcript_apologises_without_executing():
    executor = FakeExecutor()
    orch, speaker = _orchestrator(transcriber=FakeTranscriber("   "), executor=executor)
    orch._handle_command(FakeMic())
    assert executor.executed == []
    assert speaker.spoken == ["Sorry, I didn't catch that."]


def test_intent_backend_down_apologises_without_executing():
    executor = FakeExecutor()
    orch, speaker = _orchestrator(
        parser=FakeParser(exc=IntentError("refused")), executor=executor
    )
    orch._handle_command(FakeMic())
    assert executor.executed == []
    assert "language model" in speaker.spoken[0]


def test_gate_denial_blocks_execution():
    executor = FakeExecutor()
    orch, speaker = _orchestrator(
        gate=FakeGate(GateResult(False, "denied by user")), executor=executor
    )
    orch._handle_command(FakeMic())
    assert executor.executed == []
    assert speaker.spoken == ["Okay, cancelled."]


def test_unclear_confirmation_blocks_execution_with_failsafe_message():
    executor = FakeExecutor()
    orch, speaker = _orchestrator(
        gate=FakeGate(GateResult(False, "no clear confirmation — denied")),
        executor=executor,
    )
    orch._handle_command(FakeMic())
    assert executor.executed == []
    assert "didn't get a clear yes" in speaker.spoken[0]


def test_mic_flushed_before_capture():
    mic = FakeMic()
    orch, _ = _orchestrator()
    orch._handle_command(mic)
    assert mic.flushes >= 1


def test_wake_ack_spoken_before_command():
    config = Config(
        logging=LoggingConfig(), audio=AudioConfig(),
        wake_word=WakeWordConfig(ack_phrase="Yes Boss"),
        safety=SafetyConfig(), execution=ExecutionConfig(), stt=STTConfig(),
        intent=IntentConfig(), tts=TTSConfig(), ui=UIConfig(enabled=False),
        service=ServiceConfig(),
    )
    speaker = FakeSpeaker()
    mic = FakeMic()
    orch = Orchestrator(
        config, speaker,
        capturer=FakeCapturer(UTTERANCE),
        transcriber=FakeTranscriber("open notepad"),
        parser=FakeParser(Intent("open_app", {"name": "notepad"}, "open notepad")),
        gate=FakeGate(GateResult(True, "non-destructive")),
        executor=FakeExecutor(ExecutionResult(True, "Opening notepad.")),
    )
    orch._handle_command(mic)
    assert speaker.spoken == ["Yes Boss", "Opening notepad."]
    assert mic.flushes >= 2  # wake-phrase tail AND our own acknowledgment


# --------------------------------------------------------------------------- #
# Confirmation round trip (_ask_confirmation wired into a real gate)
# --------------------------------------------------------------------------- #
def test_ask_confirmation_speaks_prompt_and_returns_reply():
    orch, speaker = _orchestrator(
        capturer=FakeCapturer(UTTERANCE),
        transcriber=FakeTranscriber("yes"),
    )
    orch._mic = FakeMic()
    reply = orch._ask_confirmation("This will delete x. Proceed?")
    assert reply == "yes"
    assert speaker.spoken == ["This will delete x. Proceed?"]


def test_ask_confirmation_without_mic_denies():
    orch, _ = _orchestrator()
    orch._mic = None
    assert orch._ask_confirmation("Proceed?") is None


def test_pipeline_events_published_to_bus():
    """The HUD bus must see the state walk and the key payload events."""
    import json

    from jarvis.ui.events import BUS

    q = BUS.subscribe(replay=False)
    try:
        orch, _ = _orchestrator()
        orch._handle_command(FakeMic())
        events = []
        while True:
            try:
                events.append(json.loads(q.get_nowait()))
            except Exception:
                break
        states = [e["state"] for e in events if e["type"] == "state"]
        types = {e["type"] for e in events}
        assert states[:4] == ["listening", "transcribing", "thinking", "executing"]
        assert {"transcript", "intent", "gate", "result"} <= types
        intent_evt = next(e for e in events if e["type"] == "intent")
        assert intent_evt["action"] == "open_app"
        assert intent_evt["destructive"] is False
    finally:
        BUS.unsubscribe(q)


def test_real_gate_full_round_trip_deny():
    """End-to-end: destructive intent + real gate + spoken 'no' => no execution."""
    executor = FakeExecutor()
    speaker = FakeSpeaker()
    orch = Orchestrator(
        CONFIG, speaker,
        capturer=FakeCapturer(UTTERANCE, UTTERANCE),          # command, then reply
        transcriber=FakeTranscriber("delete report.txt", "no"),
        parser=FakeParser(Intent("delete_file", {"path": "report.txt"}, "delete report.txt")),
        executor=executor,
    )
    orch._mic = FakeMic()
    orch._handle_command(orch._mic)
    assert executor.executed == []                     # gate blocked it
    assert any("report.txt" in s for s in speaker.spoken)  # prompt named the file
    assert speaker.spoken[-1] == "Okay, cancelled."
