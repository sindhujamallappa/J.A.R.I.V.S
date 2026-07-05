"""Pipeline orchestrator for J.A.R.I.V.S.

Wires the full voice loop: wake word → utterance capture → STT → intent →
safety gate → execution → TTS. Owns the single microphone stream (shared by
the wake listener and the capturer) and flushes it around every TTS playback
so the assistant never transcribes its own voice.

Safety: every intent passes through :class:`ConfirmationGate` before the
executor sees it. The gate's spoken yes/no round trip is provided here by
:meth:`_ask_confirmation` (speak the prompt, then capture + transcribe the
reply). The orchestrator itself performs no filesystem/process operations —
those live in the gated executor.
"""

from __future__ import annotations

import logging
from typing import Optional

from . import __version__
from .actions import is_destructive
from .config import Config
from .execution.executor import Executor
from .intent.parser import IntentError, IntentParser
from .safety.gate import ConfirmationGate
from .stt.capture import UtteranceCapturer
from .stt.transcriber import Transcriber
from .tts.speaker import Speaker
from .ui.events import BUS
from .utils.audio import MicrophoneStream, play_tone
from .wake_word.listener import WakeWordListener

log = logging.getLogger(__name__)


class Orchestrator:
    """Runs the wake → listen → understand → (confirm) → act → speak loop."""

    def __init__(
        self,
        config: Config,
        speaker: Speaker,
        *,
        transcriber: Optional[Transcriber] = None,
        parser: Optional[IntentParser] = None,
        executor: Optional[Executor] = None,
        capturer: Optional[UtteranceCapturer] = None,
        gate: Optional[ConfirmationGate] = None,
    ) -> None:
        """Heavy stages (Whisper model) load eagerly here, once — not per
        command. Keyword overrides exist for testing with fakes."""
        self._config = config
        self._speaker = speaker
        self._capturer = capturer or UtteranceCapturer(config.stt, config.audio)
        self._transcriber = transcriber or Transcriber(config.stt)
        self._parser = parser or IntentParser(config.intent)
        # Live web answers reuse the parser's LLM client; getattr keeps fake
        # parsers without .answer working in tests.
        self._executor = executor or Executor(
            config.execution,
            web=config.web_answers,
            answerer=getattr(self._parser, "answer", None),
        )
        self._gate = gate or ConfirmationGate(config.safety, ask=self._ask_confirmation)
        self._mic: Optional[MicrophoneStream] = None

    def run(self) -> None:
        """Block forever handling commands; returns only on interrupt."""
        cfg = self._config
        BUS.publish(
            "status",
            version=__version__,
            wake_model=", ".join(cfg.wake_word.models),
            stt_model=cfg.stt.model_size,
            intent_model=cfg.intent.model,
            tts_voice=cfg.tts.voice,
        )
        self._parser.warm_up()  # preload the LLM so command #1 isn't a cold start
        with MicrophoneStream(cfg.audio) as mic, \
                WakeWordListener(cfg.wake_word, cfg.audio, mic=mic) as listener:
            self._mic = mic
            self._say("Jarvis is online.")
            mic.flush()
            BUS.publish("state", state="idle")  # the HUD otherwise sticks on 'speaking'
            try:
                for detection in listener.stream():
                    log.info("Wake word fired (score=%.3f) — listening for a command",
                             detection.score)
                    try:
                        self._handle_command(mic)
                    finally:
                        mic.flush()  # drop the echo of our own reply
                        BUS.publish("state", state="idle")
            finally:
                self._mic = None
                BUS.publish("state", state="offline")

    # ------------------------------------------------------------------ #
    # One command, end to end
    # ------------------------------------------------------------------ #
    def _handle_command(self, mic: MicrophoneStream) -> None:
        """Capture, understand, authorize, execute, and answer one command."""
        import time as _time

        timings: dict[str, float] = {}

        def _timed(stage: str, fn):  # noqa: ANN001, ANN202 — tiny local helper
            start = _time.perf_counter()
            try:
                return fn()
            finally:
                timings[stage] = _time.perf_counter() - start

        try:
            mic.flush()  # drop the tail of the wake phrase
            ack = self._config.wake_word.ack_phrase
            if ack:
                _timed("ack", lambda: self._say(ack))
                mic.flush()  # never capture our own acknowledgment
            BUS.publish("state", state="listening")
            self._beep()

            utterance = _timed("capture", lambda: self._capturer.capture(mic))
            if utterance is None:
                self._say("I didn't hear a command.")
                return

            BUS.publish("state", state="transcribing")
            text = _timed("stt", lambda: self._transcriber.transcribe(utterance).text)
            if not text.strip():
                self._say("Sorry, I didn't catch that.")
                return
            BUS.publish("transcript", who="user", text=text)

            BUS.publish("state", state="thinking")
            try:
                intent = _timed("intent", lambda: self._parser.parse(text))
            except IntentError:
                log.exception("Intent backend unavailable")
                self._say("I can't reach my language model right now.")
                return
            BUS.publish(
                "intent",
                action=intent.action,
                params=intent.params,
                destructive=is_destructive(
                    intent.action, self._config.safety.extra_destructive_actions
                ),
            )

            # Gate time includes the spoken yes/no round trip when destructive.
            verdict = _timed("gate", lambda: self._gate.authorize(intent))
            BUS.publish("gate", allowed=verdict.allowed, reason=verdict.reason)
            if not verdict.allowed:
                log.info("Gate refused %r: %s", intent.action, verdict.reason)
                if verdict.reason == "denied by user":
                    self._say("Okay, cancelled.")
                else:
                    self._say("I didn't get a clear yes, so I didn't do it.")
                return

            BUS.publish("state", state="executing")
            result = _timed("exec", lambda: self._executor.execute(intent))
            BUS.publish("result", ok=result.ok, message=result.message)
            _timed("speak", lambda: self._say(result.message))
        finally:
            if timings:
                log.info(
                    "Command timing: %s (total %.2fs)",
                    " ".join(f"{k}={v:.2f}s" for k, v in timings.items()),
                    sum(timings.values()),
                )

    # ------------------------------------------------------------------ #
    # Gate plumbing: speak a prompt, hear the reply
    # ------------------------------------------------------------------ #
    def _ask_confirmation(self, prompt: str) -> Optional[str]:
        """The gate's ask(): speak ``prompt``, return the transcribed reply.

        Returns ``None`` when nothing intelligible was heard — the gate
        treats that as unclear and ultimately denies (fail-safe).
        """
        if self._mic is None:
            log.error("Confirmation requested with no open microphone — denying")
            return None
        self._say(prompt)
        BUS.publish("state", state="confirming")
        self._mic.flush()  # never transcribe our own prompt
        utterance = self._capturer.capture(self._mic)
        if utterance is None:
            return None
        reply = self._transcriber.transcribe(utterance).text
        if reply:
            BUS.publish("transcript", who="user", text=reply)
        return reply or None

    def _say(self, text: str) -> None:
        """Speak ``text`` and mirror it to the HUD transcript."""
        BUS.publish("transcript", who="jarvis", text=text)
        BUS.publish("state", state="speaking")
        self._speaker.speak(text)

    def _beep(self) -> None:
        """Play the listening cue; never let audio-out kill the loop."""
        try:
            play_tone()
        except Exception:  # noqa: BLE001
            log.warning("Earcon playback failed", exc_info=True)
