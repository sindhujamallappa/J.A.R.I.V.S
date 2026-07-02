"""Configuration loading and validation for J.A.R.I.V.S.

Reads ``config/config.yaml``, applies a small set of environment-variable
overrides, and returns typed, validated config objects. This module is the
single source of truth for runtime settings — nothing downstream should read
raw YAML or ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from dotenv import load_dotenv

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_LOG_FORMATS = {"text", "json"}
_VALID_FRAMEWORKS = {"onnx", "tflite"}


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or out of range."""


# --------------------------------------------------------------------------- #
# Typed sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "text"
    file: Optional[str] = "logs/jarvis.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_length: int = 1280
    channels: int = 1
    # None -> system default. May be an int index or a name substring.
    input_device: Optional[str] = None


@dataclass(frozen=True)
class WakeWordConfig:
    model: str = "hey_jarvis"
    threshold: float = 0.5
    inference_framework: str = "onnx"
    vad_threshold: float = 0.0
    enable_noise_suppression: bool = False
    trigger_cooldown_sec: float = 2.0
    models_dir: str = "models"


@dataclass(frozen=True)
class SafetyConfig:
    # Treat actions we don't recognise as needing confirmation (fail-safe).
    confirm_unknown_actions: bool = True
    # Re-prompts allowed on an unclear yes/no before defaulting to DENY.
    confirmation_retries: int = 2
    # Spoken before a destructive action. "{action}" is filled with a summary.
    confirmation_prompt: str = "This will {action}. Should I proceed? Say yes or no."
    reprompt: str = "Sorry, I didn't catch a yes or no. Please say yes or no."
    # Extra action names to always treat as destructive (merged with built-ins).
    extra_destructive_actions: tuple[str, ...] = ()
    affirmatives: tuple[str, ...] = (
        "yes", "yeah", "yep", "yup", "sure", "confirm", "confirmed",
        "proceed", "go ahead", "do it", "okay", "ok", "affirmative",
    )
    negatives: tuple[str, ...] = (
        "no", "nope", "nah", "cancel", "stop", "don't", "do not",
        "abort", "negative", "never mind", "nevermind",
    )


@dataclass(frozen=True)
class ExecutionConfig:
    # Friendly name -> executable/path, e.g. {"notes": "notepad.exe"}.
    apps: dict[str, str] = field(default_factory=dict)
    # Arbitrary shell exec (run_command). Gated + logged, but OFF by default.
    allow_run_command: bool = False
    # delete_file sends to the Recycle Bin instead of permanent removal.
    delete_to_trash: bool = True
    search_url: str = "https://www.google.com/search?q={query}"
    read_file_max_chars: int = 2000


@dataclass(frozen=True)
class STTConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"          # "" for autodetect
    beam_size: int = 1
    # --- utterance capture / silence endpointing (int16 RMS domain) ---
    min_rms_threshold: float = 300.0    # noise-floor guard for onset
    speech_multiplier: float = 2.5      # onset = max(min_rms, ambient * this)
    calibration_sec: float = 0.3        # ambient-noise sampling window
    silence_duration_sec: float = 0.8   # trailing silence that ends capture
    max_duration_sec: float = 15.0      # hard cap on one utterance
    onset_timeout_sec: float = 4.0      # give up if speech never starts
    preroll_sec: float = 0.3            # audio kept from before onset
    models_dir: str = "models/whisper"  # where Whisper model files download to


@dataclass(frozen=True)
class IntentConfig:
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    timeout_sec: float = 60.0  # must cover a cold model load, not just inference
    temperature: float = 0.0   # deterministic structured output
    # How long Ollama keeps the model in RAM after a request. Without this the
    # model unloads after ~5 idle minutes and the next command hits a cold
    # (30s+) reload.
    keep_alive: str = "30m"
    max_tokens: int = 256      # intent JSON is small; cap runaway generations


@dataclass(frozen=True)
class TTSConfig:
    engine: str = "piper"
    voice: str = "en_US-lessac-medium"  # bundled voice name, or a path to a .onnx
    models_dir: str = "models/piper"
    speaking_rate: float = 1.0           # >1 faster, <1 slower (maps to length_scale)
    speaker_id: int = 0                  # for multi-speaker voices
    volume: float = 1.0


@dataclass(frozen=True)
class UIConfig:
    enabled: bool = True
    host: str = "127.0.0.1"   # loopback only — the HUD is private to this machine
    port: int = 8765
    open_browser: bool = True  # open the HUD when J.A.R.I.V.S starts


@dataclass(frozen=True)
class ServiceConfig:
    task_name: str = "JARVIS Voice Assistant"
    restart_on_crash: bool = True
    restart_backoff_sec: float = 5.0
    restart_backoff_max_sec: float = 60.0
    single_instance: bool = True
    run_hidden: bool = True  # launch via pythonw.exe (no console window)


@dataclass(frozen=True)
class Config:
    logging: LoggingConfig
    audio: AudioConfig
    wake_word: WakeWordConfig
    safety: SafetyConfig
    execution: ExecutionConfig
    stt: STTConfig
    intent: IntentConfig
    tts: TTSConfig
    ui: UIConfig
    service: ServiceConfig


# --------------------------------------------------------------------------- #
# Environment overrides (small, explicit allow-list — see .env.example)
# --------------------------------------------------------------------------- #
# env var -> (section, key, optional caster)
_ENV_OVERRIDES: dict[str, tuple[Any, ...]] = {
    "JARVIS_LOG_LEVEL": ("logging", "level"),
    "JARVIS_LOG_FILE": ("logging", "file"),
    "JARVIS_WAKE_MODEL": ("wake_word", "model"),
    "JARVIS_WAKE_THRESHOLD": ("wake_word", "threshold", float),
    "JARVIS_INTENT_HOST": ("intent", "host"),
    "JARVIS_INTENT_MODEL": ("intent", "model"),
    "JARVIS_AUDIO_INPUT_DEVICE": ("audio", "input_device"),
}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Overlay recognised environment variables onto the raw config dict."""
    for env_var, spec in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is None:
            continue
        section, key = spec[0], spec[1]
        caster: Optional[Callable[[str], Any]] = spec[2] if len(spec) > 2 else None
        if caster is not None:
            try:
                cast_value: Any = caster(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Environment variable {env_var}={value!r} is not valid: {exc}"
                ) from exc
        else:
            cast_value = value
        raw.setdefault(section, {})
        if not isinstance(raw[section], dict):
            raise ConfigError(f"Cannot apply {env_var}: '{section}' is not a mapping")
        raw[section][key] = cast_value
    return raw


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    val = raw.get(name, {})
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ConfigError(
            f"Config section '{name}' must be a mapping, got {type(val).__name__}"
        )
    return val


def _construct(cls: type, section: dict[str, Any], name: str, *, strict: bool):
    """Build a dataclass from a section dict.

    ``strict=True`` rejects unknown keys (catches typos in actively-used
    sections); ``strict=False`` silently ignores them (forward-compat for
    stages not yet wired up).
    """
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown and strict:
        raise ConfigError(
            f"Unknown key(s) in section '{name}': {sorted(unknown)}. "
            f"Allowed: {sorted(known)}"
        )
    kwargs = {k: v for k, v in section.items() if k in known}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Invalid config for section '{name}': {exc}") from exc


def load_config(path: str | os.PathLike[str] = "config/config.yaml") -> Config:
    """Load, override, and validate configuration.

    Args:
        path: Path to the YAML config file.

    Returns:
        A fully validated :class:`Config`.

    Raises:
        ConfigError: If the file is missing, unparseable, or any value is
            invalid or out of range.
    """
    load_dotenv()  # load .env into os.environ if present; no-op otherwise

    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p.resolve()}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse {p}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a mapping of sections")

    raw = _apply_env_overrides(raw)

    cfg = Config(
        logging=_construct(LoggingConfig, _section(raw, "logging"), "logging", strict=True),
        audio=_construct(AudioConfig, _section(raw, "audio"), "audio", strict=True),
        wake_word=_construct(WakeWordConfig, _section(raw, "wake_word"), "wake_word", strict=True),
        safety=_construct(SafetyConfig, _section(raw, "safety"), "safety", strict=True),
        execution=_construct(ExecutionConfig, _section(raw, "execution"), "execution", strict=True),
        stt=_construct(STTConfig, _section(raw, "stt"), "stt", strict=True),
        intent=_construct(IntentConfig, _section(raw, "intent"), "intent", strict=True),
        tts=_construct(TTSConfig, _section(raw, "tts"), "tts", strict=True),
        ui=_construct(UIConfig, _section(raw, "ui"), "ui", strict=True),
        service=_construct(ServiceConfig, _section(raw, "service"), "service", strict=True),
    )
    _validate(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# Validation of the actively-used sections
# --------------------------------------------------------------------------- #
def _validate(cfg: Config) -> None:
    log, audio, ww = cfg.logging, cfg.audio, cfg.wake_word

    if log.level.upper() not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"logging.level '{log.level}' invalid; choose from {sorted(_VALID_LOG_LEVELS)}"
        )
    if log.format not in _VALID_LOG_FORMATS:
        raise ConfigError(
            f"logging.format '{log.format}' invalid; choose from {sorted(_VALID_LOG_FORMATS)}"
        )
    if log.max_bytes <= 0:
        raise ConfigError("logging.max_bytes must be > 0")
    if log.backup_count < 0:
        raise ConfigError("logging.backup_count must be >= 0")

    if audio.sample_rate <= 0:
        raise ConfigError("audio.sample_rate must be > 0")
    if audio.frame_length <= 0:
        raise ConfigError("audio.frame_length must be > 0")
    if audio.channels not in (1, 2):
        raise ConfigError("audio.channels must be 1 or 2")

    if not 0.0 <= ww.threshold <= 1.0:
        raise ConfigError("wake_word.threshold must be in [0.0, 1.0]")
    if ww.inference_framework not in _VALID_FRAMEWORKS:
        raise ConfigError(
            f"wake_word.inference_framework '{ww.inference_framework}' invalid; "
            f"choose from {sorted(_VALID_FRAMEWORKS)}"
        )
    if not 0.0 <= ww.vad_threshold <= 1.0:
        raise ConfigError("wake_word.vad_threshold must be in [0.0, 1.0]")
    if ww.trigger_cooldown_sec < 0:
        raise ConfigError("wake_word.trigger_cooldown_sec must be >= 0")

    if cfg.safety.confirmation_retries < 0:
        raise ConfigError("safety.confirmation_retries must be >= 0")
    if "{action}" not in cfg.safety.confirmation_prompt:
        raise ConfigError("safety.confirmation_prompt must contain the '{action}' placeholder")

    if "{query}" not in cfg.execution.search_url:
        raise ConfigError("execution.search_url must contain the '{query}' placeholder")
    if cfg.execution.read_file_max_chars <= 0:
        raise ConfigError("execution.read_file_max_chars must be > 0")
    if not isinstance(cfg.execution.apps, dict):
        raise ConfigError("execution.apps must be a mapping of name -> executable")

    if not cfg.intent.model.strip():
        raise ConfigError("intent.model must not be empty")
    if not cfg.intent.host.strip():
        raise ConfigError("intent.host must not be empty")
    if cfg.intent.timeout_sec <= 0:
        raise ConfigError("intent.timeout_sec must be > 0")
    if not 0.0 <= cfg.intent.temperature <= 2.0:
        raise ConfigError("intent.temperature must be in [0.0, 2.0]")
    if not cfg.intent.keep_alive.strip():
        raise ConfigError("intent.keep_alive must not be empty")
    if cfg.intent.max_tokens < 32:
        raise ConfigError("intent.max_tokens must be >= 32")

    stt = cfg.stt
    if not stt.model_size.strip():
        raise ConfigError("stt.model_size must not be empty")
    if stt.beam_size < 1:
        raise ConfigError("stt.beam_size must be >= 1")
    if stt.speech_multiplier < 1.0:
        raise ConfigError("stt.speech_multiplier must be >= 1.0")
    if stt.min_rms_threshold < 0:
        raise ConfigError("stt.min_rms_threshold must be >= 0")
    for name in ("calibration_sec", "silence_duration_sec", "max_duration_sec",
                 "onset_timeout_sec"):
        if getattr(stt, name) <= 0:
            raise ConfigError(f"stt.{name} must be > 0")
    if stt.preroll_sec < 0:
        raise ConfigError("stt.preroll_sec must be >= 0")
    if not stt.models_dir.strip():
        raise ConfigError("stt.models_dir must not be empty")

    tts = cfg.tts
    if not tts.engine.strip():
        raise ConfigError("tts.engine must not be empty")
    if not tts.voice.strip():
        raise ConfigError("tts.voice must not be empty")
    if tts.speaking_rate <= 0:
        raise ConfigError("tts.speaking_rate must be > 0")
    if tts.speaker_id < 0:
        raise ConfigError("tts.speaker_id must be >= 0")
    if tts.volume < 0:
        raise ConfigError("tts.volume must be >= 0")

    ui = cfg.ui
    if not ui.host.strip():
        raise ConfigError("ui.host must not be empty")
    if not 1 <= ui.port <= 65535:
        raise ConfigError("ui.port must be in [1, 65535]")

    svc = cfg.service
    if not svc.task_name.strip():
        raise ConfigError("service.task_name must not be empty")
    if svc.restart_backoff_sec <= 0:
        raise ConfigError("service.restart_backoff_sec must be > 0")
    if svc.restart_backoff_max_sec < svc.restart_backoff_sec:
        raise ConfigError("service.restart_backoff_max_sec must be >= restart_backoff_sec")

    # If a custom wake-word model file is configured (looks like a path), it
    # must exist. Bare bundled names (e.g. "hey_jarvis") are resolved/downloaded
    # by the listener, so we don't check those here.
    if _looks_like_path(ww.model):
        model_path = Path(ww.model)
        if not model_path.is_file():
            raise ConfigError(f"wake_word.model file not found: {model_path.resolve()}")


def _looks_like_path(model: str) -> bool:
    """True if ``model`` refers to a file rather than a bundled model name."""
    return (
        os.sep in model
        or (os.altsep is not None and os.altsep in model)
        or model.endswith((".onnx", ".tflite"))
    )
