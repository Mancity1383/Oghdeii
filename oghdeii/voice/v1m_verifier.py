"""v1m System-One cloud intent verification for Oghdeii voice commands.

The Windows ``System.Speech`` helper recognizes phrases from a constrained
command grammar and sends the selected phrase plus N-best alternatives;
V1M decides whether the result is an intentional command and which action it
requests. The grammar is a first-pass acoustic constraint, not V1M's classifier.
This module sits between the recogniser's stdout reader and the action
dispatcher as a *guardrail + intent resolver*:

* ``is_valid_command`` (``Noul``)  — was the utterance an intentional
  Oghdeii command, or ambient chatter? Probabilities below the
  configured floor (default 0.55) never execute.
* ``action`` (``Choice``)          — which supported action the phrase requests.
  The ``ignore`` choice vetoes execution; V1M's action determines dispatch.
* ``execution_risk`` (``Score``)   — accidental-trigger risk, which rises
  while keyboard typing is active.

Latency contract
----------------
``verify()`` is called from the recogniser's stdout reader thread and

1. never blocks that thread longer than ``v1m_wait_timeout_ms`` (default 3000 ms),
2. never performs network I/O on that thread — the HTTP call runs on an
   internal worker pool — and
3. never raises: when cloud verification is enabled, unavailable or malformed
   V1M decisions fail closed; when the user disables V1M, constrained local
   command matches continue to work offline.

A slow answer returns immediately and is written into the phrase cache when it
finally lands, so the *next* identical utterance can be answered from cache.

Fallback ladder (all commands are held when enabled V1M cannot answer)::

    disabled -> cache hit -> SDK/key missing -> circuit open -> cloud (bounded
    wait) -> blocked (unless V1M is explicitly disabled)

Credentials come from ``V1M_API_KEY`` in the environment first, then from the
local app configuration (``config.json``); ``TYPESAFE_API_KEY`` (the SDK's own
variable) is honoured as a secondary environment source.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from types import ModuleType
from typing import Any

__all__ = [
    "ACTIONS",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "CloudVerdict",
    "SDK_IMPORT_ERROR",
    "TypingStateProbe",
    "VerificationResult",
    "V1MVoiceVerifier",
    "normalize_phrase",
    "sdk_available",
    "sdk_base_url",
]

# ---------------------------------------------------------------------------
# Constants / schema vocabulary
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINT = "https://v1m.ir/v1"
DEFAULT_MODEL = "v1m-latest"

DEFAULT_WAIT_TIMEOUT_MS = 3000     # bounded wait for a typical cloud round trip
DEFAULT_REQUEST_TIMEOUT_MS = 5000 # HTTP budget for the (background) call
DEFAULT_MIN_PROBABILITY = 0.55     # P(is_valid_command) floor
DEFAULT_MAX_EXECUTION_RISK = 4.0   # rubric is 0..4; 4.0 == gate effectively off
DEFAULT_CACHE_TTL_S = 600.0
DEFAULT_FAILURE_COOLDOWN_S = 30.0

CACHE_MAX_ENTRIES = 256
FAILURE_THRESHOLD = 3              # consecutive cloud failures -> open circuit

# Question names used on the wire; the response echoes them back.
Q_VALID = "is_valid_command"
Q_ACTION = "action"
Q_RISK = "execution_risk"

# V1M action vocabulary. Its command label is passed through the user's live
# voice_actions mapping.
ACTIONS = (
    "copy",
    "paste",
    "undo",
    "redo",
    "select_all",
    "screenshot",
    "lock",
    "show_desktop",
    "close_window",
    "open_calculator",
    "open_notepad",
    "open_browser",
    "open_terminal",
    "play_pause",
    "next",
    "previous",
    "mute",
    "volume_up",
    "volume_down",
    "ignore",
)

# Local constrained-phrase command -> schema action. An empty command means
# no supported local grammar phrase was mapped and V1M must classify the text.
COMMAND_TO_ACTION = {
    "copy": "copy",
    "paste": "paste",
    "undo": "undo",
    "redo": "redo",
    "select_all": "select_all",
    "screenshot": "screenshot",
    "lock": "lock",
    "show_desktop": "show_desktop",
    "close_window": "close_window",
    "open_calculator": "open_calculator",
    "open_notepad": "open_notepad",
    "open_browser": "open_browser",
    "open_terminal": "open_terminal",
    "play": "play_pause",
    "pause": "play_pause",
    "next": "next",
    "previous": "previous",
    "mute": "mute",
    "volume_up": "volume_up",
    "volume_down": "volume_down",
}

# Schema action -> dispatcher command understood by voice_actions.
ACTION_TO_COMMAND = {
    "copy": "copy",
    "paste": "paste",
    "undo": "undo",
    "redo": "redo",
    "select_all": "select_all",
    "screenshot": "screenshot",
    "lock": "lock",
    "show_desktop": "show_desktop",
    "close_window": "close_window",
    "open_calculator": "open_calculator",
    "open_notepad": "open_notepad",
    "open_browser": "open_browser",
    "open_terminal": "open_terminal",
    "play_pause": "play",
    "next": "next",
    "previous": "previous",
    "mute": "mute",
    "volume_up": "volume_up",
    "volume_down": "volume_down",
}

NOUL_INSTRUCTIONS = (
    "Does this speech transcript contain an intentional request for Oghdeii "
    "to run one of its supported voice commands? Windows recognized it with a "
    "constrained command grammar. When recognized_command is present, that is "
    "strong evidence that a supported phrase matched; V1M still judges intent, "
    "but should not treat a clear matched phrase as invalid just because it is "
    "short. If "
    "transcript_candidates are present, they are alternate ASR hypotheses "
    "with their own confidence scores; use them to correct a likely "
    "mishearing, while still judging whether the intended phrase is a command."
)
NOUL_CRITERIA = {
    "true": "The phrase clearly asks Oghdeii to run a supported action. "
            "A recognized_command value means Windows matched the phrase to "
            "the constrained supported-command grammar; a clear match such as "
            "recognized_command=copy with phrase 'laptop copy' is a valid "
            "command when the required wake phrase is present. "
            "Accept natural wording, paraphrases, and likely speech-to-text errors. "
            "If wake_word_required is true, accept a plausible intended wake call "
            "before the action, including a garbled or misrecognized wake name "
            "(for example, 'mamad desktop' may mean 'Laptop, show desktop'); "
            "do not require an exact fixed phrase.",
    "false": "Ambient conversation, TV, background speech, or a phrase that "
             "does not request a supported action. If wake_word_required is true, "
             "there is no plausible wake call or indication the request is "
             "addressed to Oghdeii.",
}

ACTION_INSTRUCTIONS = (
    "Which supported Oghdeii voice command does the phrase request? "
    "Consider transcript_candidates as alternate speech-recognition hypotheses "
    "when the primary phrase is unclear or sounds unrelated to an action. "
    "Use the candidate confidences as evidence, not as a rigid rule. "
    "Use a non-empty recognized_command as a strong hint because Windows "
    "matched a supported constrained-grammar phrase. If it is empty or null, "
    "infer the action from the recognized phrase and its candidates. Otherwise "
    "use it as a hint and "
    "correct it when the spoken phrase clearly requests a different listed "
    "command. Choose ignore "
    "only when the phrase does not request any listed command."
)
ACTION_CRITERIA = {
    "copy": "Copy the current selection ('Laptop Copy', 'copy that').",
    "paste": "Paste the clipboard ('Laptop Paste', 'paste that').",
    "undo": "Undo the last action ('Laptop Undo', 'undo that').",
    "redo": "Redo the last action ('Laptop Redo').",
    "select_all": "Select all content ('Laptop Select All').",
    "screenshot": "Take a screenshot ('Laptop Screenshot', 'take a screenshot').",
    "lock": "Lock the computer ('Laptop Lock', 'lock screen').",
    "show_desktop": "Minimize or hide open windows to show the desktop ('Laptop Show Desktop', 'Laptop Desktop').",
    "close_window": "Close the active window ('Laptop Close Window', 'close app').",
    "open_calculator": "Open the calculator app ('Laptop Open Calculator', 'Laptop Calculator').",
    "open_notepad": "Open Notepad ('Laptop Open Notepad', 'Laptop Notepad').",
    "open_browser": "Open the default web browser ('Laptop Open Browser', 'Laptop Browser').",
    "open_terminal": "Open a terminal ('Laptop Open Terminal', 'Laptop Terminal').",
    "play_pause": "Toggle media playback ('Laptop Play', 'pause the music').",
    "next": "Skip to the next track ('Laptop Next', 'next track').",
    "previous": "Go back to the previous track ('Laptop Previous').",
    "mute": "Toggle microphone/speaker mute ('Laptop Mute').",
    "volume_up": "Raise the system volume ('Laptop Volume Up').",
    "volume_down": "Lower the system volume ('Laptop Volume Down').",
    "ignore": "The phrase is ambient or does not request any supported voice command; do nothing.",
}

RISK_INSTRUCTIONS = (
    "Risk that acting on this phrase right now would be an accidental or "
    "damaging trigger. Raise the score while keyboard typing is active, or "
    "when the phrase is ambiguous or overheard rather than addressed to the engine."
)
# Ordered rubric: index == score level (0..4).
RISK_RUBRIC = [
    "Safe: clearly addressed to Oghdeii while the user is idle.",
    "Low: plausibly addressed to Oghdeii; the user is mostly idle.",
    "Moderate: the phrase is short or ambiguous, or the user is actively typing.",
    "High: likely ambient speech or typing is active; a misfire would interrupt work.",
    "Severe: typing is active and the action could destroy work "
    "(paste over a document, lock mid-edit).",
]

_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)


# ---------------------------------------------------------------------------
# Optional SDK import — the app must import and run without it
# ---------------------------------------------------------------------------

# ``typesafe_sdk`` is an optional dependency: the slot stays Optional so the
# "SDK missing" branch type-checks like the runtime fallback it is.
_typesafe_sdk: ModuleType | None = None
try:
    import typesafe_sdk as _typesafe_sdk
except Exception as _sdk_exc:  # pragma: no cover - depends on the environment
    _typesafe_sdk = None
    SDK_IMPORT_ERROR: str | None = str(_sdk_exc) or type(_sdk_exc).__name__
else:
    SDK_IMPORT_ERROR = None


def sdk_available() -> bool:
    """True when ``typesafe_sdk`` imported cleanly (never raises)."""
    return _typesafe_sdk is not None and hasattr(_typesafe_sdk, "TypeSafeClient")


def sdk_base_url(endpoint: str | None) -> str:
    """Turn a configured endpoint into the SDK's ``base_url``.

    The SDK appends ``/v1/systemone`` itself (``SYSTEM_ONE_PATH``), so a
    configured endpoint of ``https://v1m.ir/v1`` must be shortened to
    ``https://v1m.ir`` or the request would go to ``/v1/v1/systemone``.
    """
    url = str(endpoint or DEFAULT_ENDPOINT).strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url or DEFAULT_ENDPOINT


def normalize_phrase(text: str) -> str:
    """Case/punctuation-insensitive form of a recognised phrase for cache keys."""
    lowered = str(text or "").strip().lower()
    without_punct = _NON_WORD_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", without_punct).strip()


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CloudVerdict:
    """The raw System-One answers, before any local policy is applied."""

    probability: float
    action: str
    action_confidence: float | None = None
    action_probabilities: dict[str, float] | None = None
    execution_risk: float | None = None
    risk_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """A complete, always-usable gate decision for one recognised phrase."""

    phrase: str
    command: str
    source: str            # "cache" | "cloud" | "offline" | "disabled"
    allow: bool
    reason: str = ""
    probability: float | None = None      # None == "not evaluated" (offline/disabled)
    canonical_action: str | None = None
    action_confidence: float | None = None
    action_probability: float | None = None
    execution_risk: float | None = None
    risk_confidence: float | None = None
    latency_ms: float = 0.0

    @property
    def is_cloud(self) -> bool:
        """True when the decision rests on a v1m verdict (live or cached)."""
        return self.source in ("cloud", "cache")

    @property
    def is_valid_command(self) -> bool | None:
        """Boolean form of the Noul answer, or ``None`` when not evaluated."""
        return None if self.probability is None else bool(self.probability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "command": self.command,
            "source": self.source,
            "allow": self.allow,
            "reason": self.reason,
            "probability": self.probability,
            "action": self.canonical_action,
            "action_confidence": self.action_confidence,
            "action_probability": self.action_probability,
            "execution_risk": self.execution_risk,
            "latency_ms": round(self.latency_ms, 1),
        }

    def __str__(self) -> str:  # pragma: no cover - diagnostics helper
        prob = "n/a" if self.probability is None else f"{self.probability:.2f}"
        risk = "n/a" if self.execution_risk is None else f"{self.execution_risk:.1f}"
        verdict = "ALLOW" if self.allow else "BLOCK"
        return (
            f"<{verdict} {self.source} cmd={self.command!r} p={prob} "
            f"action={self.canonical_action or '-'} risk={risk} ({self.reason})>"
        )


# ---------------------------------------------------------------------------
# Typing-state probe (context for execution_risk)
# ---------------------------------------------------------------------------

class TypingStateProbe:
    """Best-effort "is the user typing right now?" signal.

    ``TapDetector`` releases its global keyboard hook when voice mode starts,
    so the probe owns a small lazily-started ``pynput`` listener of its own.
    Any failure (no pynput, headless session, hook refused) disables the probe
    permanently instead of raising — risk context is advisory.
    """

    def __init__(self, config_manager, listener_factory: Callable[..., Any] | None = None):
        self.config = config_manager
        self._listener_factory = listener_factory
        self._last_key_time = float("-inf")
        self._listener = None
        self._lock = threading.Lock()
        self._unavailable = False
        self._stopped = False

    def __call__(self) -> bool:
        return self.is_active()

    @property
    def cooldown_s(self) -> float:
        try:
            ms = float(self.config.get("typing_cooldown_ms", 400))
        except (TypeError, ValueError):
            ms = 400.0
        return max(0.0, ms) / 1000.0

    def start(self) -> bool:
        """Prime the keyboard hook early so the first command has context."""
        if self._unavailable or self._stopped:
            return False
        self._ensure_listener()
        return self._listener is not None

    def is_active(self) -> bool:
        if not self.start():
            return False
        return (time.monotonic() - self._last_key_time) < self.cooldown_s

    def _ensure_listener(self) -> None:
        with self._lock:
            if self._unavailable or self._stopped or self._listener is not None:
                return
            factory = self._listener_factory
            if factory is None:
                try:
                    from pynput import keyboard

                    factory = keyboard.Listener
                except Exception:
                    self._unavailable = True
                    return
            try:
                listener = factory(on_press=self._on_press)
                listener.daemon = True
                listener.start()
                self._listener = listener
            except Exception as exc:
                self._unavailable = True
                print(f"[V1M] Typing probe unavailable, risk context degrades: {exc}")

    def _on_press(self, _key) -> None:
        self._last_key_time = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# The verifier
# ---------------------------------------------------------------------------

class V1MVoiceVerifier:
    """Non-blocking guardrail + intent resolver for recognised voice commands.

    Typical wiring (see ``main.py`` for the shipped patch)::

        verifier = V1MVoiceVerifier(config)
        decision = verifier.verify(cmd, spoken_text, confidence)
        if decision.allow:
            executor.trigger(resolve_voice_action(config, decision.command))

    ``decision.command`` is the command the *dispatcher* should resolve — it
    equals the locally recognised command except when the model's ``action``
    disagrees with a command it can express, in which case the model wins.
    """

    def __init__(
        self,
        config_manager,
        typing_probe: Callable[[], bool] | None = None,
        client_factory: Callable[[str, str, float], Any] | None = None,
        max_workers: int = 2,
    ):
        self.config = config_manager
        self._typing_probe = (
            typing_probe if typing_probe is not None else TypingStateProbe(config_manager)
        )
        self._client_factory = client_factory or self._default_client_factory
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, tuple[float, CloudVerdict]] = OrderedDict()
        # Network lives here — never on the caller's (recogniser reader) thread.
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="v1m-verify"
        )
        # Fully asynchronous API (verify_async) uses a separate single lane so it
        # can never starve the bounded-wait path.
        self._async_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="v1m-gate"
        )
        self._client = None
        self._client_signature: tuple | None = None
        self._last_exchange_json: str | None = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self.last_error: str | None = None
        self.stats = {
            "cloud": 0, "cache": 0, "offline": 0, "disabled": 0,
            "blocked": 0, "timeouts": 0,
        }

    # ------------------------------------------------------------------
    # Configuration (live: every call reads the shared config)
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        try:
            return bool(self.config.get("enable_v1m_verification", False))
        except Exception:
            return False

    @property
    def endpoint(self) -> str:
        return str(self.config.get("v1m_endpoint") or DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT

    @property
    def model(self) -> str:
        return str(self.config.get("v1m_model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    def _wait_timeout_s(self) -> float:
        return max(0.01, self._ms("v1m_wait_timeout_ms", DEFAULT_WAIT_TIMEOUT_MS, 10, 10000) / 1000.0)

    def _request_timeout_s(self) -> float:
        return max(0.05, self._ms("v1m_request_timeout_ms", DEFAULT_REQUEST_TIMEOUT_MS, 100, 60000) / 1000.0)

    def _min_probability(self) -> float:
        return self._num("v1m_min_probability", DEFAULT_MIN_PROBABILITY, 0.0, 1.0)

    def _max_risk(self) -> float:
        return self._num("v1m_max_execution_risk", DEFAULT_MAX_EXECUTION_RISK, 0.0, 4.0)

    def _cache_ttl_s(self) -> float:
        return self._num("v1m_cache_ttl_s", DEFAULT_CACHE_TTL_S, 0.0, 86400.0)

    def _failure_cooldown_s(self) -> float:
        return self._num("v1m_failure_cooldown_s", DEFAULT_FAILURE_COOLDOWN_S, 0.0, 3600.0)

    def _ms(self, key: str, default: float, low: float, high: float) -> float:
        try:
            value = float(self.config.get(key, default))
        except (TypeError, ValueError):
            value = float(default)
        if value != value:  # NaN
            value = float(default)
        return min(high, max(low, value))

    def _num(self, key: str, default: float, low: float, high: float) -> float:
        return self._ms(key, default, low, high)

    def api_key(self) -> str:
        """Resolve the API key: env first (V1M_API_KEY), then local config."""
        for env_name in ("V1M_API_KEY", "TYPESAFE_API_KEY"):
            value = (os.environ.get(env_name) or "").strip()
            if value:
                return value
        try:
            return str(self.config.get("v1m_api_key") or "").strip()
        except Exception:
            return ""

    def key_source(self) -> str:
        if (os.environ.get("V1M_API_KEY") or "").strip():
            return "environment (V1M_API_KEY)"
        if (os.environ.get("TYPESAFE_API_KEY") or "").strip():
            return "environment (TYPESAFE_API_KEY)"
        if str(self.config.get("v1m_api_key") or "").strip():
            return "app configuration"
        return "missing"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Prime background resources (typing probe) when the feature is on."""
        if not self.enabled or not sdk_available():
            return False
        starter = getattr(self._typing_probe, "start", None)
        try:
            if callable(starter):
                return bool(starter())
            self._typing_probe()  # plain callable probes need no priming
        except Exception:
            return False
        return True

    def verify(
        self,
        command: str,
        phrase: str = "",
        confidence: float = 0.0,
        alternates: list[dict[str, Any]] | None = None,
    ) -> VerificationResult:
        """Gate one recognised phrase. Bounded, exception-free, thread-safe.

        Returns within ``v1m_wait_timeout_ms`` (default 3000 ms) even when the
        cloud is down or slow; never raises.
        """
        started = time.monotonic()
        command = str(command or "").strip().lower()
        phrase = str(phrase or "").strip()[:200]
        try:
            alternates = self._normalize_alternates(alternates, phrase, confidence)
            return self._verify_locked(command, phrase, confidence, alternates, started)
        except Exception as exc:  # absolute last resort; free text fails closed
            self.last_error = f"internal verifier error: {exc}"
            return self._offline_result(command, phrase, started, f"internal error: {exc}")

    def verify_async(
        self,
        command: str,
        phrase: str = "",
        confidence: float = 0.0,
        alternates: list[dict[str, Any]] | None = None,
    ) -> Future:
        """Fully non-blocking variant: resolves to a :class:`VerificationResult`."""
        future: Future = Future()
        try:
            self._async_executor.submit(
                lambda: future.set_result(
                    self.verify(command, phrase, confidence, alternates)
                )
            )
        except RuntimeError:  # pool already shut down
            future.set_result(
                self._offline_result(command, phrase, time.monotonic(), "verifier shut down")
            )
        return future

    def test_connection(self, timeout: float | None = None) -> tuple[bool, str]:
        """Probe credentials + endpoint. Blocking — call it from a worker thread.

        Returns ``(ok, human_readable_detail)`` and never raises.
        """
        key = self.api_key()
        if not key:
            return False, "No API key. Set V1M_API_KEY or paste one below."
        if not sdk_available():
            return False, f"typesafe-sdk is not installed ({SDK_IMPORT_ERROR})"
        timeout_s = float(timeout) if timeout else max(5.0, self._request_timeout_s())
        started = time.monotonic()
        client = None
        try:
            client = self._client_factory(key, sdk_base_url(self.endpoint), timeout_s)
            listing = client.models.list(timeout=timeout_s)
            names = [str(getattr(model, "name", model)) for model in getattr(listing, "models", [])]
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if self.model in names:
                detail = f"model '{self.model}' available"
            elif names:
                detail = f"available models: {', '.join(names[:6])}"
            else:
                detail = "endpoint reachable"
            return True, f"Connected in {elapsed_ms:.0f} ms — {detail}"
        except Exception as exc:
            return False, self._friendly_error(exc)
        finally:
            if client is not None and client is not self._client:
                closer = getattr(client, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass

    def clear_cache(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        return count

    def describe(self) -> dict[str, Any]:
        """Snapshot for tooltips/logs — contains no secrets."""
        with self._lock:
            circuit = max(0.0, self._circuit_open_until - time.monotonic())
            cache_size = len(self._cache)
            stats = dict(self.stats)
        return {
            "enabled": self.enabled,
            "sdk_installed": sdk_available(),
            "key_source": self.key_source(),
            "endpoint": self.endpoint,
            "model": self.model,
            "wait_timeout_ms": self._wait_timeout_s() * 1000.0,
            "min_probability": self._min_probability(),
            "max_execution_risk": self._max_risk(),
            "cache_entries": cache_size,
            "circuit_open_for_s": circuit,
            "last_error": self.last_error,
            "stats": stats,
        }

    def last_exchange_json(self) -> str | None:
        """Return the latest request/response details without credentials."""
        with self._lock:
            return self._last_exchange_json

    def shutdown(self) -> None:
        """Release the worker pools, cached client and keyboard hook."""
        for pool in (self._executor, self._async_executor):
            try:
                # Let submitted verification calls finish before closing their
                # shared HTTP client; queued verify_async futures must resolve.
                pool.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
        with self._lock:
            client, self._client = self._client, None
            self._client_signature = None
        if client is not None:
            closer = getattr(client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        stopper = getattr(self._typing_probe, "stop", None)
        if callable(stopper):
            try:
                stopper()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Decision pipeline
    # ------------------------------------------------------------------

    def _verify_locked(
        self,
        command: str,
        phrase: str,
        confidence: float,
        alternates: list[dict[str, Any]],
        started: float,
    ) -> VerificationResult:
        if not self.enabled:
            self._count("disabled")
            return self._offline_result(
                command, phrase, started,
                "v1m cloud verification is disabled", source="disabled",
            )

        typing_active = self._typing_active()
        cache_key = self._cache_key(
            phrase, command, confidence, typing_active, alternates
        )

        verdict = self._cache_get(cache_key)
        if verdict is not None:
            self._count("cache")
            return self._cloud_result(verdict, command, phrase, started, source="cache")

        unavailable = self._cloud_unavailable_reason()
        if unavailable is not None:
            return self._offline_result(command, phrase, started, unavailable)

        try:
            future = self._executor.submit(
                self._fetch_verdict, command, phrase, confidence, typing_active, alternates
            )
        except RuntimeError:  # pool shut down
            return self._offline_result(command, phrase, started, "verifier is shut down")

        wait_s = self._wait_timeout_s()
        try:
            verdict = future.result(timeout=wait_s)
        except FutureTimeoutError:
            # Do not waste the round trip: the late answer still seeds the cache.
            future.add_done_callback(self._late_cache(cache_key))
            self._count("timeouts")
            self._register_failure(
                TimeoutError(f"no answer within {wait_s * 1000:.0f} ms"), "timeout"
            )
            return self._offline_result(
                command, phrase, started,
                f"cloud verification exceeded {wait_s * 1000:.0f} ms",
            )
        except CancelledError:
            return self._offline_result(command, phrase, started, "verification cancelled")
        except Exception as exc:
            self._register_failure(exc, "cloud call failed")
            return self._offline_result(
                command, phrase, started, f"cloud unavailable: {exc}"
            )

        self._register_success()
        self._cache_put(cache_key, verdict)
        self._count("cloud")
        return self._cloud_result(verdict, command, phrase, started, source="cloud")

    def _typing_active(self) -> bool:
        try:
            return bool(self._typing_probe())
        except Exception:
            return False  # advisory context only: never fail over it

    @staticmethod
    def _normalize_alternates(
        alternates: list[dict[str, Any]] | None, phrase: str, confidence: float
    ) -> list[dict[str, Any]]:
        normalized = []
        seen = set()
        source = list(alternates or [])[:5]
        if phrase:
            source.insert(0, {"text": phrase, "confidence": confidence})
        for item in source:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()[:200]
            key = normalize_phrase(text)
            if not key or key in seen:
                continue
            try:
                probability = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                probability = 0.0
            seen.add(key)
            normalized.append({"text": text, "confidence": round(probability, 3)})
        return normalized[:5]

    def _cache_key(
        self,
        phrase: str,
        command: str,
        confidence: float,
        typing_active: bool,
        alternates: list[dict[str, Any]] | None = None,
    ) -> str:
        text = normalize_phrase(phrase) or normalize_phrase(command)
        # Every value below is sent to V1M as classification context. Omitting
        # confidence scores or wake-word policy could reuse a verdict for a
        # materially different model input.
        hypotheses = [
            {
                "text": normalize_phrase(item.get("text", "")),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
            }
            for item in (alternates or [])[:5]
        ]
        try:
            confidence_reference = round(
                float(self.config.get("voice_confidence_threshold", 0.55)), 3
            )
        except (TypeError, ValueError):
            confidence_reference = 0.55
        payload = {
            "phrase": text,
            "command": command,
            "confidence": round(float(confidence), 3),
            "confidence_reference": confidence_reference,
            "hypotheses": hypotheses,
            "typing_active": bool(typing_active),
            "wake_word_required": bool(
                self.config.get("voice_require_wake_word", True)
            ),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def _cache_get(self, key: str) -> CloudVerdict | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expiry, verdict = entry
            if expiry < time.monotonic():
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return verdict

    def _cache_put(self, key: str, verdict: CloudVerdict) -> None:
        ttl = self._cache_ttl_s()
        if ttl <= 0:
            return
        with self._lock:
            self._cache[key] = (time.monotonic() + ttl, verdict)
            self._cache.move_to_end(key)
            while len(self._cache) > CACHE_MAX_ENTRIES:
                self._cache.popitem(last=False)

    def _late_cache(self, key: str):
        """Done-callback: a slow answer that beat the wait still warms the cache."""

        def _store(future: Future) -> None:
            try:
                verdict = future.result()
            except Exception:
                return  # failures are never cached; the next call retries
            self._cache_put(key, verdict)

        return _store

    def _cloud_unavailable_reason(self) -> str | None:
        if not sdk_available():
            return "offline mode: typesafe-sdk is not installed"
        if not self.api_key():
            return "offline mode: no V1M_API_KEY configured"
        with self._lock:
            if time.monotonic() < self._circuit_open_until:
                return "offline mode: cloud paused after repeated failures"
        return None

    def _register_failure(self, exc: Exception, what: str) -> None:
        self.last_error = f"{what}: {exc}"[:300]
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= FAILURE_THRESHOLD:
                cooldown = self._failure_cooldown_s()
                self._circuit_open_until = time.monotonic() + cooldown
                self._consecutive_failures = 0
                print(
                    f"[V1M] Cloud verification paused for {cooldown:.0f}s after "
                    f"{FAILURE_THRESHOLD} failures: {self.last_error}"
                )

    def _register_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + 1

    # ------------------------------------------------------------------
    # Cloud I/O (runs on the worker pool, never on the caller's thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_client_factory(api_key: str, base_url: str, timeout_s: float):
        if not sdk_available():
            raise RuntimeError("typesafe-sdk is not installed")
        # Imported here rather than reusing the module-level slot: that one has
        # to stay Optional for the missing-SDK case, which would cost us the
        # real client/retry signatures mypy checks this call against.
        import typesafe_sdk as sdk
        return sdk.TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            # Single-pass, strict: a guardrail must never add hidden retries or
            # backoff on top of the wait budget.
            retry=sdk.RetryPolicy(max_retries=0, timeout=timeout_s),
        )

    def _get_client(self):
        key = self.api_key()
        if not key:
            raise RuntimeError("no V1M_API_KEY configured")
        signature = (key, sdk_base_url(self.endpoint), round(self._request_timeout_s(), 3))
        with self._lock:
            if self._client is not None and self._client_signature == signature:
                return self._client
            stale, self._client = self._client, None
            if stale is not None:
                closer = getattr(stale, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
            self._client = self._client_factory(*signature)
            self._client_signature = signature
            return self._client

    @staticmethod
    def _build_questions() -> dict[str, Any]:
        """The single-pass System-One schema: one Noul, one Choice, one Score."""
        if not sdk_available():
            raise RuntimeError("typesafe-sdk is not installed")
        # See _default_client_factory: the local import keeps the real question
        # signatures in view, and NoulCriteria is built explicitly so the two
        # yes/no keys are checked against the SDK's closed TypedDict instead of
        # being smuggled through dict(), which no longer type-checks.
        import typesafe_sdk as sdk
        return {
            Q_VALID: sdk.Noul(
                instructions=NOUL_INSTRUCTIONS,
                criteria=sdk.NoulCriteria(
                    true=NOUL_CRITERIA["true"],
                    false=NOUL_CRITERIA["false"],
                ),
            ),
            Q_ACTION: sdk.Choice(
                instructions=ACTION_INSTRUCTIONS,
                criteria=dict(ACTION_CRITERIA),
            ),
            Q_RISK: sdk.Score(
                instructions=RISK_INSTRUCTIONS,
                criteria=list(RISK_RUBRIC),
            ),
        }

    def _build_state(
        self,
        command: str,
        phrase: str,
        confidence: float,
        typing_active: bool,
        alternates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            conf = round(float(confidence), 3)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            confidence_reference = round(
                float(self.config.get("voice_confidence_threshold", 0.55)), 3
            )
        except (TypeError, ValueError):
            confidence_reference = 0.55
        return {
            # The model grades intent, so it needs the recogniser's own read.
            "phrase": phrase or command,
            "recognized_command": command or None,
            "recognizer_confidence": conf,
            "recognizer_confidence_reference": confidence_reference,
            "transcript_candidates": list(alternates or [])[:5],
            "wake_word_required": bool(self.config.get("voice_require_wake_word", True)),
            # Drives execution_risk: typing makes an accidental hotkey costly.
            "typing_active": bool(typing_active),
        }

    def _fetch_verdict(
        self,
        command: str,
        phrase: str,
        confidence: float,
        typing_active: bool,
        alternates: list[dict[str, Any]] | None = None,
    ) -> CloudVerdict:
        client = self._get_client()
        state = self._build_state(command, phrase, confidence, typing_active, alternates)
        questions = self._build_questions()
        model = self.model
        wire_questions = {
            name: question.model_dump(mode="json", exclude_none=True)
            for name, question in questions.items()
        }
        request = {
            "method": "POST",
            "url": f"{sdk_base_url(self.endpoint).rstrip('/')}/v1/systemone",
            "body": {"state": state, "model": model, "questions": wire_questions},
        }

        try:
            response = client.system_one(state=state, questions=questions, model=model)
        except Exception as exc:
            self._save_exchange(request, {"error": self._friendly_error(exc)})
            raise

        try:
            verdict = self._parse_response(response)
        except Exception as exc:
            self._save_exchange(
                request,
                self._response_for_inspection(response),
                parse_error=f"{type(exc).__name__}: {exc}",
            )
            raise

        self._save_exchange(
            request,
            self._response_for_inspection(response),
            parsed_verdict={
                "is_valid_command_probability": verdict.probability,
                "action": verdict.action,
                "action_confidence": verdict.action_confidence,
                "action_probabilities": verdict.action_probabilities,
                "execution_risk": verdict.execution_risk,
                "risk_confidence": verdict.risk_confidence,
            },
        )
        return verdict

    @staticmethod
    def _response_for_inspection(response: Any) -> Any:
        dumper = getattr(response, "model_dump", None)
        if callable(dumper):
            try:
                return dumper(mode="json", exclude_none=True)
            except Exception:
                pass
        return repr(response)

    def _save_exchange(
        self,
        request: dict[str, Any],
        response: Any,
        *,
        parsed_verdict: dict[str, Any] | None = None,
        parse_error: str | None = None,
    ) -> None:
        """Keep the latest wire-shaped exchange for the local details dialog."""
        exchange: dict[str, Any] = {"request": request, "response": response}
        if parsed_verdict is not None:
            exchange["parsed_verdict"] = parsed_verdict
        if parse_error:
            exchange["parse_error"] = parse_error
        try:
            rendered = json.dumps(exchange, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            rendered = json.dumps({"diagnostic_error": str(exc)}, ensure_ascii=False)
        with self._lock:
            self._last_exchange_json = rendered

    @staticmethod
    def _parse_response(response: Any) -> CloudVerdict:
        """Extract the three answers; a malformed body is a cloud failure."""
        answers = getattr(response, "answers", None)
        if not isinstance(answers, dict):
            raise ValueError("v1m response carried no answers")

        noul = answers.get(Q_VALID)
        raw_probability = getattr(noul, "noul", None)
        if raw_probability is None and isinstance(noul, dict):
            raw_probability = noul.get("noul")
        if raw_probability is None:
            raise ValueError("v1m response is missing is_valid_command")
        probability = float(raw_probability)
        if probability != probability:  # NaN
            raise ValueError("v1m returned a non-numeric is_valid_command")
        probability = min(1.0, max(0.0, probability))

        choice = answers.get(Q_ACTION)
        raw_action = getattr(choice, "choice", None)
        if raw_action is None and isinstance(choice, dict):
            raw_action = choice.get("choice")
        if raw_action is None:
            raise ValueError("v1m response is missing action")
        action = str(raw_action).strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"v1m returned an unknown action {action!r}")

        raw_action_confidence = getattr(choice, "confidence", None)
        raw_probabilities = getattr(choice, "probabilities", None)
        if isinstance(choice, dict):
            if raw_action_confidence is None:
                raw_action_confidence = choice.get("confidence")
            if raw_probabilities is None:
                raw_probabilities = choice.get("probabilities")
        action_confidence = None
        if raw_action_confidence is not None:
            action_confidence = min(1.0, max(0.0, float(raw_action_confidence)))
        action_probabilities = None
        if isinstance(raw_probabilities, dict):
            action_probabilities = {}
            for name, value in raw_probabilities.items():
                normalized_name = str(name).strip().lower()
                if normalized_name not in ACTIONS:
                    continue
                try:
                    action_probabilities[normalized_name] = min(
                        1.0, max(0.0, float(value))
                    )
                except (TypeError, ValueError):
                    continue

        risk = risk_confidence = None
        score = answers.get(Q_RISK)
        if score is not None:
            raw_risk = getattr(score, "score", None)
            if raw_risk is None and isinstance(score, dict):
                raw_risk = score.get("score")
            if raw_risk is not None:
                risk = float(raw_risk)
            raw_conf = getattr(score, "confidence", None)
            if raw_conf is None and isinstance(score, dict):
                raw_conf = score.get("confidence")
            if raw_conf is not None:
                risk_confidence = float(raw_conf)

        return CloudVerdict(
            probability=probability,
            action=action,
            action_confidence=action_confidence,
            action_probabilities=action_probabilities,
            execution_risk=risk,
            risk_confidence=risk_confidence,
        )

    # ------------------------------------------------------------------
    # Policy: turn a verdict into allow/deny + the command to execute
    # ------------------------------------------------------------------

    def _cloud_result(
        self,
        verdict: CloudVerdict,
        command: str,
        phrase: str,
        started: float,
        source: str,
    ) -> VerificationResult:
        threshold = self._min_probability()
        max_risk = self._max_risk()
        allow = True
        resolved = command

        if verdict.probability < threshold:
            allow = False
            reason = (
                f"is_valid_command {verdict.probability:.2f} < {threshold:.2f}; "
                f"action={verdict.action}"
            )
            if verdict.action_confidence is not None:
                reason += f" (Choice confidence {verdict.action_confidence:.0%})"
            if verdict.execution_risk is not None:
                reason += f", risk={verdict.execution_risk:.1f}"
        elif verdict.action == "ignore":
            allow = False
            reason = "v1m resolved the phrase to 'ignore'"
        elif verdict.execution_risk is not None and verdict.execution_risk > max_risk:
            allow = False
            reason = (
                f"execution_risk {verdict.execution_risk:.1f} > allowed {max_risk:.1f}"
            )
        elif verdict.execution_risk is None and max_risk < DEFAULT_MAX_EXECUTION_RISK:
            # A risk gate was explicitly tightened but the model did not answer
            # it — be conservative rather than executing blind.
            allow = False
            reason = "execution_risk unavailable while a risk gate is active"
        else:
            local_action = COMMAND_TO_ACTION.get(command)
            if local_action is None or local_action != verdict.action:
                # Use V1M's action when it corrected a local command or when
                # the recognizer supplied no local command hint.
                resolved = ACTION_TO_COMMAND.get(verdict.action, command)
            reason = (
                f"is_valid_command {verdict.probability:.2f}, "
                f"action={verdict.action}"
            )
            if verdict.execution_risk is not None:
                reason += f", risk={verdict.execution_risk:.1f}"

        if not allow:
            self._count("blocked")

        return VerificationResult(
            phrase=phrase or command,
            command=resolved,
            source=source,
            allow=allow,
            reason=reason,
            probability=verdict.probability,
            canonical_action=verdict.action,
            action_confidence=verdict.action_confidence,
            action_probability=(
                (verdict.action_probabilities or {}).get(verdict.action)
            ),
            execution_risk=verdict.execution_risk,
            risk_confidence=verdict.risk_confidence,
            latency_ms=(time.monotonic() - started) * 1000.0,
        )

    def _offline_result(
        self,
        command: str,
        phrase: str,
        started: float,
        reason: str,
        source: str = "offline",
    ) -> VerificationResult:
        if source == "offline":
            self._count("offline")
        verification_unavailable = source != "disabled"
        if verification_unavailable:
            reason = f"{reason}; command not executed without a V1M decision"
        return VerificationResult(
            phrase=phrase or command,
            command=command,
            source=source,
            # The user explicitly disabled V1M: retain the offline command
            # path. If V1M is enabled but unavailable/malformed/timed out,
            # fail closed even when Windows matched a local grammar phrase.
            allow=not verification_unavailable,
            reason=reason,
            probability=None,
            latency_ms=(time.monotonic() - started) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        name = type(exc).__name__
        message = str(exc) or name
        sdk = _typesafe_sdk
        if sdk is not None:
            auth = getattr(sdk, "TypeSafeAuthenticationError", ())
            perm = getattr(sdk, "TypeSafePermissionDeniedError", ())
            conn = getattr(sdk, "TypeSafeAPIConnectionError", ())
            timeout = getattr(sdk, "TypeSafeAPITimeoutError", ())
            if isinstance(exc, (auth, perm)):
                return f"v1m rejected the API key ({message})"
            if isinstance(exc, timeout):
                return f"v1m did not answer in time ({message})"
            if isinstance(exc, conn):
                return f"v1m endpoint unreachable ({message})"
        if "Authentication" in name or "401" in message:
            return f"v1m rejected the API key ({message})"
        if "Timeout" in name or "Connection" in name:
            return f"v1m endpoint unreachable ({message})"
        return f"v1m test failed ({message})"
