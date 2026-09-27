"""Offline microphone -> faster-whisper voice command detector.

The existing Windows helper is useful as a zero-install fallback, but its
finite grammar can turn a short command into an unrelated phrase.  This
detector keeps the same callback contract while replacing the recognizer with
local audio capture, a short adaptive VAD, and faster-whisper transcription.
Only the resulting text and local alternatives are passed to the existing
V1M verifier; no audio is uploaded.
"""

from __future__ import annotations

import difflib
import math
import os
import queue
import re
import threading
import time
from collections import Counter, deque

import numpy as np

COMMAND_ALIASES = {
    "copy": "copy",
    "copy that": "copy",
    "copy this": "copy",
    "paste": "paste",
    "paste that": "paste",
    "paste this": "paste",
    "undo": "undo",
    "undo that": "undo",
    "redo": "redo",
    "select all": "select_all",
    "screenshot": "screenshot",
    "take screenshot": "screenshot",
    "take a screenshot": "screenshot",
    "lock": "lock",
    "lock pc": "lock",
    "lock screen": "lock",
    "lock computer": "lock",
    "desktop": "show_desktop",
    "show desktop": "show_desktop",
    "show the desktop": "show_desktop",
    "calculator": "open_calculator",
    "open calculator": "open_calculator",
    "notepad": "open_notepad",
    "open notepad": "open_notepad",
    "browser": "open_browser",
    "open browser": "open_browser",
    "terminal": "open_terminal",
    "open terminal": "open_terminal",
    "close window": "close_window",
    "close app": "close_window",
    "play": "play",
    "play music": "play",
    "pause": "pause",
    "pause music": "pause",
    "next": "next",
    "next track": "next",
    "previous": "previous",
    "previous track": "previous",
    "mute": "mute",
    "mute sound": "mute",
    "volume up": "volume_up",
    "turn volume up": "volume_up",
    "volume down": "volume_down",
    "turn volume down": "volume_down",
}

WAKE_PREFIXES = ("hey laptop ", "laptop ", "lap top ")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    """Normalize Whisper text without pretending it is a command yet."""
    text = _NON_WORD_RE.sub(" ", str(text or "").lower())
    return _SPACE_RE.sub(" ", text).strip()[:240]


def command_from_text(text: str) -> tuple[str, float]:
    """Return the closest supported command and its conservative similarity."""
    normalized = normalize_transcript(text)
    for prefix in WAKE_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break
    if not normalized:
        return "", 0.0
    exact = COMMAND_ALIASES.get(normalized)
    if exact:
        return exact, 1.0

    # Fast substring containment for multi-word aliases (e.g. "please show desktop")
    words_set = set(normalized.split())
    for alias in sorted(COMMAND_ALIASES.keys(), key=len, reverse=True):
        alias_words = alias.split()
        if len(alias_words) >= 2 and f" {alias} " in f" {normalized} ":
            return COMMAND_ALIASES[alias], 0.95

    # Direct distinct verb containment in short phrases (e.g. "laptop screenshot now")
    if len(words_set) <= 4:
        for keyword in ("screenshot", "calculator", "notepad", "terminal", "undo", "redo"):
            if keyword in words_set:
                return COMMAND_ALIASES[keyword], 0.90

    best_command = ""
    best_score = 0.0
    for alias, command in COMMAND_ALIASES.items():
        score = difflib.SequenceMatcher(None, normalized, alias).ratio()
        if score > best_score:
            best_score, best_command = score, command
    # A fuzzy match is only a hint for V1M.  The verifier still sees the raw
    # transcript and can veto an ambiguous phrase.
    return (best_command, best_score) if best_score >= 0.62 else ("", best_score)


def has_wake_word(text: str) -> bool:
    """Return True only when the transcript starts with the configured wake phrase."""
    normalized = normalize_transcript(text)
    return any(normalized.startswith(prefix.strip()) for prefix in WAKE_PREFIXES)


def looks_like_repetition_hallucination(text: str) -> bool:
    """Reject decoder loops/noise transcripts before they reach V1M."""
    words = normalize_transcript(text).split()
    if any(words[index] == words[index + 1] == words[index + 2]
           for index in range(max(0, len(words) - 2))):
        return True
    counts = Counter(word for word in words if word not in {"laptop", "lap", "top", "hey"})
    if any(count >= 4 for count in counts.values()):
        return True
    return len(words) >= 7 and len(set(words)) / len(words) < 0.60


class WhisperVoiceDetector:
    """Capture short utterances and transcribe them with faster-whisper.

    The class intentionally mirrors ``VoiceDetector``'s public lifecycle and
    callback attributes so GUI and CLI code can use either backend.
    """

    FRAME_MS = 20
    PRE_ROLL_MS = 220
    # Shorter than before, while leaving room for a natural pause between the
    # wake phrase and its action.
    END_SILENCE_MS = 350
    MIN_UTTERANCE_MS = 280
    MAX_UTTERANCE_MS = 4200
    # Whisper can hallucinate ordinary words from very short noise bursts.
    # V1M should receive only plausible speech, even when cloud verification
    # is enabled and the normal confidence threshold is intentionally low.
    MIN_CLOUD_CONFIDENCE = 0.18

    def __init__(self, config_manager, on_voice_command=None, on_status_change=None):
        self.config_manager = config_manager
        self.on_voice_command = on_voice_command
        self.on_status_change = on_status_change
        self.on_speech_active = None
        self.state = "stopped"
        self.last_error = None
        self.is_running = False
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._session_id = 0
        self._stopping = False
        self._config_stop_requested = threading.Event()
        self._last_config_notice = 0.0
        self._startup_event = threading.Event()
        self._frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=120)
        self._utterance_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
        self._thread = None
        self._transcribe_thread = None
        self._stream = None
        self._stream_restart_event = threading.Event()
        self._opened_input_snapshot = None
        self._sounddevice = None
        self._model = None
        self._sample_rate = 16000
        self._input_rate = 16000
        self._noise_rms = 0.003
        self._last_trigger_at = float("-inf")
        config_manager.add_listener(self._on_config_changed)

    @property
    def is_enabled(self):
        return bool(self.config_manager.get("enable_voice_commands", True))

    def _set_state(self, state, message=""):
        self.state = state
        if state == "error":
            self.last_error = message or "Whisper voice engine failed"
        if state in ("ready", "error", "stopped"):
            self._startup_event.set()
        callback = self.on_status_change
        if callback:
            try:
                callback(state, message)
            except Exception as exc:
                print(f"[WhisperVoice] Status callback error: {exc}")

    def _on_config_changed(self):
        if not self.is_running:
            return
        if (
            not self.is_enabled
            or self.config_manager.get("active_mode", "knock") != "voice"
        ):
            # Config listeners may run on the DSP worker. Ask the capture
            # owner to stop; never join audio/transcription threads inline.
            self._config_stop_requested.set()
            return
        if self._input_snapshot() != self._opened_input_snapshot:
            self._stream_restart_event.set()
            now = time.monotonic()
            if now - self._last_config_notice > 0.5:
                self._last_config_notice = now
                self._set_state("starting", "Switching the voice microphone…")

    def _input_snapshot(self):
        return (
            self.config_manager.get("selected_input_device"),
            self.config_manager.get("selected_input_device_signature"),
        )

    def _model_name(self):
        return str(self.config_manager.get("whisper_model", "distil-small.en") or "distil-small.en").strip()

    def _model_options(self):
        device = str(self.config_manager.get("whisper_device", "cpu") or "cpu").strip()
        compute = str(self.config_manager.get("whisper_compute_type", "int8") or "int8").strip()
        return device, compute

    def start(self):
        with self._lifecycle_lock:
            if self._stopping:
                return False
            if self.is_running:
                return True
            for thread in (self._thread, self._transcribe_thread):
                if thread is not None and thread.is_alive():
                    return False
            if not self.is_enabled:
                self._set_state("error", "Voice commands are disabled in settings.")
                return False
            self._session_id += 1
            session_id = self._session_id
            self._stop_event = threading.Event()
            self._config_stop_requested.clear()
            self._stream_restart_event.clear()
            self._startup_event.clear()
            self.last_error = None
            for pending_queue in (self._frame_queue, self._utterance_queue):
                while True:
                    try:
                        pending_queue.get_nowait()
                    except queue.Empty:
                        break
            self.is_running = True
            self._set_state("starting", "Loading local faster-whisper model…")
            self._thread = threading.Thread(target=self._run_capture, args=(session_id,), name="WhisperVoiceCapture", daemon=True)
            self._transcribe_thread = threading.Thread(
                target=self._run_transcriber, args=(session_id,),
                name="WhisperVoiceTranscriber", daemon=True,
            )
            self._thread.start()
            self._transcribe_thread.start()
            return True

    def wait_until_ready(self, timeout_s=8.0):
        if self.state == "ready" and self.is_running:
            return True
        self._startup_event.wait(max(0.0, float(timeout_s)))
        return self.state == "ready" and self.is_running

    def _run_capture(self, session_id):
        try:
            # Windows without Developer Mode cannot create Hugging Face's
            # cache symlinks. Downloads still work through regular copies;
            # hide the non-fatal warning so it does not look like ASR failed.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except Exception as exc:
            self.is_running = False
            self._stop_event.set()
            try:
                self._utterance_queue.put_nowait(None)
            except queue.Full:
                pass
            self._set_state("error", f"Install faster-whisper and sounddevice: {exc}")
            return
        failed = False
        try:
            device, compute_type = self._model_options()
            logical_cpus = os.cpu_count() or 6
            cpu_threads = max(2, min(6, logical_cpus - 2))
            self._model = WhisperModel(
                self._model_name(),
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
            if self._stop_event.is_set():
                return
            self._sounddevice = sd
            self._stream = self._open_stream(sd)
            self._set_state("ready", f"Local Whisper ready ({self._model_name()}; {compute_type}).")
            self._capture_loop(sd, session_id)
        except Exception as exc:
            failed = True
            self.is_running = False
            self._stop_event.set()
            self._set_state("error", f"Whisper voice engine failed: {exc}")
        finally:
            self._close_stream()
            try:
                self._utterance_queue.put_nowait(None)
            except queue.Full:
                pass
            if self._stop_event.is_set() and not failed:
                self.is_running = False
                self._set_state("stopped", "Whisper voice engine stopped.")

    def _open_stream(self, sd):
        blocksize = int(self._sample_rate * self.FRAME_MS / 1000)
        self._opened_input_snapshot = self._input_snapshot()
        device = self._opened_input_snapshot[0]
        stream_kwargs = {"device": device} if device is not None else {}
        try:
            stream = sd.InputStream(
                samplerate=self._sample_rate, channels=1, dtype="float32",
                blocksize=blocksize, callback=self._audio_callback, **stream_kwargs,
            )
            self._input_rate = self._sample_rate
        except Exception:
            try:
                info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
            except Exception:
                info = sd.query_devices(kind="input")
            self._input_rate = int(float(info.get("default_samplerate", 16000)))
            blocksize = max(1, int(self._input_rate * self.FRAME_MS / 1000))
            try:
                stream = sd.InputStream(
                    samplerate=self._input_rate, channels=1, dtype="float32",
                    blocksize=blocksize, callback=self._audio_callback, **stream_kwargs,
                )
            except Exception:
                # A stale device index should not prevent Voice Mode from
                # using the current default microphone.
                try:
                    default_info = sd.query_devices(kind="input")
                    self._input_rate = int(float(default_info.get("default_samplerate", 16000)))
                    blocksize = max(1, int(self._input_rate * self.FRAME_MS / 1000))
                except Exception:
                    pass
                stream = sd.InputStream(
                    samplerate=self._input_rate, channels=1, dtype="float32",
                    blocksize=blocksize, callback=self._audio_callback,
                )
        stream.start()
        return stream

    def _close_stream(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _audio_callback(self, indata, frames, time_info, status):
        if self._stop_event.is_set():
            return
        if status:
            print(f"[WhisperVoice] Audio status: {status}")
        try:
            self._frame_queue.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            # A full queue means transcription is behind. Dropping the oldest
            # frame keeps the recognizer live instead of adding seconds of lag.
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
            except (queue.Empty, queue.Full):
                pass

    def _capture_loop(self, sd, session_id):
        pre_roll = deque(maxlen=max(1, int(self.PRE_ROLL_MS / self.FRAME_MS)))
        utterance = []
        active = False
        silence_ms = 0
        speech_frames = 0
        retry_delay = 0.25
        while session_id == self._session_id and not self._stop_event.is_set():
            if self._config_stop_requested.is_set():
                self._stop_event.set()
                break
            if self._stream_restart_event.is_set():
                try:
                    new_stream = self._open_stream(sd)
                except Exception as exc:
                    # Retain the current stream and retry in-place. Device
                    # enumeration/config errors must not kill Voice Mode.
                    self._set_state("starting", f"Microphone switch failed; retrying: {exc}")
                    self._stop_event.wait(retry_delay)
                    retry_delay = min(4.0, retry_delay * 2)
                    continue
                old_stream, self._stream = self._stream, new_stream
                if old_stream is not None:
                    try:
                        old_stream.stop(); old_stream.close()
                    except Exception:
                        pass
                retry_delay = 0.25
                self._stream_restart_event.clear()
                while True:
                    try: self._frame_queue.get_nowait()
                    except queue.Empty: break
                self._noise_rms = 0.003
                active = False; utterance.clear(); pre_roll.clear()
                speech_frames = 0; silence_ms = 0
                self._set_state("ready", "Whisper voice recognition is running.")
                continue
            try:
                frame = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            centered_frame = frame - float(np.mean(frame))
            rms = float(np.sqrt(np.mean(np.square(centered_frame), dtype=np.float64) + 1e-12))
            if not active:
                # Asymmetric background noise floor tracking:
                # Fall fast when ambient drops, rise slowly to avoid speech leaking into the noise floor
                if rms < self._noise_rms:
                    self._noise_rms = 0.90 * self._noise_rms + 0.10 * rms
                else:
                    self._noise_rms = 0.985 * self._noise_rms + 0.015 * min(rms, 0.04)
            threshold = max(0.0040, min(0.06, self._noise_rms * 2.20))
            speech = rms >= threshold
            if not active:
                pre_roll.append(frame)
                if speech:
                    speech_frames += 1
                else:
                    speech_frames = max(0, speech_frames - 1)
                # Require a short sustained onset. A single keyboard click or
                # fan transient should not wake Whisper and produce a guess.
                if speech_frames >= 3:
                    active = True
                    utterance = list(pre_roll)
                    silence_ms = 0
                    speech_callback = getattr(self, "on_speech_active", None)
                    if speech_callback:
                        try:
                            speech_callback(1.2)
                        except Exception:
                            pass
                continue
            utterance.append(frame)
            if speech:
                silence_ms = 0
            else:
                silence_ms += int(len(frame) * 1000 / self._input_rate) if self._input_rate > 0 else self.FRAME_MS
            elapsed_ms = int(sum(len(f) for f in utterance) * 1000 / self._input_rate) if self._input_rate > 0 else len(utterance) * self.FRAME_MS
            if silence_ms >= self.END_SILENCE_MS or elapsed_ms >= self.MAX_UTTERANCE_MS:
                if elapsed_ms - silence_ms >= self.MIN_UTTERANCE_MS:
                    audio = np.concatenate(utterance).astype(np.float32, copy=False)
                    try:
                        self._utterance_queue.put_nowait(audio)
                    except queue.Full:
                        pass
                active = False
                utterance = []
                pre_roll.clear()
                speech_frames = 0
                silence_ms = 0

    def _prepare_audio(self, audio):
        if audio is None or len(audio) == 0:
            return audio
        # 1. DC offset removal
        audio = audio - float(np.mean(audio))
        # 2. Resample if necessary to 16kHz
        if self._input_rate != self._sample_rate:
            try:
                from scipy.signal import resample_poly
                divisor = math.gcd(self._input_rate, self._sample_rate)
                audio = np.asarray(
                    resample_poly(audio, self._sample_rate // divisor, self._input_rate // divisor),
                    dtype=np.float32,
                )
            except Exception:
                duration = len(audio) / max(1, self._input_rate)
                target_samples = int(duration * self._sample_rate)
                orig_indices = np.linspace(0, duration, len(audio), endpoint=False)
                target_indices = np.linspace(0, duration, target_samples, endpoint=False)
                audio = np.interp(target_indices, orig_indices, audio).astype(np.float32)
        # 3. Peak normalization to optimize Whisper's log-mel dynamic range
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        if peak > 0.01:
            audio = (audio / peak) * 0.92
        return np.asarray(audio, dtype=np.float32)

    def _run_transcriber(self, session_id):
        while session_id == self._session_id and not self._stop_event.is_set():
            try:
                audio = self._utterance_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if audio is None:
                return
            if self._stop_event.is_set():
                return
            try:
                self._transcribe(audio)
            except Exception as exc:
                print(f"[WhisperVoice] Transcription failed: {exc}")

    def _transcribe(self, audio):
        if self._model is None:
            return
        audio = self._prepare_audio(audio)
        # Greedy search (beam_size=1, best_of=1) reduces decoding latency by 35-50%
        # for short domain-prompted command phrases while avoiding beam search wandering.
        transcribe_kwargs = {
            "language": "en",
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "without_timestamps": True,
            "max_new_tokens": 16,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "initial_prompt": (
                "Laptop voice commands: laptop copy, laptop paste, laptop undo, "
                "laptop redo, laptop select all, laptop screenshot, laptop lock, "
                "laptop show desktop, laptop calculator, laptop notepad, "
                "laptop browser, laptop terminal, laptop close window, "
                "laptop play, laptop pause, laptop next, laptop previous, "
                "laptop mute, laptop volume up, laptop volume down."
            ),
        }
        # Try transcription with Silero VAD filtering enabled for speed and noise rejection;
        # fallback without VAD filter if the optional Silero backend is missing.
        try:
            segments, _info = self._model.transcribe(
                audio,
                vad_filter=True,
                vad_parameters=dict(
                    min_speech_duration_ms=180,
                    min_silence_duration_ms=250,
                    speech_pad_ms=120,
                ),
                **transcribe_kwargs,
            )
            parts = list(segments)
        except Exception:
            segments, _info = self._model.transcribe(
                audio,
                vad_filter=False,
                **transcribe_kwargs,
            )
            parts = list(segments)
        raw = normalize_transcript(" ".join(str(segment.text or "") for segment in parts))
        if not raw:
            return
        if looks_like_repetition_hallucination(raw):
            print("[WhisperVoice] Discarded a repetitive transcript before V1M.")
            return
        no_speech_probs = [float(getattr(s, "no_speech_prob", 0.0)) for s in parts]
        if no_speech_probs and all(prob >= 0.65 for prob in no_speech_probs):
            return
        avg_logprob = sum(float(getattr(s, "avg_logprob", -1.5)) for s in parts) / max(1, len(parts))
        confidence = max(0.05, min(0.99, math.exp(min(0.0, avg_logprob))))
        command, similarity = command_from_text(raw)
        if not command and not bool(
            self.config_manager.get("enable_v1m_verification", False)
        ):
            # No local mapping is not a command candidate in offline mode.
            return
        try:
            threshold = float(self.config_manager.get("voice_confidence_threshold", 0.55))
        except (TypeError, ValueError):
            threshold = 0.55
        cloud_enabled = bool(self.config_manager.get("enable_v1m_verification", False))
        if confidence < threshold and not cloud_enabled:
            return
        if cloud_enabled and confidence < max(self.MIN_CLOUD_CONFIDENCE, threshold * 0.35):
            return
        if bool(self.config_manager.get("voice_require_wake_word", True)) and not has_wake_word(raw):
            return
        alternatives = [{"text": raw, "confidence": confidence}]
        if command:
            canonical = next((alias for alias, value in COMMAND_ALIASES.items() if value == command), command)
            hinted = f"laptop {canonical}"
            if hinted != raw:
                alternatives.append({"text": hinted, "confidence": max(0.35, confidence * max(0.65, similarity))})
        callback = self.on_voice_command
        if (
            callback
            and self.is_running
            and not self._stop_event.is_set()
            and time.monotonic() - self._last_trigger_at >= self._cooldown_s()
        ):
            self._last_trigger_at = time.monotonic()
            try:
                callback(command, confidence, raw, alternatives)
            except Exception as exc:
                print(f"[WhisperVoice] Command callback error: {exc}")

    def _cooldown_s(self):
        try:
            return max(0, int(self.config_manager.get("voice_command_cooldown_ms", 650))) / 1000.0
        except (TypeError, ValueError):
            return 0.65

    def stop(self):
        with self._lifecycle_lock:
            self._stopping = True
            self._session_id += 1
            self._stop_event.set()
            self._close_stream()
            for q in (self._frame_queue, self._utterance_queue):
                try: q.put_nowait(None)
                except queue.Full: pass
            threads = (self._thread, self._transcribe_thread)
        # Never report stopped or permit another session while either worker
        # can still touch this detector/model.
        for thread in threads:
            if thread and thread is not threading.current_thread():
                thread.join()
        with self._lifecycle_lock:
            self._thread = self._transcribe_thread = None
            self.is_running = False
            self._stopping = False
            self._set_state("stopped", "Whisper voice engine stopped.")

    def cleanup(self):
        """Release all resources and unsubscribe from config changes."""
        self.stop()
        self.config_manager.remove_listener(self._on_config_changed)
