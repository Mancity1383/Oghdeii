import base64
import os
import re
import subprocess
import threading
import time
from pathlib import Path

VOICE_COMMANDS = {
    "copy", "paste", "undo", "redo", "select_all", "screenshot", "lock",
    "show_desktop", "close_window", "open_calculator", "open_notepad",
    "open_browser", "open_terminal", "play", "pause", "next", "previous",
    "mute", "volume_up", "volume_down",
}

_VOICE_PHRASE_COMMANDS = {
    "copy": "copy", "copy that": "copy", "copy this": "copy",
    "paste": "paste", "paste that": "paste", "paste this": "paste",
    "screenshot": "screenshot", "take screenshot": "screenshot",
    "take a screenshot": "screenshot",
    "lock pc": "lock", "lock screen": "lock", "lock computer": "lock",
    "undo": "undo", "undo that": "undo", "redo": "redo",
    "select all": "select_all",
    "desktop": "show_desktop", "show desktop": "show_desktop",
    "calculator": "open_calculator", "open calculator": "open_calculator",
    "notepad": "open_notepad", "open notepad": "open_notepad",
    "browser": "open_browser", "open browser": "open_browser",
    "terminal": "open_terminal", "open terminal": "open_terminal",
    "close window": "close_window", "close app": "close_window",
    "play": "play", "play music": "play",
    "pause": "pause", "pause music": "pause",
    "next": "next", "next track": "next",
    "previous": "previous", "previous track": "previous",
    "mute": "mute", "mute sound": "mute",
    "volume up": "volume_up", "volume down": "volume_down",
}


def _command_from_constrained_phrase(text):
    """Return the command hint for a phrase accepted by the finite grammar."""
    phrase = " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    for prefix in ("hey laptop ", "lap top ", "laptop "):
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix):]
            break
    return _VOICE_PHRASE_COMMANDS.get(phrase, "")


class VoiceDetector:
    """Manage the Windows System.Speech helper and its validated text protocol.

    Lifecycle guarantees provided by this class:

    * Exactly one helper process per session.  Threads are bound to the exact
      ``proc`` they were created for, so a fast ``stop()`` -> ``start()``
      (slider release, wake-word toggle, auto-restart) can never let an old
      reader thread clobber the new session's state or read from the wrong
      pipe.
    * Self-healing: a helper that dies, prints ``ERROR:`` (e.g. the audio
      device was removed) or stops answering ``ping`` probes is restarted
      automatically with a bounded attempt budget instead of leaving voice
      mode silently dead.  Three failures without ever becoming READY gives
      up and surfaces an error instead of crash-looping.
    * Live config sync: relevant edits from the GUI, voice_calibrate.py, or
      other config writers schedule a debounced respawn. Confidence is checked
      before forwarding recognized command phrases; action mappings and
      cooldowns are resolved live on every event.
    * Clean shutdown: ``stop()`` sends "quit", closes stdin (EOF makes even a
      helper that missed "quit" exit), escalates terminate -> kill with tight
      timeouts, cancels pending restart timers and joins the session's IO
      threads — no orphaned helper and no restart-after-quit resurrection.
    """

    # --- tunables (overridable per-instance, e.g. in tests) ---
    STOP_QUIT_WAIT_S = 1.0        # graceful "quit" grace period
    STOP_TERM_WAIT_S = 0.6        # after terminate()
    STOP_KILL_WAIT_S = 0.6        # after kill()
    THREAD_JOIN_TIMEOUT_S = 1.5   # per-thread join cap in stop()
    RESTART_DEBOUNCE_S = 0.5      # coalesce restart triggers (slider drags)
    RESTART_MAX_ATTEMPTS = 3      # failures allowed within RESTART_WINDOW_S
    RESTART_WINDOW_S = 120.0      # budget window; READY resets the budget
    MONITOR_INTERVAL_S = 5.0      # death-watch / ping cadence
    PONG_TIMEOUT_S = 12.0         # ~2 missed pongs => hung helper

    def __init__(self, config_manager, on_voice_command=None, on_status_change=None):
        self.config_manager = config_manager
        self.on_voice_command = on_voice_command
        self.on_status_change = on_status_change
        self.proc = None
        self.is_running = False
        self.state = "stopped"
        self.last_error = None
        self.thread = None
        self.stderr_thread = None
        self.monitor_thread = None
        self.exe_path = Path(__file__).parent / "VoiceRecognizerV1M.exe"
        self._lock = threading.RLock()
        self._stdin_lock = threading.Lock()   # serializes quit/ping writes
        self._startup_event = threading.Event()
        self._monitor_wake = threading.Event()
        self._restart_timer = None
        self._restart_generation = 0
        self._restart_attempts = []
        self._spawned_conf = None
        self._spawned_mode = None
        self._session_started_at = 0.0
        self._ping_supported = False
        self._ping_exhausted = False
        self._last_ping_sent_at = 0.0
        self._last_pong_at = 0.0
        self._last_trigger_at = float("-inf")
        self._last_cmd = None
        # Live config sync: debounce helper restarts after relevant settings
        # edits. Confidence changes update the command-event gate.
        config_manager.add_listener(self._on_config_changed)

    # ------------------------------------------------------------------
    # Configuration sync
    # ------------------------------------------------------------------

    @property
    def is_enabled(self):
        return bool(self.config_manager.get("enable_voice_commands", True))

    def _desired_conf(self):
        """Threshold exactly as the helper would receive it on spawn."""
        try:
            conf = float(self.config_manager.get("voice_confidence_threshold", 0.55))
        except (TypeError, ValueError):
            conf = 0.72
        conf = min(0.95, max(0.20, conf))
        return float(f"{conf:.2f}")

    def _desired_mode(self):
        return "wake" if self.config_manager.get("voice_require_wake_word", True) else "direct"

    def _on_config_changed(self):
        """React to config edits from THIS process or any other one.

        Runs on whatever thread wrote the config (GUI, audio worker, a
        debounced-save reload, calibrate.py in a second process).  Only keys
        baked into the spawned helper trigger a respawn; action mappings and
        cooldowns are read live per event, so they apply instantly without
        touching the helper.
        """
        with self._lock:
            running = self.is_running and self.proc is not None
        if not running:
            return  # helper starts with fresh values on the next mode switch
        if not self.is_enabled:
            # Disabled externally while the helper was up: shut it down.
            self.stop()
            return
        conf = self._desired_conf()
        mode = self._desired_mode()
        with self._lock:
            if conf == self._spawned_conf and mode == self._spawned_mode:
                return  # nothing baked into the helper changed
        # Intentional respawn, not a failure: never counts against the
        # watchdog budget (the user could otherwise exhaust it by dragging
        # the slider faster than the helper reaches READY).
        self._schedule_restart(
            reason="config changed (threshold/wake-word)",
            count_toward_budget=False,
        )

    # ------------------------------------------------------------------
    # State handling
    # ------------------------------------------------------------------

    def _set_state(self, state, message=""):
        lowered = str(message).lower()
        if state == "error" and ("access is denied" in lowered or "e_accessdenied" in lowered):
            message = (
                "Microphone access was denied. Enable Microphone access and "
                "Let desktop apps access your microphone in Windows Privacy settings."
            )
        elif state == "error" and "no english windows speech recognizer" in lowered:
                message = "Install an English speech language in Windows Settings, then restart عقده ای."
        if state == "ready":
            # Healthy session: refill the auto-restart failure budget.
            with self._lock:
                self._restart_attempts = []
        self.state = state
        if state == "error":
            self.last_error = message or "Voice recognition failed"
        if state in ("ready", "error", "stopped"):
            self._startup_event.set()
        callback = self.on_status_change
        if callback:
            try:
                callback(state, message)
            except Exception as exc:
                print(f"[VoiceDetector] Status callback error: {exc}")

    def _is_current(self, proc):
        with self._lock:
            return proc is not None and self.proc is proc

    # ------------------------------------------------------------------
    # Spawning / session start
    # ------------------------------------------------------------------

    def _spawn_process(self, conf, mode):
        """Create the helper process (seam: tests replace this)."""
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        return subprocess.Popen(
            [str(self.exe_path), f"{conf:.2f}", mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            startupinfo=startupinfo,
        )

    def _start_io_threads(self, proc):
        """Start reader/stderr/monitor threads bound to this exact proc."""
        self.thread = threading.Thread(
            target=self._reader_loop, args=(proc,),
            name="VoiceRecognizerStdout", daemon=True,
        )
        self.stderr_thread = threading.Thread(
            target=self._stderr_loop, args=(proc,),
            name="VoiceRecognizerStderr", daemon=True,
        )
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(proc, self._monitor_wake),
            name="VoiceRecognizerMonitor", daemon=True,
        )
        self.thread.start()
        self.stderr_thread.start()
        self.monitor_thread.start()

    def _invalidate_pending_restart_locked(self):
        """Called with _lock held: a manual stop/start supersedes timers."""
        self._restart_generation += 1
        timer = self._restart_timer
        self._restart_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def start(self):
        with self._lock:
            if self.is_running:
                return True
            if not self.is_enabled:
                self._set_state("error", "Voice commands are disabled in settings.")
                return False
            if os.name != "nt":
                self._set_state("error", "Windows System.Speech is only available on Windows.")
                return False
            if not self.exe_path.exists():
                self._set_state("error", f"Voice engine was not found: {self.exe_path}")
                return False

            # Manual (or watchdog) start supersedes any pending restart timer.
            self._invalidate_pending_restart_locked()

            conf = self._desired_conf()
            mode = self._desired_mode()
            try:
                proc = self._spawn_process(conf, mode)
            except Exception as exc:
                self._set_state("error", f"Failed to start voice engine: {exc}")
                return False

            now = time.monotonic()
            self.proc = proc
            self.is_running = True
            self.last_error = None
            self._startup_event.clear()
            self._last_cmd = None
            self._last_trigger_at = float("-inf")
            self._spawned_conf = conf
            self._spawned_mode = mode
            self._session_started_at = now
            self._ping_supported = False
            self._ping_exhausted = False
            self._last_ping_sent_at = 0.0
            self._last_pong_at = now
            self._monitor_wake = threading.Event()
            self._set_state("starting", "Starting local speech-to-text…")
            try:
                self._start_io_threads(proc)
            except Exception as exc:
                # Roll the half-started session back so is_running never
                # points at a process without readers.
                self.proc = None
                self.is_running = False
                try:
                    proc.kill()
                except Exception:
                    pass
                self._set_state("error", f"Failed to start voice engine readers: {exc}")
                return False
            return True

    def wait_until_ready(self, timeout_s=8.0):
        """Wait for the helper's READY/error response without guessing from process creation."""
        if self.state == "ready" and self.is_running:
            return True
        self._startup_event.wait(max(0.0, float(timeout_s)))
        if self.state == "ready" and self.is_running:
            return True
        if self.state not in ("error", "stopped"):
            self._set_state("error", "Voice engine did not become ready before the startup timeout.")
        return False

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_protocol_line(line):
        if line.startswith("VOICE_TEXT2:"):
            parts = line.split(":", 2)
            if len(parts) != 3:
                return None
            try:
                conf = float(parts[1].strip().replace(",", "."))
                payload = base64.b64decode(parts[2].strip(), validate=True).decode("utf-8")
            except (ValueError, TypeError, UnicodeDecodeError):
                return None
            if not 0.0 <= conf <= 1.0:
                return None
            candidates = []
            for item in payload.split("\x1f")[:5]:
                try:
                    candidate_conf, candidate_text = item.split("\x1e", 1)
                    candidate_conf = float(candidate_conf.replace(",", "."))
                except (ValueError, TypeError):
                    continue
                candidate_text = candidate_text.strip()[:200]
                if candidate_text and 0.0 <= candidate_conf <= 1.0:
                    candidates.append({"text": candidate_text, "confidence": candidate_conf})
            if not candidates:
                return None
            spoken = candidates[0]["text"]
            command = _command_from_constrained_phrase(spoken)
            return "VOICE_TEXT", command, conf, spoken, candidates

        if line.startswith("VOICE_TEXT:"):
            parts = line.split(":", 2)
            if len(parts) != 3:
                return None
            try:
                conf = float(parts[1].strip().replace(",", "."))
            except (ValueError, TypeError):
                return None
            spoken = parts[2].strip()[:200]
            if not 0.0 <= conf <= 1.0 or not spoken:
                return None
            # Empty command means no local intent/action classification was
            # performed; V1M must choose an action from the transcript.
            return "VOICE_TEXT", "", conf, spoken, []

        if line.startswith("DEBUG_REJECTED_TEXT:"):
            # Windows may reject a low-energy or out-of-grammar utterance but
            # still provide a text hypothesis. Keep it unclassified locally;
            # V1M can use it as fallible transcript evidence when enabled.
            parts = line.split(":", 2)
            if len(parts) != 3:
                return None
            try:
                conf = float(parts[1].strip().replace(",", "."))
            except (ValueError, TypeError):
                return None
            spoken = parts[2].strip()[:200]
            if not 0.0 <= conf <= 1.0 or not spoken:
                return None
            return "VOICE_TEXT", "", conf, spoken, [
                {"text": spoken, "confidence": conf}
            ]

        parts = line.split(":", 3)
        kind = parts[0] if parts else ""
        if kind not in ("VOICE_CMD", "DEBUG_LOW_CONF", "DEBUG_REJECTED") or len(parts) < 3:
            return None
        cmd = parts[1].strip().lower()
        if cmd not in VOICE_COMMANDS:
            return None
        try:
            conf = float(parts[2].strip().replace(",", "."))
        except (ValueError, TypeError):
            return None
        if not 0.0 <= conf <= 1.0:
            return None
        spoken = parts[3].strip() if len(parts) >= 4 else cmd
        return kind, cmd, conf, spoken[:200], []

    def _handle_line(self, proc, line):
        """Process one stdout line from the helper.

        Guarded by proc identity: output from a replaced or stopped session
        can never flip the new session's state or dispatch duplicate commands.
        """
        if not self._is_current(proc):
            return
        parsed = self._parse_protocol_line(line)
        if parsed:
            kind, cmd, conf, spoken, candidates = parsed
            if kind not in ("VOICE_CMD", "VOICE_TEXT"):
                return
            try:
                threshold = float(self.config_manager.get("voice_confidence_threshold", 0.55))
            except (TypeError, ValueError):
                threshold = 0.55
            # When cloud classification is enabled, let V1M evaluate Windows'
            # transcript and N-best confidences instead of silently dropping
            # low-confidence hypotheses here. Keep the local threshold for the
            # explicit offline-only mode, where no cloud intent gate exists.
            v1m_enabled = bool(
                self.config_manager.get("enable_v1m_verification", False)
            )
            if not self.is_enabled or (conf < threshold and not v1m_enabled):
                return
            now = time.monotonic()
            try:
                cooldown_ms = int(self.config_manager.get("voice_command_cooldown_ms", 650))
            except (TypeError, ValueError):
                cooldown_ms = 650
            cooldown_s = max(0, cooldown_ms) / 1000.0
            # GLOBAL cooldown: any command inside the window is suppressed,
            # not just repeats of the same one — alternating ambient phrases
            # (TV, nearby conversation) used to slip past a per-command check.
            if now - self._last_trigger_at < cooldown_s:
                return
            self._last_cmd = cmd or spoken.lower()
            self._last_trigger_at = now
            callback = self.on_voice_command
            if callback:
                try:
                    callback(cmd, conf, spoken, candidates)
                except Exception as exc:
                    print(f"[VoiceDetector] Command callback error: {exc}")
            return

        if line == "STATUS:READY":
            self._set_state("ready", "Speech-to-text is ready; V1M selects the action.")
        elif line == "PONG":
            self._ping_supported = True
            self._last_pong_at = time.monotonic()
        elif line.startswith("DEBUG_"):
            # Diagnostics are consumed by voice_calibrate.py, not by the
            # action-running detector.
            return
        elif line.startswith("STATUS:"):
            self._set_state(self.state, line[7:].replace(":", " — "))
        elif line.startswith("ERROR:"):
            # The helper reported a fatal condition (mic lost, recognition
            # pipeline ended).  Schedule a restart so voice mode recovers on
            # its own — works even with helpers that stay alive after ERROR.
            self._set_state("error", line[6:].strip())
            self._schedule_restart(reason="helper reported a fatal error")
        else:
            print(f"[VoiceDetector] Ignoring malformed helper output: {line}")

    # ------------------------------------------------------------------
    # IO threads
    # ------------------------------------------------------------------

    def _reader_loop(self, proc):
        try:
            while True:
                with self._lock:
                    if not (self.is_running and self.proc is proc):
                        return  # session stopped or replaced: let the new one win
                if proc.poll() is not None:
                    break
                line = proc.stdout.readline()
                if not line:
                    break  # EOF: helper closed stdout (or died)
                line = line.strip()
                if not line:
                    continue
                self._handle_line(proc, line)
        except Exception as exc:
            if self._is_current(proc) and self.is_running:
                self._set_state("error", f"Voice engine reader failed: {exc}")
        finally:
            self._on_reader_exit(proc)

    def _on_reader_exit(self, proc):
        """Idempotent reader-exit handler; restarts only for the CURRENT session."""
        with self._lock:
            current = proc is not None and self.proc is proc
            unexpected = current and self.is_running
            if current:
                self.is_running = False
        if not unexpected:
            return  # stale thread from a replaced session: clobber nothing
        code = proc.poll()
        if self.state != "error":
            self._set_state("error", f"Voice engine exited unexpectedly (code {code}).")
        self._schedule_restart(proc, reason="helper exited unexpectedly")

    def _stderr_loop(self, proc):
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return
        try:
            for line in stream:
                if line.strip():
                    print(f"[VoiceDetector:stderr] {line.rstrip()}")
        except Exception:
            pass

    def _send_ping(self, proc):
        with self._stdin_lock:
            stdin = getattr(proc, "stdin", None)
            if stdin is None or getattr(stdin, "closed", None) is True:
                return
            try:
                stdin.write("ping\n")
                stdin.flush()
            except Exception:
                pass  # broken pipe: the reader/death-watch handles the exit

    def _monitor_loop(self, proc, wake):
        """Death belt + hang probe.

        The reader notices EOF, but it can stay parked on readline() forever
        if the helper wedges without closing its pipe.  Helpers supporting
        the ping probe answer "PONG"; one that answered before and stops
        answering is treated as hung and restarted.  Legacy helpers (built
        before ping support) are probed once, then watched for death only.
        """
        while True:
            if wake.wait(self.MONITOR_INTERVAL_S):
                return  # stop() requested
            with self._lock:
                if not (self.is_running and self.proc is proc):
                    return
            if proc.poll() is not None:
                # The reader fires _on_reader_exit too; it is idempotent.
                self._on_reader_exit(proc)
                return

            now = time.monotonic()
            if self._ping_supported and now - self._last_pong_at > self.PONG_TIMEOUT_S:
                # It answered before and went quiet: the main loop is wedged.
                print("[VoiceDetector] Voice engine stopped answering ping probes — assuming it hung.")
                self._set_state("error", "Voice engine hung (no response to ping).")
                self._schedule_restart(proc, reason="helper stopped answering ping probes")
                return
            if not self._ping_supported and not self._ping_exhausted \
                    and now - self._session_started_at > self.PONG_TIMEOUT_S:
                # First probe never answered: helper predates ping support.
                # Stop probing and rely on the death-watch alone.
                self._ping_exhausted = True
            if not self._ping_exhausted and now - self._last_ping_sent_at >= self.MONITOR_INTERVAL_S:
                self._send_ping(proc)
                self._last_ping_sent_at = now

    # ------------------------------------------------------------------
    # Auto-restart watchdog
    # ------------------------------------------------------------------

    def _schedule_restart(self, proc=None, delay=None, reason="automatic restart",
                          count_toward_budget=True):
        """Arm a debounced restart for the CURRENT session (idempotent).

        count_toward_budget=False marks an intentional respawn (config edit);
        only watchdog restarts (death / ERROR / hang) consume the failure
        budget that ultimately gives up.
        """
        with self._lock:
            if proc is None:
                proc = self.proc
            if proc is None or self.proc is not proc:
                return  # session already replaced or stopped: nothing to do
            self._restart_generation += 1
            gen = self._restart_generation
            if self._restart_timer is not None:
                try:
                    self._restart_timer.cancel()
                except Exception:
                    pass
            timer = threading.Timer(
                self.RESTART_DEBOUNCE_S if delay is None else delay,
                self._run_scheduled_restart,
                args=(gen,),
                kwargs={"reason": reason, "count_toward_budget": count_toward_budget},
            )
            timer.daemon = True
            self._restart_timer = timer
            timer.start()

    def _run_scheduled_restart(self, gen, reason="automatic restart",
                               count_toward_budget=True):
        """Timer target: bounded stop+start of the helper."""
        with self._lock:
            if gen != self._restart_generation:
                return  # superseded by a manual stop/start or a newer schedule
            if self.proc is None:
                return
            self._restart_timer = None
            attempts = [
                t for t in self._restart_attempts
                if time.monotonic() - t < self.RESTART_WINDOW_S
            ]
            if count_toward_budget:
                if len(attempts) >= self.RESTART_MAX_ATTEMPTS:
                    self._restart_attempts = attempts
                    give_up = True
                else:
                    attempts.append(time.monotonic())
                    self._restart_attempts = attempts
                    give_up = False
            else:
                give_up = False
        if give_up:
            self.stop()
            self._set_state(
                "error",
                f"Voice engine failed {len(attempts)} times without becoming "
                "ready — automatic restart gave up. Switch modes or restart "
                "Oghdeii to retry.",
            )
            return
        attempt_note = (
            f"attempt {len(attempts)}/{self.RESTART_MAX_ATTEMPTS}"
            if count_toward_budget else "config respawn"
        )
        print(f"[VoiceDetector] Respawning voice engine ({reason}; {attempt_note})…")
        self.stop()
        if self.is_enabled:
            self.start()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self):
        with self._lock:
            proc = self.proc
            self.proc = None
            self.is_running = False
            self._invalidate_pending_restart_locked()
            wake = self._monitor_wake
            threads = (self.thread, self.stderr_thread, self.monitor_thread)
            self.thread = self.stderr_thread = self.monitor_thread = None
        wake.set()
        if proc is not None:
            try:
                if proc.poll() is None:
                    with self._stdin_lock:
                        stdin = getattr(proc, "stdin", None)
                        if stdin is not None and getattr(stdin, "closed", None) is not True:
                            try:
                                stdin.write("quit\n")
                                stdin.flush()
                            except Exception:
                                pass
                            # Closing our write end guarantees the helper sees
                            # stdin EOF: even if "quit" was lost, its ReadLine()
                            # returns null and the child exits — no orphan.
                            try:
                                stdin.close()
                            except Exception:
                                pass
                    try:
                        proc.wait(timeout=self.STOP_QUIT_WAIT_S)
                    except subprocess.TimeoutExpired:
                        try:
                            proc.terminate()
                            proc.wait(timeout=self.STOP_TERM_WAIT_S)
                        except Exception:
                            try:
                                proc.kill()
                                proc.wait(timeout=self.STOP_KILL_WAIT_S)
                            except Exception:
                                print("[VoiceDetector] Warning: voice helper did not exit cleanly.")
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Join the session's IO threads so a fast stop() -> start() never has
        # old readers lingering next to the new session.
        me = threading.current_thread()
        for t in threads:
            if t is not None and t is not me:
                try:
                    t.join(timeout=self.THREAD_JOIN_TIMEOUT_S)
                except Exception:
                    pass
        self._set_state("stopped", "Voice engine stopped.")
