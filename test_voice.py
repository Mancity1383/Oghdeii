"""Tests for the Voice Detection subsystem.

Covers the audit fixes:
* spawn snapshot + debounced respawn when baked-in config changes
  (threshold / wake-word mode) — in-process and cross-process edits;
* no respawn for live-resolved keys (voice action mappings);
* bounded auto-restart budget with give-up;
* stale-session isolation: old reader/exit/line events can never clobber
  the new session's state or schedule restarts against it;
* ERROR-line and EOF recovery, PONG ping capability;
* global command cooldown (alternating ambient phrases are suppressed);
* clean shutdown: quit + stdin close, timers cancelled, state settled;
* tooling cleanup paths (voice_calibrate / voice_validate close stdin).
"""

import io
import tempfile
import time
import unittest
from unittest import mock

import voice_calibrate
import voice_validate
from config_manager import ConfigManager, DEFAULT_CONFIG
from voice_detector import VOICE_COMMANDS, VoiceDetector


def _fake_proc():
    """A live-looking helper process: poll() None, writable mock stdin."""
    proc = mock.MagicMock()
    proc.poll.return_value = None
    proc.stdin = mock.MagicMock()
    proc.stdin.closed = False
    return proc


class VoiceHarness(unittest.TestCase):
    """Shared fixture: temp config + detector with spawn/threads mocked out.

    RESTART_DEBOUNCE_S defaults to 30s so tests can assert that a restart
    was *scheduled* without racing the timer; tests that want the timer to
    actually fire override it with a small value.
    """

    def make_detector(self, debounce=30.0, cfg=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cm = ConfigManager(tmp.name)
        for key, value in (cfg or {}).items():
            cm.set(key, value)
        vd = VoiceDetector(cm)
        self.addCleanup(vd.stop)  # LIFO: runs before the tmp dir disappears
        vd.RESTART_DEBOUNCE_S = debounce
        vd._spawn_process = mock.Mock(return_value=_fake_proc())
        vd._start_io_threads = mock.Mock()
        return vd, cm


class SpawnSnapshotTests(VoiceHarness):
    """The helper's baked-in arguments must mirror the live config."""

    def test_start_bakes_configured_threshold_and_mode(self):
        vd, cm = self.make_detector(cfg={
            "voice_confidence_threshold": 0.85,
            "voice_require_wake_word": False,
        })
        self.assertTrue(vd.start())
        vd._spawn_process.assert_called_once()
        self.assertEqual(vd._spawn_process.call_args[0], (0.85, "direct"))
        self.assertEqual(vd._spawned_conf, 0.85)
        self.assertEqual(vd._spawned_mode, "direct")
        self.assertEqual(vd.state, "starting")

    def test_threshold_change_while_running_schedules_respawn(self):
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        self.assertIsNone(vd._restart_timer)
        cm.set("voice_confidence_threshold", 0.40)
        self.assertIsNotNone(vd._restart_timer)  # armed, long debounce

    def test_wake_word_change_while_running_schedules_respawn(self):
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        self.assertIsNone(vd._restart_timer)
        cm.set("voice_require_wake_word", False)
        self.assertIsNotNone(vd._restart_timer)

    def test_config_change_while_stopped_schedules_nothing(self):
        vd, cm = self.make_detector(debounce=30.0)
        gen = vd._restart_generation
        cm.set("voice_confidence_threshold", 0.40)
        cm.set("voice_require_wake_word", False)
        self.assertIsNone(vd._restart_timer)
        self.assertEqual(vd._restart_generation, gen)
        # Next start picks the new values up directly.
        vd.start()
        self.assertEqual(vd._spawn_process.call_args[0][0], 0.40)
        self.assertEqual(vd._spawn_process.call_args[0][1], "direct")

    def test_voice_mapping_change_does_not_restart_helper(self):
        """Binding edits apply live on the next utterance — never respawn."""
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        cm.set("voice_actions", {**DEFAULT_CONFIG["voice_actions"], "copy": "undo"})
        cm.set("voice_command_cooldown_ms", 250)
        cm.set("sensitivity", 0.55)
        self.assertIsNone(vd._restart_timer)
        self.assertEqual(vd._spawn_process.call_count, 1)

    def test_unrelated_keys_do_not_reschedule_after_spawn_is_in_sync(self):
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        cm.set("voice_confidence_threshold", 0.40)          # pending respawn
        if vd._restart_timer is not None:
            vd._restart_timer.cancel()
        vd._restart_timer = None
        vd._spawned_conf = vd._desired_conf()               # respawn "happened"
        gen = vd._restart_generation
        cm.set("sensitivity", 0.61)
        cm.set("enable_notifications", False)
        self.assertEqual(vd._restart_generation, gen)
        self.assertIsNone(vd._restart_timer)

    def test_disable_voice_commands_stops_running_helper(self):
        vd, cm = self.make_detector()
        self.assertTrue(vd.start())
        cm.set("enable_voice_commands", False)
        self.assertFalse(vd.is_running)
        self.assertIsNone(vd.proc)
        self.assertEqual(vd.state, "stopped")

    def test_detector_subscribes_to_config_changes(self):
        vd, cm = self.make_detector()
        self.assertIn(vd._on_config_changed, cm._listeners)


class AutoRestartTests(VoiceHarness):
    """Watchdog: ERROR lines, reader EOF, bounded budget, stale isolation."""

    def test_scheduled_restart_respawns_helper(self):
        vd, cm = self.make_detector(debounce=0.05)
        vd.start()
        self.assertEqual(vd._spawn_process.call_count, 1)
        cm.set("voice_confidence_threshold", 0.33)
        deadline = time.monotonic() + 3.0
        while vd._spawn_process.call_count < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(vd._spawn_process.call_count, 2)
        self.assertEqual(vd._spawned_conf, 0.33)
        self.assertTrue(vd.is_running)

    def test_config_respawns_never_consume_failure_budget(self):
        """Rapid threshold edits are intentional respawns: even four in a
        row (with no READY in between, mocked threads) must all happen and
        must never trigger the watchdog's give-up path."""
        vd, cm = self.make_detector(debounce=0.05)
        vd.start()
        for i, conf in enumerate((0.60, 0.55, 0.50, 0.45), start=1):
            cm.set("voice_confidence_threshold", conf)
            deadline = time.monotonic() + 3.0
            while vd._spawn_process.call_count < i + 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(vd._spawn_process.call_count, i + 1,
                             f"respawn #{i + 1} after threshold {conf}")
        with vd._lock:
            self.assertEqual(vd._restart_attempts, [])  # never counted
        self.assertTrue(vd.is_running)
        self.assertNotEqual(vd.state, "error")

    def test_restart_budget_gives_up_without_ready(self):
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        with vd._lock:
            vd._restart_attempts = [time.monotonic() - 10.0] * 3
        vd._run_scheduled_restart(vd._restart_generation)
        self.assertEqual(vd.state, "error")
        self.assertIn("gave up", vd.last_error or "")
        self.assertFalse(vd.is_running)
        self.assertEqual(vd._spawn_process.call_count, 1)  # no respawn

    def test_ready_refills_restart_budget(self):
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        with vd._lock:
            vd._restart_attempts = [time.monotonic()]
        vd._handle_line(vd.proc, "STATUS:READY")
        with vd._lock:
            self.assertEqual(vd._restart_attempts, [])

    def test_stale_restart_generation_is_ignored(self):
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        vd._run_scheduled_restart(vd._restart_generation + 5)
        self.assertEqual(vd._spawn_process.call_count, 1)
        self.assertTrue(vd.is_running)

    def test_old_session_exit_cannot_clobber_new_session(self):
        """Regression: a replaced session's reader finishing late must not
        flip the new session to error, clear its is_running, or schedule a
        restart against it."""
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        stale_proc = mock.MagicMock()
        vd._on_reader_exit(stale_proc)
        self.assertTrue(vd.is_running)
        self.assertEqual(vd.state, "starting")
        self.assertIsNone(vd._restart_timer)

    def test_error_line_schedules_restart_for_current_session_only(self):
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        vd._handle_line(
            vd.proc, "ERROR:Recognition completed with error: device gone"
        )
        self.assertEqual(vd.state, "error")
        self.assertIn("device gone", vd.last_error or "")
        self.assertIsNotNone(vd._restart_timer)  # recovery scheduled
        # Stale output must be ignored entirely.
        vd._handle_line(mock.MagicMock(), "ERROR:stale clobber")
        self.assertNotIn("stale clobber", vd.last_error or "")

    def test_reader_eof_marks_error_and_schedules_restart(self):
        vd, cm = self.make_detector(debounce=30.0)
        proc = _fake_proc()
        proc.stdout = io.StringIO("STATUS:READY\n")
        vd._spawn_process = mock.Mock(return_value=proc)
        vd.start()
        vd._reader_loop(proc)  # READY, then EOF -> unexpected exit
        self.assertEqual(vd.state, "error")
        self.assertIn("exited unexpectedly", vd.last_error or "")
        self.assertIsNotNone(vd._restart_timer)
        self.assertFalse(vd.is_running)

    def test_manual_stop_then_start_supersedes_pending_restart(self):
        vd, cm = self.make_detector(debounce=30.0)
        vd.start()
        cm.set("voice_confidence_threshold", 0.40)
        self.assertIsNotNone(vd._restart_timer)
        gen = vd._restart_generation
        vd.stop()
        self.assertIsNone(vd._restart_timer)  # cancelled, not just ignored
        vd.start()
        self.assertGreater(vd._restart_generation, gen)
        self.assertEqual(vd._spawn_process.call_count, 2)

    def test_pong_marks_ping_capability(self):
        vd, cm = self.make_detector()
        vd.start()
        self.assertFalse(vd._ping_supported)
        vd._handle_line(vd.proc, "PONG")
        self.assertTrue(vd._ping_supported)


class CommandDispatchTests(VoiceHarness):
    """Threshold gate, global cooldown, and session isolation on dispatch."""

    def test_global_cooldown_blocks_alternating_commands(self):
        vd, cm = self.make_detector(cfg={"voice_confidence_threshold": 0.5})
        vd.start()
        fired = []
        vd.on_voice_command = lambda cmd, conf, txt=None, candidates=None: fired.append(cmd)
        p = vd.proc
        vd._handle_line(p, "VOICE_CMD:copy:0.90:copy that")
        # A DIFFERENT command inside the cooldown window is still suppressed
        # (alternating ambient phrases used to slip past a per-command check).
        vd._handle_line(p, "VOICE_CMD:paste:0.90:paste this")
        self.assertEqual(fired, ["copy"])
        # After the window, a different command fires.
        vd._last_trigger_at = time.monotonic() - 10.0
        vd._handle_line(p, "VOICE_CMD:paste:0.90:paste this")
        self.assertEqual(fired, ["copy", "paste"])
        # Stale session output never dispatches.
        vd._handle_line(mock.MagicMock(), "VOICE_CMD:undo:0.99:undo")
        self.assertEqual(fired, ["copy", "paste"])

    def test_below_threshold_and_debug_lines_are_never_dispatched(self):
        vd, cm = self.make_detector(cfg={"voice_confidence_threshold": 0.9})
        vd.start()
        fired = []
        vd.on_voice_command = lambda *args: fired.append(args)
        vd._handle_line(vd.proc, "VOICE_CMD:copy:0.80:copy")
        vd._handle_line(vd.proc, "DEBUG_LOW_CONF:copy:0.10:copy")
        vd._handle_line(vd.proc, "DEBUG_REJECTED:copy:0.05:copy")
        self.assertEqual(fired, [])

    def test_free_transcript_reaches_callback_without_local_command_mapping(self):
        vd, cm = self.make_detector(cfg={"voice_confidence_threshold": 0.5})
        vd.start()
        fired = []
        vd.on_voice_command = lambda *args: fired.append(args)

        vd._handle_line(vd.proc, "VOICE_TEXT:0.88:mamad desktop")

        self.assertEqual(fired, [("", 0.88, "mamad desktop", [])])

    def test_low_confidence_phrase_is_filtered_before_v1m(self):
        vd, cm = self.make_detector(cfg={"voice_confidence_threshold": 0.9})
        vd.start()
        fired = []
        vd.on_voice_command = lambda *args: fired.append(args)

        vd._handle_line(vd.proc, "VOICE_TEXT:0.20:mamad desktop")

        self.assertEqual(fired, [])

    def test_v1m_receives_low_confidence_transcript_and_alternates(self):
        import base64

        vd, cm = self.make_detector(cfg={
            "voice_confidence_threshold": 0.9,
            "enable_v1m_verification": True,
        })
        vd.start()
        fired = []
        vd.on_voice_command = lambda *args: fired.append(args)
        payload = "0.20\x1elaptop copy\x1f0.18\x1elap top copy"
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        vd._handle_line(vd.proc, f"VOICE_TEXT2:0.20:{encoded}")

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "copy")
        self.assertEqual(fired[0][1], 0.20)
        self.assertEqual(fired[0][3][1]["text"], "lap top copy")

    def test_v1m_receives_windows_rejected_text_as_unclassified_evidence(self):
        vd, cm = self.make_detector(cfg={"enable_v1m_verification": True})
        vd.start()
        fired = []
        vd.on_voice_command = lambda *args: fired.append(args)

        vd._handle_line(vd.proc, "DEBUG_REJECTED_TEXT:0.24:laptop copy")

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "")  # no unsupported local command hint
        self.assertEqual(fired[0][1], 0.24)
        self.assertEqual(fired[0][2], "laptop copy")
        self.assertEqual(fired[0][3][0]["confidence"], 0.24)

    def test_alternate_transcripts_are_delivered_to_callback(self):
        import base64

        vd, cm = self.make_detector()
        vd.start()
        fired = []
        vd.on_voice_command = lambda *args: fired.append(args)
        payload = "0.82\x1elot of hockey\x1f0.72\x1eLaptop Copy"
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        vd._handle_line(vd.proc, f"VOICE_TEXT2:0.82:{encoded}")

        self.assertEqual(fired[0][2], "lot of hockey")
        self.assertEqual(fired[0][3][1]["text"], "Laptop Copy")

    def test_disabled_voice_never_dispatches(self):
        vd, cm = self.make_detector(cfg={"enable_voice_commands": False})
        # Force a session while enabled, then disable externally to prove the
        # listener stops the helper AND the dispatch-side gate holds.
        cm.set("enable_voice_commands", True)
        vd.start()
        fired = []
        vd.on_voice_command = lambda *args: fired.append(args)
        cm.set("enable_voice_commands", False)  # listener stops the helper...
        vd._handle_line(vd.proc, "VOICE_CMD:copy:0.90:copy")  # proc now None
        self.assertEqual(fired, [])
        self.assertFalse(vd.is_running)


class ShutdownTests(VoiceHarness):

    def test_stop_sends_quit_closes_stdin_and_cleans_up(self):
        vd, cm = self.make_detector()
        vd.start()
        proc = vd.proc
        vd.stop()
        self.assertFalse(vd.is_running)
        self.assertIsNone(vd.proc)
        self.assertEqual(vd.state, "stopped")
        proc.stdin.write.assert_called_with("quit\n")
        proc.stdin.close.assert_called_once()  # EOF fallback, no orphans
        proc.wait.assert_called()              # graceful path preferred

    def test_stop_without_start_is_safe_and_idempotent(self):
        vd, cm = self.make_detector()
        vd.stop()
        vd.stop()
        self.assertEqual(vd.state, "stopped")
        self.assertIsNone(vd.proc)


class VoiceProtocolTests(unittest.TestCase):

    def test_parse_valid_line_round_trips(self):
        got = VoiceDetector._parse_protocol_line("VOICE_CMD:copy:0.91:copy that")
        self.assertEqual(got, ("VOICE_CMD", "copy", 0.91, "copy that", []))

    def test_parse_free_transcript_without_a_local_command(self):
        got = VoiceDetector._parse_protocol_line("VOICE_TEXT:0.88:mamad desktop: please")
        self.assertEqual(got, ("VOICE_TEXT", "", 0.88, "mamad desktop: please", []))

    def test_parse_free_transcript_alternates(self):
        import base64

        payload = "0.31\x1elot of hockey\x1f0.72\x1eLaptop Copy"
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        got = VoiceDetector._parse_protocol_line(f"VOICE_TEXT2:0.31:{encoded}")

        self.assertEqual(got[0:4], ("VOICE_TEXT", "", 0.31, "lot of hockey"))
        self.assertEqual(got[4], [
            {"text": "lot of hockey", "confidence": 0.31},
            {"text": "Laptop Copy", "confidence": 0.72},
        ])

    def test_constrained_phrase_provides_a_v1m_command_hint(self):
        import base64

        payload = "0.756\x1elaptop copy\x1f0.658\x1elap top copy"
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        got = VoiceDetector._parse_protocol_line(f"VOICE_TEXT2:0.756:{encoded}")

        self.assertEqual(got[0:4], ("VOICE_TEXT", "copy", 0.756, "laptop copy"))

    def test_parse_rejects_unknown_command_and_bad_confidence(self):
        self.assertIsNone(VoiceDetector._parse_protocol_line("VOICE_CMD:evil:0.9:x"))
        self.assertIsNone(VoiceDetector._parse_protocol_line("VOICE_CMD:copy:1.4:x"))
        self.assertIsNone(VoiceDetector._parse_protocol_line("VOICE_CMD:copy:abc:x"))
        self.assertIsNone(VoiceDetector._parse_protocol_line("garbage"))

    def test_every_emitted_command_has_a_mapping_slot(self):
        """Binding integrity: the helper's allow-list and the default
        voice_actions keys must match exactly, so no recognized phrase can
        ever be unmapped out of the box."""
        self.assertEqual(VOICE_COMMANDS, set(DEFAULT_CONFIG["voice_actions"]))
        # And every default maps to an action the sanitizer accepts.
        from config_manager import ALLOWED_ACTIONS
        self.assertTrue(
            set(DEFAULT_CONFIG["voice_actions"].values()) <= ALLOWED_ACTIONS
        )


class ToolingCleanupTests(unittest.TestCase):
    """Calibration/validation scripts must also close stdin (EOF fallback)."""

    @staticmethod
    def _live_fake_proc():
        proc = mock.MagicMock()
        proc.poll.return_value = None
        proc.stdin = mock.MagicMock()
        proc.stdin.closed = False
        return proc

    def test_voice_calibrate_close_reaches_stdin_eof(self):
        vc = voice_calibrate.VoiceCalibrator.__new__(voice_calibrate.VoiceCalibrator)
        proc = self._live_fake_proc()
        vc._safe_close_proc(proc)
        proc.stdin.write.assert_called_with("quit\n")
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    def test_voice_validate_stop_helper_closes_stdin(self):
        proc = self._live_fake_proc()
        voice_validate.stop_helper(proc)
        proc.stdin.write.assert_called_with("quit\n")
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    def test_voice_calibrate_detects_running_helper(self):
        # Force the detection seam so the test is deterministic on any machine.
        vc = voice_calibrate.VoiceCalibrator.__new__(voice_calibrate.VoiceCalibrator)
        with mock.patch.object(
            voice_calibrate.VoiceCalibrator, "_helper_already_running",
            staticmethod(lambda: True),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                vc._start_ready_recognizer(timeout_s=0.1)
        self.assertIn("already running", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
