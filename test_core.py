import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from action_executor import (
    ActionExecutor,
    resolve_gesture_action,
    resolve_voice_action,
)
from audio_engine import AudioEngine
from config_manager import DEFAULT_CONFIG, ConfigManager
from tap_detector import TapDetector
from voice_detector import VoiceDetector


class _FakeGesture:
    def __init__(self, count, side="center"):
        self.count = count
        self.side = side


class ActionBindingTests(unittest.TestCase):
    """The knock -> action mapping must always come from live config."""

    def _config(self, tmp, actions):
        cfg = ConfigManager(tmp)
        cfg.set("actions", {**DEFAULT_CONFIG["actions"], **actions})
        return cfg

    def test_single_and_triple_follow_configured_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(tmp, {"single_tap": "paste", "triple_tap": "lock"})
            self.assertEqual(resolve_gesture_action(cfg, _FakeGesture(1)), "paste")
            self.assertEqual(resolve_gesture_action(cfg, _FakeGesture(3)), "lock")

    def test_remapping_takes_effect_without_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigManager(tmp)
            self.assertEqual(
                resolve_gesture_action(cfg, _FakeGesture(1)),
                DEFAULT_CONFIG["actions"]["single_tap"],
            )
            cfg.set("actions", {**cfg.get("actions"), "single_tap": "screenshot"})
            self.assertEqual(resolve_gesture_action(cfg, _FakeGesture(1)), "screenshot")

    def test_side_double_tap_overrides_then_falls_back_to_double(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(
                tmp,
                {"double_tap": "paste", "left_double_tap": "screenshot",
                 "right_double_tap": "none"},
            )
            self.assertEqual(
                resolve_gesture_action(cfg, _FakeGesture(2, "left")), "screenshot"
            )
            self.assertEqual(
                resolve_gesture_action(cfg, _FakeGesture(2, "right")), "none"
            )
            # Center / no side mapping -> the general double-knock choice.
            self.assertEqual(
                resolve_gesture_action(cfg, _FakeGesture(2, "center")), "paste"
            )

    def test_voice_mapping_uses_live_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigManager(tmp)
            cfg.set(
                "voice_actions",
                {**DEFAULT_CONFIG["voice_actions"], "copy": "undo"},
            )
            self.assertEqual(resolve_voice_action(cfg, "COPY"), "undo")
            self.assertIsNone(resolve_voice_action(cfg, "unknown"))


class ConfigSyncTests(unittest.TestCase):
    """config.json is shared by several processes; they must stay in sync and
    must never clobber each other's edits."""

    def test_other_process_sees_written_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ConfigManager(tmp)
            reader = ConfigManager(tmp)
            writer.set("actions", {**DEFAULT_CONFIG["actions"], "single_tap": "lock"})
            reader.reload_if_changed(force=True)
            self.assertEqual(reader.get("actions")["single_tap"], "lock")

    def test_get_picks_up_external_write_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ConfigManager(tmp)
            reader = ConfigManager(tmp)
            writer.set("sensitivity", 0.55)
            # Simulate the throttle window elapsing, then any ordinary get().
            reader._next_reload_check = 0.0
            self.assertEqual(reader.get("sensitivity"), 0.55)

    def test_debounced_save_flushes_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ConfigManager(tmp)
            reader = ConfigManager(tmp)
            writer.set("sensitivity", 0.70, flush=False)
            reader.reload_if_changed(force=True)
            self.assertNotEqual(reader.get("sensitivity"), 0.70)  # not yet on disk
            self.assertTrue(writer.flush())
            reader.reload_if_changed(force=True)
            self.assertEqual(reader.get("sensitivity"), 0.70)

    def test_concurrent_edits_from_two_processes_merge_without_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = ConfigManager(tmp)
            b = ConfigManager(tmp)  # both hold the same starting snapshot
            a.set("sensitivity", 0.77)            # process A edits key 1
            b.set(
                "actions", {**DEFAULT_CONFIG["actions"], "single_tap": "paste"}
            )                                      # process B edits key 2
            final = ConfigManager(tmp)
            self.assertEqual(final.get("sensitivity"), 0.77)          # A survives
            self.assertEqual(final.get("actions")["single_tap"], "paste")  # B survives

    def test_listeners_fire_on_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigManager(tmp)
            seen = []
            cfg.add_listener(lambda: seen.append(1))
            cfg.set("sensitivity", 0.66)
            cfg.add_listener(lambda: seen.append(2))  # added after the change
            cfg.set("sensitivity", 0.68)
            self.assertGreaterEqual(len(seen), 1)
            self.assertEqual(cfg.get("sensitivity"), 0.68)

    def test_gui_save_action_pattern_round_trips(self):
        # Mirrors gui._save_action: build a new dict, set it, verify persistence.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigManager(tmp)
            actions = dict(cfg.get("actions"))
            actions["double_tap"] = "lock"
            cfg.set("actions", actions)
            fresh = ConfigManager(tmp)
            self.assertEqual(fresh.get("actions")["double_tap"], "lock")


class ConfigManagerTests(unittest.TestCase):
    def test_invalid_values_are_sanitized_and_unsafe_actions_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "sensitivity": "99",
                        "suppress_typing": "false",
                        "actions": {"single_tap": "cmd:whoami"},
                        "voice_actions": {"copy": "cmd:calc"},
                        "unknown_key": "ignored",
                    }
                ),
                encoding="utf-8",
            )
            config = ConfigManager(tmp)
            self.assertEqual(config.get("sensitivity"), 1.0)
            self.assertFalse(config.get("suppress_typing"))
            self.assertEqual(config.get("actions")["single_tap"], DEFAULT_CONFIG["actions"]["single_tap"])
            self.assertEqual(config.get("voice_actions")["copy"], "copy")
            self.assertNotIn("unknown_key", config.config)

    def test_atomic_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ConfigManager(tmp)
            self.assertTrue(config.update({"sensitivity": 0.77, "voice_require_wake_word": False}))
            loaded = ConfigManager(tmp)
            self.assertEqual(loaded.get("sensitivity"), 0.77)
            self.assertFalse(loaded.get("voice_require_wake_word"))


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = ConfigManager(self.tmp.name)
        self.gestures = []
        self.detector = TapDetector(
            self.config,
            on_gesture_detected=self.gestures.append,
            monitor_keyboard=False,
        )

    def tearDown(self):
        self.detector.stop()
        self.tmp.cleanup()

    @staticmethod
    def _silence():
        mono = np.zeros(256, dtype=np.float32)
        return mono, np.zeros((256, 2), dtype=np.float32)

    def test_synthetic_voice_is_rejected(self):
        sr = 48000
        t = np.arange(1024, dtype=np.float32) / sr
        voice = ((np.sin(2 * np.pi * 160 * t) + 0.4 * np.sin(2 * np.pi * 1800 * t)) * 0.4).astype(np.float32)
        for i in range(4):
            block = voice[i * 256 : (i + 1) * 256]
            self.detector.process_chunk(block, block, block, block, np.column_stack([block, block]))
        self.assertFalse(self.detector.finalize_pending())
        self.assertEqual(self.gestures, [])
        diagnostics = self.detector.get_diagnostics()
        self.assertTrue(diagnostics["counts"])
        self.assertNotIn("accepted_pulse", diagnostics["counts"])

    def test_isolated_left_knock_is_detected(self):
        self.detector.reset()
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:30] = 0.35
        raw = np.zeros((256, 2), dtype=np.float32)
        raw[10:30, 0] = 0.35
        raw[10:30, 1] = 0.08
        self.detector.process_chunk(impulse, impulse * 0.2, silence, silence, raw)
        self.assertTrue(self.detector.finalize_pending())
        self.assertEqual(len(self.gestures), 1)
        self.assertEqual(self.gestures[0].count, 1)
        self.assertEqual(self.gestures[0].side, "left")
        self.assertEqual(
            self.detector.get_diagnostics()["counts"].get("accepted_pulse"), 1
        )

    def test_three_separate_impulses_form_one_triple_gesture(self):
        self.config.set("refractory_period_ms", 20)
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:24] = np.linspace(0.05, 0.4, 14, dtype=np.float32)
        raw = np.column_stack([impulse, impulse * 0.25])
        for index in range(3):
            # This test isolates grouping; refractory behavior is covered by
            # the dedicated rebound/cooldown test below.
            self.detector.last_tap_time = float("-inf")
            self.detector.process_chunk(
                impulse, impulse * 0.25, silence, silence, raw
            )
            if index < 2:
                for _ in range(20):
                    self.detector.process_chunk(
                        silence, silence, silence, silence,
                        np.zeros((256, 2), dtype=np.float32),
                    )
        self.assertTrue(self.detector.finalize_pending())
        self.assertEqual([gesture.count for gesture in self.gestures], [3])

    def test_recent_keyboard_event_suppresses_candidate(self):
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:24] = 0.4
        raw = np.column_stack([impulse, impulse])
        self.detector.last_key_time = 10.0
        with mock.patch("tap_detector.time.monotonic", return_value=10.1):
            self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.assertFalse(self.detector.finalize_pending())
        self.assertEqual(
            self.detector.get_diagnostics()["counts"].get("typing_suppressed"), 1
        )

    def test_sustained_sound_revokes_pending_impulse(self):
        self.config.update({
            "max_sustained_ms": 30,
            "sustained_revoke_ms": 140,
            "sustained_recovery_ms": 120,
            "voice_like_hold_ms": 900,
        })
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:24] = 0.4
        raw = np.column_stack([impulse, impulse])
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.assertEqual(self.detector.current_taps, 1)

        sustained = np.full(256, 0.08, dtype=np.float32)
        for _ in range(30):
            self.detector.process_chunk(
                sustained, sustained, silence, silence,
                np.column_stack([sustained, sustained]),
            )
        self.assertEqual(self.detector.current_taps, 0)
        self.assertFalse(self.detector.finalize_pending())
        self.assertEqual(self.gestures, [])
        self.assertGreater(
            self.detector.get_diagnostics()["counts"].get(
                "sustained_recovery", 0
            ),
            0,
        )
        # Even after the waveform briefly falls below threshold, the confirmed
        # voice-like burst keeps the following syllable from becoming a tap.
        self.detector.process_chunk(
            silence, silence, silence, silence,
            np.zeros((256, 2), dtype=np.float32),
        )
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.assertEqual(
            self.detector.get_diagnostics()["last_candidate"]["result"],
            "voice_like_suppression",
        )
        short_deadline = time.monotonic() + 0.05
        self.detector.voice_like_suppression_until = short_deadline
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.assertGreater(
            self.detector.voice_like_suppression_until,
            short_deadline + 0.50,
        )

    def test_short_chassis_ring_does_not_revoke_pending_tap(self):
        self.config.update({
            "max_sustained_ms": 30,
            "sustained_revoke_ms": 140,
            "sustained_recovery_ms": 120,
            "voice_like_hold_ms": 900,
        })
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:24] = 0.4
        raw = np.column_stack([impulse, impulse])
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        ring = np.full(256, 0.08, dtype=np.float32)
        for _ in range(12):
            self.detector.process_chunk(
                ring, ring, silence, silence, np.column_stack([ring, ring])
            )
        self.assertEqual(self.detector.current_taps, 1)
        self.assertTrue(self.detector.finalize_pending())
        self.assertEqual(len(self.gestures), 1)

    def test_post_gesture_cooldown_blocks_chassis_rebound(self):
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:24] = 0.4
        raw = np.column_stack([impulse, impulse])
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.assertTrue(self.detector.finalize_pending())
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.assertFalse(self.detector.finalize_pending())
        self.assertEqual(len(self.gestures), 1)
        self.assertEqual(
            self.detector.get_diagnostics()["counts"].get(
                "post_gesture_cooldown"
            ),
            1,
        )

    def test_dispatch_refuses_pending_group_during_confirmed_voice(self):
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:24] = 0.4
        raw = np.column_stack([impulse, impulse])
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.detector.voice_like_suppression_until = time.monotonic() + 1.0
        self.assertTrue(self.detector.finalize_pending())
        self.assertEqual(self.gestures, [])
        self.assertEqual(
            self.detector.get_diagnostics()["counts"].get(
                "voice_like_group_cancelled"
            ),
            1,
        )

    def test_crowded_mode_rejects_center_airborne_impulse(self):
        self.config.update({"crowded_mode": True, "detect_sides": True})
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:24] = 0.4
        raw = np.column_stack([impulse, impulse])
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.assertTrue(self.detector.finalize_pending())
        self.assertEqual(self.gestures, [])
        self.assertEqual(
            self.detector.get_diagnostics()["counts"].get(
                "spatially_ambiguous"
            ),
            1,
        )

    def test_crowded_mode_accepts_asymmetric_chassis_tap(self):
        self.config.update({"crowded_mode": True, "detect_sides": True})
        silence, _ = self._silence()
        left = silence.copy()
        left[10:24] = 0.4
        right = left * 0.2
        raw = np.column_stack([left, right])
        self.detector.process_chunk(left, right, silence, silence, raw)
        self.assertTrue(self.detector.finalize_pending())
        self.assertEqual(len(self.gestures), 1)
        self.assertEqual(self.gestures[0].side, "left")

    def test_crowded_mode_does_not_deadlock_on_mono_microphone(self):
        # Mono capture can never localize a side; rejecting every "center"
        # gesture would disable knock detection entirely.
        self.config.update({"crowded_mode": True, "detect_sides": True})
        silence, _ = self._silence()
        impulse = silence.copy()
        impulse[10:24] = 0.4
        raw = np.zeros((256, 1), dtype=np.float32)  # single-channel device
        raw[10:24, 0] = 0.4
        self.detector.process_chunk(impulse, impulse, silence, silence, raw)
        self.assertTrue(self.detector.finalize_pending())
        self.assertEqual(len(self.gestures), 1)
        self.assertEqual(self.gestures[0].count, 1)


class AudioEngineTests(unittest.TestCase):
    def test_worker_processing_updates_history_and_callback(self):
        chunks = []
        engine = AudioEngine(
            on_audio_chunk=lambda *args: chunks.append(args),
            warmup_seconds=0,
        )
        raw = np.zeros((256, 2), dtype=np.float32)
        raw[10:30, 0] = 0.35
        engine._process_chunk(raw)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(engine.get_waveform_history().shape, (1024,))

    def test_stream_warmup_primes_filters_without_emitting_candidate(self):
        chunks = []
        engine = AudioEngine(
            on_audio_chunk=lambda *args: chunks.append(args),
            warmup_seconds=0,
        )
        engine._warmup_frames_remaining = 300
        raw = np.zeros((256, 2), dtype=np.float32)
        engine._process_chunk(raw)
        self.assertEqual(chunks, [])
        engine._process_chunk(raw)
        self.assertEqual(chunks, [])
        engine._process_chunk(raw)
        self.assertEqual(len(chunks), 1)

    def test_callback_only_enqueues(self):
        engine = AudioEngine()
        engine.is_running = True
        raw = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(raw, 256, None, None)
        queued = engine._audio_queue.get_nowait()
        self.assertEqual(queued.shape, (256, 1))
        engine.is_running = False

    def test_start_falls_back_to_next_microphone(self):
        engine = AudioEngine(warmup_seconds=0)
        stream = mock.MagicMock()
        dev = {"name": "Working microphone", "hostapi": 0}

        def open_device(index):
            if index == 1:
                raise RuntimeError("device busy")
            return stream, dev, 48000, 1, 256

        with (
            mock.patch.object(engine, "_candidate_devices", return_value=[1, 2]),
            mock.patch.object(engine, "_open_stream_for_device", side_effect=open_device),
            mock.patch.object(engine, "_device_signature", return_value="working|wasapi"),
        ):
            self.assertTrue(engine.start())
            self.assertEqual(engine.active_device_index, 2)
            stream.start.assert_called_once()
            engine.stop()


class SecurityTests(unittest.TestCase):
    def test_shell_command_action_is_rejected(self):
        executor = ActionExecutor()
        with mock.patch("action_executor.subprocess.Popen") as popen:
            self.assertFalse(executor.trigger("cmd:calc.exe"))
            popen.assert_not_called()

    def test_voice_protocol_rejects_unknown_commands(self):
        self.assertIsNone(VoiceDetector._parse_protocol_line("VOICE_CMD:cmd:calc:0.99:bad"))
        self.assertIsNone(VoiceDetector._parse_protocol_line("VOICE_CMD:unknown:0.99:bad"))
        parsed = VoiceDetector._parse_protocol_line("VOICE_CMD:copy:0.91:laptop copy")
        self.assertEqual(parsed, ("VOICE_CMD", "copy", 0.91, "laptop copy", []))
        rejected = VoiceDetector._parse_protocol_line("DEBUG_REJECTED:copy:0.31:laptop copy")
        self.assertEqual(rejected, ("DEBUG_REJECTED", "copy", 0.31, "laptop copy", []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
