"""Regression coverage for settings, dispatch, packaging, and audio lifecycle."""

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from action_executor import ActionExecutor
from config_manager import DEFAULT_CONFIG, ConfigManager
from gui import MainWindow
from oghdeii.voice.v1m_verifier import V1MVoiceVerifier
from tap_detector import TapDetector
from whisper_voice_detector import WhisperVoiceDetector


class SettingsRegressionTests(unittest.TestCase):
    def test_simultaneous_process_writes_preserve_both_edits(self):
        script = """
import sys, time
from pathlib import Path
from config_manager import ConfigManager
folder, key, value = sys.argv[1:]
config = ConfigManager(folder)
config.set(key, int(value), flush=False)
# Prevent the debounce timer from writing before both writers are ready.
with config._io_lock:
    config._cancel_debounce_locked()
Path(folder, key + '.ready').touch()
deadline = time.monotonic() + 10
while not Path(folder, 'go').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('test barrier timed out')
    time.sleep(0.01)
assert config.flush(), config.last_error
"""
        import time
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            processes = []
            try:
                for key, value in (("grouping_window_ms", 600), ("typing_cooldown_ms", 900)):
                    processes.append(subprocess.Popen(
                        [sys.executable, "-c", script, tmp, key, str(value)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    ))
                deadline = time.monotonic() + 10
                while len(list(Path(tmp).glob("*.ready"))) < 2:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                Path(tmp, "go").touch()
                for process in processes:
                    output, error = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, (output, error))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()
            saved = ConfigManager(tmp)
            self.assertEqual(saved.get("grouping_window_ms"), 600)
            self.assertEqual(saved.get("typing_cooldown_ms"), 900)

    def test_nonfinite_integer_settings_use_defaults(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                config = ConfigManager._merge_defaults({
                    "grouping_window_ms": value, "selected_input_device": value,
                })
                self.assertEqual(config["grouping_window_ms"], DEFAULT_CONFIG["grouping_window_ms"])
                self.assertIsNone(config["selected_input_device"])

    def test_deferred_updates_preserve_cross_setting_invariants(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ConfigManager(tmp)
            self.addCleanup(config.flush)
            with mock.patch.object(config, "_schedule_save"):
                config.set("detect_sides", False, flush=False)
                config.set("crowded_mode", True, flush=False)
                self.assertTrue(config.get("detect_sides"))
                config.update({"detect_sides": False}, flush=False)
                self.assertTrue(config.get("detect_sides"))
            config.flush()

    def test_repeated_external_edits_survive_dirty_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = ConfigManager(tmp)
            external = ConfigManager(tmp)
            with mock.patch.object(local, "_schedule_save"):
                local.set("sensitivity", 0.7, flush=False)
                external.set("grouping_window_ms", 500)
                local.reload_if_changed(force=True)
                external.set("grouping_window_ms", 600)
                local.flush()
            saved = ConfigManager(tmp)
            self.assertEqual(saved.get("sensitivity"), 0.7)
            self.assertEqual(saved.get("grouping_window_ms"), 600)


class ActionFailureTests(unittest.TestCase):
    def test_cli_help_works_with_legacy_redirected_encoding(self):
        from pathlib import Path

        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "main.py"), "--help"],
            env={**os.environ, "PYTHONIOENCODING": "ascii"},
            capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertIn("عقده ای", result.stdout.decode("utf-8"))

    def test_failed_linux_actions_never_report_success_or_beep(self):
        executor = ActionExecutor()
        with (
            mock.patch("action_executor.IS_WINDOWS", False),
            mock.patch.object(executor, "_execute_linux_hotkey", return_value=False),
            mock.patch("action_executor.shutil.which", return_value=None),
            mock.patch("action_executor.webbrowser.open", return_value=False),
            mock.patch.object(executor, "play_feedback_beep") as beep,
        ):
            for action in ("copy", "paste", "undo", "redo", "select_all", "screenshot",
                           "task_view", "show_desktop", "close_window", "switch_window", "open_browser"):
                with self.subTest(action=action):
                    self.assertFalse(executor.trigger(action))
            beep.assert_not_called()


class CloudNumericValidationTests(unittest.TestCase):
    def test_invalid_risk_and_intent_values_are_rejected(self):
        for field, values in (
            ("execution_risk", [float("nan"), float("inf"), -1, 5]),
            ("is_valid_command", [float("nan"), float("inf"), -0.1, 1.1]),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    answers = {
                        "is_valid_command": {"noul": 0.95},
                        "action": {"choice": "copy"},
                        "execution_risk": {"score": 1.0},
                    }
                    answers[field]["score" if field == "execution_risk" else "noul"] = value
                    with self.assertRaises(ValueError):
                        V1MVoiceVerifier._parse_response(SimpleNamespace(answers=answers))


class AudioLifecycleRegressionTests(unittest.TestCase):
    def test_pause_cancels_gesture_and_rejects_new_audio_until_start(self):
        config = ConfigManager._merge_defaults({})
        output = []
        detector = TapDetector(config, on_gesture_detected=output.append, monitor_keyboard=False)
        self.addCleanup(detector.stop)
        low = np.zeros(256, dtype=np.float32)
        low[10:24] = np.linspace(0.05, 0.4, 14, dtype=np.float32)
        high = np.zeros_like(low)
        raw = np.column_stack([low, low * 0.2])
        detector.process_chunk(low, low * 0.2, high, high, raw)
        detector.pause()
        detector.process_chunk(low, low * 0.2, high, high, raw)
        self.assertFalse(detector.finalize_pending())
        self.assertEqual(output, [])
        detector.reset()
        detector.start()
        detector.process_chunk(low, low * 0.2, high, high, raw)
        self.assertTrue(detector.finalize_pending())
        self.assertEqual(len(output), 1)

    def test_queued_audio_keeps_capture_rate_after_microphone_change(self):
        config = mock.Mock()
        config.get.side_effect = DEFAULT_CONFIG.get
        detector = WhisperVoiceDetector(config)
        self.addCleanup(detector.cleanup)
        detector._input_rate = 16000  # new microphone
        captured = np.sin(np.arange(48000, dtype=np.float32) * 0.02)
        prepared = detector._prepare_audio(captured, input_rate=48000)
        self.assertEqual(len(prepared), 16000)


class GuiDispatchRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = ConfigManager(self.tmp.name)
        self.audio = mock.Mock(is_running=False, last_start_error="test microphone unavailable")
        self.audio.start.return_value = False
        self.voice = mock.Mock(is_running=False)
        self.executor = mock.Mock()
        verifier = mock.Mock()
        verifier.describe.return_value = {"sdk_installed": False, "endpoint": "test endpoint"}
        with (
            mock.patch("gui.AudioEngine.get_input_devices", return_value=[]),
            mock.patch("gui.AudioEngine.resolve_device_index", return_value=None),
            mock.patch.object(MainWindow, "init_tray"),
        ):
            self.window = MainWindow(self.config, self.audio, mock.Mock(), self.executor,
                                     voice_detector=self.voice, voice_verifier=verifier)
        self.window.tray_icon = mock.Mock()

    def tearDown(self):
        self.window.clean_quit()
        self.window._action_pool.shutdown(wait=True, cancel_futures=True)
        self.window.deleteLater()
        self.app.processEvents()
        self.tmp.cleanup()

    def test_pause_blocks_queued_tap_and_voice_results(self):
        generation = self.window._dispatch_generation
        self.window.toggle_monitoring()
        self.window.on_gesture_received(SimpleNamespace(count=1, side="center"), generation)
        self.config.set("active_mode", "voice")
        self.window.on_voice_received("copy", 0.9, "laptop copy", None, generation)
        self.executor.trigger.assert_not_called()
        self.window.tap_detector.pause.assert_called()

    def test_voice_verification_finishing_after_pause_is_discarded(self):
        self.config.set("active_mode", "voice")

        def verify(*args, **kwargs):
            self.window.toggle_monitoring()
            return SimpleNamespace(allow=True, command="copy")

        self.window.voice_verifier.verify.side_effect = verify
        with mock.patch.object(self.window.bridge, "voice_signal") as signal:
            self.window._on_voice_captured("copy", 0.9, "laptop copy")
            signal.emit.assert_not_called()

    def test_resume_does_not_revive_an_old_queued_result(self):
        self.config.set("active_mode", "voice")
        old_generation = self.window._dispatch_generation
        self.window.toggle_monitoring()
        self.window.toggle_monitoring()
        with mock.patch.object(self.window, "_dispatch_action") as dispatch:
            self.window.on_voice_received("copy", 0.9, "laptop copy", None, old_generation)
            dispatch.assert_not_called()

    def test_waiting_action_is_cancelled_when_paused(self):
        release = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        self.window._action_pool.shutdown()
        self.window._action_pool = pool
        pool.submit(release.wait, 5)
        try:
            self.window._dispatch_action("copy", "Copy")
            self.window.toggle_monitoring()
        finally:
            release.set()
            pool.shutdown(wait=True)
        self.executor.trigger.assert_not_called()

    def test_failed_calibration_launch_restores_monitoring(self):
        with (
            mock.patch("gui.sys.platform", "win32"),
            mock.patch("subprocess.Popen", side_effect=OSError("cannot launch")),
            mock.patch.object(self.window, "switch_mode") as resume,
        ):
            self.window.run_tap_calibration()
            self.assertFalse(self.window._calibrating)
            resume.assert_called_once_with("knock")

    def test_quit_cleans_current_voice_backend_and_unregisters_listener(self):
        replacement = mock.Mock()
        self.window.voice_detector = replacement
        self.window.clean_quit()
        replacement.cleanup.assert_called_once()
        self.assertNotIn(self.window._config_listener, self.config._listeners)


if __name__ == "__main__":
    unittest.main()
