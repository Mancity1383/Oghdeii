"""Focused tests for transcript cleanup before command verification."""

import unittest

from whisper_voice_detector import (
    command_from_text,
    has_wake_word,
    looks_like_repetition_hallucination,
    normalize_transcript,
)


class TranscriptClassificationTests(unittest.TestCase):
    def test_normalizes_whitespace_and_punctuation(self):
        self.assertEqual(normalize_transcript("  Laptop,   Copy! "), "laptop copy")

    def test_maps_wake_word_command(self):
        command, score = command_from_text("Laptop copy")
        self.assertEqual(command, "copy")
        self.assertEqual(score, 1.0)

    def test_maps_supported_variant(self):
        command, _score = command_from_text("hey laptop show the desktop")
        self.assertEqual(command, "show_desktop")

    def test_rejects_unrelated_phrase_as_command_hint(self):
        command, _score = command_from_text("thanks for watching")
        self.assertEqual(command, "")

    def test_detects_wake_word_only_at_start(self):
        self.assertTrue(has_wake_word("Laptop paste"))
        self.assertFalse(has_wake_word("please laptop paste"))

    def test_rejects_decoder_repetition(self):
        self.assertTrue(looks_like_repetition_hallucination(
            "laptop ticket screen screen screen screen screen"
        ))

    def test_keeps_short_natural_command(self):
        self.assertFalse(looks_like_repetition_hallucination("laptop show desktop"))

    def test_maps_lock_and_volume_variants(self):
        cmd, score = command_from_text("laptop lock")
        self.assertEqual(cmd, "lock")
        self.assertEqual(score, 1.0)

        cmd, _ = command_from_text("laptop turn volume up")
        self.assertEqual(cmd, "volume_up")

        cmd, _ = command_from_text("laptop turn volume down")
        self.assertEqual(cmd, "volume_down")


class WhisperLifecycleTests(unittest.TestCase):
    def test_cleanup_unregisters_config_listener(self):
        import tempfile

        from config_manager import ConfigManager
        from whisper_voice_detector import WhisperVoiceDetector

        with tempfile.TemporaryDirectory() as tmp:
            cm = ConfigManager(tmp)
            detector = WhisperVoiceDetector(cm)
            self.assertIn(detector._on_config_changed, cm._listeners)
            detector.cleanup()
            self.assertNotIn(detector._on_config_changed, cm._listeners)

    def test_voice_detector_cleanup_unregisters_listener(self):
        import tempfile

        from config_manager import ConfigManager
        from voice_detector import VoiceDetector

        with tempfile.TemporaryDirectory() as tmp:
            cm = ConfigManager(tmp)
            detector = VoiceDetector(cm)
            self.assertIn(detector._on_config_changed, cm._listeners)
            detector.cleanup()
            self.assertNotIn(detector._on_config_changed, cm._listeners)

    def test_prepare_audio_resampling(self):
        import tempfile

        import numpy as np

        from config_manager import ConfigManager
        from whisper_voice_detector import WhisperVoiceDetector

        with tempfile.TemporaryDirectory() as tmp:
            cm = ConfigManager(tmp)
            detector = WhisperVoiceDetector(cm)
            detector._input_rate = 48000
            detector._sample_rate = 16000
            # 48000 samples = 1 second at 48kHz
            audio_48k = np.sin(np.linspace(0, 100, 48000), dtype=np.float32)
            resampled = detector._prepare_audio(audio_48k)
            # Should be approximately 16000 samples (1 second at 16kHz)
            self.assertEqual(len(resampled), 16000)
            detector.cleanup()

    def test_silero_empty_or_truncated_result_retries_without_vad(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import Mock

        import numpy as np

        from config_manager import ConfigManager
        from whisper_voice_detector import WhisperVoiceDetector

        with tempfile.TemporaryDirectory() as tmp:
            cm = ConfigManager(tmp)
            detector = WhisperVoiceDetector(cm)
            detector.is_running = True
            output = []
            detector.on_voice_command = lambda *args: output.append(args)
            detected_command = SimpleNamespace(
                text=" laptop copy ", avg_logprob=-0.1, no_speech_prob=0.0
            )
            # Simulate Silero first dropping the utterance completely, then
            # returning only the wake word. The non-VAD retry recovers it.
            detector._model = Mock()
            detector._model.transcribe.side_effect = [
                (iter(()), None),
                (iter([detected_command]), None),
            ]

            detector._transcribe(np.ones(16000, dtype=np.float32) * 0.02)

            self.assertEqual(len(output), 1)
            self.assertEqual(output[0][0], "copy")
            self.assertEqual(detector._model.transcribe.call_count, 2)
            self.assertTrue(detector._model.transcribe.call_args_list[0].kwargs["vad_filter"])
            self.assertFalse(detector._model.transcribe.call_args_list[1].kwargs["vad_filter"])
            detector.cleanup()

    def test_silero_wake_word_only_retries_to_recover_action(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import Mock

        import numpy as np

        from config_manager import ConfigManager
        from whisper_voice_detector import WhisperVoiceDetector

        with tempfile.TemporaryDirectory() as tmp:
            detector = WhisperVoiceDetector(ConfigManager(tmp))
            detector.is_running = True
            output = []
            detector.on_voice_command = lambda *args: output.append(args)
            wake_only = SimpleNamespace(
                text=" laptop ", avg_logprob=-0.1, no_speech_prob=0.0
            )
            full_command = SimpleNamespace(
                text=" laptop copy ", avg_logprob=-0.1, no_speech_prob=0.0
            )
            detector._model = Mock()
            detector._model.transcribe.side_effect = [
                (iter([wake_only]), None),
                (iter([full_command]), None),
            ]

            detector._transcribe(np.ones(16000, dtype=np.float32) * 0.02)

            self.assertEqual(len(output), 1)
            self.assertEqual(output[0][0], "copy")
            self.assertEqual(detector._model.transcribe.call_count, 2)
            detector.cleanup()

    def test_input_stream_silent_device_falls_back_to_system_default(self):
        import tempfile
        from unittest.mock import Mock

        import numpy as np

        from config_manager import ConfigManager
        from whisper_voice_detector import WhisperVoiceDetector

        class Stream:
            def __init__(self, callback, device):
                self.callback = callback
                self.device = device
                self.active = False
                self.closed = False

            def start(self):
                if self.device is None:
                    self.callback(np.zeros((320, 1), dtype=np.float32), 320, None, None)
                    self.active = True

            def stop(self):
                self.active = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            cm = ConfigManager(tmp)
            cm.update({
                "selected_input_device": 42,
                "selected_input_device_signature": "inactive-device",
            })
            detector = WhisperVoiceDetector(cm)
            sd = Mock()
            sd.query_devices.side_effect = lambda *args, **kwargs: {
                "default_samplerate": 16000
            }
            opened = []

            def input_stream(**kwargs):
                stream = Stream(kwargs["callback"], kwargs.get("device"))
                opened.append(stream)
                return stream

            sd.InputStream.side_effect = input_stream
            stream = detector._open_stream(sd)

            self.assertIs(stream, opened[-1])
            self.assertTrue(detector._using_default_input_fallback)
            self.assertTrue(opened[0].closed)
            self.assertIsNone(stream.device)
            detector.cleanup()


if __name__ == "__main__":
    unittest.main()
