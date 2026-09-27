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
        import numpy as np
        import tempfile
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


if __name__ == "__main__":
    unittest.main()
