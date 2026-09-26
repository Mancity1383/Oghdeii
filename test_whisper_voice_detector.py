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


if __name__ == "__main__":
    unittest.main()
