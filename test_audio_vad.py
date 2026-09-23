import unittest

import numpy as np

from config_manager import ConfigManager
from tap_detector import TapDetector


class AcousticRejectionTests(unittest.TestCase):
    class Config:
        def __init__(self):
            self.values = ConfigManager._merge_defaults({})

        def get(self, key, default=None):
            return self.values.get(key, default)

    def setUp(self):
        self.gestures = []
        self.detector = TapDetector(
            self.Config(),
            on_gesture_detected=self.gestures.append,
            sample_rate=48000,
            monitor_keyboard=False,
        )

    def tearDown(self):
        self.detector.stop()

    def test_periodic_airborne_signal_is_rejected(self):
        t = np.arange(2048, dtype=np.float32) / 48000.0
        voice = ((np.sin(2 * np.pi * 160 * t) + 0.4 * np.sin(2 * np.pi * 1800 * t)) * 0.4).astype(np.float32)
        for start in range(0, len(voice), 256):
            block = voice[start : start + 256]
            self.detector.process_chunk(block, block, block, block, np.column_stack([block, block]))
        self.assertFalse(self.detector.finalize_pending())
        self.assertEqual(self.gestures, [])

    def test_short_impulse_is_accepted(self):
        low = np.zeros(256, dtype=np.float32)
        low[10:24] = np.linspace(0.05, 0.4, 14, dtype=np.float32)
        raw = np.column_stack([low, low * 0.2])
        high = np.zeros_like(low)
        self.detector.process_chunk(low, low * 0.2, high, high, raw)
        self.assertTrue(self.detector.finalize_pending())
        self.assertEqual(self.gestures[0].count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
