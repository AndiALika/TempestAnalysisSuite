"""
test_tempest_signal.py
=====================
Validation suite for ``tempest_signal`` (stdlib unittest only).
"""

import unittest
import numpy as np

import tempest_signal as tsig


def _spectrum_with_peaks(peak_freqs, fmax=200000.0, n=4000, floor=-80.0, peak=0.0):
    freqs = np.linspace(0.0, fmax, n)
    mag = np.full(n, floor)
    for pf in peak_freqs:
        i = int(np.argmin(np.abs(freqs - pf)))
        w = max(1, n // 400)
        lo, hi = max(0, i - w), min(n, i + w)
        x = np.arange(lo, hi)
        mag[lo:hi] = np.maximum(mag[lo:hi],
                                floor + (peak - floor) * np.exp(-((x - i) ** 2) / (2 * (w / 3) ** 2)))
    return freqs, mag


class TestDetectPeaks(unittest.TestCase):
    def test_finds_known_tones(self):
        want = [15625, 31250, 62500]
        f, m = _spectrum_with_peaks(want)
        peaks = tsig.detect_peaks(f, m, prominence_db=10, min_freq=1000)
        got = sorted(p[0] for p in peaks)
        for w in want:
            self.assertTrue(any(abs(g - w) < 200 for g in got), f"missed {w}")

    def test_limits_and_sorts(self):
        f, m = _spectrum_with_peaks([1e4, 2e4, 3e4, 4e4, 5e4])
        peaks = tsig.detect_peaks(f, m, prominence_db=6, max_peaks=3)
        self.assertLessEqual(len(peaks), 3)
        levels = [p[1] for p in peaks]
        self.assertEqual(levels, sorted(levels, reverse=True))


class TestHarmonicFamilies(unittest.TestCase):
    def test_detects_family(self):
        f0 = 15625.0
        fams = tsig.harmonic_families([f0, 2 * f0, 3 * f0, 4 * f0])
        self.assertTrue(fams)
        self.assertAlmostEqual(fams[0]["fundamental"], f0, delta=1.0)
        self.assertEqual(fams[0]["count"], 4)

    def test_no_family_for_random(self):
        fams = tsig.harmonic_families([1000.0, 1731.0, 4390.0], min_harmonics=3)
        self.assertEqual(fams, [])


class TestClassify(unittest.TestCase):
    def test_crt_wins_for_crt_peaks(self):
        ranked = tsig.classify_device([15625, 31250, 62500, 125000])
        self.assertEqual(ranked[0]["device"], "CRT Monitor")
        self.assertAlmostEqual(ranked[0]["score"], 1.0, delta=1e-6)

    def test_keyboard_identified(self):
        ranked = tsig.classify_device([12000, 24000, 48000])
        self.assertEqual(ranked[0]["device"], "Keyboard (PS/2)")

    def test_all_devices_scored(self):
        ranked = tsig.classify_device([])
        self.assertEqual(len(ranked), len(__import__("tempest_physics").DEVICES))
        self.assertTrue(all(d["score"] == 0.0 for d in ranked))


if __name__ == "__main__":
    unittest.main(verbosity=2)
