"""
test_tempest_video.py
====================
Validation suite for ``tempest_video`` (stdlib unittest only).
"""

import unittest
import numpy as np

import tempest_video as tv


class TestRasterRoundTrip(unittest.TestCase):
    def test_exact_reconstruction_no_highpass_no_noise(self):
        # With the correct timing and no distortion, folding recovers the image.
        rng = np.random.default_rng(0)
        img = rng.random((30, 40))
        sig, h_total, v_total = tv.raster_scan(img, h_blank=8, v_blank=5)
        em = tv.emanate(sig, noise=0.0, highpass=False)
        rec = tv.reconstruct(em, h_total, v_total)
        np.testing.assert_allclose(rec[:30, :40], img, atol=1e-9)

    def test_timing_totals(self):
        img = np.zeros((20, 32))
        _, h_total, v_total = tv.raster_scan(img, h_blank=16, v_blank=4)
        self.assertEqual(h_total, 48)
        self.assertEqual(v_total, 24)

    def test_wrong_line_length_shears(self):
        img = tv.demo_image("checker", 64, 48)
        sig, h_total, v_total = tv.raster_scan(img, h_blank=10, v_blank=6)
        em = tv.emanate(sig, noise=0.0, highpass=False)
        good = tv.reconstruct(em, h_total, v_total)[:48, :64]
        bad = tv.reconstruct(em, h_total + 3, v_total)[:48, :64]
        self.assertLess(np.corrcoef(good.ravel(), img.ravel())[0, 1], 1.01)
        # a mistuned line length must reconstruct worse than the correct one
        self.assertGreater(
            np.corrcoef(good.ravel(), img.ravel())[0, 1],
            np.corrcoef(bad.ravel(), img.ravel())[0, 1])


class TestTimingRecovery(unittest.TestCase):
    def test_estimate_line_length(self):
        # Vertical bars → every line identical → strong line-period autocorrelation.
        img = tv.demo_image("bars", 80, 60)
        sig, h_total, v_total = tv.raster_scan(img, h_blank=12, v_blank=8)
        em = tv.emanate(sig, noise=0.02, seed=1)
        est = tv.estimate_line_length(em, min_len=20, max_len=200)
        self.assertEqual(est, h_total)

    def test_estimate_frame_length_multiframe(self):
        # Horizontal bands give a per-line profile with the frame period; needs
        # several frames, as a real capture has.
        yy, xx = np.mgrid[0:60, 0:80]
        img = ((yy // 8) % 2).astype(float)
        sig, h_total, v_total = tv.raster_scan(img, h_blank=12, v_blank=8)
        multi = np.tile(sig, 4)
        em = tv.emanate(multi, noise=0.01, seed=2)
        est = tv.estimate_frame_length(em, h_total, min_lines=20, max_lines=120)
        self.assertEqual(est, v_total)


class TestReconstructRobustness(unittest.TestCase):
    def test_shape_and_padding(self):
        sig = np.arange(100, dtype=float)
        rec = tv.reconstruct(sig, 15, 10)      # needs 150 > 100 → zero-padded
        self.assertEqual(rec.shape, (10, 15))

    def test_offsets_roll(self):
        sig = np.arange(200, dtype=float)
        a = tv.reconstruct(sig, 20, 5, h_offset=0)
        b = tv.reconstruct(sig, 20, 5, h_offset=3)
        self.assertFalse(np.array_equal(a, b))

    def test_frame_averaging_reduces_noise(self):
        # A flat frame repeated N times with independent noise: averaging the
        # frames must reduce the residual noise (√N SNR gain).
        img = np.full((20, 30), 0.5)
        sig, h_total, v_total = tv.raster_scan(img, h_blank=6, v_blank=4)
        multi = np.tile(sig, 4)
        em = tv.emanate(multi, noise=0.2, seed=5, highpass=False)
        one = tv.reconstruct(em, h_total, v_total, n_frames=1)[:20, :30]
        four = tv.reconstruct(em, h_total, v_total, n_frames=4)[:20, :30]
        self.assertLess(four.std(), one.std())


class TestSources(unittest.TestCase):
    def test_text_image_shape_and_range(self):
        img = tv.demo_image("text", 160, 120, text="SECRET")
        self.assertEqual(img.shape, (120, 160))
        self.assertGreaterEqual(img.min(), 0.0)
        self.assertLessEqual(img.max(), 1.0)
        self.assertGreater(img.sum(), 0.0)     # some pixels lit

    def test_emanate_reproducible(self):
        a = tv.emanate(np.linspace(0, 1, 500), noise=0.1, seed=7)
        b = tv.emanate(np.linspace(0, 1, 500), noise=0.1, seed=7)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
