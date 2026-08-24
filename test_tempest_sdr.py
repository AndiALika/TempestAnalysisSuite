"""
test_tempest_sdr.py
==================
Validation suite for ``tempest_sdr`` (stdlib unittest only).
"""

import unittest
import numpy as np

import tempest_sdr as sdr


class TestBackends(unittest.TestCase):
    def test_simulated_always_available(self):
        self.assertTrue(sdr.available_backends()["simulated"])

    def test_open_simulated(self):
        s = sdr.open_sdr("simulated", sample_rate=1e6, center_freq=100e6)
        self.assertTrue(s.is_open)
        self.assertEqual(s.read(1024).shape, (1024,))
        s.close()
        self.assertFalse(s.is_open)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            sdr.open_sdr("banana")

    def test_rtlsdr_missing_is_informative(self):
        # pyrtlsdr is not installed in this environment → clear RuntimeError.
        with self.assertRaises(RuntimeError):
            sdr.open_sdr("rtlsdr")

    def test_soapy_missing_is_informative(self):
        with self.assertRaises(RuntimeError):
            sdr.open_sdr("soapy")

    def test_uhd_missing_is_informative(self):
        with self.assertRaises(RuntimeError):
            sdr.open_sdr("uhd")


class TestSimulatedSDR(unittest.TestCase):
    def test_read_is_complex(self):
        s = sdr.SimulatedSDR(sample_rate=1e6).open()
        block = s.read(2048)
        self.assertTrue(np.iscomplexobj(block))
        self.assertEqual(len(block), 2048)

    def test_streaming_advances_time(self):
        # The time base must advance and the stream must not repeat verbatim.
        s = sdr.SimulatedSDR(sample_rate=1e6).open()   # default noise
        a = s.read(1000)
        b = s.read(1000)
        self.assertEqual(s._t0, 2000)          # continuous time base
        self.assertFalse(np.allclose(a, b))    # noise advances too

    def test_tone_appears_at_expected_rf(self):
        fs, fc, off = 2.0e6, 100e6, 0.25e6
        s = sdr.SimulatedSDR(sample_rate=fs, center_freq=fc,
                             tones=[off], noise=0.0).open()
        iq = s.read(8192)
        f, mag, _ = sdr.capture_spectrum(iq, fs, fc, nfft=8192, window="Rectangular")
        peak_f = f[np.argmax(mag)]
        self.assertAlmostEqual(peak_f, fc + off, delta=fs / 8192 * 3)

    def test_device_tones_within_band(self):
        # A device-seeded source must place all its tones inside ±fs/2.
        fs = 2.0e6
        s = sdr.SimulatedSDR(sample_rate=fs, center_freq=300e6, device="GPU")
        self.assertTrue(all(abs(t) <= fs / 2 for t in s.tones))


if __name__ == "__main__":
    unittest.main(verbosity=2)
