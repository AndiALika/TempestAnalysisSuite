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


class TestFileSDRSource(unittest.TestCase):
    def _tone(self, n=2000, fs=2e6, off=100e3, amp=0.5):
        t = np.arange(n) / fs
        return (amp * np.exp(2j * np.pi * off * t)).astype(np.complex64)

    def test_read_returns_exact_samples(self):
        iq = self._tone(n=1000)
        s = sdr.FileSDRSource(iq, sample_rate=2e6, center_freq=100e6).open()
        block = s.read(400)
        np.testing.assert_array_equal(block, iq[:400])

    def test_loops_by_default(self):
        iq = self._tone(n=100)
        s = sdr.FileSDRSource(iq, sample_rate=2e6, loop=True).open()
        first = s.read(100)                # exactly one full pass
        second = s.read(100)               # loops back to the start
        np.testing.assert_array_equal(first, second)

    def test_no_loop_pads_with_zeros_at_end(self):
        iq = self._tone(n=50)
        s = sdr.FileSDRSource(iq, sample_rate=2e6, loop=False).open()
        block = s.read(80)                 # longer than the recording
        np.testing.assert_array_equal(block[:50], iq)
        np.testing.assert_array_equal(block[50:], np.zeros(30, dtype=np.complex64))
        self.assertTrue(s.at_end)

    def test_progress_and_duration(self):
        iq = self._tone(n=1000, fs=2e6)
        s = sdr.FileSDRSource(iq, sample_rate=2e6).open()
        self.assertAlmostEqual(s.duration_s, 1000 / 2e6)
        s.read(250)
        self.assertAlmostEqual(s.progress, 0.25, delta=1e-9)

    def test_empty_source_reads_silence(self):
        s = sdr.FileSDRSource(np.array([], dtype=np.complex64), sample_rate=2e6).open()
        block = s.read(10)
        self.assertEqual(len(block), 10)
        self.assertTrue(np.all(block == 0))

    def test_tone_recovered_via_capture_spectrum(self):
        fs, fc, off = 2.0e6, 100e6, 0.3e6
        iq = self._tone(n=8192, fs=fs, off=off, amp=0.7)
        s = sdr.FileSDRSource(iq, sample_rate=fs, center_freq=fc).open()
        block = s.read(8192)
        f, mag, _ = sdr.capture_spectrum(block, fs, fc, nfft=8192, window="Rectangular")
        peak_f = f[np.argmax(mag)]
        self.assertAlmostEqual(peak_f, fc + off, delta=fs / 8192 * 3)


class TestOpenFileSource(unittest.TestCase):
    def test_opens_raw_cu8_file(self):
        import os, tempfile
        iq = np.exp(2j * np.pi * 0.1e6 * np.arange(500) / 2e6).astype(np.complex64) * 0.5
        reals = np.empty(iq.size * 2, dtype=np.float32)
        reals[0::2] = iq.real; reals[1::2] = iq.imag
        raw = np.clip(reals * 127.5 + 127.5, 0, 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cap.cu8")
            raw.tofile(p)
            src = sdr.open_file_source(p, dtype_key="cu8", fs_hint=2e6, fc_hint=433.92e6)
        self.assertTrue(src.is_open)
        self.assertEqual(src.sample_rate, 2e6)
        self.assertEqual(src.center_freq, 433.92e6)
        self.assertEqual(len(src.info["iq"]), 500)
        self.assertEqual(len(src.read(200)), 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
