"""
test_tempest_dsp.py
===================
Validation suite for ``tempest_dsp``.  Runs with the stdlib only:

    python -m unittest test_tempest_dsp -v
"""

import os
import tempfile
import unittest
import wave

import numpy as np

import tempest_dsp as dsp


class TestWavLoading(unittest.TestCase):
    def _write_wav(self, path, data_int16, fs, channels=1):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(data_int16.tobytes())

    def test_mono_int16_roundtrip(self):
        fs = 8000
        t = np.arange(fs) / fs
        sig = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.wav")
            self._write_wav(p, sig, fs)
            data, got_fs = dsp.load_wav(p)
        self.assertEqual(got_fs, fs)
        self.assertEqual(len(data), fs)
        self.assertLessEqual(np.max(np.abs(data)), 1.0 + 1e-9)   # normalised

    def test_stereo_downmix_to_mono(self):
        fs = 8000
        n = 1000
        stereo = np.zeros(n * 2, dtype=np.int16)
        stereo[0::2] = 10000    # L
        stereo[1::2] = -10000   # R
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.wav")
            self._write_wav(p, stereo, fs, channels=2)
            data, _ = dsp.load_wav(p, mono=True)
        # length is per-channel (not the interleaved count) — the old code got
        # this wrong and read 2× the samples as mono.
        self.assertEqual(len(data), n)
        self.assertTrue(np.allclose(data, 0.0, atol=1e-6))   # L+R average ≈ 0


class TestIQLoading(unittest.TestCase):
    def test_iq_preserves_quadrature(self):
        # A complex tone at +f has energy ONLY at the positive frequency; if Q is
        # discarded it would appear at ±f. This guards the original bug.
        fs = 1_000_000
        t = np.arange(4096) / fs
        iq = np.exp(2j * np.pi * 100_000 * t).astype(np.complex64)
        inter = np.empty(iq.size * 2, dtype=np.float32)
        inter[0::2] = iq.real
        inter[1::2] = iq.imag
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.iq")
            inter.tofile(p)
            data, fs2 = dsp.load_iq(p, fs=fs)
        self.assertEqual(fs2, fs)
        self.assertTrue(np.iscomplexobj(data))
        f, mag, _ = dsp.averaged_spectrum(data, fs, nfft=4096, window="Rectangular")
        peak_f = f[np.argmax(mag)]
        self.assertAlmostEqual(peak_f, 100_000, delta=fs / 4096 * 2)
        # Ensure it is NOT symmetric about 0 (which is what losing Q would give).
        neg_energy = mag[f < -50_000].max()
        pos_energy = mag[f > 50_000].max()
        self.assertGreater(pos_energy, neg_energy + 20)


class TestAveragedSpectrum(unittest.TestCase):
    def test_unit_sine_reads_zero_db(self):
        # Coherent-gain correction: a 1.0-amplitude tone → ≈ 0 dB for ANY window.
        # The tone is placed exactly on an FFT bin (fs == nfft, freq integer) so
        # that scalloping loss does not confound the gain check.
        fs = nfft = 8192
        t = np.arange(nfft) / fs
        sig = 1.0 * np.sin(2 * np.pi * 2000 * t)          # 2000 → exact bin 2000
        for win in ("Rectangular", "Hann", "Hamming", "Blackman", "Flat Top"):
            f, mag, _ = dsp.averaged_spectrum(sig, fs, nfft=nfft, window=win)
            self.assertAlmostEqual(mag.max(), 0.0, delta=0.3,
                                   msg=f"window {win} peak {mag.max():.2f} dB")

    def test_peak_frequency_correct(self):
        fs = 44100
        t = np.arange(44100) / fs
        sig = np.sin(2 * np.pi * 5000 * t)
        f, mag, _ = dsp.averaged_spectrum(sig, fs, nfft=8192, window="Hann")
        self.assertAlmostEqual(f[np.argmax(mag)], 5000, delta=fs / 8192 * 2)

    def test_averaging_reduces_noise_variance(self):
        fs = 48000
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(48000)
        _, m1, n1 = dsp.averaged_spectrum(noise, fs, nfft=1024, overlap=0.5)
        _, m8, n8 = dsp.averaged_spectrum(noise, fs, nfft=8192, overlap=0.5)
        self.assertGreater(n1, n8)                    # smaller nfft → more averages
        self.assertLess(np.std(m1), np.std(m8))       # more averaging → smoother

    def test_dc_is_removed(self):
        fs = 8000
        sig = 5.0 + np.sin(2 * np.pi * 1000 * np.arange(8000) / fs)  # big DC offset
        f, mag, _ = dsp.averaged_spectrum(sig, fs, nfft=4096, window="Hann")
        dc_level = mag[0]
        tone_level = mag.max()
        self.assertGreater(tone_level, dc_level)      # DC must not dominate


class TestSpectrogram(unittest.TestCase):
    def test_shapes(self):
        fs = 16000
        sig = np.sin(2 * np.pi * 1000 * np.arange(16000) / fs)
        f, t, S = dsp.make_spectrogram(sig, fs, nfft=512)
        self.assertEqual(S.shape[0], f.size)
        self.assertEqual(S.shape[1], t.size)


class TestSynthSignal(unittest.TestCase):
    def test_reproducible(self):
        a, _ = dsp.synth_tempest_signal(seed=3)
        b, _ = dsp.synth_tempest_signal(seed=3)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
