"""
test_tempest_capture_io.py
==========================
Validation suite for ``tempest_capture_io`` (stdlib unittest only).
"""

import json
import os
import struct
import tempfile
import unittest
import wave

import numpy as np

import tempest_capture_io as cio


def _make_tone(n=2000, fs=2_000_000.0, f_off=100_000.0, amp=0.5):
    t = np.arange(n) / fs
    return (amp * np.exp(2j * np.pi * f_off * t)).astype(np.complex64)


class TestFormatDetection(unittest.TestCase):
    def test_sigmf(self):
        self.assertEqual(cio.detect_format("capture.sigmf-data"), "sigmf")
        self.assertEqual(cio.detect_format("capture.sigmf-meta"), "sigmf")

    def test_wav(self):
        self.assertEqual(cio.detect_format("capture.WAV"), "wav")   # case-insensitive

    def test_raw(self):
        for ext in (".iq", ".dat", ".bin", ".cu8", ".cs16", ".cf32"):
            self.assertEqual(cio.detect_format(f"capture{ext}"), "raw")


class TestRawIQRoundTrip(unittest.TestCase):
    def _roundtrip(self, dtype_key, tol):
        iq = _make_tone(amp=0.5)
        spec = cio.RAW_DTYPES[dtype_key]
        np_dtype = spec["dtype"]
        if np_dtype == np.uint8:
            reals = np.empty(iq.size * 2, dtype=np.float32)
            reals[0::2] = iq.real; reals[1::2] = iq.imag
            raw = np.clip(reals * 127.5 + 127.5, 0, 255).astype(np.uint8)
        elif np_dtype == np.int8:
            reals = np.empty(iq.size * 2, dtype=np.float32)
            reals[0::2] = iq.real; reals[1::2] = iq.imag
            raw = np.clip(reals * 127, -127, 127).astype(np.int8)
        elif np_dtype == np.int16:
            reals = np.empty(iq.size * 2, dtype=np.float32)
            reals[0::2] = iq.real; reals[1::2] = iq.imag
            raw = np.clip(reals * 32767, -32767, 32767).astype(np.int16)
        else:  # cf32
            raw = np.empty(iq.size * 2, dtype=np.float32)
            raw[0::2] = iq.real; raw[1::2] = iq.imag

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, f"test.{dtype_key}")
            raw.tofile(p)
            out = cio.load_raw_iq(p, dtype_key, fs=2_000_000.0, fc=433.92e6)

        self.assertEqual(out["fs"], 2_000_000.0)
        self.assertEqual(out["fc"], 433.92e6)
        self.assertEqual(out["format"], "raw")
        self.assertEqual(len(out["iq"]), len(iq))
        self.assertFalse(out["truncated"])
        # amplitude should round-trip within the format's quantisation tolerance
        self.assertLess(np.max(np.abs(out["iq"] - iq)), tol)

    def test_cu8(self):  self._roundtrip("cu8", tol=0.02)
    def test_cs8(self):  self._roundtrip("cs8", tol=0.02)
    def test_cs16(self): self._roundtrip("cs16", tol=1e-3)
    def test_cf32(self): self._roundtrip("cf32", tol=1e-6)

    def test_missing_fs_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.cu8")
            np.zeros(20, dtype=np.uint8).tofile(p)
            with self.assertRaises(ValueError):
                cio.load_raw_iq(p, "cu8", fs=0)

    def test_unknown_dtype_key_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.iq")
            np.zeros(20, dtype=np.uint8).tofile(p)
            with self.assertRaises(ValueError):
                cio.load_raw_iq(p, "not_a_format", fs=2e6)

    def test_truncation_flag(self):
        iq = _make_tone(n=1000)
        raw = np.empty(iq.size * 2, dtype=np.float32)
        raw[0::2] = iq.real; raw[1::2] = iq.imag
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.cf32")
            raw.tofile(p)
            out = cio.load_raw_iq(p, "cf32", fs=2e6, max_samples=100)
        self.assertEqual(len(out["iq"]), 100)
        self.assertEqual(out["n_samples_total"], 1000)
        self.assertTrue(out["truncated"])


class TestSigMF(unittest.TestCase):
    def _write_pair(self, d, datatype="cf32_le", fs=2_000_000, freq=433.92e6):
        iq = _make_tone(n=1500, fs=fs, amp=0.4)
        data_path = os.path.join(d, "rec.sigmf-data")
        meta_path = os.path.join(d, "rec.sigmf-meta")
        raw = np.empty(iq.size * 2, dtype=np.float32)
        raw[0::2] = iq.real; raw[1::2] = iq.imag
        raw.tofile(data_path)
        meta = {"global": {"core:datatype": datatype, "core:sample_rate": fs,
                           "core:description": "unit test recording",
                           "core:author": "pytest"},
               "captures": [{"core:sample_start": 0, "core:frequency": freq}],
               "annotations": []}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return data_path, meta_path, iq

    def test_load_from_meta_path(self):
        with tempfile.TemporaryDirectory() as d:
            _data, meta, iq = self._write_pair(d)
            out = cio.load_sigmf(meta)
        self.assertEqual(out["fs"], 2_000_000.0)
        self.assertAlmostEqual(out["fc"], 433.92e6)
        self.assertEqual(out["format"], "sigmf")
        self.assertEqual(len(out["iq"]), len(iq))
        self.assertEqual(out["description"], "unit test recording")

    def test_load_from_data_path(self):
        with tempfile.TemporaryDirectory() as d:
            data, _meta, iq = self._write_pair(d)
            out = cio.load_sigmf(data)
        self.assertEqual(len(out["iq"]), len(iq))

    def test_via_load_capture_dispatches_to_sigmf(self):
        with tempfile.TemporaryDirectory() as d:
            data, _meta, iq = self._write_pair(d)
            out = cio.load_capture(data)
        self.assertEqual(out["format"], "sigmf")
        self.assertAlmostEqual(out["fc"], 433.92e6)

    def test_unsupported_datatype_raises(self):
        with tempfile.TemporaryDirectory() as d:
            _data, meta, _iq = self._write_pair(d, datatype="cf64_le")
            with self.assertRaises(ValueError):
                cio.load_sigmf(meta)

    def test_missing_data_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            meta_path = os.path.join(d, "orphan.sigmf-meta")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"global": {"core:datatype": "cf32_le",
                                      "core:sample_rate": 2e6}}, f)
            with self.assertRaises(FileNotFoundError):
                cio.load_sigmf(meta_path)

    def test_cu8_sigmf_datatype(self):
        with tempfile.TemporaryDirectory() as d:
            iq = _make_tone(n=500, amp=0.3)
            reals = np.empty(iq.size * 2, dtype=np.float32)
            reals[0::2] = iq.real; reals[1::2] = iq.imag
            raw = np.clip(reals * 127.5 + 127.5, 0, 255).astype(np.uint8)
            data_path = os.path.join(d, "r.sigmf-data")
            meta_path = os.path.join(d, "r.sigmf-meta")
            raw.tofile(data_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"global": {"core:datatype": "cu8", "core:sample_rate": 2e6},
                          "captures": [{"core:sample_start": 0}]}, f)
            out = cio.load_sigmf(meta_path)
        self.assertEqual(out["dtype_key"], "cu8")
        self.assertIsNone(out["fc"])          # no core:frequency given


class TestWav(unittest.TestCase):
    def test_stereo_iq_wav(self):
        fs = 48000
        n = 2000
        t = np.arange(n) / fs
        i = (0.5 * np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
        q = (0.5 * np.cos(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
        inter = np.empty(n * 2, dtype=np.int16)
        inter[0::2] = i; inter[1::2] = q
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "iq.wav")
            with wave.open(p, "wb") as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(fs)
                wf.writeframes(inter.tobytes())
            out = cio.load_wav_iq(p, fc=100e6)
        self.assertEqual(out["fs"], fs)
        self.assertEqual(out["fc"], 100e6)
        self.assertTrue(np.iscomplexobj(out["iq"]))
        self.assertEqual(len(out["iq"]), n)

    def test_via_load_capture_dispatches_to_wav(self):
        fs = 8000
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.wav")
            with wave.open(p, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
                wf.writeframes((np.zeros(100, dtype=np.int16)).tobytes())
            out = cio.load_capture(p)
        self.assertEqual(out["format"], "wav")


class TestScanFolder(unittest.TestCase):
    def test_lists_mixed_formats_and_flags_raw(self):
        with tempfile.TemporaryDirectory() as d:
            np.zeros(40, dtype=np.uint8).tofile(os.path.join(d, "a.cu8"))
            sub = os.path.join(d, "nested"); os.makedirs(sub)
            np.zeros(40, dtype=np.uint8).tofile(os.path.join(sub, "b.iq"))
            with open(os.path.join(d, "ignore.txt"), "w") as f:
                f.write("not a capture")
            entries = cio.scan_folder(d)
        names = {e["name"] for e in entries}
        self.assertIn("a.cu8", names)
        self.assertIn("b.iq", names)
        self.assertNotIn("ignore.txt", names)
        self.assertTrue(all(e["needs_format"] for e in entries))  # both raw

    def test_sigmf_pair_listed_once_with_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            iq = _make_tone(n=200)
            raw = np.empty(iq.size * 2, dtype=np.float32)
            raw[0::2] = iq.real; raw[1::2] = iq.imag
            raw.tofile(os.path.join(d, "r.sigmf-data"))
            with open(os.path.join(d, "r.sigmf-meta"), "w", encoding="utf-8") as f:
                json.dump({"global": {"core:datatype": "cf32_le", "core:sample_rate": 2e6},
                          "captures": [{"core:sample_start": 0, "core:frequency": 915e6}]}, f)
            entries = cio.scan_folder(d)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["format"], "sigmf")
        self.assertEqual(entries[0]["fs"], 2e6)
        self.assertEqual(entries[0]["fc"], 915e6)
        self.assertFalse(entries[0]["needs_format"])

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cio.scan_folder(d), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
