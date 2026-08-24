"""
test_tempest_measure.py
======================
Validation suite for ``tempest_measure`` (stdlib unittest only).
"""

import os
import tempfile
import unittest

import numpy as np

import tempest_measure as tm
import tempest_physics as tp


class TestUnitConversions(unittest.TestCase):
    def test_dbm_to_dbuv_50ohm(self):
        # 0 dBm across 50 Ω ≈ 107 dBµV (classic conversion constant).
        self.assertAlmostEqual(float(tm.dbm_to_dbuv(0.0)), 107.0, delta=0.1)

    def test_dbm_dbuv_roundtrip(self):
        v = tm.dbm_to_dbuv(-30.0)
        self.assertAlmostEqual(float(tm.dbuv_to_dbm(v)), -30.0, delta=1e-6)


class TestCalibrationChain(unittest.TestCase):
    def test_field_strength_formula(self):
        # E = V + AF + cable − gain
        E = tm.field_strength_dbuv_per_m(40.0, antenna_factor_db=12.0,
                                         cable_loss_db=3.0, gain_db=20.0)
        self.assertAlmostEqual(float(E), 40 + 12 + 3 - 20)

    def test_field_strength_vector_af(self):
        v = np.array([40.0, 50.0])
        af = np.array([10.0, 15.0])
        E = tm.field_strength_dbuv_per_m(v, af, cable_loss_db=0.0, gain_db=0.0)
        np.testing.assert_allclose(E, [50.0, 65.0])

    def test_reference_offset(self):
        # Known 0 dBm (=107 dBµV) reading -4.4 relative → offset 111.4;
        # applying it to a later -10 relative reading gives 101.4 dBµV.
        off = tm.reference_offset_db(-4.4, tm.dbm_to_dbuv(0.0))
        self.assertAlmostEqual(off, 107.0 + 4.4, delta=0.1)
        self.assertAlmostEqual(-10.0 + off, 101.4, delta=0.1)

    def test_interp_antenna_factor(self):
        af_f = [1e6, 1e9]
        af_v = [10.0, 30.0]
        # geometric midpoint in log-freq → linear midpoint of AF
        mid = tm.interp_antenna_factor([np.sqrt(1e6 * 1e9)], af_f, af_v)
        self.assertAlmostEqual(float(mid[0]), 20.0, delta=0.1)

    def test_interp_af_clamps(self):
        af = tm.interp_antenna_factor([1e3, 1e12], [1e6, 1e9], [10.0, 30.0])
        self.assertAlmostEqual(af[0], 10.0)    # below table → first value
        self.assertAlmostEqual(af[1], 30.0)    # above table → last value


class TestTraceImport(unittest.TestCase):
    def _write(self, path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_load_comma_with_header_mhz(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.csv")
            self._write(p, "freq_MHz,level_dBuV\n# comment\n10,45\n20,50\n30,42\n")
            f, v = tm.load_trace_csv(p, freq_col="freq_MHz",
                                     level_col="level_dBuV", freq_unit="MHz")
        self.assertEqual(len(f), 3)
        self.assertAlmostEqual(f[0], 10e6)          # MHz → Hz
        self.assertAlmostEqual(v[np.argmax(v)], 50)

    def test_load_semicolon_dbm(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.csv")
            self._write(p, "1000000;-60\n2000000;-55\n")
            f, v = tm.load_trace_csv(p, 0, 1, freq_unit="Hz", level_is_dbm=True)
        self.assertAlmostEqual(f[0], 1e6)
        self.assertAlmostEqual(v[0], -60 + 107.0, delta=0.1)   # dBm→dBµV


class TestDeviceCalibration(unittest.TestCase):
    def test_back_projection(self):
        # A trace peaking at 30 dBµV/m measured at 10 m → baseline at 1 m is
        # 30 + 20·log10(10) = 50 dBµV/m.
        e0 = tm.emission_baseline_from_field(np.array([20.0, 30.0, 25.0]), 10.0)
        self.assertAlmostEqual(e0, 50.0, delta=0.1)

    def test_calibrate_device_updates_model(self):
        original = tp.DEVICES["LCD Monitor"]["emission_dbuv"]
        try:
            e0 = tm.calibrate_device("LCD Monitor",
                                     np.array([55.0, 60.0]), distance_m=1.0)
            self.assertEqual(tp.DEVICES["LCD Monitor"]["emission_dbuv"], e0)
            self.assertAlmostEqual(e0, 60.0, delta=0.1)   # peak at 1 m
            self.assertTrue(tp.DEVICES["LCD Monitor"].get("measured"))
        finally:
            tp.DEVICES["LCD Monitor"]["emission_dbuv"] = original
            tp.DEVICES["LCD Monitor"].pop("measured", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
