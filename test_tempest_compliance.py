"""
test_tempest_compliance.py
=========================
Validation suite for ``tempest_compliance`` (stdlib unittest only).
"""

import unittest
import numpy as np

import tempest_compliance as tc


class TestMaskLevels(unittest.TestCase):
    def test_cispr32_b_step(self):
        f = np.array([100e6, 500e6])
        lim = tc.mask_levels(f, "CISPR 32 Class B (10 m)")
        self.assertAlmostEqual(lim[0], 30.0)     # 30–230 MHz band
        self.assertAlmostEqual(lim[1], 37.0)     # 230–1000 MHz band

    def test_out_of_scope_is_nan(self):
        lim = tc.mask_levels(np.array([1e6, 2e9]), "CISPR 32 Class B (10 m)")
        self.assertTrue(np.all(np.isnan(lim)))   # below 30 MHz / above 1 GHz

    def test_distance_correction(self):
        # CISPR B is at 10 m; at 3 m the allowed field is higher by 20·log10(10/3).
        base = tc.mask_levels(np.array([100e6]), "CISPR 32 Class B (10 m)")[0]
        near = tc.mask_levels(np.array([100e6]), "CISPR 32 Class B (10 m)",
                              at_distance_m=3.0)[0]
        self.assertAlmostEqual(near - base, 20 * np.log10(10.0 / 3.0), delta=0.05)

    def test_fcc_bands(self):
        f = np.array([50e6, 150e6, 500e6])
        lim = tc.mask_levels(f, "FCC Part 15 Class B (3 m)")
        self.assertAlmostEqual(lim[0], 40.0)
        self.assertAlmostEqual(lim[1], 43.5)
        self.assertAlmostEqual(lim[2], 46.0)


class TestEvaluate(unittest.TestCase):
    def test_fail_and_margin(self):
        f = np.array([100e6, 500e6])
        levels = np.array([35.0, 30.0])           # 35 > 30 at 100 MHz
        r = tc.evaluate(f, levels, "CISPR 32 Class B (10 m)")
        self.assertFalse(r["passed"])
        self.assertEqual(r["n_exceed"], 1)
        self.assertAlmostEqual(r["worst_margin"], 5.0)
        self.assertAlmostEqual(r["worst_freq"], 100e6)

    def test_pass(self):
        f = np.array([100e6, 500e6])
        levels = np.array([20.0, 25.0])           # both under the limit
        r = tc.evaluate(f, levels, "CISPR 32 Class B (10 m)")
        self.assertTrue(r["passed"])
        self.assertEqual(r["n_exceed"], 0)

    def test_all_out_of_scope_passes_vacuously(self):
        f = np.array([1e6, 5e6])                  # below 30 MHz
        r = tc.evaluate(f, np.array([90.0, 90.0]), "CISPR 32 Class B (10 m)")
        self.assertTrue(r["passed"])
        self.assertEqual(r["n_exceed"], 0)

    def test_illustrative_flag(self):
        self.assertTrue(tc.is_illustrative("NATO SDIP-27 Level A (illustrative)"))
        self.assertFalse(tc.is_illustrative("CISPR 32 Class A (10 m)"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
