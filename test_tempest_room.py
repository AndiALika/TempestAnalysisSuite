"""
test_tempest_room.py
===================
Validation suite for ``tempest_room`` (stdlib unittest only).
"""

import unittest
import numpy as np

import tempest_room as tr
import tempest_physics as tp


class TestGeometry(unittest.TestCase):
    def test_exit_wall_directions(self):
        W, L = 10.0, 8.0
        c = (5.0, 4.0)
        self.assertEqual(tr.exit_wall(W, L, c, (20.0, 4.0)), "E")
        self.assertEqual(tr.exit_wall(W, L, c, (-5.0, 4.0)), "W")
        self.assertEqual(tr.exit_wall(W, L, c, (5.0, 20.0)), "N")
        self.assertEqual(tr.exit_wall(W, L, c, (5.0, -5.0)), "S")


class TestWallSE(unittest.TestCase):
    def test_aperture_lowers_wall_se(self):
        base = 60.0
        # a 1 m window is a large aperture → effective SE far below the bulk wall
        eff = tr.wall_effective_se(base, {"length": 1.0, "width": 0.5}, freq_hz=1e8)
        self.assertLess(eff, base)
        self.assertLess(eff, 20.0)          # windows are terrible shields

    def test_no_aperture_returns_bulk(self):
        self.assertEqual(tr.wall_effective_se(42.0, None), 42.0)


class TestFieldAndAssess(unittest.TestCase):
    def _one_device(self):
        return [{"type": "HDMI Cable", "x": 5.0, "y": 4.0, "power": 10.0}]

    def test_wall_se_reduces_field(self):
        eff_open = {s: 0.0 for s in tr.WALLS}
        eff_shield = {s: 40.0 for s in tr.WALLS}
        f_open, _ = tr.field_at_point(self._one_device(), eff_open, 10, 8, (20, 4))
        f_sh, _ = tr.field_at_point(self._one_device(), eff_shield, 10, 8, (20, 4))
        self.assertAlmostEqual(f_open - f_sh, 40.0, delta=0.5)   # exactly the SE

    def test_farther_perimeter_leaks_less(self):
        eff = {s: 10.0 for s in tr.WALLS}
        near, _ = tr.perimeter_worst_case(self._one_device(), eff, 10, 8, 2.0)
        far, _ = tr.perimeter_worst_case(self._one_device(), eff, 10, 8, 30.0)
        self.assertGreater(near, far)

    def test_assess_pass_fail_flips_with_wall_se(self):
        dev = self._one_device()
        fail = tr.assess(dev, {s: 0.0 for s in tr.WALLS}, 10, 8, standoff=3.0)
        ok = tr.assess(dev, {s: 80.0 for s in tr.WALLS}, 10, 8, standoff=3.0)
        self.assertFalse(fail["passed"])
        self.assertTrue(ok["passed"])
        self.assertGreater(fail["risk"], ok["risk"])

    def test_window_increases_leak(self):
        dev = [{"type": "HDMI Cable", "x": 9.0, "y": 4.0, "power": 10.0}]  # near E wall
        walls = {s: 60.0 for s in tr.WALLS}
        no_win = tr.assess(dev, walls, 10, 8, 3.0)
        with_win = tr.assess(dev, walls, 10, 8, 3.0,
                             apertures={"E": {"length": 1.0, "width": 0.5}})
        self.assertGreater(with_win["worst_field"], no_win["worst_field"])

    def test_eavesdropper_reported(self):
        res = tr.assess(self._one_device(), {s: 20.0 for s in tr.WALLS}, 10, 8,
                        standoff=5.0, eavesdropper=(15.0, 4.0))
        self.assertIn("eaves_field", res)
        self.assertIn("eaves_margin", res)


class TestDesignSolver(unittest.TestCase):
    def _devs(self):
        return [{"type": "CRT Monitor", "x": 5.0, "y": 4.0, "power": 15.0}]

    def test_required_wall_se_makes_it_pass(self):
        dev = self._devs()
        walls = {s: 5.0 for s in tr.WALLS}
        rec = tr.design_recommendations(dev, walls, 10, 8, standoff=3.0)
        if not rec["passed"]:
            add = rec["add_wall_se_db"]
            raised = {s: 5.0 + add for s in tr.WALLS}
            res = tr.assess(dev, raised, 10, 8, 3.0)
            self.assertLessEqual(res["worst_field"], res["floor"] + 0.5)

    def test_required_standoff_monotone(self):
        dev = self._devs()
        walls = {s: 10.0 for s in tr.WALLS}
        s = tr.required_standoff(dev, walls, 10, 8)
        if s is not None and s > 0.2:
            worst = tr.perimeter_worst_case(
                dev, tr.effective_walls(walls), 10, 8, s)[0]
            self.assertLessEqual(worst, tp.DETECTION_FLOOR_DBUV + 0.5)

    def test_recommend_material(self):
        rec = tr.recommend_material(60.0)
        self.assertIsNotNone(rec)
        self.assertIn("material", rec)
        self.assertGreater(rec["thickness_mm"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
