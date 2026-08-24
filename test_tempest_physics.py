"""
test_tempest_physics.py
=======================
Validation suite for ``tempest_physics``.  Uses only the standard-library
``unittest`` framework (no external dependencies) so it runs anywhere Python
runs:

    python -m unittest test_tempest_physics -v

The important tests are the *reference-value* ones: they pin the model to
numbers that can be derived independently from textbooks (Ott, Schelkunoff),
which is what lets the thesis claim the tool is validated rather than merely
plausible.
"""

import unittest
import numpy as np

import tempest_physics as tp


class TestReferenceValues(unittest.TestCase):
    """Pin the model to independently-known textbook values."""

    def test_skin_depth_copper_1MHz(self):
        # Classic reference: δ_Cu(1 MHz) ≈ 66 µm.
        d = tp.skin_depth(1e6, tp.SIGMA_CU, 1.0)
        self.assertAlmostEqual(d * 1e6, 66.0, delta=1.0)   # within 1 µm

    def test_skin_depth_scales_as_inverse_sqrt_f(self):
        d1 = tp.skin_depth(1e6, tp.SIGMA_CU)
        d100 = tp.skin_depth(1e8, tp.SIGMA_CU)
        # 100× frequency → 10× thinner skin depth.
        self.assertAlmostEqual(d1 / d100, 10.0, delta=0.05)

    def test_absorption_copper_1mm_1MHz(self):
        # A = 8.686 · t/δ = 8.686 · 1mm/66µm ≈ 131.6 dB.
        A = tp.absorption_loss_db(1e6, tp.SIGMA_CU, 1.0, 1e-3)
        self.assertAlmostEqual(A, 131.6, delta=1.5)

    def test_plane_wave_reflection_copper_1MHz(self):
        # Ott's plane-wave reflection loss for copper at 1 MHz ≈ 108 dB.
        R = tp.reflection_loss_db(1e6, tp.SIGMA_CU, 1.0,
                                  field=tp.FIELD_PLANE)
        self.assertAlmostEqual(float(R), 108.0, delta=1.0)

    def test_eta0_value(self):
        self.assertAlmostEqual(tp.ETA0, 376.730, delta=0.01)


class TestPhysicalTrends(unittest.TestCase):
    """Sanity/monotonicity properties that must hold for any correct model."""

    def test_se_increases_with_thickness(self):
        se = [tp.shielding_effectiveness(1e6, tp.SIGMA_CU, 1.0, t)["SE"]
              for t in (0.1e-3, 0.5e-3, 1e-3, 2e-3)]
        self.assertTrue(all(x < y for x, y in zip(se, se[1:])),
                        f"SE must grow with thickness, got {se}")

    def test_absorption_grows_with_frequency(self):
        A = tp.absorption_loss_db(np.array([1e3, 1e6, 1e9]),
                                  tp.SIGMA_CU, 1.0, 1e-3)
        self.assertTrue(A[0] < A[1] < A[2])

    def test_mu_metal_beats_copper_at_low_freq_magnetic(self):
        # Mu-metal's whole point: superior low-frequency magnetic shielding.
        f = 1e3
        cu = tp.MATERIALS["Copper"]; mu = tp.MATERIALS["Mu-Metal"]
        se_cu = tp.shielding_effectiveness(f, cu["sigma"], cu["mu_r"], 1e-3,
                                           r=0.3, field=tp.FIELD_MAGNETIC)["SE"]
        se_mu = tp.shielding_effectiveness(f, mu["sigma"], mu["mu_r"], 1e-3,
                                           r=0.3, field=tp.FIELD_MAGNETIC)["SE"]
        self.assertGreater(float(se_mu), float(se_cu))

    def test_near_field_electric_gt_magnetic_reflection(self):
        # At small kr, a high-impedance (E) field reflects far more than a
        # low-impedance (H) field — a defining near-field behaviour.
        f, r = 1e5, 0.1
        R_E = tp.reflection_loss_db(f, tp.SIGMA_CU, 1.0, r, tp.FIELD_ELECTRIC)
        R_H = tp.reflection_loss_db(f, tp.SIGMA_CU, 1.0, r, tp.FIELD_MAGNETIC)
        self.assertGreater(float(R_E), float(R_H))

    def test_field_types_converge_in_far_field(self):
        # Far from the source (k r >> 1) all field types → the plane-wave value.
        f, r = 1e9, 50.0          # k r ≈ 1047 ≫ 1
        vals = [float(tp.reflection_loss_db(f, tp.SIGMA_CU, 1.0, r, ft))
                for ft in tp.FIELD_TYPES]
        self.assertAlmostEqual(max(vals), min(vals), delta=0.5)


class TestMultipleReflection(unittest.TestCase):
    def test_B_negligible_for_thick_shield(self):
        # Thick / high-absorption shield: B ≈ 0.
        B = tp.multiple_reflection_db(1e6, tp.SIGMA_CU, 1.0, 5e-3)
        self.assertLess(abs(float(B)), 0.5)

    def test_B_significant_for_thin_foil(self):
        # Very thin foil at low frequency (A ≪ 10 dB): B must be non-trivial
        # and negative (it reduces SE).
        B = tp.multiple_reflection_db(1e3, tp.SIGMA_CU, 1.0, 1e-6)
        self.assertLess(float(B), -0.5)


class TestMinThickness(unittest.TestCase):
    def test_meets_requirement(self):
        t = tp.min_thickness_for_se(tp.SIGMA_CU, 1.0, 60.0,
                                    1e6, 1e9, 5e-3)
        self.assertIsNotNone(t)
        worst = float(np.min(tp.shielding_effectiveness(
            np.logspace(6, 9, 120), tp.SIGMA_CU, 1.0, t)["SE"]))
        # Bisection returns the upper bound, so worst-case SE ≥ requirement.
        self.assertGreaterEqual(worst, 60.0 - 0.5)

    def test_returns_none_when_impossible(self):
        # Absurdly high SE requirement in a vanishingly thin budget.
        t = tp.min_thickness_for_se(tp.SIGMA_CU, 1.0, 500.0,
                                    1e6, 1e9, 1e-6)
        self.assertIsNone(t)


class TestArrayBroadcasting(unittest.TestCase):
    def test_vectorized_over_frequency(self):
        f = np.logspace(3, 9, 50)
        out = tp.shielding_effectiveness(f, tp.SIGMA_CU, 1.0, 1e-3)
        self.assertEqual(out["SE"].shape, f.shape)
        self.assertTrue(np.all(np.isfinite(out["SE"])))


class TestUnitsConversion(unittest.TestCase):
    def test_dbm_to_field_monotonic_in_distance(self):
        near = tp.dbm_to_field_dbuv_per_m(10.0, r=1.0)
        far = tp.dbm_to_field_dbuv_per_m(10.0, r=10.0)
        # 10× distance → −20 dB field strength.
        self.assertAlmostEqual(near - far, 20.0, delta=0.1)


class TestFrequencyDependentMu(unittest.TestCase):
    def test_rolls_off_above_relaxation(self):
        f_relax = 2e3
        lo = tp.effective_mu_r(f_relax / 100, 20000.0, f_relax)
        at = tp.effective_mu_r(f_relax, 20000.0, f_relax)
        hi = tp.effective_mu_r(f_relax * 1000, 20000.0, f_relax)
        self.assertAlmostEqual(float(lo), 20000.0, delta=200)      # ≈ DC below
        self.assertAlmostEqual(float(at), 1 + 19999 / np.sqrt(2), delta=50)
        self.assertLess(float(hi), 100.0)                          # collapsed
        self.assertTrue(lo > at > hi)

    def test_nonmagnetic_unchanged(self):
        mu = tp.effective_mu_r(np.array([1e3, 1e9]), 1.0, None)
        self.assertTrue(np.allclose(mu, 1.0))

    def test_reduces_hf_magnetic_se(self):
        # Mu-metal SE at 10 MHz must be lower with the roll-off than with a
        # frozen DC permeability (the old over-statement).
        mm = tp.MATERIALS["Mu-Metal"]
        f = 1e7
        se_const = tp.shielding_effectiveness(f, mm["sigma"], mm["mu_r"], 1e-3,
                                              r=0.3, field=tp.FIELD_MAGNETIC)["SE"]
        mu_eff = tp.effective_mu_r(f, mm["mu_r"], mm["f_mu"])
        se_roll = tp.shielding_effectiveness(f, mm["sigma"], mu_eff, 1e-3,
                                             r=0.3, field=tp.FIELD_MAGNETIC)["SE"]
        self.assertLess(float(se_roll), float(se_const))


class TestMultilayer(unittest.TestCase):
    CU = tp.MATERIALS["Copper"]

    def test_single_layer_matches_arb(self):
        # One layer via ABCD == the A+R+B result (both rigorous) within ~1 dB,
        # over the range where neither overflows / clips.
        f = np.logspace(3, 7, 40)
        se_abcd = tp.multilayer_se_db(f, [(self.CU["sigma"], 1.0, 1e-3)])
        se_arb = tp.shielding_effectiveness(f, self.CU["sigma"], 1.0, 1e-3)["SE"]
        self.assertTrue(np.allclose(se_abcd, se_arb, atol=1.0))

    def test_two_thin_equal_one_thick(self):
        # Two 1 mm copper layers in contact == one 2 mm copper layer (below the
        # attenuation cap so the equivalence is exact).
        f = np.logspace(3, 7, 30)
        two = tp.multilayer_se_db(f, [(self.CU["sigma"], 1.0, 1e-3)] * 2)
        one = tp.multilayer_se_db(f, [(self.CU["sigma"], 1.0, 2e-3)])
        self.assertTrue(np.allclose(two, one, atol=1e-6))

    def test_laminate_beats_single(self):
        # Adding a mu-metal layer to copper improves low-frequency SE.
        mm = tp.MATERIALS["Mu-Metal"]
        f = 1e3
        cu_only = tp.multilayer_se_db(f, [(self.CU["sigma"], 1.0, 1e-3)],
                                      r=0.3, field=tp.FIELD_MAGNETIC)
        lam = tp.multilayer_se_db(f, [(mm["sigma"], mm["mu_r"], 1e-3),
                                      (self.CU["sigma"], 1.0, 1e-3)],
                                  r=0.3, field=tp.FIELD_MAGNETIC)
        self.assertGreater(float(lam), float(cu_only))

    def test_shield_mass(self):
        m = tp.shield_mass_kg([(8960, 1e-3), (7870, 0.5e-3)], area_m2=2.0)
        self.assertAlmostEqual(m, 8960 * 1e-3 * 2 + 7870 * 0.5e-3 * 2, delta=1e-6)


class TestBestLaminates(unittest.TestCase):
    def test_returns_designs_meeting_target(self):
        cands = tp.best_laminates(1e4, 1e8, 60.0, t_max_each_mm=3.0, area_m2=1.0)
        self.assertTrue(cands)
        for c in cands:
            self.assertGreaterEqual(c["se"], 60.0 - 1.0)
            self.assertGreater(c["mass_kg"], 0.0)
            self.assertIn("layers", c)

    def test_includes_a_two_layer_option(self):
        cands = tp.best_laminates(1e3, 1e9, 80.0, t_max_each_mm=3.0)
        self.assertTrue(any(len(c["layers"]) == 2 for c in cands))

    def test_infeasible_returns_empty(self):
        # 400 dB across a wide band within 1 µm each is impossible.
        cands = tp.best_laminates(1e6, 1e9, 400.0, t_max_each_mm=0.001)
        self.assertEqual(cands, [])

    def test_sorted_by_mass(self):
        cands = tp.best_laminates(1e4, 1e8, 50.0)
        masses = [c["mass_kg"] for c in cands]
        self.assertEqual(masses, sorted(masses))

    def test_floor_prevents_submicron_designs(self):
        # Plane-wave reflection-limited case: without a manufacturing floor the
        # optimizer returns meaningless sub-micron foils that display as 0.
        # Every layer must be at least the 0.01 mm floor.
        cands = tp.best_laminates(1e6, 1e9, 60.0, field=tp.FIELD_PLANE,
                                  t_max_each_mm=5.0)
        self.assertTrue(cands)
        for c in cands:
            self.assertGreater(c["se"], 0.0)
            for _mat, t_mm in c["layers"]:
                self.assertGreaterEqual(t_mm, 0.01 - 1e-9)

    def test_cost_usd_is_real_currency(self):
        cands = tp.best_laminates(1e4, 1e8, 50.0, area_m2=1.0)
        cu = [c for c in cands if len(c["layers"]) == 1 and c["layers"][0][0] == "Copper"]
        self.assertTrue(cu)
        c = cu[0]; t_m = c["layers"][0][1] / 1000.0
        m = tp.MATERIALS["Copper"]
        expected = m["price_per_kg"] * m["density"] * t_m * 1.0
        self.assertAlmostEqual(c["cost_usd"], expected, delta=abs(expected) * 0.01 + 1e-9)

    def test_frequency_dependent_requirement(self):
        # A requirement that is high at low frequency and low at high frequency;
        # every returned design must satisfy SE(f) >= req(f) everywhere.
        req = lambda f: np.where(f < 1e5, 55.0, 20.0)
        cands = tp.best_laminates(1e3, 1e9, req, field=tp.FIELD_MAGNETIC, r=0.3,
                                  t_max_each_mm=3.0)
        self.assertTrue(cands)
        for c in cands:
            self.assertGreaterEqual(c["margin"], -0.5)

    def test_required_se_curve(self):
        f = np.array([1e6, 1e7])
        req = tp.required_se_curve(f, source_dbuv=np.array([80.0, 50.0]), limit_dbuv=30.0)
        np.testing.assert_allclose(req, [50.0, 20.0])
        # clamps negatives (source below limit) and NaN (out-of-scope limit) to 0
        req2 = tp.required_se_curve(f, np.array([10.0, 50.0]), np.array([30.0, np.nan]))
        np.testing.assert_allclose(req2, [0.0, 0.0])


class TestPropagation(unittest.TestCase):
    def test_fraunhofer_distance(self):
        # D=1 m at 1 GHz (λ=0.2998 m) → 2·1²/λ ≈ 6.67 m.
        self.assertAlmostEqual(float(tp.fraunhofer_distance(1.0, 1e9)), 6.671, delta=0.05)

    def test_friis_path_loss_reference(self):
        # 1 m @ 1 GHz: 20·log10(4π·1/0.2998) ≈ 32.4 dB.
        self.assertAlmostEqual(float(tp.friis_path_loss_db(1e9, 1.0)), 32.44, delta=0.1)

    def test_friis_received_decays_20db_per_decade(self):
        near = tp.friis_received_dbm(10.0, 1e8, 1.0)
        far = tp.friis_received_dbm(10.0, 1e8, 10.0)
        self.assertAlmostEqual(near - far, 20.0, delta=0.1)

    def test_gains_add(self):
        base = tp.friis_received_dbm(0.0, 1e8, 5.0)
        with_gain = tp.friis_received_dbm(0.0, 1e8, 5.0, g_tx_dbi=3.0, g_rx_dbi=2.0)
        self.assertAlmostEqual(with_gain - base, 5.0, delta=1e-6)


class TestCoverage(unittest.TestCase):
    def test_pattern_isotropic_and_dipole(self):
        self.assertEqual(float(tp.radiation_pattern_db(1.0, "isotropic")), 0.0)
        # dipole: ~0 dB broadside (θ=π/2), strongly down along the axis (θ=0)
        self.assertAlmostEqual(float(tp.radiation_pattern_db(np.pi / 2, "dipole")), 0.0, delta=0.05)
        self.assertLess(float(tp.radiation_pattern_db(0.0, "dipole")), -20.0)

    def test_coverage_decays_and_anchors(self):
        # field at 1 m equals the source level; falls with distance (isotropic)
        near = tp.coverage_field_dbuv(60.0, 1.0, 0.0)
        far = tp.coverage_field_dbuv(60.0, 10.0, 0.0)
        self.assertAlmostEqual(float(near), 60.0, delta=0.1)
        self.assertAlmostEqual(float(near - far), 20.0, delta=0.1)   # −20 dB/decade

    def test_two_ray_has_lobes_and_far_null(self):
        r = np.linspace(1, 200, 4000)
        tr = tp.two_ray_factor_db(r, 1e8, h_tx=2.0, h_rx=2.0)
        self.assertGreater(tr.max() - tr.min(), 6.0)          # interference structure
        self.assertLess(float(tr[-1]), 0.0)                    # cancels beyond last lobe

    def test_interception_range(self):
        # E0=60, threshold=20 → r = 10^((60−20)/20) = 100 m
        self.assertAlmostEqual(tp.interception_range_m(60.0, 20.0), 100.0, delta=0.5)
        # higher threshold ⇒ shorter interception range
        self.assertLess(tp.interception_range_m(60.0, 40.0), tp.interception_range_m(60.0, 20.0))


class TestApertureLeakage(unittest.TestCase):
    def test_slot_leakage_reference(self):
        # 100 mm slot at 100 MHz (λ=3 m): SE = 20·log10(3/0.2) = 23.5 dB.
        se = tp.slot_leakage_db(100e6, 0.1)
        self.assertAlmostEqual(float(se), 23.52, delta=0.2)

    def test_slot_resonance_zero(self):
        # At the half-wave resonance L = λ/2 the slot radiates freely (SE→0).
        f_res = tp.C0 / (2 * 0.1)             # λ = 0.2 m
        self.assertAlmostEqual(float(tp.slot_leakage_db(f_res, 0.1)), 0.0, delta=0.1)

    def test_slot_more_leaky_with_frequency(self):
        lo = tp.slot_leakage_db(10e6, 0.1)
        hi = tp.slot_leakage_db(500e6, 0.1)
        self.assertGreater(float(lo), float(hi))

    def test_n_apertures_correction(self):
        one = tp.slot_leakage_db(100e6, 0.1, 1)
        four = tp.slot_leakage_db(100e6, 0.1, 4)
        self.assertAlmostEqual(float(one - four), 10 * np.log10(4), delta=0.05)

    def test_waveguide_rectangular_reference(self):
        # Well below cutoff: A ≈ 27.3·t/w. w=10 mm, t=2 mm → 5.46 dB.
        A = tp.waveguide_below_cutoff_db(1e6, 0.010, 0.002, "rectangular")
        self.assertAlmostEqual(float(A), 27.3 * 2 / 10, delta=0.1)

    def test_waveguide_circular_reference(self):
        # A ≈ 32·t/d. d=5 mm, t=3 mm → 19.2 dB.
        A = tp.waveguide_below_cutoff_db(1e6, 0.005, 0.003, "circular")
        self.assertAlmostEqual(float(A), 32.0 * 3 / 5, delta=0.2)

    def test_waveguide_zero_above_cutoff(self):
        # Above the aperture's cutoff there is no evanescent attenuation.
        w = 0.010
        fc = tp.C0 / (2 * w)
        self.assertAlmostEqual(
            float(tp.waveguide_below_cutoff_db(fc * 2, w, 0.002)), 0.0, delta=1e-9)

    def test_combine_dominated_by_worst(self):
        f = np.array([1e8, 5e8])
        good = np.array([120.0, 110.0])
        bad = np.array([25.0, 20.0])
        tot = tp.combine_se_db(good, bad)
        self.assertTrue(np.all(tot <= bad + 1e-6))          # never better than worst
        self.assertTrue(np.all(np.abs(tot - bad) < 0.1))    # ≈ the bad path

    def test_enclosure_limited_by_aperture(self):
        # A thick copper wall with a 150 mm seam: overall SE ≈ the seam, far
        # below the (enormous) bulk-material SE.
        f = np.logspace(7, 9, 50)
        material = tp.shielding_effectiveness(f, tp.SIGMA_CU, 1.0, 2e-3)["SE"]
        enc = tp.enclosure_se_db(f, tp.SIGMA_CU, 1.0, 2e-3,
                                 apertures=[{"length": 0.15, "width": 0.002}])
        self.assertTrue(np.all(enc < material))
        self.assertLess(float(np.min(enc)), 60.0)           # seam dominates


class TestEmanationAndZoning(unittest.TestCase):
    def test_field_decays_20db_per_decade(self):
        near = tp.field_at_distance_dbuv("CRT Monitor", 10.0, 1.0)
        far = tp.field_at_distance_dbuv("CRT Monitor", 10.0, 10.0)
        self.assertAlmostEqual(near - far, 20.0, delta=0.1)

    def test_stronger_emitter_has_larger_zone(self):
        crt = tp.zone_radius_m("CRT Monitor", 10.0,
                               tp.ZONE_THRESHOLDS_DBUV["Zone 0 (Controlled)"])
        kbd = tp.zone_radius_m("Keyboard (PS/2)", 10.0,
                               tp.ZONE_THRESHOLDS_DBUV["Zone 0 (Controlled)"])
        self.assertGreater(crt, kbd)

    def test_higher_power_enlarges_zone(self):
        lo = tp.zone_radius_m("CPU (3 GHz)", 0.0, 30.0)
        hi = tp.zone_radius_m("CPU (3 GHz)", 20.0, 30.0)
        self.assertGreater(hi, lo)

    def test_outer_zone_radius_larger_than_inner(self):
        r0 = tp.zone_radius_m("GPU", 10.0, tp.ZONE_THRESHOLDS_DBUV["Zone 0 (Controlled)"])
        r3 = tp.zone_radius_m("GPU", 10.0, tp.ZONE_THRESHOLDS_DBUV["Zone 3 (Public)"])
        self.assertGreater(r3, r0)   # lower threshold ⇒ farther contour

    def test_wall_shielding_reduces_risk(self):
        devs = [{"type": "CRT Monitor", "x": 1.0, "y": 1.0, "power": 10.0}]
        r_open, _ = tp.room_risk_score(devs, 10, 8, wall_se_db=0.0)
        r_shielded, _ = tp.room_risk_score(devs, 10, 8, wall_se_db=40.0)
        self.assertGreater(r_open, r_shielded)   # wall SE must matter now

    def test_device_type_affects_risk(self):
        crt = [{"type": "CRT Monitor", "x": 1.0, "y": 1.0, "power": 10.0}]
        kbd = [{"type": "Keyboard (PS/2)", "x": 1.0, "y": 1.0, "power": 10.0}]
        r_crt, _ = tp.room_risk_score(crt, 10, 8, 10.0)
        r_kbd, _ = tp.room_risk_score(kbd, 10, 8, 10.0)
        self.assertGreater(r_crt, r_kbd)   # not identical any more

    def test_risk_score_bounded(self):
        many = [{"type": "CRT Monitor", "x": 0.2, "y": 0.2, "power": 30.0}
                for _ in range(20)]
        score, _ = tp.room_risk_score(many, 10, 8, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertGreaterEqual(score, 0.0)

    def test_emanation_spectrum_reproducible(self):
        f1, s1 = tp.emanation_spectrum("CRT Monitor", 1.0, seed=42)
        f2, s2 = tp.emanation_spectrum("CRT Monitor", 1.0, seed=42)
        np.testing.assert_array_equal(s1, s2)     # deterministic given seed

    def test_emanation_spectrum_attenuates_with_distance(self):
        _, near = tp.emanation_spectrum("CRT Monitor", 1.0, seed=1)
        _, far = tp.emanation_spectrum("CRT Monitor", 10.0, seed=1)
        self.assertGreater(near.max(), far.max())


if __name__ == "__main__":
    unittest.main(verbosity=2)
