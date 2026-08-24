"""
tempest_physics.py
==================
Pure physics / engineering core for the TEMPEST Analysis Suite.

This module contains **no GUI code**. Every routine is a pure NumPy function so
that it can be unit-tested, reused, and cited independently of the interface.
Keeping the science separate from the presentation is what makes the numbers in
the thesis reproducible and defensible.

All quantities are SI unless a function name explicitly says otherwise
(e.g. ``*_mm`` returns millimetres, ``*_db`` returns decibels).

References
----------
[1] H. W. Ott, *Electromagnetic Compatibility Engineering*, Wiley, 2009, Ch. 6
    (shielding: absorption, reflection and multiple-reflection loss).
[2] S. A. Schelkunoff, *Electromagnetic Waves*, Van Nostrand, 1943
    (impedance formulation of shielding effectiveness).
[3] IEEE Std 299-2006, *Measuring the Effectiveness of Electromagnetic
    Shielding Enclosures*.
[4] C. R. Paul, *Introduction to Electromagnetic Compatibility*, 2nd ed., Wiley.
[5] M. G. Kuhn, "Compromising emanations: eavesdropping risks of computer
    displays", University of Cambridge, Tech. Report UCAM-CL-TR-577, 2003.
[6] NATO SDIP-27 (formerly AMSG 720B) emanation-security zoning model.
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Physical constants (CODATA 2018)
# ─────────────────────────────────────────────────────────────────────────────
MU0      = 4.0e-7 * np.pi          # vacuum permeability            [H/m]
EPS0     = 8.8541878128e-12        # vacuum permittivity            [F/m]
C0       = 299_792_458.0           # speed of light (exact)         [m/s]
ETA0     = MU0 * C0                # free-space wave impedance ≈376.730 Ω
SIGMA_CU = 5.8e7                   # reference conductivity, copper  [S/m]

# Field / coupling type identifiers used across the suite.
FIELD_PLANE    = "Plane Wave"
FIELD_ELECTRIC = "Electric Field"
FIELD_MAGNETIC = "Magnetic Field"
FIELD_TYPES    = (FIELD_PLANE, FIELD_ELECTRIC, FIELD_MAGNETIC)


# ─────────────────────────────────────────────────────────────────────────────
#  Material library  (σ in S/m, μ_r dimensionless, ρ relative cost index)
# ─────────────────────────────────────────────────────────────────────────────
#  ``cost`` is a *relative* $/kg index (copper ≈ 8) used only for ranking, never
#  presented as an absolute price.
#  ``f_mu`` (relaxation frequency, Hz) is only set for ferromagnetic materials:
#  their permeability rolls off above it (see ``effective_mu_r``).  ``density`` is
#  kg/m³.  ``price_per_kg`` is a representative raw-material price [USD/kg] — an
#  order-of-magnitude figure (metal prices fluctuate; fabrication not included) —
#  used for a real currency cost estimate (mass × price) rather than an index.
MATERIALS: dict[str, dict] = {
    "Copper":        {"sigma": 5.8e7, "mu_r": 1.0,     "density": 8960,  "f_mu": None,  "price_per_kg": 9.5},
    "Aluminum":      {"sigma": 3.5e7, "mu_r": 1.0,     "density": 2700,  "f_mu": None,  "price_per_kg": 2.6},
    "Steel (mild)":  {"sigma": 1.0e7, "mu_r": 100.0,   "density": 7870,  "f_mu": 2.0e5, "price_per_kg": 0.9},
    "Steel (SS304)": {"sigma": 1.4e6, "mu_r": 1.0,     "density": 8000,  "f_mu": None,  "price_per_kg": 3.5},
    "Mu-Metal":      {"sigma": 1.6e6, "mu_r": 20000.0, "density": 8747,  "f_mu": 2.0e3, "price_per_kg": 65.0},
    "Brass":         {"sigma": 1.5e7, "mu_r": 1.0,     "density": 8530,  "f_mu": None,  "price_per_kg": 6.5},
    "Tin":           {"sigma": 8.7e6, "mu_r": 1.0,     "density": 7265,  "f_mu": None,  "price_per_kg": 30.0},
    "Silver":        {"sigma": 6.1e7, "mu_r": 1.0,     "density": 10490, "f_mu": None,  "price_per_kg": 850.0},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Skin effect & intrinsic impedances
# ─────────────────────────────────────────────────────────────────────────────
def skin_depth(f, sigma, mu_r=1.0):
    """Skin depth δ = 1/√(π f μ σ)  [m].

    Reference check: copper at 1 MHz → δ ≈ 66 µm.
    """
    f = np.asarray(f, dtype=float)
    return 1.0 / np.sqrt(np.pi * f * mu_r * MU0 * sigma)


def shield_impedance(f, sigma, mu_r=1.0):
    """Intrinsic (characteristic) impedance of a good conductor.

    Z_s = √(jωμ/σ)  [Ω], complex.  For a good conductor |Z_s| = √(ωμ/σ).
    ``mu_r`` may be a scalar or a per-frequency array.
    """
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return np.sqrt(1j * w * mu_r * MU0 / sigma)


def effective_mu_r(f, mu_r_dc, f_relax=None):
    """Frequency-dependent relative permeability of a ferromagnetic material.

    Above the relaxation frequency the permeability rolls off (single-pole
    magnitude model, Snoek-like):

        µr(f) = 1 + (µr_dc − 1) / √(1 + (f/f_relax)²)

    So it is ≈ µr_dc well below f_relax and → 1 well above it.  Non-magnetic
    materials (µr_dc ≈ 1 or ``f_relax`` None) are returned unchanged.  This is
    why the earlier constant-µr model *over-stated* high-frequency magnetic
    shielding for mu-metal.
    """
    f = np.asarray(f, dtype=float)
    if not f_relax or mu_r_dc <= 1.0:
        return np.full_like(f, float(mu_r_dc))
    return 1.0 + (mu_r_dc - 1.0) / np.sqrt(1.0 + (f / float(f_relax)) ** 2)


def wave_impedance(f, r, field=FIELD_PLANE):
    """Wave (field) impedance of the *incident* field at distance ``r`` from the
    source, following the small-dipole near-field model in Ott [1, §6.6].

    * Plane wave / far field:      Z_w = η0 (≈ 377 Ω)
    * Electric (high-impedance)    Z_w = η0 / (k r)   for k r < 1   → η0 else
    * Magnetic (low-impedance)     Z_w = η0 · (k r)   for k r < 1   → η0 else

    where k = 2πf/c.  Both near-field cases converge to η0 at the reactive
    near-/far-field boundary r = λ/2π (k r = 1), which is exactly why the plane
    wave case is the large-``r`` limit.  ``r`` is ignored for a plane wave.
    """
    f = np.asarray(f, dtype=float)
    if field == FIELD_PLANE:
        return np.full_like(f, ETA0, dtype=float)

    kr = (2.0 * np.pi * f / C0) * float(r)
    kr = np.maximum(kr, 1e-12)              # guard r→0 / f→0
    if field == FIELD_ELECTRIC:
        # high-impedance field: falls from ∞ toward η0 as kr→1
        return ETA0 * np.maximum(1.0, 1.0 / kr)
    elif field == FIELD_MAGNETIC:
        # low-impedance field: rises from 0 toward η0 as kr→1
        return ETA0 * np.minimum(1.0, kr)
    raise ValueError(f"unknown field type: {field!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  Shielding effectiveness   SE = A + R + B
# ─────────────────────────────────────────────────────────────────────────────
def absorption_loss_db(f, sigma, mu_r, t_m):
    """Absorption loss A = 8.686 · t/δ  [dB]  (t and δ in metres).

    Reference check: 1 mm copper at 1 MHz → A ≈ 131.6 dB.
    """
    delta = skin_depth(f, sigma, mu_r)
    return 8.686 * float(t_m) / delta


def reflection_loss_db(f, sigma, mu_r, r=1.0, field=FIELD_PLANE):
    """Reflection loss R  [dB] from the impedance mismatch at the two air/metal
    interfaces (Schelkunoff formulation [2]):

        R = 20·log10( |Z_w + Z_s|² / (4 |Z_w| |Z_s|) )

    This exact expression (rather than the |Z_w|/(4|Z_s|) approximation) stays
    correct even when the mismatch is small.  For a plane wave it reproduces the
    textbook value (copper @ 1 MHz → ≈ 108 dB).
    """
    Zs = shield_impedance(f, sigma, mu_r)
    Zw = wave_impedance(f, r, field)
    R = 20.0 * np.log10(np.abs(Zw + Zs) ** 2 / (4.0 * np.abs(Zw) * np.abs(Zs)))
    return np.clip(R, 0.0, None)


def multiple_reflection_db(f, sigma, mu_r, t_m, r=1.0, field=FIELD_PLANE):
    """Multiple-reflection correction term B  [dB], Ott [1, eq. 6-9].

        B = 20·log10 | 1 − q · e^(−2γt) | ,   γ = (1+j)/δ,
        q = ((Z_s − Z_w)/(Z_s + Z_w))²

    B is negative (it *reduces* SE) and is only significant for electrically
    thin shields where the absorption loss A ≲ 10 dB (e.g. thin foils, or any
    shield at low frequency).  It vanishes as e^(−2t/δ) → 0.
    """
    f = np.asarray(f, dtype=float)
    delta = skin_depth(f, sigma, mu_r)
    Zs = shield_impedance(f, sigma, mu_r)
    Zw = wave_impedance(f, r, field)
    q = ((Zs - Zw) / (Zs + Zw)) ** 2
    gamma = (1.0 + 1j) / delta
    B = 20.0 * np.log10(np.abs(1.0 - q * np.exp(-2.0 * gamma * float(t_m))))
    return B


def shielding_effectiveness(f, sigma, mu_r, t_m, r=1.0, field=FIELD_PLANE,
                            include_multiple_reflection=True):
    """Total shielding effectiveness SE = A + R + B  [dB].

    Parameters
    ----------
    f : float or array   frequency [Hz]
    sigma : float        conductivity [S/m]
    mu_r : float         relative permeability
    t_m : float          shield thickness [m]
    r : float            distance from emanation source [m] (near-field only)
    field : str          one of FIELD_TYPES
    include_multiple_reflection : bool
        Include the B correction (recommended; matters for thin shields).

    Returns
    -------
    dict with keys ``SE``, ``A``, ``R``, ``B``, ``skin_depth`` (all arrays if
    ``f`` is an array).
    """
    A = absorption_loss_db(f, sigma, mu_r, t_m)
    R = reflection_loss_db(f, sigma, mu_r, r, field)
    B = (multiple_reflection_db(f, sigma, mu_r, t_m, r, field)
         if include_multiple_reflection else np.zeros_like(A))
    return {"SE": A + R + B, "A": A, "R": R, "B": B,
            "skin_depth": skin_depth(f, sigma, mu_r)}


def min_thickness_for_se(sigma, mu_r, req_se_db, f_lo, f_hi, t_max_m,
                         r=1.0, field=FIELD_PLANE, n_freq=120, iters=40,
                         t_floor_m=1e-5):
    """Smallest thickness [m] whose *worst-case* SE across [f_lo, f_hi] meets
    ``req_se_db``.  Returns ``None`` if unattainable within ``t_max_m``.

    ``t_floor_m`` is a practical manufacturing floor (default 10 µm): in a
    reflection-dominated (far-field / plane-wave) scenario the reflection loss
    alone can already exceed the requirement, so the *thinnest* shield that
    works is essentially zero.  Reporting a sub-micron thickness (and the
    divide-by-tiny "SE per mm" that follows) is meaningless, so the search is
    clamped at ``t_floor_m`` and returns that floor when even it suffices.

    SE increases monotonically with thickness (A ∝ t, B → 0), so bisection is
    exact to machine precision within ``iters`` iterations.
    """
    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_freq)

    def worst_se(t_m):
        return float(np.min(
            shielding_effectiveness(freqs, sigma, mu_r, t_m, r, field)["SE"]))

    if worst_se(t_max_m) < req_se_db:
        return None
    if worst_se(t_floor_m) >= req_se_db:
        return t_floor_m           # reflection-limited: any thin foil suffices
    lo, hi = t_floor_m, t_max_m
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if worst_se(mid) >= req_se_db:
            hi = mid
        else:
            lo = mid
    return hi


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-layer (laminated) shields  —  ABCD transmission-matrix cascade
# ─────────────────────────────────────────────────────────────────────────────
#  A laminate (e.g. mu-metal for low-frequency magnetic + copper for HF) is
#  handled rigorously by cascading each layer's transmission (ABCD) matrix and
#  terminating both sides in the incident wave impedance.  For a single layer
#  this reduces *exactly* to the A + R + B result above, so it is a strict
#  generalisation, not a separate model.  ``layers`` is a list of
#  ``(sigma, mu_r, thickness_m)`` (``mu_r`` may be a per-frequency array).

_GT_CAP = 150.0   # cap on per-layer attenuation (t/δ) to keep cosh/sinh finite;
                  # SE beyond ~1300 dB/layer is physically unmeasurable anyway.


def _layer_abcd(f, sigma, mu_r, t_m):
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    Zm = np.sqrt(1j * w * mu_r * MU0 / sigma)          # intrinsic impedance
    gamma = np.sqrt(1j * w * mu_r * MU0 * sigma)        # propagation constant
    gt = gamma * float(t_m)
    gt = np.minimum(gt.real, _GT_CAP) + 1j * gt.imag    # avoid cosh/sinh overflow
    ch, sh = np.cosh(gt), np.sinh(gt)
    return ch, Zm * sh, sh / Zm, ch                     # A, B, C, D


def multilayer_se_db(f, layers, r=1.0, field=FIELD_PLANE):
    """Shielding effectiveness [dB] of a stack of layers via the ABCD cascade.

    SE = 20·log10 |(A·Zw + B + C·Zw² + D·Zw) / (2·Zw)|,  Zw the wave impedance.
    """
    f = np.asarray(f, dtype=float)
    Zw = wave_impedance(f, r, field)
    A = np.ones_like(f, dtype=complex)
    B = np.zeros_like(f, dtype=complex)
    C = np.zeros_like(f, dtype=complex)
    D = np.ones_like(f, dtype=complex)
    for sigma, mu_r, t_m in layers:
        a, b, c, d = _layer_abcd(f, sigma, mu_r, t_m)
        A, B, C, D = A * a + B * c, A * b + B * d, C * a + D * c, C * b + D * d
    return 20.0 * np.log10(np.abs((A * Zw + B + C * Zw ** 2 + D * Zw) / (2.0 * Zw)))


def shield_mass_kg(layers, area_m2):
    """Total mass [kg] of a laminate over ``area_m2``.

    ``layers`` is a list of ``(density_kg_m3, thickness_m)``.
    """
    return float(sum(rho * t * float(area_m2) for rho, t in layers))


def _material_mu(freqs, m, use_rolloff):
    if use_rolloff and m.get("f_mu"):
        return effective_mu_r(freqs, m["mu_r"], m["f_mu"])
    return m["mu_r"]


def required_se_curve(freqs, source_dbuv, limit_dbuv):
    """Required shielding SE(f) [dB] to bring a source under a limit.

        SE_required(f) = max(0, source_field(f) − limit(f))

    ``source_dbuv`` and ``limit_dbuv`` are field strengths [dBµV/m] on ``freqs``
    (scalars broadcast).  Where the limit is NaN (out of a standard's scope) the
    requirement is 0 (no constraint).
    """
    src = np.broadcast_to(np.asarray(source_dbuv, float), np.shape(freqs)).astype(float)
    lim = np.broadcast_to(np.asarray(limit_dbuv, float), np.shape(freqs)).astype(float)
    req = src - lim
    req = np.where(np.isnan(req), 0.0, req)
    return np.clip(req, 0.0, None)


def best_laminates(f_lo, f_hi, req_se_db, r=1.0, field=FIELD_PLANE,
                   t_max_each_mm=3.0, area_m2=1.0, use_mu_rolloff=True,
                   n_grid=8, n_freq=60, materials=None, include_laminates=True,
                   t_floor_mm=0.01):
    """Search single materials and **two-layer laminates** for the shield that
    meets the requirement across the band with the least mass.

    ``req_se_db`` may be a **scalar** (flat SE target) OR a **callable**
    ``req(freqs)->array`` (a frequency-dependent requirement, e.g. from
    :func:`required_se_curve`).  A design "meets" the requirement when its
    SE(f) ≥ req(f) at **every** frequency, i.e. the worst SE margin ≥ 0.

    Returns dicts: ``layers`` = ``[(material, thickness_mm), ...]``, ``se``
    (worst SE across band), ``margin`` (worst SE−req, dB, ≥0), ``mass_kg``,
    ``cost_usd`` (real, mass × representative price), ``thickness_mm`` (total).
    """
    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_freq)
    req = np.asarray(req_se_db(freqs) if callable(req_se_db) else req_se_db, float)
    names = list(materials or MATERIALS.keys())
    t_max = t_max_each_mm / 1000.0
    # Manufacturing floor: in reflection-limited (far-field) cases the required
    # SE is met by a sub-micron foil, which is meaningless and displays as 0.
    t_floor = min(t_floor_mm / 1000.0, t_max)
    grid = np.linspace(max(t_floor, t_max / n_grid), t_max, n_grid)
    cands = []

    def _cost(layers_m_t):        # real USD = Σ price[$/kg]·density·t·area
        return float(sum(m["price_per_kg"] * m["density"] * t * area_m2
                         for m, t in layers_m_t))

    # single materials (bisection on thickness, clamped at the floor)
    for nm in names:
        m = MATERIALS[nm]
        mu = _material_mu(freqs, m, use_mu_rolloff)

        def margin(t, _sig=m["sigma"], _mu=mu):
            return float(np.min(multilayer_se_db(freqs, [(_sig, _mu, t)], r, field) - req))

        if margin(t_max) < 0:
            continue
        if margin(t_floor) >= 0:
            hi = t_floor                    # requirement met at the floor
        else:
            lo, hi = t_floor, t_max
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if margin(mid) >= 0:
                    hi = mid
                else:
                    lo = mid
        se_curve = multilayer_se_db(freqs, [(m["sigma"], mu, hi)], r, field)
        cands.append({"layers": [(nm, hi * 1000)], "se": float(np.min(se_curve)),
                      "margin": float(np.min(se_curve - req)),
                      "mass_kg": shield_mass_kg([(m["density"], hi)], area_m2),
                      "cost_usd": _cost([(m, hi)]), "thickness_mm": hi * 1000})

    # two-layer laminates (thickness grid, keep the lightest that passes)
    for i, n1 in enumerate(names if include_laminates else []):
        m1 = MATERIALS[n1]; mu1 = _material_mu(freqs, m1, use_mu_rolloff)
        for n2 in names[i + 1:]:
            m2 = MATERIALS[n2]; mu2 = _material_mu(freqs, m2, use_mu_rolloff)
            best = None
            for t1 in grid:
                for t2 in grid:
                    se_curve = multilayer_se_db(
                        freqs, [(m1["sigma"], mu1, t1), (m2["sigma"], mu2, t2)], r, field)
                    mg = float(np.min(se_curve - req))
                    if mg >= 0:
                        mass = shield_mass_kg([(m1["density"], t1), (m2["density"], t2)], area_m2)
                        if best is None or mass < best["mass_kg"]:
                            best = {"layers": [(n1, t1 * 1000), (n2, t2 * 1000)],
                                    "se": float(np.min(se_curve)), "margin": mg, "mass_kg": mass,
                                    "cost_usd": _cost([(m1, t1), (m2, t2)]),
                                    "thickness_mm": (t1 + t2) * 1000}
            if best:
                cands.append(best)

    cands.sort(key=lambda c: c["mass_kg"])
    return cands


# ─────────────────────────────────────────────────────────────────────────────
#  Free-space propagation helpers
# ─────────────────────────────────────────────────────────────────────────────
def wavelength(f):
    """Wavelength λ = c/f  [m]."""
    return C0 / np.asarray(f, dtype=float)


def near_far_boundary(f):
    """Reactive near-field boundary r = λ/2π  [m] (edge of the reactive region)."""
    return wavelength(f) / (2.0 * np.pi)


def fraunhofer_distance(aperture_m, f):
    """Far-field (Fraunhofer) boundary r = 2·D²/λ  [m] for an antenna/source of
    largest dimension ``D``.  Beyond this the pattern is fully formed and fields
    fall as 1/r.  (The rough ``2λ`` used previously ignores the source size.)"""
    return 2.0 * float(aperture_m) ** 2 / wavelength(f)


def friis_path_loss_db(f, r):
    """Free-space path loss [dB] = 20·log10(4π r / λ)."""
    lam = wavelength(f)
    return 20.0 * np.log10(4.0 * np.pi * np.maximum(np.asarray(r, float), 1e-9) / lam)


def friis_received_dbm(p_tx_dbm, f, r, g_tx_dbi=0.0, g_rx_dbi=0.0):
    """Received power [dBm] via Friis:  Pr = Pt + Gt + Gr − FSPL."""
    return (float(p_tx_dbm) + float(g_tx_dbi) + float(g_rx_dbi)
            - friis_path_loss_db(f, r))


# ─────────────────────────────────────────────────────────────────────────────
#  Emanation coverage & interception (spatial propagation)
# ─────────────────────────────────────────────────────────────────────────────
def radiation_pattern_db(theta, pattern="isotropic"):
    """Radiation-pattern gain [dB] relative to isotropic vs angle ``theta`` [rad].

    * ``isotropic`` — 0 dB everywhere.
    * ``dipole``    — short-dipole power pattern sin²θ (a figure-8; a small floor
      avoids −∞ along the axis).  Representative of a wire/cable radiator.
    """
    theta = np.asarray(theta, dtype=float)
    if pattern == "dipole":
        return 10.0 * np.log10(np.sin(theta) ** 2 + 1e-3)
    return np.zeros_like(theta)


def two_ray_factor_db(r, f, h_tx=1.0, h_rx=1.0, gamma=-1.0):
    """Two-ray (direct + ground-reflected) field factor [dB] relative to the
    direct free-space field, vs horizontal distance ``r``.

    Produces the characteristic interference lobes/nulls of real ground
    propagation; well beyond the last lobe the two rays cancel and the field
    rolls off faster than free space.
    """
    lam = wavelength(f)
    r = np.asarray(r, dtype=float)
    d1 = np.sqrt(r ** 2 + (h_tx - h_rx) ** 2)          # direct path
    d2 = np.sqrt(r ** 2 + (h_tx + h_rx) ** 2)          # reflected path
    dphi = 2.0 * np.pi * (d2 - d1) / lam
    field = np.abs(1.0 + gamma * (d1 / d2) * np.exp(-1j * dphi))
    return 20.0 * np.log10(field + 1e-9)


def coverage_field_dbuv(emission_1m_dbuv, X, Y, pattern="isotropic",
                        two_ray=False, f=1e8, h_tx=1.0, h_rx=1.0):
    """2-D field-strength map [dBµV/m] around a source at the origin.

    ``emission_1m_dbuv`` is the source field at the 1 m reference distance; the
    map applies 1/r free-space decay, the radiation ``pattern`` and (optionally)
    ``two_ray`` ground-reflection multipath.
    """
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    r = np.hypot(X, Y)
    field = (float(emission_1m_dbuv)
             - free_space_field_decay_db(np.maximum(r, REF_DISTANCE), REF_DISTANCE))
    if pattern != "isotropic":
        field = field + radiation_pattern_db(np.arctan2(Y, X), pattern)
    if two_ray:
        field = field + two_ray_factor_db(r, f, h_tx, h_rx)
    return field


def interception_range_m(emission_1m_dbuv, threshold_dbuv, pattern_peak_db=0.0):
    """Maximum free-space distance [m] at which the field (in the pattern's
    strongest direction) drops to ``threshold_dbuv``:  r = 10^((E0+G−T)/20)."""
    return max(REF_DISTANCE, 10.0 ** ((float(emission_1m_dbuv) + pattern_peak_db
                                       - float(threshold_dbuv)) / 20.0))


def free_space_field_decay_db(r, r_ref=1.0):
    """Field-strength (1/r) attenuation relative to a reference distance [dB].

    Returns a *positive* attenuation.  20·log10(r/r_ref); a doubling of distance
    is −6 dB of field strength.
    """
    r = np.asarray(r, dtype=float)
    return 20.0 * np.log10(np.maximum(r, 1e-9) / r_ref)


def dbm_to_field_dbuv_per_m(p_dbm, gain_dbi=0.0, r=1.0):
    """Rough conversion from radiated power [dBm] to field strength [dBµV/m] at
    distance ``r`` for an isotropic-ish source, using the far-field relation
    E = √(30 · P · G)/r.  Provided so the suite can express results in the
    dBµV/m units that TEMPEST limits are actually written in (NATO SDIP-27),
    instead of the ambiguous "dBm" that appeared on earlier plots.
    """
    p_w = 10.0 ** ((p_dbm - 30.0) / 10.0)          # dBm → W
    g   = 10.0 ** (gain_dbi / 10.0)
    e_v_per_m = np.sqrt(30.0 * p_w * g) / np.maximum(r, 1e-9)
    return 20.0 * np.log10(e_v_per_m / 1e-6)        # V/m → dBµV/m


# ─────────────────────────────────────────────────────────────────────────────
#  Aperture / seam leakage   (usually the *dominant* limit on real enclosures)
# ─────────────────────────────────────────────────────────────────────────────
#  A perfectly-conducting box would have infinite SE; in practice seams, vents,
#  connector holes and display cut-outs radiate and set the real shielding limit.
#  Follows Ott [1, §6.9]: slot radiation + waveguide-below-cutoff + N-aperture
#  correction, combined with the bulk-material SE as parallel leakage paths.

def slot_leakage_db(f, slot_length_m, n_apertures=1):
    """Aperture radiation SE = 20·log10(λ / 2L) − 10·log10(N)  [dB].

    ``slot_length_m`` is the *longest* linear dimension of the aperture; SE falls
    to 0 at the slot resonance L = λ/2 (a half-wave slot radiates freely).  N
    apertures spaced within λ/2 degrade SE by 10·log10(N).
    """
    lam = wavelength(f)
    se = 20.0 * np.log10(lam / (2.0 * float(slot_length_m)))
    se = np.clip(se, 0.0, None)
    if n_apertures and n_apertures > 1:
        se = se - 10.0 * np.log10(float(n_apertures))
    return np.clip(se, 0.0, None)


def waveguide_below_cutoff_db(f, transverse_dim_m, depth_m, shape="rectangular"):
    """Attenuation of an aperture behaving as a waveguide below cutoff  [dB].

    A hole with depth (panel thickness, or a honeycomb-vent tube) is a waveguide;
    below its cutoff frequency the field is evanescent and strongly attenuated:

        A = 8.686 · (2π/λc)·√(1 − (f/fc)²) · depth      (0 above cutoff)

    with λc = 2·w (rectangular) or 1.706·d (circular).  Well below cutoff this
    reduces to Ott's rules of thumb 27.3·t/w (rect) and 32·t/d (round).
    """
    f = np.asarray(f, dtype=float)
    lam_c = (1.706 if shape == "circular" else 2.0) * float(transverse_dim_m)
    fc = C0 / lam_c
    ratio = np.clip(f / fc, 0.0, 1.0)
    alpha = (2.0 * np.pi / lam_c) * np.sqrt(1.0 - ratio ** 2)   # 1/m, evanescent
    A = 8.686 * alpha * float(depth_m)
    return np.where(f < fc, A, 0.0)


def aperture_se_db(f, length_m, width_m, depth_m, n_apertures=1,
                   shape="rectangular"):
    """Total SE of one aperture type: slot radiation + waveguide-below-cutoff."""
    return (slot_leakage_db(f, length_m, n_apertures)
            + waveguide_below_cutoff_db(f, width_m, depth_m, shape))


def combine_se_db(*se_arrays):
    """Combine parallel leakage paths.  Leaked fields add, so

        SE_total = −20·log10( Σ 10^(−SE_i/20) )

    The total is always ≤ the worst individual path — a superb wall with one bad
    seam is only as good as the seam.
    """
    leak = np.zeros_like(np.asarray(se_arrays[0], dtype=float))
    for se in se_arrays:
        leak = leak + 10.0 ** (-np.asarray(se, dtype=float) / 20.0)
    return -20.0 * np.log10(leak)


def enclosure_se_db(f, sigma, mu_r, t_m, apertures=None, r=1.0, field=FIELD_PLANE):
    """Overall enclosure SE combining the bulk material with a list of apertures.

    ``apertures`` is a list of dicts with keys ``length``, ``width`` [m],
    optional ``depth`` (defaults to the panel thickness ``t_m``), ``count`` and
    ``shape`` ("rectangular" | "circular").
    """
    material = shielding_effectiveness(f, sigma, mu_r, t_m, r, field)["SE"]
    paths = [material]
    for ap in (apertures or []):
        paths.append(aperture_se_db(
            f, ap["length"], ap["width"], ap.get("depth", t_m),
            ap.get("count", 1), ap.get("shape", "rectangular")))
    return combine_se_db(*paths)


# ─────────────────────────────────────────────────────────────────────────────
#  Emanation source model  &  NATO SDIP-27 style zoning
# ─────────────────────────────────────────────────────────────────────────────
#  NOTE ON MODELLING HONESTY
#  -------------------------
#  Real compromising-emanation levels vary by orders of magnitude with the
#  specific unit, cabling and environment, and can only be established by
#  measurement (IEEE 299 / NATO SDIP-27 test procedures).  The values below are
#  *representative, illustrative* baselines chosen so that the visualiser
#  produces plausible, self-consistent, and — crucially — **reproducible**
#  results.  They are NOT calibrated measurements and must not be cited as
#  absolute emission figures.  ``emission_dbuv`` is a nominal radiated field
#  strength [dBµV/m] at the reference distance ``REF_DISTANCE`` (1 m), ranking
#  devices by their well-documented relative emanation risk (e.g. an unshielded
#  CRT / video cable radiates far more than a shielded Ethernet link [5]).

REF_DISTANCE          = 1.0     # reference measurement distance [m]
DETECTION_FLOOR_DBUV  = 20.0    # nominal attacker detectability floor [dBµV/m]

DEVICES: dict[str, dict] = {
    "CRT Monitor":     {"peaks": [15625, 31250, 62500, 125000], "harmonics": 8, "noise": 0.15, "emission_dbuv": 45.0},
    "LCD Monitor":     {"peaks": [67500, 135000, 270000],       "harmonics": 5, "noise": 0.08, "emission_dbuv": 40.0},
    "CPU (3 GHz)":     {"peaks": [100e6, 300e6, 600e6, 1e9],    "harmonics": 4, "noise": 0.20, "emission_dbuv": 41.0},
    "GPU":             {"peaks": [150e6, 300e6, 900e6],          "harmonics": 5, "noise": 0.22, "emission_dbuv": 42.0},
    "HDMI Cable":      {"peaks": [165e6, 330e6, 495e6, 660e6],   "harmonics": 6, "noise": 0.10, "emission_dbuv": 44.0},
    "USB 3.0":         {"peaks": [2.5e8, 5e8],                   "harmonics": 3, "noise": 0.09, "emission_dbuv": 38.0},
    "Ethernet (1Gb)":  {"peaks": [125e6, 250e6],                 "harmonics": 4, "noise": 0.07, "emission_dbuv": 37.0},
    "Keyboard (PS/2)": {"peaks": [12000, 24000, 48000],          "harmonics": 6, "noise": 0.05, "emission_dbuv": 36.0},
}

# Field-strength contour levels [dBµV/m] defining each emanation-security zone.
ZONE_THRESHOLDS_DBUV: dict[str, float] = {
    "Zone 0 (Controlled)": 42.0,
    "Zone 1 (Sensitive)":  36.0,
    "Zone 2 (Monitored)":  30.0,
    "Zone 3 (Public)":     23.0,
}


def _effective_emission_dbuv(device, power_dbm=10.0):
    """Nominal field at 1 m [dBµV/m], shifted by the declared source power
    relative to the 10 dBm reference used to set the baselines."""
    return DEVICES[device]["emission_dbuv"] + (float(power_dbm) - 10.0)


def field_at_distance_dbuv(device, power_dbm=10.0, r=1.0):
    """Radiated field strength [dBµV/m] at distance ``r`` using 1/r (−20 dB per
    decade) free-space decay.  Clamped at ``REF_DISTANCE`` so the near-field does
    not blow up unphysically."""
    e0 = _effective_emission_dbuv(device, power_dbm)
    return e0 - free_space_field_decay_db(max(float(r), REF_DISTANCE), REF_DISTANCE)


def zone_radius_m(device, power_dbm, threshold_dbuv):
    """Radius [m] of the field-strength contour where the emanation drops to
    ``threshold_dbuv``.  From E(r) = E0 − 20·log10(r) ⇒ r = 10^((E0−T)/20)."""
    e0 = _effective_emission_dbuv(device, power_dbm)
    return max(0.2, 10.0 ** ((e0 - threshold_dbuv) / 20.0))


def leaked_field_dbuv(device, power_dbm, wall_distance_m, wall_se_db):
    """Field strength [dBµV/m] that escapes the room: the internal field reaching
    the nearest wall, attenuated by the wall's shielding effectiveness."""
    return (field_at_distance_dbuv(device, power_dbm, wall_distance_m)
            - float(wall_se_db))


def room_risk_score(devices, room_w, room_l, wall_se_db):
    """Physically-grounded TEMPEST risk score for a room layout.

    For each device the field reaching the *nearest wall* is computed, attenuated
    by the wall SE, and compared with the attacker detectability floor.  The
    per-device exceedance (dB above the floor) is accumulated into a 0–100 score.
    Unlike the previous purely-geometric heuristic, this uses the device type,
    its power, its position **and the wall shielding**.

    Parameters
    ----------
    devices : list of dict with keys ``type``, ``x``, ``y``, ``power``.
    room_w, room_l : float   room dimensions [m].
    wall_se_db : float       wall shielding effectiveness [dB].

    Returns
    -------
    (score, details) : (float in [0, 100], list of per-device dicts)
    """
    total = 0.0
    details = []
    for d in devices:
        # distance from the device to the closest wall (0.1 m floor)
        d_wall = max(0.1, min(d["x"], room_w - d["x"],
                              d["y"], room_l - d["y"]))
        field_wall = field_at_distance_dbuv(d["type"], d["power"], d_wall)
        leaked = field_wall - float(wall_se_db)
        margin = leaked - DETECTION_FLOOR_DBUV        # dB above detectability
        points = float(np.clip(margin, 0.0, 50.0) * 2.0)   # 0..100 per device
        total += points
        details.append({
            **d, "wall_distance": d_wall, "field_at_wall": field_wall,
            "leaked_field": leaked, "margin_db": margin, "risk_points": points,
        })
    return float(min(100.0, total)), details


def emanation_spectrum(device, distance=1.0, n=4000, harmonics=True, seed=0):
    """Reproducible synthetic emanation spectrum for the visualiser.

    Contributions are summed in the **linear** field domain (µV/m) — physically
    correct superposition — then converted to dBµV/m at the end.  The noise floor
    is drawn from a *seeded* generator so the same inputs always give the same
    figure (essential for a reproducible thesis plot).

    Returns
    -------
    (freqs [Hz], field_dbuv [dBµV/m]) : both 1-D arrays of length ``n``.
    """
    dev = DEVICES[device]
    rng = np.random.default_rng(seed)
    freqs = np.linspace(1.0, max(dev["peaks"]) * 1.5, n)
    r = max(float(distance), REF_DISTANCE)

    floor_lin = 10.0 ** (5.0 / 20.0)                     # ~5 dBµV/m noise floor
    spec = floor_lin * (1.0 + dev["noise"] * rng.random(n))
    e0_lin = 10.0 ** (dev["emission_dbuv"] / 20.0)       # peak field at 1 m
    n_harm = dev["harmonics"] if harmonics else 1
    bw = max(1, int(n * 0.003))
    for p in dev["peaks"]:
        for h in range(1, n_harm + 1):
            f = p * h
            if f < freqs[-1]:
                amp = e0_lin / r / (h ** 1.5)            # 1/r decay + roll-off
                idx = int(np.argmin(np.abs(freqs - f)))
                lo, hi = max(0, idx - bw), min(n, idx + bw)
                x = np.arange(lo, hi)
                spec[lo:hi] += amp * np.exp(-((x - idx) ** 2) / (2 * (bw / 3) ** 2))
    return freqs, 20.0 * np.log10(spec + 1e-6)


__all__ = [
    "MU0", "EPS0", "C0", "ETA0", "SIGMA_CU", "MATERIALS",
    "FIELD_PLANE", "FIELD_ELECTRIC", "FIELD_MAGNETIC", "FIELD_TYPES",
    "skin_depth", "shield_impedance", "effective_mu_r", "wave_impedance",
    "absorption_loss_db", "reflection_loss_db", "multiple_reflection_db",
    "shielding_effectiveness", "min_thickness_for_se",
    "multilayer_se_db", "shield_mass_kg", "best_laminates", "required_se_curve",
    "slot_leakage_db", "waveguide_below_cutoff_db", "aperture_se_db",
    "combine_se_db", "enclosure_se_db",
    "wavelength", "near_far_boundary", "fraunhofer_distance",
    "friis_path_loss_db", "friis_received_dbm", "free_space_field_decay_db",
    "dbm_to_field_dbuv_per_m",
    "radiation_pattern_db", "two_ray_factor_db", "coverage_field_dbuv",
    "interception_range_m",
    "REF_DISTANCE", "DETECTION_FLOOR_DBUV", "DEVICES", "ZONE_THRESHOLDS_DBUV",
    "field_at_distance_dbuv", "zone_radius_m", "leaked_field_dbuv",
    "room_risk_score", "emanation_spectrum",
]
