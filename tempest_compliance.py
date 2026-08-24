"""
tempest_compliance.py
====================
Emission-limit masks and pass/fail evaluation for the TEMPEST Analysis Suite.

Overlaying a standardised limit line on a measured or simulated spectrum turns a
plot into an assessment: every point above the mask is a violation.  This module
provides the masks and the evaluation; the GUI draws them.

Honesty about the standards
---------------------------
* **CISPR 32 (EN 55032)** and **FCC Part 15** radiated-emission limits below
  1 GHz are public and reproduced here at their normative measurement distances.
* **NATO SDIP-27** limits are *controlled/undisclosed*.  The Level A/B/C curves
  here are **illustrative placeholders** (flagged ``illustrative=True``) so the
  workflow can be demonstrated — they must NOT be cited as the real limits.

A mask is a list of segments ``(f_start_Hz, f_end_Hz, level_start, level_end)``
interpolated linearly in log-frequency; frequencies outside every segment are
out of the standard's scope and are excluded from the pass/fail decision.

References
----------
[1] CISPR 32:2015 / EN 55032, radiated emission limits (Class A/B).
[2] FCC 47 CFR Part 15 Subpart B, radiated emission limits.
"""

from __future__ import annotations

import numpy as np

import tempest_physics as tp

# Each standard: measurement distance (m), whether illustrative, and segments
# (f0_Hz, f1_Hz, level0_dBµV/m, level1_dBµV/m).  Flat limits use level0 == level1.
STANDARDS: dict[str, dict] = {
    "CISPR 32 Class A (10 m)": {
        "distance_m": 10.0, "illustrative": False,
        "segments": [(30e6, 230e6, 40.0, 40.0), (230e6, 1000e6, 47.0, 47.0)],
    },
    "CISPR 32 Class B (10 m)": {
        "distance_m": 10.0, "illustrative": False,
        "segments": [(30e6, 230e6, 30.0, 30.0), (230e6, 1000e6, 37.0, 37.0)],
    },
    "FCC Part 15 Class B (3 m)": {
        "distance_m": 3.0, "illustrative": False,
        "segments": [(30e6, 88e6, 40.0, 40.0), (88e6, 216e6, 43.5, 43.5),
                     (216e6, 960e6, 46.0, 46.0), (960e6, 1000e6, 54.0, 54.0)],
    },
    "NATO SDIP-27 Level A (illustrative)": {
        "distance_m": 1.0, "illustrative": True,
        "segments": [(1e4, 1e9, 20.0, 20.0)],
    },
    "NATO SDIP-27 Level B (illustrative)": {
        "distance_m": 1.0, "illustrative": True,
        "segments": [(1e4, 1e9, 40.0, 40.0)],
    },
    "NATO SDIP-27 Level C (illustrative)": {
        "distance_m": 1.0, "illustrative": True,
        "segments": [(1e4, 1e9, 60.0, 60.0)],
    },
}


def standard_names():
    return list(STANDARDS.keys())


def is_illustrative(standard):
    return bool(STANDARDS[standard]["illustrative"])


def mask_levels(freqs, standard, at_distance_m=None):
    """Limit level [dBµV/m] at each frequency for ``standard``.

    Points outside every segment return ``np.nan`` (out of scope).  If
    ``at_distance_m`` is given, the mask is distance-corrected from the
    standard's measurement distance using free-space 1/r scaling so it can be
    compared against data taken at a different distance:

        limit(at_distance) = limit(spec) + 20·log10(d_spec / d_at)
    """
    if standard not in STANDARDS:
        raise KeyError(standard)
    spec = STANDARDS[standard]
    freqs = np.asarray(freqs, dtype=float)
    out = np.full(freqs.shape, np.nan, dtype=float)
    for f0, f1, l0, l1 in spec["segments"]:
        m = (freqs >= f0) & (freqs <= f1)
        if not np.any(m):
            continue
        if l0 == l1:
            out[m] = l0
        else:                              # linear in log-frequency
            t = (np.log10(freqs[m]) - np.log10(f0)) / (np.log10(f1) - np.log10(f0))
            out[m] = l0 + t * (l1 - l0)
    if at_distance_m:
        out = out + tp.free_space_field_decay_db(spec["distance_m"], float(at_distance_m))
    return out


def evaluate(freqs, levels_dbuv, standard, at_distance_m=None):
    """Compare a spectrum against a standard's mask.

    Returns a dict:
      * ``limits``        the mask level at each frequency (NaN out of scope)
      * ``in_scope``      boolean mask of evaluated points
      * ``margin``        levels − limits (positive = exceedance) where in scope
      * ``passed``        True if no in-scope point exceeds the limit
      * ``worst_margin``  largest exceedance in dB (−inf-safe: NaN if nothing in scope)
      * ``worst_freq``    frequency of the worst point
      * ``n_exceed``      number of exceeding points
    """
    freqs = np.asarray(freqs, dtype=float)
    levels = np.asarray(levels_dbuv, dtype=float)
    limits = mask_levels(freqs, standard, at_distance_m)
    in_scope = ~np.isnan(limits)
    result = {"limits": limits, "in_scope": in_scope,
              "margin": np.where(in_scope, levels - limits, np.nan)}
    if not np.any(in_scope):
        result.update(passed=True, worst_margin=float("nan"),
                      worst_freq=float("nan"), n_exceed=0)
        return result
    margins = levels[in_scope] - limits[in_scope]
    scope_freqs = freqs[in_scope]
    idx = int(np.argmax(margins))
    result.update(passed=bool(np.all(margins <= 0.0)),
                  worst_margin=float(margins[idx]),
                  worst_freq=float(scope_freqs[idx]),
                  n_exceed=int(np.sum(margins > 0.0)))
    return result


__all__ = ["STANDARDS", "standard_names", "is_illustrative",
           "mask_levels", "evaluate"]
