"""
tempest_signal.py
================
Spectral *interpretation* for the Signal Capture & Analysis module: turn a
spectrum into a diagnosis.  Given a computed spectrum it

* **detects peaks** (prominence-based),
* groups them into **harmonic families** (fundamental + integer multiples — the
  signature of a clocked digital source such as a pixel clock or sync), and
* **classifies** the capture against the device emanation library (which known
  device does this comb of peaks look like?).

GUI-free and unit-tested.  Depends on ``tempest_physics`` for the device library.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks as _find_peaks

import tempest_physics as tp


def detect_peaks(freqs, mag_db, prominence_db=6.0, max_peaks=12, min_freq=0.0):
    """Return the strongest spectral peaks as ``[(freq_Hz, level_dB), ...]``,
    sorted by level (descending).  ``prominence_db`` sets how much a peak must
    stand out from the surrounding spectrum; ``min_freq`` skips DC / very low
    bins.
    """
    freqs = np.asarray(freqs, dtype=float)
    mag = np.asarray(mag_db, dtype=float)
    mask = freqs >= float(min_freq)
    if not np.any(mask):
        return []
    fsel, msel = freqs[mask], mag[mask]
    idx, _ = _find_peaks(msel, prominence=float(prominence_db))
    peaks = [(float(fsel[i]), float(msel[i])) for i in idx]
    peaks.sort(key=lambda t: -t[1])
    return peaks[:int(max_peaks)]


def harmonic_families(peak_freqs, tol=0.03, min_harmonics=3):
    """Group peaks into harmonic families.

    For each candidate fundamental f0, collect peaks lying within ``tol`` (a
    fractional tolerance) of an integer multiple n·f0.  Families with at least
    ``min_harmonics`` members are returned, richest first, as dicts with keys
    ``fundamental``, ``harmonics`` (list of ``(n, freq)``) and ``count``.
    """
    peaks = sorted(float(f) for f in peak_freqs if f > 0)
    families, used = [], set()
    for f0 in peaks:
        if f0 in used:
            continue
        harmonics = []
        for f in peaks:
            n = int(round(f / f0))
            if n >= 1 and abs(f - n * f0) <= tol * n * f0:
                harmonics.append((n, f))
        if len(harmonics) >= int(min_harmonics):
            families.append({"fundamental": f0, "harmonics": harmonics,
                             "count": len(harmonics)})
            used.update(f for _, f in harmonics)
    return sorted(families, key=lambda d: -d["count"])


def classify_device(peak_freqs, tol=0.05):
    """Score every device in the library by how many of its fundamental
    emanation frequencies appear among ``peak_freqs``.

    Returns ``[{device, matched, n_peaks, score}, ...]`` ranked by score
    (fraction of the device's peaks matched), best first.
    """
    peaks = [float(f) for f in peak_freqs]
    out = []
    for name, dev in tp.DEVICES.items():
        exp = dev["peaks"]
        matched = sum(1 for e in exp
                      if any(abs(pk - e) <= tol * e for pk in peaks))
        out.append({"device": name, "matched": matched, "n_peaks": len(exp),
                    "score": matched / max(1, len(exp))})
    return sorted(out, key=lambda d: (-d["score"], -d["matched"]))


__all__ = ["detect_peaks", "harmonic_families", "classify_device"]
