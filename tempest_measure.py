"""
tempest_measure.py
=================
Ingestion and calibration of **real measured** emanation data for the TEMPEST
Analysis Suite.  GUI-free and unit-tested like the other core modules.

The point of this module is to move the suite from illustrative baselines to
data-driven ones: import a spectrum-analyzer / EMI-receiver trace, convert the
receiver reading into an absolute electric field strength using the measurement
chain, and (optionally) use it to calibrate the device emission baselines that
the simulator and room model rely on.

Calibration chain (standard EMC radiated-emission measurement)
--------------------------------------------------------------
    E[dBµV/m] = V[dBµV] + AF[dB/m] + cable_loss[dB] − gain[dB]

where
    V           receiver reading at its input (dBµV),
    AF          antenna factor of the measuring antenna (may vary with freq),
    cable_loss  loss of the cable between antenna and receiver (added back),
    gain        gain of any preamp/LNA in the chain (subtracted out).

References
----------
[1] CISPR 16-1 / ANSI C63.x radiated-emission measurement methodology.
[2] H. W. Ott, *Electromagnetic Compatibility Engineering*, Wiley, 2009.
"""

from __future__ import annotations

import numpy as np

import tempest_physics as tp

_FREQ_UNITS = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


# ─────────────────────────────────────────────────────────────────────────────
#  Unit conversions
# ─────────────────────────────────────────────────────────────────────────────
def dbm_to_dbuv(p_dbm, impedance_ohm=50.0):
    """Convert power [dBm] to voltage [dBµV] across ``impedance_ohm``.

    dBµV = dBm + 90 + 10·log10(R).  For the usual 50 Ω this is + ~107 dB.
    """
    return np.asarray(p_dbm, float) + 90.0 + 10.0 * np.log10(impedance_ohm)


def dbuv_to_dbm(v_dbuv, impedance_ohm=50.0):
    """Inverse of :func:`dbm_to_dbuv`."""
    return np.asarray(v_dbuv, float) - 90.0 - 10.0 * np.log10(impedance_ohm)


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration chain
# ─────────────────────────────────────────────────────────────────────────────
def field_strength_dbuv_per_m(v_dbuv, antenna_factor_db=0.0,
                              cable_loss_db=0.0, gain_db=0.0):
    """Apply the measurement chain to convert a receiver reading to field
    strength.  ``antenna_factor_db`` may be a scalar or a per-point array
    (same length as ``v_dbuv``) for a frequency-dependent antenna factor.

        E = V + AF + cable_loss − gain
    """
    return (np.asarray(v_dbuv, float) + np.asarray(antenna_factor_db, float)
            + float(cable_loss_db) - float(gain_db))


def reference_offset_db(measured_level_db, known_level_dbuv):
    """Reference-level calibration constant for an SDR.

    Feed a known CW signal of ``known_level_dbuv`` into the receiver, read the
    peak the SDR reports (``measured_level_db``, in its arbitrary relative units)
    and this returns the offset to add to any future reading so it becomes an
    absolute dBµV value:

        reading_dBµV = reading_relative + reference_offset_db

    After this, the full chain (E = reading_dBµV + AF + cable − gain) yields
    calibrated field strength from live SDR data.
    """
    return float(known_level_dbuv) - float(measured_level_db)


def interp_antenna_factor(freqs, af_freqs, af_values):
    """Interpolate a tabulated antenna factor [dB/m] onto ``freqs`` (log-freq
    linear interpolation; clamped to the table ends)."""
    freqs = np.asarray(freqs, float)
    order = np.argsort(af_freqs)
    af_freqs = np.asarray(af_freqs, float)[order]
    af_values = np.asarray(af_values, float)[order]
    return np.interp(np.log10(freqs),
                     np.log10(af_freqs), af_values,
                     left=af_values[0], right=af_values[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  Trace import
# ─────────────────────────────────────────────────────────────────────────────
def load_trace_csv(path, freq_col=0, level_col=1, freq_unit="Hz",
                   level_is_dbm=False, impedance_ohm=50.0):
    """Load a spectrum-analyzer / EMI-receiver trace from CSV.

    Robust to headers, comment lines (``#``) and comma/semicolon/tab
    delimiters (auto-sniffed).  ``freq_col`` / ``level_col`` may be integer
    indices or column names.

    Returns
    -------
    (freqs_Hz, level_dbuv) : two 1-D float arrays, sorted by frequency.
    ``level_dbuv`` is the raw receiver reading (converted from dBm if
    ``level_is_dbm``) — the calibration chain is applied separately.
    """
    freqs, levels = _read_two_columns(path, freq_col, level_col)
    scale = _FREQ_UNITS.get(freq_unit, 1.0)
    freqs = freqs * scale
    if level_is_dbm:
        levels = dbm_to_dbuv(levels, impedance_ohm)
    order = np.argsort(freqs)
    return freqs[order], levels[order]


def _sniff_has_header(path):
    """True if the first non-comment line is not fully numeric (i.e. a header)."""
    import re
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p for p in re.split(r"[,;\t ]+", s) if p != ""]
            try:
                [float(p) for p in parts]
                return False               # all numeric → headerless
            except ValueError:
                return True                # some text → header row
    return True


def _read_two_columns(path, freq_col, level_col):
    """Pull two columns from a messy CSV, preferring pandas when available."""
    has_header = _sniff_has_header(path)
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python", comment="#",
                         header=0 if has_header else None)
        fc = df.columns[freq_col] if isinstance(freq_col, int) else freq_col
        lc = df.columns[level_col] if isinstance(level_col, int) else level_col
        f = pd.to_numeric(df[fc], errors="coerce").to_numpy(float)
        v = pd.to_numeric(df[lc], errors="coerce").to_numpy(float)
    except Exception:
        # stdlib/numpy fallback (integer columns only)
        if not isinstance(freq_col, int) or not isinstance(level_col, int):
            raise
        data = np.genfromtxt(path, delimiter=None, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        f = data[:, freq_col].astype(float)
        v = data[:, level_col].astype(float)
    mask = np.isfinite(f) & np.isfinite(v)
    if not mask.any():
        raise ValueError("no numeric (frequency, level) rows found in trace")
    return f[mask], v[mask]


# ─────────────────────────────────────────────────────────────────────────────
#  Model calibration from measurement
# ─────────────────────────────────────────────────────────────────────────────
def emission_baseline_from_field(field_dbuv, distance_m):
    """Back-project a measured field-strength trace (at ``distance_m``) to the
    emission baseline at the 1 m reference distance used by the model.

        E0 = max(E_measured) + 20·log10(distance / 1 m)

    The result is directly usable as a ``DEVICES[...]['emission_dbuv']`` value.
    """
    peak = float(np.max(field_dbuv))
    return peak + tp.free_space_field_decay_db(distance_m, tp.REF_DISTANCE)


def calibrate_device(device, field_dbuv, distance_m):
    """Overwrite ``tp.DEVICES[device]['emission_dbuv']`` from a measured trace
    and return the new baseline.  This is how a real measurement replaces the
    illustrative default for that device across the whole suite."""
    if device not in tp.DEVICES:
        raise KeyError(device)
    e0 = emission_baseline_from_field(field_dbuv, distance_m)
    tp.DEVICES[device]["emission_dbuv"] = e0
    tp.DEVICES[device]["measured"] = True
    return e0


__all__ = [
    "dbm_to_dbuv", "dbuv_to_dbm",
    "field_strength_dbuv_per_m", "reference_offset_db", "interp_antenna_factor",
    "load_trace_csv", "emission_baseline_from_field", "calibrate_device",
]
