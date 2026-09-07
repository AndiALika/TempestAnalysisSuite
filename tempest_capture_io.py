"""
tempest_capture_io.py
=====================
Import of externally captured SDR recordings (files on disk) for the TEMPEST
Analysis Suite — the counterpart to ``tempest_sdr.py``'s *live* capture.

This is for the common real-world situation: you (or a colleague, or another
SDR device — RTL-SDR, HackRF, USRP, SDR#, GQRX, ...) already recorded IQ
samples to a file, and want to load and analyse them the same way as a live
capture.  It supports the file conventions actually used by the SDR community:

* **SigMF** (``.sigmf-data`` + ``.sigmf-meta``) — the self-describing standard
  (sample rate, centre frequency and sample format are read from the metadata,
  no guessing required). See https://github.com/sigmf/SigMF.
* **Raw interleaved IQ** — headerless binary, the native output of tools like
  ``rtl_sdr`` (``cu8``), ``hackrf_transfer`` (``cs8``), and GNU Radio file
  sinks (``cf32``/``cs16``). These carry **no metadata**, so the sample format,
  sample rate and centre frequency must be supplied by the caller (the GUI
  prompts for them, pre-filled with sensible defaults).
* **WAV** — some tools (SDR#, HDSDR) save baseband IQ as a stereo WAV file
  (left = I, right = Q); reuses :mod:`tempest_dsp`'s loader.

GUI-free and unit-tested, like the other core modules.
"""

from __future__ import annotations

import json
import os
import struct
import numpy as np

import tempest_dsp as dsp

# ─────────────────────────────────────────────────────────────────────────────
#  Raw sample-format registry
# ─────────────────────────────────────────────────────────────────────────────
#  key -> (numpy dtype, bytes per real sample, normalisation to ~[-1, 1])
#  Matches the conventions of rtl_sdr (cu8), hackrf_transfer (cs8), GNU Radio
#  file sinks (cf32/cs16) and most SigMF captures.
RAW_DTYPES: dict[str, dict] = {
    "cu8":  {"label": "uint8  (cu8 — RTL-SDR native, e.g. rtl_sdr)",
             "dtype": np.uint8,   "normalize": lambda x: (x.astype(np.float32) - 127.5) / 127.5},
    "cs8":  {"label": "int8  (cs8/ci8 — HackRF native, e.g. hackrf_transfer)",
             "dtype": np.int8,    "normalize": lambda x: x.astype(np.float32) / 127.0},
    "cs16": {"label": "int16  (cs16/ci16 — common GNU Radio / SDR export)",
             "dtype": np.int16,   "normalize": lambda x: x.astype(np.float32) / 32767.0},
    "cf32": {"label": "float32  (cf32 — GNU Radio file sink, SigMF default)",
             "dtype": np.float32, "normalize": lambda x: x.astype(np.float32)},
}
DEFAULT_RAW_DTYPE = "cu8"          # the single most common headerless format

# SigMF core:datatype -> one of the RAW_DTYPES keys above.  Only little-endian
# (the overwhelming majority of real captures, x86/ARM native) is supported;
# big-endian raises a clear error rather than silently misreading.
_SIGMF_DATATYPE_MAP = {
    "cu8": "cu8", "ci8": "cs8", "ci16_le": "cs16", "cf32_le": "cf32",
}

RAW_EXTENSIONS = (".iq", ".dat", ".bin", ".cfile", ".cf32", ".fc32",
                  ".cs16", ".ci16", ".s16", ".cs8", ".ci8", ".s8", ".cu8", ".u8")
SIGMF_EXTENSIONS = (".sigmf-data", ".sigmf-meta")
WAV_EXTENSIONS = (".wav",)
ALL_EXTENSIONS = RAW_EXTENSIONS + SIGMF_EXTENSIONS + WAV_EXTENSIONS

MAX_SAMPLES_DEFAULT = 20_000_000    # ~160 MB as complex64; guards against
                                    # accidentally loading a multi-GB capture


# ─────────────────────────────────────────────────────────────────────────────
#  Format detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_format(path):
    """Classify a file as ``"sigmf"``, ``"wav"`` or ``"raw"`` from its name."""
    low = str(path).lower()
    if low.endswith(SIGMF_EXTENSIONS):
        return "sigmf"
    if low.endswith(WAV_EXTENSIONS):
        return "wav"
    return "raw"


def _sigmf_pair(path):
    """Given either half of a SigMF pair, return ``(data_path, meta_path)``."""
    base = str(path)
    if base.endswith(".sigmf-meta"):
        base = base[: -len(".sigmf-meta")]
    elif base.endswith(".sigmf-data"):
        base = base[: -len(".sigmf-data")]
    return base + ".sigmf-data", base + ".sigmf-meta"


# ─────────────────────────────────────────────────────────────────────────────
#  SigMF
# ─────────────────────────────────────────────────────────────────────────────
def read_sigmf_meta(meta_path):
    """Parse a ``.sigmf-meta`` JSON file into a flat, defensively-read dict:
    ``datatype``, ``dtype_key`` (mapped to :data:`RAW_DTYPES`), ``sample_rate``,
    ``frequency`` (centre, from the first capture segment, or None),
    ``description``, ``author``.
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    glob = meta.get("global", {}) or {}
    captures = meta.get("captures", []) or []
    datatype = glob.get("core:datatype")
    dtype_key = _SIGMF_DATATYPE_MAP.get(datatype)
    if dtype_key is None:
        raise ValueError(
            f"Unsupported SigMF core:datatype {datatype!r} — this suite reads "
            f"little-endian cu8/ci8/ci16_le/cf32_le. Convert the capture first "
            f"(e.g. with the 'sigmf' Python package) or open it as raw IQ with "
            f"an explicit format.")
    freq = None
    if captures:
        freq = captures[0].get("core:frequency")
    return {
        "datatype": datatype, "dtype_key": dtype_key,
        "sample_rate": glob.get("core:sample_rate"),
        "frequency": float(freq) if freq is not None else None,
        "description": glob.get("core:description", ""),
        "author": glob.get("core:author", ""),
    }


def load_sigmf(path, max_samples=MAX_SAMPLES_DEFAULT):
    """Load a SigMF recording (either the ``.sigmf-data`` or ``.sigmf-meta``
    half of the pair).  Returns the same dict shape as :func:`load_capture`.
    """
    data_path, meta_path = _sigmf_pair(path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"SigMF metadata not found: {meta_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"SigMF data file not found: {data_path}")
    meta = read_sigmf_meta(meta_path)
    if not meta["sample_rate"]:
        raise ValueError(f"{meta_path} has no core:sample_rate")
    iq, n_total = _read_raw_iq(data_path, meta["dtype_key"], max_samples)
    return {
        "iq": iq, "fs": float(meta["sample_rate"]), "fc": meta["frequency"],
        "format": "sigmf", "dtype_key": meta["dtype_key"], "path": data_path,
        "n_samples": len(iq), "n_samples_total": n_total,
        "duration_s": len(iq) / float(meta["sample_rate"]),
        "truncated": n_total > len(iq),
        "description": meta["description"], "author": meta["author"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Raw interleaved IQ
# ─────────────────────────────────────────────────────────────────────────────
def _read_raw_iq(path, dtype_key, max_samples=MAX_SAMPLES_DEFAULT):
    """Read a headerless interleaved-IQ file → complex64 array.  Returns
    ``(iq, n_complex_samples_in_file)`` — the second value lets the caller
    detect truncation when the file exceeds ``max_samples``."""
    if dtype_key not in RAW_DTYPES:
        raise ValueError(f"Unknown IQ sample format {dtype_key!r}; choose one of "
                         f"{list(RAW_DTYPES)}")
    spec = RAW_DTYPES[dtype_key]
    np_dtype = np.dtype(spec["dtype"])
    file_bytes = os.path.getsize(path)
    n_reals_total = file_bytes // np_dtype.itemsize
    n_complex_total = n_reals_total // 2
    n_complex = min(n_complex_total, int(max_samples))
    raw = np.fromfile(path, dtype=np_dtype, count=n_complex * 2)
    if raw.size % 2:
        raw = raw[:-1]
    norm = spec["normalize"](raw)
    iq = norm[0::2] + 1j * norm[1::2]
    return iq.astype(np.complex64), n_complex_total


def load_raw_iq(path, dtype_key, fs, fc=None, max_samples=MAX_SAMPLES_DEFAULT):
    """Load a headerless raw IQ file with an explicit format/sample-rate
    (the caller — the GUI — is responsible for asking the user for these,
    since raw files carry no self-describing metadata)."""
    if not fs or fs <= 0:
        raise ValueError("sample rate must be given for a raw IQ file")
    iq, n_total = _read_raw_iq(path, dtype_key, max_samples)
    return {
        "iq": iq, "fs": float(fs), "fc": float(fc) if fc is not None else None,
        "format": "raw", "dtype_key": dtype_key, "path": path,
        "n_samples": len(iq), "n_samples_total": n_total,
        "duration_s": len(iq) / float(fs), "truncated": n_total > len(iq),
        "description": "", "author": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WAV (stereo I/Q, as saved by some SDR tools)
# ─────────────────────────────────────────────────────────────────────────────
def load_wav_iq(path, fc=None):
    """Load a stereo WAV file as complex IQ (left=I, right=Q).  A mono WAV is
    loaded as a real-valued (baseband-audio) signal instead."""
    try:
        import soundfile as sf
        data, fs = sf.read(path, always_2d=True, dtype="float32")
    except Exception:
        # Fall back to the stdlib-based mono loader if soundfile is absent or
        # the file needs it; stereo WAV specifically needs channel separation
        # so this fallback only supports mono.
        mono, fs = dsp.load_wav(path, mono=True)
        return {"iq": mono.astype(np.complex64), "fs": float(fs), "fc": fc,
                "format": "wav", "dtype_key": "wav-mono", "path": path,
                "n_samples": len(mono), "n_samples_total": len(mono),
                "duration_s": len(mono) / float(fs), "truncated": False,
                "description": "", "author": ""}
    if data.shape[1] >= 2:
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
    else:
        iq = data[:, 0].astype(np.complex64)
    return {"iq": iq, "fs": float(fs), "fc": fc, "format": "wav",
            "dtype_key": "wav-stereo" if data.shape[1] >= 2 else "wav-mono",
            "path": path, "n_samples": len(iq), "n_samples_total": len(iq),
            "duration_s": len(iq) / float(fs), "truncated": False,
            "description": "", "author": ""}


# ─────────────────────────────────────────────────────────────────────────────
#  Unified entry point
# ─────────────────────────────────────────────────────────────────────────────
def load_capture(path, dtype_key=None, fs_hint=None, fc_hint=None,
                 max_samples=MAX_SAMPLES_DEFAULT):
    """Load any supported capture file, auto-detecting SigMF/WAV/raw from the
    extension.  For **raw** files (no self-describing metadata) ``dtype_key``
    and ``fs_hint`` are required — the GUI collects these from the user.

    Returns a dict: ``iq`` (complex64 ndarray), ``fs``, ``fc`` (may be None),
    ``format``, ``dtype_key``, ``n_samples``, ``n_samples_total``,
    ``duration_s``, ``truncated``, ``path``, ``description``, ``author``.
    """
    fmt = detect_format(path)
    if fmt == "sigmf":
        out = load_sigmf(path, max_samples)
    elif fmt == "wav":
        out = load_wav_iq(path, fc=fc_hint)
    else:
        if not dtype_key:
            raise ValueError(
                "This file has no self-describing metadata (not SigMF/WAV) — "
                "specify the sample format (dtype_key) and sample rate.")
        if not fs_hint:
            raise ValueError("Specify the sample rate (fs_hint) for a raw IQ file.")
        out = load_raw_iq(path, dtype_key, fs_hint, fc_hint, max_samples)
    if fc_hint is not None and out.get("fc") is None:
        out["fc"] = float(fc_hint)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Folder scanning ("a folder full of signal samples")
# ─────────────────────────────────────────────────────────────────────────────
def scan_folder(folder, recursive=True):
    """List every recognised capture file under ``folder`` with a lightweight
    preview (no full data load): filename, format, detected fs/fc where
    self-describing (SigMF), file size, and whether it needs manual
    format/rate entry before it can be played back.

    Returns a list of dicts, sorted by relative path, each with keys:
    ``path``, ``name``, ``rel_path``, ``format``, ``size_bytes``, ``fs``,
    ``fc``, ``needs_format`` (bool).
    """
    results = []
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for root, _dirs, files in walker:
        for name in files:
            low = name.lower()
            # A .sigmf-meta is the pair's descriptor; skip its .sigmf-data twin
            # here to avoid listing the pair twice (meta carries the metadata).
            if low.endswith(".sigmf-data"):
                continue
            if not low.endswith(ALL_EXTENSIONS):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, folder)
            fmt = detect_format(full)
            entry = {"path": full, "name": name, "rel_path": rel, "format": fmt,
                     "size_bytes": os.path.getsize(full) if fmt != "sigmf"
                     else os.path.getsize(_sigmf_pair(full)[0])
                     if os.path.exists(_sigmf_pair(full)[0]) else 0,
                     "fs": None, "fc": None, "needs_format": fmt == "raw"}
            if fmt == "sigmf":
                try:
                    meta = read_sigmf_meta(_sigmf_pair(full)[1])
                    entry["fs"] = meta["sample_rate"]
                    entry["fc"] = meta["frequency"]
                    entry["dtype_key"] = meta["dtype_key"]
                except Exception as e:
                    entry["error"] = str(e)
            results.append(entry)
    results.sort(key=lambda e: e["rel_path"].lower())
    return results


__all__ = [
    "RAW_DTYPES", "DEFAULT_RAW_DTYPE", "RAW_EXTENSIONS", "SIGMF_EXTENSIONS",
    "WAV_EXTENSIONS", "ALL_EXTENSIONS", "MAX_SAMPLES_DEFAULT",
    "detect_format", "read_sigmf_meta", "load_sigmf", "load_raw_iq",
    "load_wav_iq", "load_capture", "scan_folder",
]
