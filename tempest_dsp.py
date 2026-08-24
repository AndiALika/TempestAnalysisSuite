"""
tempest_dsp.py
==============
Signal ingestion and spectral analysis for the TEMPEST Analysis Suite.

Like ``tempest_physics``, this module is GUI-free and unit-tested.  It fixes the
signal-path problems in the original tool:

* **IQ files keep their Q channel** (the old code did ``np.real(iq)`` and threw
  the quadrature component away, collapsing a complex baseband capture to a real
  signal and discarding negative-frequency information).
* **Robust WAV loading** — 8/16/32-bit PCM via the stdlib, plus 24-bit and float
  formats when the optional ``soundfile`` package is present; multi-channel files
  are handled instead of being mis-read as interleaved mono.
* **Averaged, window-corrected spectra** — a Welch-style average over overlapping
  windowed segments with proper *coherent-gain* correction, so a unit-amplitude
  tone reads ≈ 0 dB regardless of the window (the old single-snapshot FFT was off
  by a window-dependent factor and never removed DC).
"""

from __future__ import annotations

import wave
import numpy as np
from scipy.signal import get_window, spectrogram as _spectrogram

try:
    import soundfile as _sf
    HAVE_SOUNDFILE = True
except Exception:                      # pragma: no cover - optional dependency
    HAVE_SOUNDFILE = False

# GUI-friendly window name  →  scipy.signal.get_window name.
WINDOWS = {
    "Rectangular": "boxcar",
    "Hann":        "hann",
    "Hamming":     "hamming",
    "Blackman":    "blackman",
    "Flat Top":    "flattop",
}


def _scipy_window(name, n):
    return get_window(WINDOWS.get(name, str(name).lower()), n, fftbins=True)


# ─────────────────────────────────────────────────────────────────────────────
#  File loading
# ─────────────────────────────────────────────────────────────────────────────
def load_wav(path, mono=True):
    """Load a WAV file → ``(data, fs)`` with ``data`` float64 normalised to ±1.

    Uses ``soundfile`` when available (24-bit / float support); otherwise falls
    back to the stdlib ``wave`` module for 8/16/32-bit PCM.  Multi-channel audio
    is averaged to mono (``mono=True``) or reduced to channel 0.
    """
    if HAVE_SOUNDFILE:
        data, fs = _sf.read(path, always_2d=True, dtype="float64")
        data = data.mean(axis=1) if mono else data[:, 0]
        return data.astype(np.float64), int(fs)

    with wave.open(path, "rb") as wf:
        fs, n, ch, sw = (wf.getframerate(), wf.getnframes(),
                         wf.getnchannels(), wf.getsampwidth())
        raw = wf.readframes(n)

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
    if dtype is None:
        raise ValueError(
            f"Unsupported WAV sample width ({sw * 8}-bit). "
            "Install the 'soundfile' package for 24-bit / float WAV support.")

    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if ch > 1:
        data = data.reshape(-1, ch)
        data = data.mean(axis=1) if mono else data[:, 0]

    # Normalise to ±1 (8-bit PCM is unsigned, centred on 128).
    if dtype == np.uint8:
        data = (data - 128.0) / 128.0
    else:
        data = data / float(np.iinfo(dtype).max)
    return data, int(fs)


def load_iq(path, fs=2_000_000, dtype=np.float32):
    """Load an interleaved I/Q file → ``(complex128 array, fs)``.

    The quadrature (Q) channel is **preserved**, giving a proper complex baseband
    signal whose spectrum is two-sided about 0 Hz.
    """
    raw = np.fromfile(path, dtype=dtype).astype(np.float64)
    if raw.size % 2:                    # drop a trailing unpaired sample
        raw = raw[:-1]
    iq = raw[0::2] + 1j * raw[1::2]
    return iq, int(fs)


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral analysis
# ─────────────────────────────────────────────────────────────────────────────
def averaged_spectrum(data, fs, nfft=4096, window="Hann", overlap=0.5):
    """Welch-style **amplitude** spectrum, averaged over overlapping windowed
    segments and corrected for the window's coherent gain.

    Returns ``(freqs, mag_db, n_avg)``.  Real input yields a one-sided spectrum;
    complex (IQ) input yields a two-sided spectrum centred at 0 Hz.  A unit-
    amplitude sinusoid reads ≈ 0 dB irrespective of the chosen window.
    """
    x = np.asarray(data)
    complex_in = np.iscomplexobj(x)
    n = min(int(nfft), len(x))
    if n < 2:
        raise ValueError("signal too short to analyse")
    step = max(1, int(n * (1.0 - overlap)))
    w = _scipy_window(window, n)
    cg = np.sum(w)                                   # coherent gain

    mags = []
    starts = range(0, len(x) - n + 1, step) or [0]
    for s in starts:
        seg = x[s:s + n]
        seg = seg - np.mean(seg)                     # remove DC
        spec = (np.fft.fftshift(np.fft.fft(seg * w)) if complex_in
                else np.fft.rfft(seg * w))
        mags.append(np.abs(spec))
    mag = np.mean(mags, axis=0)

    if complex_in:
        freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
        amp = mag / cg
    else:
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        amp = mag / cg * 2.0                         # one-sided → ×2 …
        amp[0] /= 2.0                                # … except DC bin
    return freqs, 20.0 * np.log10(amp + 1e-12), len(mags)


def make_spectrogram(data, fs, nfft=512, window="Hann", overlap=0.5):
    """Thin wrapper around ``scipy.signal.spectrogram`` returning power in dB.

    Returns ``(f, t, Sxx_db)``.  Complex input produces a two-sided (fftshifted)
    spectrogram.
    """
    x = np.asarray(data)
    n = min(int(nfft), len(x))
    noverlap = min(int(n * overlap), n - 1)
    f, t, Sxx = _spectrogram(
        x, fs=fs, window=_scipy_window(window, n), nperseg=n,
        noverlap=noverlap, nfft=n, detrend="constant",
        return_onesided=not np.iscomplexobj(x), scaling="density",
        mode="magnitude")
    if np.iscomplexobj(x):
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)
    return f, t, 20.0 * np.log10(Sxx + 1e-12)


def synth_tempest_signal(fs=44100, duration=1.0, seed=0):
    """Reproducible synthetic composite signal for the 'generate test' button."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)
    sig = (0.6 * np.sin(2 * np.pi * 1000 * t)
           + 0.3 * np.sin(2 * np.pi * 3000 * t)
           + 0.15 * np.sin(2 * np.pi * 7000 * t)
           + 0.05 * np.sin(2 * np.pi * 15000 * t)
           + 0.02 * rng.standard_normal(t.size))
    return sig, fs


__all__ = [
    "HAVE_SOUNDFILE", "WINDOWS",
    "load_wav", "load_iq",
    "averaged_spectrum", "make_spectrogram", "synth_tempest_signal",
]
