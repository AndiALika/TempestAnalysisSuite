"""
tempest_video.py
===============
Van Eck / video-emanation raster reconstruction for the TEMPEST Analysis Suite.

This is the signature TEMPEST demonstration: a raster display drives its pixels
line by line, and the unintended electromagnetic emanation of that video signal
can be captured at a distance and **reassembled into a readable image** if the
eavesdropper recovers the line and frame timing (Kuhn, UCAM-CL-TR-577, 2003; van
Eck, 1985).

The module is GUI-free and unit-tested.  It provides both directions:

* **Forward model** — :func:`raster_scan` + :func:`emanate` turn a source image
  into a 1-D "captured" waveform (displays radiate on intensity *transitions*, so
  the emanation is modelled as the derivative of pixel intensity plus noise).
* **Attack / reconstruction** — :func:`reconstruct` folds the 1-D waveform back
  into a 2-D image given guessed samples-per-line and lines-per-frame, and
  :func:`estimate_line_length` / :func:`estimate_frame_length` recover that timing
  automatically by autocorrelation (the "line-rate lock" of a van Eck receiver).

References
----------
[1] W. van Eck, "Electromagnetic radiation from video display units: an
    eavesdropping risk?", Computers & Security, 1985.
[2] M. G. Kuhn, "Compromising emanations: eavesdropping risks of computer
    displays", University of Cambridge, UCAM-CL-TR-577, 2003.
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Forward model  (image → emanated waveform)
# ─────────────────────────────────────────────────────────────────────────────
def raster_scan(image, h_blank=0, v_blank=0):
    """Serialise a 2-D image into the 1-D raster waveform a display would scan.

    ``h_blank`` samples of horizontal blanking are appended after every active
    line and ``v_blank`` blank lines after the frame, exactly as a real display's
    sync intervals extend the line/frame totals.

    Returns ``(signal, h_total, v_total)`` where ``h_total = W + h_blank`` is the
    samples-per-line and ``v_total = H + v_blank`` the lines-per-frame.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2-D (grayscale)")
    H, W = image.shape
    h_total = W + int(h_blank)
    v_total = H + int(v_blank)
    frame = np.zeros((v_total, h_total), dtype=float)
    frame[:H, :W] = image
    return frame.reshape(-1), h_total, v_total


def emanate(signal, noise=0.05, seed=0, highpass=True):
    """Model the leaked emanation of a raster waveform.

    A display radiates on intensity *transitions*, so the emanated signal is
    modelled as the first difference (high-pass) of the pixel stream, plus
    additive white Gaussian noise (seeded for reproducibility).
    """
    sig = np.asarray(signal, dtype=float)
    em = np.diff(sig, prepend=sig[:1]) if highpass else sig.copy()
    if noise:
        em = em + float(noise) * np.random.default_rng(seed).standard_normal(em.size)
    return em


# ─────────────────────────────────────────────────────────────────────────────
#  Reconstruction  (waveform → image)
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct(signal, h_total, v_total, h_offset=0, v_offset=0, n_frames=1):
    """Fold the 1-D captured waveform into a ``(v_total, h_total)`` image.

    ``h_offset`` / ``v_offset`` emulate the horizontal- and vertical-hold
    adjustments of a van Eck receiver: getting ``h_total`` wrong shears the image
    diagonally, and the offsets slide the frame into registration.

    ``n_frames`` averages that many consecutive frames.  Because the emanation
    repeats every frame while the noise does not, frame-synchronous averaging is
    a real attacker technique that improves SNR by ~√n and can pull an otherwise
    buried image out of the noise.
    """
    sig = np.asarray(signal, dtype=float)
    h_total, v_total = int(h_total), int(v_total)
    n_frames = max(1, int(n_frames))
    if h_total < 1 or v_total < 1:
        raise ValueError("h_total and v_total must be ≥ 1")
    shift = int(h_offset) + int(v_offset) * h_total
    sig = np.roll(sig, -shift)
    frame_len = h_total * v_total
    need = frame_len * n_frames
    if sig.size < need:
        sig = np.concatenate([sig, np.zeros(need - sig.size)])
    stack = sig[:need].reshape(n_frames, v_total, h_total)
    return stack.mean(axis=0)


def _autocorr_peak(x, min_lag, max_lag):
    """Lag of the strongest autocorrelation peak in ``[min_lag, max_lag]``."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 4:
        return int(min_lag)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(F * np.conj(F))[:n]
    lo = max(1, int(min_lag))
    hi = min(int(max_lag), n - 1)
    if hi <= lo:
        return lo
    return lo + int(np.argmax(ac[lo:hi]))


def estimate_line_length(signal, min_len=16, max_len=None):
    """Recover samples-per-line (``h_total``) by autocorrelation.

    Vertically-correlated content (text rows, screen furniture) makes the
    emanation autocorrelation peak at the line period — this is what a real
    attacker sweeps to 'lock' the horizontal hold.
    """
    n = np.asarray(signal).size
    return _autocorr_peak(signal, min_len, max_len if max_len else n // 4)


def estimate_frame_length(signal, h_total, min_lines=8, max_lines=None):
    """Recover lines-per-frame (``v_total``) from the per-line activity profile.

    Needs the capture to span more than one frame (as a real one does).
    """
    x = np.asarray(signal, dtype=float)
    h_total = int(h_total)
    nb = x.size // h_total
    if nb < 4:
        return int(min_lines)
    prof = x[:nb * h_total].reshape(nb, h_total).std(axis=1)
    return _autocorr_peak(prof, min_lines, max_lines if max_lines else nb // 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Source images (for the demo)
# ─────────────────────────────────────────────────────────────────────────────
def demo_image(kind="text", width=160, height=120, text="TOP SECRET"):
    """Build a grayscale (0..1) source image for the demonstration."""
    if kind == "text":
        return text_image(text, width, height)
    yy, xx = np.mgrid[0:height, 0:width]
    if kind == "bars":
        return ((xx // 10) % 2).astype(float)
    if kind == "checker":
        return (((xx // 16) + (yy // 16)) % 2).astype(float)
    if kind == "gradient":
        return xx / max(1, width - 1)
    return np.zeros((height, width), dtype=float)


def text_image(text, width=160, height=120):
    """Render ``text`` to a grayscale (0..1) array using PIL."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(img)
    lines = str(text).split("\n") or [""]
    size = max(8, int(height * 0.30 / max(1, len(lines))))
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            font = ImageFont.load_default()
    y = height * 0.12
    for line in lines:
        draw.text((width * 0.06, y), line, fill=255, font=font)
        y += height * 0.9 / max(1, len(lines))
    return np.asarray(img, dtype=float) / 255.0


def load_image(path, width=160, height=120):
    """Load any image file, convert to grayscale and resize to (height, width)."""
    from PIL import Image
    img = Image.open(path).convert("L").resize((int(width), int(height)))
    return np.asarray(img, dtype=float) / 255.0


__all__ = [
    "raster_scan", "emanate", "reconstruct",
    "estimate_line_length", "estimate_frame_length",
    "demo_image", "text_image", "load_image",
]
