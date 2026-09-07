"""
tempest_sdr.py
=============
Software-defined-radio capture abstraction for the TEMPEST Analysis Suite.

Design goal: the live-capture UI must run **with no hardware** (for development,
demos and the thesis defence), yet swap in a real receiver by changing one
string.  So there is:

* :class:`SDRSource`   — the abstract interface every backend implements;
* :class:`SimulatedSDR`— a synthetic complex-IQ source (tones + noise), optionally
  seeded to a device's emanation peaks;
* :func:`open_sdr`     — a factory returning an *opened* source for a named
  backend.  ``"simulated"`` works everywhere; ``"rtlsdr"`` is wired to pyrtlsdr;
  ``"soapy"`` / ``"uhd"`` are stubs that raise an informative install message.

Adding real hardware later is a localised change — implement ``read()`` in a new
``SDRSource`` subclass and register it in :func:`open_sdr`.
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Abstract interface
# ─────────────────────────────────────────────────────────────────────────────
class SDRSource:
    """Common interface for all SDR backends.  Subclasses must implement
    :meth:`read`; most also override :meth:`open` / :meth:`close`."""

    def __init__(self, sample_rate=2.048e6, center_freq=100e6, gain=20.0):
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.gain = float(gain)
        self._open = False

    def open(self):
        self._open = True
        return self

    def read(self, n):                     # pragma: no cover - abstract
        raise NotImplementedError

    def close(self):
        self._open = False

    @property
    def is_open(self):
        return self._open

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Simulated backend (no hardware required)
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedSDR(SDRSource):
    """Synthetic complex-baseband source: a handful of tones (offsets from the
    tuned centre frequency) in complex Gaussian noise.  Streaming is continuous
    — successive :meth:`read` calls advance the time base so a waterfall scrolls
    realistically.

    Pass ``device="CRT Monitor"`` to place that device's emanation harmonics that
    fall inside the tuned band, linking the live view to the emanation model.
    """

    def __init__(self, sample_rate=2.048e6, center_freq=100e6, gain=20.0,
                 tones=None, noise=0.05, seed=0, device=None):
        super().__init__(sample_rate, center_freq, gain)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)
        self._t0 = 0
        if device is not None:
            self.tones = self._peaks_in_band(device)
        elif tones is not None:
            self.tones = list(tones)
        else:
            self.tones = [0.15e6, -0.35e6]

    def _peaks_in_band(self, device):
        import tempest_physics as tp
        half = self.sample_rate / 2.0
        offs = []
        for p in tp.DEVICES[device]["peaks"]:
            for h in range(1, tp.DEVICES[device]["harmonics"] + 1):
                off = p * h - self.center_freq
                if -half < off < half:
                    offs.append(off)
        return offs or [0.1e6]

    def read(self, n):
        n = int(n)
        t = (self._t0 + np.arange(n)) / self.sample_rate
        self._t0 += n
        # gain scales the signal amplitude (rough, dimensionless)
        g = 10.0 ** ((self.gain - 20.0) / 40.0)
        iq = (self.noise / np.sqrt(2.0)) * (self._rng.standard_normal(n)
                                            + 1j * self._rng.standard_normal(n))
        for i, off in enumerate(self.tones):
            iq += g * (0.6 / (i + 1)) * np.exp(2j * np.pi * off * t)
        return iq.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
#  File-playback source ("upload a capture" — files from other SDR devices)
# ─────────────────────────────────────────────────────────────────────────────
class FileSDRSource(SDRSource):
    """Replays a pre-loaded IQ array through the exact same interface as a live
    backend, so an imported recording (RTL-SDR/HackRF/USRP/... capture, or a
    SigMF file) flows through the same worker-thread/spectrum/waterfall
    pipeline as live hardware — no other code needs to know the difference.

    Pass the ``iq`` array (from :func:`tempest_capture_io.load_capture`) plus
    its ``sample_rate`` and ``center_freq``.  When the file is exhausted it
    either loops back to the start (``loop=True``, default — useful for a
    short recording) or pads with silence (``loop=False``).
    """

    def __init__(self, iq, sample_rate, center_freq=0.0, loop=True):
        super().__init__(sample_rate, center_freq, gain=0.0)
        self._iq = np.asarray(iq, dtype=np.complex64)
        self._pos = 0
        self.loop = bool(loop)

    def read(self, n):
        n = int(n)
        out = np.zeros(n, dtype=np.complex64)
        if self._iq.size == 0:
            return out
        filled = 0
        while filled < n:
            remaining = self._iq.size - self._pos
            if remaining <= 0:
                if self.loop:
                    self._pos = 0
                    continue
                break                                   # leave the rest zero-padded
            take = min(remaining, n - filled)
            out[filled:filled + take] = self._iq[self._pos:self._pos + take]
            self._pos += take
            filled += take
        return out

    @property
    def progress(self):
        """Fraction of the recording played so far, in [0, 1]."""
        return self._pos / self._iq.size if self._iq.size else 0.0

    @property
    def duration_s(self):
        return self._iq.size / self.sample_rate if self.sample_rate else 0.0

    @property
    def at_end(self):
        return (not self.loop) and self._pos >= self._iq.size


def open_file_source(path, dtype_key=None, fs_hint=None, fc_hint=None, loop=True,
                     max_samples=None):
    """Load a capture file (SigMF / WAV / raw IQ — see
    :mod:`tempest_capture_io`) and return an **opened** :class:`FileSDRSource`.

    For headerless raw files, ``dtype_key`` and ``fs_hint`` are required —
    the GUI collects these from the user when the format can't be
    auto-detected. ``fc_hint`` overrides/fills the centre frequency when the
    file doesn't carry one (SigMF captures usually do; raw files never do).
    """
    import tempest_capture_io as cio
    kwargs = dict(dtype_key=dtype_key, fs_hint=fs_hint, fc_hint=fc_hint)
    if max_samples is not None:
        kwargs["max_samples"] = max_samples
    info = cio.load_capture(path, **kwargs)
    src = FileSDRSource(info["iq"], info["fs"],
                        center_freq=info["fc"] or 0.0, loop=loop).open()
    src.info = info          # stash the full metadata dict for the GUI to show
    return src


# ─────────────────────────────────────────────────────────────────────────────
#  Backend discovery & factory
# ─────────────────────────────────────────────────────────────────────────────
def available_backends():
    """Return ``{name: bool}`` — which backends can actually be opened here."""
    out = {"simulated": True}
    for name, module in [("rtlsdr", "rtlsdr"), ("soapy", "SoapySDR"), ("uhd", "uhd")]:
        try:
            __import__(module)
            out[name] = True
        except Exception:
            out[name] = False
    return out


def open_sdr(backend="simulated", sample_rate=2.048e6, center_freq=100e6,
             gain=20.0, **kw):
    """Return an **opened** :class:`SDRSource` for ``backend``.

    Extra keyword args:
      * simulated → ``tones`` / ``noise`` / ``seed`` / ``device``
      * soapy     → ``driver`` (e.g. "hackrf", "lime", "airspy"; "" = auto)
      * uhd       → ``args``   (UHD device address string; "" = auto)
    """
    backend = backend.lower()
    if backend == "simulated":
        return SimulatedSDR(sample_rate, center_freq, gain, **kw).open()
    # Real hardware backends do not use the emanation-device tie.
    kw.pop("device", None)
    if backend == "rtlsdr":
        return _open_rtlsdr(sample_rate, center_freq, gain)
    if backend in ("soapy", "soapysdr"):
        return _open_soapy(sample_rate, center_freq, gain,
                           driver=kw.get("driver", ""))
    if backend == "uhd":
        return _open_uhd(sample_rate, center_freq, gain, args=kw.get("args", ""))
    raise ValueError(f"unknown SDR backend: {backend!r}")


def _open_soapy(sample_rate, center_freq, gain, driver=""):
    """Generic SoapySDR backend — covers HackRF, LimeSDR, Airspy, PlutoSDR,
    RTL-SDR-via-Soapy, etc. Kept isolated so a missing library gives a clear
    message rather than a stack trace."""
    try:
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    except Exception as e:                 # pragma: no cover - needs hardware/lib
        raise RuntimeError(
            "SoapySDR is not installed. Install SoapySDR + its Python bindings "
            "and the device support module (e.g. SoapyHackRF / SoapyLMS7), then "
            "select this backend. Optionally set driver=... (hackrf, lime, …).") from e

    class _SoapySource(SDRSource):         # pragma: no cover - needs hardware
        def open(self):
            self._dev = SoapySDR.Device(dict(driver=driver) if driver else {})
            self._dev.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self._dev.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
            try:
                self._dev.setGain(SOAPY_SDR_RX, 0, self.gain)
            except Exception:
                pass                        # some devices only support AGC
            self._rx = self._dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            self._dev.activateStream(self._rx)
            self._open = True
            return self

        def read(self, n):
            n = int(n)
            out = np.empty(n, np.complex64)
            filled = 0
            while filled < n:
                buff = np.empty(n - filled, np.complex64)
                sr = self._dev.readStream(self._rx, [buff], len(buff),
                                          timeoutUs=1_000_000)
                if sr.ret > 0:
                    out[filled:filled + sr.ret] = buff[:sr.ret]
                    filled += sr.ret
                else:
                    break                   # timeout / overflow / error
            return out[:filled]

        def close(self):
            try:
                self._dev.deactivateStream(self._rx)
                self._dev.closeStream(self._rx)
            finally:
                self._open = False

    return _SoapySource(sample_rate, center_freq, gain).open()


def _open_uhd(sample_rate, center_freq, gain, args=""):
    """Ettus USRP backend via the UHD Python API."""
    try:
        import uhd
    except Exception as e:                  # pragma: no cover - needs hardware/lib
        raise RuntimeError(
            "The UHD Python API is not installed. Install UHD with Python "
            "support, then select this backend (optionally set args=... for the "
            "device address).") from e

    class _UhdSource(SDRSource):            # pragma: no cover - needs hardware
        def open(self):
            self._usrp = uhd.usrp.MultiUSRP(args)
            self._usrp.set_rx_rate(self.sample_rate)
            self._usrp.set_rx_freq(uhd.types.TuneRequest(self.center_freq))
            self._usrp.set_rx_gain(self.gain)
            self._open = True
            return self

        def read(self, n):
            samps = self._usrp.recv_num_samps(
                int(n), self.center_freq, self.sample_rate, [0], self.gain)
            return np.asarray(samps).reshape(-1).astype(np.complex64)

        def close(self):
            self._open = False

    return _UhdSource(sample_rate, center_freq, gain).open()


def _open_rtlsdr(sample_rate, center_freq, gain):
    """Real RTL-SDR backend via pyrtlsdr (kept isolated so import failure gives a
    clear, actionable message instead of a stack trace)."""
    try:
        from rtlsdr import RtlSdr
    except Exception as e:                 # pragma: no cover - needs hardware/lib
        raise RuntimeError(
            "pyrtlsdr is not installed. Run 'pip install pyrtlsdr' and install "
            "the RTL-SDR USB driver (Zadig → WinUSB on Windows).") from e

    class _RtlSdrSource(SDRSource):        # pragma: no cover - needs hardware
        def open(self):
            self._sdr = RtlSdr()
            self._sdr.sample_rate = self.sample_rate
            self._sdr.center_freq = self.center_freq
            self._sdr.gain = self.gain
            self._open = True
            return self

        def read(self, n):
            return self._sdr.read_samples(int(n)).astype(np.complex64)

        def close(self):
            try:
                self._sdr.close()
            finally:
                self._open = False

    return _RtlSdrSource(sample_rate, center_freq, gain).open()


# ─────────────────────────────────────────────────────────────────────────────
#  Spectrum helper (absolute RF axis)
# ─────────────────────────────────────────────────────────────────────────────
def capture_spectrum(iq, sample_rate, center_freq, nfft=4096, window="Hann"):
    """Averaged spectrum of an IQ block on an **absolute RF** frequency axis
    (baseband bins shifted up by ``center_freq``).

    Returns ``(freqs_Hz, mag_db, n_avg)``.
    """
    import tempest_dsp as dsp
    f, mag_db, n = dsp.averaged_spectrum(iq, sample_rate, nfft=nfft, window=window)
    return f + float(center_freq), mag_db, n


__all__ = [
    "SDRSource", "SimulatedSDR", "FileSDRSource",
    "available_backends", "open_sdr", "open_file_source", "capture_spectrum",
]
