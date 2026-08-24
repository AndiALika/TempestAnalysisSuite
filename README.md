# TEMPEST Analysis Suite

**MSc Telecommunications Engineering — thesis tool by Andi Lika**

A desktop application for analysing **compromising electromagnetic emanations
(TEMPEST)** and the shielding / zoning countermeasures against them. It combines
an emanation model, signal analysis, a rigorous shielding-effectiveness engine,
facility (room) assessment, live SDR capture, a van Eck video-reconstruction
demonstration, and standards-based compliance checking.

The codebase is deliberately split into **GUI-free, unit-tested "core" modules**
(the physics and signal processing) and a **customtkinter GUI** (`tempest_suite22.py`).
Every number the tool produces comes from a core function that is validated
against textbook reference values, so the results are reproducible and citable.

---

## Quick start

```bash
pip install -r requirements.txt
python tempest_suite22.py
```

Run the test suite (standard-library `unittest`, no pytest needed):

```bash
python -m unittest test_tempest_physics test_tempest_dsp test_tempest_report \
    test_tempest_measure test_tempest_sdr test_tempest_video \
    test_tempest_compliance test_tempest_room test_tempest_signal
```

> On Windows, set `PYTHONIOENCODING=utf-8` before running any script that prints
> ✅/❌/µ characters (the GUI itself is unaffected).

---

## Application modules (GUI)

| # | Module | What it does |
|---|--------|--------------|
| 1 | **EM Emanation Simulator** | Reproducible emanation spectra (dBµV/m) per device; overlay a **measured trace**, calibrate a device from it, and check against a **compliance mask** (CISPR 32 / FCC 15 / SDIP-27). |
| 2 | **Signal Capture & Analysis** | Load WAV/IQ, Welch-averaged window-corrected spectrum + spectrogram; **peak detection, harmonic-family grouping and emanation-source classification**. |
| 3 | **Shielding Effectiveness Calculator** | SE = A+R+B with near-field E/H, **multi-layer laminates** (ABCD cascade), **frequency-dependent µ**, aperture/seam leakage, **measured-SE overlay + target pass/fail**, mass. |
| 4 | **Room TEMPEST Assessment** | Per-wall shielding + window/door apertures, **inspectable-space / eavesdropper** field analysis, and a **design solver** (required wall SE / standoff / material). |
| 5 | **Emanation Coverage & Interception Map** | Field-strength **coverage heatmap** (dBµV/m) with the **interception boundary** contour, **two-ray multipath**, near/far-field boundaries, a placed **eavesdropper** verdict, and a radial field profile. Wavefront animation kept as a secondary view. |
| 6 | **Shielding Design Optimizer** | Finds the lightest/cheapest shield (single or **2-layer laminate**) that brings a **source under a limit** — a *derived*, frequency-dependent requirement `SE_req(f)=source−limit`; real SE margin, mass (kg) & cost (USD), with a mass–cost Pareto. |
| 7 | **Live SDR Capture** | Real-time spectrum + waterfall. Backends: simulated (no hardware), RTL-SDR, SoapySDR, UHD. Reference-level calibration to dBµV/m. |
| 8 | **Van Eck Reconstruction** | Reconstructs an eavesdropped screen image from a display's video emanation; autocorrelation line/frame-rate lock and frame averaging. |

---

## Architecture

```
tempest_suite22.py        GUI (customtkinter) — 8 modules, presentation only
│
├─ tempest_physics.py     shielding (A+R+B, near-field, multilayer, apertures),
│                         emanation model, zoning, propagation, materials
├─ tempest_dsp.py         WAV/IQ loading, Welch spectra, spectrogram
├─ tempest_signal.py      peak detection, harmonic families, device classification
├─ tempest_measure.py     measured-trace import + antenna-factor calibration chain
├─ tempest_compliance.py  emission-limit masks + pass/fail
├─ tempest_room.py        per-wall shielding, inspectable space, design solver
├─ tempest_sdr.py         SDR capture abstraction + simulated/real backends
├─ tempest_video.py       van Eck raster reconstruction
└─ tempest_report.py      PNG / CSV / PDF / JSON-session export

test_tempest_*.py         119 unit tests (one suite per core module)
tempest_samples/          ready-made .wav / .iq / .csv test inputs (+ its README)
```

Each core module is pure NumPy/SciPy with no GUI dependency, so the physics can
be tested, reused and cited independently of the interface.

---

## Scientific basis & references

* **Shielding** — Schelkunoff impedance method and Ott's A+R+B decomposition;
  multilayer via ABCD transmission-matrix cascade. Validated: copper skin depth
  66 µm, 1 mm-copper absorption 131.6 dB, plane-wave reflection 108 dB @ 1 MHz.
  *H. W. Ott, EMC Engineering, Wiley 2009; S. A. Schelkunoff, EM Waves, 1943;
  IEEE Std 299; MIL-STD-285.*
* **Apertures/seams** — slot radiation `20log10(λ/2L)` + waveguide-below-cutoff.
* **Zoning / inspectable space** — *NATO SDIP-27* (Level A/B/C curves in the tool
  are **illustrative placeholders**; the real limits are controlled).
* **Compliance masks** — *CISPR 32:2015* and *FCC 47 CFR Part 15* (public limits).
* **Emanation reconstruction** — *W. van Eck, 1985; M. G. Kuhn, UCAM-CL-TR-577, 2003.*
* **Calibration** — CISPR 16 radiated-emission chain: `E = V + AF + cable − gain`.
* **Propagation** — Friis link budget; Fraunhofer far-field 2D²/λ.

### Honesty note
Device emission baselines and the detectability floor are **representative,
illustrative values**, not calibrated measurements — they are clearly flagged in
the code and in exported reports. Feed real spectrum-analyzer / IEEE-299 traces
via the measured-overlay features to obtain data-driven results.

---

## Sample test data

`tempest_samples/` contains ready-made inputs (5 WAV, 3 IQ, 3 CSV traces) with
their own README describing which module and settings each exercises.

---

## Building a standalone executable

PyInstaller produces a self-contained `.exe` (no Python install needed on the
target machine — handy for a defence laptop):

```powershell
pip install pyinstaller
./build_exe.ps1        # or: pyinstaller tempest_suite.spec
```

The result appears in `dist/TempestSuite/`.
