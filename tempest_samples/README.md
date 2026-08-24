# TEMPEST Analysis Suite — sample test files

Generated sample inputs for trying out the **Signal Capture & Analysis** and
**EM Emanation Simulator** modules. All files were verified by loading them back
through the app's own loaders (`tempest_dsp`, `tempest_measure`).

## Signal Capture & Analysis  →  "📂 Load WAV File"

### WAV audio files
| File | fs | What it shows |
|------|----|----|
| `sig_tempest_composite.wav` | 48 kHz | The classic multi-tone TEMPEST composite (1/3/7/15 kHz) — clean FFT peaks. |
| `sig_crt_video_harmonics.wav` | 192 kHz | CRT h-sync fundamental 15.625 kHz + harmonics — a harmonic **comb** in the FFT. |
| `sig_keyboard_bursts.wav` | 48 kHz | Repeated "keystroke" bursts — best seen in the **spectrogram** (vertical stripes). |
| `sig_chirp_sweep.wav` | 48 kHz | 0.5→18 kHz sweep — a clean diagonal in the **spectrogram**. |
| `sig_stereo_1k_5k.wav` | 48 kHz | **Stereo** (L=1 kHz, R=5 kHz) — tests the stereo→mono down-mix (both tones appear). |

Try changing the **Window** (Rectangular → Flat Top) and **FFT Size** while a file
is loaded to see the averaged-spectrum resolution/leakage trade-off.

### IQ files (interleaved float32, fs = 2.0 MHz)
Load these the same way (choose *IQ files* / `*.iq` in the dialog).

| File | What it shows |
|------|----|
| `iq_single_carrier_+200kHz.iq` | One complex carrier at **+200 kHz** — note it appears only on the positive side (Q channel preserved). |
| `iq_two_carriers.iq` | Carriers at **+300 kHz and −450 kHz** — a **two-sided** spectrum about 0. |
| `iq_am_carrier_100kHz.iq` | AM carrier at +100 kHz with 5 kHz tone — shows **sidebands**. |

The time-domain panel plots **I and Q** separately for IQ files.

## EM Emanation Simulator  →  "📈 Load CSV" (measured trace)

Set the **freq-unit** dropdown and **dBm-in** switch to match the file:

| File | Unit | dBm switch | Notes |
|------|------|-----------|-------|
| `trace_hdmi_dBuV_MHz.csv` | MHz | off | HDMI-style peaks at 165/330/495/660 MHz. Pair with device **HDMI Cable**. |
| `trace_cpu_dBm_MHz.csv` | MHz | **on** | Levels in **dBm** (auto-converted to dBµV). Pair with **CPU (3 GHz)**. |
| `trace_crt_dBuV_kHz.csv` | **kHz** | off | CRT sync harmonics 15.6/31.25/62.5/125 kHz. Pair with **CRT Monitor**. |

Suggested workflow:
1. Pick the matching **Device**, set unit + dBm switch, click **📈 Load CSV** — the
   measured trace overlays the model in orange.
2. Enter your antenna factor / cable loss / gain and **@dist**, then **🎯 Calibrate
   device** to replace that device's emission baseline with the measurement
   (this also updates the Room Assessment zones).
3. Turn on a **Compliance mask** (e.g. *CISPR 32 Class B*) to get a PASS/FAIL verdict.

*(All traces have a realistic noise floor plus harmonic peaks; the CSVs have a
header row and are comma-delimited, but the loader also sniffs headerless /
semicolon / tab files.)*
