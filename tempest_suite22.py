"""
TEMPEST Analysis Suite
MSc Telecommunications Engineering - Master Thesis Tool
Author: Andi Lika
Description: Comprehensive TEMPEST device and room emanation analysis application.
"""

import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from tkinter import filedialog, messagebox
import os, json, tempfile

import tempest_physics as tp   # validated pure-physics core (see tempest_physics.py)
import tempest_dsp as dsp      # signal ingestion & spectral analysis core (see tempest_dsp.py)
import tempest_report as rep   # PNG / CSV / PDF / session export (see tempest_report.py)
import tempest_measure as tm   # measured-trace import & field-strength calibration
import tempest_sdr as sdrlib   # SDR capture abstraction + simulated backend
import tempest_capture_io as cio  # import recorded captures (SigMF / raw IQ / WAV)
import tempest_video as tv     # van Eck video-emanation reconstruction
import tempest_compliance as tc  # emission-limit masks & pass/fail evaluation
import tempest_room as tr       # per-wall shielding, inspectable-space & design solver
import tempest_signal as tsig   # peak detection, harmonic families & device classification
from tkinter import filedialog as _fd
import threading, queue, time

# ─────────────────────────────────────────────
#  App-wide theme
# ─────────────────────────────────────────────
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")     # base theme; colours overridden below

# Retint the widgets whose accent isn't set per-instance (sliders, switches,
# checkboxes, dropdowns) so the crimson accent is applied consistently
# everywhere.  Values are hardcoded here (rather than referencing the palette
# below) because this runs before that block is defined.
def _retint():
    _A, _H = "#c62f55", "#ad2647"
    _TXT, _TXT2 = "#141417", "#6b6b73"     # TEXT_PRI / TEXT_SEC
    _WHITE, _TINT = "#ffffff", "#f4dbe2"   # BG_PANEL / soft crimson hover tint
    try:
        t = ctk.ThemeManager.theme
        for k in ("progress_color", "button_color"):
            t["CTkSlider"][k] = _A
        t["CTkSlider"]["button_hover_color"] = _H
        t["CTkSwitch"]["progress_color"] = _A
        for k in ("fg_color", "hover_color"):
            t["CTkCheckBox"][k] = _A if k == "fg_color" else _H
        # CTkOptionMenu's default text_color (#DCE4EE, pale) is meant for a
        # saturated-blue fill; on our light grey fg_color it was near-invisible
        # ("grey on grey", reads as unselected). Fix the text/dropdown colours
        # app-wide — fg_color/button_color are already set per-instance.
        om = t["CTkOptionMenu"]
        om["text_color"] = _TXT
        om["text_color_disabled"] = _TXT2
        om["button_hover_color"] = _H
        # The dropdown *popup* is a separate internal widget with its own theme
        # section ("DropdownMenu", not "CTkOptionMenu") — style that too.
        dm = t["DropdownMenu"]
        dm["fg_color"] = _WHITE
        dm["text_color"] = _TXT
        dm["hover_color"] = _TINT
    except Exception:
        pass
_retint()

# ── Light minimalist palette:  black · raspberry-crimson · greys · off-white ──
BG_DARK   = "#f0f0f1"   # app ground (module frames)
BG_PANEL  = "#ffffff"   # cards / panels / plot background
BG_CARD   = "#f4f4f6"   # inner tiles / inputs
ACCENT    = "#c62f55"   # crimson — primary, active, brand, alert
ACCENT_HV = "#ad2647"   # crimson hover
ACCENT_SEL    = "#8a1d38"   # darker crimson — persistent "selected" state (nav, chosen option)
ACCENT_SEL_HV = "#6e1730"   # darker still — hover while selected
ACCENT2   = "#2f8f5b"   # muted green — success / positive
WARN      = "#b57d1e"   # amber — warnings / thresholds
TEXT_PRI  = "#141417"   # near-black text
TEXT_SEC  = "#6b6b73"   # grey text
BORDER    = "#e4e4e7"   # hairline border
DANGER    = "#c62f55"   # crimson — fail / danger

# Sidebar (black)
BG_SIDEBAR    = "#0c0c0e"
SIDEBAR_TEXT  = "#9a9aa2"
SIDEBAR_HOVER = "#17171b"
SIDEBAR_DIV   = "#1e1e23"

# ── Matplotlib: light, minimalist, cohesive across every figure ──
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": BG_PANEL, "axes.facecolor": BG_PANEL, "savefig.facecolor": BG_PANEL,
    "axes.edgecolor": BORDER, "axes.linewidth": 0.8,
    "axes.labelcolor": TEXT_SEC, "axes.titlecolor": TEXT_PRI,
    "xtick.color": TEXT_SEC, "ytick.color": TEXT_SEC, "text.color": TEXT_PRI,
    "grid.color": "#ededf0", "grid.linewidth": 0.8, "axes.grid": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "Segoe UI", "DejaVu Sans"],
    "font.size": 9, "figure.dpi": 110,
    "axes.prop_cycle": plt.cycler(color=[ACCENT, ACCENT2, WARN, "#3a6ea5", "#8a5cd1", TEXT_SEC]),
})
PLOT_PARAMS = dict(facecolor=BG_PANEL, edgecolor=BORDER)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  BUTTON
# ─────────────────────────────────────────────────────────────────────────────
class SidebarButton(ctk.CTkButton):
    def __init__(self, master, text, icon, command=None, **kw):
        super().__init__(
            master,
            text=f"  {icon}  {text}",
            anchor="w",
            height=40,
            corner_radius=8,
            fg_color="transparent",
            hover_color=SIDEBAR_HOVER,
            text_color=SIDEBAR_TEXT,
            font=ctk.CTkFont(size=13, weight="normal"),
            command=command,
            **kw
        )

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 1 – EM EMANATION SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class EmanationSimulator(ctk.CTkFrame):
    # Device library is shared with the physics core (single source of truth).
    DEVICES = tp.DEVICES

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._measured = None          # (freqs_Hz, field_dBµV/m) of a loaded trace
        self._measured_name = ""
        self._build_ui()

    def _build_ui(self):
        # Title
        title = ctk.CTkLabel(self, text="EM Emanation Simulator",
                             font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRI)
        title.pack(anchor="w", padx=24, pady=(20, 2))
        sub = ctk.CTkLabel(self, text="Visualize electromagnetic leakage from common computing devices",
                           font=ctk.CTkFont(size=12), text_color=TEXT_SEC)
        sub.pack(anchor="w", padx=24, pady=(0, 16))

        # Controls row
        ctrl = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        ctrl.pack(fill="x", padx=24, pady=(0, 16))

        ctk.CTkLabel(ctrl, text="Device:", text_color=TEXT_SEC, font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=14, pady=12)
        self.device_var = ctk.StringVar(value="CRT Monitor")
        dev_menu = ctk.CTkOptionMenu(ctrl, values=list(self.DEVICES.keys()),
                                     variable=self.device_var, width=200,
                                     fg_color=BG_CARD, button_color=ACCENT,
                                     command=lambda _: self._plot())
        dev_menu.grid(row=0, column=1, padx=8, pady=12)

        ctk.CTkLabel(ctrl, text="Distance (m):", text_color=TEXT_SEC, font=ctk.CTkFont(size=12)).grid(row=0, column=2, padx=14)
        self.dist_var = ctk.DoubleVar(value=1.0)
        dist_slider = ctk.CTkSlider(ctrl, from_=0.1, to=50, variable=self.dist_var,
                                    width=180, command=lambda _: self._plot())
        dist_slider.grid(row=0, column=3, padx=8)
        self.dist_label = ctk.CTkLabel(ctrl, text="1.0 m", text_color=ACCENT, font=ctk.CTkFont(size=12, weight="bold"))
        self.dist_label.grid(row=0, column=4, padx=4)

        ctk.CTkLabel(ctrl, text="Show harmonics:", text_color=TEXT_SEC, font=ctk.CTkFont(size=12)).grid(row=0, column=5, padx=14)
        self.harm_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(ctrl, text="", variable=self.harm_var, command=self._plot).grid(row=0, column=6, padx=8)

        ctk.CTkButton(ctrl, text="🖼 Save PNG", width=110, fg_color=BG_CARD,
                      hover_color=BORDER, command=self._save_png).grid(row=0, column=7, padx=10)

        # ── Measured-trace / calibration row ─────────────────────────────────
        meas = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10,
                            border_width=1, border_color=BORDER)
        meas.pack(fill="x", padx=24, pady=(0, 12))
        ctk.CTkLabel(meas, text="Measured trace:", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(14, 6), pady=10)
        ctk.CTkButton(meas, text="📈 Load CSV", width=100, fg_color=ACCENT,
                      hover_color=ACCENT_HV, command=self._load_measured
                      ).grid(row=0, column=1, padx=4)

        self.meas_unit = ctk.StringVar(value="MHz")
        ctk.CTkOptionMenu(meas, values=["Hz", "kHz", "MHz", "GHz"], width=72,
                          variable=self.meas_unit, fg_color=BG_CARD, button_color=ACCENT
                          ).grid(row=0, column=2, padx=4)
        self.meas_dbm = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(meas, text="dBm in", variable=self.meas_dbm,
                      font=ctk.CTkFont(size=11)).grid(row=0, column=3, padx=8)

        def _mfield(col, label, var_name, default):
            ctk.CTkLabel(meas, text=label, text_color=TEXT_SEC,
                         font=ctk.CTkFont(size=11)).grid(row=0, column=col, padx=(10, 2))
            v = ctk.StringVar(value=default)
            setattr(self, var_name, v)
            ctk.CTkEntry(meas, textvariable=v, width=54, fg_color=BG_CARD,
                         border_color=BORDER).grid(row=0, column=col + 1, padx=(0, 2))
        _mfield(4, "AF", "meas_af", "0")
        _mfield(6, "Cable", "meas_cable", "0")
        _mfield(8, "Gain", "meas_gain", "0")
        _mfield(10, "@dist m", "meas_dist", "1.0")

        ctk.CTkButton(meas, text="🎯 Calibrate device", width=140, fg_color=ACCENT2,
                      text_color="#ffffff", hover_color="#26744a",
                      command=self._calibrate_from_trace).grid(row=0, column=12, padx=(10, 4))
        ctk.CTkButton(meas, text="✖", width=32, fg_color=BG_CARD, hover_color=BORDER,
                      command=self._clear_measured).grid(row=0, column=13, padx=(0, 12))

        # ── Compliance limit mask row ────────────────────────────────────────
        comp = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10,
                            border_width=1, border_color=BORDER)
        comp.pack(fill="x", padx=24, pady=(0, 12))
        ctk.CTkLabel(comp, text="Compliance mask:", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(14, 6), pady=10)
        self.compl_var = ctk.StringVar(value="None")
        ctk.CTkOptionMenu(comp, values=["None"] + tc.standard_names(), width=250,
                          variable=self.compl_var, fg_color=BG_CARD, button_color=ACCENT,
                          command=lambda _: self._plot()).grid(row=0, column=1, padx=4)
        self.compl_verdict = ctk.CTkLabel(comp, text="", font=ctk.CTkFont(size=12, weight="bold"),
                                          text_color=TEXT_SEC)
        self.compl_verdict.grid(row=0, column=2, padx=16)

        # matplotlib canvas
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(11, 6), **PLOT_PARAMS)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=24, pady=(0, 20))

        self.dist_var.trace_add("write", lambda *_: self.dist_label.configure(text=f"{self.dist_var.get():.1f} m"))
        self._plot()

    def _generate_spectrum(self, device_key, distance):
        """Reproducible emanation spectrum in dBµV/m (delegates to the core)."""
        return tp.emanation_spectrum(device_key, distance,
                                     harmonics=self.harm_var.get())

    def _plot(self):
        dev = self.device_var.get()
        dist = self.dist_var.get()
        freqs, spec_db = self._generate_spectrum(dev, dist)   # already dBµV/m
        limit = tp.DETECTION_FLOOR_DBUV

        for ax in (self.ax1, self.ax2):
            ax.clear()
            ax.set_facecolor(BG_PANEL)
            ax.tick_params(colors=TEXT_SEC, labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

        self.ax1.plot(freqs / 1e6, spec_db, color=ACCENT, lw=0.8, alpha=0.9, label="Model")
        self.ax1.fill_between(freqs / 1e6, spec_db.min(), spec_db, alpha=0.15, color=ACCENT)
        # Overlay a real measured trace (already calibrated to field strength).
        if self._measured is not None:
            mf, mv = self._measured
            self.ax1.plot(mf / 1e6, mv, color=WARN, lw=1.0, alpha=0.95,
                          label=f"Measured: {self._measured_name}")
        self.ax1.set_xlabel("Frequency (MHz)", color=TEXT_SEC, fontsize=10)
        self.ax1.set_ylabel("Field strength (dBµV/m)", color=TEXT_SEC, fontsize=10)
        title = f"{dev}  —  Emanation Spectrum  @  {dist:.1f} m"
        if tp.DEVICES[dev].get("measured"):
            title += "  [calibrated]"
        self.ax1.set_title(title, color=TEXT_PRI, fontsize=12)
        self.ax1.axhline(limit, color=WARN, lw=0.8, linestyle="--", label="Detectability floor")
        self._overlay_compliance(freqs, spec_db, dist)
        self.ax1.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_PRI)
        self.ax1.grid(True, alpha=0.15, color=BORDER)

        # Analytic peak field vs distance (1/r free-space decay — reproducible).
        distances = np.linspace(0.5, 100, 200)
        peak_db = tp.field_at_distance_dbuv(dev, 10.0, 1.0) - \
            tp.free_space_field_decay_db(np.maximum(distances, tp.REF_DISTANCE), tp.REF_DISTANCE)
        self.ax2.plot(distances, peak_db, color=ACCENT2, lw=1.5)
        self.ax2.plot(dist, tp.field_at_distance_dbuv(dev, 10.0, dist), "o",
                      color=ACCENT, ms=8, label=f"Current: {dist:.1f} m")
        self.ax2.axhline(limit, color=WARN, lw=0.8, linestyle="--", label="Detectability floor")
        self.ax2.set_xlabel("Distance (m)", color=TEXT_SEC, fontsize=10)
        self.ax2.set_ylabel("Peak field strength (dBµV/m)", color=TEXT_SEC, fontsize=10)
        self.ax2.set_title("Field Attenuation vs Distance (1/r)", color=TEXT_PRI, fontsize=12)
        self.ax2.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_PRI)
        self.ax2.grid(True, alpha=0.15, color=BORDER)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _save_png(self):
        path = _fd.asksaveasfilename(defaultextension=".png",
                                     filetypes=[("PNG image", "*.png")],
                                     title="Save emanation plot")
        if not path: return
        try:
            rep.save_figure(self.fig, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    # ── measured-trace ingestion & calibration ──────────────────────────────
    def _cal_floats(self):
        """Read AF / cable / gain / distance entries, tolerating blanks."""
        def g(v, d):
            try: return float(v.get())
            except Exception: return d
        return (g(self.meas_af, 0.0), g(self.meas_cable, 0.0),
                g(self.meas_gain, 0.0), g(self.meas_dist, 1.0))

    def _load_measured(self):
        path = _fd.askopenfilename(
            filetypes=[("CSV trace", "*.csv"), ("Text", "*.txt"), ("All", "*.*")],
            title="Load measured emission trace (freq, level)")
        if not path: return
        try:
            freqs, level = tm.load_trace_csv(
                path, freq_col=0, level_col=1,
                freq_unit=self.meas_unit.get(), level_is_dbm=self.meas_dbm.get())
            af, cable, gain, _ = self._cal_floats()
            field = tm.field_strength_dbuv_per_m(level, af, cable, gain)
            self._measured = (freqs, field)
            self._measured_name = os.path.basename(path)
            self._plot()
        except Exception as e:
            messagebox.showerror("Trace load failed", str(e))

    def _calibrate_from_trace(self):
        if self._measured is None:
            messagebox.showinfo("No trace", "Load a measured trace first."); return
        try:
            _, _, _, dist = self._cal_floats()
            _, field = self._measured
            e0 = tm.calibrate_device(self.device_var.get(), field, dist)
            messagebox.showinfo(
                "Device calibrated",
                f"{self.device_var.get()} emission baseline set to {e0:.1f} dBµV/m "
                f"from the measured trace (referred to 1 m).\n\n"
                f"This now feeds the Room Assessment and zone model too.")
            self._plot()
        except Exception as e:
            messagebox.showerror("Calibration failed", str(e))

    def _clear_measured(self):
        self._measured = None
        self._measured_name = ""
        self._plot()

    def _overlay_compliance(self, freqs, spec_db, dist):
        """Draw the selected emission-limit mask over the spectrum and report a
        pass/fail verdict.  Evaluates measured data if loaded, else the model."""
        std = self.compl_var.get()
        if std == "None":
            self.compl_verdict.configure(text="")
            return
        limits = tc.mask_levels(freqs, std, at_distance_m=dist)
        self.ax1.plot(freqs / 1e6, limits, color="#f43f5e", lw=1.4, linestyle="--",
                      label=std.split(" (")[0] + " limit")
        # Assess measured trace if present (its own distance), otherwise the model.
        if self._measured is not None:
            mf, mv = self._measured
            _, _, _, mdist = self._cal_floats()
            res = tc.evaluate(mf, mv, std, at_distance_m=mdist)
            who = "Measured"
        else:
            res = tc.evaluate(freqs, spec_db, std, at_distance_m=dist)
            who = "Model"
        if res["n_exceed"] == 0:
            txt, col = f"✅ {who} PASS", ACCENT2
        else:
            txt = (f"❌ {who} FAIL  (+{res['worst_margin']:.1f} dB @ "
                   f"{res['worst_freq']/1e6:.2f} MHz)")
            col = WARN
        if tc.is_illustrative(std):
            txt += "  ⚠ illustrative limits"
        self.compl_verdict.configure(text=txt, text_color=col)


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 2 – SIGNAL CAPTURE & ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
class SignalAnalysis(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._data = None
        self._fs   = 44100
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Signal Capture & Analysis",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(self, text="Load .wav / .iq files or generate a test signal — FFT spectrum & spectrogram analysis",
                     font=ctk.CTkFont(size=12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=24, pady=(0, 16))

        # Top controls
        ctrl = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        ctrl.pack(fill="x", padx=24, pady=(0, 14))

        ctk.CTkButton(ctrl, text="📂  Load WAV File", width=160,
                      fg_color=ACCENT, hover_color=ACCENT_HV,
                      command=self._load_wav).grid(row=0, column=0, padx=12, pady=12)

        ctk.CTkButton(ctrl, text="🎲  Generate Test Signal", width=180,
                      fg_color=BG_CARD, hover_color=BORDER,
                      command=self._gen_test).grid(row=0, column=1, padx=8, pady=12)

        ctk.CTkLabel(ctrl, text="Window:", text_color=TEXT_SEC, font=ctk.CTkFont(size=12)).grid(row=0, column=2, padx=12)
        self.win_var = ctk.StringVar(value="Hann")
        ctk.CTkOptionMenu(ctrl, values=["Rectangular", "Hann", "Hamming", "Blackman", "Flat Top"],
                          variable=self.win_var, width=140, fg_color=BG_CARD, button_color=ACCENT,
                          command=lambda _: self._update_plots()
                          ).grid(row=0, column=3, padx=6, pady=12)

        ctk.CTkLabel(ctrl, text="FFT Size:", text_color=TEXT_SEC, font=ctk.CTkFont(size=12)).grid(row=0, column=4, padx=12)
        self.fft_var = ctk.StringVar(value="4096")
        ctk.CTkOptionMenu(ctrl, values=["512", "1024", "2048", "4096", "8192"],
                          variable=self.fft_var, width=100, fg_color=BG_CARD, button_color=ACCENT,
                          command=lambda _: self._update_plots()
                          ).grid(row=0, column=5, padx=6, pady=12)

        self.status_lbl = ctk.CTkLabel(ctrl, text="No signal loaded", text_color=TEXT_SEC,
                                       font=ctk.CTkFont(size=11))
        self.status_lbl.grid(row=0, column=6, padx=16)

        ctk.CTkButton(ctrl, text="🖼 Save PNG", width=110, fg_color=BG_CARD,
                      hover_color=BORDER, command=self._save_png).grid(row=0, column=7, padx=10)

        # Detection / classification banner
        self.detect_lbl = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12, weight="bold"),
                                       text_color=TEXT_SEC, anchor="w", justify="left")
        self.detect_lbl.pack(fill="x", padx=24, pady=(0, 6))

        # Canvas
        self.fig, self.axes = plt.subplots(3, 1, figsize=(11, 7), **PLOT_PARAMS)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self._draw_empty()

    def _draw_empty(self):
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(BG_PANEL)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)
        self.axes[0].text(0.5, 0.5, "Load a file or generate a test signal",
                          ha="center", va="center", color=TEXT_SEC, fontsize=13,
                          transform=self.axes[0].transAxes)
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _load_wav(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("IQ files", "*.iq"), ("All", "*.*")])
        if not path: return
        try:
            if path.lower().endswith(".iq"):
                self._data, self._fs = dsp.load_iq(path)
                kind = "IQ (complex)"
            else:
                self._data, self._fs = dsp.load_wav(path)
                kind = "WAV"
            self.status_lbl.configure(
                text=f"✅  {os.path.basename(path)}  |  {kind}  |  fs={self._fs/1e3:.1f} kHz  |  {len(self._data)} samples",
                text_color=ACCENT2)
            self._update_plots()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _gen_test(self):
        self._data, self._fs = dsp.synth_tempest_signal(fs=44100, duration=1.0)
        self.status_lbl.configure(
            text=f"✅  Synthetic TEMPEST signal  |  fs=44.1 kHz  |  {len(self._data)} samples",
            text_color=ACCENT2)
        self._update_plots()

    def _update_plots(self):
        if self._data is None: return
        N = int(self.fft_var.get())
        win = self.win_var.get()
        complex_in = np.iscomplexobj(self._data)
        try:
            freqs, mag_db, n_avg = dsp.averaged_spectrum(self._data, self._fs, nfft=N, window=win)
        except ValueError as e:
            messagebox.showwarning("Signal too short", str(e)); return

        for ax in self.axes:
            ax.clear(); ax.set_facecolor(BG_PANEL)
            ax.tick_params(colors=TEXT_SEC, labelsize=9)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)

        # Time domain — show I (and Q for complex captures)
        n_show = min(len(self._data), 2048)
        t_ax = np.arange(n_show) / self._fs * 1000
        self.axes[0].plot(t_ax, np.real(self._data[:n_show]), color=ACCENT, lw=0.7,
                          label=("I" if complex_in else None))
        if complex_in:
            self.axes[0].plot(t_ax, np.imag(self._data[:n_show]), color=ACCENT2, lw=0.7, label="Q")
            self.axes[0].legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_PRI)
        self.axes[0].set_ylabel("Amplitude", color=TEXT_SEC, fontsize=10)
        self.axes[0].set_xlabel("Time (ms)", color=TEXT_SEC, fontsize=10)
        self.axes[0].set_title("Time Domain" + (" (I/Q)" if complex_in else ""), color=TEXT_PRI, fontsize=11)
        self.axes[0].grid(True, alpha=0.15, color=BORDER)

        # Averaged, window-gain-corrected amplitude spectrum
        self.axes[1].plot(freqs / 1e3, mag_db, color=ACCENT2, lw=0.8)
        self.axes[1].fill_between(freqs / 1e3, mag_db.min(), mag_db, alpha=0.15, color=ACCENT2)
        # Peak detection + emanation-source classification
        self._analyse_peaks(freqs, mag_db)
        self.axes[1].set_ylabel("Amplitude (dB, win-corr.)", color=TEXT_SEC, fontsize=10)
        self.axes[1].set_xlabel("Frequency (kHz)", color=TEXT_SEC, fontsize=10)
        self.axes[1].set_title(
            f"Averaged Spectrum  |  {win}  |  N={N}  |  {n_avg} averages"
            + ("  |  two-sided (IQ)" if complex_in else ""),
            color=TEXT_PRI, fontsize=11)
        self.axes[1].grid(True, alpha=0.15, color=BORDER)

        # Spectrogram (tied to the selected window)
        f_sg, t_sg, S = dsp.make_spectrogram(self._data, self._fs,
                                             nfft=min(512, N), window=win)
        self.axes[2].pcolormesh(t_sg, f_sg / 1e3, S, cmap="plasma", shading="auto")
        self.axes[2].set_ylabel("Frequency (kHz)", color=TEXT_SEC, fontsize=10)
        self.axes[2].set_xlabel("Time (s)", color=TEXT_SEC, fontsize=10)
        self.axes[2].set_title("Spectrogram", color=TEXT_PRI, fontsize=11)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _analyse_peaks(self, freqs, mag_db):
        """Detect peaks, mark them, and classify the emanation source."""
        peaks = tsig.detect_peaks(freqs, mag_db, prominence_db=8, max_peaks=10, min_freq=0.0)
        for pf, pl in peaks:
            self.axes[1].plot(pf / 1e3, pl, "v", color=WARN, ms=6, zorder=5)
        if not peaks:
            self.detect_lbl.configure(text="🔍 No prominent peaks detected.", text_color=TEXT_SEC)
            return
        pfreqs = [p[0] for p in peaks]
        fams = tsig.harmonic_families(pfreqs)
        top = tsig.classify_device(pfreqs)[0]
        parts = ["🔍 Peaks: " + ", ".join(f"{p/1e3:.1f}k" for p in sorted(pfreqs)[:6])]
        if fams:
            parts.append(f"harmonic family f₀≈{fams[0]['fundamental']/1e3:.2f} kHz "
                         f"({fams[0]['count']} harmonics)")
        parts.append(f"likely source: {top['device']} ({top['matched']}/{top['n_peaks']})"
                     if top["score"] > 0 else "no device-library match")
        col = ACCENT2 if top["score"] >= 0.5 else (WARN if top["score"] > 0 else TEXT_SEC)
        self.detect_lbl.configure(text="   |   ".join(parts), text_color=col)

    def _save_png(self):
        path = _fd.asksaveasfilename(defaultextension=".png",
                                     filetypes=[("PNG image", "*.png")],
                                     title="Save signal analysis")
        if not path: return
        try:
            rep.save_figure(self.fig, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3 – SHIELDING EFFECTIVENESS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
class ShieldingCalculator(ctk.CTkFrame):
    # Single source of truth for material properties lives in the physics core.
    MATERIALS = tp.MATERIALS

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Shielding Effectiveness Calculator",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(self, text="Compute SE (dB) via absorption + reflection loss — IEEE 299 / MIL-STD-285 methodology",
                     font=ctk.CTkFont(size=12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=24, pady=(0, 16))

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=24)
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)

        # ── LEFT PANEL ──────────────────────────────────────────────
        left = ctk.CTkScrollableFrame(content, fg_color=BG_PANEL, corner_radius=12,
                                      border_width=1, border_color=BORDER, width=270)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 16), pady=(0, 20))

        def lbl(parent, text, col=TEXT_SEC):
            ctk.CTkLabel(parent, text=text, text_color=col,
                         font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(10, 0))

        lbl(left, "Shield Material", TEXT_PRI)
        self.mat_var = ctk.StringVar(value="Copper")
        ctk.CTkOptionMenu(left, values=list(self.MATERIALS.keys()), variable=self.mat_var,
                          fg_color=BG_CARD, button_color=ACCENT, width=250
                          ).pack(padx=16, pady=(4, 0))

        lbl(left, "Thickness (mm)")
        self.thick_var = ctk.StringVar(value="1.0")
        ctk.CTkEntry(left, textvariable=self.thick_var, width=250,
                     fg_color=BG_CARD, border_color=BORDER).pack(padx=16, pady=(4, 0))

        lbl(left, "Frequency Range")
        freq_row = ctk.CTkFrame(left, fg_color="transparent")
        freq_row.pack(fill="x", padx=16, pady=(4, 0))
        self.fmin_var = ctk.StringVar(value="1e3")
        self.fmax_var = ctk.StringVar(value="3e9")
        ctk.CTkEntry(freq_row, textvariable=self.fmin_var, width=110,
                     fg_color=BG_CARD, border_color=BORDER, placeholder_text="f_min Hz").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(freq_row, textvariable=self.fmax_var, width=110,
                     fg_color=BG_CARD, border_color=BORDER, placeholder_text="f_max Hz").pack(side="left")

        lbl(left, "Field Type")
        self.field_var = ctk.StringVar(value="Plane Wave")
        ctk.CTkOptionMenu(left, values=list(tp.FIELD_TYPES),
                          variable=self.field_var, fg_color=BG_CARD, button_color=ACCENT, width=250
                          ).pack(padx=16, pady=(4, 0))

        lbl(left, "Source Distance (m) — near-field E/H")
        self.dist_var = ctk.StringVar(value="1.0")
        ctk.CTkEntry(left, textvariable=self.dist_var, width=250,
                     fg_color=BG_CARD, border_color=BORDER).pack(padx=16, pady=(4, 0))

        lbl(left, "Aperture / seam (enclosure leakage)", TEXT_PRI)
        self.ap_on = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(left, text="Include aperture leakage", variable=self.ap_on,
                      command=self._calculate).pack(anchor="w", padx=16, pady=(2, 0))
        ap1 = ctk.CTkFrame(left, fg_color="transparent"); ap1.pack(fill="x", padx=16, pady=(4, 0))
        self.ap_len = ctk.StringVar(value="100")     # longest dimension (mm)
        self.ap_wid = ctk.StringVar(value="2")        # transverse dimension (mm)
        for var, ph in [(self.ap_len, "L mm"), (self.ap_wid, "W mm")]:
            ctk.CTkEntry(ap1, textvariable=var, width=112, fg_color=BG_CARD,
                         border_color=BORDER, placeholder_text=ph).pack(side="left", padx=(0, 6))
        ap2 = ctk.CTkFrame(left, fg_color="transparent"); ap2.pack(fill="x", padx=16, pady=(4, 0))
        self.ap_depth = ctk.StringVar(value="")       # blank → panel thickness
        self.ap_count = ctk.StringVar(value="1")
        ctk.CTkEntry(ap2, textvariable=self.ap_depth, width=74, fg_color=BG_CARD,
                     border_color=BORDER, placeholder_text="depth mm").pack(side="left", padx=(0, 6))
        ctk.CTkEntry(ap2, textvariable=self.ap_count, width=44, fg_color=BG_CARD,
                     border_color=BORDER, placeholder_text="N").pack(side="left", padx=(0, 6))
        self.ap_shape = ctk.StringVar(value="rectangular")
        ctk.CTkOptionMenu(ap2, values=["rectangular", "circular"], width=104,
                          variable=self.ap_shape, fg_color=BG_CARD, button_color=ACCENT
                          ).pack(side="left")

        lbl(left, "Laminate (multi-layer)", TEXT_PRI)
        self.mu_freq = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(left, text="Frequency-dependent µ (ferro)", variable=self.mu_freq,
                      command=self._calculate).pack(anchor="w", padx=16, pady=(2, 0))
        lam_row = ctk.CTkFrame(left, fg_color="transparent"); lam_row.pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkButton(lam_row, text="➕ Add as layer", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._add_layer).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(lam_row, text="✖ Clear", width=60, fg_color=BG_CARD, hover_color=BORDER,
                      command=self._clear_layers).pack(side="left", padx=(2, 0))
        self.stack_lbl = ctk.CTkLabel(left, text="Stack: (single layer)", font=ctk.CTkFont(size=11),
                                      text_color=TEXT_SEC, wraplength=250, justify="left")
        self.stack_lbl.pack(anchor="w", padx=16, pady=(4, 0))

        lbl(left, "Verification / target", TEXT_PRI)
        mrow = ctk.CTkFrame(left, fg_color="transparent"); mrow.pack(fill="x", padx=16, pady=(2, 0))
        ctk.CTkButton(mrow, text="📈 Load measured SE", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._load_measured_se).pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.mse_unit = ctk.StringVar(value="MHz")
        ctk.CTkOptionMenu(mrow, values=["Hz", "kHz", "MHz", "GHz"], width=72,
                          variable=self.mse_unit, fg_color=BG_CARD, button_color=ACCENT).pack(side="left")
        trow = ctk.CTkFrame(left, fg_color="transparent"); trow.pack(fill="x", padx=16, pady=(4, 0))
        self.target_se = ctk.StringVar(value="")
        self.area_var = ctk.StringVar(value="")
        ctk.CTkEntry(trow, textvariable=self.target_se, width=112, fg_color=BG_CARD,
                     border_color=BORDER, placeholder_text="target SE dB").pack(side="left", padx=(0, 6))
        ctk.CTkEntry(trow, textvariable=self.area_var, width=112, fg_color=BG_CARD,
                     border_color=BORDER, placeholder_text="area m²").pack(side="left")
        self.se_verdict = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=12, weight="bold"),
                                       text_color=TEXT_SEC, wraplength=250, justify="left")
        self.se_verdict.pack(anchor="w", padx=16, pady=(6, 0))

        ctk.CTkButton(left, text="⚡  Calculate", width=250, height=40,
                      fg_color=ACCENT, hover_color=ACCENT_HV, font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._calculate).pack(padx=16, pady=(16, 4))

        exp_row = ctk.CTkFrame(left, fg_color="transparent")
        exp_row.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkButton(exp_row, text="💾 CSV", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._export_csv).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(exp_row, text="🖼 PNG", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._save_png).pack(side="left", expand=True, fill="x", padx=(2, 0))

        # Result cards
        self._last = None                     # cached curves for CSV export
        self._layers = []                     # laminate stack [{material, thickness_mm}]
        self._measured_se = None              # (freqs_Hz, SE_dB) of a loaded trace
        self._measured_se_name = ""
        self.result_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.result_frame.pack(fill="x", padx=16, pady=(0, 16))

        # ── RIGHT PANEL (plot) ───────────────────────────────────────
        right = ctk.CTkFrame(content, fg_color=BG_PANEL, corner_radius=12,
                             border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="nsew", pady=(0, 20))

        self.fig, (self.ax_se, self.ax_sd) = plt.subplots(2, 1, figsize=(9, 6.5), **PLOT_PARAMS)
        self.fig.tight_layout(pad=3.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._placeholder()

    def _placeholder(self):
        for ax in (self.ax_se, self.ax_sd):
            ax.clear(); ax.set_facecolor(BG_PANEL)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)
        self.ax_se.text(0.5, 0.5, "Configure parameters and click Calculate",
                        ha="center", va="center", color=TEXT_SEC, fontsize=12,
                        transform=self.ax_se.transAxes)
        self.fig.tight_layout(pad=3.5)
        self.canvas.draw()

    def _get_distance(self):
        """Source distance [m] for the near-field E/H model; robust to bad input."""
        try:
            return max(1e-3, float(self.dist_var.get()))
        except Exception:
            return 1.0

    def _se_formula(self, f, sigma, mu_r, t_m):
        """Delegate to the validated physics core (see tempest_physics.py)."""
        out = tp.shielding_effectiveness(
            f, sigma, mu_r, t_m,
            r=self._get_distance(), field=self.field_var.get())
        return out["SE"], out["A"], out["R"], out["B"], out["skin_depth"] * 1000

    def _aperture_spec(self, t_m):
        """Build an aperture dict from the inputs, or None if disabled/invalid.
        Depth defaults to the panel thickness when the field is left blank."""
        if not self.ap_on.get():
            return None
        try:
            depth_s = self.ap_depth.get().strip()
            return {"length": float(self.ap_len.get()) / 1000.0,
                    "width": float(self.ap_wid.get()) / 1000.0,
                    "depth": float(depth_s) / 1000.0 if depth_s else t_m,
                    "count": max(1, int(float(self.ap_count.get() or 1))),
                    "shape": self.ap_shape.get()}
        except Exception:
            return None

    @staticmethod
    def _float_or_none(s):
        try:
            s = s.strip()
            return float(s) if s else None
        except Exception:
            return None

    def _mu_r(self, f, mat):
        """Relative permeability — frequency-dependent for ferromagnetics if the
        toggle is on and the material has a relaxation frequency."""
        if self.mu_freq.get() and mat.get("f_mu"):
            return tp.effective_mu_r(f, mat["mu_r"], mat["f_mu"])
        return mat["mu_r"]

    def _layer_specs(self):
        """Laminate as [{material, thickness_mm}]; the single input if none added."""
        if self._layers:
            return list(self._layers)
        return [{"material": self.mat_var.get(), "thickness_mm": self.thick_var.get()}]

    def _add_layer(self):
        try:
            float(self.thick_var.get())
        except Exception:
            messagebox.showerror("Input Error", "Enter a valid thickness first."); return
        self._layers.append({"material": self.mat_var.get(), "thickness_mm": self.thick_var.get()})
        self._calculate()

    def _clear_layers(self):
        self._layers.clear()
        self._calculate()

    def _load_measured_se(self):
        path = _fd.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")],
                                   title="Load measured SE trace (frequency, SE dB)")
        if not path: return
        try:
            f, se = tm.load_trace_csv(path, 0, 1, freq_unit=self.mse_unit.get())
            self._measured_se = (f, se); self._measured_se_name = os.path.basename(path)
            self._calculate()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _calculate(self):
        try:
            fmin = float(self.fmin_var.get()); fmax = float(self.fmax_var.get())
            specs = self._layer_specs()
            for ly in specs:
                float(ly["thickness_mm"])
        except Exception as e:
            messagebox.showerror("Input Error", str(e)); return

        freqs = np.logspace(np.log10(fmin), np.log10(fmax), 500)
        multilayer = len(specs) > 1
        stack, mass_layers, total_t = [], [], 0.0
        for ly in specs:
            m = self.MATERIALS[ly["material"]]
            t = float(ly["thickness_mm"]) / 1000.0
            total_t += t
            stack.append((m["sigma"], self._mu_r(freqs, m), t))
            mass_layers.append((m["density"], t))

        if multilayer:
            SE = tp.multilayer_se_db(freqs, stack, r=self._get_distance(), field=self.field_var.get())
            A = R = B = None
            skin = tp.skin_depth(freqs, stack[0][0], stack[0][1]) * 1000.0
        else:
            SE, A, R, B, skin = self._se_formula(freqs, stack[0][0], stack[0][1], stack[0][2])

        # Aperture / enclosure leakage (optional): usually the real limit.
        ap = self._aperture_spec(total_t)
        enc = ap_se = None
        if ap is not None:
            ap_se = tp.aperture_se_db(freqs, ap["length"], ap["width"], ap["depth"], ap["count"], ap["shape"])
            enc = tp.combine_se_db(SE, ap_se)
        effective = enc if enc is not None else SE

        stack_txt = " + ".join(f"{s['material']} {float(s['thickness_mm']):g}mm" for s in specs)
        self._last = {"freqs": freqs, "SE": SE, "A": A, "R": R, "B": B, "skin": skin,
                      "enclosure": enc, "aperture": ap_se, "stack": stack_txt}
        self.stack_lbl.configure(text="Stack: " + (stack_txt if multilayer else "(single layer)"))

        mid = len(freqs) // 2
        target = self._float_or_none(self.target_se.get())
        area = self._float_or_none(self.area_var.get())
        mass = tp.shield_mass_kg(mass_layers, area) if area else None
        worst = float(np.min(effective)); wf = freqs[int(np.argmin(effective))]
        if target is not None:
            ok = worst >= target
            self.se_verdict.configure(
                text=(f"✅ meets {target:g} dB (min {worst:.0f} dB @ {wf/1e6:.2f} MHz)" if ok
                      else f"❌ fails {target:g} dB (min {worst:.0f} dB @ {wf/1e6:.2f} MHz)"),
                text_color=ACCENT2 if ok else WARN)
        else:
            self.se_verdict.configure(text="")

        # Result cards
        for w in self.result_frame.winfo_children(): w.destroy()
        cards = []
        if enc is not None:
            cards.append(("Enclosure SE at Mid-Band Frequency", enc[mid], "dB", "#f43f5e"))
        cards.append(("Shielding Effectiveness at Mid-Band Frequency", SE[mid], "dB", ACCENT2))
        cards.append(("Min SE (band)", worst, "dB", "#22d3ee"))
        if not multilayer:
            cards += [("Absorption (A)", A[mid], "dB", ACCENT),
                      ("Reflection (R)", R[mid], "dB", "#d29922"),
                      ("Multi-Reflect (B)", B[mid], "dB", "#c084fc")]
        cards.append(("Skin Depth", skin[mid], "mm", WARN))
        if mass is not None:
            cards.append(("Shield mass", mass, "kg", "#a3e635"))
        for label, val, unit, color in cards:
            card = ctk.CTkFrame(self.result_frame, fg_color=BG_CARD, corner_radius=8)
            card.pack(fill="x", pady=3)
            ctk.CTkLabel(card, text=label, font=ctk.CTkFont(size=11), text_color=TEXT_SEC,
                        wraplength=230, justify="left").pack(anchor="w", padx=10, pady=(5, 0))
            ctk.CTkLabel(card, text=f"{val:.1f} {unit}", font=ctk.CTkFont(size=15, weight="bold"),
                         text_color=color).pack(anchor="w", padx=10, pady=(0, 5))

        # Plots
        for ax in (self.ax_se, self.ax_sd):
            ax.clear(); ax.set_facecolor(BG_PANEL)
            ax.tick_params(colors=TEXT_SEC, labelsize=9)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)

        self.ax_se.semilogx(freqs, SE, color=ACCENT2, lw=2.0,
                            label=("Laminate SE" if multilayer else "Total SE = A+R+B"))
        if not multilayer:
            self.ax_se.semilogx(freqs, A, color=ACCENT, lw=1.5, linestyle="--", label="Absorption (A)")
            self.ax_se.semilogx(freqs, R, color="#d29922", lw=1.5, linestyle=":", label="Reflection (R)")
            self.ax_se.semilogx(freqs, B, color="#c084fc", lw=1.0, linestyle="-.", label="Multi-reflection (B)")
        if enc is not None:
            self.ax_se.semilogx(freqs, ap_se, color="#f97316", lw=1.3, linestyle="--", label="Aperture SE")
            self.ax_se.semilogx(freqs, enc, color="#f43f5e", lw=2.4, label="Enclosure SE")
        if self._measured_se is not None:
            mf, mv = self._measured_se
            self.ax_se.semilogx(mf, mv, color="#22d3ee", lw=1.4, marker="o", ms=2,
                                label=f"Measured: {self._measured_se_name}")
        if target is not None:
            self.ax_se.axhline(target, color=ACCENT2, lw=1.0, linestyle="--", label=f"Target {target:g} dB")
        self.ax_se.axhline(20, color=WARN, lw=0.8, linestyle="--", label="20 dB minimum")
        self.ax_se.axhline(80, color="#8b949e", lw=0.7, linestyle="--", label="NATO SDIP-27 Level A")
        if enc is not None:
            self.ax_se.set_ylim(0, max(120.0, float(np.nanmax(enc)) + 40.0))
        elif multilayer:
            self.ax_se.set_ylim(0, min(400.0, float(np.nanmax(SE)) + 40.0))
        self.ax_se.set_xlabel("Frequency (Hz)", color=TEXT_SEC, fontsize=10)
        self.ax_se.set_ylabel("Shielding Effectiveness (dB)", color=TEXT_SEC, fontsize=10)
        title = stack_txt if multilayer else f"{self.mat_var.get()} — t={float(self.thick_var.get()):.2f} mm"
        self.ax_se.set_title(f"{title}  |  {self.field_var.get()}  |  r={self._get_distance():.2f} m",
                             color=TEXT_PRI, fontsize=10)
        self.ax_se.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_PRI)
        self.ax_se.grid(True, which="both", alpha=0.15, color=BORDER)

        self.ax_sd.semilogx(freqs, skin, color=WARN, lw=2.0)
        self.ax_sd.set_xlabel("Frequency (Hz)", color=TEXT_SEC, fontsize=10)
        self.ax_sd.set_ylabel("Skin Depth (mm)", color=TEXT_SEC, fontsize=10)
        self.ax_sd.set_title("Skin Depth vs Frequency", color=TEXT_PRI)
        self.ax_sd.grid(True, which="both", alpha=0.15, color=BORDER)

        self.fig.tight_layout(pad=3.5)
        self.canvas.draw()

    def _export_csv(self):
        if not self._last:
            messagebox.showinfo("Nothing to export", "Run a calculation first."); return
        path = _fd.asksaveasfilename(defaultextension=".csv",
                                     filetypes=[("CSV", "*.csv")],
                                     title="Export SE vs frequency")
        if not path: return
        d = self._last
        n = len(d["freqs"])
        col = lambda k: d[k] if d.get(k) is not None else [""] * n
        rows = zip(d["freqs"], d["SE"], col("A"), col("R"), col("B"),
                   col("enclosure"), d["skin"])
        try:
            rep.export_csv(path,
                           ["frequency_Hz", "SE_dB", "absorption_dB", "reflection_dB",
                            "multiref_dB", "enclosure_dB", "skin_depth_mm"], rows)
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _save_png(self):
        path = _fd.asksaveasfilename(defaultextension=".png",
                                     filetypes=[("PNG image", "*.png")],
                                     title="Save plot")
        if not path: return
        try:
            rep.save_figure(self.fig, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 4 – ROOM TEMPEST ASSESSMENT TOOL
# ─────────────────────────────────────────────────────────────────────────────
class RoomAssessment(ctk.CTkFrame):
    ZONE_COLORS = {
        "Zone 0 (Controlled)": ("#ff6b6b", 0.30),
        "Zone 1 (Sensitive)":  ("#ffa94d", 0.25),
        "Zone 2 (Monitored)":  ("#ffe066", 0.20),
        "Zone 3 (Public)":     ("#69db7c", 0.15),
    }
    WALL_POS = {"N": "North", "E": "East", "S": "South", "W": "West"}

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._devices: list[dict] = []
        self._build_ui()
        self._plot()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Room TEMPEST Assessment Tool",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(self, text="Per-wall shielding, window/door apertures & inspectable-space "
                     "analysis — does emanation escape the controlled perimeter?",
                     font=ctk.CTkFont(size=12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=24, pady=(0, 16))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(body, fg_color=BG_PANEL, corner_radius=12,
                                      border_width=1, border_color=BORDER, width=280)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        def sec(text):
            ctk.CTkLabel(left, text=text, font=ctk.CTkFont(size=14, weight="bold"),
                         text_color=TEXT_PRI).pack(anchor="w", padx=14, pady=(14, 2))

        def field(label, var_name, default, cb=False):
            ctk.CTkLabel(left, text=label, font=ctk.CTkFont(size=12),
                         text_color=TEXT_SEC).pack(anchor="w", padx=14, pady=(6, 0))
            v = ctk.StringVar(value=default)
            setattr(self, var_name, v)
            ctk.CTkEntry(left, textvariable=v, fg_color=BG_CARD,
                         border_color=BORDER).pack(fill="x", padx=14, pady=(2, 0))

        sec("Room")
        field("Room Width (m)", "room_w", "10")
        field("Room Length (m)", "room_l", "8")

        sec("Wall Shielding SE (dB)")
        wall_row = ctk.CTkFrame(left, fg_color="transparent")
        wall_row.pack(fill="x", padx=14, pady=(2, 0))
        self.wall_vars = {}
        for side in ("N", "E", "S", "W"):
            col = ctk.CTkFrame(wall_row, fg_color="transparent")
            col.pack(side="left", expand=True, fill="x", padx=1)
            ctk.CTkLabel(col, text=side, font=ctk.CTkFont(size=10),
                         text_color=TEXT_SEC).pack()
            v = ctk.StringVar(value="20")
            self.wall_vars[side] = v
            ctk.CTkEntry(col, textvariable=v, width=52, fg_color=BG_CARD,
                         border_color=BORDER).pack()

        sec("Window / Door (aperture)")
        self.ap_on = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(left, text="Include aperture", variable=self.ap_on,
                      command=self._plot).pack(anchor="w", padx=14, pady=(2, 0))
        aprow = ctk.CTkFrame(left, fg_color="transparent"); aprow.pack(fill="x", padx=14, pady=(4, 0))
        ctk.CTkLabel(aprow, text="on wall", font=ctk.CTkFont(size=11),
                     text_color=TEXT_SEC).pack(side="left", padx=(0, 6))
        self.ap_wall = ctk.StringVar(value="E")
        ctk.CTkOptionMenu(aprow, values=["N", "E", "S", "W"], width=64,
                          variable=self.ap_wall, fg_color=BG_CARD, button_color=ACCENT,
                          command=lambda _: self._plot()).pack(side="left")
        aprow2 = ctk.CTkFrame(left, fg_color="transparent"); aprow2.pack(fill="x", padx=14, pady=(4, 0))
        self.ap_len = ctk.StringVar(value="1.0")
        self.ap_wid = ctk.StringVar(value="0.5")
        self.ap_depth = ctk.StringVar(value="0.1")
        for var, ph in [(self.ap_len, "L m"), (self.ap_wid, "W m"), (self.ap_depth, "t m")]:
            ctk.CTkEntry(aprow2, textvariable=var, width=70, fg_color=BG_CARD,
                         border_color=BORDER, placeholder_text=ph).pack(side="left", padx=(0, 4))
        field("Assessment frequency (MHz)", "assess_freq", "100")

        sec("Inspectable space")
        field("Controlled-perimeter standoff (m)", "standoff", "5")
        erow = ctk.CTkFrame(left, fg_color="transparent"); erow.pack(fill="x", padx=14, pady=(6, 0))
        ctk.CTkLabel(erow, text="Eavesdropper X,Y (blank=none)", font=ctk.CTkFont(size=11),
                     text_color=TEXT_SEC).pack(anchor="w")
        erow2 = ctk.CTkFrame(left, fg_color="transparent"); erow2.pack(fill="x", padx=14, pady=(2, 0))
        self.eav_x = ctk.StringVar(value=""); self.eav_y = ctk.StringVar(value="")
        for var, ph in [(self.eav_x, "X m"), (self.eav_y, "Y m")]:
            ctk.CTkEntry(erow2, textvariable=var, width=108, fg_color=BG_CARD,
                         border_color=BORDER, placeholder_text=ph).pack(side="left", padx=(0, 6))

        sec("Add Device")
        ctk.CTkLabel(left, text="Device Type", font=ctk.CTkFont(size=12),
                     text_color=TEXT_SEC).pack(anchor="w", padx=14, pady=(6, 0))
        self.dev_type = ctk.StringVar(value="CRT Monitor")
        ctk.CTkOptionMenu(left, values=list(tp.DEVICES.keys()), variable=self.dev_type,
                          fg_color=BG_CARD, button_color=ACCENT).pack(fill="x", padx=14, pady=(2, 0))
        drow = ctk.CTkFrame(left, fg_color="transparent"); drow.pack(fill="x", padx=14, pady=(4, 0))
        self.dev_x = ctk.StringVar(value="5"); self.dev_y = ctk.StringVar(value="4")
        self.dev_pow = ctk.StringVar(value="10")
        for var, ph in [(self.dev_x, "X m"), (self.dev_y, "Y m"), (self.dev_pow, "dBm")]:
            ctk.CTkEntry(drow, textvariable=var, width=70, fg_color=BG_CARD,
                         border_color=BORDER, placeholder_text=ph).pack(side="left", padx=(0, 4))

        ctk.CTkButton(left, text="➕  Add Device", fg_color=ACCENT2, text_color="#ffffff",
                      hover_color="#26744a", font=ctk.CTkFont(weight="bold"),
                      command=self._add_device).pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkButton(left, text="🔄  Update assessment", fg_color=ACCENT,
                      hover_color=ACCENT_HV, command=self._plot).pack(fill="x", padx=14, pady=2)
        ctk.CTkButton(left, text="🧮  Solve requirements", fg_color=ACCENT,
                      hover_color=ACCENT_HV, command=self._solve).pack(fill="x", padx=14, pady=2)
        ctk.CTkButton(left, text="🗑  Clear All", fg_color=BG_CARD,
                      hover_color=BORDER, command=self._clear).pack(fill="x", padx=14, pady=2)

        ctk.CTkButton(left, text="📊  Report", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._report).pack(fill="x", padx=14, pady=(10, 2))
        ctk.CTkButton(left, text="📄  Export PDF", fg_color=BG_CARD,
                      hover_color=BORDER, command=self._export_pdf).pack(fill="x", padx=14, pady=2)
        row_io = ctk.CTkFrame(left, fg_color="transparent"); row_io.pack(fill="x", padx=14, pady=2)
        ctk.CTkButton(row_io, text="💾 Save", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._save_layout).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(row_io, text="📂 Load", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._load_layout).pack(side="left", expand=True, fill="x", padx=(2, 0))
        ctk.CTkButton(left, text="🖼  Save Map PNG", fg_color=BG_CARD,
                      hover_color=BORDER, command=self._save_png).pack(fill="x", padx=14, pady=(2, 8))

        # Verdict + risk
        self.verdict_lbl = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=13, weight="bold"),
                                        text_color=TEXT_SEC, wraplength=250, justify="left")
        self.verdict_lbl.pack(anchor="w", padx=14, pady=(12, 2))
        self.risk_lbl = ctk.CTkLabel(left, text="Risk Score: —",
                                     font=ctk.CTkFont(size=13, weight="bold"), text_color=WARN)
        self.risk_lbl.pack(anchor="w", padx=14, pady=(2, 10))

        # Map
        right = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=12,
                             border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="nsew")
        self.fig_room, self.ax_room = plt.subplots(1, 1, figsize=(9, 7), **PLOT_PARAMS)
        self.fig_room.tight_layout(pad=2.0)
        self.canvas_room = FigureCanvasTkAgg(self.fig_room, master=right)
        self.canvas_room.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    # ── input helpers ───────────────────────────────────────────────────────
    def _room_dims(self):
        try:
            return float(self.room_w.get()), float(self.room_l.get())
        except Exception:
            return 10.0, 8.0

    def _wall_se_dict(self):
        out = {}
        for s, v in self.wall_vars.items():
            try: out[s] = float(v.get())
            except Exception: out[s] = 0.0
        return out

    def _assess_freq(self):
        try:
            return max(1e3, float(self.assess_freq.get()) * 1e6)
        except Exception:
            return 1e8

    def _standoff(self):
        try:
            return max(0.1, float(self.standoff.get()))
        except Exception:
            return 5.0

    def _aperture_dict(self):
        if not self.ap_on.get():
            return {}
        try:
            return {self.ap_wall.get(): {"length": float(self.ap_len.get()),
                                         "width": float(self.ap_wid.get()),
                                         "depth": float(self.ap_depth.get())}}
        except Exception:
            return {}

    def _eavesdropper(self):
        try:
            return (float(self.eav_x.get()), float(self.eav_y.get()))
        except Exception:
            return None

    def _assessment(self):
        W, L = self._room_dims()
        return tr.assess(self._devices, self._wall_se_dict(), W, L, self._standoff(),
                         self._aperture_dict(), self._assess_freq(), self._eavesdropper())

    # ── actions ─────────────────────────────────────────────────────────────
    def _add_device(self):
        try:
            self._devices.append({"type": self.dev_type.get(),
                                  "x": float(self.dev_x.get()), "y": float(self.dev_y.get()),
                                  "power": float(self.dev_pow.get())})
            self._plot()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    def _clear(self):
        self._devices.clear()
        self._plot()

    def _plot(self):
        W, L = self._room_dims()
        so = self._standoff()
        eff = tr.effective_walls(self._wall_se_dict(), self._aperture_dict(), self._assess_freq())
        res = self._assessment()
        ax = self.ax_room
        ax.clear(); ax.set_facecolor(BG_PANEL)
        pad = so + 1.5
        ax.set_xlim(-pad, W + pad); ax.set_ylim(-pad, L + pad)
        ax.tick_params(colors=TEXT_SEC, labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor(BORDER)

        # Controlled perimeter
        ax.add_patch(Rectangle((-so, -so), W + 2 * so, L + 2 * so, linewidth=1.4,
                               edgecolor="#8b949e", facecolor="none", linestyle="--", zorder=1))
        ax.text(W / 2, L + so + 0.2, f"Controlled perimeter (standoff {so:g} m)",
                ha="center", color=TEXT_SEC, fontsize=9)

        # Room + per-wall SE labels
        ax.add_patch(Rectangle((0, 0), W, L, linewidth=2.5, edgecolor=TEXT_PRI,
                               facecolor="#1c2333", zorder=2))
        for side, (px, py, rot) in {"N": (W/2, L, 0), "S": (W/2, 0, 0),
                                    "E": (W, L/2, 90), "W": (0, L/2, 90)}.items():
            ax.text(px, py, f" {side}:{eff[side]:.0f}dB ", ha="center", va="center",
                    color=ACCENT2 if eff[side] >= 40 else WARN, fontsize=8, rotation=rot,
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG_CARD, ec=BORDER, alpha=0.9), zorder=7)

        # Aperture marker
        aps = self._aperture_dict()
        for side, ap in aps.items():
            seg = {"N": [(W/2 - ap["length"]/2, L), (W/2 + ap["length"]/2, L)],
                   "S": [(W/2 - ap["length"]/2, 0), (W/2 + ap["length"]/2, 0)],
                   "E": [(W, L/2 - ap["length"]/2), (W, L/2 + ap["length"]/2)],
                   "W": [(0, L/2 - ap["length"]/2), (0, L/2 + ap["length"]/2)]}[side]
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                    color="#f43f5e", lw=5, zorder=8, solid_capstyle="butt")

        # Device zones
        for dev in self._devices:
            x, y, pw = dev["x"], dev["y"], dev["power"]
            for zone, (color, alpha) in reversed(list(self.ZONE_COLORS.items())):
                r = tp.zone_radius_m(dev["type"], pw, tp.ZONE_THRESHOLDS_DBUV[zone])
                ax.add_patch(plt.Circle((x, y), r, color=color, alpha=alpha, zorder=3))
            ax.plot(x, y, "w^", ms=9, zorder=5)
            ax.annotate(dev["type"], (x, y), textcoords="offset points", xytext=(6, 6),
                        color=TEXT_PRI, fontsize=8, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.2", fc=BG_CARD, ec=BORDER, alpha=0.85))

        # Worst-case perimeter leak point
        if res["worst_point"] is not None:
            wx, wy = res["worst_point"]
            leaked = res["worst_field"]
            ax.plot(wx, wy, "X", color=("#f43f5e" if not res["passed"] else ACCENT2),
                    ms=14, mew=2, zorder=9)
            ax.annotate(f"worst leak\n{leaked:.0f} dBµV/m", (wx, wy), textcoords="offset points",
                        xytext=(8, 8), color=TEXT_PRI, fontsize=8, zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc=BG_CARD, ec=BORDER, alpha=0.9))

        # Eavesdropper
        eav = self._eavesdropper()
        if eav is not None and "eaves_field" in res:
            ax.plot(eav[0], eav[1], "*", color="#22d3ee", ms=16, zorder=9)
            ax.annotate(f"eavesdropper\n{res['eaves_field']:.0f} dBµV/m", eav,
                        textcoords="offset points", xytext=(8, -18), color="#22d3ee",
                        fontsize=8, zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc=BG_CARD, ec=BORDER, alpha=0.9))

        ax.set_xlabel("X (m)", color=TEXT_SEC, fontsize=10)
        ax.set_ylabel("Y (m)", color=TEXT_SEC, fontsize=10)
        ax.set_title("Inspectable-Space Assessment", color=TEXT_PRI, fontsize=13)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.1, color=BORDER, zorder=0)
        self.fig_room.tight_layout(pad=2.0)
        self.canvas_room.draw()

        # Verdict + risk
        if not self._devices:
            self.verdict_lbl.configure(text="Add a device to assess.", text_color=TEXT_SEC)
        elif res["passed"]:
            self.verdict_lbl.configure(
                text=f"✅ CONTAINED — worst perimeter field {res['worst_field']:.0f} dBµV/m "
                     f"({-res['margin']:.0f} dB below detectability).", text_color=ACCENT2)
        else:
            self.verdict_lbl.configure(
                text=f"❌ INTERCEPTABLE — {res['worst_field']:.0f} dBµV/m at the perimeter, "
                     f"+{res['margin']:.0f} dB over detectability.", text_color=WARN)
        risk = int(round(res["risk"]))
        rc = ACCENT2 if risk < 30 else ("#d29922" if risk < 70 else WARN)
        self.risk_lbl.configure(text=f"Risk Score: {risk}/100", text_color=rc)

    def _solve(self):
        if not self._devices:
            messagebox.showinfo("No devices", "Add at least one device first."); return
        W, L = self._room_dims()
        rec = tr.design_recommendations(self._devices, self._wall_se_dict(), W, L,
                                        self._standoff(), self._aperture_dict(), self._assess_freq())
        if rec["passed"]:
            messagebox.showinfo("Design solver",
                                f"✅ Already contained.\nWorst perimeter field "
                                f"{rec['worst_field']:.0f} dBµV/m, {-rec['margin']:.0f} dB "
                                f"below the detectability floor.")
            return
        lines = ["❌ Emanation escapes the controlled perimeter.", "",
                 f"Worst leak: {rec['worst_field']:.0f} dBµV/m  (+{rec['margin']:.0f} dB over floor)",
                 f"Dominant path: {rec['offending_device']} through the "
                 f"{self.WALL_POS.get(rec['offending_wall'], '?')} wall", "",
                 "To contain it, EITHER:",
                 f"  • raise {self.WALL_POS.get(rec['offending_wall'],'that')}-wall SE "
                 f"{rec['current_wall_se']:.0f} → {rec['target_wall_se']:.0f} dB "
                 f"(+{rec['add_wall_se_db']:.0f} dB)"]
        if rec.get("aperture_limited"):
            lines.append("    ⚠ that wall is APERTURE-limited — improve the window/door "
                         "(honeycomb vent / waveguide gasket), bulk shielding won't help.")
        m = rec.get("material")
        if m:
            lines.append(f"    e.g. {m['material']} {m['thickness_mm']:.2f} mm")
        if rec.get("required_standoff_m"):
            lines.append(f"  • OR extend the controlled perimeter to "
                         f"{rec['required_standoff_m']:.1f} m standoff")
        messagebox.showinfo("Design solver — mitigation", "\n".join(lines))

    def _report(self):
        W, L = self._room_dims()
        res = self._assessment()
        eff = res["eff_walls"]
        lines = ["TEMPEST ROOM ASSESSMENT REPORT", "=" * 42, "",
                 f"Room:        {W:g} m x {L:g} m",
                 f"Standoff:    {self._standoff():g} m   Assess freq: {self._assess_freq()/1e6:g} MHz",
                 "Wall SE (effective, dB):  " + "  ".join(f"{s}={eff[s]:.0f}" for s in ("N", "E", "S", "W")),
                 f"Devices:     {len(self._devices)}", ""]
        if self._devices and res["worst_point"] is not None:
            verdict = "CONTAINED" if res["passed"] else "INTERCEPTABLE"
            lines.append(f"VERDICT:  {verdict}   (worst perimeter field "
                         f"{res['worst_field']:.0f} dBuV/m, margin {res['margin']:+.0f} dB)")
            lines.append(f"Risk score: {res['risk']:.0f}/100")
            if "eaves_field" in res:
                lines.append(f"Eavesdropper field: {res['eaves_field']:.0f} dBuV/m "
                             f"(margin {res['eaves_margin']:+.0f} dB)")
        lines += ["", "Devices:"]
        for i, d in enumerate(self._devices, 1):
            lines.append(f"  [{i}] {d['type']} @ ({d['x']},{d['y']}) — {d['power']} dBm")
        lines += ["", "Generated by TEMPEST Analysis Suite — MSc Thesis Tool"]
        messagebox.showinfo("TEMPEST Assessment Report", "\n".join(lines))

    def _export_pdf(self):
        path = _fd.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
                                     title="Export TEMPEST assessment PDF")
        if not path: return
        try:
            W, L = self._room_dims()
            res = self._assessment(); eff = res["eff_walls"]
            png = os.path.join(tempfile.gettempdir(), "_tempest_zonemap.png")
            rep.save_figure(self.fig_room, png)
            verdict = "—"
            if self._devices and res["worst_point"] is not None:
                verdict = ("CONTAINED" if res["passed"] else "INTERCEPTABLE") + \
                          f"  ({res['worst_field']:.0f} dBµV/m, {res['margin']:+.0f} dB)"
            # per-device contribution at the worst-case point
            rows = []
            if res["worst_point"] is not None:
                _, contribs = tr.field_at_point(self._devices, eff, W, L, res["worst_point"])
                rows = [[i, c["type"], f'({c["x"]},{c["y"]})',
                         self.WALL_POS.get(c["wall"], "-"), f'{c["e_free"]:.0f}',
                         f'{c["leaked"]:.0f}'] for i, c in enumerate(contribs, 1)]
            rep.build_pdf_report(
                path, "TEMPEST Room Assessment Report",
                subtitle="MSc Telecommunications Engineering — TEMPEST Analysis Suite",
                meta={"Room": f"{W:g} m × {L:g} m",
                      "Wall SE (N/E/S/W)": "/".join(f"{eff[s]:.0f}" for s in ("N", "E", "S", "W")) + " dB",
                      "Standoff": f"{self._standoff():g} m",
                      "Assessment freq": f"{self._assess_freq()/1e6:g} MHz",
                      "Verdict": verdict, "Risk": f"{res['risk']:.0f}/100"},
                tables=[("Per-Device Contribution at Worst Perimeter Point",
                         ["#", "Device", "Position", "Exit wall",
                          "Free-space dBµV/m", "Leaked dBµV/m"], rows)],
                image_path=png,
                notes=["Emission baselines and the detectability floor are representative, "
                       "illustrative values (NATO SDIP-27 style) — not calibrated "
                       "measurements. See tempest_physics.py / tempest_room.py."])
            messagebox.showinfo("Export complete", f"PDF saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _save_layout(self):
        path = _fd.asksaveasfilename(defaultextension=".json", filetypes=[("JSON layout", "*.json")],
                                     title="Save room layout")
        if not path: return
        W, L = self._room_dims()
        try:
            rep.save_session(path, {
                "room_w": W, "room_l": L, "walls": self._wall_se_dict(),
                "aperture": self._aperture_dict(), "standoff": self._standoff(),
                "assess_freq_mhz": self._assess_freq() / 1e6,
                "eavesdropper": self._eavesdropper(), "devices": self._devices})
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _load_layout(self):
        path = _fd.askopenfilename(filetypes=[("JSON layout", "*.json")], title="Load room layout")
        if not path: return
        try:
            obj = rep.load_session(path)
            self.room_w.set(str(obj.get("room_w", 10)))
            self.room_l.set(str(obj.get("room_l", 8)))
            walls = obj.get("walls") or {s: obj.get("wall_se", 20) for s in ("N", "E", "S", "W")}
            for s in ("N", "E", "S", "W"):
                self.wall_vars[s].set(str(walls.get(s, 20)))
            self.standoff.set(str(obj.get("standoff", 5)))
            self.assess_freq.set(str(obj.get("assess_freq_mhz", 100)))
            ap = (obj.get("aperture") or {})
            if ap:
                side = next(iter(ap)); spec = ap[side]
                self.ap_on.set(True); self.ap_wall.set(side)
                self.ap_len.set(str(spec.get("length", 1.0)))
                self.ap_wid.set(str(spec.get("width", 0.5)))
                self.ap_depth.set(str(spec.get("depth", 0.1)))
            else:
                self.ap_on.set(False)
            eav = obj.get("eavesdropper")
            self.eav_x.set("" if not eav else str(eav[0]))
            self.eav_y.set("" if not eav else str(eav[1]))
            self._devices = list(obj.get("devices", []))
            self._plot()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _save_png(self):
        path = _fd.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")],
                                     title="Save assessment map")
        if not path: return
        try:
            rep.save_figure(self.fig_room, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 5 – EMANATION COVERAGE & INTERCEPTION MAP
# ─────────────────────────────────────────────────────────────────────────────
class WavePropagation(ctk.CTkFrame):
    """Emanation coverage & interception map: where can a device's emanation be
    received?  Shows the field-strength coverage (dBµV/m), the interception
    boundary (field = detectability floor), optional two-ray ground multipath and
    a placed eavesdropper's received field.  A wavefront animation is kept as a
    secondary near/far-field view."""

    PATTERNS = ["Isotropic", "Dipole"]
    MODELS = ["Free-space", "Two-ray (ground)"]
    VIEWS = ["Coverage map", "Wavefront (animated)"]

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._anim_running = False
        self._frame = 0
        self._after_id = None
        self._cbar = None
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Emanation Coverage & Interception Map",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(self, text="How far — and where — can this device's emanation be received? "
                     "Field-strength coverage, interception boundary & multipath",
                     font=ctk.CTkFont(size=12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=24, pady=(0, 16))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        body.columnconfigure(1, weight=1); body.rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(body, fg_color=BG_PANEL, corner_radius=12,
                                      border_width=1, border_color=BORDER, width=270)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        def sec(t):
            ctk.CTkLabel(left, text=t, font=ctk.CTkFont(size=14, weight="bold"),
                         text_color=TEXT_PRI).pack(anchor="w", padx=14, pady=(14, 2))

        def lbl(t):
            ctk.CTkLabel(left, text=t, font=ctk.CTkFont(size=12),
                         text_color=TEXT_SEC).pack(anchor="w", padx=14, pady=(8, 0))

        def entry(var_name, default):
            v = ctk.StringVar(value=default); setattr(self, var_name, v)
            ctk.CTkEntry(left, textvariable=v, fg_color=BG_CARD, border_color=BORDER
                         ).pack(fill="x", padx=14, pady=(2, 0))

        sec("Source")
        lbl("Device")
        self.dev_var = ctk.StringVar(value="HDMI Cable")
        ctk.CTkOptionMenu(left, values=list(tp.DEVICES.keys()), variable=self.dev_var,
                          fg_color=BG_CARD, button_color=ACCENT, command=self._pick_device
                          ).pack(fill="x", padx=14, pady=(2, 0))
        lbl("Emission @1 m (dBµV/m)")
        entry("emis_var", f"{tp.DEVICES['HDMI Cable']['emission_dbuv']:.0f}")
        lbl("Frequency (MHz)")
        entry("freq_var", f"{max(tp.DEVICES['HDMI Cable']['peaks']) / 1e6:.4g}")
        lbl("Radiation pattern")
        self.pattern_var = ctk.StringVar(value="Isotropic")
        ctk.CTkOptionMenu(left, values=self.PATTERNS, variable=self.pattern_var,
                          fg_color=BG_CARD, button_color=ACCENT, command=lambda _: self._redraw()
                          ).pack(fill="x", padx=14, pady=(2, 0))

        sec("Propagation")
        lbl("Model")
        self.model_var = ctk.StringVar(value="Free-space")
        ctk.CTkOptionMenu(left, values=self.MODELS, variable=self.model_var,
                          fg_color=BG_CARD, button_color=ACCENT, command=lambda _: self._redraw()
                          ).pack(fill="x", padx=14, pady=(2, 0))
        hrow = ctk.CTkFrame(left, fg_color="transparent"); hrow.pack(fill="x", padx=14, pady=(6, 0))
        self.htx_var = ctk.StringVar(value="1.5"); self.hrx_var = ctk.StringVar(value="1.5")
        self.ant_var = ctk.StringVar(value="0.3")
        for lab, var in [("h_tx", self.htx_var), ("h_rx", self.hrx_var), ("D m", self.ant_var)]:
            ctk.CTkLabel(hrow, text=lab, font=ctk.CTkFont(size=10), text_color=TEXT_SEC).pack(side="left")
            ctk.CTkEntry(hrow, textvariable=var, width=44, fg_color=BG_CARD,
                         border_color=BORDER).pack(side="left", padx=(2, 6))

        sec("Interception")
        lbl("Detectability floor (dBµV/m)")
        entry("floor_var", "20")
        lbl("Eavesdropper X, Y (m)")
        erow = ctk.CTkFrame(left, fg_color="transparent"); erow.pack(fill="x", padx=14, pady=(2, 0))
        self.eav_x = ctk.StringVar(value=""); self.eav_y = ctk.StringVar(value="")
        for var, ph in [(self.eav_x, "X"), (self.eav_y, "Y")]:
            ctk.CTkEntry(erow, textvariable=var, width=104, fg_color=BG_CARD,
                         border_color=BORDER, placeholder_text=ph).pack(side="left", padx=(0, 6))
        lbl("Map range (m)")
        entry("range_var", "30")

        sec("View")
        self.view_var = ctk.StringVar(value="Coverage map")
        ctk.CTkOptionMenu(left, values=self.VIEWS, variable=self.view_var,
                          fg_color=BG_CARD, button_color=ACCENT, command=self._on_view
                          ).pack(fill="x", padx=14, pady=(2, 0))
        wrow = ctk.CTkFrame(left, fg_color="transparent"); wrow.pack(fill="x", padx=14, pady=(6, 0))
        self.play_btn = ctk.CTkButton(wrow, text="▶ Play", width=90, fg_color=ACCENT2,
                                      text_color="#ffffff", hover_color="#26744a",
                                      font=ctk.CTkFont(weight="bold"), command=self._toggle_anim)
        self.play_btn.pack(side="left", padx=(0, 4))
        ctk.CTkButton(wrow, text="🔄", width=40, fg_color=BG_CARD, hover_color=BORDER,
                      command=self._reset).pack(side="left")
        self.speed_var = ctk.IntVar(value=3)
        ctk.CTkSlider(left, from_=1, to=10, variable=self.speed_var).pack(fill="x", padx=14, pady=(6, 0))
        ctk.CTkButton(left, text="🔍 Update", fg_color=ACCENT, hover_color=ACCENT_HV,
                      command=self._redraw).pack(fill="x", padx=14, pady=(8, 2))
        ctk.CTkButton(left, text="🖼 Save PNG", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._save_png).pack(fill="x", padx=14, pady=(2, 10))

        right = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=12,
                             border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="nsew")
        self.fig, (self.ax_map, self.ax_profile) = plt.subplots(
            2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1.5, 1]},
            layout="constrained", **PLOT_PARAMS)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._redraw()

    # ── input helpers ────────────────────────────────────────────────────────
    def _f(self, var, default):
        try: return float(var.get())
        except Exception: return default

    def _emission(self): return self._f(self.emis_var, 44.0)
    def _freq(self): return max(1e3, self._f(self.freq_var, 100.0) * 1e6)
    def _floor(self): return self._f(self.floor_var, 20.0)
    def _range(self): return max(2.0, self._f(self.range_var, 30.0))
    def _antenna(self): return max(1e-3, self._f(self.ant_var, 0.3))
    def _pattern(self): return "dipole" if self.pattern_var.get() == "Dipole" else "isotropic"
    def _two_ray(self): return self.model_var.get().startswith("Two-ray")

    def _heights(self):
        return max(0.01, self._f(self.htx_var, 1.5)), max(0.01, self._f(self.hrx_var, 1.5))

    def _eaves(self):
        try: return float(self.eav_x.get()), float(self.eav_y.get())
        except Exception: return None

    def _radial_field(self):
        """Radial field profile (peak direction) [dBµV/m] vs distance, model-aware."""
        E0 = self._emission(); f = self._freq(); floor = self._floor()
        tr = self._two_ray(); htx, hrx = self._heights()
        r_int0 = tp.interception_range_m(E0, floor)
        r = np.linspace(0.5, max(self._range() * 1.5, r_int0 * 1.3, 20.0), 600)
        field = E0 - tp.free_space_field_decay_db(np.maximum(r, tp.REF_DISTANCE), tp.REF_DISTANCE)
        if tr:
            field = field + tp.two_ray_factor_db(r, f, htx, hrx)
        return r, field

    def _interception_range(self):
        """Furthest distance at which the field still exceeds the floor (model-aware)."""
        r, field = self._radial_field()
        above = np.where(field >= self._floor())[0]
        return float(r[above[-1]]) if above.size else 0.0

    def _pick_device(self, name):
        d = tp.DEVICES[name]
        self.emis_var.set(f"{d['emission_dbuv']:.0f}")
        self.freq_var.set(f"{max(d['peaks']) / 1e6:.4g}")
        self._redraw()

    def _on_view(self, _):
        if self.view_var.get() != "Wavefront (animated)":
            self._stop_anim()
        self._redraw()

    # ── drawing ──────────────────────────────────────────────────────────────
    def _style(self, ax):
        ax.clear(); ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_SEC, labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor(BORDER)

    def _clear_cbar(self):
        if self._cbar is not None:
            try: self._cbar.remove()
            except Exception: pass
            self._cbar = None

    def _redraw(self):
        self._clear_cbar()
        self._draw_profile()
        if self.view_var.get() == "Coverage map":
            self._draw_coverage()
        else:
            self._draw_wavefront()
        self.canvas.draw()

    def _draw_coverage(self):
        self._style(self.ax_map)
        R = self._range(); E0 = self._emission(); f = self._freq()
        floor = self._floor(); pat = self._pattern(); tr = self._two_ray()
        htx, hrx = self._heights()
        grid = np.linspace(-R, R, 320)
        X, Y = np.meshgrid(grid, grid)
        Z = tp.coverage_field_dbuv(E0, X, Y, pat, tr, f, htx, hrx)
        im = self.ax_map.imshow(Z, extent=[-R, R, -R, R], origin="lower", cmap="turbo",
                                vmin=floor - 15, vmax=E0 + 3, aspect="equal", interpolation="bilinear")
        self._cbar = self.fig.colorbar(im, ax=self.ax_map, fraction=0.046, pad=0.02)
        self._cbar.set_label("Field strength (dBµV/m)", color=TEXT_SEC, fontsize=9)
        self._cbar.ax.tick_params(colors=TEXT_SEC, labelsize=8)
        try:
            self.ax_map.contour(X, Y, Z, levels=[floor], colors=["white"],
                                linewidths=1.6, linestyles="--")
        except Exception:
            pass
        for r_b, col in [(float(tp.near_far_boundary(f)), WARN),
                         (float(tp.fraunhofer_distance(self._antenna(), f)), ACCENT2)]:
            if 0 < r_b < R:
                self.ax_map.add_patch(plt.Circle((0, 0), r_b, color=col, fill=False,
                                                 lw=1.1, linestyle=":"))
        self.ax_map.plot(0, 0, "*", color="white", ms=15, zorder=6)
        eav = self._eaves()
        if eav is not None:
            ef = float(tp.coverage_field_dbuv(E0, eav[0], eav[1], pat, tr, f, htx, hrx))
            det = ef >= floor
            self.ax_map.plot(eav[0], eav[1], "P", color=("#f43f5e" if det else ACCENT2),
                             ms=13, mew=1.5, zorder=7)
            self.ax_map.annotate(f"{ef:.0f} dBµV/m\n{'DETECTABLE' if det else 'below floor'}",
                                 eav, textcoords="offset points", xytext=(8, 6),
                                 color=TEXT_PRI, fontsize=8, zorder=7,
                                 bbox=dict(boxstyle="round,pad=0.2", fc=BG_CARD, ec=BORDER, alpha=0.9))
        r_int = self._interception_range()
        self.ax_map.set_xlim(-R, R); self.ax_map.set_ylim(-R, R)
        self.ax_map.set_xlabel("X (m)", color=TEXT_SEC, fontsize=10)
        self.ax_map.set_ylabel("Y (m)", color=TEXT_SEC, fontsize=10)
        self.ax_map.set_title(f"Coverage  |  {self.dev_var.get()}  |  interception range ≈ {r_int:.0f} m "
                              f"(white = floor {floor:.0f} dBµV/m)", color=TEXT_PRI, fontsize=10)

    def _draw_profile(self):
        self._style(self.ax_profile)
        floor = self._floor(); tr = self._two_ray()
        r, field = self._radial_field()
        self.ax_profile.plot(r, field, color=ACCENT, lw=1.6,
                             label=("two-ray" if tr else "free-space") + " (peak dir.)")
        self.ax_profile.axhline(floor, color=WARN, lw=1.0, linestyle="--",
                                label=f"detectability floor {floor:.0f} dBµV/m")
        above = np.where(field >= floor)[0]
        if above.size:
            r_int = r[above[-1]]
            self.ax_profile.axvline(r_int, color="#f43f5e", lw=1.2)
            self.ax_profile.annotate(f"interception ≈ {r_int:.0f} m", (r_int, floor),
                                     textcoords="offset points", xytext=(-95, 14),
                                     color="#f43f5e", fontsize=8)
        eav = self._eaves()
        if eav is not None:
            self.ax_profile.axvline(float(np.hypot(*eav)), color="#22d3ee", lw=0.8, linestyle=":")
        self.ax_profile.set_xlabel("Distance (m)", color=TEXT_SEC, fontsize=10)
        self.ax_profile.set_ylabel("Field (dBµV/m)", color=TEXT_SEC, fontsize=10)
        self.ax_profile.set_title("Radial field vs distance", color=TEXT_PRI, fontsize=11)
        self.ax_profile.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_PRI)
        self.ax_profile.grid(True, alpha=0.15, color=BORDER)

    def _intensity_map(self, X, Y, t, lam, pattern):
        R = np.hypot(X, Y) + 1e-3
        wave = np.cos(2 * np.pi * (R / lam - t)) / R
        if pattern == "dipole":
            wave = wave * (np.sin(np.arctan2(Y, X)) ** 2 + 0.05)
        return wave

    def _draw_wavefront(self):
        self._style(self.ax_map)
        R = self._range(); f = self._freq(); lam = tp.C0 / f
        t = self._frame * 0.04 * self.speed_var.get()
        grid = np.linspace(-R, R, 300)
        X, Y = np.meshgrid(grid, grid)
        Z = self._intensity_map(X, Y, t, lam, self._pattern())
        vmax = float(np.percentile(np.abs(Z), 97)) or 1.0
        self.ax_map.imshow(Z, extent=[-R, R, -R, R], origin="lower", cmap="plasma",
                           vmin=-vmax, vmax=vmax, aspect="equal", interpolation="bilinear")
        for r_b, col, lab in [(float(tp.near_far_boundary(f)), WARN, "reactive λ/2π"),
                              (float(tp.fraunhofer_distance(self._antenna(), f)), ACCENT2, "far-field 2D²/λ")]:
            if 0 < r_b < R:
                self.ax_map.add_patch(plt.Circle((0, 0), r_b, color=col, fill=False,
                                                 lw=1.2, linestyle="--", label=f"{lab}={r_b:.2f} m"))
        self.ax_map.plot(0, 0, "w*", ms=14, zorder=5, label="Source")
        self.ax_map.set_xlim(-R, R); self.ax_map.set_ylim(-R, R)
        self.ax_map.set_xlabel("X (m)", color=TEXT_SEC, fontsize=10)
        self.ax_map.set_ylabel("Y (m)", color=TEXT_SEC, fontsize=10)
        self.ax_map.set_title(f"Wavefronts  |  f={f/1e6:.3g} MHz, λ={lam:.3g} m",
                              color=TEXT_PRI, fontsize=11)
        self.ax_map.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER,
                           labelcolor=TEXT_PRI, loc="upper right")

    def _animate(self):
        if not self._anim_running: return
        self._frame += 1
        self._draw_wavefront()
        self.canvas.draw()
        self._after_id = self.after(max(30, 120 - self.speed_var.get() * 10), self._animate)

    def _stop_anim(self):
        self._anim_running = False
        if self._after_id:
            self.after_cancel(self._after_id); self._after_id = None
        self.play_btn.configure(text="▶ Play", fg_color=ACCENT2, text_color="#ffffff")

    def _toggle_anim(self):
        if self.view_var.get() != "Wavefront (animated)":
            self.view_var.set("Wavefront (animated)")
            self._clear_cbar(); self._draw_profile()
        if self._anim_running:
            self._stop_anim()
        else:
            self._anim_running = True
            self.play_btn.configure(text="⏸ Pause", fg_color=WARN, text_color=BG_DARK)
            self._animate()

    def _reset(self):
        self._stop_anim()
        self._frame = 0
        self._redraw()

    def _save_png(self):
        path = _fd.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")],
                                     title="Save coverage map")
        if not path: return
        try:
            rep.save_figure(self.fig, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 6 – SHIELDING DESIGN OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class MaterialAdvisor(ctk.CTkFrame):
    """Find the lightest / cheapest shield that meets a requirement across a band.

    The requirement can be a flat SE target, OR — the accurate mode — derived
    from a real source and a real limit: SE_required(f) = source_field(f) −
    limit(f).  All output values are physical: SE margin (dB), thickness (mm),
    mass (kg) and a real currency cost (mass × representative price)."""

    MATERIALS = ShieldingCalculator.MATERIALS   # single source of truth

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._results: list[dict] = []
        self._measured_src = None          # (freqs_Hz, field_dBµV/m)
        self._measured_src_name = ""
        self._my_layers: list[tuple] = []  # user-built stack [(material, thickness_mm)]
        self._my_design = None             # evaluated manual design
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Shielding Design Optimizer",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(self, text="Lightest / cheapest shield that brings a source under a limit — "
                     "real SE margin, mass (kg) and cost (USD)",
                     font=ctk.CTkFont(size=12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=24, pady=(0, 16))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(body, fg_color=BG_PANEL, corner_radius=12,
                                      border_width=1, border_color=BORDER, width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        def sec(text):
            ctk.CTkLabel(left, text=text, font=ctk.CTkFont(size=14, weight="bold"),
                         text_color=TEXT_PRI).pack(anchor="w", padx=16, pady=(14, 2))

        def lbl(text):
            ctk.CTkLabel(left, text=text, font=ctk.CTkFont(size=12),
                         text_color=TEXT_SEC).pack(anchor="w", padx=16, pady=(8, 0))

        def entry(var_name, default):
            v = ctk.StringVar(value=default); setattr(self, var_name, v)
            ctk.CTkEntry(left, textvariable=v, fg_color=BG_CARD,
                         border_color=BORDER).pack(fill="x", padx=16, pady=(2, 0))

        sec("Requirement")
        self.req_mode = ctk.StringVar(value="Source → limit")
        ctk.CTkOptionMenu(left, values=["Source → limit", "Flat SE target"],
                          variable=self.req_mode, fg_color=BG_CARD,
                          button_color=ACCENT).pack(fill="x", padx=16, pady=(2, 0))

        lbl("Source")
        self.src_mode = ctk.StringVar(value="Device (model)")
        ctk.CTkOptionMenu(left, values=["Device (model)", "Measured trace"],
                          variable=self.src_mode, fg_color=BG_CARD,
                          button_color=ACCENT).pack(fill="x", padx=16, pady=(2, 0))
        self.dev_var = ctk.StringVar(value="HDMI Cable")
        ctk.CTkOptionMenu(left, values=list(tp.DEVICES.keys()), variable=self.dev_var,
                          fg_color=BG_CARD, button_color=ACCENT).pack(fill="x", padx=16, pady=(4, 0))
        trow = ctk.CTkFrame(left, fg_color="transparent"); trow.pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkButton(trow, text="📈 Load trace", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._load_src).pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.src_unit = ctk.StringVar(value="MHz")
        ctk.CTkOptionMenu(trow, values=["Hz", "kHz", "MHz", "GHz"], width=70,
                          variable=self.src_unit, fg_color=BG_CARD, button_color=ACCENT).pack(side="left")
        self.src_lbl = ctk.CTkLabel(left, text="(device model — illustrative)",
                                    font=ctk.CTkFont(size=10), text_color=TEXT_SEC)
        self.src_lbl.pack(anchor="w", padx=16, pady=(2, 0))

        lbl("Limit")
        self.limit_var = ctk.StringVar(value="Detectability floor (20 dBµV/m)")
        ctk.CTkOptionMenu(left, values=["Detectability floor (20 dBµV/m)"] + tc.standard_names(),
                          variable=self.limit_var, fg_color=BG_CARD,
                          button_color=ACCENT).pack(fill="x", padx=16, pady=(2, 0))
        lbl("Observation distance (m)")
        entry("dist_var", "1.0")

        lbl("Flat mode only — band Hz & target dB")
        frow = ctk.CTkFrame(left, fg_color="transparent"); frow.pack(fill="x", padx=16, pady=(2, 0))
        self.band_lo = ctk.StringVar(value="1e6"); self.band_hi = ctk.StringVar(value="1e9")
        self.req_se = ctk.StringVar(value="60")
        for var, ph in [(self.band_lo, "f_lo"), (self.band_hi, "f_hi"), (self.req_se, "SE dB")]:
            ctk.CTkEntry(frow, textvariable=var, width=76, fg_color=BG_CARD,
                         border_color=BORDER, placeholder_text=ph).pack(side="left", padx=(0, 4))

        sec("Constraints")
        lbl("Field type")
        self.field_var = ctk.StringVar(value="Plane Wave")
        ctk.CTkOptionMenu(left, values=list(tp.FIELD_TYPES), variable=self.field_var,
                          fg_color=BG_CARD, button_color=ACCENT).pack(fill="x", padx=16, pady=(2, 0))
        crow = ctk.CTkFrame(left, fg_color="transparent"); crow.pack(fill="x", padx=16, pady=(6, 0))
        self.max_thick = ctk.StringVar(value="5"); self.area_var = ctk.StringVar(value="1.0")
        ctk.CTkLabel(crow, text="max t mm", font=ctk.CTkFont(size=11), text_color=TEXT_SEC).pack(side="left")
        ctk.CTkEntry(crow, textvariable=self.max_thick, width=56, fg_color=BG_CARD,
                     border_color=BORDER).pack(side="left", padx=(2, 8))
        ctk.CTkLabel(crow, text="area m²", font=ctk.CTkFont(size=11), text_color=TEXT_SEC).pack(side="left")
        ctk.CTkEntry(crow, textvariable=self.area_var, width=56, fg_color=BG_CARD,
                     border_color=BORDER).pack(side="left", padx=(2, 0))
        self.lam_on = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(left, text="🔬 Include 2-layer laminates", variable=self.lam_on
                      ).pack(anchor="w", padx=16, pady=(8, 0))
        self.mu_roll = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(left, text="Frequency-dependent µ", variable=self.mu_roll
                      ).pack(anchor="w", padx=16, pady=(4, 0))
        lbl("Rank by")
        self.priority_var = ctk.StringVar(value="Lowest mass")
        ctk.CTkOptionMenu(left, values=["Lowest mass", "Lowest cost", "Minimum thickness", "Highest margin"],
                          variable=self.priority_var, fg_color=BG_CARD,
                          button_color=ACCENT).pack(fill="x", padx=16, pady=(2, 0))

        ctk.CTkButton(left, text="⚙  Optimize", height=42, fg_color=ACCENT, hover_color=ACCENT_HV,
                      font=ctk.CTkFont(size=13, weight="bold"), command=self._solve
                      ).pack(fill="x", padx=16, pady=(14, 4))
        exp = ctk.CTkFrame(left, fg_color="transparent"); exp.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkButton(exp, text="💾 CSV", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._export_csv).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(exp, text="📄 PDF", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._export_pdf).pack(side="left", expand=True, fill="x", padx=2)
        ctk.CTkButton(exp, text="🖼 PNG", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._save_png).pack(side="left", expand=True, fill="x", padx=(2, 0))

        sec("Evaluate a specific design")
        mrow = ctk.CTkFrame(left, fg_color="transparent"); mrow.pack(fill="x", padx=16, pady=(2, 0))
        self.my_mat = ctk.StringVar(value="Copper")
        ctk.CTkOptionMenu(mrow, values=list(self.MATERIALS.keys()), variable=self.my_mat, width=150,
                          fg_color=BG_CARD, button_color=ACCENT).pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.my_thick = ctk.StringVar(value="1.0")
        ctk.CTkEntry(mrow, textvariable=self.my_thick, width=70, fg_color=BG_CARD,
                     border_color=BORDER, placeholder_text="t mm").pack(side="left")
        arow = ctk.CTkFrame(left, fg_color="transparent"); arow.pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkButton(arow, text="➕ Add layer", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._add_my_layer).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(arow, text="✖ Clear", width=60, fg_color=BG_CARD, hover_color=BORDER,
                      command=self._clear_my_layers).pack(side="left", padx=(2, 0))
        self.my_stack_lbl = ctk.CTkLabel(left, text="Stack: (single layer)", font=ctk.CTkFont(size=10),
                                         text_color=TEXT_SEC, wraplength=250, justify="left")
        self.my_stack_lbl.pack(anchor="w", padx=16, pady=(4, 0))
        ctk.CTkButton(left, text="🔍 Evaluate my design", fg_color="#f59e0b", text_color="#ffffff",
                      hover_color="#d97706", font=ctk.CTkFont(weight="bold"),
                      command=self._evaluate_design).pack(fill="x", padx=16, pady=(6, 2))
        self.my_verdict = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=12, weight="bold"),
                                       text_color=TEXT_SEC, wraplength=250, justify="left")
        self.my_verdict.pack(anchor="w", padx=16, pady=(2, 8))

        self.res_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.res_frame.pack(fill="x", padx=8, pady=(0, 12))

        right = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=12,
                             border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="nsew")
        self.fig, (self.ax_top, self.ax_bot) = plt.subplots(
            2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [1.3, 1]}, **PLOT_PARAMS)
        self.fig.tight_layout(pad=3.2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._placeholder()

    def _placeholder(self):
        for ax in (self.ax_top, self.ax_bot):
            ax.clear(); ax.set_facecolor(BG_PANEL)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)
        self.ax_top.text(0.5, 0.5, "Configure the requirement and click Optimize",
                         ha="center", va="center", color=TEXT_SEC, fontsize=12,
                         transform=self.ax_top.transAxes)
        self.fig.tight_layout(pad=3.2)
        self.canvas.draw()

    # ── input helpers ───────────────────────────────────────────────────────
    def _distance(self):
        try: return max(1e-3, float(self.dist_var.get()))
        except Exception: return 1.0

    def _mu_for(self, freqs, m):
        if self.mu_roll.get() and m.get("f_mu"):
            return tp.effective_mu_r(freqs, m["mu_r"], m["f_mu"])
        return m["mu_r"]

    @staticmethod
    def _desc(c):
        return " + ".join(f"{m} {t:.3g}mm" for m, t in c["layers"])

    def _load_src(self):
        path = _fd.askopenfilename(filetypes=[("CSV trace", "*.csv"), ("All", "*.*")],
                                   title="Load measured emission trace (freq, dBµV/m)")
        if not path: return
        try:
            f, v = tm.load_trace_csv(path, 0, 1, freq_unit=self.src_unit.get())
            self._measured_src = (f, v); self._measured_src_name = os.path.basename(path)
            self.src_mode.set("Measured trace")
            self.src_lbl.configure(text=f"trace: {self._measured_src_name} ({len(f)} pts)",
                                   text_color=ACCENT2)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _band(self):
        if self.req_mode.get() == "Flat SE target":
            return float(self.band_lo.get()), float(self.band_hi.get())
        if self.src_mode.get() == "Measured trace" and self._measured_src is not None:
            f, _ = self._measured_src
            return float(np.min(f)), float(np.max(f))
        peaks = tp.DEVICES[self.dev_var.get()]["peaks"]
        return max(1e3, min(peaks) / 2.0), max(peaks) * 1.5

    def _source_field(self, freqs):
        """Source field strength [dBµV/m] on ``freqs`` at the observation distance."""
        if self.src_mode.get() == "Measured trace" and self._measured_src is not None:
            mf, mv = self._measured_src
        else:
            mf, mv = tp.emanation_spectrum(self.dev_var.get(), self._distance())
        return np.interp(np.log10(freqs), np.log10(np.maximum(mf, 1.0)), mv,
                         left=mv[0], right=mv[-1])

    def _limit_field(self, freqs):
        """Limit field strength [dBµV/m] on ``freqs`` (NaN out of a standard's scope)."""
        lim = self.limit_var.get()
        if lim.startswith("Detectability"):
            return np.full(np.shape(freqs), tp.DETECTION_FLOOR_DBUV)
        return tc.mask_levels(freqs, lim, at_distance_m=self._distance())

    def _requirement(self):
        """Return the requirement passed to best_laminates: a scalar (flat mode)
        or a callable req(freqs)->array (source→limit)."""
        if self.req_mode.get() == "Flat SE target":
            return float(self.req_se.get())
        return lambda freqs: tp.required_se_curve(
            freqs, self._source_field(freqs), self._limit_field(freqs))

    def _stack_se(self, freqs, layers):
        stack = [(self.MATERIALS[nm]["sigma"], self._mu_for(freqs, self.MATERIALS[nm]), t / 1000.0)
                 for nm, t in layers]
        return tp.multilayer_se_db(freqs, stack, r=self._distance(), field=self.field_var.get())

    # ── solve ───────────────────────────────────────────────────────────────
    def _solve(self):
        try:
            f_lo, f_hi = self._band()
            t_max = float(self.max_thick.get()); area = max(1e-6, float(self.area_var.get()))
        except Exception as e:
            messagebox.showerror("Input Error", str(e)); return
        try:
            cands = tp.best_laminates(f_lo, f_hi, self._requirement(), r=self._distance(),
                                      field=self.field_var.get(), t_max_each_mm=t_max,
                                      area_m2=area, use_mu_rolloff=self.mu_roll.get(),
                                      include_laminates=self.lam_on.get())
        except Exception as e:
            messagebox.showerror("Optimization failed", str(e)); return
        if not cands:
            messagebox.showwarning("No Solution",
                "No design meets the requirement within the thickness constraint.\n"
                "Increase max thickness, relax the limit, or widen the distance.")
            return
        key = {"Lowest mass": lambda c: c["mass_kg"],
               "Lowest cost": lambda c: c["cost_usd"],
               "Minimum thickness": lambda c: c["thickness_mm"],
               "Highest margin": lambda c: -c["margin"]}
        cands.sort(key=key.get(self.priority_var.get(), lambda c: c["mass_kg"]))
        self._results = cands
        self._update_results_panel(cands)
        self._redraw()

    def _update_results_panel(self, results):
        for w in self.res_frame.winfo_children(): w.destroy()
        ctk.CTkLabel(self.res_frame, text="🏆  Recommended designs",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=TEXT_PRI).pack(anchor="w", padx=8, pady=(8, 4))
        MEDAL = ["🥇", "🥈", "🥉"]
        for i, c in enumerate(results[:6]):
            medal = MEDAL[i] if i < 3 else f"  {i+1}."
            color = [ACCENT2, "#d29922", WARN][i] if i < 3 else TEXT_SEC
            card = ctk.CTkFrame(self.res_frame, fg_color=BG_CARD, corner_radius=8)
            card.pack(fill="x", pady=3, padx=4)
            tag = "  [laminate]" if len(c["layers"]) == 2 else ""
            ctk.CTkLabel(card, text=f"{medal}  {self._desc(c)}{tag}",
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=color).pack(anchor="w", padx=10, pady=(6, 0))
            ctk.CTkLabel(card,
                         text=f"margin +{c['margin']:.0f} dB   |   {c['thickness_mm']:.3g} mm   |   "
                              f"{c['mass_kg']:.3g} kg   |   ${c['cost_usd']:,.2f}",
                         font=ctk.CTkFont(size=11), text_color=TEXT_SEC
                         ).pack(anchor="w", padx=10, pady=(0, 6))

    # ── manual design evaluation ────────────────────────────────────────────
    def _area(self):
        try: return max(1e-6, float(self.area_var.get()))
        except Exception: return 1.0

    def _mass(self, layers):
        return tp.shield_mass_kg([(self.MATERIALS[nm]["density"], t / 1000.0)
                                  for nm, t in layers], self._area())

    def _real_cost(self, layers):
        a = self._area()
        return float(sum(self.MATERIALS[nm]["price_per_kg"] * self.MATERIALS[nm]["density"]
                         * (t / 1000.0) * a for nm, t in layers))

    def _add_my_layer(self):
        try: t = float(self.my_thick.get())
        except Exception:
            messagebox.showerror("Input Error", "Enter a valid thickness (mm)."); return
        self._my_layers.append((self.my_mat.get(), t))
        self.my_stack_lbl.configure(text="Stack: " + " + ".join(
            f"{m} {th:.3g}mm" for m, th in self._my_layers))

    def _clear_my_layers(self):
        self._my_layers = []; self._my_design = None
        self.my_stack_lbl.configure(text="Stack: (single layer)")
        self.my_verdict.configure(text="")
        self._redraw()

    def _evaluate_design(self):
        layers = list(self._my_layers)
        if not layers:
            try: layers = [(self.my_mat.get(), float(self.my_thick.get()))]
            except Exception:
                messagebox.showerror("Input Error", "Enter a valid thickness (mm)."); return
        try:
            f_lo, f_hi = self._band()
        except Exception as e:
            messagebox.showerror("Input Error", str(e)); return
        freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), 300)
        req = self._requirement()
        req_arr = np.asarray(req(freqs) if callable(req) else np.full(freqs.shape, req), float)
        se = self._stack_se(freqs, layers)
        margin = float(np.min(se - req_arr))
        self._my_design = {"layers": layers, "se": float(np.min(se)), "margin": margin,
                           "mass_kg": self._mass(layers), "cost_usd": self._real_cost(layers),
                           "thickness_mm": sum(t for _, t in layers)}
        md = self._my_design
        if margin >= 0:
            self.my_verdict.configure(
                text=f"✅ MEETS requirement  (margin +{margin:.0f} dB)   |   "
                     f"{md['mass_kg']:.3g} kg   |   ${md['cost_usd']:,.2f}", text_color=ACCENT2)
        else:
            self.my_verdict.configure(
                text=f"❌ SHORTFALL {margin:.0f} dB (SE below required)   |   "
                     f"{md['mass_kg']:.3g} kg   |   ${md['cost_usd']:,.2f}", text_color=WARN)
        self._redraw()

    def _redraw(self):
        for ax in (self.ax_top, self.ax_bot):
            ax.clear(); ax.set_facecolor(BG_PANEL)
            ax.tick_params(colors=TEXT_SEC, labelsize=9)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)
        try:
            f_lo, f_hi = self._band()
        except Exception:
            self.fig.tight_layout(pad=3.2); self.canvas.draw(); return
        freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), 300)
        req = self._requirement()
        req_arr = np.asarray(req(freqs) if callable(req) else np.full(freqs.shape, req), float)

        # TOP — required vs achieved SE
        self.ax_top.semilogx(freqs, req_arr, color="#f43f5e", lw=1.8, linestyle="--",
                             label="Required SE(f)")
        tops = [float(np.nanmax(req_arr)) if req_arr.size else 60.0]
        title_bits = []
        if self._results:
            rec = self._results[0]; rec_se = self._stack_se(freqs, rec["layers"])
            tops.append(float(np.nanmax(rec_se)))
            self.ax_top.semilogx(freqs, rec_se, color=ACCENT2, lw=2.0,
                                 label=f"Optimum: {self._desc(rec)}")
            for c in self._results[1:4]:
                self.ax_top.semilogx(freqs, self._stack_se(freqs, c["layers"]),
                                     color=ACCENT, lw=0.8, alpha=0.35)
            if not self._my_design:
                self.ax_top.fill_between(freqs, req_arr, rec_se, where=(rec_se >= req_arr),
                                         color=ACCENT2, alpha=0.15)
                title_bits.append(f"optimum {float(np.min(rec_se - req_arr)):+.0f} dB")
        if self._my_design:
            my_se = self._stack_se(freqs, self._my_design["layers"]); tops.append(float(np.nanmax(my_se)))
            self.ax_top.semilogx(freqs, my_se, color="#f59e0b", lw=2.4,
                                 label=f"My design: {self._desc(self._my_design)}")
            self.ax_top.fill_between(freqs, req_arr, my_se, where=(my_se >= req_arr),
                                     color="#f59e0b", alpha=0.15)
            self.ax_top.fill_between(freqs, req_arr, my_se, where=(my_se < req_arr),
                                     color=WARN, alpha=0.30)
            title_bits.append(f"my design {float(np.min(my_se - req_arr)):+.0f} dB")
        self.ax_top.set_ylim(0, max(60.0, max(tops) * 1.1))
        self.ax_top.set_xlabel("Frequency (Hz)", color=TEXT_SEC, fontsize=10)
        self.ax_top.set_ylabel("Shielding effectiveness (dB)", color=TEXT_SEC, fontsize=10)
        self.ax_top.set_title("Achieved vs Required SE" + (f"   ({' · '.join(title_bits)})" if title_bits else ""),
                              color=TEXT_PRI, fontsize=11)
        self.ax_top.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_PRI)
        self.ax_top.grid(True, which="both", alpha=0.15, color=BORDER)

        # BOTTOM — mass–cost trade-off
        if self._results:
            masses = [c["mass_kg"] for c in self._results]
            costs = [max(1e-3, c["cost_usd"]) for c in self._results]
            self.ax_bot.scatter(masses, costs, s=26, color=ACCENT, edgecolor=BORDER,
                                linewidth=0.5, alpha=0.75, zorder=3)
            rec = self._results[0]
            self.ax_bot.scatter([rec["mass_kg"]], [max(1e-3, rec["cost_usd"])], marker="*",
                                s=300, color=ACCENT2, edgecolor="white", linewidth=1.1, zorder=5)
        if self._my_design:
            md = self._my_design
            self.ax_bot.scatter([md["mass_kg"]], [max(1e-3, md["cost_usd"])], marker="D",
                                s=90, color="#f59e0b", edgecolor="white", linewidth=1.0, zorder=6)
            self.ax_bot.annotate("my design", (md["mass_kg"], max(1e-3, md["cost_usd"])),
                                textcoords="offset points", xytext=(8, 6), color="#f59e0b", fontsize=8)
        if self._results or self._my_design:
            self.ax_bot.set_xscale("log"); self.ax_bot.set_yscale("log")
        self.ax_bot.set_xlabel("Shield mass (kg)", color=TEXT_SEC, fontsize=10)
        self.ax_bot.set_ylabel("Cost (USD, material)", color=TEXT_SEC, fontsize=10)
        self.ax_bot.set_title("Mass-Cost Trade-off  (star = optimum, diamond = your design)",
                              color=TEXT_PRI, fontsize=11)
        self.ax_bot.grid(True, which="both", alpha=0.15, color=BORDER)

        self.fig.tight_layout(pad=3.2)
        self.canvas.draw()

    # ── export ──────────────────────────────────────────────────────────────
    def _rows(self):
        return [[self._desc(c), f'{c["thickness_mm"]:.4f}', f'{c["se"]:.2f}',
                 f'{c["margin"]:.2f}', f'{c["mass_kg"]:.4f}', f'{c["cost_usd"]:.2f}']
                for c in self._results]

    def _export_csv(self):
        if not self._results:
            messagebox.showinfo("Nothing to export", "Run Optimize first."); return
        path = _fd.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                     title="Export ranked designs")
        if not path: return
        try:
            rep.export_csv(path, ["design", "thickness_mm", "min_SE_dB", "margin_dB",
                                  "mass_kg", "cost_usd"], self._rows())
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _export_pdf(self):
        if not self._results:
            messagebox.showinfo("Nothing to export", "Run Optimize first."); return
        path = _fd.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
                                     title="Export shielding design report")
        if not path: return
        try:
            png = os.path.join(tempfile.gettempdir(), "_shield_design.png")
            rep.save_figure(self.fig, png)
            rec = self._results[0]
            rep.build_pdf_report(
                path, "Shielding Design Report",
                subtitle="MSc Telecommunications Engineering — TEMPEST Analysis Suite",
                meta={"Requirement": self.req_mode.get(),
                      "Source": (self._measured_src_name if self.src_mode.get() == "Measured trace"
                                 and self._measured_src else self.dev_var.get()),
                      "Limit": self.limit_var.get(), "Field / distance":
                      f"{self.field_var.get()} @ {self._distance():g} m",
                      "Recommended": self._desc(rec),
                      "Worst SE margin": f"+{rec['margin']:.0f} dB",
                      "Mass / cost": f"{rec['mass_kg']:.3g} kg / ${rec['cost_usd']:,.2f}"},
                tables=[("Ranked Designs",
                         ["Design", "t (mm)", "min SE (dB)", "margin (dB)", "mass (kg)", "cost (USD)"],
                         [[r[0], r[1], r[2], r[3], r[4], f"${float(r[5]):,.2f}"] for r in self._rows()[:10]])],
                image_path=png,
                notes=["Material prices are representative raw-material estimates (fabrication "
                       "excluded). Device-model source spectra are illustrative; load a measured "
                       "trace for calibrated results. See tempest_physics.py."])
            messagebox.showinfo("Export complete", f"PDF saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _save_png(self):
        path = _fd.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")],
                                     title="Save plot")
        if not path: return
        try:
            rep.save_figure(self.fig, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 7 – LIVE SDR CAPTURE
# ─────────────────────────────────────────────────────────────────────────────
class LiveSDRCapture(ctk.CTkFrame):
    """Real-time spectrum + waterfall from an SDR — live hardware (RTL-SDR, …)
    or the simulated backend, OR an **imported recording** (SigMF / raw IQ /
    WAV, from this device or another SDR) played back through the exact same
    pipeline. Capture runs on a worker thread; the UI is refreshed from a
    queue via ``after`` so Tk is only ever touched on the main thread."""

    WATERFALL_ROWS = 120
    BLOCK = 4096

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._sdr = None
        self._thread = None
        self._running = False
        self._active_mode = None       # "live" | "file" — which button/status to reset on stop
        self._queue = queue.Queue(maxsize=8)
        self._wf = None
        self._freqs = None
        self._after_id = None
        self._last_raw = None          # (freqs, relative-dB mag) for ref calibration
        self._loaded_capture = None    # dict from tempest_capture_io.load_capture()
        self._file_rows = []           # widgets in the folder-scan list, for cleanup
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Live SDR Capture",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(self, text="Real-time spectrum & waterfall from a software-defined radio "
                     "(simulated backend if no hardware is connected)",
                     font=ctk.CTkFont(size=12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=24, pady=(0, 16))

        ctrl = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10,
                            border_width=1, border_color=BORDER)
        ctrl.pack(fill="x", padx=24, pady=(0, 10))

        avail = sdrlib.available_backends()
        def _blabel(k):
            return k + ("" if avail.get(k) else "  (n/a)")
        ctk.CTkLabel(ctrl, text="Backend:", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(14, 4), pady=12)
        self.backend_var = ctk.StringVar(value="simulated")
        ctk.CTkOptionMenu(ctrl, values=[_blabel(k) for k in avail], width=150,
                          variable=self.backend_var, fg_color=BG_CARD, button_color=ACCENT
                          ).grid(row=0, column=1, padx=4)

        def _num(col, label, var_name, default, width=70):
            ctk.CTkLabel(ctrl, text=label, text_color=TEXT_SEC,
                         font=ctk.CTkFont(size=12)).grid(row=0, column=col, padx=(12, 2))
            v = ctk.StringVar(value=default)
            setattr(self, var_name, v)
            ctk.CTkEntry(ctrl, textvariable=v, width=width, fg_color=BG_CARD,
                         border_color=BORDER).grid(row=0, column=col + 1, padx=(0, 2))
        _num(2, "Center (MHz)", "fc_var", "100")
        _num(4, "Rate (MHz)", "fs_var", "2.048")
        _num(6, "Gain (dB)", "gain_var", "20", width=56)

        ctk.CTkLabel(ctrl, text="Tie device:", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=12)).grid(row=0, column=8, padx=(12, 2))
        self.tie_var = ctk.StringVar(value="(none)")
        ctk.CTkOptionMenu(ctrl, values=["(none)"] + list(tp.DEVICES.keys()), width=140,
                          variable=self.tie_var, fg_color=BG_CARD, button_color=ACCENT
                          ).grid(row=0, column=9, padx=4)

        self.start_btn = ctk.CTkButton(ctrl, text="▶  Start", width=100, fg_color=ACCENT2,
                                       text_color="#ffffff", hover_color="#26744a",
                                       font=ctk.CTkFont(weight="bold"), command=self._toggle)
        self.start_btn.grid(row=0, column=10, padx=(12, 4), pady=12)
        ctk.CTkButton(ctrl, text="🖼 PNG", width=64, fg_color=BG_CARD,
                      hover_color=BORDER, command=self._save_png).grid(row=0, column=11, padx=(0, 12))

        # Calibration row
        cal = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10,
                           border_width=1, border_color=BORDER)
        cal.pack(fill="x", padx=24, pady=(0, 10))
        self.cal_on = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(cal, text="Calibrate → dBµV/m", variable=self.cal_on,
                      font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=14, pady=8)
        def _cnum(col, label, var_name, default):
            ctk.CTkLabel(cal, text=label, text_color=TEXT_SEC,
                         font=ctk.CTkFont(size=11)).grid(row=0, column=col, padx=(12, 2))
            v = ctk.StringVar(value=default)
            setattr(self, var_name, v)
            ctk.CTkEntry(cal, textvariable=v, width=64, fg_color=BG_CARD,
                         border_color=BORDER).grid(row=0, column=col + 1, padx=(0, 2))
        _cnum(1, "Ref offset (dB→dBµV)", "ref_var", "107")
        _cnum(3, "AF", "af_var", "0")
        _cnum(5, "Cable", "cable_var", "0")
        _cnum(7, "Gain", "lna_var", "0")
        ctk.CTkButton(cal, text="🎯 Calibrate ref…", width=132, fg_color=ACCENT,
                      hover_color=ACCENT_HV, command=self._calibrate_reference
                      ).grid(row=0, column=9, padx=(12, 4))
        self.status = ctk.CTkLabel(cal, text="Idle", text_color=TEXT_SEC,
                                   font=ctk.CTkFont(size=11))
        self.status.grid(row=0, column=10, padx=16)

        # ── File Playback ("upload a capture") ──────────────────────────────
        fp = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10,
                          border_width=1, border_color=BORDER)
        fp.pack(fill="x", padx=24, pady=(0, 10))
        ctk.CTkLabel(fp, text="📂  Imported capture:", text_color=TEXT_PRI,
                     font=ctk.CTkFont(size=12, weight="bold")
                     ).grid(row=0, column=0, padx=(14, 10), pady=(10, 2), sticky="w")
        ctk.CTkButton(fp, text="Load File…", width=110, fg_color=BG_CARD, hover_color=BORDER,
                      text_color=TEXT_PRI, command=self._load_file_dialog
                      ).grid(row=0, column=1, padx=4, pady=(10, 2))
        ctk.CTkButton(fp, text="Load Folder…", width=110, fg_color=BG_CARD, hover_color=BORDER,
                      text_color=TEXT_PRI, command=self._load_folder_dialog
                      ).grid(row=0, column=2, padx=4, pady=(10, 2))
        self.file_loop = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(fp, text="Loop", variable=self.file_loop,
                      font=ctk.CTkFont(size=11)).grid(row=0, column=3, padx=(10, 4), pady=(10, 2))
        self.file_play_btn = ctk.CTkButton(fp, text="▶ Play file", width=100, fg_color=ACCENT2,
                                           text_color="#ffffff", hover_color="#26744a",
                                           font=ctk.CTkFont(weight="bold"), state="disabled",
                                           command=self._toggle_file)
        self.file_play_btn.grid(row=0, column=4, padx=(4, 14), pady=(10, 2))

        self.file_info_lbl = ctk.CTkLabel(fp, text="No capture loaded — load a file or a folder "
                                          "of recordings (SigMF, raw IQ, or WAV).",
                                          text_color=TEXT_SEC, font=ctk.CTkFont(size=11),
                                          anchor="w", justify="left")
        self.file_info_lbl.grid(row=1, column=0, columnspan=5, padx=14, pady=(0, 8), sticky="w")

        self.file_list = ctk.CTkScrollableFrame(fp, fg_color="transparent", height=110)
        self.file_list.grid(row=2, column=0, columnspan=5, padx=10, pady=(0, 10), sticky="ew")
        fp.columnconfigure(4, weight=1)
        self.file_list.grid_remove()      # shown only once a folder is scanned

        self.fig, (self.ax_spec, self.ax_wf) = plt.subplots(
            2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [1, 1.5]}, **PLOT_PARAMS)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self._draw_idle()

    def _draw_idle(self):
        for ax in (self.ax_spec, self.ax_wf):
            ax.clear(); ax.set_facecolor(BG_PANEL)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)
        self.ax_spec.text(0.5, 0.5, "Press ▶ Start to begin capture",
                          ha="center", va="center", color=TEXT_SEC, fontsize=13,
                          transform=self.ax_spec.transAxes)
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _params(self):
        def g(v, d):
            try: return float(v.get())
            except Exception: return d
        return (g(self.fc_var, 100) * 1e6, g(self.fs_var, 2.048) * 1e6,
                g(self.gain_var, 20))

    def _toggle(self):
        if self._running and self._active_mode == "live":
            self.stop()
        else:
            self.stop()                # stop file playback first if that was active
            self._start()

    def _start(self):
        fc, fs, gain = self._params()
        backend = self.backend_var.get().split()[0]        # strip "(n/a)"
        device = None if self.tie_var.get() == "(none)" else self.tie_var.get()
        try:
            src = sdrlib.open_sdr(backend, sample_rate=fs, center_freq=fc,
                                  gain=gain, device=device)
        except Exception as e:
            messagebox.showerror("SDR open failed", str(e)); return
        self._begin_capture(src, "live", f"Capturing @ {fc/1e6:.3f} MHz  |  {backend}")

    def _toggle_file(self):
        if self._running and self._active_mode == "file":
            self.stop(); return
        if not self._loaded_capture:
            messagebox.showinfo("No capture loaded", "Load a file or folder first."); return
        self.stop()                    # stop live capture first if that was active
        info = self._loaded_capture
        src = sdrlib.FileSDRSource(info["iq"], info["fs"], center_freq=info["fc"] or 0.0,
                                   loop=self.file_loop.get()).open()
        status = (f"Playing {os.path.basename(info['path'])}  |  fs={info['fs']/1e6:.3f} MHz")
        if info["fc"]:
            status += f"  |  fc={info['fc']/1e6:.3f} MHz"
        status += f"  |  {info['duration_s']:.1f}s"
        status += "  |  looping" if self.file_loop.get() else "  |  single pass"
        self._begin_capture(src, "file", status)

    def _begin_capture(self, src, mode, status_text):
        """Shared start-up for both live hardware and file-playback sources —
        both are just an :class:`SDRSource`, so the worker/poll/plot pipeline
        below never needs to know which one it's driving."""
        self._sdr = src
        self._active_mode = mode
        self._wf = None; self._freqs = None
        self._running = True
        btn = self.start_btn if mode == "live" else self.file_play_btn
        btn.configure(text="⏸  Stop", fg_color=WARN, text_color=BG_DARK)
        self.status.configure(text=status_text, text_color=ACCENT2)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._after_id = self.after(60, self._poll)

    def stop(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id); self._after_id = None
        if self._thread:
            self._thread.join(timeout=1.0); self._thread = None
        if self._sdr:
            try: self._sdr.close()
            except Exception: pass
            self._sdr = None
        self._active_mode = None
        self.start_btn.configure(text="▶  Start", fg_color=ACCENT2, text_color="#ffffff")
        self.file_play_btn.configure(text="▶ Play file", fg_color=ACCENT2, text_color="#ffffff")
        self.status.configure(text="Idle", text_color=TEXT_SEC)

    def _worker(self):
        while self._running and self._sdr is not None:
            try:
                iq = self._sdr.read(self.BLOCK)
                f, mag, _ = sdrlib.capture_spectrum(
                    iq, self._sdr.sample_rate, self._sdr.center_freq,
                    nfft=self.BLOCK, window="Hann")
            except Exception:
                break
            if self._queue.full():
                try: self._queue.get_nowait()
                except queue.Empty: pass
            try: self._queue.put_nowait((f, mag))
            except queue.Full: pass
            if isinstance(self._sdr, sdrlib.FileSDRSource):
                # File reads are near-instant numpy slicing (unlike hardware
                # I/O, which paces itself) — sleep to the recording's own time
                # base so the waterfall stays meaningful and the thread doesn't
                # spin the CPU.
                time.sleep(min(0.2, self.BLOCK / self._sdr.sample_rate))

    def _apply_cal(self, mag):
        if not self.cal_on.get():
            return mag, "Relative power (dB)"
        def g(v, d):
            try: return float(v.get())
            except Exception: return d
        E = tm.field_strength_dbuv_per_m(mag + g(self.ref_var, 107),
                                         g(self.af_var, 0), g(self.cable_var, 0),
                                         g(self.lna_var, 0))
        return E, "Field strength (dBµV/m)"

    def _poll(self):
        if not self._running: return
        if self._sdr is not None and getattr(self._sdr, "at_end", False):
            self.stop(); return        # single-pass file playback reached the end
        latest = None
        while not self._queue.empty():
            try: latest = self._queue.get_nowait()
            except queue.Empty: break
        if latest is not None:
            self._update_plots(*latest)
        self._after_id = self.after(60, self._poll)

    def _update_plots(self, f, mag):
        self._last_raw = (f, mag)          # keep uncalibrated data for ref-cal
        vals, ylabel = self._apply_cal(mag)
        if self._wf is None or self._freqs is None or len(self._freqs) != len(f):
            self._freqs = f
            self._wf = np.full((self.WATERFALL_ROWS, len(f)), float(vals.min()))
        self._wf = np.roll(self._wf, -1, axis=0)
        self._wf[-1] = vals

        self.ax_spec.clear(); self.ax_spec.set_facecolor(BG_PANEL)
        self.ax_spec.tick_params(colors=TEXT_SEC, labelsize=9)
        for s in self.ax_spec.spines.values(): s.set_edgecolor(BORDER)
        self.ax_spec.plot(f / 1e6, vals, color=ACCENT2, lw=0.8)
        self.ax_spec.set_ylabel(ylabel, color=TEXT_SEC, fontsize=9)
        self.ax_spec.set_title("Live Spectrum", color=TEXT_PRI, fontsize=11)
        self.ax_spec.grid(True, alpha=0.15, color=BORDER)

        self.ax_wf.clear(); self.ax_wf.set_facecolor(BG_PANEL)
        self.ax_wf.tick_params(colors=TEXT_SEC, labelsize=9)
        self.ax_wf.imshow(self._wf, aspect="auto", origin="lower", cmap="plasma",
                          extent=[f[0] / 1e6, f[-1] / 1e6, 0, self.WATERFALL_ROWS])
        self.ax_wf.set_xlabel("Frequency (MHz)", color=TEXT_SEC, fontsize=9)
        self.ax_wf.set_ylabel("Time (frames)", color=TEXT_SEC, fontsize=9)
        self.ax_wf.set_title("Waterfall", color=TEXT_PRI, fontsize=11)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _save_png(self):
        path = _fd.asksaveasfilename(defaultextension=".png",
                                     filetypes=[("PNG image", "*.png")],
                                     title="Save SDR capture")
        if not path: return
        try:
            rep.save_figure(self.fig, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _calibrate_reference(self):
        """Reference-level calibration: with a known CW signal injected, capture
        its peak and solve for the offset that turns relative dB into dBµV."""
        if self._last_raw is None:
            messagebox.showinfo(
                "No signal",
                "Start capture with your known reference signal applied, then "
                "run the reference calibration."); return
        dlg = ctk.CTkInputDialog(
            title="Reference-level calibration",
            text="Known injected level at the SDR input (dBm):")
        raw = dlg.get_input()
        if raw is None:                    # user cancelled
            return
        try:
            known_dbm = float(raw)
        except (TypeError, ValueError):
            messagebox.showerror("Invalid input", "Enter a number in dBm."); return
        f, mag = self._last_raw
        peak_rel = float(np.max(mag))
        known_dbuv = float(tm.dbm_to_dbuv(known_dbm))
        offset = tm.reference_offset_db(peak_rel, known_dbuv)
        self.ref_var.set(f"{offset:.1f}")
        self.cal_on.set(True)
        messagebox.showinfo(
            "Reference calibrated",
            f"Measured peak {peak_rel:.1f} dB (relative)  ↔  {known_dbm:g} dBm "
            f"= {known_dbuv:.1f} dBµV.\n\nReference offset set to {offset:.1f} dB. "
            f"The field-strength axis now applies AF / cable / gain on top of this.")
        self._update_plots(f, mag)

    # ── imported-capture loading ("upload a file / folder") ─────────────────
    def _load_file_dialog(self):
        filetypes = [
            ("SigMF", "*.sigmf-meta *.sigmf-data"),
            ("WAV", "*.wav"),
            ("Raw IQ", " ".join(f"*{e}" for e in cio.RAW_EXTENSIONS)),
            ("All files", "*.*"),
        ]
        path = _fd.askopenfilename(title="Load a captured SDR recording",
                                   filetypes=filetypes)
        if not path: return
        self._load_path(path)

    def _load_folder_dialog(self):
        folder = _fd.askdirectory(title="Select a folder of captured recordings")
        if not folder: return
        try:
            entries = cio.scan_folder(folder)
        except Exception as e:
            messagebox.showerror("Scan failed", str(e)); return
        self._populate_file_list(entries)

    def _populate_file_list(self, entries):
        for w in self._file_rows: w.destroy()
        self._file_rows = []
        if not entries:
            self.file_list.grid_remove()
            messagebox.showinfo("No captures found",
                "No recognised capture files (SigMF / raw IQ / WAV) were found "
                "in that folder.")
            return
        self.file_list.grid()
        for e in entries:
            row = ctk.CTkFrame(self.file_list, fg_color=BG_CARD, corner_radius=6)
            row.pack(fill="x", pady=2, padx=2)
            detail = e["format"].upper()
            if e.get("fs"):
                detail += f"  ·  {e['fs']/1e6:.3f} MHz"
            if e.get("fc"):
                detail += f"  @ {e['fc']/1e6:.3f} MHz"
            detail += f"  ·  {e['size_bytes']/1024:,.0f} KB"
            if e["needs_format"]:
                detail += "  ·  needs format"
            ctk.CTkButton(row, text=f"{e['rel_path']}   —   {detail}", anchor="w",
                         fg_color="transparent", hover_color=BORDER, text_color=TEXT_PRI,
                         font=ctk.CTkFont(size=11),
                         command=lambda p=e["path"]: self._load_path(p)
                         ).pack(fill="x", padx=6, pady=4)
            self._file_rows.append(row)

    def _load_path(self, path):
        fmt = cio.detect_format(path)
        try:
            if fmt == "raw":
                picked = self._prompt_raw_format(path)
                if picked is None: return          # user cancelled the dialog
                dtype_key, fs_hint, fc_hint = picked
                info = cio.load_capture(path, dtype_key=dtype_key,
                                        fs_hint=fs_hint, fc_hint=fc_hint)
            else:
                info = cio.load_capture(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e)); return

        self._loaded_capture = info
        self.file_play_btn.configure(state="normal")
        fc_txt = f"{info['fc']/1e6:.3f} MHz" if info["fc"] else "not set"
        trunc = (f"  ⚠ truncated to first {info['n_samples']:,} of "
                 f"{info['n_samples_total']:,} samples" if info["truncated"] else "")
        self.file_info_lbl.configure(
            text=(f"✅ {os.path.basename(info['path'])}  ·  {info['format'].upper()}"
                 f"  ·  fs={info['fs']/1e6:.3f} MHz  ·  fc={fc_txt}"
                 f"  ·  {info['duration_s']:.2f}s  ·  {info['n_samples']:,} samples{trunc}"),
            text_color=ACCENT2)

    def _prompt_raw_format(self, path):
        """Small modal collecting sample format / rate / centre-frequency for a
        headerless raw IQ file (it carries no self-describing metadata, unlike
        SigMF). Returns ``(dtype_key, fs_hz, fc_hz)``, or ``None`` if cancelled.
        """
        top = ctk.CTkToplevel(self)
        top.title("Raw IQ format")
        top.geometry("400x300")
        top.transient(self.winfo_toplevel())
        top.grab_set()
        result = {}

        ctk.CTkLabel(top, text=os.path.basename(path),
                     font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=16, pady=(16, 4))
        ctk.CTkLabel(top, text="This file has no metadata — specify how it was "
                     "recorded (as with rtl_sdr, hackrf_transfer, or a GNU Radio "
                     "file sink):", text_color=TEXT_SEC, font=ctk.CTkFont(size=11),
                     wraplength=360, justify="left").pack(anchor="w", padx=16, pady=(0, 10))

        ctk.CTkLabel(top, text="Sample format", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16)
        labels = [spec["label"] for spec in cio.RAW_DTYPES.values()]
        label_to_key = {spec["label"]: key for key, spec in cio.RAW_DTYPES.items()}
        fmt_var = ctk.StringVar(value=cio.RAW_DTYPES[cio.DEFAULT_RAW_DTYPE]["label"])
        ctk.CTkOptionMenu(top, values=labels, variable=fmt_var, fg_color=BG_CARD,
                          button_color=ACCENT).pack(fill="x", padx=16, pady=(2, 10))

        ctk.CTkLabel(top, text="Sample rate (MHz)", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16)
        fs_var = ctk.StringVar(value=self.fs_var.get())
        ctk.CTkEntry(top, textvariable=fs_var, fg_color=BG_CARD, border_color=BORDER
                    ).pack(fill="x", padx=16, pady=(2, 10))

        ctk.CTkLabel(top, text="Centre frequency (MHz) — optional", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16)
        fc_var = ctk.StringVar(value=self.fc_var.get())
        ctk.CTkEntry(top, textvariable=fc_var, fg_color=BG_CARD, border_color=BORDER
                    ).pack(fill="x", padx=16, pady=(2, 4))

        def _ok():
            try:
                fs_hz = float(fs_var.get()) * 1e6
                fc_hz = float(fc_var.get()) * 1e6 if fc_var.get().strip() else None
                if fs_hz <= 0: raise ValueError("sample rate must be positive")
            except ValueError as e:
                messagebox.showerror("Invalid input",
                                     f"Sample rate / frequency must be numbers ({e})."); return
            result["value"] = (label_to_key[fmt_var.get()], fs_hz, fc_hz)
            top.destroy()

        btns = ctk.CTkFrame(top, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=(14, 16))
        ctk.CTkButton(btns, text="Cancel", fg_color=BG_CARD, hover_color=BORDER,
                     text_color=TEXT_PRI, command=top.destroy
                     ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(btns, text="Load", fg_color=ACCENT, hover_color=ACCENT_HV,
                     command=_ok).pack(side="left", expand=True, fill="x", padx=(4, 0))

        top.wait_window()
        return result.get("value")


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 8 – VAN ECK VIDEO RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
class VideoReconstruction(ctk.CTkFrame):
    """Demonstrate reconstructing an eavesdropped screen image from a display's
    video emanation — synthesise the leaked waveform from a source image, then
    fold it back into a picture by recovering the line/frame timing."""

    SOURCES = {"Text": "text", "Bars": "bars", "Checker": "checker", "Gradient": "gradient"}

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._signal = None
        self._orig = None
        self._true_h = self._true_v = 0
        self._override_img = None
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Van Eck Video Reconstruction",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(self, text="Reconstruct an eavesdropped screen image from a display's video "
                     "emanation — recover the line/frame timing to lock the picture (van Eck, Kuhn)",
                     font=ctk.CTkFont(size=12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=24, pady=(0, 16))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(body, fg_color=BG_PANEL, corner_radius=12,
                                      border_width=1, border_color=BORDER, width=290)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        def sec(t):
            ctk.CTkLabel(left, text=t, font=ctk.CTkFont(size=14, weight="bold"),
                         text_color=TEXT_PRI).pack(anchor="w", padx=16, pady=(16, 4))

        def entry(label, var_name, default):
            ctk.CTkLabel(left, text=label, font=ctk.CTkFont(size=12),
                         text_color=TEXT_SEC).pack(anchor="w", padx=16, pady=(6, 0))
            v = ctk.StringVar(value=default)
            setattr(self, var_name, v)
            ctk.CTkEntry(left, textvariable=v, fg_color=BG_CARD,
                         border_color=BORDER).pack(fill="x", padx=16, pady=(2, 0))

        sec("Source (victim screen)")
        ctk.CTkLabel(left, text="Content", font=ctk.CTkFont(size=12),
                     text_color=TEXT_SEC).pack(anchor="w", padx=16, pady=(6, 0))
        self.src_var = ctk.StringVar(value="Text")
        ctk.CTkOptionMenu(left, values=list(self.SOURCES.keys()), variable=self.src_var,
                          fg_color=BG_CARD, button_color=ACCENT,
                          command=lambda _: (setattr(self, "_override_img", None), self._synthesize())
                          ).pack(fill="x", padx=16, pady=(2, 0))
        entry("Text (for Text source)", "text_var", "TOP\nSECRET")
        entry("Active width (px)", "w_var", "160")
        entry("Active height (px)", "h_var", "120")
        entry("H-blank (px)", "hblank_var", "24")
        entry("V-blank (lines)", "vblank_var", "12")

        ctk.CTkLabel(left, text="Capture noise", font=ctk.CTkFont(size=12),
                     text_color=TEXT_SEC).pack(anchor="w", padx=16, pady=(8, 0))
        self.noise_var = ctk.DoubleVar(value=0.05)
        ctk.CTkSlider(left, from_=0.0, to=0.4, variable=self.noise_var,
                      command=lambda _: self._synthesize()).pack(fill="x", padx=16, pady=(2, 0))

        ctk.CTkButton(left, text="🛰  Synthesize emanation", height=40, fg_color=ACCENT,
                      hover_color=ACCENT_HV, font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._synthesize).pack(fill="x", padx=16, pady=(14, 4))
        ctk.CTkButton(left, text="📂  Load image as screen", fg_color=BG_CARD,
                      hover_color=BORDER, command=self._load_image).pack(fill="x", padx=16, pady=2)

        sec("Receiver tuning (the attack)")
        self.htot_var = ctk.IntVar(value=184)
        self.vtot_var = ctk.IntVar(value=132)
        self.hoff_var = ctk.IntVar(value=0)
        self.voff_var = ctk.IntVar(value=0)
        self._sliders = {}
        for label, var, vn in [("Samples / line (h_total)", self.htot_var, "htot"),
                               ("Lines / frame (v_total)", self.vtot_var, "vtot"),
                               ("H-hold offset", self.hoff_var, "hoff"),
                               ("V-hold offset", self.voff_var, "voff")]:
            lbl = ctk.CTkLabel(left, text=label, font=ctk.CTkFont(size=12), text_color=TEXT_SEC)
            lbl.pack(anchor="w", padx=16, pady=(8, 0))
            self._sliders[vn + "_lbl"] = lbl
            s = ctk.CTkSlider(left, from_=0, to=300, variable=var,
                              command=lambda _v: self._reconstruct_draw())
            s.pack(fill="x", padx=16, pady=(2, 0))
            self._sliders[vn] = s

        self.avg_lbl = ctk.CTkLabel(left, text="Frame averaging = 3", font=ctk.CTkFont(size=12),
                                    text_color=TEXT_SEC)
        self.avg_lbl.pack(anchor="w", padx=16, pady=(8, 0))
        self.avg_var = ctk.IntVar(value=3)
        ctk.CTkSlider(left, from_=1, to=3, number_of_steps=2, variable=self.avg_var,
                      command=lambda _v: self._reconstruct_draw()).pack(fill="x", padx=16, pady=(2, 0))

        ctk.CTkButton(left, text="🔒  Auto-lock timing", fg_color=ACCENT2,
                      text_color="#ffffff", hover_color="#26744a",
                      font=ctk.CTkFont(weight="bold"), command=self._auto_lock
                      ).pack(fill="x", padx=16, pady=(14, 4))
        ctk.CTkButton(left, text="🖼  Save PNG", fg_color=BG_CARD, hover_color=BORDER,
                      command=self._save_png).pack(fill="x", padx=16, pady=(2, 12))

        right = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=12,
                             border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="nsew")
        self.fig, self.axd = plt.subplot_mosaic(
            [["orig", "recon"], ["sig", "recon"]],
            figsize=(10, 6.5), gridspec_kw={"width_ratios": [1, 1.6],
                                            "height_ratios": [1.2, 1]}, **PLOT_PARAMS)
        self.fig.tight_layout(pad=2.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._synthesize()

    # ── forward model ───────────────────────────────────────────────────────
    def _dims(self):
        def g(v, d):
            try: return max(8, int(float(v.get())))
            except Exception: return d
        return g(self.w_var, 160), g(self.h_var, 120)

    def _current_image(self):
        if self._override_img is not None:
            return self._override_img
        W, H = self._dims()
        kind = self.SOURCES[self.src_var.get()]
        return tv.demo_image(kind, W, H, text=self.text_var.get())

    def _synthesize(self):
        img = self._current_image()
        try:
            hblank = int(float(self.hblank_var.get())); vblank = int(float(self.vblank_var.get()))
        except Exception:
            hblank, vblank = 24, 12
        self._orig = img
        frame_sig, h_total, v_total = tv.raster_scan(img, hblank, vblank)
        multi = np.tile(frame_sig, 3)                     # a few frames, as a real capture
        self._signal = tv.emanate(multi, noise=float(self.noise_var.get()), seed=0)
        self._true_h, self._true_v = h_total, v_total

        # configure the tuning sliders around the (unknown to attacker) true values
        self._config_slider("htot", max(8, h_total - 40), h_total + 40, h_total)
        self._config_slider("vtot", max(8, v_total - 30), v_total + 30, v_total)
        self._config_slider("hoff", 0, h_total, 0)
        self._config_slider("voff", 0, v_total, 0)
        self._reconstruct_draw()

    def _config_slider(self, key, lo, hi, val):
        s = self._sliders[key]
        s.configure(from_=lo, to=hi, number_of_steps=max(1, int(hi - lo)))
        {"htot": self.htot_var, "vtot": self.vtot_var,
         "hoff": self.hoff_var, "voff": self.voff_var}[key].set(int(val))

    def _load_image(self):
        path = _fd.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All", "*.*")],
            title="Load an image to use as the victim screen")
        if not path: return
        try:
            W, H = self._dims()
            self._override_img = tv.load_image(path, W, H)
            self._synthesize()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    # ── reconstruction ──────────────────────────────────────────────────────
    def _reconstruct_draw(self):
        if self._signal is None: return
        h = int(self.htot_var.get()); v = int(self.vtot_var.get())
        ho = int(self.hoff_var.get()); vo = int(self.voff_var.get())
        nf = int(self.avg_var.get())
        self._sliders["htot_lbl"].configure(text=f"Samples / line (h_total) = {h}")
        self._sliders["vtot_lbl"].configure(text=f"Lines / frame (v_total) = {v}")
        self.avg_lbl.configure(text=f"Frame averaging = {nf}")
        rec = tv.reconstruct(self._signal, h, v, ho, vo, n_frames=nf)
        locked = (h == self._true_h and v == self._true_v)

        for name in ("orig", "recon", "sig"):
            ax = self.axd[name]; ax.clear(); ax.set_facecolor(BG_PANEL)
            ax.tick_params(colors=TEXT_SEC, labelsize=8)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)

        self.axd["orig"].imshow(self._orig, cmap="gray", aspect="auto")
        self.axd["orig"].set_title("Victim screen (ground truth)", color=TEXT_PRI, fontsize=10)
        self.axd["orig"].set_xticks([]); self.axd["orig"].set_yticks([])

        vmin, vmax = np.percentile(rec, [2, 98])
        self.axd["recon"].imshow(rec, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        self.axd["recon"].set_title(
            "Reconstruction  " + ("🔒 LOCKED" if locked else "— tune h_total/offsets to lock"),
            color=(ACCENT2 if locked else WARN), fontsize=11)
        self.axd["recon"].set_xticks([]); self.axd["recon"].set_yticks([])

        snippet = self._signal[:min(self._signal.size, 3 * self._true_h)]
        self.axd["sig"].plot(snippet, color=ACCENT, lw=0.6)
        self.axd["sig"].set_title("Captured emanation waveform (first 3 lines)",
                                  color=TEXT_PRI, fontsize=9)
        self.axd["sig"].set_xlabel("Sample", color=TEXT_SEC, fontsize=8)
        self.axd["sig"].grid(True, alpha=0.15, color=BORDER)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def _auto_lock(self):
        if self._signal is None: return
        h = tv.estimate_line_length(self._signal, min_len=16,
                                    max_len=min(self._signal.size // 4, self._true_h * 3))
        v = tv.estimate_frame_length(self._signal, h, min_lines=16,
                                     max_lines=self._true_v * 3)
        # widen slider ranges if the estimate falls outside the current window
        self._config_slider("htot", min(h, self.htot_var.get()) - 5,
                            max(h, self.htot_var.get()) + 5, h)
        self._config_slider("vtot", min(v, self.vtot_var.get()) - 5,
                            max(v, self.vtot_var.get()) + 5, v)
        self.hoff_var.set(0); self.voff_var.set(0)
        self._reconstruct_draw()
        messagebox.showinfo("Auto-lock",
                            f"Recovered timing by autocorrelation:\n"
                            f"  samples/line = {h}  (true {self._true_h})\n"
                            f"  lines/frame  = {v}  (true {self._true_v})")

    def _save_png(self):
        path = _fd.asksaveasfilename(defaultextension=".png",
                                     filetypes=[("PNG image", "*.png")],
                                     title="Save reconstruction")
        if not path: return
        try:
            rep.save_figure(self.fig, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD  (Home)
# ─────────────────────────────────────────────────────────────────────────────
class Dashboard(ctk.CTkFrame):
    CARDS = [
        ("📡", "EM Emanation Simulator",    "Visualize EM leakage from computing devices across frequency spectrum",         ACCENT),
        ("📊", "Signal Capture & Analysis", "FFT spectrum analysis with window functions & spectrogram",                     ACCENT2),
        ("🛡️", "Shielding Effectiveness",   "Compute SE (dB) for materials using absorption & reflection loss formulas",    "#d29922"),
        ("🏢", "Room Assessment Tool",      "Interactive room layout with NATO SDIP-27 emanation zone mapping",              "#9c59d1"),
        ("🌊", "Coverage & Interception",    "Field-strength coverage map, interception boundary, multipath & eavesdropper",  "#0ea5e9"),
        ("🧱", "Shielding Design Optimizer", "Lightest / cheapest shield to bring a source under a limit — real SE margin, kg & USD", "#f43f5e"),
        ("📻", "Live SDR Capture",          "Real-time spectrum & waterfall from an SDR (simulated backend if no hardware)", "#22c55e"),
        ("🖥️", "Van Eck Reconstruction",    "Reconstruct an eavesdropped screen image from a display's video emanation",     "#e879f9"),
    ]

    def __init__(self, master, switch_cb, **kw):
        super().__init__(master, fg_color=BG_DARK, **kw)
        self._switch = switch_cb
        self._build()

    def _build(self):
        hero = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=16,
                            border_width=1, border_color=BORDER)
        hero.pack(fill="x", padx=28, pady=(28, 20))

        ctk.CTkLabel(hero, text="TEMPEST Analysis Suite",
                     font=ctk.CTkFont(size=28, weight="bold"), text_color=TEXT_PRI
                     ).pack(anchor="w", padx=28, pady=(24, 4))
        ctk.CTkLabel(hero,
                     text="Comprehensive tool for device and room emanation analysis.\n"
                          "MSc Telecommunications Engineering  ·  NATO SDIP-27  ·  IEEE 299  ·  MIL-STD-285  ·  Andi Lika",
                     font=ctk.CTkFont(size=13), text_color=TEXT_SEC, justify="left"
                     ).pack(anchor="w", padx=28, pady=(0, 24))

        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=28, pady=(0, 28))
        grid.columnconfigure((0, 1, 2), weight=1)

        modules = ["emanation", "signal", "shielding", "room", "wave", "material", "sdr", "video"]
        for i, (icon, title, desc, _color) in enumerate(self.CARDS):
            card = ctk.CTkFrame(grid, fg_color=BG_PANEL, corner_radius=12,
                                border_width=1, border_color=BORDER)
            card.grid(row=i // 3, column=i % 3, sticky="nsew", padx=10, pady=10)
            card.columnconfigure(0, weight=1)

            ctk.CTkLabel(card, text=f"{icon}  {title}",
                         font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT_PRI
                         ).pack(anchor="w", padx=20, pady=(18, 4))
            ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(size=12),
                         text_color=TEXT_SEC, wraplength=360, justify="left"
                         ).pack(anchor="w", padx=20, pady=(0, 14))

            mod = modules[i]
            ctk.CTkButton(card, text="Open  →", width=90, height=30, anchor="w",
                          fg_color="transparent", hover_color=BG_CARD, text_color=ACCENT,
                          font=ctk.CTkFont(size=12, weight="bold"),
                          command=lambda m=mod: self._switch(m)
                          ).pack(anchor="w", padx=16, pady=(0, 16))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN  APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class TempestApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TEMPEST Analysis Suite  —  MSc Telecommunications Engineering")
        self.geometry("1280x780")
        self.minsize(1100, 680)
        self.configure(fg_color=BG_DARK)
        self._build_layout()

    def _build_layout(self):
        # ── Sidebar (black) ───────────────────────────────────────────────────
        self.sidebar = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, corner_radius=0,
                                    border_width=0, width=224)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="🔐  TEMPEST Suite",
                     font=ctk.CTkFont(size=15, weight="bold"), text_color="#ffffff"
                     ).pack(anchor="w", padx=18, pady=(20, 0))
        ctk.CTkLabel(self.sidebar, text="E M S E C   A N A L Y S I S",
                     font=ctk.CTkFont(size=9), text_color="#7a7a82"
                     ).pack(anchor="w", padx=18, pady=(1, 14))

        nav_groups = [
            (None, [("Home", "🏠", "home")]),
            ("Analysis", [("EM Emanation", "📡", "emanation"),
                          ("Signal Analysis", "📊", "signal"),
                          ("Coverage Map", "🌊", "wave")]),
            ("Design", [("Shielding Calc.", "🛡️", "shielding"),
                        ("Room Assessment", "🏢", "room"),
                        ("Shield Optimizer", "🧱", "material")]),
            ("Capture", [("Live SDR", "📻", "sdr"),
                         ("Van Eck Recon", "🖥️", "video")]),
        ]
        self._nav_btns = {}
        for group, items in nav_groups:
            if group:
                ctk.CTkLabel(self.sidebar, text=group.upper(),
                             font=ctk.CTkFont(size=10, weight="bold"), text_color="#5c5c64"
                             ).pack(anchor="w", padx=20, pady=(14, 2))
            for label, icon, key in items:
                btn = SidebarButton(self.sidebar, label, icon,
                                    command=lambda k=key: self._switch(k))
                btn.pack(fill="x", padx=10, pady=1)
                self._nav_btns[key] = btn

        ctk.CTkLabel(self.sidebar, text="v1.0 · MSc Thesis", font=ctk.CTkFont(size=10),
                     text_color=SIDEBAR_TEXT).pack(side="bottom", padx=18, pady=16)

        # ── Content area ─────────────────────────────────────────────────────
        self.content = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        self.content.pack(side="left", fill="both", expand=True)

        self._frames = {
            "home":      Dashboard(self.content, self._switch),
            "emanation": EmanationSimulator(self.content),
            "signal":    SignalAnalysis(self.content),
            "shielding": ShieldingCalculator(self.content),
            "room":      RoomAssessment(self.content),
            "wave":      WavePropagation(self.content),
            "material":  MaterialAdvisor(self.content),
            "sdr":       LiveSDRCapture(self.content),
            "video":     VideoReconstruction(self.content),
        }
        for f in self._frames.values():
            f.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._switch("home")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        # Stop any running SDR capture thread before tearing down the window.
        sdr_frame = self._frames.get("sdr")
        if sdr_frame is not None:
            try: sdr_frame.stop()
            except Exception: pass
        self.destroy()

    def _switch(self, key):
        for k, f in self._frames.items():
            f.lift() if k == key else f.lower()
        for k, btn in self._nav_btns.items():
            active = k == key
            # Selected nav item gets a darker, muted crimson fill (ACCENT_SEL) so
            # it reads as "you are here" and is never confused with the brighter
            # ACCENT used for primary action buttons elsewhere in the app.
            btn.configure(
                fg_color=ACCENT_SEL if active else "transparent",
                hover_color=ACCENT_SEL_HV if active else SIDEBAR_HOVER,
                text_color="#ffffff" if active else SIDEBAR_TEXT,
                font=ctk.CTkFont(size=13, weight="bold" if active else "normal")
            )


if __name__ == "__main__":
    app = TempestApp()
    app.mainloop()
