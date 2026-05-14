import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# matplotlib.animation removed — using root.after() loops instead
import numpy as np
import os
from paths import REPORTS_DIR

from data_engine import DataEngine
from ai_analyzer import AIAnalyzer
from report_generator import ReportGenerator

# ── Color palette ───────────────────────────────────────────────
BG        = "#080C14"
BG_PANEL  = "#0D1220"
BG_CARD   = "#101828"
BORDER    = "#1A2535"
GRID      = "#141E2C"
ACCENT    = "#00D4FF"
PURPLE    = "#7C3AED"
GREEN     = "#22C55E"
YELLOW    = "#F59E0B"
ORANGE    = "#F97316"
RED       = "#EF4444"
TXT       = "#E2E8F0"
TXT_DIM   = "#475569"

CHANNELS_BIPOLAR = [
    ("Fp1-F7","#3B82F6"),("F7-T3","#3B82F6"),("T3-T5","#3B82F6"),("T5-O1","#3B82F6"),
    ("Fp1-F3","#60A5FA"),("F3-C3","#60A5FA"),("C3-P3","#60A5FA"),("P3-O1","#60A5FA"),
    ("Fp2-F8","#F87171"),("F8-T4","#F87171"),("T4-T6","#F87171"),("T6-O2","#F87171"),
    ("Fp2-F4","#FCA5A5"),("F4-C4","#FCA5A5"),("C4-P4","#FCA5A5"),("P4-O2","#FCA5A5"),
    ("Fz-Cz","#34D399"),("Cz-Pz","#34D399"),
]

# Clinical monitor: fixed sensitivity (~7 uV/mm), single trace color, 200 ms sub-grid
EEG_TRACE = "#7FD899"
EEG_SUBGRID = "#0B1018"
EEG_UV_PER_MM = 7.0
EEG_MM_PER_DIV_Y = 3.0
EEG_TRACE_LW = 0.78

def _rcolor(r):
    if r >= 66: return RED
    if r >= 48: return ORANGE
    if r >= 24: return YELLOW
    return GREEN


class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CerebraWatch PRO — Clinical EEG Monitoring")
        self.root.geometry("1440x900")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        plt.rcParams.update({
            "figure.facecolor": BG_PANEL,
            "axes.facecolor":   BG_PANEL,
            "axes.edgecolor":   BORDER,
            "text.color":       TXT,
            "xtick.color":      TXT_DIM,
            "ytick.color":      TXT,
            "font.family":      "sans-serif",
        })

        self.engine   = DataEngine()
        self.ai       = AIAnalyzer()
        self.reporter = ReportGenerator()

        self.display_sec     = 10
        self.samples_to_show = self.engine.sample_rate * self.display_sec
        self.data_buffer     = np.full((18, self.samples_to_show), np.nan)
        self.time_mask       = np.linspace(-self.display_sec, 0, self.samples_to_show)

        self.is_running      = False
        self._eeg_job        = None   # after() EEG job id

        self._build_ui()
        self.root.after(200, self._deferred_init)

    # ════════════════════════════════════════════════════════════
    def _deferred_init(self):
        self.update_patient_ui()
        self.root.after(800, self.toggle_run)

    # ════════════════════════════════════════════════════════════
    #  UI BUILD
    # ════════════════════════════════════════════════════════════
    def _build_ui(self):
        self._topbar()
        self._main_area()
        self._statusbar()

    # ── Top Bar ─────────────────────────────────────────────────
    def _topbar(self):
        bar = tk.Frame(self.root, bg="#060A12", height=52)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)

        # Logo
        logo = tk.Frame(bar, bg="#060A12")
        logo.pack(side=tk.LEFT, padx=16)
        tk.Label(logo, text="Cerebra", bg="#060A12", fg=ACCENT,
                 font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        tk.Label(logo, text="Watch", bg="#060A12", fg=TXT,
                 font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        tk.Label(logo, text=" PRO", bg="#060A12", fg=PURPLE,
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, pady=(5,0))

        # Nav tabs (clickable)
        nav = tk.Frame(bar, bg="#060A12")
        nav.pack(side=tk.LEFT, padx=24)

        self._nav_frames = {}
        tab_defs = [
            ("Live View", None),
            ("Analysis",  self._open_analysis),
            ("Archives",  self._open_archives),
            ("Reports",   self._open_reports),
        ]
        for i, (t, cmd) in enumerate(tab_defs):
            f = tk.Frame(nav, bg="#060A12", cursor="hand2" if cmd else "arrow")
            f.pack(side=tk.LEFT, padx=6)
            lbl = tk.Label(f, text=t, bg="#060A12",
                           fg=ACCENT if i == 0 else TXT_DIM,
                           font=("Arial", 10), cursor="hand2" if cmd else "arrow")
            lbl.pack()
            bar2 = tk.Frame(f, bg=ACCENT if i == 0 else "#060A12", height=2)
            bar2.pack(fill=tk.X)
            if cmd:
                lbl.bind("<Button-1>", lambda e, c=cmd, l=lbl, b=bar2: c())
                f.bind("<Button-1>",   lambda e, c=cmd: c())
                lbl.bind("<Enter>", lambda e, l=lbl: l.config(fg=TXT))
                lbl.bind("<Leave>", lambda e, l=lbl: l.config(fg=TXT_DIM))

        # Buttons
        right = tk.Frame(bar, bg="#060A12")
        right.pack(side=tk.RIGHT, padx=16)

        self.btn_run = tk.Button(right, text="▶  Start", command=self.toggle_run,
            bg=GREEN, fg="#0A0A0A", font=("Arial",10,"bold"),
            relief=tk.FLAT, padx=14, pady=5, cursor="hand2")
        self.btn_run.pack(side=tk.RIGHT, padx=5)

        tk.Button(right, text="⟳  Next Patient", command=self.next_patient,
            bg=ACCENT, fg="#0A0A0A", font=("Arial",10,"bold"),
            relief=tk.FLAT, padx=14, pady=5, cursor="hand2").pack(side=tk.RIGHT, padx=5)

        tk.Button(right, text="MRI", command=self._open_mri,
            bg="#1E3A5F", fg=TXT, font=("Arial", 10, "bold"),
            relief=tk.FLAT, padx=12, pady=5, cursor="hand2").pack(side=tk.RIGHT, padx=5)

        tk.Button(right, text="⬇  Export PDF", command=self.generate_report,
            bg=PURPLE, fg="white", font=("Arial",10,"bold"),
            relief=tk.FLAT, padx=14, pady=5, cursor="hand2").pack(side=tk.RIGHT, padx=5)

    # ── Main Area ───────────────────────────────────────────────
    def _main_area(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Left EEG panel 70%
        left = tk.Frame(main, bg=BG_PANEL,
                        highlightbackground=BORDER, highlightthickness=1)
        left.place(relx=0, rely=0, relwidth=0.695, relheight=1.0)

        # Right panels 30%
        right = tk.Frame(main, bg=BG)
        right.place(relx=0.705, rely=0, relwidth=0.295, relheight=1.0)

        self._build_eeg(left)
        self._build_right(right)

    # ── EEG Panel ───────────────────────────────────────────────
    def _build_eeg(self, parent):
        # Header strip
        hdr = tk.Frame(parent, bg="#0A0F1C", height=30)
        hdr.pack(side=tk.TOP, fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="● LIVE STREAMING  18 CH", bg="#0A0F1C",
                 fg=RED, font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=12, pady=6)
        tk.Label(hdr, text="SENSITIVITY: 7μV/mm   SWEEP: 30mm/sec   FILTER: 0.5–70Hz   NOTCH: 60Hz",
                 bg="#0A0F1C", fg=TXT_DIM, font=("Arial", 8)).pack(side=tk.LEFT, padx=8)

        self.lbl_state = tk.Label(hdr, text="NORMAL", bg="#0A0F1C",
                                  fg=GREEN, font=("Arial", 9, "bold"))
        self.lbl_state.pack(side=tk.RIGHT, padx=12)

        self.lbl_data_src = tk.Label(hdr, text="◌ waiting for data...", bg="#0A0F1C",
                                     fg=TXT_DIM, font=("Arial", 8))
        self.lbl_data_src.pack(side=tk.RIGHT, padx=8)

        # EEG Figure
        self.fig_eeg, self.ax_eeg = plt.subplots(figsize=(11,8), dpi=95)
        self.fig_eeg.patch.set_facecolor(BG_PANEL)
        self.ax_eeg.set_facecolor(BG_PANEL)
        for s in self.ax_eeg.spines.values():
            s.set_visible(False)

        self.ax_eeg.set_xlim(-self.display_sec, 0)
        self.ax_eeg.set_xticks(np.arange(-self.display_sec, 1, 1))
        self.ax_eeg.set_xticklabels([])
        self.ax_eeg.tick_params(axis="x", length=0)

        for x in range(-self.display_sec, 1):
            self.ax_eeg.axvline(x=x, color=GRID, lw=0.8, alpha=0.9)
        for x in np.arange(-self.display_sec, 0, 0.2):
            if np.isclose(x % 1.0, 0.0, atol=0.02):
                continue
            self.ax_eeg.axvline(x=x, color=EEG_SUBGRID, lw=0.35, alpha=0.55)

        self.y_off = np.arange(18, 0, -1) * 3
        self.ax_eeg.set_yticks(self.y_off)
        self.ax_eeg.set_yticklabels([c[0] for c in CHANNELS_BIPOLAR],
                                    fontsize=8.5, color=TXT)
        self.ax_eeg.tick_params(axis="y", length=0)
        self.ax_eeg.set_ylim(-2, 57)

        for offset in self.y_off:
            self.ax_eeg.axhline(y=offset, color=GRID, lw=0.4, alpha=0.6)

        self.eeg_lines = []
        for i, (_, _c) in enumerate(CHANNELS_BIPOLAR):
            ln, = self.ax_eeg.plot(self.time_mask,
                                   np.nan_to_num(self.data_buffer[i], nan=0.0) + self.y_off[i],
                                   color=EEG_TRACE, lw=EEG_TRACE_LW, alpha=0.98)
            self.eeg_lines.append(ln)

        self.fig_eeg.tight_layout(pad=0.3)
        cv = FigureCanvasTkAgg(self.fig_eeg, master=parent)
        cv.draw()
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_eeg = cv

    # ── Right Panel ─────────────────────────────────────────────
    def _build_right(self, parent):
        # ── Risk Score Card ─────────────────────────────────────
        risk_card = tk.Frame(parent, bg=BG_CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        risk_card.place(relx=0, rely=0, relwidth=1.0, relheight=0.24)

        tk.Label(risk_card, text="REAL-TIME RISK SCORE", bg=BG_CARD,
                 fg=TXT_DIM, font=("Arial", 8, "bold")).pack(anchor="w", padx=10, pady=(8,0))

        self.lbl_risk_num = tk.Label(risk_card, text="—.—%", bg=BG_CARD,
                                     fg=GREEN, font=("Arial", 38, "bold"))
        self.lbl_risk_num.pack()

        self.lbl_risk_cls = tk.Label(risk_card, text="Starting...",
                                     bg=BG_CARD, fg=TXT_DIM, font=("Arial", 11))
        self.lbl_risk_cls.pack()

        self.risk_bar_cv = tk.Canvas(risk_card, bg=BG_CARD,
                                     height=12, highlightthickness=0)
        self.risk_bar_cv.pack(fill=tk.X, padx=10, pady=(4,8))

        # ── Spectral Chart ───────────────────────────────────────
        spec_card = tk.Frame(parent, bg=BG_CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        spec_card.place(relx=0, rely=0.25, relwidth=1.0, relheight=0.27)

        tk.Label(spec_card, text="SPECTRAL POWER DISTRIBUTION",
                 bg=BG_CARD, fg=TXT_DIM, font=("Arial", 8, "bold")).pack(anchor="w", padx=10, pady=(8,2))

        self.fig_spec, self.ax_spec = plt.subplots(figsize=(3.4, 2.0), dpi=92)
        self.fig_spec.patch.set_facecolor(BG_CARD)
        self.ax_spec.set_facecolor(BG_CARD)

        bands  = ["Delta","Theta","Alpha","Beta","Gamma"]
        bclrs  = [ACCENT, "#818CF8", GREEN, YELLOW, RED]
        self.spec_bars = self.ax_spec.bar(bands, [0]*5, color=bclrs, width=0.6)
        self.ax_spec.set_ylim(0, 1.0)
        self.ax_spec.set_xticklabels(bands, fontsize=8, color=TXT)
        self.ax_spec.set_yticks([0, 0.5, 1.0])
        self.ax_spec.set_yticklabels(["0","0.5","1"], fontsize=7, color=TXT_DIM)
        self.ax_spec.tick_params(length=0)
        for sp in self.ax_spec.spines.values():
            sp.set_visible(False)
        self.ax_spec.yaxis.grid(True, color=GRID, lw=0.7)
        self.ax_spec.set_axisbelow(True)
        self.fig_spec.tight_layout(pad=0.3)

        cv2 = FigureCanvasTkAgg(self.fig_spec, master=spec_card)
        cv2.draw()
        cv2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=(0,4))
        self.canvas_spec = cv2

        # ── Patient Profile Card ─────────────────────────────────
        pt_card = tk.Frame(parent, bg=BG_CARD,
                           highlightbackground=BORDER, highlightthickness=1)
        pt_card.place(relx=0, rely=0.53, relwidth=1.0, relheight=0.19)

        hdr = tk.Frame(pt_card, bg=BG_CARD)
        hdr.pack(fill=tk.X, padx=10, pady=(6,2))
        tk.Label(hdr, text="PATIENT PROFILE", bg=BG_CARD,
                 fg=TXT_DIM, font=("Arial", 8, "bold")).pack(side=tk.LEFT)
        self.lbl_db = tk.Label(hdr, text="", bg=BG_CARD,
                               fg=ACCENT, font=("Arial", 8))
        self.lbl_db.pack(side=tk.RIGHT)

        grid = tk.Frame(pt_card, bg=BG_CARD)
        grid.pack(fill=tk.X, padx=10)

        self._pt_vals = {}
        fields = [("ID","id"),("Age","age"),("Gender","gender"),
                  ("Outcome","outcome"),("Engel","engel"),("Follow-Up","follow_up")]

        for idx, (label, key) in enumerate(fields):
            col = (idx % 2) * 2
            row = (idx // 2) * 2
            tk.Label(grid, text=label.upper(), bg=BG_CARD,
                     fg=TXT_DIM, font=("Arial", 7, "bold")).grid(
                         row=row, column=col, sticky="w", padx=(0,20), pady=(1,0))
            v = tk.Label(grid, text="—", bg=BG_CARD,
                         fg=TXT, font=("Arial", 9, "bold"))
            v.grid(row=row+1, column=col, sticky="w", padx=(0,20))
            self._pt_vals[key] = v

        # ── AI clinical notes card ───────────────────────────────
        ai_card = tk.Frame(parent, bg=BG_CARD,
                           highlightbackground=BORDER, highlightthickness=1)
        ai_card.place(relx=0, rely=0.73, relwidth=1.0, relheight=0.27)

        ai_hdr = tk.Frame(ai_card, bg=BG_CARD)
        ai_hdr.pack(fill=tk.X, padx=10, pady=(6,2))
        tk.Label(ai_hdr, text="  AI CLINICAL NOTES", bg=BG_CARD,
                 fg=TXT_DIM, font=("Arial", 8, "bold")).pack(side=tk.LEFT)
        self.lbl_ai_time = tk.Label(ai_hdr, text="", bg=BG_CARD,
                                    fg=TXT_DIM, font=("Arial", 7))
        self.lbl_ai_time.pack(side=tk.RIGHT)

        # Scrollable text
        txt_frame = tk.Frame(ai_card, bg=BG_CARD)
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))

        scrollbar = tk.Scrollbar(txt_frame, bg=BORDER, troughcolor=BG_CARD,
                                 highlightthickness=0)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_clinical = tk.Text(
            txt_frame, bg="#060A14", fg=TXT,
            font=("Courier New", 8), wrap=tk.WORD,
            relief=tk.FLAT, state=tk.DISABLED,
            yscrollcommand=scrollbar.set,
            highlightthickness=0, cursor="arrow"
        )
        self.txt_clinical.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.txt_clinical.yview)

        # Color tags for clinical notes text widget
        self.txt_clinical.tag_config("header",  foreground=ACCENT,  font=("Courier New", 8, "bold"))
        self.txt_clinical.tag_config("warning", foreground=YELLOW)
        self.txt_clinical.tag_config("danger",  foreground=RED,     font=("Courier New", 8, "bold"))
        self.txt_clinical.tag_config("ok",      foreground=GREEN)
        self.txt_clinical.tag_config("dim",     foreground=TXT_DIM)
        self.txt_clinical.tag_config("rec",     foreground="#818CF8")

        self._analytics_cycle = 0


    # ── Status Bar ──────────────────────────────────────────────
    def _statusbar(self):
        bar = tk.Frame(self.root, bg="#060A12", height=24)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)

        self.lbl_status = tk.Label(bar, text="System ready.", bg="#060A12",
                                   fg=TXT_DIM, font=("Arial", 8))
        self.lbl_status.pack(side=tk.LEFT, padx=12)

        n = self.engine.get_patient_count()
        tk.Label(bar, text=f"Database: {n} patients loaded",
                 bg="#060A12", fg=ACCENT, font=("Arial", 8)).pack(side=tk.RIGHT, padx=12)

        self.lbl_eeg_source = tk.Label(bar, text="", bg="#060A12",
                                       fg="#22C55E", font=("Arial", 8, "bold"))
        self.lbl_eeg_source.pack(side=tk.RIGHT, padx=8)

    # ════════════════════════════════════════════════════════════
    #  PATIENT UI
    # ════════════════════════════════════════════════════════════
    def update_patient_ui(self):
        info = self.engine.get_patient_info()

        self._pt_vals["id"].config(text=info.get("id","—"))
        self._pt_vals["age"].config(text=f"{info.get('age','—')} yrs")
        self._pt_vals["gender"].config(text=info.get("gender","—"))
        self._pt_vals["follow_up"].config(text=f"{info.get('follow_up',0):.1f} yrs")
        self._pt_vals["engel"].config(text=info.get("engel","—"))

        oc = info.get("outcome","—")
        oc_clr = GREEN if "Successful" in oc else (RED if "Failed" in oc else YELLOW)
        self._pt_vals["outcome"].config(text=oc, fg=oc_clr)

        self.lbl_db.config(text=info.get("database",""))

        # Real / simulated / loading EEG indicator
        self._update_data_src_label()
        self.root.after(2000, self._update_data_src_label)
        self.root.after(5000, self._update_data_src_label)

    def _update_data_src_label(self):
        has_real   = (self.engine._eeg_loader is not None and
                      self.engine._eeg_loader.available)
        is_loading = self.engine._eeg_loading
        pid        = self.engine.current_patient.get("id", "—")

        if has_real:
            loader = self.engine._eeg_loader
            n_ch   = getattr(loader, "n_channels", "?")
            dur    = getattr(loader, "duration_sec", 0)
            txt_h  = f"● REAL iEEG  {n_ch} ch  {dur:.0f}s"
            txt_b  = "● REAL EEG"
            self.lbl_data_src.config(text=txt_h, fg=GREEN)
            self.lbl_eeg_source.config(text=txt_b, fg=GREEN)
        elif is_loading:
            self.lbl_data_src.config(text=f"⟳ Loading {pid}...", fg=YELLOW)
            self.lbl_eeg_source.config(text="⟳ EEG LOADING", fg=YELLOW)
        else:
            self.lbl_data_src.config(text="◌ SIMULATED EEG (no raw data)", fg=TXT_DIM)
            self.lbl_eeg_source.config(text="● SIMULATED EEG", fg=TXT_DIM)


    # ════════════════════════════════════════════════════════════
    #  CONTROLS
    # ════════════════════════════════════════════════════════════
    def toggle_run(self):
        if self.is_running:
            self.is_running = False
            self.btn_run.config(text="▶  Start", bg=GREEN)
            if self._eeg_job:
                self.root.after_cancel(self._eeg_job)
                self._eeg_job = None
        else:
            self.is_running = True
            self.btn_run.config(text="⏸  Pause", bg=ORANGE)
            self._schedule_eeg()

    def next_patient(self):
        was = self.is_running
        if was: self.toggle_run()
        self.engine.generate_new_patient()
        self.ai.reset_history()
        self.update_patient_ui()
        self.data_buffer[:] = np.nan
        for i, ln in enumerate(self.eeg_lines):
            ln.set_ydata(np.full_like(self.time_mask, self.y_off[i], dtype=float))
        self.canvas_eeg.draw_idle()
        self.lbl_risk_num.config(text="—.—%", fg=GREEN)
        self.lbl_risk_cls.config(text="Waiting for new patient...", fg=TXT_DIM)
        self.lbl_status.config(text="New patient loaded.", fg=TXT_DIM)
        if was: self.toggle_run()

    # ════════════════════════════════════════════════════════════
    #  ANIMATION (root.after loops — no FuncAnimation)
    # ════════════════════════════════════════════════════════════
    def _schedule_eeg(self):
        if not self.is_running:
            return
        self._update_eeg()
        self._update_analytics()
        self._eeg_job = self.root.after(80, self._schedule_eeg)

    def _clinical_eeg_y(self, row: np.ndarray, y0: float) -> np.ndarray:
        d = np.asarray(row, dtype=np.float64)
        d = np.nan_to_num(d, nan=0.0)
        span_pp = 10.0 * EEG_UV_PER_MM
        half = EEG_MM_PER_DIV_Y * 0.40
        g = half / max(span_pp, 1e-6)
        amp = np.clip(d * g, -half, half)
        return amp + y0

    def _update_eeg(self):
        chunk = self.engine.eeg_frame_samples
        win, state = self.engine.generate_window()
        self.data_buffer = np.roll(self.data_buffer, -chunk, axis=1)
        self.data_buffer[:, -chunk:] = win[:, :chunk]

        prev = getattr(self, "_prev_state", "normal")
        has_real = (self.engine._eeg_loader is not None and
                    self.engine._eeg_loader.available)
        if has_real and prev in ("seizure", "warning", "attention") and state == "normal":
            self.ai.reset_history()
        self._prev_state = state

        for i, ln in enumerate(self.eeg_lines):
            ln.set_ydata(self._clinical_eeg_y(self.data_buffer[i], self.y_off[i]))

        state_cfg = {
            "normal":    ("NORMAL",    GREEN),
            "attention": ("ATTENTION", YELLOW),
            "warning":   ("WARNING",   ORANGE),
            "seizure":   ("⚠ SEIZURE", RED),
        }
        lbl, clr = state_cfg.get(state, ("—", TXT_DIM))
        self.lbl_state.config(text=lbl, fg=clr)
        self.canvas_eeg.draw_idle()


    def _update_analytics(self):
        recent = self.data_buffer[:, -self.engine.samples_per_window:]
        if np.isnan(recent).any():
            return
        if np.std(recent) < 0.12:
            return

        base_risk = self.engine.current_patient.get("base_risk", 1.0)

        res    = self.ai.analyze_window(recent, base_risk)
        powers = res["band_powers"]

        risk = res["risk_score"]
        cls  = res["classification"]

        has_real = (self.engine._eeg_loader is not None and
                    self.engine._eeg_loader.available)
        if not has_real:
            st = self.engine.current_patient.get("state", "normal")
            floor = {"normal": 0, "attention": 30, "warning": 52, "seizure": 82}
            risk = max(risk, floor.get(st, 0))
            cls, sev = self.ai._classify(risk)
            res["risk_score"] = risk
            res["classification"] = cls
            res["severity"] = sev

        clr = _rcolor(risk)
        self.lbl_risk_num.config(text=f"{risk:.1f}%", fg=clr)
        self.lbl_risk_cls.config(text=cls, fg=clr)
        self._draw_risk_bar(risk, clr)

        # Spectral bars
        bands = ["Delta","Theta","Alpha","Beta","Gamma"]
        n = len(powers)
        avgs = [sum(ch.get(b,0) for ch in powers)/n for b in bands]
        for bar, h in zip(self.spec_bars, avgs):
            bar.set_height(h)
        self.canvas_spec.draw_idle()

        pid = self.engine.current_patient.get("id","")
        self.lbl_status.config(
            text=f"AI: {cls}  |  Risk: {risk:.1f}%  |  {pid}",
            fg=clr
        )

        # Clinical notes ~every 5 s at 80 ms/tick
        self._analytics_cycle = getattr(self, "_analytics_cycle", 0) + 1
        if self._analytics_cycle % 63 == 1:
            self._refresh_clinical_notes(res, risk, cls)

    def _refresh_clinical_notes(self, res, risk, cls):
        """Refresh AI clinical notes panel."""
        import datetime
        info = self.engine.get_patient_info()
        try:
            cs = self.ai.get_clinical_summary(info, res)
        except Exception:
            return

        interp = cs.get("interpretation", [])
        recs   = cs.get("recommendations", [])
        dom    = cs.get("dominant_band", "—")
        now    = datetime.datetime.now().strftime("%H:%M:%S")

        self.lbl_ai_time.config(text=now)

        t = self.txt_clinical
        t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        t.insert(tk.END, f"── {cls}  |  Risk: {risk:.1f}%  |  Dom. band: {dom} ──\n", "header")
        t.insert(tk.END, "\n")

        for line in interp:
            if "CRITICAL" in line or "critical" in line.lower():
                t.insert(tk.END, f"⚠ {line}\n", "danger")
            elif "WARNING" in line or "warning" in line.lower():
                t.insert(tk.END, f"⚡ {line}\n", "warning")
            elif "CAUTION" in line or "caution" in line.lower():
                t.insert(tk.END, f"⚡ {line}\n", "warning")
            elif "Normal" in line or "normal" in line.lower():
                t.insert(tk.END, f"✓ {line}\n", "ok")
            else:
                t.insert(tk.END, f"• {line}\n")
            t.insert(tk.END, "\n")

        if recs:
            t.insert(tk.END, "── RECOMMENDATIONS ──\n", "header")
            for rec in recs:
                t.insert(tk.END, f"{rec}\n", "rec")

        t.config(state=tk.DISABLED)
        t.see("1.0")


    def _draw_risk_bar(self, risk, color):
        self.risk_bar_cv.update_idletasks()
        w = self.risk_bar_cv.winfo_width()
        h = self.risk_bar_cv.winfo_height()
        if w <= 1: return
        self.risk_bar_cv.delete("all")
        self.risk_bar_cv.create_rectangle(0, 0, w, h, fill=BORDER, outline="")
        filled = int(w * risk / 100)
        if filled > 0:
            self.risk_bar_cv.create_rectangle(0, 0, filled, h,
                                               fill=color, outline="")

    # ════════════════════════════════════════════════════════════
    #  NAV TAB WINDOWS
    # ════════════════════════════════════════════════════════════

    def _open_mri(self):
        """MRI/FLAIR — NIfTI volumes under mri/ only."""
        try:
            from mri_viewer import MRIViewerWindow
        except ImportError as e:
            messagebox.showerror("MRI", f"Could not load module:\n{e}")
            return
        info = self.engine.get_patient_info()
        MRIViewerWindow(self.root, info)

    def _open_analysis(self):
        """Analysis tab — live AI metrics window."""
        win = tk.Toplevel(self.root)
        win.title("CerebraWatch — Analysis")
        win.geometry("640x540")
        win.configure(bg=BG)
        win.resizable(True, True)

        tk.Label(win, text="  DETAILED ANALYSIS", bg=BG, fg=ACCENT,
                 font=("Arial", 13, "bold")).pack(pady=(16, 4))
        tk.Frame(win, bg=BORDER, height=1).pack(fill=tk.X, padx=20)

        # Live metric grid
        grid = tk.Frame(win, bg=BG)
        grid.pack(fill=tk.X, padx=24, pady=12)

        info = self.engine.get_patient_info()
        recent = self.data_buffer[:, -self.engine.samples_per_window:]

        if not np.isnan(recent).any() and np.std(recent) > 0.5:
            res = self.ai.analyze_window(recent, info.get("base_risk", 1.0))
            cs  = self.ai.get_clinical_summary(info, res)
            metrics = [
                ("Patient ID",      info.get("id", "—")),
                ("Risk score",      f"{res['risk_score']:.1f}%"),
                ("Raw risk",        f"{res.get('raw_risk', 0):.1f}%"),
                ("Classification", res["classification"]),
                ("Dominant band",   cs.get("dominant_band", "—")),
                ("Total spikes",    str(res.get("total_spikes", "—"))),
                ("Seizure count",   str(cs.get("seizure_count", 0))),
                ("Avg risk (~1m)",  f"{cs.get('avg_risk_1min', 0):.1f}%"),
                ("Max risk",        f"{cs.get('max_risk', 0):.1f}%"),
                ("Windows analyzed", str(cs.get("window_analyzed", "—"))),
            ]
        else:
            metrics = [("Status", "Waiting for sufficient data...")]

        for r, (label, val) in enumerate(metrics):
            tk.Label(grid, text=label, bg=BG, fg=TXT_DIM,
                     font=("Arial", 9)).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 30))
            clr = RED if "%" in val and float(val.replace("%","")) >= 66 \
                  else YELLOW if "%" in val and float(val.replace("%","")) >= 48 \
                  else TXT
            tk.Label(grid, text=val, bg=BG, fg=clr,
                     font=("Courier New", 10, "bold")).grid(row=r, column=1, sticky="w")

        # Band powers
        tk.Frame(win, bg=BORDER, height=1).pack(fill=tk.X, padx=20, pady=(8,4))
        tk.Label(win, text="BAND POWERS (18-ch mean)", bg=BG, fg=TXT_DIM,
                 font=("Arial", 8, "bold")).pack(anchor="w", padx=24)

        if not np.isnan(recent).any() and np.std(recent) > 0.5:
            bands = ["Delta","Theta","Alpha","Beta","Gamma"]
            bclrs = [ACCENT, "#818CF8", GREEN, YELLOW, RED]
            bf = tk.Frame(win, bg=BG); bf.pack(fill=tk.X, padx=24, pady=8)
            for b, c in zip(bands, bclrs):
                avg = sum(res["band_powers"][i].get(b, 0) for i in range(len(res["band_powers"]))) / max(1, len(res["band_powers"]))
                val_pct = int(avg * 100)
                col_f = tk.Frame(bf, bg=BG); col_f.pack(side=tk.LEFT, expand=True)
                tk.Label(col_f, text=b, bg=BG, fg=c, font=("Arial", 8)).pack()
                bar_bg = tk.Frame(col_f, bg="#1A2030", width=60, height=80); bar_bg.pack()
                bar_bg.pack_propagate(False)
                bar_h = int(80 * avg)
                tk.Frame(bar_bg, bg=c, height=max(2, bar_h), width=44).place(relx=0.5, rely=1.0, anchor="s")
                tk.Label(col_f, text=f"{val_pct}%", bg=BG, fg=TXT_DIM, font=("Arial", 7)).pack()

        tk.Button(win, text="Close", command=win.destroy,
                  bg=BORDER, fg=TXT, font=("Arial", 10),
                  relief=tk.FLAT, padx=20, pady=6).pack(pady=14)

    def _open_archives(self):
        """Archives tab — patient session seizure log."""
        win = tk.Toplevel(self.root)
        win.title("CerebraWatch — Archives")
        win.geometry("700x480")
        win.configure(bg=BG)

        tk.Label(win, text="🗄  PATIENT ARCHIVE / SEIZURE LOG", bg=BG, fg=ACCENT,
                 font=("Arial", 13, "bold")).pack(pady=(16, 4))
        tk.Frame(win, bg=BORDER, height=1).pack(fill=tk.X, padx=20)

        cols = ("Patient ID", "Risk %", "Class", "Seizure", "Time")
        import tkinter.ttk as ttk
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview", background="#0B0F1A", foreground=TXT,
                        fieldbackground="#0B0F1A", rowheight=26,
                        font=("Courier New", 9))
        style.configure("Dark.Treeview.Heading", background=BG_CARD,
                        foreground=ACCENT, font=("Arial", 9, "bold"))
        style.map("Dark.Treeview", background=[("selected", ACCENT)],
                  foreground=[("selected", "#000")])

        tree_f = tk.Frame(win, bg=BG); tree_f.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)
        sb = tk.Scrollbar(tree_f, bg=BORDER); sb.pack(side=tk.RIGHT, fill=tk.Y)
        tree = ttk.Treeview(tree_f, columns=cols, show="headings",
                            style="Dark.Treeview", yscrollcommand=sb.set)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=130, anchor="center")
        sb.config(command=tree.yview)
        tree.pack(fill=tk.BOTH, expand=True)

        # Seizure log
        sz_log = self.ai._seizure_log
        if sz_log:
            for entry in reversed(sz_log[-50:]):
                tree.insert("", tk.END, values=(
                    self.engine.current_patient.get("id", "—"),
                    f"{entry['risk']:.1f}%",
                    "Critical (Seizure)",
                    f"#{entry['window_no']}",
                    entry["time"]
                ))
        else:
            tree.insert("", tk.END, values=("—", "—", "No seizure this session", "—", "—"))

        tk.Button(win, text="Close", command=win.destroy,
                  bg=BORDER, fg=TXT, font=("Arial", 10),
                  relief=tk.FLAT, padx=20, pady=6).pack(pady=10)

    def _open_reports(self):
        """Reports tab — PDF list and open."""
        import subprocess, platform
        win = tk.Toplevel(self.root)
        win.title("CerebraWatch — Reports")
        win.geometry("680x460")
        win.configure(bg=BG)

        tk.Label(win, text="📄  REPORTS", bg=BG, fg=ACCENT,
                 font=("Arial", 13, "bold")).pack(pady=(16, 4))
        tk.Frame(win, bg=BORDER, height=1).pack(fill=tk.X, padx=20)
        tk.Label(win, text="Double-click to open a report — or generate a new one.",
                 bg=BG, fg=TXT_DIM, font=("Arial", 9)).pack(pady=4)

        rdir = REPORTS_DIR
        os.makedirs(rdir, exist_ok=True)

        listbox_f = tk.Frame(win, bg=BG); listbox_f.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)
        sb2 = tk.Scrollbar(listbox_f, bg=BORDER); sb2.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(listbox_f, bg="#0B0F1A", fg=TXT, font=("Courier New", 10),
                        selectbackground=ACCENT, selectforeground="#000",
                        relief=tk.FLAT, yscrollcommand=sb2.set,
                        highlightthickness=0, activestyle="none")
        lb.pack(fill=tk.BOTH, expand=True)
        sb2.config(command=lb.yview)

        pdfs = sorted([f for f in os.listdir(rdir) if f.endswith(".pdf")], reverse=True)
        if pdfs:
            for f in pdfs:
                size = os.path.getsize(os.path.join(rdir, f)) // 1024
                lb.insert(tk.END, f"  📄  {f}   ({size} KB)")
        else:
            lb.insert(tk.END, "  No reports generated yet.")

        def open_selected(event=None):
            sel = lb.curselection()
            if not sel or not pdfs: return
            path = os.path.join(rdir, pdfs[sel[0]])
            if platform.system() == "Windows":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])

        lb.bind("<Double-Button-1>", open_selected)

        btn_f = tk.Frame(win, bg=BG); btn_f.pack(pady=10)
        tk.Button(btn_f, text="📄  Open Selected Report", command=open_selected,
                  bg=ACCENT, fg="#000", font=("Arial", 10, "bold"),
                  relief=tk.FLAT, padx=16, pady=6).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_f, text="⬇  Generate New Report", command=lambda: [win.destroy(), self.generate_report()],
                  bg=PURPLE, fg="white", font=("Arial", 10, "bold"),
                  relief=tk.FLAT, padx=16, pady=6).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_f, text="Close", command=win.destroy,
                  bg=BORDER, fg=TXT, font=("Arial", 10),
                  relief=tk.FLAT, padx=16, pady=6).pack(side=tk.LEFT, padx=8)

    # ════════════════════════════════════════════════════════════
    #  PDF REPORT
    # ════════════════════════════════════════════════════════════
    def generate_report(self):
        was = self.is_running
        if was: self.toggle_run()

        self.lbl_status.config(text="Generating PDF report...", fg=TXT)
        self.root.update()

        info   = self.engine.get_patient_info()
        recent = self.data_buffer[:, -self.engine.samples_per_window:]

        if np.isnan(recent).any():
            messagebox.showwarning("Warning", "Not enough data yet.")
            if was: self.toggle_run()
            return

        res             = self.ai.analyze_window(recent, info["base_risk"])
        clinical_summary = self.ai.get_clinical_summary(info, res)
        max_risk        = max(self.ai.risk_history)

        try:
            fname = self.reporter.generate_pdf(
                patient_info=info,
                eeg_figure=self.fig_eeg,
                spectral_figure=self.fig_spec,
                max_risk=max_risk,
                avg_risk=res["risk_score"],
                final_decision=res["classification"],
                clinical_summary=clinical_summary,
            )
            messagebox.showinfo("Success", f"4-page clinical report:\n{fname}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate report:\n{e}")

        if was: self.toggle_run()


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()
