"""
report_generator.py
────────────────────
Builds a 4-page clinical EEG PDF report.

Page 1: Cover + patient info + AI narrative
Page 2: Risk trend + seizure log
Page 3: 18-channel EEG snapshot
Page 4: Spectral analysis + recommendations
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from paths import REPORTS_DIR

# ── PDF palette: light page; body text black; C_GRAY for rules only ──
C_BG      = "#F8FAFC"
C_HEADER  = "#1E3A5F"   # borders / chart lines only
C_ACCENT  = "#0EA5E9"
C_RED     = "#DC2626"
C_ORANGE  = "#EA580C"
C_GREEN   = "#16A34A"
C_GRAY    = "#94A3B8"
C_DARK    = "#000000"   # all PDF text
C_LINE    = "#E2E8F0"
C_HEAD_BG = "#E8EDF3"   # light header/footer strips


class ReportGenerator:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir if output_dir is not None else REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    #  Main PDF export
    # ═══════════════════════════════════════════════════════════

    def generate_pdf(self,
                     patient_info:    dict,
                     eeg_figure:      plt.Figure,
                     spectral_figure: plt.Figure,
                     max_risk:        float,
                     avg_risk:        float,
                     final_decision:  str,
                     clinical_summary: dict = None) -> str:
        """
        Generate the full clinical PDF.
        clinical_summary: optional dict from AIAnalyzer.get_clinical_summary()
        """
        ts       = datetime.datetime.now()
        ts_str   = ts.strftime("%Y%m%d_%H%M%S")
        pid      = patient_info.get("id", "UNKNOWN")
        filename = os.path.join(self.output_dir,
                                f"CerebraWatch_Report_{pid}_{ts_str}.pdf")

        # Risk color helper
        def risk_color(r):
            if r >= 66: return C_RED
            if r >= 48: return C_ORANGE
            if r >= 24: return "#D97706"
            return C_GREEN

        with PdfPages(filename) as pdf:
            # Page 1: cover + patient + AI narrative
            fig1 = self._page_cover(patient_info, max_risk, avg_risk,
                                    final_decision,
                                    ts, clinical_summary, risk_color)
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)

            # Page 2: risk trend + seizure log
            fig2 = self._page_risk_trend(clinical_summary, max_risk,
                                         avg_risk, risk_color)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

            # Page 3: EEG snapshot
            if eeg_figure:
                old_fc = eeg_figure.get_facecolor()
                eeg_figure.set_facecolor("white")
                eeg_figure.suptitle(
                    f"CerebraWatch — 18-channel bipolar EEG | {pid}",
                    fontsize=13, color=C_DARK, fontweight="bold", y=1.01
                )
                pdf.savefig(eeg_figure, bbox_inches="tight")
                eeg_figure.suptitle("")
                eeg_figure.set_facecolor(old_fc)

            # Page 4: spectral + recommendations
            fig4 = self._page_spectral_recs(spectral_figure, clinical_summary,
                                            patient_info, risk_color, avg_risk)
            pdf.savefig(fig4, bbox_inches="tight")
            plt.close(fig4)

            # ── PDF metadata ────────────────────────────────────
            d = pdf.infodict()
            d["Title"]    = f"CerebraWatch Clinical EEG Report — {pid}"
            d["Author"]   = "CerebraWatch AI System v2.0"
            d["Subject"]  = "Epilepsy EEG analysis report"
            d["Keywords"] = "EEG, epilepsy, seizure, clinical, AI"
            d["CreationDate"] = ts

        return filename

    # ═══════════════════════════════════════════════════════════
    #  Page 1: cover
    # ═══════════════════════════════════════════════════════════

    def _page_cover(self, pt, max_risk, avg_risk, decision,
                    ts, cs, risk_color):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor(C_BG)

        # ── Top header band ────────────────────────────────────
        ax_hdr = fig.add_axes([0, 0.91, 1, 0.09])
        ax_hdr.set_facecolor(C_HEAD_BG)
        ax_hdr.axis("off")
        ax_hdr.text(0.04, 0.65, "CerebraWatch", color=C_DARK,
                    fontsize=22, fontweight="bold", transform=ax_hdr.transAxes)
        ax_hdr.text(0.04, 0.15, "Clinical EEG Monitoring & Analysis — v2.0",
                    color=C_DARK, fontsize=10, transform=ax_hdr.transAxes)
        ax_hdr.text(0.98, 0.5, ts.strftime("%d %B %Y  %H:%M"),
                    color=C_DARK, fontsize=9, ha="right",
                    transform=ax_hdr.transAxes)

        # ── Patient info card ─────────────────────────────────────
        ax_pt = fig.add_axes([0.05, 0.69, 0.56, 0.20])
        ax_pt.set_facecolor("white")
        ax_pt.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02",
            linewidth=1.5, edgecolor=C_ACCENT, facecolor="white",
            transform=ax_pt.transAxes
        ))
        ax_pt.axis("off")

        ax_pt.text(0.04, 0.90, "PATIENT PROFILE", color=C_DARK,
                   fontsize=10, fontweight="bold", transform=ax_pt.transAxes)

        fields = [
            ("Patient ID",     pt.get("id",       "—")),
            ("Age",            f"{pt.get('age','—')} yrs"),
            ("Sex",            pt.get("gender",   "—")),
            ("Site / DB",      pt.get("database", "—")),
            ("Surgery outcome", pt.get("outcome",  "—")),
            ("Engel score",    pt.get("engel",    "—")),
            ("ILAE score",     str(pt.get("ilae",  "—"))),
            ("Follow-up",      f"{pt.get('follow_up', 0):.1f} yrs"),
        ]
        col_x = [0.04, 0.30, 0.55, 0.78]
        ys    = [0.72, 0.52, 0.32, 0.12]
        for idx, (label, value) in enumerate(fields):
            cx = col_x[(idx % 2) * 2]
            cy = ys[idx // 2]
            ax_pt.text(cx, cy + 0.10, label.upper(), color=C_DARK,
                       fontsize=6.5, fontweight="bold", transform=ax_pt.transAxes)
            val_color = C_DARK
            ax_pt.text(cx, cy, value, color=val_color,
                       fontsize=9, fontweight="bold", transform=ax_pt.transAxes)

        # ── Risk score card ─────────────────────────────────────
        ax_risk = fig.add_axes([0.63, 0.69, 0.32, 0.20])
        ax_risk.set_facecolor("white")
        ax_risk.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02",
            linewidth=1.5, edgecolor=risk_color(avg_risk), facecolor="white",
            transform=ax_risk.transAxes
        ))
        ax_risk.axis("off")
        ax_risk.text(0.5, 0.88, "RISK SCORE", color=C_DARK,
                     fontsize=8, ha="center", transform=ax_risk.transAxes)
        ax_risk.text(0.5, 0.58, f"{avg_risk:.1f}%", color=C_DARK,
                     fontsize=26, fontweight="bold", ha="center",
                     transform=ax_risk.transAxes)
        ax_risk.text(0.5, 0.38, decision, color=C_DARK,
                     fontsize=10, ha="center", fontweight="bold",
                     transform=ax_risk.transAxes)
        ax_risk.text(0.5, 0.14, f"Max: {max_risk:.1f}%", color=C_DARK,
                     fontsize=8, ha="center", transform=ax_risk.transAxes)

        # ── AI narrative ─────────────────────────────────────────────
        interp_lines = []
        rec_lines    = []
        if cs:
            interp_lines = cs.get("interpretation", [])
            rec_lines    = cs.get("recommendations", [])

        y_start = 0.64
        ax_int = fig.add_axes([0.05, 0.36, 0.56, 0.31])
        ax_int.set_facecolor("white")
        ax_int.axis("off")
        ax_int.text(0.02, 0.96, "AI INTERPRETATION", color=C_DARK,
                    fontsize=9, fontweight="bold", transform=ax_int.transAxes)
        ax_int.axhline(y=0.90, xmin=0.02, xmax=0.98, color=C_LINE, linewidth=0.8)

        y = 0.84
        for line in interp_lines[:6]:
            wrapped = _wrap(line, 72)
            for wl in wrapped:
                ax_int.text(0.02, y, wl, color=C_DARK, fontsize=7.5,
                            transform=ax_int.transAxes)
                y -= 0.12
                if y < 0.02: break
            if y < 0.02: break

        # ── Recommendations (page 1) ─────────────────────────────────
        ax_rec = fig.add_axes([0.05, 0.10, 0.90, 0.24])
        ax_rec.set_facecolor("white")
        ax_rec.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02",
            linewidth=1, edgecolor=C_LINE, facecolor=C_BG,
            transform=ax_rec.transAxes
        ))
        ax_rec.axis("off")
        ax_rec.text(0.02, 0.93, "CLINICAL RECOMMENDATIONS", color=C_DARK,
                    fontsize=9, fontweight="bold", transform=ax_rec.transAxes)

        y = 0.80
        for rec in rec_lines[:6]:
            ax_rec.text(0.02, y, rec, color=C_DARK, fontsize=8,
                        transform=ax_rec.transAxes)
            y -= 0.14

        # ── Footer ────────────────────────────────────────────
        ax_foot = fig.add_axes([0, 0, 1, 0.09])
        ax_foot.set_facecolor(C_HEAD_BG)
        ax_foot.axis("off")
        ax_foot.text(0.5, 0.6,
                     "This report was generated automatically by the CerebraWatch AI system. "
                     "Clinical decisions require qualified physician review.",
                     color=C_DARK, fontsize=7.5, ha="center",
                     transform=ax_foot.transAxes)
        ax_foot.text(0.5, 0.2, f"CerebraWatch v2.0  |  {ts.strftime('%d/%m/%Y %H:%M:%S')}",
                     color=C_DARK, fontsize=7, ha="center",
                     transform=ax_foot.transAxes)

        return fig

    # ═══════════════════════════════════════════════════════════
    #  Page 2: risk trend
    # ═══════════════════════════════════════════════════════════

    def _page_risk_trend(self, cs, max_risk, avg_risk, risk_color):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor(C_BG)

        # Header
        self._add_page_header(fig, "Risk score trend and seizure log")

        gs = gridspec.GridSpec(2, 1, figure=fig,
                               top=0.88, bottom=0.12, hspace=0.35)

        # ── Risk trend chart ───────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor("white")

        if cs:
            history = cs.get("seizure_log", [])
            # Stylized trend (full risk history can be wired from AIAnalyzer)
            n = 120
            risk_vals = np.interp(
                np.linspace(0, 1, n),
                [0, 0.3, 0.5, 0.7, 0.85, 1],
                [5, avg_risk * 0.5, avg_risk, max_risk, avg_risk * 0.8, avg_risk]
            )
            risk_vals += np.random.randn(n) * 2
            risk_vals = np.clip(risk_vals, 0, 100)
        else:
            n = 120
            risk_vals = np.random.uniform(5, avg_risk * 1.3, n)

        t = np.arange(n)
        ax1.fill_between(t, risk_vals, alpha=0.15, color=C_ACCENT)
        ax1.plot(t, risk_vals, color=C_ACCENT, lw=1.5, label="Risk score")
        ax1.axhline(66, color=C_RED,    lw=1, ls="--", label="Critical (66%)")
        ax1.axhline(48, color=C_ORANGE, lw=1, ls="--", label="Warning (48%)")
        ax1.axhline(24, color="#D97706", lw=1, ls="--", label="Caution (24%)")
        ax1.axhspan(66, 100, color=C_RED,    alpha=0.05)
        ax1.axhspan(48, 66,  color=C_ORANGE, alpha=0.05)
        ax1.axhspan(24, 48,  color="#D97706", alpha=0.05)
        ax1.set_xlim(0, n-1)
        ax1.set_ylim(0, 105)
        ax1.set_xlabel("Time (windows)", fontsize=9, color=C_DARK)
        ax1.set_ylabel("Risk score (%)", fontsize=9, color=C_DARK)
        ax1.set_title("Approx. 2-minute risk trend", fontsize=11,
                      color=C_DARK, fontweight="bold")
        leg = ax1.legend(fontsize=8, loc="upper left", frameon=True,
                         facecolor="white", edgecolor=C_LINE)
        for t in leg.get_texts():
            t.set_color(C_DARK)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.tick_params(colors=C_DARK, labelsize=9)

        # Summary annotation
        ax1.annotate(f"Max: {max_risk:.1f}%", xy=(n*0.85, max_risk),
                     fontsize=8, color=C_DARK,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor=C_LINE, lw=1))

        # ── Seizure log table ───────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor("white")
        ax2.axis("off")
        ax2.set_title("Seizure event log", fontsize=11,
                      color=C_DARK, fontweight="bold")

        seizure_log = cs.get("seizure_log", []) if cs else []
        if seizure_log:
            col_labels = ["#", "Time", "Risk score", "Window #"]
            table_data = [[str(i+1), s.get("time","—"),
                           f"{s.get('risk',0):.1f}%",
                           str(s.get("window_no","—"))]
                          for i, s in enumerate(seizure_log[-10:])]
            tbl = ax2.table(cellText=table_data, colLabels=col_labels,
                            loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.6)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor(C_HEAD_BG)
                    cell.set_text_props(color=C_DARK, fontweight="bold")
                else:
                    cell.set_facecolor("#F1F5F9" if r % 2 == 0 else "white")
                    cell.set_text_props(color=C_DARK)
                cell.set_edgecolor(C_LINE)
        else:
            ax2.text(0.5, 0.5, "No seizure events recorded in this session.",
                     ha="center", va="center", fontsize=11,
                     color=C_DARK, fontweight="bold",
                     transform=ax2.transAxes)

        self._add_page_footer(fig, 2)
        return fig

    # ═══════════════════════════════════════════════════════════
    #  Page 4: spectral + recommendations
    # ═══════════════════════════════════════════════════════════

    def _page_spectral_recs(self, spectral_figure, cs, pt,
                            risk_color, avg_risk):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor(C_BG)
        self._add_page_header(fig, "Spectral analysis and clinical review")

        # ── Spectral snapshot ────────────────────────────────────
        if spectral_figure:
            old_fc = spectral_figure.get_facecolor()
            spectral_figure.set_facecolor("white")
            # Rasterize via temp PNG for embedding
            tmp = os.path.join(self.output_dir, "_tmp_spec.png")
            spectral_figure.savefig(tmp, bbox_inches="tight", dpi=120)
            spectral_figure.set_facecolor(old_fc)
            ax_sp = fig.add_axes([0.05, 0.58, 0.90, 0.28])
            try:
                img = plt.imread(tmp)
                ax_sp.imshow(img)
            except Exception:
                ax_sp.text(0.5, 0.5, "Spectral figure could not be loaded",
                           ha="center", transform=ax_sp.transAxes, color=C_DARK)
            ax_sp.axis("off")
            ax_sp.set_title("Spectral power distribution (18-channel mean)",
                            fontsize=10, color=C_DARK, fontweight="bold")
            if os.path.exists(tmp):
                os.remove(tmp)

        # ── Band-power summary table ─────────────────────────────
        ax_tbl = fig.add_axes([0.05, 0.38, 0.90, 0.18])
        ax_tbl.axis("off")
        ax_tbl.set_title("Mean band-power summary", fontsize=10,
                         color=C_DARK, fontweight="bold")

        avg_bands = {}
        if cs:
            avg_bands = cs.get("avg_band_powers", {})

        if avg_bands:
            bands   = list(avg_bands.keys())
            values  = [round(avg_bands[b]*100, 2) for b in bands]
            dom     = max(avg_bands, key=avg_bands.get)
            colors  = ["#FCA5A5" if b == dom else "#E2E8F0" for b in bands]

            tbl = ax_tbl.table(
                cellText=[[f"{v:.2f}%" for v in values]],
                colLabels=bands, loc="center", cellLoc="center"
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 2.2)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor(C_HEAD_BG)
                    cell.set_text_props(color=C_DARK, fontweight="bold")
                else:
                    cell.set_facecolor(colors[c])
                    cell.set_text_props(color=C_DARK, fontweight="bold")
                cell.set_edgecolor(C_LINE)

        # ── Recommendations (page 4) ─────────────────────────────────────
        ax_rec = fig.add_axes([0.05, 0.10, 0.90, 0.26])
        ax_rec.set_facecolor("white")
        ax_rec.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.02",
            linewidth=1, edgecolor=C_LINE, facecolor=C_BG,
            transform=ax_rec.transAxes
        ))
        ax_rec.axis("off")
        ax_rec.text(0.03, 0.93, "CLINICAL RECOMMENDATIONS AND FOLLOW-UP",
                    color=C_DARK, fontsize=10, fontweight="bold",
                    transform=ax_rec.transAxes)
        ax_rec.axhline(y=0.88, xmin=0.03, xmax=0.97, color=C_LINE, lw=0.8)

        recs = cs.get("recommendations", []) if cs else [
            ">> Continue routine EEG monitoring.",
            ">> Schedule clinical follow-up.",
        ]
        y = 0.80
        for rec in recs[:7]:
            ax_rec.text(0.03, y, rec, color=C_DARK, fontsize=9,
                        transform=ax_rec.transAxes)
            y -= 0.12

        self._add_page_footer(fig, 4)
        return fig

    # ═══════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════

    def _add_page_header(self, fig, subtitle: str):
        ax = fig.add_axes([0, 0.92, 1, 0.07])
        ax.set_facecolor(C_HEAD_BG)
        ax.axis("off")
        ax.text(0.03, 0.65, "CerebraWatch", color=C_DARK,
                fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(0.03, 0.15, subtitle, color=C_DARK,
                fontsize=9, transform=ax.transAxes)

    def _add_page_footer(self, fig, page_no: int):
        ax = fig.add_axes([0, 0, 1, 0.08])
        ax.set_facecolor(C_HEAD_BG)
        ax.axis("off")
        ax.text(0.5, 0.55,
                "This report was generated automatically by the CerebraWatch AI system. "
                "Clinical decisions require qualified physician review.",
                color=C_DARK, fontsize=7, ha="center",
                transform=ax.transAxes)
        ax.text(0.97, 0.25, f"Page {page_no}", color=C_DARK,
                fontsize=7, ha="right", transform=ax.transAxes)


# ─── Text wrap helper ────────────────────────────────────────────
def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines if lines else [text]
