import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.stats import kurtosis, skew
import datetime

# Max raw risk samples (~60 s when analytics runs every ~80 ms)
_RISK_HISTORY_MAX = 750


class AIAnalyzer:
    """
    Advanced EEG analysis engine.
    Extracts clinical features from 18-channel windows, computes risk,
    and builds structured report data.
    """

    def __init__(self, sample_rate: int = 256):
        self.sample_rate  = sample_rate
        self.risk_history = [0.0] * 30   # priming; grows to _RISK_HISTORY_MAX (rolling mean of last 3)

        # Frequency bands (Hz)
        self.bands = {
            "Delta": (0.5, 4),
            "Theta": (4,   8),
            "Alpha": (8,  13),
            "Beta":  (13, 30),
            "Gamma": (30, 80),
        }

        # Pre-ictal band markers
        self._seizure_log: list[dict] = []   # recent seizure detections
        self._window_count = 0

    # ═══════════════════════════════════════════════════════════
    #  Core analysis
    # ═══════════════════════════════════════════════════════════

    def analyze_window(self, window_data: np.ndarray,
                       patient_base_risk: float = 1.0) -> dict:
        """
        Analyze an 18-channel EEG window.

        Returns
        -------
        dict — risk_score, classification, band_powers, features, alert_channels, ...
        """
        self._window_count += 1
        n_ch = window_data.shape[0]

        # ── Feature extraction ────────────────────────────────────
        all_band_powers = []
        hjorth_list     = []
        spike_list      = []
        line_length_list= []
        kurt_list       = []
        skew_list       = []

        total_delta_theta = 0.0
        total_beta_gamma  = 0.0
        total_spikes      = 0

        for i in range(n_ch):
            ch = window_data[i, :]

            # Band powers
            powers = self._band_power(ch)
            all_band_powers.append(powers)
            total_delta_theta += powers["Delta"] + powers["Theta"]
            total_beta_gamma  += powers["Beta"]  + powers["Gamma"]

            # Hjorth parameters
            act, mob, comp = self._hjorth(ch)
            hjorth_list.append({"activity": act, "mobility": mob, "complexity": comp})

            # Spike count
            spikes = self._detect_spikes(ch)
            spike_list.append(spikes)
            total_spikes += spikes

            # Line length (high-frequency activity)
            ll = self._line_length(ch)
            line_length_list.append(ll)

            # Distribution stats
            kurt_list.append(float(kurtosis(ch)))
            skew_list.append(float(skew(ch)))

        # ── Risk score ───────────────────────────────────────────
        risk, factors = self._compute_risk(
            n_ch, total_spikes, hjorth_list, total_beta_gamma,
            total_delta_theta, line_length_list, kurt_list, patient_base_risk
        )

        # Temporal smoothing (last 3 values)
        self.risk_history.append(risk)
        if len(self.risk_history) > _RISK_HISTORY_MAX:
            self.risk_history.pop(0)
        smoothed = float(np.mean(self.risk_history[-3:]))

        # Classification
        classification, severity = self._classify(smoothed)

        # Seizure log
        if severity == "seizure":
            self._seizure_log.append({
                "time":       datetime.datetime.now().isoformat(timespec="seconds"),
                "risk":       round(smoothed, 1),
                "window_no":  self._window_count,
            })

        # Top 3 channels by spike count
        alert_ch_idx = sorted(range(n_ch), key=lambda i: spike_list[i], reverse=True)[:3]

        return {
            "risk_score":        round(smoothed, 2),
            "raw_risk":          round(risk, 2),
            "classification":    classification,
            "severity":          severity,
            "band_powers":       all_band_powers,
            "hjorth":            hjorth_list,
            "spikes_per_ch":     spike_list,
            "line_length":       line_length_list,
            "kurtosis":          kurt_list,
            "skewness":          skew_list,
            "alert_channels":    alert_ch_idx,
            "risk_factors":      factors,
            "total_spikes":      total_spikes,
            "avg_spikes":        total_spikes / n_ch,
            "seizure_log":       list(self._seizure_log),
            "window_count":      self._window_count,
        }

    # ════════════════════════════════════════════════════════════
    #  History reset (new patient / manual)
    # ════════════════════════════════════════════════════════════

    def reset_history(self):
        """Reset risk history when a new patient is loaded."""
        self.risk_history  = [0.0] * 30
        self._seizure_log  = []
        self._window_count = 0

    # ════════════════════════════════════════════════════════════
    #  Clinical summary
    # ════════════════════════════════════════════════════════════

    def get_clinical_summary(self, patient_info: dict,
                             last_analysis: dict) -> dict:
        """
        Full clinical summary for PDF reporting.
        """
        risk  = last_analysis.get("risk_score", 0)
        sev   = last_analysis.get("severity", "normal")
        cls   = last_analysis.get("classification", "Normal")

        # Mean band powers (18 channels)
        bp_list = last_analysis.get("band_powers", [])
        avg_bands = {b: 0.0 for b in self.bands}
        if bp_list:
            for ch_p in bp_list:
                for b in self.bands:
                    avg_bands[b] += ch_p.get(b, 0)
            for b in avg_bands:
                avg_bands[b] = round(avg_bands[b] / len(bp_list), 4)

        # Dominant band
        dom_band = max(avg_bands, key=avg_bands.get)

        # Seizure count
        sz_count = len(last_analysis.get("seizure_log", []))

        # Interpretation lines
        interpretation = self._generate_interpretation(
            patient_info, risk, sev, dom_band, avg_bands,
            last_analysis.get("total_spikes", 0), sz_count
        )

        # Recommendations
        recommendations = self._generate_recommendations(sev, risk, patient_info)

        return {
            "patient":           patient_info,
            "risk_score":        risk,
            "max_risk":          round(max(self.risk_history), 2),
            "min_risk":          round(min(self.risk_history), 2),
            "avg_risk_1min":     round(float(np.mean(self.risk_history)), 2),
            "classification":    cls,
            "severity":          sev,
            "dominant_band":     dom_band,
            "avg_band_powers":   avg_bands,
            "total_spikes":      last_analysis.get("total_spikes", 0),
            "seizure_count":     sz_count,
            "seizure_log":       last_analysis.get("seizure_log", []),
            "alert_channels":    last_analysis.get("alert_channels", []),
            "interpretation":    interpretation,
            "recommendations":   recommendations,
            "report_time":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "window_analyzed":   last_analysis.get("window_count", 0),
        }

    # ═══════════════════════════════════════════════════════════
    #  Interpretation and recommendations
    # ═══════════════════════════════════════════════════════════

    def _generate_interpretation(self, pt, risk, severity, dom_band,
                                  avg_bands, total_spikes, sz_count) -> list[str]:
        lines = []
        outcome = pt.get("outcome", "")
        engel   = pt.get("engel", "")
        age     = pt.get("age", "?")
        db      = pt.get("database", "")

        lines.append(
            f"EEG analysis completed for patient {pt.get('id','?')} (age {age}, {db})."
        )

        # Dominant-band narrative
        if dom_band == "Delta":
            lines.append(
                "Dominant delta activity. May reflect deep sleep, sedation, "
                "or pathological slow-wave activity."
            )
        elif dom_band == "Theta":
            lines.append(
                "Dominant theta activity. May relate to drowsiness or mild encephalopathy."
            )
        elif dom_band == "Alpha":
            lines.append(
                "Alpha-dominant pattern — consistent with relaxed wakefulness "
                "or eyes-closed resting state."
            )
        elif dom_band == "Beta":
            lines.append(
                "Elevated beta activity. Consider anxiety, medication effect, "
                "or motor cortex activation."
            )
        elif dom_band == "Gamma":
            lines.append(
                "High-frequency gamma activity observed. May align with "
                "pathological discharges or high cognitive load."
            )

        # Risk narrative (aligned with _classify thresholds)
        if risk >= 66:
            lines.append(
                f"CRITICAL: Risk score {risk:.1f}% exceeds the high-risk threshold. "
                "Clinical review is advised."
            )
        elif risk >= 48:
            lines.append(
                f"WARNING: Risk score {risk:.1f}% — possible pre-ictal features."
            )
        elif risk >= 24:
            lines.append(
                f"CAUTION: Risk score {risk:.1f}% — mildly abnormal EEG pattern."
            )
        else:
            lines.append(
                f"Risk score {risk:.1f}% within normal monitoring limits. "
                "No clear pathology flagged."
            )

        # Spike narrative
        if total_spikes > 50:
            lines.append(
                f"High-amplitude spike activity detected (total spikes {total_spikes}). "
                "May be consistent with epileptiform discharges."
            )
        elif total_spikes > 15:
            lines.append(
                f"Moderate spike activity ({total_spikes} spikes). "
                "Closer clinical monitoring is reasonable."
            )

        # Surgical history narrative
        if "Failed" in outcome:
            lines.append(
                f"History: unsuccessful epilepsy surgery (Engel: {engel}). "
                "Recurrent seizure risk may be elevated."
            )
        elif "Successful" in outcome:
            lines.append(
                f"History: successful surgery (Engel: {engel}). "
                "Current EEG should be interpreted in a post-operative follow-up context."
            )
        elif "No Resection" in outcome or "NR" in outcome:
            lines.append(
                "No resection performed. Monitoring continues under conservative management."
            )

        if sz_count > 0:
            lines.append(
                f"{sz_count} seizure event(s) logged during this session."
            )

        return lines

    def _generate_recommendations(self, severity: str,
                                   risk: float, pt: dict) -> list[str]:
        recs = []
        outcome = pt.get("outcome", "")

        if severity == "seizure":
            recs += [
                ">> Start emergency response protocol.",
                ">> Request neurology consultation.",
                ">> Consider urgent antiseizure medication review.",
                ">> Document seizure duration, spread, and termination.",
            ]
        elif severity == "warning":
            recs += [
                ">> Increase EEG monitoring frequency.",
                ">> Prepare for possible seizure escalation.",
                ">> Review current antiseizure drug dosing.",
            ]
        elif severity == "attention":
            recs += [
                ">> Increase observation frequency.",
                ">> Review triggers (sleep deprivation, stress, adherence).",
            ]
        else:
            recs += [
                ">> Continue routine EEG monitoring.",
                ">> Schedule periodic clinical review.",
            ]

        if "Failed" in outcome and risk > 40:
            recs.append(">> Consider epilepsy surgery re-evaluation consult.")

        recs.append(f">> Next EEG follow-up within: {self._next_followup(risk)}.")
        return recs

    @staticmethod
    def _next_followup(risk: float) -> str:
        if risk >= 66: return "24 hours"
        if risk >= 48: return "1 week"
        if risk >= 24: return "1 month"
        return "3 months"

    # ═══════════════════════════════════════════════════════════
    #  Feature computation
    # ═══════════════════════════════════════════════════════════

    def _band_power(self, data: np.ndarray) -> dict:
        freqs, psd = welch(data, self.sample_rate,
                           nperseg=min(int(self.sample_rate), len(data)))
        total = np.sum(psd)
        powers = {}
        for band, (flo, fhi) in self.bands.items():
            idx = np.logical_and(freqs >= flo, freqs <= fhi)
            powers[band] = float(np.sum(psd[idx]) / total) if total > 0 else 0.0
        return powers

    def _hjorth(self, x: np.ndarray):
        d1 = np.diff(x)
        d2 = np.diff(d1)
        vx = np.var(x)
        v1 = np.var(d1)
        v2 = np.var(d2)
        activity   = float(vx)
        mobility   = float(np.sqrt(v1 / vx)) if vx > 0 else 0.0
        complexity = float((np.sqrt(v2 / v1) / mobility)
                           if v1 > 0 and mobility > 0 else 0.0)
        return activity, mobility, complexity

    def _detect_spikes(self, data: np.ndarray, threshold: float = 80.0) -> int:
        return int(np.sum(np.abs(data) > threshold))

    def _line_length(self, data: np.ndarray) -> float:
        return float(np.sum(np.abs(np.diff(data))))

    def _compute_risk(self, n_ch, total_spikes, hjorth_list,
                      total_beta_gamma, total_delta_theta,
                      line_length_list, kurt_list,
                      base_risk) -> tuple[float, dict]:
        base = 10.0

        # Factor A — spike burden
        factor_spike = min(total_spikes / (n_ch * 2), 1.0) * 50.0

        # Factor B — Hjorth complexity (lower = more synchronized)
        avg_comp = float(np.mean([h["complexity"] for h in hjorth_list]))
        factor_comp = (2.2 - avg_comp) * 20.0 if avg_comp < 2.2 else 0.0

        # Factor C — beta/gamma dominance
        factor_freq = min((total_beta_gamma / n_ch) * 60.0, 35.0)

        # Factor D — line length (sharp transients)
        avg_ll = float(np.mean(line_length_list))
        factor_ll = min(avg_ll / 5000.0 * 15.0, 15.0)

        # Factor E — kurtosis (peaked / burst-like activity)
        avg_kurt = float(np.mean(kurt_list))
        factor_kurt = min(max(avg_kurt - 3.0, 0) * 5.0, 15.0)

        # Delta/theta dominance
        factor_slow = min((total_delta_theta / n_ch) * 10.0, 10.0)

        raw = base + factor_spike + factor_comp + factor_freq + \
              factor_ll + factor_kurt + factor_slow

        # base_risk caps modulation (aligned with ~66% critical threshold)
        capped_risk = min(1.25, max(0.80, base_risk))
        final = max(0.0, min(100.0, raw * capped_risk))

        factors = {
            "spike_factor":    round(factor_spike, 2),
            "complexity_drop": round(factor_comp, 2),
            "freq_factor":     round(factor_freq, 2),
            "line_length":     round(factor_ll, 2),
            "kurtosis":        round(factor_kurt, 2),
            "slow_wave":       round(factor_slow, 2),
        }
        return final, factors

    def _classify(self, score: float) -> tuple[str, str]:
        if score >= 66: return "Critical (Seizure)",  "seizure"
        if score >= 48: return "Warning",            "warning"
        if score >= 24: return "Attention",          "attention"
        return "Normal", "normal"
