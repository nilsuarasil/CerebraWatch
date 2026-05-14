import numpy as np
import random
import os
import csv
import threading

from paths import DATA_DIR, PARTICIPANTS_TSV

SITE_NAMES = {
    "JHH":  "Johns Hopkins Hospital",
    "NIH":  "National Inst. of Health",
    "UMF":  "U. of Miami Florida",
    "UMMC": "U. of Maryland Med. Ctr.",
    "CC":   "Cleveland Clinic",
}
GENDER_MAP   = {"M": "Male", "F": "Female"}
ENGEL_DESC   = {"1.0":"Seizure Free","2.0":"Sig. Reduction","3.0":"Slight Reduction",
                "4.0":"No Change","-1.0":"No Surgery"}
OUTCOME_DESC = {"S":"Successful Surgery","F":"Failed Surgery","NR":"No Resection"}


def _load_patients_from_tsv(tsv_path: str) -> list:
    patients = []
    if not os.path.exists(tsv_path):
        return patients
    try:
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pid = row.get("participant_id","").strip()
                if not pid:
                    continue
                try:    age = int(float(row.get("age","30")))
                except: age = random.randint(18,65)

                sex     = row.get("sex","").strip()
                gender  = GENDER_MAP.get(sex, "Unknown")
                site    = row.get("site","").strip().upper()
                database= SITE_NAMES.get(site, site or "Unknown DB")
                out_raw = row.get("outcome","NR").strip()
                outcome = OUTCOME_DESC.get(out_raw, out_raw)

                try:    engel_key = str(float(row.get("engel_score","-1.0")))
                except: engel_key = "-1.0"
                engel   = ENGEL_DESC.get(engel_key, engel_key)

                try:    ilae = float(row.get("ilae_score",-1))
                except: ilae = -1.0
                try:    follow = float(row.get("years_follow_up",0))
                except: follow = 0.0

                risk_mod = {"S":0.80,"F":1.40,"NR":1.20}.get(out_raw, 1.0)
                base_risk= round(random.uniform(0.7, 1.3) * risk_mod, 3)

                sub_dir = os.path.join(DATA_DIR, pid)
                patients.append({
                    "id":         pid,
                    "age":        age,
                    "gender":     gender,
                    "database":   database,
                    "site":       site,
                    "outcome":    outcome,
                    "engel":      engel,
                    "ilae":       ilae,
                    "follow_up":  follow,
                    "base_risk":  base_risk,
                    "has_eeg":    os.path.isdir(sub_dir),
                    "sub_dir":    sub_dir,
                    "state":       "normal",
                    "state_ticks": 0,
                })
    except Exception as e:
        print(f"[DataEngine] TSV read error: {e}")
    return patients


class DataEngine:
    def __init__(self, sample_rate=256, window_size_sec=2):
        self.sample_rate        = sample_rate
        self.window_size_sec    = window_size_sec
        self.samples_per_window = sample_rate * window_size_sec

        self.channels = [
            "Fp1-F7","F7-T3","T3-T5","T5-O1",
            "Fp1-F3","F3-C3","C3-P3","P3-O1",
            "Fp2-F8","F8-T4","T4-T6","T6-O2",
            "Fp2-F4","F4-C4","C4-P4","P4-O2",
            "Fz-Cz","Cz-Pz",
        ]
        self.num_channels = len(self.channels)

        os.makedirs(DATA_DIR, exist_ok=True)
        tsv = PARTICIPANTS_TSV
        self._patients = _load_patients_from_tsv(tsv)

        if not self._patients:
            print("[DataEngine] WARNING: participants.tsv missing/unreadable, using simulation.")
            self._patients = self._simulated(20)
        else:
            print(f"[DataEngine] Loaded {len(self._patients)} patients from TSV.")

        self._idx = random.randint(0, len(self._patients)-1)
        self.current_patient = {}
        self._eeg_lock    = threading.Lock()
        self._eeg_loader  = None
        self._eeg_loading = False
        self.generate_new_patient()

    # ── Public API ──────────────────────────────────────────────

    def generate_new_patient(self):
        self._idx = (self._idx + 1) % len(self._patients)
        self.current_patient = dict(self._patients[self._idx])
        self.current_patient["state"]       = "normal"
        self.current_patient["state_ticks"] = 0
        # Clear old loader; load new one in background
        with self._eeg_lock:
            self._eeg_loader = None
        self._try_load_eeg(self.current_patient)

    def get_patient_info(self) -> dict:
        return self.current_patient

    def get_all_patients(self) -> list:
        return self._patients

    def get_patient_count(self) -> int:
        return len(self._patients)

    @property
    def eeg_frame_samples(self) -> int:
        """EEG samples per UI frame (~0.1 s @ sample_rate); must match dashboard."""
        return max(1, int(self.sample_rate * 0.1))

    # ── EEGLoader integration (background thread) ───────────────────────

    def _try_load_eeg(self, patient: dict):
        """
        Start EEG load in a background thread (UI stays responsive).
        On completion, assigns self._eeg_loader if the same patient is still selected.
        """
        if not patient.get("has_eeg", False):
            return
        sub_dir = patient.get("sub_dir", "")
        if not sub_dir or not os.path.isdir(sub_dir):
            return
        pid = patient["id"]

        def _worker():
            try:
                from eeg_loader import EEGLoader
                loader = EEGLoader(sub_dir)
                if loader.available:
                    with self._eeg_lock:
                        # Assign only if this patient is still selected
                        if self.current_patient.get("id") == pid:
                            self._eeg_loader = loader
                            print(f"[DataEngine] Real EEG ready: {pid} "
                                  f"({loader.n_channels} ch, "
                                  f"{loader.duration_sec:.0f} s)")
            except Exception as e:
                print(f"[DataEngine] EEG thread error ({pid}): {e}")
            finally:
                self._eeg_loading = False

        self._eeg_loading = True
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── EEG Generation ─────────────────────────────────────────

    def _pink_noise(self, n):
        w = np.random.randn(n)
        X = np.fft.rfft(w)
        S = np.fft.rfftfreq(n)
        S[0] = S[1]
        X /= np.sqrt(S)
        p = np.fft.irfft(X, n=n)
        return (p - p.mean()) / p.std()

    def _rhythm(self, freqs, amps, n):
        t = np.linspace(0, self.window_size_sec, n, endpoint=False)
        s = np.zeros(n)
        for f, a in zip(freqs, amps):
            s += a * np.sin(2*np.pi*(f+random.uniform(-0.5,0.5))*t
                            + random.uniform(0, 2*np.pi))
        return s

    def _noise_layer(self, w: np.ndarray) -> np.ndarray:
        return w + (np.random.randn(*w.shape) * 0.72).astype(np.float32)

    def generate_window(self):
        """
        Return one EEG window for the current patient.
        Uses real data when available; otherwise simulated data.
        Returns: (data: np.ndarray shape (18, samples), state: str)
        """
        self.current_patient["state_ticks"] += 1
        state = self._tick_state()

        # ── Real EEG (thread-safe) ───────────────────────────────
        with self._eeg_lock:
            loader = self._eeg_loader

        if loader is not None and loader.available:
            # Dashboard consumes eeg_frame_samples per frame only.
            frame_n = self.eeg_frame_samples
            window = loader.next_samples(frame_n)
            if window is not None:
                n_ch, n_samp = window.shape
                if n_ch < self.num_channels:
                    pad = np.zeros((self.num_channels - n_ch, n_samp), dtype=np.float32)
                    window = np.vstack([window, pad])
                elif n_ch > self.num_channels:
                    window = window[:self.num_channels, :]
                return self._noise_layer(window), state

        # ── Simulated EEG ───────────────────────────────────────
        return self._noise_layer(self._simulate_window(state)), state

    def _simulate_window(self, state):
        data = np.zeros((self.num_channels, self.samples_per_window))
        for i in range(self.num_channels):
            amp = 15.0 if state == "normal" else 20.0
            ch  = self._pink_noise(self.samples_per_window) * amp

            if state == "normal":
                ch += self._rhythm([2,5,10,20],[5,5,15,3], self.samples_per_window)
                if i in [0,4,8,12] and random.random() < 0.3:
                    bp = random.randint(0, self.samples_per_window - int(0.5*self.sample_rate))
                    bl = np.sin(np.linspace(0,np.pi,int(0.5*self.sample_rate)))*60
                    ch[bp:bp+len(bl)] += bl
            elif state == "attention":
                ch += self._rhythm([3,6,9,18,25],[8,10,8,8,5], self.samples_per_window)
            elif state == "warning":
                ch += self._rhythm([1.5,4,15,25,40],[15,12,5,15,10], self.samples_per_window)
                if random.random() < 0.4:
                    sp = random.randint(0, self.samples_per_window-20)
                    ch[sp:sp+5]    += 80
                    ch[sp+5:sp+15] -= 60
            elif state == "seizure":
                sf = random.uniform(3,7)
                ch += self._rhythm([sf,sf*2,40,60],[45,20,25,15], self.samples_per_window)
                spk = np.random.randn(self.samples_per_window)
                spk[spk < 2.5] = 0
                ch += spk * 100

            ch += np.random.randn(self.samples_per_window) * 4.6
            data[i] = ch
        return data

    def _tick_state(self):
        p = self.current_patient
        t = p["state_ticks"]
        r = p["base_risk"]
        s = p["state"]

        if s == "normal":
            # ~80 ms/tick per UI frame
            if t > 24 or (t > 4 and random.random() < 0.38 * min(r, 1.4)):
                p["state"], p["state_ticks"] = "attention", 0
        elif s == "attention":
            if t > 4:
                p["state"]       = "warning" if random.random() < 0.98 else "normal"
                p["state_ticks"] = 0
        elif s == "warning":
            if t > 4:
                p["state"]       = "seizure" if random.random() < 1.0 else "attention"
                p["state_ticks"] = 0
        elif s == "seizure":
            if t > 22:
                p["state"], p["state_ticks"] = "normal", 0

        return p["state"]


    def _simulated(self, n):
        dbs = ["Kaggle AES","PhysioNet EPILEPSIAE","TUH EEG Corpus","CHB-MIT"]
        return [{
            "id": f"SIM-{1000+i}", "age": random.randint(8,75),
            "gender": random.choice(["Male","Female"]),
            "database": random.choice(dbs), "site":"SIM",
            "outcome":"Simulated","engel":"N/A","ilae":-1.0,"follow_up":0.0,
            "base_risk": round(random.uniform(0.5,1.5),3),
            "has_eeg": False, "sub_dir": "", "state":"normal","state_ticks":0,
        } for i in range(n)]
