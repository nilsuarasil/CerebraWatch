"""
mri_viewer.py
─────────────
Real-brain MRI (NIfTI T1w / FLAIR) viewer.
Picks an eligible MRI from mri/ (sex/outcome-filtered pool); a new random case each time the viewer opens.
Three planes (Axial / Coronal / Sagittal), animated scrub.
"""
import os
import csv
import random
import threading
import tkinter as tk
import numpy as np

from paths import MRI_DIR


# ── nibabel loader ───────────────────────────────────────────────────
def _try_nibabel():
    try:
        import nibabel as nib
        return nib
    except ImportError:
        return None


# ── MRI database ─────────────────────────────────────────────────────
class MRIDatabase:
    """Reads mri/participants.tsv and maps .nii.gz paths."""

    def __init__(self, mri_dir: str = MRI_DIR):
        self.mri_dir = mri_dir
        self.patients = []
        self._load()

    def _load(self):
        tsv = os.path.join(self.mri_dir, "participants.tsv")
        if not os.path.isfile(tsv):
            return
        try:
            with open(tsv, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    pid = row.get("participant_id", "").strip()
                    if not pid:
                        continue
                    anat_dir = os.path.join(self.mri_dir, pid, "anat")
                    t1    = self._find_modality(anat_dir, "T1w")
                    flair = self._find_modality(anat_dir, "FLAIR")
                    try:
                        age = int(float(row.get("age_scan", "10")))
                    except Exception:
                        age = 10
                    self.patients.append({
                        "id":       pid,
                        "group":    row.get("group", "fcd"),
                        "sex":      row.get("sex", "M"),
                        "age":      age,
                        "diagnosis": row.get("mri_diagnosis", "suspicion"),
                        "lobe":     row.get("lobe", "FL"),
                        "hemisphere": row.get("hemisphere", "L"),
                        "outcome":  row.get("1year_outcome", "n/a"),
                        "t1_path":  t1,
                        "flair_path": flair,
                    })
        except Exception as e:
            print(f"[MRIDatabase] {e}")

    @staticmethod
    def _find_modality(anat_dir: str, modality: str) -> str | None:
        """
        In BIDS anat/, find *T1w*.nii.gz or *FLAIR*.nii.gz (case-insensitive).
        Picks the largest file (datalad pointer files are usually tiny).
        """
        if not os.path.isdir(anat_dir):
            return None
        best_path = None
        best_size = -1
        for name in os.listdir(anat_dir):
            low = name.lower()
            if not low.endswith(".nii.gz"):
                continue
            if modality == "T1w":
                if "t1w" not in low:
                    continue
            else:
                if "flair" not in low:
                    continue
            full = os.path.join(anat_dir, name)
            try:
                sz = os.path.getsize(full)
            except OSError:
                continue
            if sz > best_size:
                best_size = sz
                best_path = full
        return best_path

    def find_match(self, eeg_info: dict) -> dict | None:
        """Eligible MRI subject (random pick from filtered pool on each open)."""
        has_scan = [p for p in self.patients if p.get("t1_path") or p.get("flair_path")]
        if not has_scan:
            return None
        gender = eeg_info.get("gender", "Male")
        sex = "M" if "Male" in gender else "F"
        outcome = eeg_info.get("outcome", "")
        prefer_fcd = True

        pool = [p for p in has_scan if p["sex"] == sex and
                (p["group"] == "fcd") == prefer_fcd]
        if not pool:
            pool = has_scan[:]

        if "Successful" in outcome:
            good = [p for p in pool if p["outcome"] in ("IA", "IB")]
            if good:
                pool = good
        elif "Failed" in outcome:
            bad = [p for p in pool if p["outcome"] in ("IVB", "IVC", "IIIA")]
            if bad:
                pool = bad

        if not pool:
            pool = has_scan[:]

        if len(pool) == 1 and len(has_scan) > 1:
            pool = has_scan[:]

        return random.choice(pool)


# ── MRI volume loader ────────────────────────────────────────────────
class MRILoader:
    """Loads one NIfTI file into a numpy array."""

    def __init__(self, path: str):
        self.path    = path
        self.volume  = None   # (X, Y, Z) float32
        self.loaded  = False
        self._load()

    def _load(self):
        nib = _try_nibabel()
        if nib is None:
            print("[MRILoader] nibabel not installed — pip install nibabel")
            return
        try:
            img = nib.load(self.path)
            data = np.asanyarray(img.dataobj).astype(np.float32)
            # Orientation for axial view (X, Y, Z)
            data = np.squeeze(data)
            if data.ndim == 4:
                data = data[..., 0]
            self.volume = data
            self.loaded = True
        except Exception as e:
            print(f"[MRILoader] {e}")

    @property
    def n_slices(self):
        return self.volume.shape[2] if self.loaded else 0

    def axial(self, z):
        sl = self.volume[:, :, z]
        return self._window(sl)

    def coronal(self, y):
        sl = self.volume[:, y, :]
        return self._window(sl)

    def sagittal(self, x):
        sl = self.volume[x, :, :]
        return self._window(sl)

    @staticmethod
    def _window(sl, w=400, l=200):
        lo = l - w / 2
        hi = l + w / 2
        p2, p98 = np.percentile(sl, 2), np.percentile(sl, 98)
        if p98 > p2:
            lo = p2; hi = p98
        sl = np.clip(sl, lo, hi)
        sl = ((sl - lo) / (hi - lo) * 255).astype(np.uint8)
        return sl


# ── slice → PhotoImage ───────────────────────────────────────────────
def _slice_to_photo(arr2d, w, h):
    """numpy (H,W) uint8 → tk.PhotoImage (grayscale)."""
    from PIL import Image, ImageTk
    img = Image.fromarray(arr2d).convert("L")
    img = img.resize((w, h), Image.BILINEAR)
    img = img.rotate(90, expand=True)  # anatomical orientation
    return ImageTk.PhotoImage(img)


# ── main viewer window ───────────────────────────────────────────────
class MRIViewerWindow:
    BG   = "#000000"
    GOLD = "#F59E0B"
    DIM  = "#4B5563"
    TXT  = "#E2E8F0"
    GRN  = "#22C55E"

    def __init__(self, parent, eeg_info: dict):
        self.win = tk.Toplevel(parent)
        self.win.title("CerebraWatch — MRI/FLAIR Viewer")
        self.win.geometry("1100x720")
        self.win.configure(bg=self.BG)
        self.win.resizable(True, True)

        self.loader: MRILoader | None = None
        self.mri_info  = {}
        self._z        = 0
        self._playing  = False
        self._after_id = None
        self._modality = "T1w"
        self._photo_ax = self._photo_cor = self._photo_sag = None
        self._mri_cfg_job = None

        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui(eeg_info)
        self._load_async(eeg_info)

    def _build_ui(self, eeg_info):
        # top info bar
        top = tk.Frame(self.win, bg="#0A0F1C", height=40)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        self.lbl_title = tk.Label(
            top, text="MRI / FLAIR  —  Loading...",
            bg="#0A0F1C", fg=self.GOLD, font=("Arial", 11, "bold"))
        self.lbl_title.pack(side=tk.LEFT, padx=14, pady=8)

        self.lbl_mod = tk.Label(
            top, text="", bg="#0A0F1C", fg=self.GRN,
            font=("Arial", 9, "bold"))
        self.lbl_mod.pack(side=tk.RIGHT, padx=14)

        # image area — grid: axial expands; coronal/sagittal share right column
        img_frame = tk.Frame(self.win, bg=self.BG)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=0, minsize=300)
        img_frame.rowconfigure(0, weight=1)

        left = tk.Frame(img_frame, bg="#050810",
                        highlightbackground="#1E3A5F", highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)

        self.lbl_ax_info = tk.Label(
            left, text="AXIAL", bg="#050810", fg=self.GOLD,
            font=("Courier New", 9, "bold"))
        self.lbl_ax_info.pack(anchor="nw", padx=8, pady=4)

        self.cv_ax = tk.Canvas(left, bg="black", highlightthickness=0)
        self.cv_ax.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        right = tk.Frame(img_frame, bg=self.BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(4, weight=1)

        tk.Label(right, text="CORONAL", bg=self.BG,
                 fg=self.DIM, font=("Courier New", 8)).grid(row=0, column=0, sticky="w", padx=6)
        self.cv_cor = tk.Canvas(right, bg="black", highlightthickness=0)
        self.cv_cor.grid(row=1, column=0, sticky="nsew", padx=4)

        tk.Frame(right, bg="#1E3A5F", height=1).grid(row=2, column=0, sticky="ew", pady=3)

        tk.Label(right, text="SAGITTAL", bg=self.BG,
                 fg=self.DIM, font=("Courier New", 8)).grid(row=3, column=0, sticky="w", padx=6)
        self.cv_sag = tk.Canvas(right, bg="black", highlightthickness=0)
        self.cv_sag.grid(row=4, column=0, sticky="nsew", padx=4)

        def _on_img_cfg(_ev):
            if self._mri_cfg_job is not None:
                self.win.after_cancel(self._mri_cfg_job)
            self._mri_cfg_job = self.win.after(48, self._debounced_render)

        img_frame.bind("<Configure>", _on_img_cfg)

        # control bar
        ctrl = tk.Frame(self.win, bg="#0A0F1C", height=54)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)
        ctrl.pack_propagate(False)

        tk.Button(ctrl, text="◀◀", command=self._first,
                  bg="#1E3A5F", fg=self.TXT, font=("Arial", 10),
                  relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=4, pady=10)

        self.btn_play = tk.Button(ctrl, text="▶ Play",
                  command=self._toggle_play,
                  bg=self.GOLD, fg="#000", font=("Arial", 10, "bold"),
                  relief=tk.FLAT, padx=10, pady=4)
        self.btn_play.pack(side=tk.LEFT, padx=4, pady=10)

        tk.Button(ctrl, text="▶▶", command=self._last,
                  bg="#1E3A5F", fg=self.TXT, font=("Arial", 10),
                  relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=4, pady=10)

        # slice slider
        self.slider_var = tk.IntVar(value=0)
        self.slider = tk.Scale(
            ctrl, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.slider_var, command=self._on_slider,
            bg="#0A0F1C", fg=self.TXT, troughcolor="#1E3A5F",
            highlightthickness=0, sliderrelief=tk.FLAT,
            length=350, showvalue=False)
        self.slider.pack(side=tk.LEFT, padx=12, pady=8)

        self.lbl_slice = tk.Label(ctrl, text="— / —",
                                  bg="#0A0F1C", fg=self.TXT,
                                  font=("Courier New", 10))
        self.lbl_slice.pack(side=tk.LEFT, padx=6)

        # modality selector
        self.mod_var = tk.StringVar(value="T1w")
        for m in ("T1w", "FLAIR"):
            tk.Radiobutton(ctrl, text=m, variable=self.mod_var,
                           value=m, command=self._switch_modality,
                           bg="#0A0F1C", fg=self.TXT,
                           selectcolor="#1E3A5F", font=("Arial", 9),
                           activebackground="#0A0F1C").pack(side=tk.RIGHT, padx=8)

        tk.Label(ctrl, text="Modality:", bg="#0A0F1C",
                 fg=self.DIM, font=("Arial", 8)).pack(side=tk.RIGHT)

        # speed
        self.speed_var = tk.IntVar(value=80)
        tk.Scale(ctrl, from_=20, to=300, orient=tk.HORIZONTAL,
                 variable=self.speed_var, label="ms/slice",
                 bg="#0A0F1C", fg=self.DIM, troughcolor="#1E3A5F",
                 highlightthickness=0, sliderrelief=tk.FLAT,
                 length=100, font=("Arial", 7)).pack(side=tk.RIGHT, padx=8)

    def _load_async(self, eeg_info):
        def worker():
            db = MRIDatabase()
            match = db.find_match(eeg_info)
            if match is None:
                self.win.after(0, self._show_failure_no_data)
                return

            paths_to_try: list[tuple[str, str]] = []
            seen: set[str] = set()
            for key, mod in (("t1_path", "T1w"), ("flair_path", "FLAIR")):
                p = match.get(key)
                if p and os.path.isfile(p) and p not in seen:
                    paths_to_try.append((mod, p))
                    seen.add(p)
            anat_dir = os.path.join(db.mri_dir, match["id"], "anat")
            if os.path.isdir(anat_dir):
                for name in os.listdir(anat_dir):
                    low = name.lower()
                    if not low.endswith(".nii.gz"):
                        continue
                    fp = os.path.join(anat_dir, name)
                    if fp in seen:
                        continue
                    mod = "FLAIR" if "flair" in low else ("T1w" if "t1w" in low else "T1w")
                    paths_to_try.append((mod, fp))
                    seen.add(fp)

            def _sz(tp):
                try:
                    return os.path.getsize(tp[1])
                except OSError:
                    return 0

            paths_to_try.sort(key=_sz, reverse=True)
            # Prefer real files over tiny git-annex pointers, but vary which large volume loads first
            big = [p for p in paths_to_try if _sz(p) >= 1_000_000]
            head = big if big else paths_to_try[:]
            tail = [p for p in paths_to_try if p not in head]
            random.shuffle(head)
            paths_to_try = head + tail

            loader = None
            modality = "T1w"
            for mod, path in paths_to_try:
                if not os.path.isfile(path):
                    continue
                L = MRILoader(path)
                if L.loaded:
                    loader, modality = L, mod
                    break

            if loader is not None:
                ld, mcpy, md = loader, dict(match), modality
                self.win.after(0, lambda: self._apply_loaded(ld, mcpy, md))
            else:
                mc = dict(match)
                self.win.after(0, lambda: self._show_failure_nifti(mc))

        threading.Thread(target=worker, daemon=True).start()

    def _debounced_render(self):
        self._mri_cfg_job = None
        try:
            self._render()
        except tk.TclError:
            pass

    def _show_failure_no_data(self):
        self.loader = None
        self.mri_info = {}
        self.lbl_title.config(
            text="No NIfTI: add mri/<patient>/anat/*.nii.gz from the dataset.")
        self.lbl_mod.config(text="")
        self.lbl_ax_info.config(text="")
        self.slider.config(to=0, state="disabled")
        self.lbl_slice.config(text="--")
        self._blank_canvases("MRI TSV present but no .nii.gz files found.")

    def _show_failure_nifti(self, match: dict):
        self.loader = None
        self.mri_info = match
        pid = match.get("id", "?")
        self.lbl_title.config(text=f"{pid}: NIfTI could not be read (nibabel / file).")
        self.lbl_mod.config(text="")
        self.slider.config(to=0, state="disabled")
        self.lbl_slice.config(text="--")
        self._blank_canvases("Corrupt file or nibabel not installed.")

    def _blank_canvases(self, msg: str):
        for cv, w, h in (
            (self.cv_ax, 700, 580),
            (self.cv_cor, 300, 280),
            (self.cv_sag, 300, 280),
        ):
            cv.delete("all")
            cv.create_text(
                w // 2, h // 2, text=msg, fill=self.DIM,
                font=("Arial", 10), width=w - 40, justify=tk.CENTER,
            )

    def _apply_loaded(self, loader: MRILoader, match: dict, modality: str):
        self.loader = loader
        self.mri_info = match
        self._modality = modality
        self.mod_var.set(modality)
        n = max(1, loader.n_slices)
        self._z = n // 2
        self.slider.config(state="normal", to=n - 1)
        self.slider_var.set(self._z)
        m = self.mri_info
        self.lbl_title.config(
            text=f"{m['id']}  |  {str(m.get('group','')).upper()}  |  "
                 f"Lobe: {m['lobe']}  |  Hem: {m['hemisphere']}  |  "
                 f"Dx: {m['diagnosis']}  |  Outcome: {m['outcome']}")
        self.lbl_mod.config(text=f"* {self._modality}")
        self._render()
        self.win.after(300, self._render)

    def _render(self):
        if not self.loader or not self.loader.loaded:
            return
        n = self.loader.n_slices
        z = max(0, min(self._z, n - 1))
        mid_x = self.loader.volume.shape[0] // 2
        mid_y = self.loader.volume.shape[1] // 2

        # Axial
        w = self.cv_ax.winfo_width()  or 700
        h = self.cv_ax.winfo_height() or 580
        sl_ax  = self.loader.axial(z)
        self._photo_ax = _slice_to_photo(sl_ax, w, h)
        self.cv_ax.delete("all")
        self.cv_ax.create_image(w//2, h//2, image=self._photo_ax, anchor=tk.CENTER)
        # crosshair
        self.cv_ax.create_line(w//2, 0, w//2, h, fill="#1E3A5F", width=1)
        self.cv_ax.create_line(0, h//2, w, h//2, fill="#1E3A5F", width=1)
        # overlay
        self.cv_ax.create_text(8, 8, anchor="nw",
            text=f"AXIAL  z={z+1}/{n}\nT:{self.mri_info.get('id','')}\nMR {self._modality}",
            fill=self.GOLD, font=("Courier New", 9))

        # Coronal
        wc = self.cv_cor.winfo_width()  or 300
        hc = self.cv_cor.winfo_height() or 280
        sl_cor = self.loader.coronal(mid_y)
        self._photo_cor = _slice_to_photo(sl_cor, wc, hc)
        self.cv_cor.delete("all")
        self.cv_cor.create_image(wc//2, hc//2, image=self._photo_cor, anchor=tk.CENTER)
        self.cv_cor.create_text(6, 6, anchor="nw", text=f"y={mid_y}",
                                fill=self.DIM, font=("Courier New", 8))

        # Sagittal
        ws = self.cv_sag.winfo_width()  or 300
        hs = self.cv_sag.winfo_height() or 280
        sl_sag = self.loader.sagittal(mid_x)
        self._photo_sag = _slice_to_photo(sl_sag, ws, hs)
        self.cv_sag.delete("all")
        self.cv_sag.create_image(ws//2, hs//2, image=self._photo_sag, anchor=tk.CENTER)
        self.cv_sag.create_text(6, 6, anchor="nw", text=f"x={mid_x}",
                                fill=self.DIM, font=("Courier New", 8))

        self.lbl_slice.config(text=f"{z+1:3d} / {n}")
        self.lbl_ax_info.config(
            text=f"AXIAL  |  {self.mri_info.get('id','')}  |  "
                 f"Lobe: {self.mri_info.get('lobe','')}  "
                 f"Hemisphere: {self.mri_info.get('hemisphere','')}  |  "
                 f"Slice {z+1}/{n}")

    def _on_slider(self, val):
        self._z = int(float(val))
        self._render()

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self.btn_play.config(text="|| Pause", bg="#EF4444")
            self._animate()
        else:
            self.btn_play.config(text="▶ Play", bg=self.GOLD)
            if self._after_id:
                self.win.after_cancel(self._after_id)

    def _animate(self):
        if not self._playing or not self.loader:
            return
        self._z = (self._z + 1) % self.loader.n_slices
        self.slider_var.set(self._z)
        self._render()
        self._after_id = self.win.after(self.speed_var.get(), self._animate)

    def _first(self):
        self._z = 0
        self.slider_var.set(0)
        self._render()

    def _last(self):
        if self.loader:
            self._z = self.loader.n_slices - 1
            self.slider_var.set(self._z)
            self._render()

    def _switch_modality(self):
        if not self.mri_info:
            return
        m = self.mod_var.get()
        path = self.mri_info.get("t1_path") if m == "T1w" \
               else self.mri_info.get("flair_path")
        if not path:
            return
        self._modality = m
        self.lbl_mod.config(text=f"* {m}")
        def worker():
            loader = MRILoader(path)
            if loader.loaded:
                self.loader = loader
                self._z = loader.n_slices // 2
                self.win.after(0, self._render)
        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self):
        self._playing = False
        if self._mri_cfg_job is not None:
            try:
                self.win.after_cancel(self._mri_cfg_job)
            except tk.TclError:
                pass
            self._mri_cfg_job = None
        if self._after_id is not None:
            try:
                self.win.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        self.win.destroy()
