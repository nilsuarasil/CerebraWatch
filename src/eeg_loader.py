"""
eeg_loader.py
─────────────
Loads real iEEG/ECoG with MNE: BrainVision (.vhdr) or European Data Format (.edf).
Requires: pip install mne
"""
import os
import glob
import numpy as np
import random

# Target output sample rate (match simulator)
TARGET_SRATE    = 256
# Window length returned per call (seconds)
WINDOW_SEC      = 2
TARGET_CHANNELS = 18  # dashboard expects 18 channels


def _try_import_mne():
    try:
        import mne
        mne.set_log_level("ERROR")
        return mne
    except ImportError:
        return None


def find_vhdr_files(sub_dir: str) -> list:
    """
    Find all ses-*/ieeg/*.vhdr under BIDS sub_dir.
    """
    pattern_bids = os.path.join(sub_dir, "ses-*", "ieeg", "*.vhdr")
    pattern_flat = os.path.join(sub_dir, "ieeg", "*.vhdr")
    files = sorted(glob.glob(pattern_bids)) + sorted(glob.glob(pattern_flat))
    seen = set()
    result = []
    for f in files:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def find_edf_files(sub_dir: str) -> list:
    """Find all ses-*/ieeg/*.edf (and flat ieeg/*.edf) under sub_dir."""
    pattern_bids = os.path.join(sub_dir, "ses-*", "ieeg", "*.edf")
    pattern_flat = os.path.join(sub_dir, "ieeg", "*.edf")
    files = sorted(glob.glob(pattern_bids)) + sorted(glob.glob(pattern_flat))
    seen = set()
    result = []
    for f in files:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def _safe_eeg_size(vhdr_path: str) -> int:
    """Size of the .eeg/.dat file paired with .vhdr (0 if missing)."""
    for ext in (".eeg", ".dat", ".bin"):
        candidate = os.path.splitext(vhdr_path)[0] + ext
        if os.path.isfile(candidate):
            try:
                return os.path.getsize(candidate)
            except OSError:
                return 0
    return 0


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path) if os.path.isfile(path) else 0
    except OSError:
        return 0


class EEGLoader:
    """
    Loads one subject folder: prefers BrainVision with non-empty binary,
    then falls back to largest .edf in ieeg/.
    """

    def __init__(self, sub_dir: str):
        self.sub_dir   = sub_dir
        self.available = False
        self._raw      = None
        self._data     = None
        self._srate    = TARGET_SRATE
        self._cursor   = 0
        self._win_samples = 0
        self.source_file = ""

        mne = _try_import_mne()
        if mne is None:
            print("[EEGLoader] MNE not installed — pip install mne")
            return

        vhdr_files = find_vhdr_files(sub_dir)
        edf_files  = find_edf_files(sub_dir)
        if not vhdr_files and not edf_files:
            return

        last_err = None

        for vhdr_path in sorted(vhdr_files, key=_safe_eeg_size, reverse=True):
            if _safe_eeg_size(vhdr_path) <= 0:
                continue
            try:
                raw = mne.io.read_raw_brainvision(
                    vhdr_path, preload=True, verbose=False
                )
                self._ingest_raw(raw, mne, os.path.basename(vhdr_path))
                return
            except Exception as e:
                last_err = e

        for edf_path in sorted(edf_files, key=_file_size, reverse=True):
            if _file_size(edf_path) <= 0:
                continue
            try:
                raw = mne.io.read_raw_edf(
                    edf_path, preload=True, verbose=False, stim_channel=False
                )
                self._ingest_raw(raw, mne, os.path.basename(edf_path))
                return
            except Exception as e:
                last_err = e

        if last_err is not None:
            print(f"[EEGLoader] {os.path.basename(sub_dir)} load failed: {last_err}")

    def _ingest_raw(self, raw, mne, source_basename: str) -> None:
        # ECoG / SEEG / EEG picks (EDF is often typed as eeg)
        picks = mne.pick_types(raw.info, ecog=True, exclude="bads")
        if len(picks) == 0:
            picks = mne.pick_types(raw.info, seeg=True, exclude="bads")
        if len(picks) == 0:
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(picks) == 0:
            picks = mne.pick_types(raw.info, misc=False, exclude="bads")
        if len(picks) == 0:
            picks = list(range(min(TARGET_CHANNELS, len(raw.ch_names))))
        raw.pick(picks)

        orig_srate = raw.info["sfreq"]
        if int(orig_srate) != TARGET_SRATE:
            raw.resample(TARGET_SRATE, npad="auto", verbose=False)

        raw.filter(0.5, 100.0, method="iir", verbose=False)
        raw.notch_filter(60.0, verbose=False)

        data = raw.get_data()

        if data.shape[0] >= TARGET_CHANNELS:
            data = data[:TARGET_CHANNELS, :]
        else:
            pad = np.zeros((TARGET_CHANNELS - data.shape[0], data.shape[1]),
                           dtype=np.float32)
            data = np.vstack([data, pad])

        data = data * 1e6  # V → µV

        global_std = np.std(data)
        if global_std > 0:
            data = (data - np.mean(data)) * (15.0 / global_std)

        self._data = data.astype(np.float32)
        self._srate = TARGET_SRATE
        self._win_samples = int(TARGET_SRATE * WINDOW_SEC)
        max_start = max(0, self._data.shape[1] // 4)
        self._cursor = random.randint(0, max_start)
        self.available = True
        self.source_file = source_basename
        print(f"[EEGLoader] OK {os.path.basename(self.sub_dir)} | "
              f"{self.n_channels} ch | "
              f"{self.duration_sec:.0f} s | "
              f"{self.source_file}")

    def next_samples(self, n: int) -> np.ndarray | None:
        if not self.available or self._data is None or n <= 0:
            return None
        n = int(n)
        n_total = self._data.shape[1]
        if n_total == 0:
            return None
        out = np.empty((TARGET_CHANNELS, n), dtype=np.float32)
        pos = 0
        while pos < n:
            if self._cursor >= n_total:
                self._cursor = 0
            take = min(n - pos, n_total - self._cursor)
            out[:, pos : pos + take] = self._data[:, self._cursor : self._cursor + take]
            self._cursor += take
            pos += take
        return out

    def next_window(self) -> np.ndarray:
        if not self.available or self._data is None:
            return None

        n_samples = self._data.shape[1]
        end = self._cursor + self._win_samples

        if end > n_samples:
            self._cursor = 0
            end = self._win_samples

        if end > n_samples:
            return None

        window = self._data[:, self._cursor:end].copy()
        self._cursor = end
        return window

    @property
    def sample_rate(self) -> int:
        return self._srate

    @property
    def duration_sec(self) -> float:
        if self._data is None:
            return 0.0
        return self._data.shape[1] / self._srate

    @property
    def n_channels(self) -> int:
        if self._data is None:
            return 0
        return self._data.shape[0]
