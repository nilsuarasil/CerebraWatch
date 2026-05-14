# CerebraWatch PRO

Desktop clinical EEG monitor with AI-assisted risk scoring and PDF reporting. Built with **Python**, **Tkinter**, and **Matplotlib**. Supports **real iEEG/ECoG** (BrainVision and EDF via MNE) and a **simulated** patient mode when no recordings are available.

All user-facing strings in the application and generated PDFs are **English**.

---

## Features

### Live EEG

- 18-channel bipolar montage, clinical-style layout  
- Scrolling waveforms, grid timing (1 s major / 200 ms minor)  
- Data source indicator: real BrainVision/EDF vs simulated  

### Patients

- Loads cohort from `data/participants.tsv` and `data/sub-*` folders  
- **Next Patient** cycles the queue; EEG loads in a background thread  
- Falls back to simulated patients if TSV or folders are missing  

### Real EEG (MNE)

Loader priority per subject:

1. **BrainVision** — largest non-empty companion (`.eeg`, `.dat`, or `.bin`) next to `.vhdr`  
2. **EDF** — largest `*.edf` under `ieeg/` if BrainVision is missing or fails  

Layouts:

```text
data/sub-<id>/ses-*/ieeg/*.vhdr
data/sub-<id>/ieeg/*.vhdr
data/sub-<id>/ses-*/ieeg/*.edf
data/sub-<id>/ieeg/*.edf
```

- Resampling to **256 Hz**, basic filtering and notch (60 Hz)  
- Streaming via `next_samples()` for smooth scrolling  
- Amplitude normalization for stable UI and risk metrics  

### MRI viewer

- Reads `data/mri/participants.tsv` and `data/mri/<id>/anat/*.nii.gz` (T1w / FLAIR)  
- Picks an eligible subject from a **sex/outcome-filtered pool**, then **randomizes** among candidates each time the viewer opens (and among large volumes when several NIfTI files exist)  
- Axial / coronal / sagittal views, slice scrub and playback  
- Requires **nibabel**  

### AI analysis and reports

- Band powers (Welch), Hjorth-style features, spike heuristics, line length, kurtosis  
- Smoothed risk score and tiered classification  
- Live spectral bars, clinical notes panel, status line  
- **4-page PDF** under `data/reports/` — text styled for print (**black body copy**, light header strips)  
- **Analysis**, **Archives**, **Reports** windows from the top navigation  

### Simulation

- State machine: **Normal → Attention → Warning → Seizure** with configurable dynamics  
- When not using real EEG, displayed risk is aligned with the simulation state so alerts stay visible  

---

## Requirements

- **Python 3.10+** recommended  

| Package        | Role                                      |
| -------------- | ----------------------------------------- |
| `numpy`, `scipy` | Signal processing, statistics           |
| `matplotlib`   | EEG, spectrum, PDF figures                |
| `Pillow`       | MRI slice export, imaging helpers       |
| `mne`          | BrainVision + EDF EEG I/O               |
| `nibabel`      | NIfTI MRI                               |

---

## Installation

```bash
git clone <your-repo-url>
cd CerebraWatch
pip install -r requirements.txt
```

### Windows

Double-click or run:

```bat
setup_and_run.bat
```

The batch file uses `cd /d "%~dp0"` so it runs from its own folder (no path edit required).

---

## Run

From the repository root (where `main.py` lives):

```bash
python main.py
```

`main.py` adds `src/` to `sys.path`, sets the working directory to the project root, and ensures `data/` exists. The live session auto-starts shortly after the window opens (configurable in code).

For development you can also set `PYTHONPATH=src` and import from `src/`.

---

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `src/` | Application code (`dashboard.py`, `data_engine.py`, `paths.py`, …) |
| `main.py` | Launcher: `sys.path`, working directory, starts the UI |
| `data/participants.tsv` | EEG cohort metadata (`participant_id`, `sex`, `outcome`, `site`, …) |
| `data/sub-<id>/` | Per-subject BIDS-style tree with `ses-*/ieeg/` or `ieeg/` |
| `data/mri/` | Optional MRI sidecar (`participants.tsv`, `sub-*/anat/*.nii.gz`) |
| `data/reports/` | Exported PDFs (created on demand) |

### EEG files

BrainVision triplets are typical:

- `.vhdr` — header  
- `.eeg` / `.dat` — binary signal  
- `.vmrk` — markers (optional for display)  

EDF files in the same `ieeg/` folders are used when BrainVision is not usable.

### MRI files

- `data/mri/participants.tsv` (UTF-8 or UTF-8 BOM)  
- `data/mri/sub-<id>/anat/*T1w*.nii.gz`, `*FLAIR*.nii.gz` — prefer real files over tiny git-annex placeholders  

---

## Validation

Check TSV rows against `data/sub-*` folders and list BrainVision / EDF coverage:

```bash
python src/verify_patients.py
```

---

## Core modules

| File | Role |
| ---- | ---- |
| `main.py` | Launcher: path setup, working directory, Tk entrypoint |
| `src/paths.py` | `PROJECT_ROOT`, `DATA_DIR`, `MRI_DIR`, `REPORTS_DIR` |
| `src/dashboard.py` | Main window, EEG loop, analytics, PDF, MRI launcher |
| `src/data_engine.py` | Patients, real/sim EEG routing, simulation state machine |
| `src/eeg_loader.py` | MNE BrainVision + EDF → NumPy stream |
| `src/ai_analyzer.py` | Features, risk, classification, clinical summary |
| `src/mri_viewer.py` | MRI database match, NIfTI load, Tk viewer |
| `src/report_generator.py` | Multi-page PDF export |
| `src/verify_patients.py` | Folder vs TSV sanity check |

---

## Troubleshooting

### Status shows simulated EEG

- No `.vhdr` (with data) and no `.edf` under `ieeg/` for that subject  
- **MNE** not installed: `pip install mne`  

### MRI window empty or errors

- Add real `.nii.gz` under `data/mri/<id>/anat/`  
- Install **nibabel**: `pip install nibabel`  

### Waveforms look flat or clipped

The UI uses a fixed µV-style scale; the loader normalizes variance for display. Very quiet or very loud recordings may need dataset-specific tuning in `src/eeg_loader.py`.

---

## GitHub and raw data

This repo is intended to hold **source code only**. The root `.gitignore` already excludes:

- `data/sub-*/` subject folders (EEG + BIDS sidecars)  
- `data/mri/sub-*/` anatomical volumes  
- Raw EEG (`.vhdr`, `.eeg`, `.dat`, `.edf`, `.vmrk`) and NIfTI (`.nii`, `.nii.gz`) anywhere matched by those rules  
- Generated `data/reports/`  

GitHub blocks files **> 100 MB**; large histories also make clones painful. **Do not** `git add -f` those paths.

**For collaborators:** after `git clone`, download the public dataset you use (e.g. from [OpenNeuro](https://openneuro.org)) under the terms of that dataset’s license, extract or symlink it so `data/participants.tsv` and `data/sub-*` (and optional `data/mri/`) match the layout above, then run `python src/verify_patients.py` and `python main.py`.

Before every push, run `git status` and confirm no large or sensitive files are listed as *new file*.

`data/participants.tsv` is **not** ignored: only commit it if IDs and fields are appropriate for public hosting under your data agreement.

---

## License and data

**Software (this repo’s code):** [MIT License](LICENSE) — use, change, and share freely; no warranty.

**Data:** EEG, MRI, and cohort tables are **not** covered by that license; they follow their **original dataset** terms. Do not redistribute clinical or identifiable data without permission.
