"""
verify_patients.py
──────────────────
Validates all sub-* folders and BIDS layout.
Reports consistency between participants.tsv and on-disk folders.
"""
import os
import csv
import glob

from paths import DATA_DIR, PARTICIPANTS_TSV


def check_patients():
    os.makedirs(DATA_DIR, exist_ok=True)
    tsv_path = PARTICIPANTS_TSV
    if not os.path.exists(tsv_path):
        print("[!] participants.tsv not found!")
        return

    tsv_ids = set()
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row.get("participant_id", "").strip()
            if pid:
                tsv_ids.add(pid)

    sub_dirs = [
        d for d in os.listdir(DATA_DIR)
        if d.startswith("sub-") and os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    print(f"\n{'='*60}")
    print(f"  CerebraWatch - patient data validation report")
    print(f"{'='*60}")
    print(f"\nparticipants.tsv: {len(tsv_ids)} patient row(s).")
    print(f"On disk: {len(sub_dirs)} sub-* folder(s).\n")

    ok_count = 0
    missing_ieeg = []
    only_in_dir  = []
    only_in_tsv  = []

    for pid in sorted(tsv_ids):
        sub_dir = os.path.join(DATA_DIR, pid)
        if not os.path.isdir(sub_dir):
            only_in_tsv.append(pid)
            continue

        vhdr_files = glob.glob(os.path.join(sub_dir, "ses-*", "ieeg", "*.vhdr")) + \
                     glob.glob(os.path.join(sub_dir, "ieeg", "*.vhdr"))
        edf_files = glob.glob(os.path.join(sub_dir, "ses-*", "ieeg", "*.edf")) + \
                    glob.glob(os.path.join(sub_dir, "ieeg", "*.edf"))
        if not vhdr_files and not edf_files:
            missing_ieeg.append(pid)
        else:
            ok_count += 1
            total_eeg_mb = 0.0
            for v in vhdr_files:
                for ext in (".eeg", ".dat", ".bin"):
                    eeg_file = os.path.splitext(v)[0] + ext
                    if os.path.isfile(eeg_file):
                        total_eeg_mb += os.path.getsize(eeg_file) / (1024**2)
            for f in edf_files:
                if os.path.isfile(f):
                    total_eeg_mb += os.path.getsize(f) / (1024**2)
            tag = []
            if vhdr_files:
                tag.append(f"BV:{len(vhdr_files)}")
            if edf_files:
                tag.append(f"EDF:{len(edf_files)}")
            print(f"  [OK] {pid:<20} | {' '.join(tag):<14} | {total_eeg_mb:>7.1f} MB")

    for d in sorted(sub_dirs):
        if d not in tsv_ids:
            only_in_dir.append(d)

    print(f"\n{'-'*60}")
    print(f"  Real EEG data present : {ok_count} patient(s)")
    print(f"  No ieeg (.vhdr/.edf)  : {len(missing_ieeg)} patient(s)")
    if missing_ieeg:
        for p in missing_ieeg:
            print(f"    [!] {p}")
    print(f"  TSV only (no folder)  : {len(only_in_tsv)} patient(s)")
    if only_in_tsv:
        for p in only_in_tsv:
            print(f"    [i] {p}")
    print(f"  Folder only (no TSV)  : {len(only_in_dir)} entry(ies)")
    if only_in_dir:
        for d in only_in_dir:
            print(f"    [i] {d}")
    print(f"{'='*60}\n")

    if ok_count == 0:
        print("[!] No real EEG data - system will run in simulation mode.")
    else:
        print(f"[OK] Real EEG data ready for {ok_count} patient(s).")
        print("   If MNE is installed, these recordings are used automatically.")

if __name__ == "__main__":
    check_patients()
