"""
Central paths: code lives in src/, all datasets and outputs under data/.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MRI_DIR = os.path.join(DATA_DIR, "mri")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
PARTICIPANTS_TSV = os.path.join(DATA_DIR, "participants.tsv")
