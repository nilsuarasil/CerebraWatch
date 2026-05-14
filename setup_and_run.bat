@echo off
echo ============================================================
echo  CerebraWatch PRO - setup and launch
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/4] Installing dependencies...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo WARNING: Some packages failed to install.
) else (
    echo OK: Dependencies ready.
)

echo.
echo [2/4] Checking MNE...
python -c "import mne; print('MNE version:', mne.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: MNE not installed - simulation will be used instead of real EEG.
    echo          For real EEG: pip install mne
) else (
    echo OK: MNE installed.
)

echo.
echo [3/4] Validating patient database and sub-* folders...
python src\verify_patients.py
if %errorlevel% neq 0 (
    echo WARNING: verify_patients.py could not run.
)

echo.
echo [4/4] Starting CerebraWatch...
python main.py

pause
