@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo      JAK-AIR AI - TKT 6 PROTOTYPE
echo ==========================================
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating Python environment...
  py -m venv .venv
)
call ".venv\Scripts\activate.bat"
echo [2/3] Installing/updating requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo [3/3] Starting JAK-AIR AI...
streamlit run app.py
pause
