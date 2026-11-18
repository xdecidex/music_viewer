@echo off
rem Launch Music Viewer without a console window. Creates the venv on first run.
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
    echo First run: setting up Python environment...
    python -m venv .venv || goto :error
    .venv\Scripts\python -m pip install -r requirements.txt || goto :error
)
start "" .venv\Scripts\pythonw.exe main.py %*
exit /b 0
:error
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause
