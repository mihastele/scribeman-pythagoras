@echo off
REM ──────────────────────────────────────────────────────────────
REM setup.bat  —  Windows dependency installer
REM Run once: double-click or run from cmd/PowerShell
REM ──────────────────────────────────────────────────────────────

echo =^> Installing Python dependencies...
uv pip install --upgrade pip
uv pip install -r requirements.txt

echo.
echo =^> Done.
echo    WASAPI loopback is built into Windows — no extra drivers needed.
echo    Run: python transcribe.py --list-devices
pause
