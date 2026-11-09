@echo off
rem Synthetic street video with known truth, no models needed. Prints per-stage timings.
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
scenelive demo %*
