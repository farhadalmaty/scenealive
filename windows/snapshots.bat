@echo off
rem Usage: windows\snapshots.bat VIDEO [--every 4]   frames named by the second of the video, to label them
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: snapshots.bat VIDEO [--every 4] ^& exit /b 1)
scenelive snapshots %*
