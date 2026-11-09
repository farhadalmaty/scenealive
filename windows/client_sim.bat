@echo off
rem A simulated phone: streams a video to a running server. Usage: windows\client_sim.bat VIDEO --url "https://localhost:8443/?token=XXXX" --chat "Что вокруг?"
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: client_sim.bat VIDEO [--url ...] [--chat "..."] ^& exit /b 1)
scenelive client-sim %*
