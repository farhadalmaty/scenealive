@echo off
rem Usage: windows\bench.bat VIDEO [--detector world^|yolo] [--imgsz 320,480,640]   detector speed on this machine
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: bench.bat VIDEO [options] ^& exit /b 1)
scenelive bench %*
