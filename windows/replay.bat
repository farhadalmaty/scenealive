@echo off
rem Usage: windows\replay.bat VIDEO [options]   e.g.  windows\replay.bat D:\videos\street.mp4 --detector world --fps 3 --out-video
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: replay.bat VIDEO [options] ^& exit /b 1)
scenelive replay %*
