@echo off
rem Usage: windows\bench_seg.bat VIDEO [--frames 12]   speed of the segmentation model + overlay pictures (floor green, walls blue ...)
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: bench_seg.bat VIDEO [--frames 12] ^& exit /b 1)
scenelive bench-seg %*
