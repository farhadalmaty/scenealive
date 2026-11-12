@echo off
rem Usage: windows\bench_ocr.bat PHOTO_OR_FOLDER_OR_VIDEO [--bus]   time and results of text reading
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: bench_ocr.bat PHOTO_OR_FOLDER_OR_VIDEO [--bus] ^& exit /b 1)
scenelive bench-ocr %*
