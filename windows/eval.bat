@echo off
rem Usage: windows\eval.bat RUN_FOLDER labels.csv   precision/recall against your labels, suggests the confidence threshold
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~2"=="" (echo Usage: eval.bat RUN_FOLDER labels.csv ^& exit /b 1)
scenelive eval %*
