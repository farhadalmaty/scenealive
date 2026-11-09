@echo off
rem Usage: windows\labels_template.bat RUN_FOLDER [--every 4] [--prefill]   writes labels.csv (skeleton) in the project folder
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: labels_template.bat RUN_FOLDER [--every 4] [--prefill] ^& exit /b 1)
scenelive labels-template %*
