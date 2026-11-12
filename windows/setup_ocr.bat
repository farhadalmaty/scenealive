@echo off
rem Package for reading text (bus numbers, signs). It downloads its models on first use.
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
pip install easyocr
echo.
echo Done. Measure on your photos: windows\bench_ocr.bat FOLDER_OR_VIDEO   Start: windows\serve.bat --ocr easyocr
