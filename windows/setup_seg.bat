@echo off
rem Package for the segmentation model (torch comes with ultralytics).
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
pip install transformers
echo.
echo Done. Measure on your video: windows\bench_seg.bat VIDEO
