@echo off
rem Real detectors (ultralytics + torch). CPU-only torch first: much smaller than the default CUDA build.
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
echo.
echo Done. Weights download on first use. Try: windows\bench.bat VIDEO
