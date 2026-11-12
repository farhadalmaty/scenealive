@echo off
rem Where do the models run? PyTorch build, CUDA, graphics card.
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python -m scenelive.device
