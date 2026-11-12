@echo off
rem Puts the CUDA build of PyTorch into scenelive's .venv, the same way as .venv-gpu in abbench.
rem After this the app uses the graphics card by itself; on a computer without a card the same build runs on the processor.
rem The CUDA index must suit your driver: https://pytorch.org/get-started/locally/ (Windows / Pip / CUDA). Default cu128.
rem Another index:  windows\setup_gpu.bat cu126
cd /d "%~dp0\.."
set TORCH_INDEX=https://download.pytorch.org/whl/cu128
if not "%~1"=="" set TORCH_INDEX=https://download.pytorch.org/whl/%~1
nvidia-smi >nul 2>&1 || (echo [!] nvidia-smi not found: install or update the NVIDIA driver first & exit /b 1)
if not exist .venv\Scripts\activate.bat (echo [!] .venv not found: run windows\setup.bat first & exit /b 1)
call .venv\Scripts\activate.bat
echo Installing PyTorch with CUDA from %TORCH_INDEX% (about 2-3 GB to download) ...
rem torch last and without dependencies, so that nothing replaces it with the CPU build (exactly as in .venv-gpu)
pip install --force-reinstall --no-deps torch torchvision --index-url %TORCH_INDEX% || goto :err
echo.
python -m scenelive.device
echo.
echo Must show "CUDA available: True" and your graphics card.
echo If a later "pip install" of another package brings back a version ending in +cpu, run this script again.
goto :eof
:err
echo [!] Installation failed, see the message above.
exit /b 1
