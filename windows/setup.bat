@echo off
rem Core install (numpy, opencv). Real detectors: run windows\setup_models.bat afterwards.
cd /d "%~dp0\.."
set PY=
for %%V in (3.12 3.11 3.10 3.13) do (
  if not defined PY (py -%%V -c "import sys" >nul 2>&1 && set PY=py -%%V)
)
if not defined PY (echo Python 3.10-3.13 not found. Install it from python.org and try again. & exit /b 1)
%PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
echo.
echo Done. Try: windows\demo.bat
