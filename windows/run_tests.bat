@echo off
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
python -m unittest discover -s tests -p "test_*.py" -v
