@echo off
rem Catalogue of everything that can be named. windows\vocab.bat --out my_vocab.txt  writes a file to edit; then windows\serve.bat --vocab my_vocab.txt
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
scenelive vocab %*
