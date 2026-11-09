@echo off
rem Packages for the server: cryptography (https for the phone camera) and anthropic (Claude chat agent).
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
pip install cryptography anthropic
echo.
echo Done. For Claude set the key first:  setx ANTHROPIC_API_KEY your_key   (then open a NEW window).  Start: windows\serve.bat
