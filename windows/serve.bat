@echo off
rem Web version: the page for the phone + the server. Options: windows\serve.bat --detector yolo --imgsz 480 --agent offline
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
scenelive serve %*
