@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" rita.py run --seconds 60
pause
