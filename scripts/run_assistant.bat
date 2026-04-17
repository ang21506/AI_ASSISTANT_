@echo off
cd /d "%~dp0.."
echo Starting MediWaste AI Assistant...
.\venv\Scripts\python.exe src\rag.py
pause
