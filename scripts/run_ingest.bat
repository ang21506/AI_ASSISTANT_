@echo off
cd /d "%~dp0.."
echo Running Knowledge Base Ingestion...
.\venv\Scripts\python.exe src\ingest.py
pause
