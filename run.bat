@echo off
cd /d "%~dp0"
call venv\Scripts\activate
uvicorn main:app --reload
pause
