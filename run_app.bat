@echo off
rem Abre o Taekwondo Trainer usando o ambiente virtual do projeto
cd /d "%~dp0"
".venv\Scripts\python.exe" "Taekwondo_App\app.py"
if errorlevel 1 pause
