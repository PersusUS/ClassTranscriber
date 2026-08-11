@echo off
REM Doble clic para abrir la ventana de ClassTranscriber.
REM Si tienes un entorno virtual en .venv, se usa automaticamente.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui.py
) else (
    python gui.py
)
if errorlevel 1 pause
