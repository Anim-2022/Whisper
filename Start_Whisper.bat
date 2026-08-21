@echo off
cd /d "%~dp0"

echo Starting Whisper Unified GUI...

rem .venv-fw is the faster-whisper environment created during the engine
rem migration; .venv may still hold the old torch stack on machines that have
rem not been rebuilt yet. Prefer the new one, fall back to the standard name so
rem a fresh clone following the README (python -m venv .venv) also works.
if exist ".venv-fw\Scripts\python.exe" (
    set "VENV_DIR=.venv-fw"
) else (
    set "VENV_DIR=.venv"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo ----------------------------------------------------------------
    echo No virtual environment found.
    echo Create one first:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt -r requirements-gpu.txt
    echo ----------------------------------------------------------------
    pause
    exit /b 1
)

echo Using %VENV_DIR%...
call "%VENV_DIR%\Scripts\activate.bat"
"%VENV_DIR%\Scripts\python.exe" start_gui.py

if %errorlevel% neq 0 (
    echo.
    echo ----------------------------------------------------------------
    echo An error occurred. Please check the message above.
    echo ----------------------------------------------------------------
    pause
) else (
    echo.
    echo Application closed.
    pause
)
