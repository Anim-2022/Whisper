@echo off
cd /d "%~dp0"

echo Starting Whisper Unified GUI...
echo Using Virtual Environment...

call .venv\Scripts\activate.bat
python start_gui.py

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
