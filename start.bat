@echo off
setlocal enabledelayedexpansion
title Smart Choice Launcher
cd /d "%~dp0"

echo ========================================
echo       Smart Choice Launcher
echo ========================================

IF NOT EXIST ".env" (
    echo.
    echo [Initial Setup]
    set /p API_KEY="Please paste your API key here (Right-Click) and press Enter: "
    echo GEMINI_API_KEY=!API_KEY!> .env
    echo.
    echo API key saved!
    echo ========================================
)

echo.
echo Launching...
timeout /t 2 >nul
start http://localhost:8000/
python app.py
pause
