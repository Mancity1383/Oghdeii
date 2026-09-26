@echo off
setlocal EnableDelayedExpansion
title Oghdeii - Standalone EXE Builder
color 0E

echo =====================================================================
echo                  Oghdeii - Standalone EXE Builder
echo =====================================================================
echo.

cd /d "%~dp0"

:: Check PyInstaller in .venv or system
set "PY_CMD="
if exist ".venv\Scripts\pyinstaller.exe" (
    set "PY_CMD=.venv\Scripts\pyinstaller.exe"
) else (
    where pyinstaller >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PY_CMD=pyinstaller"
    ) else (
        where py >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            set "PY_CMD=py -3.12 -m PyInstaller"
        ) else (
            set "PY_CMD=python -m PyInstaller"
        )
    )
)

echo [*] Using PyInstaller: %PY_CMD%

echo [*] Compiling standalone package into dist\Oghdeii...
%PY_CMD% --noconsole ^
    --name="Oghdeii" ^
    --icon="assets/icon.ico" ^
    --add-data="assets;assets" ^
    --add-data="VoiceRecognizerV1M.exe;." ^
    --add-data="calibrate.py;." ^
    --add-data="voice_calibrate.py;." ^
    --collect-all="sounddevice" ^
    --collect-all="faster_whisper" ^
    --collect-all="ctranslate2" ^
    --collect-all="av" ^
    --collect-all="scipy" ^
    --collect-all="pynput" ^
    --clean ^
    --noconfirm ^
    main.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================================================
    echo [*] Standalone build complete!
    echo [*] Executable directory: dist\Oghdeii\Oghdeii.exe
    echo =====================================================================
) else (
    echo.
    echo [ERROR] Build failed. Please check error logs above.
)

pause
