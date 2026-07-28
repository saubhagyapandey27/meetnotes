@echo off
title MeetNotes Packaging Wizard
echo ====================================================
echo             MeetNotes Executable Compiler
echo ====================================================
echo.

echo Step 1: Compiling source files to build directory...
:: Run PyInstaller using active environment settings
call conda run -n flux_intel pyinstaller --clean build.spec

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] PyInstaller compilation failed!
    echo Please verify that all dependencies are installed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ====================================================
echo SUCCESS: MeetNotes has been compiled to:
echo %CD%\dist\MeetNotes\
echo ====================================================
echo.
pause
