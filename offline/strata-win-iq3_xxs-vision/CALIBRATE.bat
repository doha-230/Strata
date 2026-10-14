@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo Stop RUN.bat and other GPU jobs before calibration.
"%~dp0python\python.exe" "%~dp0prepare.py" --calibrate
if errorlevel 1 goto failed
echo.
echo Calibration saved. Start RUN.bat to use the measured settings.
pause
exit /b 0

:failed
echo.
echo Calibration failed. Previous saved settings, if any, are kept.
pause
exit /b 1
