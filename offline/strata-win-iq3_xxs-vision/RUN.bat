@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0python\python.exe" "%~dp0prepare.py"
if errorlevel 1 goto failed
"%~dp0python\python.exe" "%~dp0serve\server.py" --engine strata --config "%~dp0strata-iq3_xxs.json" --port 8080 --open
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Strata stopped with an error. Check strata-iq3_xxs.log if present.
pause
exit /b 1
