@echo off
setlocal
set "STRATA_CONTEXT_TOKENS=131072"
set "STRATA_PIN_BY_NEED=1"
call "%~dp0RUN.bat"
exit /b %errorlevel%
