@echo off
setlocal
set "STRATA_CONTEXT_TOKENS=32768"
call "%~dp0RUN.bat"
exit /b %errorlevel%
