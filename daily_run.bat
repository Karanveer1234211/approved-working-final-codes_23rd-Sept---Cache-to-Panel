@echo off
REM ===================================================================
REM  daily_run.bat - scheduled 23:45 IST, weekdays
REM
REM  v2: output now goes to BOTH the screen and the log. The previous
REM  version redirected everything into the log file, so a manual run
REM  looked frozen while the pipeline was working.
REM ===================================================================
setlocal

cd /d "%~dp0"
call "%~dp0env_daily.bat"

if "%CACHE_DAILY_ROOT%"=="" (
    echo.
    echo   CACHE_DAILY_ROOT is not set - env_daily.bat did not run.
    echo   Refusing to continue: there is no default cache path.
    echo.
    exit /b 1
)

echo   Root: %CACHE_DAILY_ROOT%
if not exist "%CACHE_DAILY_ROOT%" (
    echo   Cache root does not exist. Refusing to create one silently.
    exit /b 1
)

set "LOG=%CACHE_DAILY_ROOT%\daily_run.log"

REM PowerShell Tee-Object gives screen + log in one pass. If PowerShell is
REM unavailable the fallback below still logs, just without live output.
where powershell >nul 2>&1
if %ERRORLEVEL%==0 (
    powershell -NoProfile -Command ^
      "python run_daily.py --symbols-file watchlist.txt --only-stale %* 2>&1 | Tee-Object -FilePath '%LOG%' -Append"
    set RC=%ERRORLEVEL%
) else (
    python run_daily.py --symbols-file watchlist.txt --only-stale %* >> "%LOG%" 2>&1
    set RC=%ERRORLEVEL%
)

if not "%RC%"=="0" (
    echo.
    echo   PIPELINE FAILED ^(exit %RC%^) - panel NOT rebuilt.
    echo   See %LOG%
    exit /b %RC%
)

echo.
echo   Report: %CACHE_DAILY_ROOT%\reports\latest_daily_report.html
endlocal
exit /b 0
