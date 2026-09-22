@echo off
REM Daily job. Schedule this for 23:45 IST on weekdays.
REM Task Scheduler: Action = Start a program, Program = this file.

call "%~dp0env_daily.bat"

python "%~dp0run_daily.py" --symbols-file "%~dp0watchlist.txt" --only-stale >> "%CACHE_DAILY_ROOT%\daily_run.log" 2>&1

if errorlevel 1 (
    echo [%date% %time%] RUN FAILED - panel NOT built >> "%CACHE_DAILY_ROOT%\daily_run.log"
    exit /b 1
)
echo [%date% %time%] OK >> "%CACHE_DAILY_ROOT%\daily_run.log"
