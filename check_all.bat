@echo off
REM ===================================================================
REM  Run after ANY change to features_daily.py or Daily_cache_v27.py.
REM  Not part of the daily job. Takes under a minute. No Kite needed.
REM ===================================================================
call "%~dp0env_daily.bat"
set FAILED=0

echo.
echo [1/3] Feature maths: reference implementations, invariants,
echo       pathological inputs, duplicates, silent columns
python "%~dp0verify_features.py" || set FAILED=1

echo.
echo [2/3] Leak detector: does it catch known leaks, and stay quiet
echo       on legitimate causal code
python "%~dp0test_leakcheck.py" || set FAILED=1

echo.
echo [3/3] Cache + gate regression: calendar, gaps, absence classes,
echo       roll marking, quarantine behaviour
python "%~dp0test_cache_v27.py" && python "%~dp0test_panel.py" && python "%~dp0test_reconcile.py" && python "%~dp0test_orphan.py" || set FAILED=1

echo.
if %FAILED%==1 (
    echo RESULT: FAILURES ABOVE - do not run the pipeline until fixed.
    exit /b 1
)
echo RESULT: all suites passed.
