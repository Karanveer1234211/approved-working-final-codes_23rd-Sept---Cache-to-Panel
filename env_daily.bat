@echo off
REM ===================================================================
REM  Environment for the NSE quant cache. Edit paths, then call this
REM  before any run:   call env_daily.bat
REM ===================================================================

REM --- Where the code lives -----------------------------------------
set QUANT_HOME=C:\Users\karanvsi\PyCharmMiscProject\bigmove_deploy

REM --- Where the raw cache is written -------------------------------
REM Keep this OFF OneDrive. OneDrive's sync locks files mid-write and
REM has broken hardcoded paths in this pipeline before.
set CACHE_DAILY_ROOT=C:\QuantData\cache_daily

REM --- Kite credentials ---------------------------------------------
set KITE_TOKEN_FILE=C:\Users\karanvsi\PyCharmMiscProject\kite_token.json

REM --- Pin the earliest stored date ---------------------------------
REM Without this, the expanding-rank WQ alphas take different values for
REM the same date whenever the run's start date moves. Set it once and
REM never change it, or you invalidate every cached feature value.
set CACHE_HISTORY_ANCHOR=2015-01-01

REM --- Refetch a trailing window each run ---------------------------
REM Repairs provisional bars (MCX runs to 23:30) and vendor corrections.
set CACHE_REFETCH_TAIL_DAYS=7

cd /d %QUANT_HOME%
echo Environment set. Cache root: %CACHE_DAILY_ROOT%
