#!/usr/bin/env python3
"""
FAST Daily Cache Builder for Zerodha Kite historical data (equities only)
Warm-up backfill, leak-free indicators, feature finalization, and logical combos.

v24 (2026-09-21)  — RAW CACHE ONLY + EXOGENOUS SERIES
=====================================================
Indicators live in features_daily.py, together with the v20 static leak check
and runtime canary. The cache makes exactly one promise: what is on disk
matches what the exchange printed, on one consistent corporate-action basis,
with no gaps. Feature definitions change constantly; the cache schema does
not, so the cache never needs rebuilding.

CORPORATE ACTION GUARD
----------------------
The incremental path dedups keep="last", which only refreshes the tail, so a
split would otherwise weld pre-split history to post-split bars permanently.
A split-shaped discontinuity discards the incremental result and refetches
the series whole. Split detection is disabled for indices (nothing adjusts)
and for futures (every roll is a legitimate discontinuity).

EXOGENOUS SERIES
----------------
NIFTY, sector indices, INDIA VIX, USDINR, crude and gold are cached on every
run via DEFAULT_EXOGENOUS, before equities, because NIFTY50 is the panel's
master calendar. Three things differ from equities:

  1. CALENDAR. MCX and CDS trade on days NSE does not. Everything is
     reindexed onto NSE sessions by align_to_sessions(), forward-filled only,
     never backfilled, with a <name>_stale counter so a four-day-old crude
     print cannot masquerade as today's.

  2. SESSION END - THE LEAK THAT MATTERS. NSE closes 15:30, currency 17:00,
     MCX 23:30. A day-t gold close did not exist when NSE closed on day t.
     Feeding it to a model predicting t+1 is a time machine that looks
     exactly like alpha. lag_sessions is set to 1 automatically for anything
     closing after 15:30.

  3. ROLLS. Kite's continuous=1 stitches expired contracts but does NOT
     back-adjust, so there is a real price step at every expiry.

Indices print no volume; the cache stores NaN rather than the API's 0, so
volume features cannot read a phantom zero-volume session.

ROLL CALENDAR (v23) - replaced price inference
----------------------------------------------
v22's _mark_rolls inferred rolls from unusual price moves. That was wrong in
both directions: a genuine crude shock was labelled a roll, and a quiet roll
was missed. v23 records contract expiries observed in the instrument dump
(update_roll_calendar, append-only) and marks rolls from that calendar.

D_roll is a NULLABLE Int8 with three states, and the third is the point:
  1 = first session after a known expiry, 0 = inside a known contract,
  NA = predates the calendar, i.e. UNKNOWN. v22 wrote 0 for everything it had
not flagged, claiming knowledge it did not have. The dump holds live
contracts only, so the calendar grows forward from your first run; history
before that is genuinely unknown and the gate refuses it rather than guessing.

SESSION GAPS (v23)
------------------
Coverage now compares every symbol against the master calendar inside its own
first..last span, not just its last date. v22 could report a symbol current
while it was missing forty sessions in the middle of 2023.

FAILING LOUDLY (v23)
--------------------
resolve_front_month raises instead of returning an unresolved Series, and
strict_exogenous aborts the run if a macro series cannot be resolved - the
alternative is a panel silently missing a whole feature family.

RATE LIMITS (v23)
-----------------
Kite caps the historical endpoint at 3 requests/sec. The old defaults (16
CLI / 16 GUI / 32 Config - three different numbers) were 5-10x over, so 429s
and retry backoff were doing the real throttling. One constant now:
KITE_HISTORICAL_RPS. More workers above the rate limit only buys retry churn.

rows_valid (v24)
----------------
rows_valid and first_valid_timestamp were warm-up concepts that left with the
indicators in v22, and _save never wrote them again - so meta.get() returned
None -> 0 for EVERY symbol and "ZERO USABLE ROWS" fired for the whole
universe on every run. A raw cache has one row count. They are gone.

NOT SOLVED HERE
---------------
Sector MEMBERSHIP is point-in-time and is not handled. Index constituents
change; a sector map scraped today and applied to 2018 data is both lookahead
and survivorship bias, in the one feature family most likely to carry real
cross-sectional alpha.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import datetime as dt
import functools
import json
import math
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
import pandas as pd
from requests.exceptions import (
    ReadTimeout,
    ConnectTimeout,
    ConnectionError as RequestsConnectionError,
)

try:
    from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError
except Exception:
    Urllib3ReadTimeoutError = Exception

# -------------------- GUI (tkinter) --------------------
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    from tkinter import ttk
    TK_OK = True
except Exception:
    TK_OK = False


# v28: a warning dialog must NEVER block a batch run.
#
# A CLI run hung for an hour inside tk.messagebox.showwarning, AFTER every
# parquet had been written - the work was finished and the process sat waiting
# on a message loop that never ran. Under Task Scheduler there is often no
# desktop session at all, so the nightly job would hang every night and the
# batch file would never log anything.
#
# Dialogs are now GUI-only and never appear on a CLI path.
_HEADLESS = False


def set_headless(flag: bool = True) -> None:
    """Suppress every dialog. Set automatically for CLI and scheduled runs."""
    global _HEADLESS
    _HEADLESS = bool(flag)


def notify(title: str, message: str) -> None:
    """
    Show a dialog if a GUI is genuinely available; otherwise just print.

    Never raises, never blocks. The console output is the real record - the
    dialog is a convenience for the GUI path only.
    """
    print(f"[{title}] {message}")
    if _HEADLESS or not TK_OK:
        return
    try:
        root = tk._default_root
        if root is None:
            return          # no live Tk mainloop: printing is the whole job
        tk.messagebox.showwarning(title, message)
    except Exception:
        pass

# -------------------- Timezone / market session --------------------
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DEFAULT_SESSION_OPEN  = dt.time(9, 15, tzinfo=IST)
DEFAULT_SESSION_CLOSE = dt.time(15, 30, tzinfo=IST)


def today_ist() -> dt.date:
    return dt.datetime.now(tz=IST).date()


# -------------------- Schema + .ok metadata --------------------
SCHEMA_VERSION = 20
OK_VERSION_KEY = "schema_version"

# -------------------- Windows default cache roots --------------------
# v29: the silent Windows fallback is GONE.
#
# It pointed at a OneDrive-synced Desktop folder holding a stale copy of the
# cache. Whenever CACHE_DAILY_ROOT was unset - a new terminal, a forgotten
# `call env_daily.bat` - every tool quietly read and wrote THAT cache
# instead, and reported success. It produced a mixed panel, a memory index
# built on 89 RELIANCE episodes instead of 2,544, and a feature audit run
# against a panel three sessions stale. Each time the output looked correct.
#
# A default that silently selects a different dataset is worse than no
# default. An unset root is now a hard error naming the fix.
WIN_DEFAULT_BASE = None


def _require_root_env() -> None:
    raise RuntimeError(
        "CACHE_DAILY_ROOT is not set.\n"
        "  Run:  call env_daily.bat\n"
        "  Then: echo %CACHE_DAILY_ROOT%   (expect C:\\QuantData\\cache_daily)\n"
        "There is deliberately no default - guessing a cache path silently "
        "corrupts every downstream artefact."
    )


def _expand_path(value: str) -> Path:
    return Path(value).expanduser()


def _default_base_dir() -> Path:
    env_base = os.environ.get("CACHE_BASE_DIR")
    if env_base:
        return _expand_path(env_base)
    if os.name == "nt":
        _require_root_env()
    return Path.home() / ".kite_cache"


def _platform_default(
    env_var: str, *, windows_default: Path, unix_suffix: str
) -> Path:
    val = os.environ.get(env_var)
    if val:
        return _expand_path(val)
    base = _default_base_dir()
    if os.name == "nt":
        return windows_default
    return base / unix_suffix


def _default_daily_root() -> Path:
    return _platform_default(
        "CACHE_DAILY_ROOT",
        windows_default=None,
        unix_suffix="cache_daily_new",
    )


# -------------------- Config --------------------
# Kite Connect's documented cap on /instruments/historical. Verify against
# current docs before raising it; exceeding it returns 429, not an error
# you would notice in the data.
KITE_HISTORICAL_RPS = 3.0

# v24.1: how many trailing sessions to REFETCH on every incremental run.
#
# Without this, `cached_last + 1 day` means any bar already on disk is never
# looked at again. That is fine for an equity fetched after 15:30, and wrong
# for everything else: run at 16:00 and today's MCX crude/gold candle is
# stored PARTIAL (their session runs to 23:30) and then frozen forever.
# Vendors also issue late corrections to recent bars.
#
# Refetching a short tail costs one extra API call per symbol and lets
# drop_duplicates(keep="last") do what it was always meant to do: replace
# provisional bars with settled ones.
REFETCH_TAIL_DAYS = int(os.environ.get("CACHE_REFETCH_TAIL_DAYS", "7"))

# v26: never store a bar from a session that has not finished.
#
# Running at 16:00 is fine for equities (NSE closed 15:30) but not for MCX,
# which trades to 23:30 - that bar would be a partial candle frozen on disk.
# Running at 15:00 is worse: even the equity bars are mid-session snapshots.
#
# Rather than store provisional data and hope the tail refetch repairs it,
# each series' fetch window is CLAMPED to its last completed session. An
# early run simply gets one fewer bar; nothing provisional ever enters the
# cache, so nothing downstream has to know about provisional state.
SESSION_GRACE_MIN = int(os.environ.get("CACHE_SESSION_GRACE_MIN", "15"))


# v27: ORPHAN BARS BEFORE THE REAL LISTING
#
# Kite returns a handful of stray bars years before a stock's actual listing.
# DELHIVERY (IPO May 2022) came back with 8 bars in 2016 - dated weeks apart -
# then nothing until 2022. Exchanges reuse tickers after a delisting, so those
# bars may well belong to a different company that once held the symbol.
#
# They are poison for two reasons:
#   * they drag the series' first date back by years, so a gap check measures
#     the span as 2016-2026 and correctly reports ~1560 "missing" sessions
#   * if they are a different company, the prices are simply another firm's
#
# This was 1505 of 2262 symbols flagged with interior gaps. The measurement
# was right; the input was poisoned.
#
# Detection is density, not price: real history is dense from day one. Scan
# forward for the first bar after which the series becomes continuous, and
# treat everything before it as orphan.
# v28 rule: trim on the HOLE, not on a bar count.
#
# The first version capped trimming at 60 leading bars. That was wrong against
# real data: FUSION carries 123 pre-listing bars from 2016, ELIN 183, KOVAI
# 435 - all above the cap, so the detector refused and left every one of them
# in place. The cap protected nothing; a real listing does not contain a
# multi-year hole.
#
# The rule is now: find the last gap of a year or more and drop everything
# before it. A ticker reassigned after a delisting leaves exactly that
# signature. So does a stock suspended for years - and its pre-suspension
# prices are equally unusable, because every feature spanning the hole is
# meaningless whether or not it is the same company.
ORPHAN_MIN_GAP_DAYS = int(os.environ.get("CACHE_ORPHAN_GAP_DAYS", "365"))
ORPHAN_MIN_REMAINING = 252   # never trim down to less than a year of history


def find_listing_start(
    ts: pd.Series,
    *,
    min_gap_days: int = ORPHAN_MIN_GAP_DAYS,
    min_remaining: int = ORPHAN_MIN_REMAINING,
) -> Optional[int]:
    """
    Index of the first bar after the last multi-year hole, or None if the
    series has no such hole.

    Works backwards from the most recent qualifying gap, so a series with
    several holes keeps only the final continuous run - and refuses if that
    would leave less than min_remaining bars.
    """
    if ts is None or len(ts) < 2:
        return None
    t = pd.DatetimeIndex(pd.to_datetime(ts))
    try:
        t = t.tz_localize(None)
    except TypeError:
        t = t.tz_convert(None)
    t = t.normalize()

    gaps = (t[1:] - t[:-1]).days
    idx = np.where(gaps >= min_gap_days)[0]
    if not len(idx):
        return None
    for j in idx[::-1]:
        start = int(j) + 1
        if len(t) - start >= min_remaining:
            return start
    return None


def trim_orphan_bars(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop pre-listing orphan bars. Returns the frame and what was removed."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return df, []
    i = find_listing_start(df["timestamp"])
    if not i:
        return df, []
    dropped = [
        _maybe_iso(v) for v in df["timestamp"].iloc[:i].tolist()
    ]
    return df.iloc[i:].reset_index(drop=True), dropped


def last_completed_session(series: "Series", now: Optional[dt.datetime] = None) -> dt.date:
    """
    The most recent date whose session has fully closed for this series.

    Today counts only if the clock is past that series' session end plus a
    grace period (the exchange's last prints settle a few minutes late).
    """
    now = now or dt.datetime.now(tz=IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    hh, mm = (int(x) for x in str(series.session_end_ist).split(":"))
    close_today = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    close_today += dt.timedelta(minutes=SESSION_GRACE_MIN)
    return now.date() if now >= close_today else now.date() - dt.timedelta(days=1)

@dataclass(frozen=True)
class Config:
    daily_root: Path = field(default_factory=_default_daily_root)
    trading_open: dt.time = DEFAULT_SESSION_OPEN
    trading_close: dt.time = DEFAULT_SESSION_CLOSE
    # v23: Kite limits the historical endpoint to 3 requests/sec. The old
    # defaults (16 CLI / 16 GUI / 32 here - three different numbers) were
    # 5-10x over, so 429s and retry backoff were doing the actual
    # throttling. Concurrency above the rate limit buys nothing and just
    # converts into retry churn, which is why more workers never helped.
    max_workers: int = 6
    rate_limit_per_sec: float = KITE_HISTORICAL_RPS
    request_timeout_s: float = 15.0
    retry_tries: int = 6
    retry_backoff_base: float = 0.45
    parquet_engine: str = os.environ.get("PARQUET_ENGINE", "pyarrow")
    parquet_compression: Optional[str] = os.environ.get("PARQUET_COMPRESSION", "snappy")
    parquet_use_dictionary: bool = True

    def day_root(self) -> Path:
        return self.daily_root

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        kwargs: dict = {}
        val = os.environ.get("CACHE_DAILY_ROOT")
        if val:
            kwargs["daily_root"] = _expand_path(val)
        kwargs.update(overrides)
        return cls(**kwargs)

    def with_updates(self, **updates) -> "Config":
        return replace(self, **updates)


# -------------------- Sanitization / paths --------------------
_ILLEGAL = set('<>:"/\n?*')
_HEADER_WORDS = {"symbol", "symbols", "ticker", "tickers", "scrip", "scrips", "name"}


def sanitize_symbol(sym: str) -> Optional[str]:
    if sym is None:
        return None
    s = str(sym)
    s = s.replace("\x00", "").replace("\r", " ").replace("\t", " ").replace("\n", " ")
    s = s.lstrip("\ufeff")
    s = " ".join(s.strip().split())
    s = "".join(ch for ch in s if ch not in _ILLEGAL)
    if not s:
        return None
    if s.strip().casefold() in _HEADER_WORDS:
        return None
    return s


def assert_path_safe(p: Path):
    sp = str(p)
    if "\x00" in sp:
        raise ValueError(f"Path contains NUL (\\x00): {sp!r}")


def daily_path(config: Config, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    p = config.day_root() / f"{s}_daily.parquet"
    assert_path_safe(p)
    return p


def ok_path(config: Config, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    p = config.day_root() / f"{s}_daily.ok.json"
    assert_path_safe(p)
    return p


def ok_meta_base() -> dict:
    return {
        OK_VERSION_KEY: SCHEMA_VERSION,
        "created_ts": dt.datetime.now(tz=IST).isoformat(),
    }


# -------------------- Atomic IO --------------------
class FileLock:
    def __init__(self, path: Path, poll_ms: int = 50, timeout_s: float = 30.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(str(path) + ".lock")
        self.poll_ms = poll_ms
        self.timeout_s = timeout_s
        self._fd: Optional[int] = None

    def acquire(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self._fd = os.open(
                    self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                os.write(self._fd, str(os.getpid()).encode())
                return
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"Timeout acquiring lock {self.lock_path}")
                time.sleep(self.poll_ms / 1000.0)

    def release(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.lock_path)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def atomic_write_bytes(target: Path, data: bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def write_json_atomic(path: Path, obj: dict):
    raw = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode()
    atomic_write_bytes(path, raw)


def to_parquet(
    path: Path,
    df: pd.DataFrame,
    *,
    engine: str,
    compression: Optional[str],
    use_dictionary: bool,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        path,
        index=False,
        engine=engine,
        compression=compression,
        use_dictionary=use_dictionary,
    )


def read_parquet(
    path: Path, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    if columns is not None:
        columns = list(columns)
    return pd.read_parquet(path, columns=columns)


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- Rate limit + retry --------------------
class RateLimiter:
    def __init__(self, per_sec: float):
        self.per_sec = float(per_sec)
        self._lock = threading.Lock()
        self._tokens = per_sec
        self._updated = time.perf_counter()

    def acquire(self):
        while True:
            with self._lock:
                now = time.perf_counter()
                self._tokens = min(
                    self.per_sec,
                    self._tokens + (now - self._updated) * self.per_sec,
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                need = max(0.0, 1.0 - self._tokens)
                wait = need / self.per_sec if self.per_sec > 0 else 0.0
            time.sleep(wait if wait > 0 else 0)


def with_retry(fn, *, tries: int, backoff: float):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        attempt = 0
        last_exc = None
        while attempt < tries:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                if "too many" in msg or "rate" in msg or "429" in msg:
                    sleep = min(15.0, backoff * (2.2**attempt))
                else:
                    sleep = min(
                        8.0,
                        backoff * (1.8**attempt) + np.random.random() * (backoff / 2),
                    )
                last_exc = e
                time.sleep(sleep)
                attempt += 1
        raise last_exc

    return wrapper


class DataFrameCache:
    def __init__(self, maxsize: int = 256):
        self.maxsize = max(1, int(maxsize))
        self._store: Dict[tuple, pd.DataFrame] = {}
        self._order: List[tuple] = []
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[pd.DataFrame]:
        with self._lock:
            df = self._store.get(key)
            if df is None:
                return None
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return df.copy(deep=True)

    def put(self, key: tuple, df: pd.DataFrame) -> pd.DataFrame:
        clone = df.copy(deep=True)
        with self._lock:
            self._store[key] = clone
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            while len(self._order) > self.maxsize:
                old = self._order.pop(0)
                self._store.pop(old, None)
        return clone.copy(deep=True)


# -------------------- Provider + resolver (Kite) --------------------
try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import TokenException, KiteException, InputException
except Exception:
    KiteConnect = None
    TokenException = KiteException = InputException = Exception


class AuthExpired(Exception):
    """Kite access token missing/expired/invalid."""


def _token_file_path() -> str:
    env_path = os.environ.get("KITE_TOKEN_FILE")
    if env_path:
        return env_path
    default_win = r"C:\Users\karanvsi\PyCharmMiscProject\kite_token.json"
    if os.name == "nt" and os.path.exists(default_win):
        return default_win
    return os.path.join(os.path.dirname(__file__), "kite_token.json")


def _instrument_cache_path() -> Path:
    p = os.environ.get("INSTRUMENT_CACHE_FILE")
    if p:
        return Path(p)
    # Secondary artefact: falls back to CWD rather than raising, because it
    # is resolved at import time and importing must never demand a root. The
    # hard error lives on the CACHE ROOT itself, which is the path whose
    # silent substitution actually corrupts things.
    _root = os.environ.get("CACHE_DAILY_ROOT")
    default_win = str(Path(_root) / "instrument_cache.json") if _root \
        else "instrument_cache.json"
    return Path(default_win) if os.name == "nt" else Path("instrument_cache.json")


def _load_instrument_cache() -> dict:
    try:
        p = _instrument_cache_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_instrument_cache(cache: dict) -> None:
    try:
        p = _instrument_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


import difflib

UNRESOLVED_SYMBOLS_LOG = Path(
    os.environ.get("UNRESOLVED_SYMBOLS_LOG")
    or str(_instrument_cache_path().parent / "unresolved_symbols.jsonl")
)
OVERRIDES_FILE = Path(
    os.environ.get("SYMBOL_OVERRIDES_FILE")
    or str(_instrument_cache_path().parent / "symbol_overrides.json")
)


class UnresolvedSymbol(Exception):
    """Symbol cannot be mapped to instrument_token."""


def _normalize_sym(s: str) -> str:
    s = (s or "").upper().strip()
    for suf in ("-EQ", "-BE", "-BZ", "-BL", "-SM", "-GS", "-GB"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return "".join(ch for ch in s if ch.isalnum())


def _load_overrides() -> dict:
    try:
        if OVERRIDES_FILE.exists():
            with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {_normalize_sym(k): str(v).upper() for k, v in d.items()}
    except Exception:
        pass
    return {}


def _append_unresolved_log(symbol: str, suggestions: list):
    val = os.environ.get("SKIP_UNRESOLVED", "").strip().lower()
    if val in ("1", "true", "yes"):
        return
    UNRESOLVED_SYMBOLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": dt.datetime.now(tz=IST).isoformat(),
        "symbol": symbol,
        "suggestions": suggestions[:5],
    }
    with open(UNRESOLVED_SYMBOLS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class InvalidInstrument(Exception):
    """Instrument token invalid/stale."""


class SymbolResolver:
    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self.overrides = _load_overrides()
        self._built = False
        self.exact: Dict[str, int] = {}
        self.base: Dict[str, int] = {}
        self.names: List[str] = []

    def _build_maps(self):
        if self._built:
            return
        rows = self.kite.instruments("NSE") + self.kite.instruments("BSE")
        by_exact: dict = {}
        by_base: dict = {}
        for r in rows:
            ts = str(r.get("tradingsymbol", "")).upper()
            itok = r.get("instrument_token")
            seg = str(r.get("segment", ""))
            inst_type = str(r.get("instrument_type", ""))
            if not itok or not ts:
                continue
            score = (
                2
                if (seg.upper().startswith(("NSE", "BSE")) and inst_type.upper() == "EQ")
                else 1
                if seg.upper().startswith(("NSE", "BSE"))
                else 0
            )
            prev = by_exact.get(ts)
            if prev is None or score > prev[0]:
                by_exact[ts] = (score, int(itok))
            base_key = _normalize_sym(ts)
            prevb = by_base.get(base_key)
            if prevb is None or score > prevb[0]:
                by_base[base_key] = (score, int(itok))
        self.exact = {k: v[1] for k, v in by_exact.items()}
        self.base = {k: v[1] for k, v in by_base.items()}
        self.names = list(self.exact.keys())
        self._built = True

    def resolve(self, symbol: str) -> Optional[int]:
        norm = _normalize_sym(symbol)
        if norm in self.overrides:
            want = self.overrides[norm]
            self._build_maps()
            tok = (
                self.exact.get(want)
                or self.exact.get(f"{want}-EQ")
                or self.base.get(_normalize_sym(want))
            )
            if tok:
                return int(tok)
        self._build_maps()
        if symbol.upper() in self.exact:
            return int(self.exact[symbol.upper()])
        if f"{symbol.upper()}-EQ" in self.exact:
            return int(self.exact[f"{symbol.upper()}-EQ"])
        if norm in self.base:
            return int(self.base[norm])
        close_matches = difflib.get_close_matches(
            symbol.upper(), self.names, n=5, cutoff=0.77
        )
        _append_unresolved_log(symbol, close_matches)
        return None


class KiteProvider:
    """Zerodha Kite-backed provider with symbol->instrument caching (daily-only)."""

    def __init__(self, *, exchange_prefix: str = "NSE:"):
        if KiteConnect is None:
            raise RuntimeError("kiteconnect not installed. `pip install kiteconnect`")
        self.exchange_prefix = exchange_prefix.rstrip(":") + ":"
        self._kite: Optional[KiteConnect] = None
        self._instruments: Dict[str, int] = {}
        self._inst_cache_file: Path = _instrument_cache_path()
        self._inst_cache_data: Dict[str, int] = {
            k.upper(): int(v)
            for k, v in _load_instrument_cache().items()
            if isinstance(v, (int, float, str)) and str(v).isdigit()
        }
        self._dumps: Dict[str, list] = {}
        self._dump_lock = threading.Lock()
        self._load_token()
        self._resolver = SymbolResolver(self._kite)

    def _load_token(self) -> None:
        token_file = _token_file_path()
        if not os.path.exists(token_file):
            raise AuthExpired(f"Token file missing: {token_file}. Refresh it first.")
        with open(token_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        api_key = data.get("api_key")
        access_token = data.get("access_token")
        if not api_key or not access_token:
            raise AuthExpired("api_key/access_token missing in token file.")
        http_timeout = float(os.environ.get("KITE_HTTP_TIMEOUT", "15"))
        max_tries, backoff = 5, 0.6
        last_err = None
        kite = KiteConnect(api_key=api_key)
        try:
            kite.timeout = http_timeout
        except Exception:
            pass
        kite.set_access_token(access_token)
        for attempt in range(max_tries):
            try:
                _ = kite.profile()
                self._kite = kite
                return
            except TokenException as e:
                raise AuthExpired(str(e)) from e
            except (
                ReadTimeout,
                ConnectTimeout,
                RequestsConnectionError,
                Urllib3ReadTimeoutError,
                socket.timeout,
            ) as e:
                last_err = e
                sleep_s = min(12.0, backoff * (1.8**attempt))
                time.sleep(sleep_s)
                continue
            except Exception as e:
                last_err = e
                break
        if last_err:
            raise RuntimeError(
                f"Kite API connectivity failed after retries: {last_err}"
            ) from last_err
        raise RuntimeError("Kite API connectivity failed for an unknown reason.")

    def _instrument_dump(self, exchange: str) -> list:
        """
        v25 SPEED: memoise the instrument dump per process.

        Each instruments(exchange) call downloads and parses the whole
        exchange master - several MB. Resolving three futures series was
        doing it six times per run (once in resolve_front_month, once in
        token_for, per series), and _symbol_to_instrument_token pulls
        NSE+BSE again on every symbol that misses the ltp() fast path.
        One fetch per exchange per run is enough; the dump does not change
        intraday.
        """
        with self._dump_lock:
            cached = self._dumps.get(exchange)
            if cached is not None:
                return cached
        rows = self._kite.instruments(exchange)
        with self._dump_lock:
            self._dumps[exchange] = rows
        return rows

    def token_for(self, series) -> int:
        """
        v22: resolve any Series to an instrument token.

        NSE equities and indices go through the existing resolver (ltp()
        accepts 'NSE:NIFTY 50' as readily as 'NSE:INFY'). CDS/MCX futures
        are looked up in that exchange's instrument dump by tradingsymbol,
        and are NEVER cached: exchanges reuse instrument tokens for
        derivatives after expiry, so a stale token silently returns a
        different contract's data.
        """
        exch = getattr(series, "exchange", "NSE")
        tsym = getattr(series, "tradingsymbol", None) or getattr(
            series, "name", str(series)
        )
        if exch in ("NSE", "BSE"):
            return self._symbol_to_instrument_token(tsym)
        rows = self._instrument_dump(exch)
        want = tsym.upper()
        for r in rows:
            if str(r.get("tradingsymbol", "")).upper() == want:
                return int(r["instrument_token"])
        raise UnresolvedSymbol(f"{exch}:{tsym}")

    def _symbol_to_instrument_token(self, symbol: str) -> int:
        sym = symbol.strip().upper()
        if sym in self._instruments:
            return self._instruments[sym]
        if sym in self._inst_cache_data:
            tok = int(self._inst_cache_data[sym])
            self._instruments[sym] = tok
            return tok
        assert self._kite is not None
        qual = f"{self.exchange_prefix}{sym}"
        try:
            quote = self._kite.ltp([qual])
            if quote and isinstance(quote, dict):
                info = quote.get(qual) or (
                    list(quote.values())[0] if list(quote.values()) else None
                )
                if info and "instrument_token" in info:
                    inst = int(info["instrument_token"])
                    self._instruments[sym] = inst
                    self._inst_cache_data[sym] = inst
                    _save_instrument_cache(self._inst_cache_data)
                    return inst
        except TokenException as e:
            raise AuthExpired(str(e))
        except InputException:
            pass
        except KiteException:
            pass
        rows = self._instrument_dump("NSE") + self._instrument_dump("BSE")
        by_exact: dict = {}
        by_base: dict = {}
        for r in rows:
            ts = str(r.get("tradingsymbol", "")).upper()
            itok = r.get("instrument_token")
            seg = str(r.get("segment", ""))
            inst_type = str(r.get("instrument_type", ""))
            if not itok or not ts:
                continue
            score = (
                2
                if (seg.upper().startswith(("NSE", "BSE")) and inst_type.upper() == "EQ")
                else 1
                if seg.upper().startswith(("NSE", "BSE"))
                else 0
            )
            prev = by_exact.get(ts)
            if prev is None or score > prev[0]:
                by_exact[ts] = (score, int(itok))
            base = ts[:-3] if ts.endswith("-EQ") else ts
            prevb = by_base.get(base)
            if prevb is None or score > prevb[0]:
                by_base[base] = (score, int(itok))
        token = None
        if sym in by_exact:
            token = by_exact[sym][1]
        elif f"{sym}-EQ" in by_exact:
            token = by_exact[f"{sym}-EQ"][1]
        elif sym in by_base:
            token = by_base[sym][1]
        else:
            base = sym[:-3] if sym.endswith("-EQ") else sym
            token = by_exact.get(base, (None, None))[1] or by_base.get(
                base, (None, None)
            )[1]
        if token is None:
            resolved = self._resolver.resolve(symbol)
            if resolved is not None:
                token = int(resolved)
        if token is None:
            names = list(by_exact.keys())
            close_matches = difflib.get_close_matches(sym, names, n=3, cutoff=0.80)
            _append_unresolved_log(symbol, close_matches)
            raise UnresolvedSymbol(sym)
        inst = int(token)
        self._instruments[sym] = inst
        self._inst_cache_data[sym] = inst
        _save_instrument_cache(self._inst_cache_data)
        return inst

    def _hist(
        self,
        instrument_token: int,
        start_dt: dt.datetime,
        end_dt: dt.datetime,
        interval: str,
        continuous: bool = False,
    ):
        assert self._kite is not None
        try:
            return self._kite.historical_data(
                instrument_token,
                from_date=start_dt,
                to_date=end_dt,
                interval=interval,
                oi=False,
                continuous=bool(continuous),
            )
        except TokenException as e:
            raise AuthExpired(str(e))
        except InputException as e:
            msg = str(e).lower()
            if "invalid token" in msg or "instrument_token" in msg:
                raise InvalidInstrument(str(e))
            if "too many" in msg or "429" in msg:
                raise
            raise RuntimeError(f"Kite historical data failed: {e}")

    def _ensure_ist_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "date" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            try:
                import zoneinfo
                tz = zoneinfo.ZoneInfo("Asia/Kolkata")
            except Exception:
                tz = IST
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize(tz)
            else:
                ts = ts.dt.tz_convert(tz)
            df["timestamp"] = ts
        return df

    def fetch_daily(self, symbol, start: dt.date, end: dt.date) -> pd.DataFrame:
        """
        Fetch daily candles for [start, end] inclusive, chunked to respect
        Kite's 2000-day per-call maximum.

        v22: `symbol` may be a bare string (NSE equity) or a Series, which
        carries the exchange and whether to stitch expired contracts.
        """
        MAX_DAYS = 1999
        continuous = bool(getattr(symbol, "continuous", False))
        has_volume = bool(getattr(symbol, "has_volume", True))

        def _iter_chunks(s: dt.date, e: dt.date):
            cur = s
            while cur <= e:
                hi = min(e, cur + dt.timedelta(days=MAX_DAYS))
                yield cur, hi
                cur = hi + dt.timedelta(days=1)

        inst = (
            self.token_for(symbol)
            if not isinstance(symbol, str)
            else self._symbol_to_instrument_token(symbol)
        )
        all_rows: list = []
        for chunk_start, chunk_end in _iter_chunks(start, end):
            start_dt_c = dt.datetime.combine(chunk_start, dt.time(0, 0))
            end_dt_c = dt.datetime.combine(chunk_end, dt.time(23, 59))
            try:
                rows = self._hist(
                    inst, start_dt_c, end_dt_c, interval="day",
                    continuous=continuous,
                )
            except InvalidInstrument:
                if isinstance(symbol, str):
                    sym = symbol.strip().upper()
                    self._instruments.pop(sym, None)
                    self._inst_cache_data.pop(sym, None)
                    _save_instrument_cache(self._inst_cache_data)
                    inst = self._symbol_to_instrument_token(symbol)
                else:
                    inst = self.token_for(symbol)
                rows = self._hist(
                    inst, start_dt_c, end_dt_c, interval="day",
                    continuous=continuous,
                )
            all_rows.extend(rows or [])
        df = pd.DataFrame(all_rows)
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = self._ensure_ist_timestamp(df)
        use_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in use_cols:
            if c not in df.columns:
                df[c] = pd.NA
        out = df[use_cols]
        if not has_volume:
            # Indices print no volume. Storing the API's 0 would make every
            # volume feature read as a genuine zero-volume session.
            out = out.assign(volume=np.nan)
        return out


# -------------------- Helpers: tz, indicators, validation --------------------

def _ensure_ist(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df = df.copy()
    df["timestamp"] = ts
    return df


def _validate_monotonic(df: pd.DataFrame):
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be strictly monotonic increasing")
    if df["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps detected")


def _raw_manifest(df: pd.DataFrame) -> dict:
    """v22: the cache stores raw OHLCV only, so the manifest describes data
    integrity rather than feature dtypes."""
    out = {
        "schema_version": SCHEMA_VERSION,
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "build_ts": dt.datetime.now(tz=IST).isoformat(),
    }
    if not df.empty:
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                out[f"null_{c}"] = int(v.isna().sum())
        c_, h_, l_, o_ = (
            pd.to_numeric(df.get(x), errors="coerce")
            for x in ("close", "high", "low", "open")
        )
        bad = (
            (h_ < l_)
            | (c_ > h_ + 1e-9)
            | (c_ < l_ - 1e-9)
            | (o_ > h_ + 1e-9)
            | (o_ < l_ - 1e-9)
            | (c_ <= 0)
        )
        out["ohlc_violations"] = int(bad.fillna(False).sum())
        out["zero_volume_days"] = int(
            (pd.to_numeric(df.get("volume"), errors="coerce") == 0).sum()
        )
    return out


def _history_anchor() -> Optional[dt.date]:
    """
    Optional fixed earliest date for every symbol's stored history.

    WHY THIS MATTERS: `_expand_rank_pct` is an EXPANDING rank measured from
    index 0 of the stored frame. If the frame's first row moves between
    runs (because `start` moved, so `warmup_start` moved), the expanding
    base changes and the 10 WQ rank alphas take DIFFERENT values for dates
    that were already in the cache. Set CACHE_HISTORY_ANCHOR=2015-01-01 to
    pin the base and make those columns reproducible run-to-run.
    """
    raw = os.environ.get("CACHE_HISTORY_ANCHOR", "").strip()
    if not raw:
        return None
    try:
        return pd.Timestamp(raw).date()
    except Exception:
        return None


_SPLIT_RATIOS = (
    2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 100.0,
    1.5, 1.25, 1.2, 4.0 / 3.0, 5.0 / 3.0, 5.0 / 2.0, 7.0 / 5.0, 3.0 / 2.0,
)


def detect_price_discontinuities(
    df: pd.DataFrame,
    *,
    min_move: float = 0.30,
    ratio_tol: float = 0.04,
) -> List[dict]:
    """
    Find bars that look like an UNADJUSTED split/bonus rather than a trade.

    Signature of a corporate action stitched badly into a cache:
      * close[t]/close[t-1] is a large move, AND
      * it sits within tolerance of a clean ratio (1:2, 1:5, 1:10, 3:2 ...),
        AND
      * the bar's own intraday range is ordinary - the whole move happened
        overnight, which is what separates an adjustment artefact from a
        real limit move, AND
      * volume moves the other way (a 1:5 split multiplies share count).

    Returns a list of dicts; empty means clean.
    """
    if df is None or df.empty or len(df) < 3:
        return []
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype="float64")
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype="float64")
    l = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype="float64")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c[:-1] / c[1:]          # prev_close / close
        move = np.abs(c[1:] / c[:-1] - 1.0)
        intraday = (h[1:] - l[1:]) / np.where(c[1:] == 0, np.nan, c[1:])

    cand = np.where(np.isfinite(move) & (move >= min_move))[0]
    hits: List[dict] = []
    for i in cand:
        r = ratio[i]
        if not np.isfinite(r) or r <= 0:
            continue
        r_eff = r if r >= 1.0 else 1.0 / r
        near = None
        for target in _SPLIT_RATIOS:
            if abs(r_eff - target) / target <= ratio_tol:
                near = target
                break
        if near is None:
            continue
        # A real one-day crash/rally prints a huge intraday range. An
        # adjustment artefact does not - the gap is entirely overnight.
        if np.isfinite(intraday[i]) and intraday[i] > 0.5 * move[i]:
            continue
        # A split is a PERMANENT level shift. A crash-and-bounce is not.
        # If the level reverts within the next few sessions, this was
        # trading, not an adjustment artefact.
        fwd = c[i + 1 : i + 6]
        fwd = fwd[np.isfinite(fwd)]
        if fwd.size:
            level_after = float(np.median(fwd))
            if abs(level_after / c[i + 1] - 1.0) > 0.5 * move[i]:
                continue
        vol_ok = True
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] > 0:
            vr = v[i + 1] / v[i]
            # price down by k => share count up by ~k (and vice versa)
            expect_up = c[i + 1] < c[i]
            vol_ok = (vr > 1.3) if expect_up else (vr < 0.77)
        hits.append(
            {
                "timestamp": _maybe_iso(ts.iloc[i + 1]),
                "prev_close": float(c[i]),
                "close": float(c[i + 1]),
                "ratio": float(r),
                "nearest_split": float(near),
                "move_pct": float(move[i] * 100.0),
                "volume_consistent": bool(vol_ok),
                "confidence": "high" if vol_ok else "medium",
            }
        )
    return hits


# -------------------- Daily build --------------------

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _read_ohlcv(path: Path) -> pd.DataFrame:
    """
    v21 speed: the incremental path recomputes every indicator from raw
    OHLCV anyway, so reading the other ~200 derived columns off disk was
    pure waste. Column pushdown cuts read I/O by roughly 30x per symbol.
    """
    try:
        return read_parquet(path, columns=OHLCV_COLS)
    except (ValueError, KeyError):
        df = read_parquet(path)
        keep = [c for c in OHLCV_COLS if c in df.columns]
        return df[keep]


def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    df = _ensure_ist(df)
    return (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def _maybe_iso(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, dt.datetime)):
        ts = pd.Timestamp(val)
        ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
        return ts.isoformat()
    return str(val)


def _parse_meta_day(value) -> Optional[dt.date]:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
        return ts.date()
    except Exception:
        return None


def _cached_span(
    path: Path, meta: Optional[dict]
) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    meta = meta or {}
    first = _parse_meta_day(meta.get("first_timestamp"))
    last = _parse_meta_day(meta.get("last_timestamp"))
    if (first is None or last is None) and path.exists():
        try:
            ts_df = read_parquet(path, columns=["timestamp"])
        except (ValueError, KeyError):
            ts_df = read_parquet(path)
        if "timestamp" in ts_df.columns and not ts_df.empty:
            ts_df = _ensure_ist(ts_df)
            ts = pd.to_datetime(ts_df["timestamp"], errors="coerce")
            dates = ts.dt.date.dropna()
            actual_first = dates.min() if not dates.empty else None
            actual_last = dates.max() if not dates.empty else None
            if first is None:
                first = actual_first
            if last is None:
                last = actual_last
    return first, last


def build_daily(
    provider: KiteProvider,
    config: Config,
    symbol: str,
    start: dt.date,
    end: dt.date,
    *,
    force=True,
    recompute_only=False,
    series: Optional["Series"] = None,
    only_stale: bool = False,
) -> Path:
    """
    Fetch and store RAW OHLCV for one instrument. No indicators (v22).

    The cache's only contract: what is on disk matches what the exchange
    printed, on one consistent corporate-action basis, with no gaps. Feature
    computation lives in features_daily.py and reads this.

    `series` carries per-instrument rules (exchange, continuous futures,
    whether volume exists, whether split detection applies). Equities get
    the default.
    """
    series = series or Series.equity(symbol)
    out_pq = daily_path(config, series.cache_key)
    ok = ok_path(config, series.cache_key)
    ok_meta = read_json(ok)
    cached_first, cached_last = _cached_span(out_pq, ok_meta)
    schema_ok = (
        bool(ok_meta)
        and ok_meta.get(OK_VERSION_KEY) == SCHEMA_VERSION
        and out_pq.exists()
    )

    anchor = _history_anchor()
    fetch_start = min(start, anchor) if anchor is not None else start

    # v26: clamp to this series' last CLOSED session. MCX at 23:30 and NSE at
    # 15:30 have different answers at 16:00, which is exactly why the clamp is
    # per-series rather than global.
    session_end = last_completed_session(series)
    if end > session_end:
        end = session_end
    if end < fetch_start:
        # Nothing has closed yet in the requested window.
        return out_pq if out_pq.exists() else _save(
            pd.DataFrame(columns=OHLCV_COLS), splits=[]
        )

    roll_cal = None
    if series.continuous:
        rc = roll_calendar_path(config, series.name)
        roll_cal = read_parquet(rc) if rc.exists() else None

    def _save(df: pd.DataFrame, *, splits: Optional[List[dict]] = None) -> Path:
        first_ts = _maybe_iso(df["timestamp"].iloc[0]) if not df.empty else None
        last_ts = _maybe_iso(df["timestamp"].iloc[-1]) if not df.empty else None
        meta = ok_meta_base() | {
            "rows": int(df.shape[0]),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "requested_start": start.isoformat() if start else None,
            "requested_end": end.isoformat() if end else None,
            "history_anchor": anchor.isoformat() if anchor else None,
            "listing_start": first_ts,
            "series_kind": series.kind,
            "exchange": series.exchange,
            "tradingsymbol": series.tradingsymbol,
            "continuous": bool(series.continuous),
            "has_volume": bool(series.has_volume),
            "session_end_ist": series.session_end_ist,
            "lag_sessions": int(series.lag_sessions),
            "corporate_action_suspects": splits or [],
            **_raw_manifest(df),
        }
        with FileLock(out_pq):
            to_parquet(
                out_pq,
                df,
                engine=config.parquet_engine,
                compression=config.parquet_compression,
                use_dictionary=config.parquet_use_dictionary,
            )
            write_json_atomic(ok, meta)
        return out_pq

    def _scan(df: pd.DataFrame) -> List[dict]:
        # Split detection is meaningless for an index (no adjustment happens)
        # and actively wrong for stitched futures, where every roll is a
        # legitimate discontinuity.
        if not series.detect_splits or df.empty:
            return []
        return detect_price_discontinuities(df)

    def _fetch(a: dt.date, b: dt.date) -> pd.DataFrame:
        return _normalize_daily(provider.fetch_daily(series, a, b))

    def _full_fetch() -> Path:
        df = _fetch(fetch_start, end)
        if df.empty:
            return _save(df, splits=[])
        df, orphans = trim_orphan_bars(df)
        if orphans:
            try:
                append_token_error_log(
                    config.day_root(), symbol=series.cache_key,
                    phase="orphan_bars",
                    error=(f"dropped {len(orphans)} pre-listing bar(s) before "
                           f"the continuous series began: {orphans[:5]}"),
                )
            except Exception:
                pass
        _validate_monotonic(df)
        df = apply_roll_marks(df, roll_cal, series)
        return _save(df, splits=_scan(df))

    # ---- Recompute-only: v22 has nothing to recompute, just re-audit ----
    if recompute_only and out_pq.exists():
        df = _normalize_daily(_read_ohlcv(out_pq))
        if not df.empty:
            df, _ = trim_orphan_bars(df)
            _validate_monotonic(df)
            df = apply_roll_marks(df, roll_cal, series)
        return _save(df, splits=_scan(df))

    # ---- Incremental ----
    if out_pq.exists() and cached_last is not None and not force:
        # v25 SPEED: decide BEFORE touching the disk or the network.
        #
        # v24 unconditionally re-fetched a 7-day tail and read the whole
        # parquet for every symbol, so a rerun with nothing new still cost
        # ~1000 API calls and ~1000 file reads. The tail refetch exists to
        # repair PROVISIONAL bars - ones stored before their session
        # closed. Only a series whose session ends after we last wrote it
        # can have those, which in practice means the post-NSE-close macro
        # series (lag_sessions > 0) and any symbol still behind `end`.
        needs_tail = bool(series.lag_sessions) or cached_last < end
        if only_stale and cached_last >= end and schema_ok:
            # Explicit "skip anything already current" for a repeat run.
            return out_pq
        needs_backfill = cached_first is not None and cached_first > fetch_start
        if schema_ok and not needs_tail and not needs_backfill:
            return out_pq

        base_df = _normalize_daily(_read_ohlcv(out_pq))
        frames = [base_df]
        fetched_any = False

        if needs_backfill:
            back_df = _fetch(fetch_start, cached_first - dt.timedelta(days=1))
            if not back_df.empty:
                frames.insert(0, back_df)
                fetched_any = True

        # Overlap the tail so provisional bars get corrected (see
        # REFETCH_TAIL_DAYS). keep="last" below prefers the fresh copy.
        tail_from = cached_last - dt.timedelta(days=REFETCH_TAIL_DAYS)
        if needs_tail and tail_from <= end:
            inc_df = _fetch(max(tail_from, fetch_start), end)
            if not inc_df.empty:
                frames.append(inc_df)
                fetched_any = True

        merged = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
            if len(frames) > 1 else base_df
        )

        if (not fetched_any) and schema_ok:
            return out_pq

        _validate_monotonic(merged)

        # CORPORATE ACTION GUARD: keep="last" only ever refreshes the tail, so
        # a split leaves pre-split history welded to post-split new bars. On a
        # hit, throw the incremental result away and refetch the whole series
        # so every bar shares one adjustment basis.
        suspects = _scan(merged)
        if suspects:
            try:
                append_token_error_log(
                    config.day_root(),
                    symbol=series.cache_key,
                    phase="corporate_action",
                    error=(
                        f"{len(suspects)} split-shaped discontinuity(ies); "
                        f"forcing full refetch. First: {suspects[0]}"
                    ),
                )
            except Exception:
                pass
            return _full_fetch()

        # v25 SPEED: a tail refetch that returned identical bars means
        # there is nothing to write. Rewriting a multi-MB parquet per
        # symbol per day for no change is pure I/O.
        if schema_ok and len(merged) == len(base_df):
            same = merged[OHLCV_COLS].equals(base_df[OHLCV_COLS])
            if same:
                return out_pq
        merged = apply_roll_marks(merged, roll_cal, series)
        return _save(merged, splits=[])

    return _full_fetch()


# ==================== exogenous series =================================
#
# NIFTY, sector indices, INDIA VIX, USDINR, crude and gold are NOT stocks and
# must not be cached as if they were. Three things differ:
#
#   1. CALENDAR. MCX and CDS trade on days NSE does not, and vice versa. The
#      NSE session list is the master calendar; everything else is reindexed
#      onto it, forward-filled only, with the staleness recorded.
#
#   2. SESSION END. NSE closes 15:30. Currency runs to 17:00 and MCX to 23:30.
#      A day-t gold close did not exist when NSE closed on day t. Using it in a
#      day-t feature is a time machine that will look exactly like alpha.
#      Every series therefore carries lag_sessions, defaulting to 1 for
#      anything that closes after NSE.
#
#   3. ROLLS. Continuous futures are stitched, not back-adjusted, so there is a
#      price jump at every expiry. Derive features from within-contract returns;
#      never run a 252-day z-score over a raw stitched price.
#
# =======================================================================

NSE_CLOSE_IST = "15:30"


@dataclass(frozen=True)
class Series:
    """One cacheable instrument and the rules that apply to it."""

    name: str                      # cache key / panel column prefix
    exchange: str = "NSE"
    tradingsymbol: str = ""
    kind: str = "equity"           # equity | index | futures
    continuous: bool = False       # stitch expired contracts (NFO/MCX futures)
    has_volume: bool = True        # indices print no volume
    detect_splits: bool = True     # meaningless for indices, wrong for futures
    session_end_ist: str = NSE_CLOSE_IST
    lag_sessions: int = 0          # sessions to lag before panel use
    underlying: str = ""           # instrument-dump `name` for futures
    front_expiry: Optional[dt.date] = None

    @property
    def cache_key(self) -> str:
        return self.name

    @property
    def qualified(self) -> str:
        return f"{self.exchange}:{self.tradingsymbol or self.name}"

    @staticmethod
    def equity(symbol: str) -> "Series":
        return Series(name=symbol, tradingsymbol=symbol)

    @staticmethod
    def index(name: str, tradingsymbol: str) -> "Series":
        return Series(
            name=name,
            exchange="NSE",
            tradingsymbol=tradingsymbol,
            kind="index",
            has_volume=False,
            detect_splits=False,
            session_end_ist=NSE_CLOSE_IST,
            lag_sessions=0,
        )

    @staticmethod
    def future(
        name: str,
        exchange: str,
        tradingsymbol: str,
        *,
        session_end_ist: str,
    ) -> "Series":
        # lag_sessions=1 whenever the contract closes after NSE does.
        return Series(
            name=name,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            underlying=tradingsymbol,
            kind="futures",
            continuous=True,
            has_volume=True,
            detect_splits=False,
            session_end_ist=session_end_ist,
            lag_sessions=1 if session_end_ist > NSE_CLOSE_IST else 0,
        )


# Default exogenous set, cached alongside equities on every run.
# Futures tradingsymbols are the CURRENT front-month; Kite's continuous=1
# walks back through that instrument's expired contracts from there, so the
# symbol must be refreshed each expiry (see resolve_front_month below).
DEFAULT_EXOGENOUS: List[Series] = [
    Series.index("NIFTY50", "NIFTY 50"),
    Series.index("NIFTYBANK", "NIFTY BANK"),
    Series.index("NIFTY500", "NIFTY 500"),
    Series.index("NIFTYMIDCAP150", "NIFTY MIDCAP 150"),
    Series.index("NIFTYSMLCAP250", "NIFTY SMLCAP 250"),
    Series.index("INDIAVIX", "INDIA VIX"),
    # Sector indices
    Series.index("NIFTYAUTO", "NIFTY AUTO"),
    Series.index("NIFTYFMCG", "NIFTY FMCG"),
    Series.index("NIFTYIT", "NIFTY IT"),
    Series.index("NIFTYMETAL", "NIFTY METAL"),
    Series.index("NIFTYPHARMA", "NIFTY PHARMA"),
    Series.index("NIFTYREALTY", "NIFTY REALTY"),
    Series.index("NIFTYENERGY", "NIFTY ENERGY"),
    Series.index("NIFTYPSUBANK", "NIFTY PSU BANK"),
    Series.index("NIFTYFINSERVICE", "NIFTY FIN SERVICE"),
    Series.index("NIFTYMEDIA", "NIFTY MEDIA"),
    Series.index("NIFTYCONSUMPTION", "NIFTY CONSUMPTION"),
    Series.index("NIFTYINFRA", "NIFTY INFRA"),
    # Macro. Front-month symbols are resolved at run time.
    Series.future("USDINR", "CDS", "USDINR", session_end_ist="17:00"),
    Series.future("CRUDEOIL", "MCX", "CRUDEOIL", session_end_ist="23:30"),
    Series.future("GOLD", "MCX", "GOLD", session_end_ist="23:30"),
]


def resolve_front_month(provider: KiteProvider, s: Series) -> Series:
    """
    Point a futures Series at the nearest unexpired contract.

    Kite's continuous=1 returns day candles for the *given instrument's*
    expired contracts, so the token must belong to a live contract. Exchanges
    also reuse instrument tokens after expiry, which is why the contract is
    re-resolved by tradingsymbol on every run rather than cached.
    """
    if not s.continuous:
        return s
    # v23: NO silent fallback. Returning the unresolved Series handed back
    # a base symbol that either failed later or, worse, resolved to some
    # other contract. Research infrastructure must refuse to run rather
    # than quietly use data it cannot vouch for.
    rows = provider._instrument_dump(s.exchange)
    today = today_ist()
    best, best_exp = None, None
    for r in rows:
        if str(r.get("instrument_type", "")).upper() != "FUT":
            continue
        if str(r.get("name", "")).upper() != s.tradingsymbol.upper():
            continue
        exp = r.get("expiry")
        if isinstance(exp, str):
            try:
                exp = dt.date.fromisoformat(exp[:10])
            except Exception:
                continue
        if isinstance(exp, dt.datetime):
            exp = exp.date()
        if not isinstance(exp, dt.date) or exp < today:
            continue
        if best_exp is None or exp < best_exp:
            best, best_exp = str(r.get("tradingsymbol", "")), exp
    if not best:
        raise UnresolvedSymbol(
            f"no live {s.exchange} FUT contract for {s.tradingsymbol} "
            f"(expiry >= {today.isoformat()}); refusing to guess"
        )
    return replace(s, tradingsymbol=best, front_expiry=best_exp)


def roll_calendar_path(config: Config, name: str) -> Path:
    p = config.day_root() / "roll_calendars"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.parquet"


def update_roll_calendar(
    provider: KiteProvider, config: Config, series: "Series"
) -> pd.DataFrame:
    """
    Record every live contract expiry we can currently see, append-only.

    WHY OBSERVED AND NOT DERIVED
    ----------------------------
    Kite's instrument dump contains LIVE contracts only - expired ones are
    gone. So the honest calendar is one we accumulate: every run writes down
    the expiries visible today, and the record grows forward from the day you
    start running it. History before that first run is simply unknown, and
    v23 says so rather than guessing (see apply_roll_marks).

    This replaces v22's _mark_rolls, which inferred rolls from unusual price
    moves. That was wrong in both directions: a genuine crude shock was
    labelled a roll, and a quiet roll was missed entirely.
    """
    if not series.continuous:
        return pd.DataFrame()
    path = roll_calendar_path(config, series.name)
    rows = provider._instrument_dump(series.exchange)
    base = (series.underlying or series.tradingsymbol).upper()
    seen: List[dict] = []
    for r in rows:
        if str(r.get("instrument_type", "")).upper() != "FUT":
            continue
        if str(r.get("name", "")).upper() != base:
            continue
        exp = r.get("expiry")
        if isinstance(exp, str):
            try:
                exp = dt.date.fromisoformat(exp[:10])
            except Exception:
                continue
        if isinstance(exp, dt.datetime):
            exp = exp.date()
        if not isinstance(exp, dt.date):
            continue
        seen.append(
            {
                "expiry": pd.Timestamp(exp),
                "tradingsymbol": str(r.get("tradingsymbol", "")),
                "observed_on": pd.Timestamp(today_ist()),
            }
        )
    new = pd.DataFrame(seen)
    if path.exists():
        old = read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    if new.empty:
        return new
    new = (
        new.sort_values(["expiry", "observed_on"])
        .drop_duplicates("expiry", keep="first")   # keep FIRST observation
        .reset_index(drop=True)
    )
    with FileLock(path):
        to_parquet(
            path,
            new,
            engine=config.parquet_engine,
            compression=config.parquet_compression,
            use_dictionary=config.parquet_use_dictionary,
        )
    return new


def apply_roll_marks(
    df: pd.DataFrame, calendar: Optional[pd.DataFrame], series: "Series"
) -> pd.DataFrame:
    """
    Mark contract rolls from the observed expiry calendar. Deterministic.

    D_roll is a NULLABLE Int8 with three states, and the third is the point:

        1    this session is the first after a known expiry -> a roll
        0    this session is inside a known contract        -> not a roll
        NA   this session predates the calendar             -> WE DO NOT KNOW

    v22 wrote 0 for everything it had not flagged, which claimed knowledge it
    did not have. NA forces the feature layer to make a decision: either drop
    futures history before the calendar starts, or accept that differences
    there may span a roll. Silence is not an option.
    """
    if df is None or df.empty or not series.continuous:
        return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    tz = getattr(ts.dt, "tz", None)
    out = pd.Series(pd.NA, index=df.index, dtype="Int8")

    if calendar is not None and not calendar.empty:
        exps = pd.to_datetime(calendar["expiry"]).dt.tz_localize(None).sort_values()
        naive = ts.dt.tz_localize(None) if tz is not None else ts
        known_from = exps.min()
        in_range = naive >= known_from
        out[in_range] = 0
        # First session strictly after each expiry is the roll.
        for e in exps:
            after = naive > e
            if after.any():
                out.iloc[int(np.argmax(after.to_numpy()))] = 1

    df["D_roll"] = out
    return df


def align_to_sessions(
    df: pd.DataFrame,
    sessions: Sequence[pd.Timestamp],
    *,
    series: Series,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Put an exogenous series onto the NSE trading calendar, safely.

    Forward-fill only, never backfill, and lag by series.lag_sessions so a
    close that printed after 15:30 IST can never appear in a same-day feature.
    `<prefix>_stale` counts sessions since the value actually changed, so the
    panel can down-weight or drop held-over values instead of silently
    treating a four-day-old crude print as today's.
    """
    px = prefix or series.name
    idx = pd.DatetimeIndex(sessions)
    if df is None or df.empty:
        return pd.DataFrame(index=idx)

    w = df.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce")
    w = w.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if w.index.tz is None:
        w.index = w.index.tz_localize(IST)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    w.index = w.index.normalize()
    idx_n = idx.normalize()

    # v28: normalising to midnight can collapse two bars onto one date, and
    # reindex refuses a duplicated index ("cannot reindex on an axis with
    # duplicate labels"). Keep the LAST bar for a date - the same rule the
    # cache uses when merging a refetched tail, so the settled print wins.
    if w.index.has_duplicates:
        w = w[~w.index.duplicated(keep="last")]
    if idx_n.has_duplicates:
        idx_n = pd.DatetimeIndex(idx_n).drop_duplicates()
        idx = pd.DatetimeIndex(idx).drop_duplicates()
    w = w.sort_index()

    cols = [c for c in ("open", "high", "low", "close", "volume", "D_roll") if c in w.columns]
    if not series.has_volume:
        cols = [c for c in cols if c != "volume"]
    w = w[cols]

    fresh = w.reindex(idx_n).notna().any(axis=1)
    out = w.reindex(idx_n).ffill()

    if series.lag_sessions:
        out = out.shift(series.lag_sessions)
        # v29: .fillna(False) on an object-dtype column triggers a pandas
        # downcasting FutureWarning that floods the log. Cast explicitly
        # rather than relying on the deprecated silent behaviour.
        fresh = fresh.shift(series.lag_sessions).fillna(False).astype(bool)

    grp = fresh.cumsum()
    stale = fresh.groupby(grp).cumcount()
    out[f"{px}_stale"] = stale.to_numpy().astype("int16")
    out.columns = [
        c if c.endswith("_stale") else f"{px}_{c}" for c in out.columns
    ]
    out.index = idx
    return out


def session_gaps(
    stored: Sequence[pd.Timestamp],
    master: Sequence[pd.Timestamp],
) -> List[dt.date]:
    """
    Sessions the master calendar had that this series does not, WITHIN the
    series' own first..last span.

    This is the check v22 was missing. Comparing only last_timestamp means a
    series can be missing forty sessions in the middle of 2023 and still
    report as perfectly current - the exact failure mode where data looks
    clean while being wrong.

    Bounded by the series' own span on purpose: a stock listed in 2023 is not
    "missing" 2018, and a delisted one is not missing last week.
    """
    if stored is None or len(stored) == 0:
        return []
    s_idx = pd.DatetimeIndex(pd.to_datetime(list(stored))).tz_localize(None).normalize()
    m_idx = pd.DatetimeIndex(pd.to_datetime(list(master))).tz_localize(None).normalize()
    lo, hi = s_idx.min(), s_idx.max()
    m_in = m_idx[(m_idx >= lo) & (m_idx <= hi)]
    missing = m_in.difference(s_idx)
    return [d.date() for d in missing]


def summarize_gaps(missing: Sequence[dt.date], max_runs: int = 5) -> List[str]:
    """Collapse missing dates into contiguous runs for readable reporting."""
    if not missing:
        return []
    ds = sorted(missing)
    runs, start, prev = [], ds[0], ds[0]
    for d in ds[1:]:
        if (d - prev).days <= 4:      # tolerate weekends inside a run
            prev = d
            continue
        runs.append((start, prev))
        start = prev = d
    runs.append((start, prev))
    out = []
    for a, b in runs[:max_runs]:
        out.append(a.isoformat() if a == b else f"{a.isoformat()}..{b.isoformat()}")
    if len(runs) > max_runs:
        out.append(f"+{len(runs) - max_runs} more runs")
    return out


def nse_sessions(config: Config, reference: str = "NIFTY50") -> pd.DatetimeIndex:
    """
    The master trading calendar, taken from cached index data rather than a
    hardcoded holiday list, so it can never drift out of date.
    """
    p = daily_path(config, reference)
    if not p.exists():
        raise FileNotFoundError(
            f"{reference} is not cached; it is the master calendar for the panel."
        )
    ts = pd.to_datetime(read_parquet(p, columns=["timestamp"])["timestamp"])
    return pd.DatetimeIndex(ts).sort_values()


# -------------------- Symbols file loader --------------------

def _load_symbols_from_file(path: str) -> List[str]:
    p = Path(path)
    ext = p.suffix.lower()
    items: List[str] = []
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(p)
        if df.empty:
            return []
        col0 = df.columns[0]
        items = [str(x) for x in df[col0].dropna().tolist()]
    else:
        try:
            df = pd.read_csv(p, header=None)
            items = (
                [str(x) for x in df.iloc[:, 0].dropna().tolist()]
                if df.shape[1] >= 1
                else []
            )
        except Exception:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            tokens = [
                t.strip()
                for t in raw.replace("\n", ",").replace("\t", ",").split(",")
            ]
            items = [t for t in tokens if t]
    cleaned: List[str] = []
    seen: set = set()
    for t in items:
        s = sanitize_symbol(t)
        if s and s.casefold() not in seen:
            seen.add(s.casefold())
            cleaned.append(s)
    return cleaned


# -------------------- Date range normalization --------------------

def parse_date_input(value) -> dt.date:
    if isinstance(value, dt.date):
        return value
    if value is None:
        raise ValueError("Date value is required")
    text = str(value).strip()
    if not text:
        raise ValueError("Date value is required")
    formats = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y")
    for fmt in formats:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {text!r}")


def normalize_requested_range(
    start_value, end_value
) -> Tuple[dt.date, dt.date, List[str]]:
    start = parse_date_input(start_value)
    end = parse_date_input(end_value)
    if start > end:
        start, end = end, start
    notes: List[str] = []
    today = today_ist()
    if end > today:
        notes.append(
            f"End date {end.isoformat()} trimmed to {today.isoformat()} "
            "because future data is unavailable."
        )
        end = today
    if start > today:
        notes.append(
            f"Start date {start.isoformat()} adjusted to {today.isoformat()} "
            "because the market has not traded yet."
        )
        start = today
    if start > end:
        raise ValueError(
            "Requested date range does not contain any trading days after adjustments."
        )
    return start, end, notes


def append_token_error_log(
    base_dir: Path, *, symbol: str, phase: str, error: str
) -> None:
    log_path = (
        (base_dir / "_token_expired.log") if base_dir else Path("_token_expired.log")
    )
    rec = {
        "ts": dt.datetime.now(tz=IST).isoformat(),
        "symbol": symbol,
        "day": None,
        "phase": phase,
        "error": str(error),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------- GUI inputs --------------------

def _ask_user_inputs_gui_file_only():
    if not TK_OK:
        raise SystemExit("Tkinter is not available.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select symbols file (Excel/CSV/TXT; first column = symbols)",
        filetypes=[
            ("Excel/CSV/TXT", "*.xlsx *.xls *.csv *.txt"),
            ("All", "*.*"),
        ],
    )
    if not path:
        messagebox.showerror("Required", "No symbols file selected.")
        raise SystemExit(1)
    symbols = _load_symbols_from_file(path)
    if not symbols:
        messagebox.showerror(
            "Invalid file", "Could not parse any symbols from the file."
        )
        raise SystemExit(1)
    today = today_ist()
    default_start = today - dt.timedelta(days=600)
    start_input = simpledialog.askstring(
        "Start date",
        "Enter start date (YYYY-MM-DD or DD-MM-YYYY):",
        initialvalue=default_start.isoformat(),
    )
    end_input = simpledialog.askstring(
        "End date",
        "Enter end date (YYYY-MM-DD or DD-MM-YYYY):",
        initialvalue=today.isoformat(),
    )
    try:
        start_date, end_date, adjustments = normalize_requested_range(
            start_input, end_input
        )
    except Exception as exc:
        messagebox.showerror("Invalid dates", str(exc))
        raise SystemExit(1)
    if adjustments:
        messagebox.showinfo("Adjusted dates", "\n".join(adjustments))
    mw = simpledialog.askinteger(
        "Parallel workers",
        "Max threads (IO-bound; 16-64 works well):",
        initialvalue=32,
        minvalue=1,
        maxvalue=128,
    )
    if not mw:
        mw = 32
    base_config = Config.from_env()
    messagebox.showinfo(
        "Summary",
        (
            "Symbols file: {}\n\nSymbols parsed: {}\n\n"
            "Date range: {} to {}\nWorkers: {}\n\nDaily folder:\n{}"
        ).format(
            path,
            len(symbols),
            start_date.isoformat(),
            end_date.isoformat(),
            mw,
            str(base_config.daily_root),
        ),
    )
    root.destroy()
    return {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "max_workers": mw,
        "base_config": base_config,
    }


# -------------------- CLI --------------------

def parse_cli_args():
    import argparse

    ap = argparse.ArgumentParser(
        description="FAST daily-only cache builder for Kite (no intraday)."
    )
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--only-stale", action="store_true",
                    help="skip symbols already current; for a repeat run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rate", type=float, default=KITE_HISTORICAL_RPS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute-only", action="store_true")
    ap.add_argument("--incremental", action="store_true")
    args = ap.parse_args()
    syms: Optional[List[str]] = args.symbols
    if args.symbols_file:
        syms = _load_symbols_from_file(args.symbols_file)
    return args, syms


def _resolve_requested_dates(
    start_value, end_value, days_value, *, incremental: bool = False
) -> Tuple[dt.date, dt.date]:
    if incremental:
        today = today_ist()
        start = today - dt.timedelta(days=600)
        start, end, _ = normalize_requested_range(start, today)
        return start, end
    if start_value and end_value:
        start, end, _ = normalize_requested_range(start_value, end_value)
        return start, end
    if days_value and days_value > 0:
        today = today_ist()
        start = today - dt.timedelta(days=days_value)
        start, end, _ = normalize_requested_range(start, today)
        return start, end
    raise SystemExit(
        "Provide --start-date/--end-date or --days to define the caching window."
    )


# -------------------- Pipeline --------------------

class Pipeline:
    def __init__(
        self,
        provider: KiteProvider,
        config: Config,
        progress_cb: Optional[callable] = None,
    ):
        self.provider = provider
        self.cfg = config
        self.ratelimiter = RateLimiter(config.rate_limit_per_sec)
        self._daily_cache = DataFrameCache(maxsize=256)
        self.fetch_daily = with_retry(
            self._cached_fetch_daily,
            tries=config.retry_tries,
            backoff=config.retry_backoff_base,
        )
        self.progress_cb = progress_cb or (lambda msg: None)

    def _cached_fetch_daily(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        key = (symbol, start, end)
        cached = self._daily_cache.get(key)
        if cached is not None:
            return cached
        self.ratelimiter.acquire()
        df = self.provider.fetch_daily(symbol, start, end)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fetch_daily must return a DataFrame")
        return self._daily_cache.put(key, df)

    def build(
        self,
        symbols: Sequence[str],
        start_date: dt.date,
        end_date: dt.date,
        *,
        force: bool,
        recompute_only: bool,
        include_exogenous: bool = True,
        exogenous: Optional[Sequence["Series"]] = None,
        strict_exogenous: bool = True,
        only_stale: bool = False,
    ):
        cfg = self.cfg
        unresolved_exo: List[str] = []
        start_date, end_date, adjustments = normalize_requested_range(
            start_date, end_date
        )
        for note in adjustments:
            try:
                self.progress_cb(f"NOTE: {note}")
            except Exception:
                print(note)

        # v22: exogenous series are cached on EVERY run, before equities.
        # NIFTY50 is the master trading calendar for panel alignment, so a
        # run that skipped it would leave the panel with nothing to align
        # to. Futures contracts are re-resolved to the live front month
        # each run: exchanges reuse instrument tokens after expiry, so a
        # cached token silently returns a different contract's history.
        exo: List[Series] = []
        if include_exogenous:
            for e in (exogenous if exogenous is not None else DEFAULT_EXOGENOUS):
                try:
                    e2 = resolve_front_month(self.provider, e)
                    _ = self.provider.token_for(e2)
                    if e2.continuous:
                        update_roll_calendar(self.provider, cfg, e2)
                    exo.append(e2)
                except Exception as ex:
                    unresolved_exo.append(f"{e.qualified} ({ex})")
                    append_token_error_log(
                        cfg.day_root(),
                        symbol=e.qualified,
                        phase="preresolve_exogenous",
                        error=str(ex),
                    )
                    if strict_exogenous:
                        # v23: an unresolvable macro series is not a
                        # degraded run, it is a wrong one. The panel would
                        # silently lose a whole feature family.
                        raise RuntimeError(
                            f"exogenous series {e.qualified} failed to "
                            f"resolve: {ex}"
                        ) from ex

        pre: List[Series] = []
        unresolved: List[str] = []
        for sym in symbols:
            try:
                _ = self.provider._symbol_to_instrument_token(sym)
                pre.append(Series.equity(sym))
            except UnresolvedSymbol as e:
                unresolved.append(str(e))
                append_token_error_log(
                    cfg.day_root(),
                    symbol=str(e),
                    phase="preresolve",
                    error="UNRESOLVED_SYMBOL",
                )
                continue
        work: List[Series] = exo + pre

        def task(s: "Series"):
            try:
                # v21: warm-up is applied ONCE, inside build_daily. v20
                # subtracted WARMUP_DAYS here and again in build_daily,
                # so the real lookback was silently 2x the configured
                # value and `first_valid` was impossible to reason about.
                return build_daily(
                    self.provider,
                    cfg,
                    s.cache_key,
                    start_date,
                    end_date,
                    force=force,
                    recompute_only=recompute_only,
                    series=s,
                    only_stale=only_stale,
                )
            except UnresolvedSymbol:
                self.progress_cb(f"SKIP unresolved: {s} (daily)")
            except AuthExpired as e:
                append_token_error_log(
                    cfg.day_root(), symbol=s, phase="daily", error=str(e)
                )
                raise
            except Exception as e:
                append_token_error_log(
                    cfg.day_root(), symbol=s, phase="daily", error=str(e)
                )
                raise

        results: list = []
        built: Dict[str, Path] = {}
        failed: Dict[str, str] = {}
        kinds: Dict[str, str] = {w.cache_key: w.kind for w in work}
        with cf.ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futures = {ex.submit(task, w): w for w in work}
            for fut in cf.as_completed(futures):
                w = futures[fut]
                key = w.cache_key
                try:
                    p = fut.result()
                    results.append(p)
                    if p is not None:
                        built[key] = p
                    else:
                        failed[key] = "unresolved"
                    self.progress_cb(f"Daily: {key}")
                except AuthExpired:
                    raise
                except Exception as e:
                    failed[key] = str(e)
                    self.progress_cb(f"ERROR {key}: {e}")

        if unresolved:
            summary = (
                f"Unresolved symbols skipped ({len(unresolved)}): "
                + ", ".join(sorted(set(unresolved)))
            )
            print(summary)
            notify("Unresolved", summary)

        self._coverage_report(
            built=built,
            failed=failed,
            unresolved=sorted(set(unresolved)) + sorted(set(unresolved_exo)),
            end_date=end_date,
            kinds=kinds,
        )

        return results


    # ------------------------------------------------------------------
    # v21: end-of-run coverage audit
    # ------------------------------------------------------------------
    def _coverage_report(
        self,
        *,
        built: Dict[str, Path],
        failed: Dict[str, str],
        unresolved: Sequence[str],
        end_date: dt.date,
        kinds: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Flag every symbol that is NOT current as of the run's end date.

        The expected last session is taken from the data itself - the latest
        timestamp any symbol reached - rather than from a hardcoded holiday
        calendar. If the whole universe stops at Friday because Monday was a
        holiday, nothing is flagged; if ONE symbol stops at Friday while the
        rest reached Monday, that symbol is flagged. That is the failure you
        actually care about: a silently stale name entering the panel with a
        stale last bar.
        """
        cfg = self.cfg
        rows: List[dict] = []
        suspects: List[dict] = []
        kinds = kinds or {}

        # v23: the master session list, used to find gaps INSIDE each
        # series rather than only checking its last date. Taken from the
        # cached index so it can never drift from a hardcoded holiday list.
        master: Optional[pd.DatetimeIndex] = None
        try:
            master = nse_sessions(cfg)
        except Exception as e:
            self.progress_cb(f"WARNING: no master calendar ({e}); gap check skipped")

        for sym, path in built.items():
            meta = read_json(ok_path(cfg, sym)) or {}
            last = _parse_meta_day(meta.get("last_timestamp"))
            if last is None:
                _, last = _cached_span(path, meta)
            missing: List[dt.date] = []
            # Only NSE-calendar series are checked against NSE sessions.
            # MCX and CDS trade on days NSE does not, so measuring them
            # against this calendar would manufacture phantom gaps.
            if master is not None and kinds.get(sym, "equity") in ("equity", "index"):
                try:
                    ts = pd.to_datetime(
                        read_parquet(path, columns=["timestamp"])["timestamp"]
                    )
                    missing = session_gaps(ts, master)
                except Exception:
                    missing = []
            rows.append(
                {
                    "symbol": sym,
                    "last_timestamp": last.isoformat() if last else None,
                    # v24: rows_valid / first_valid_timestamp were warm-up
                    # concepts that left with the indicators in v22. _save
                    # never wrote them again, so meta.get() returned None ->
                    # 0 for EVERY symbol, and "ZERO USABLE ROWS" fired for
                    # the whole universe on every run. The raw cache has
                    # only one row count.
                    "rows": int(meta.get("rows") or 0),
                    "missing_sessions": len(missing),
                    "missing_runs": summarize_gaps(missing),
                }
            )
            for hit in meta.get("corporate_action_suspects") or []:
                suspects.append({"symbol": sym, **hit})

        # The expected last session is taken from EQUITIES only. MCX and
        # CDS trade on days NSE does not, so including them would push the
        # benchmark past the last NSE session and flag every stock stale.
        for r in rows:
            r["kind"] = kinds.get(r["symbol"], "equity")
        eq_dates = [
            dt.date.fromisoformat(r["last_timestamp"])
            for r in rows
            if r["last_timestamp"] and r["kind"] == "equity"
        ]
        all_dates = [
            dt.date.fromisoformat(r["last_timestamp"])
            for r in rows
            if r["last_timestamp"]
        ]
        expected_last = max(eq_dates) if eq_dates else (max(all_dates) if all_dates else None)

        # An exogenous series only needs to be current as of the last NSE
        # session; it is allowed to be AHEAD (MCX trades later and on more
        # days). Missing entirely is always a failure - a stale NIFTY50
        # means the panel's master calendar is stale.
        stale = [
            r
            for r in rows
            if r["last_timestamp"] is None
            or (
                expected_last
                and dt.date.fromisoformat(r["last_timestamp"]) < expected_last
            )
        ]
        stale.sort(key=lambda r: (r["last_timestamp"] or "", r["symbol"]))

        empty = [r["symbol"] for r in rows if r["rows"] == 0]
        gappy = sorted(
            (r for r in rows if r["missing_sessions"] > 0),
            key=lambda r: -r["missing_sessions"],
        )

        report = {
            "run_end_date": end_date.isoformat(),
            "expected_last_session": expected_last.isoformat() if expected_last else None,
            "symbols_requested": len(built) + len(failed) + len(unresolved),
            "symbols_built": len(built),
            "symbols_current": len(rows) - len(stale),
            "symbols_stale": len(stale),
            "symbols_with_gaps": len(gappy),
            "gaps": [
                {
                    "symbol": r["symbol"],
                    "missing_sessions": r["missing_sessions"],
                    "runs": r["missing_runs"],
                }
                for r in gappy
            ],
            "stale": stale,
            "failed": [{"symbol": k, "error": v} for k, v in sorted(failed.items())],
            "unresolved": list(unresolved),
            "no_valid_rows": empty,
            "corporate_action_suspects": suspects,
            "exogenous": [r for r in rows if r["kind"] != "equity"],
            "generated_at": dt.datetime.now(tz=IST).isoformat(),
        }

        try:
            write_json_atomic(cfg.day_root() / "cache_coverage_report.json", report)
        except Exception:
            pass

        lines: List[str] = []
        lines.append("")
        lines.append("=" * 68)
        lines.append("CACHE COVERAGE REPORT")
        lines.append("=" * 68)
        lines.append(f"  Requested end date   : {end_date.isoformat()}")
        lines.append(
            f"  Last session reached : "
            f"{expected_last.isoformat() if expected_last else 'n/a'}"
        )
        lines.append(
            f"  Built {len(built)} | current {len(rows) - len(stale)} | "
            f"STALE {len(stale)} | GAPS {len(gappy)} | failed {len(failed)} | "
            f"unresolved {len(unresolved)}"
        )
        if master is None:
            lines.append(
                "  WARNING: gap check did NOT run - no master calendar. "
                "'current' below means last-date-only and proves nothing."
            )

        if expected_last and expected_last < end_date:
            lines.append(
                f"  NOTE: no symbol reached {end_date.isoformat()} - likely a "
                f"market holiday or a run before close."
            )

        if stale:
            lines.append("")
            lines.append(f"  NOT CACHED TO {expected_last}:")
            for r in stale[:60]:
                lines.append(
                    f"    {r['symbol']:<18} last={r['last_timestamp'] or 'NONE':<12} "
                    f"rows={r['rows']}"
                )
            if len(stale) > 60:
                lines.append(f"    ... +{len(stale) - 60} more (see JSON report)")

        if gappy:
            lines.append("")
            lines.append(
                f"  INTERIOR SESSION GAPS ({len(gappy)}) - these look "
                f"current but are NOT complete:"
            )
            for r in gappy[:40]:
                lines.append(
                    f"    {r['symbol']:<18} missing={r['missing_sessions']:<5} "
                    f"{', '.join(r['missing_runs'])}"
                )
            if len(gappy) > 40:
                lines.append(f"    ... +{len(gappy) - 40} more (see JSON report)")

        if empty:
            lines.append("")
            lines.append(
                f"  ZERO USABLE ROWS ({len(empty)}): "
                + ", ".join(empty[:25])
                + (" ..." if len(empty) > 25 else "")
            )

        if failed:
            lines.append("")
            lines.append(f"  FAILED ({len(failed)}):")
            for k, v in sorted(failed.items())[:30]:
                lines.append(f"    {k:<18} {v[:70]}")

        if suspects:
            lines.append("")
            lines.append(
                f"  CORPORATE ACTION SUSPECTS ({len(suspects)}) - verify the "
                f"adjustment basis before trusting these symbols:"
            )
            for h in suspects[:25]:
                lines.append(
                    f"    {h['symbol']:<18} {str(h.get('timestamp'))[:10]} "
                    f"move={h.get('move_pct', 0):.1f}% "
                    f"~1:{h.get('nearest_split')} ({h.get('confidence')})"
                )

        exo_rows = [r for r in rows if r["kind"] != "equity"]
        if exo_rows:
            missing_exo = [r for r in exo_rows if r in stale]
            lines.append("")
            lines.append(
                f"  EXOGENOUS: {len(exo_rows) - len(missing_exo)}/{len(exo_rows)} "
                f"current"
            )
            for r in exo_rows:
                mark = "STALE" if r in stale else "ok"
                lines.append(
                    f"    {r['symbol']:<18} {r['kind']:<8} "
                    f"last={r['last_timestamp'] or 'NONE':<12} {mark}"
                )
            if not any(r["symbol"] == "NIFTY50" and r not in stale for r in exo_rows):
                lines.append(
                    "    *** NIFTY50 is the panel's master calendar. "
                    "Do not build a panel until it is current. ***"
                )

        lines.append("")
        lines.append(f"  JSON: {cfg.day_root() / 'cache_coverage_report.json'}")
        lines.append("=" * 68)

        text = "\n".join(lines)
        print(text)
        try:
            self.progress_cb(text)
        except Exception:
            pass

        if stale or failed or empty or gappy:
            try:
                if TK_OK:
                    tk.messagebox.showwarning(
                        "Cache coverage",
                        f"{len(stale)} stale, {len(failed)} failed, "
                        f"{len(empty)} empty.\nSee cache_coverage_report.json",
                    )
            except Exception:
                pass

        return report


# -------------------- Entry --------------------

def main():
    # v22: the leak gates moved to features_daily.py with the code they
    # protect. The cache itself computes nothing, so it has nothing to leak.
    # Feature builds must call features_daily.assert_leak_free() themselves.

    try:
        if len(sys.argv) > 1:
            # v28: a CLI invocation is headless by definition. This is the
            # path Task Scheduler takes, where no desktop session exists and
            # a modal dialog would hang the job forever.
            set_headless(True)
            args, symbols = parse_cli_args()
            if not symbols:
                raise SystemExit(
                    "No symbols provided. Use --symbols-file <path> or --symbols ..."
                )
            base_config = Config.from_env()
            cfg = base_config.with_updates(
                max_workers=int(args.workers),
                rate_limit_per_sec=float(args.rate),
                request_timeout_s=15.0,
                retry_tries=6,
            )
            provider = KiteProvider()
            pipeline = Pipeline(provider, cfg, progress_cb=lambda m: print(m))
            cfg.daily_root.mkdir(parents=True, exist_ok=True)
            start_date, end_date = _resolve_requested_dates(
                args.start_date,
                args.end_date,
                args.days,
                incremental=bool(args.incremental),
            )
            pipeline.build(
                symbols,
                start_date,
                end_date,
                force=bool(args.force),
                recompute_only=bool(args.recompute_only),
                only_stale=bool(getattr(args, "only_stale", False)),
            )
            print("FAST daily-only cache build completed.")
        else:
            ui = _ask_user_inputs_gui_file_only()
            base_config = ui.get("base_config") or Config.from_env()
            cfg = base_config.with_updates(
                max_workers=int(ui["max_workers"]),
                rate_limit_per_sec=KITE_HISTORICAL_RPS,
                request_timeout_s=15.0,
                retry_tries=6,
            )
            symbols = ui["symbols"]
            start_date = ui["start_date"]
            end_date = ui["end_date"]

            if not TK_OK:
                raise SystemExit("Tkinter not available; GUI mode is required.")

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            class ProgressUI:
                def __init__(self, total: int):
                    self.total = max(1, int(total))
                    self.start = time.perf_counter()
                    self.completed = 0
                    self.root = tk.Toplevel()
                    self.root.title("Building daily cache (FAST)...")
                    self.root.geometry("560x160")
                    self.root.resizable(False, False)
                    self.label = tk.Label(
                        self.root, text="Starting...", anchor="w"
                    )
                    self.label.pack(fill="x", padx=12, pady=(12, 6))
                    self.pb = ttk.Progressbar(
                        self.root,
                        orient="horizontal",
                        mode="determinate",
                        maximum=self.total,
                        length=520,
                    )
                    self.pb.pack(padx=12, pady=6)
                    self.eta = tk.Label(self.root, text="ETA: --:--", anchor="w")
                    self.eta.pack(fill="x", padx=12, pady=(6, 12))
                    self.root.attributes("-topmost", True)
                    self.root.update_idletasks()

                def _fmt_eta(self, secs: float) -> str:
                    if secs is None or secs != secs or secs == float("inf"):
                        return "--:--"
                    m, s = divmod(int(secs), 60)
                    h, m = divmod(m, 60)
                    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

                def tick(self, msg: str = ""):
                    self.completed += 1
                    self.pb["value"] = self.completed
                    elapsed = max(0.001, time.perf_counter() - self.start)
                    rate = self.completed / elapsed
                    remaining = max(0, self.total - self.completed)
                    # v25: this displayed `elapsed` under an "ETA:" label,
                    # so it counted UP while claiming to count down. rate
                    # and remaining were computed and then thrown away.
                    eta_s = remaining / rate if rate > 0 else float("inf")
                    self.label.config(
                        text=msg or f"Completed {self.completed}/{self.total}"
                    )
                    self.eta.config(
                        text=(
                            f"ETA: {self._fmt_eta(eta_s)}  "
                            f"Elapsed: {self._fmt_eta(elapsed)}"
                        )
                    )
                    self.root.update_idletasks()

                def done(self):
                    self.pb["value"] = self.total
                    self.label.config(
                        text=f"Done: {self.total}/{self.total}"
                    )
                    self.eta.config(
                        text=f"ETA: 00:00  Elapsed: {self._fmt_eta(time.perf_counter() - self.start)}"
                    )
                    self.root.update_idletasks()

            pui = ProgressUI(total=len(symbols))

            def on_progress(msg: str):
                pui.tick(msg)

            provider = KiteProvider()
            pipeline = Pipeline(provider, cfg, progress_cb=on_progress)
            cfg.daily_root.mkdir(parents=True, exist_ok=True)
            pipeline.build(symbols, start_date, end_date, force=False, recompute_only=False)
            pui.done()
            messagebox.showinfo("Done", "FAST daily-only cache build completed.")

    except Exception as e:
        if TK_OK:
            try:
                messagebox.showerror("Error", str(e))
            except Exception:
                print("ERROR:", e, file=sys.stderr)
        else:
            print("ERROR:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()