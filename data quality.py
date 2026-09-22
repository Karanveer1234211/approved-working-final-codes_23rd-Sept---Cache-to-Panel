#!/usr/bin/env python3
"""
Data quality gate for the NSE quant cache.

WHAT THIS IS FOR
================
Every check in Daily_cache is advisory. Advisory checks get ignored at 2am,
and then a backtest runs on a symbol with forty missing sessions and produces
a Sharpe you believe. This module is the thing that says no.

Nothing downstream - features, panel, labels, backtest - should read the
cache directly. It should call assert_panel_ready(), which raises unless
every symbol it is about to use has passed all four gates:

    COMPLETE    every session the exchange had, this series has
    VALID       the numbers are numbers, and OHLC obeys its own arithmetic
    CONSISTENT  one row per session, sorted, one adjustment basis, declared
                schema for its asset kind
    VERIFIED    the .ok metadata exists, matches the file, and is current

A series that fails any gate is QUARANTINED: recorded in quarantine.json with
its reasons, and excluded from the allowed set. Quarantine is not deletion -
the data stays on disk for inspection. It is simply not usable.

WHY A SEPARATE MODULE
---------------------
The gate must be able to fail the cache builder's own output. A checker that
lives inside the thing it checks, and shares its assumptions, is not a check.

USAGE
-----
    from data_quality import assert_panel_ready, gate_all

    report = gate_all(root, symbols)          # inspect
    allowed = assert_panel_ready(root, symbols)  # or raise

    python data_quality.py --root <cache_root>   # CLI audit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA_OK_KEY = "schema_version"
OHLCV = ("open", "high", "low", "close", "volume")

# Per-kind schema contract. v22's schema was NOT identical across asset types
# and did not say so; this declares the differences instead of pretending.
KIND_SCHEMA: Dict[str, dict] = {
    "equity": {"required": ("timestamp", *OHLCV), "volume_required": True,
               "allow_roll_col": False},
    "index": {"required": ("timestamp", "open", "high", "low", "close"),
              "volume_required": False, "allow_roll_col": False},
    "futures": {"required": ("timestamp", *OHLCV), "volume_required": True,
                "allow_roll_col": True},
}

# A symbol may legitimately miss the odd session (a trading halt). Beyond this
# it is a data problem, not a market event.
MAX_MISSING_SESSIONS = 3
MAX_STALE_SESSIONS = 1

# v27: "completeness could not be verified" now FAILS.
#
# v26 let a futures series pass with a warning when it had no exchange peer
# to audit against - freshness checked, completeness assumed. For research
# that is the wrong default: an unverifiable claim is not a weaker pass, it
# is no evidence. USDINR is the live case (sole CDS series), and the fix is
# to cache a second CDS contract, not to lower the bar.
#
# Set allow_unverified=True on gate_all/assert_panel_ready to accept them
# deliberately, e.g. for exploratory work where the macro series is context
# rather than signal.
ALLOW_UNVERIFIED_DEFAULT = False


@dataclass
class GateResult:
    symbol: str
    kind: str = "equity"
    complete: bool = False
    valid: bool = False
    consistent: bool = False
    verified: bool = False
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.complete and self.valid and self.consistent and self.verified

    @property
    def status(self) -> str:
        if not self.allowed:
            return "QUARANTINE"
        return "PASS_UNVERIFIED" if self.warnings else "PASS"

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "kind": self.kind,
            "status": self.status,
            "COMPLETE": self.complete,
            "VALID": self.valid,
            "CONSISTENT": self.consistent,
            "VERIFIED": self.verified,
            "reasons": self.reasons,
            "warnings": self.warnings,
            **self.detail,
        }


class DataQualityError(RuntimeError):
    """Raised by assert_panel_ready when any requested series is quarantined."""


# The cache writes "<SYMBOL>_daily.parquet" / "<SYMBOL>_daily.ok.json".
# Keep this in one place - a mismatch here means the gate silently finds
# nothing and reports a clean run over an empty set.
CACHE_SUFFIX = "_daily"


def _paths(root: Path, symbol: str):
    return (root / f"{symbol}{CACHE_SUFFIX}.parquet",
            root / f"{symbol}{CACHE_SUFFIX}.ok.json")


def discover_symbols(root) -> List[str]:
    """Every cached series under `root`, by the name the cache keys them on."""
    root = Path(root)
    n = len(CACHE_SUFFIX)
    return sorted(
        p.stem[:-n] if p.stem.endswith(CACHE_SUFFIX) else p.stem
        for p in root.glob(f"*{CACHE_SUFFIX}.parquet")
    )


def _read_ok(ok_path: Path) -> Optional[dict]:
    if not ok_path.exists():
        return None
    try:
        return json.loads(ok_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def master_sessions(root: Path, reference: str = "NIFTY50") -> pd.DatetimeIndex:
    """
    The exchange's own session list, taken from a cached index.

    If this is missing there is no way to tell a complete series from an
    incomplete one, so COMPLETE cannot pass. That is deliberate: an
    unverifiable gate must fail, not default to true.
    """
    pq, _ = _paths(root, reference)
    if not pq.exists():
        raise FileNotFoundError(
            f"master calendar {reference} not cached under {root}. "
            "COMPLETE cannot be evaluated without it."
        )
    ts = pd.to_datetime(pd.read_parquet(pq, columns=["timestamp"])["timestamp"])
    return pd.DatetimeIndex(ts).tz_localize(None).normalize().sort_values().unique()


def _check_complete(
    df: pd.DataFrame,
    master: Optional[pd.DatetimeIndex],
    kind: str,
    r: GateResult,
    absence: Optional[dict] = None,
    trusted: bool = True,
    peer_sessions: Optional[pd.DatetimeIndex] = None,
    allow_unverified: bool = ALLOW_UNVERIFIED_DEFAULT,
) -> bool:
    if df.empty:
        r.reasons.append("no rows")
        return False
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"])).tz_localize(None).normalize()
    r.detail["first"] = str(ts.min().date())
    r.detail["last"] = str(ts.max().date())
    r.detail["rows"] = int(len(df))

    if master is None or len(master) == 0:
        r.reasons.append("no master calendar; completeness unverifiable")
        return False

    # v25: an UNTRUSTED calendar is single-sourced, so it can share the very
    # gap it is meant to detect - the circularity that made a NIFTY50-only
    # reference unsafe. v24 detected this and then passed `master` down
    # anyway, so every symbol inherited a completeness pass it had not
    # earned. Unverifiable must fail, not default to true.
    if not trusted:
        r.reasons.append(
            "calendar is single-sourced (untrusted); completeness cannot be "
            "verified against an independent reference"
        )
        return False

    # ------------------------------------------------------------------
    # Futures trade MCX/CDS calendars, which NSE sessions cannot measure.
    # v24 checked NSE freshness and then returned True, which reported
    # "COMPLETE" on the strength of a test that never looked for a gap.
    # v25 separates the two claims:
    #
    #     freshness    - is the series current?           (checkable here)
    #     completeness - are any sessions missing?        (needs peers)
    #
    # Peers are the other cached series on the SAME exchange. With two or
    # more, a date one has and another lacks is a real gap. With fewer,
    # completeness is genuinely unverifiable and is reported as such rather
    # than assumed.
    # ------------------------------------------------------------------
    if kind == "futures":
        stale = int((master > ts.max()).sum())
        r.detail["stale_sessions"] = stale
        if stale > MAX_STALE_SESSIONS:
            r.reasons.append(f"{stale} NSE sessions behind")
            return False

        if peer_sessions is None or len(peer_sessions) == 0:
            r.detail["completeness_verified"] = False
            r.detail["missing_sessions"] = None
            msg = ("completeness NOT verifiable: no peer series on this "
                   "exchange to derive a session calendar from (cache a "
                   "second contract on it, or pass allow_unverified=True)")
            if allow_unverified:
                r.warnings.append("freshness checked only; " + msg)
                return True
            r.reasons.append(msg)
            return False

        in_span = peer_sessions[(peer_sessions >= ts.min()) & (peer_sessions <= ts.max())]
        miss = in_span.difference(ts)
        r.detail["completeness_verified"] = True
        r.detail["missing_sessions"] = int(len(miss))
        r.detail["peer_sessions"] = int(len(peer_sessions))
        if len(miss) > MAX_MISSING_SESSIONS:
            r.detail["missing_sample"] = [str(d.date()) for d in miss[:8]]
            r.reasons.append(
                f"{len(miss)} sessions missing that peer series on this "
                f"exchange have"
            )
            return False
        return True

    in_span = master[(master >= ts.min()) & (master <= ts.max())]
    missing = in_span.difference(ts)
    r.detail["missing_sessions"] = int(len(missing))

    stale = int((master > ts.max()).sum())
    r.detail["stale_sessions"] = stale

    ok = True
    if stale > MAX_STALE_SESSIONS:
        r.reasons.append(f"{stale} sessions behind the exchange")
        ok = False

    if absence is None:
        # No cross-section available, so a legitimate no-trade day cannot be
        # told apart from a dropped fetch. Fall back to the blunt count.
        if len(missing) > MAX_MISSING_SESSIONS:
            r.reasons.append(
                f"{len(missing)} interior sessions missing (unclassified - "
                f"run gate_all over the universe to separate suspensions "
                f"from data loss)"
            )
            ok = False
        return ok

    r.detail["absence"] = {k: absence.get(k, 0) for k in ABSENCE_CLASSES}
    if absence.get("suspension"):
        r.detail["suspension_runs"] = absence.get("suspension_runs", [])
    # market_wide is never the symbol's fault; suspensions are real events
    # where the data is correct. Only unexplained isolated gaps are defects.
    defects = int(absence.get("isolated", 0))
    if defects > MAX_MISSING_SESSIONS:
        r.reasons.append(
            f"{defects} isolated sessions missing with no market-wide or "
            f"suspension explanation"
        )
        ok = False
        r.detail["isolated_dates"] = absence.get("isolated_dates", [])
    return ok


def _check_valid(df: pd.DataFrame, kind: str, r: GateResult) -> bool:
    if df.empty:
        return False
    spec = KIND_SCHEMA.get(kind, KIND_SCHEMA["equity"])
    ok = True

    o, h, l, c = (pd.to_numeric(df.get(x), errors="coerce") for x in
                  ("open", "high", "low", "close"))
    nan_px = int(pd.concat([o, h, l, c], axis=1).isna().any(axis=1).sum())
    r.detail["rows_with_nan_price"] = nan_px
    if nan_px:
        r.reasons.append(f"{nan_px} rows with NaN OHLC")
        ok = False

    nonpos = int((c <= 0).fillna(False).sum())
    if nonpos:
        r.reasons.append(f"{nonpos} rows with close <= 0")
        ok = False

    viol = (
        (h < l)
        | (c > h + 1e-9) | (c < l - 1e-9)
        | (o > h + 1e-9) | (o < l - 1e-9)
    ).fillna(False)
    n_viol = int(viol.sum())
    r.detail["ohlc_violations"] = n_viol
    if n_viol:
        r.reasons.append(f"{n_viol} rows violate OHLC arithmetic")
        ok = False

    if spec["volume_required"]:
        v = pd.to_numeric(df.get("volume"), errors="coerce")
        if v is None or v.isna().all():
            r.reasons.append("volume required for this kind but absent")
            ok = False
        else:
            neg = int((v < 0).fillna(False).sum())
            if neg:
                r.reasons.append(f"{neg} rows with negative volume")
                ok = False
    return ok


def _check_consistent(df: pd.DataFrame, meta: dict, kind: str, r: GateResult) -> bool:
    if df.empty:
        return False
    spec = KIND_SCHEMA.get(kind, KIND_SCHEMA["equity"])
    ok = True

    missing_cols = [c for c in spec["required"] if c not in df.columns]
    if missing_cols:
        r.reasons.append(f"missing required columns: {missing_cols}")
        ok = False

    if "D_roll" in df.columns and not spec["allow_roll_col"]:
        r.reasons.append("D_roll present on a non-futures series")
        ok = False

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        r.reasons.append("unparseable timestamps")
        ok = False
    else:
        if not ts.is_monotonic_increasing:
            r.reasons.append("timestamps not sorted")
            ok = False
        dupes = int(ts.duplicated().sum())
        if dupes:
            r.reasons.append(f"{dupes} duplicate sessions")
            ok = False

    suspects = meta.get("corporate_action_suspects") or []
    r.detail["corporate_action_suspects"] = len(suspects)
    if suspects:
        r.reasons.append(
            f"{len(suspects)} unresolved split-shaped discontinuity(ies) - "
            "adjustment basis not proven"
        )
        ok = False

    # Futures history older than the observed roll calendar carries D_roll=NA.
    # That is honest, but it is not usable without a decision, so the gate
    # surfaces it rather than letting it through silently.
    if kind == "futures" and "D_roll" in df.columns:
        unknown = int(pd.isna(df["D_roll"]).sum())
        r.detail["rows_with_unknown_roll"] = unknown
        if unknown:
            r.reasons.append(
                f"{unknown} rows predate the roll calendar (D_roll unknown); "
                "trim them or accept differences may span a roll"
            )
            ok = False
    return ok


def _check_verified(
    pq: Path, ok_path: Path, meta: Optional[dict], df: pd.DataFrame,
    r: GateResult, schema_version=None,
) -> bool:
    ok = True
    if meta is None:
        r.reasons.append("no .ok metadata - build never completed")
        return False
    rows_meta = meta.get("rows")
    if rows_meta is not None and int(rows_meta) != int(len(df)):
        r.reasons.append(
            f"metadata claims {rows_meta} rows, file has {len(df)} - "
            "parquet written outside the builder?"
        )
        ok = False
    if schema_version is not None and meta.get(SCHEMA_OK_KEY) != schema_version:
        r.reasons.append(
            f"schema {meta.get(SCHEMA_OK_KEY)} != expected {schema_version}"
        )
        ok = False
    try:
        if ok_path.stat().st_mtime < pq.stat().st_mtime - 1:
            r.reasons.append("parquet modified after its metadata")
            ok = False
    except OSError:
        pass
    return ok


def exchange_peer_sessions(root, symbols: Sequence[str]) -> Dict[str, pd.DatetimeIndex]:
    """
    A session calendar per non-NSE exchange, from the series cached on it.

    Union of peer sessions: if crude traded on a date, that date was an MCX
    session, so gold lacking it is a real gap. Two peers is the minimum for
    this to say anything; one series cannot audit itself.
    """
    root = Path(root)
    by_exch: Dict[str, List[pd.DatetimeIndex]] = {}
    for sym in symbols:
        pq, okp = _paths(root, sym)
        if not pq.exists():
            continue
        meta = _read_ok(okp) or {}
        if meta.get("series_kind") != "futures":
            continue
        exch = meta.get("exchange") or "UNKNOWN"
        try:
            ts = _norm(pd.read_parquet(pq, columns=["timestamp"])["timestamp"])
        except Exception:
            continue
        by_exch.setdefault(exch, []).append(ts)
    out: Dict[str, pd.DatetimeIndex] = {}
    for exch, idxs in by_exch.items():
        if len(idxs) < 2:
            continue                      # one series cannot audit itself
        u = idxs[0]
        for ix in idxs[1:]:
            u = u.union(ix)
        out[exch] = pd.DatetimeIndex(u).sort_values()
    return out


def gate_symbol(
    root: Path,
    symbol: str,
    *,
    master: Optional[pd.DatetimeIndex] = None,
    schema_version=None,
    absence: Optional[dict] = None,
    trusted: bool = True,
    peers: Optional[Dict[str, pd.DatetimeIndex]] = None,
    allow_unverified: bool = ALLOW_UNVERIFIED_DEFAULT,
) -> GateResult:
    pq, okp = _paths(root, symbol)
    meta = _read_ok(okp)
    kind = (meta or {}).get("series_kind", "equity")
    r = GateResult(symbol=symbol, kind=kind)

    if not pq.exists():
        r.reasons.append("not cached")
        return r
    try:
        df = pd.read_parquet(pq)
    except Exception as e:
        r.reasons.append(f"unreadable parquet: {e}")
        return r

    peer_ix = None
    if kind == "futures" and peers:
        peer_ix = peers.get((meta or {}).get("exchange") or "UNKNOWN")
    r.complete = _check_complete(
        df, master, kind, r, absence=absence, trusted=trusted,
        peer_sessions=peer_ix, allow_unverified=allow_unverified,
    )
    r.valid = _check_valid(df, kind, r)
    r.consistent = _check_consistent(df, meta or {}, kind, r)
    r.verified = _check_verified(pq, okp, meta, df, r, schema_version=schema_version)
    return r


def gate_all(
    root,
    symbols: Sequence[str],
    *,
    reference: str = "NIFTY50",
    schema_version=None,
    holidays_csv=None,
    classify: bool = True,
    allow_unverified: bool = ALLOW_UNVERIFIED_DEFAULT,
) -> dict:
    """
    v24: builds a cross-checked calendar and classifies absences before
    gating, so a suspended stock and a stock with dropped bars are no longer
    treated as the same failure.
    """
    root = Path(root)
    master, master_err, cal, absence = None, None, None, {}
    try:
        cal = build_calendar(
            root, symbols, reference=reference, holidays_csv=holidays_csv
        )
        if len(cal):
            master = cal.sessions
        else:
            master_err = "calendar empty"
        if classify and master is not None:
            absence = classify_absences(root, symbols, calendar=cal).get("symbols", {})
    except Exception as e:
        master_err = str(e)

    if cal is not None and not cal.trusted:
        # A single-sourced calendar cannot prove completeness. Say so rather
        # than letting every symbol inherit a false pass.
        master_err = (master_err or "") + " | calendar untrusted (single source)"

    trusted = bool(cal.trusted) if cal is not None else False
    peers = exchange_peer_sessions(root, symbols)
    results = [
        gate_symbol(
            root, s, master=master, schema_version=schema_version,
            absence=absence.get(s), trusted=trusted, peers=peers,
            allow_unverified=allow_unverified,
        )
        for s in symbols
    ]
    allowed = [r.symbol for r in results if r.allowed]
    unverified = [r for r in results if r.allowed and r.warnings]
    quarantined = [r for r in results if not r.allowed]

    report = {
        "generated_at": dt.datetime.now().isoformat(),
        "root": str(root),
        "master_calendar": reference,
        "master_error": master_err,
        "master_sessions": int(len(master)) if master is not None else 0,
        "calendar": cal.as_dict() if cal is not None else None,
        "requested": len(symbols),
        "allowed": len(allowed),
        "quarantined": len(quarantined),
        "unverified": len(unverified),
        "allowed_symbols": allowed,
        "unverified_detail": [r.as_dict() for r in unverified],
        "quarantine": [r.as_dict() for r in quarantined],
    }
    try:
        (root / "quarantine.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        pass
    return report


def assert_panel_ready(
    root, symbols: Sequence[str], *, reference: str = "NIFTY50",
    schema_version=None, allow_partial: bool = False,
    allow_unverified: bool = ALLOW_UNVERIFIED_DEFAULT,
) -> List[str]:
    """
    Return the symbols cleared for use, or raise.

    allow_partial=True proceeds with whatever passed, after printing what was
    dropped. Use it for research runs where a thinner universe is acceptable.
    Leave it False for anything whose numbers you intend to believe.
    """
    rep = gate_all(root, symbols, reference=reference,
                   schema_version=schema_version,
                   allow_unverified=allow_unverified)
    if rep["quarantined"]:
        lines = [
            f"DATA QUALITY: {rep['quarantined']}/{rep['requested']} series "
            f"quarantined."
        ]
        for q in rep["quarantine"][:25]:
            lines.append(f"  {q['symbol']:<18} {'; '.join(q['reasons'])[:100]}")
        if rep["quarantined"] > 25:
            lines.append(f"  ... +{rep['quarantined'] - 25} more")
        lines.append(f"  Full report: {Path(rep['root']) / 'quarantine.json'}")
        msg = "\n".join(lines)
        if not allow_partial:
            raise DataQualityError(msg)
        print(msg, file=sys.stderr)
    return rep["allowed_symbols"]


# ======================================================================
# TRADING CALENDAR (v24)
# ======================================================================
#
# v23 took the session list from cached NIFTY50 alone. That is circular: if
# NIFTY50 itself is missing a date, the date vanishes from the "master"
# calendar and every symbol that also lacks it reports as complete. A gap
# check whose reference can share the gap is not a check.
#
# v24 builds the calendar from up to three INDEPENDENT sources and
# cross-checks them:
#
#   authoritative  pandas_market_calendars / exchange_calendars XNSE, or a
#                  holidays CSV you supply. Knows the published schedule.
#   consensus      derived from the cached universe: a date is a session if
#                  most symbols that were live on that date have a bar. Knows
#                  what actually traded.
#   reference      cached NIFTY50. Fast, and the v23 behaviour.
#
# Neither of the first two dominates. Published calendars miss special
# sessions - NSE's Diwali Muhurat session is a real trading day that holiday
# lists routinely omit - while consensus can be fooled by a vendor outage
# that hit the whole universe at once. So the calendar is built from
# consensus, corrected by authoritative where they agree, and every
# disagreement is reported rather than silently resolved.
#
# A calendar backed by only one source is marked untrusted, and the gate
# treats completeness as unverifiable rather than assuming it passed.

DEFAULT_CONSENSUS_FRAC = 0.60   # share of live symbols that must have a bar
MARKET_WIDE_FRAC = 0.50         # absence share that makes it a calendar issue
SUSPENSION_MIN_RUN = 5          # contiguous absent sessions => suspension


@dataclass
class Calendar:
    sessions: pd.DatetimeIndex
    source: str
    sources: Dict[str, int] = field(default_factory=dict)
    disagreements: List[dict] = field(default_factory=list)
    trusted: bool = False
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sessions)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "trusted": self.trusted,
            "sessions": len(self.sessions),
            "first": str(self.sessions.min().date()) if len(self.sessions) else None,
            "last": str(self.sessions.max().date()) if len(self.sessions) else None,
            "sources": self.sources,
            "disagreements": self.disagreements[:50],
            "disagreement_count": len(self.disagreements),
            "notes": self.notes,
        }


def _norm(ix) -> pd.DatetimeIndex:
    ix = pd.DatetimeIndex(pd.to_datetime(list(ix)))
    try:
        ix = ix.tz_localize(None)
    except TypeError:
        ix = ix.tz_convert(None)
    return pd.DatetimeIndex(ix.normalize().unique()).sort_values()


def authoritative_sessions(
    lo: pd.Timestamp, hi: pd.Timestamp, *, holidays_csv=None, name: str = "XNSE"
) -> Optional[pd.DatetimeIndex]:
    """
    The published exchange schedule, if we can get one.

    Tries pandas_market_calendars, then exchange_calendars, then a CSV of
    holiday dates (one ISO date per line) which is generated off business
    days minus holidays. Returns None if none is available - the caller must
    then treat the calendar as single-sourced.
    """
    if holidays_csv:
        try:
            hol = _norm(pd.read_csv(holidays_csv, header=None)[0])
            bdays = pd.DatetimeIndex(pd.bdate_range(lo, hi))
            return bdays.difference(hol)
        except Exception:
            return None
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar(name)
        return _norm(cal.valid_days(start_date=lo, end_date=hi))
    except Exception:
        pass
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar(name)
        return _norm(cal.sessions_in_range(lo, hi))
    except Exception:
        return None


MIN_CONSENSUS_SYMBOLS = 10   # below this, "consensus" is circular


def consensus_sessions(
    spans: Dict[str, pd.DatetimeIndex],
    *,
    frac: float = DEFAULT_CONSENSUS_FRAC,
    min_symbols: int = MIN_CONSENSUS_SYMBOLS,
) -> Optional[pd.DatetimeIndex]:
    """
    Dates on which most of the LIVE universe traded.

    "Live" matters: a stock listed in 2023 must not drag down the vote for
    2019. Each symbol only votes on dates inside its own first..last span.
    """
    # A vote needs voters. With a handful of symbols, consensus just echoes
    # whatever those symbols happen to contain - exactly the circularity that
    # made a NIFTY50-only calendar unsafe.
    if not spans or len(spans) < min_symbols:
        return None
    all_dates = sorted({d for ix in spans.values() for d in ix})
    if not all_dates:
        return None
    arr = pd.DatetimeIndex(all_dates)
    present = np.zeros(len(arr), dtype=np.int32)
    live = np.zeros(len(arr), dtype=np.int32)
    pos = {d: i for i, d in enumerate(arr)}
    for ix in spans.values():
        if len(ix) == 0:
            continue
        lo, hi = ix.min(), ix.max()
        mask = (arr >= lo) & (arr <= hi)
        live += mask.astype(np.int32)
        for d in ix:
            present[pos[d]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(live > 0, present / np.maximum(live, 1), 0.0)
    return pd.DatetimeIndex(arr[share >= frac]).sort_values()


def _symbol_spans(root: Path, symbols: Sequence[str]) -> Dict[str, pd.DatetimeIndex]:
    out: Dict[str, pd.DatetimeIndex] = {}
    for s in symbols:
        pq, okp = _paths(root, s)
        if not pq.exists():
            continue
        meta = _read_ok(okp) or {}
        if meta.get("series_kind", "equity") not in ("equity", "index"):
            continue          # MCX/CDS trade a different calendar
        try:
            ts = pd.read_parquet(pq, columns=["timestamp"])["timestamp"]
            out[s] = _norm(ts)
        except Exception:
            continue
    return out


def build_calendar(
    root,
    symbols: Optional[Sequence[str]] = None,
    *,
    reference: str = "NIFTY50",
    holidays_csv=None,
    consensus_frac: float = DEFAULT_CONSENSUS_FRAC,
    calendar_name: str = "XNSE",
) -> Calendar:
    root = Path(root)
    if symbols is None:
        symbols = discover_symbols(root)

    spans = _symbol_spans(root, symbols)
    cons = consensus_sessions(spans, frac=consensus_frac)

    ref = None
    pq, _ = _paths(root, reference)
    if pq.exists():
        try:
            ref = _norm(pd.read_parquet(pq, columns=["timestamp"])["timestamp"])
        except Exception:
            ref = None

    pool = [ix for ix in (cons, ref) if ix is not None and len(ix)]
    if not pool:
        return Calendar(pd.DatetimeIndex([]), "none", {},
                        notes=["no cached data to build a calendar from"])
    lo = min(ix.min() for ix in pool)
    hi = max(ix.max() for ix in pool)
    auth = authoritative_sessions(
        lo, hi, holidays_csv=holidays_csv, name=calendar_name
    )

    avail = {
        k: v for k, v in
        (("authoritative", auth), ("consensus", cons), ("reference", ref))
        if v is not None and len(v)
    }
    counts = {k: int(len(v)) for k, v in avail.items()}
    notes: List[str] = []

    # Consensus is the operative source: it records what actually traded.
    if "consensus" in avail:
        sessions, source = avail["consensus"], "consensus"
    elif "authoritative" in avail:
        sessions, source = avail["authoritative"], "authoritative"
    else:
        sessions, source = avail["reference"], "reference"
        notes.append(
            "single-sourced from NIFTY50 - a gap in NIFTY50 will hide gaps "
            "everywhere else"
        )

    disagreements: List[dict] = []
    for a, b in (("authoritative", "consensus"),
                 ("authoritative", "reference"),
                 ("consensus", "reference")):
        if a not in avail or b not in avail:
            continue
        ia, ib = avail[a], avail[b]
        lo2, hi2 = max(ia.min(), ib.min()), min(ia.max(), ib.max())
        ia2 = ia[(ia >= lo2) & (ia <= hi2)]
        ib2 = ib[(ib >= lo2) & (ib <= hi2)]
        for d in ia2.difference(ib2):
            disagreements.append({"date": str(d.date()), "in": a, "not_in": b})
        for d in ib2.difference(ia2):
            disagreements.append({"date": str(d.date()), "in": b, "not_in": a})

    if "authoritative" in avail and "consensus" in avail:
        extra = sessions.difference(avail["authoritative"])
        if len(extra):
            notes.append(
                f"{len(extra)} session(s) traded that the published calendar "
                f"omits (e.g. Muhurat); keeping them: "
                + ", ".join(str(d.date()) for d in extra[:5])
            )
        missed = avail["authoritative"].difference(sessions)
        missed = missed[(missed >= sessions.min()) & (missed <= sessions.max())]
        if len(missed):
            notes.append(
                f"{len(missed)} published session(s) absent from the whole "
                f"universe - likely a fetch outage, NOT a holiday: "
                + ", ".join(str(d.date()) for d in missed[:5])
            )

    n_voters = len(spans)
    counts["consensus_symbols"] = int(n_voters)
    trusted = len(avail) >= 2 and "consensus" in avail
    if "consensus" not in avail:
        notes.append(
            f"no consensus source: only {n_voters} NSE-calendar symbol(s) "
            f"cached, need >= {MIN_CONSENSUS_SYMBOLS}. Completeness cannot be "
            f"verified against the universe."
        )
    if not trusted:
        notes.append(
            "untrusted calendar: needs consensus plus at least one "
            "independent source"
        )

    return Calendar(
        sessions=sessions, source=source, sources=counts,
        disagreements=disagreements, trusted=trusted, notes=notes,
    )


# ======================================================================
# ABSENCE CLASSIFICATION (v24)
# ======================================================================
#
# v23 treated every absent session as a defect. Most are not. A stock can be
# legitimately absent because it was suspended, halted, or had no trades at
# all - and separately, a date can be absent across the whole universe, which
# is a calendar or fetch problem rather than a per-symbol one.
#
# Classification needs the cross-section, so it runs over the universe once:
#
#   market_wide   absent for most live symbols -> not that symbol's fault.
#                 Either a real holiday the calendar got wrong, or a fetch
#                 outage that hit everything. Never counted against a symbol.
#   suspension    a contiguous run of >= SUSPENSION_MIN_RUN sessions ->
#                 almost certainly a real trading suspension. Reported, and
#                 tolerated, because the data is not wrong: the stock did
#                 not trade.
#   isolated      one to four scattered sessions -> a halt, a no-trade day,
#                 or missing data, and OHLCV alone CANNOT tell these apart.
#                 Treated as a defect because the alternative is assuming the
#                 benign case, which is how bad data gets in.
#
# The honest boundary: "isolated" conflates a genuine no-trade day with a
# dropped fetch. Separating them needs trade-count or suspension data the
# cache does not hold.

ABSENCE_CLASSES = ("market_wide", "suspension", "isolated")


def _runs(dates: Sequence[pd.Timestamp], sessions: pd.DatetimeIndex) -> List[List]:
    """Group absent dates into runs contiguous in SESSION space, not days."""
    if not len(dates):
        return []
    pos = {d: i for i, d in enumerate(sessions)}
    idx = sorted(pos[d] for d in dates if d in pos)
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    return [[sessions[i] for i in run] for run in runs]


def classify_absences(
    root,
    symbols: Optional[Sequence[str]] = None,
    *,
    calendar: Optional[Calendar] = None,
    market_wide_frac: float = MARKET_WIDE_FRAC,
    suspension_min_run: int = SUSPENSION_MIN_RUN,
) -> dict:
    root = Path(root)
    if symbols is None:
        symbols = discover_symbols(root)
    cal = calendar or build_calendar(root, symbols)
    sessions = cal.sessions
    if not len(sessions):
        return {"calendar": cal.as_dict(), "symbols": {}, "market_wide_dates": []}

    spans = _symbol_spans(root, symbols)

    absent_count: Dict[pd.Timestamp, int] = {}
    live_count: Dict[pd.Timestamp, int] = {}
    per_symbol_missing: Dict[str, List[pd.Timestamp]] = {}
    for sym, ix in spans.items():
        if not len(ix):
            continue
        in_span = sessions[(sessions >= ix.min()) & (sessions <= ix.max())]
        for d in in_span:
            live_count[d] = live_count.get(d, 0) + 1
        miss = in_span.difference(ix)
        per_symbol_missing[sym] = list(miss)
        for d in miss:
            absent_count[d] = absent_count.get(d, 0) + 1

    market_wide = {
        d for d, n in absent_count.items()
        if live_count.get(d, 0) > 0 and n / live_count[d] >= market_wide_frac
    }

    out: Dict[str, dict] = {}
    for sym, miss in per_symbol_missing.items():
        mw = [d for d in miss if d in market_wide]
        own = [d for d in miss if d not in market_wide]
        susp, iso = [], []
        for run in _runs(own, sessions):
            (susp if len(run) >= suspension_min_run else iso).extend(run)
        out[sym] = {
            "market_wide": len(mw),
            "suspension": len(susp),
            "isolated": len(iso),
            "suspension_runs": [
                f"{r[0].date()}..{r[-1].date()}"
                for r in _runs(susp, sessions)
            ][:5],
            "isolated_dates": [str(d.date()) for d in iso[:10]],
            "defects": len(iso),
        }

    return {
        "calendar": cal.as_dict(),
        "market_wide_dates": sorted(str(d.date()) for d in market_wide),
        "symbols": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit the cache before use.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--reference", default="NIFTY50")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="accept futures whose completeness cannot be checked")
    ap.add_argument("--holidays-csv", default=None,
                    help="optional CSV of holiday dates, one ISO date per line")
    args = ap.parse_args()
    root = Path(args.root)
    syms = args.symbols or discover_symbols(root)
    rep = gate_all(root, syms, reference=args.reference,
                   holidays_csv=args.holidays_csv,
                   allow_unverified=args.allow_unverified)
    print(f"PASS {rep['allowed']} / QUARANTINE {rep['quarantined']} "
          f"of {rep['requested']}")
    c = rep.get("calendar") or {}
    if c:
        print(f"  calendar: {c['source']} | {c['sessions']} sessions | "
              f"trusted={c['trusted']} | sources={c['sources']}")
        for n in c.get("notes", []):
            print(f"    NOTE: {n}")
        if c.get("disagreement_count"):
            print(f"    {c['disagreement_count']} source disagreement(s), e.g. "
                  f"{c['disagreements'][:3]}")
    if rep["master_error"]:
        print(f"  calendar warning: {rep['master_error']}")
    for q in rep["quarantine"][:40]:
        print(f"  {q['symbol']:<18} {'; '.join(q['reasons'])[:110]}")
    return 1 if rep["quarantined"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
