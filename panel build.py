#!/usr/bin/env python3
"""
panel_build.py - assemble the model-ready panel.

    python panel_build.py --root <cache_root> --panel <panel_dir>
    python panel_build.py --root <cache_root> --panel <panel_dir> --full

ORDER, AND WHY IT IS THIS ORDER
===============================
    cache -> GATE -> per-symbol features -> point-in-time cross-section
          -> cross-sectional features -> exogenous join -> labels -> panel

The gate comes BEFORE the cross-section, not after. A quarantined symbol left
in a cross-sectional rank contaminates every other symbol's rank on that date,
not just its own row. Filtering afterwards does not undo that.

THE THREE THINGS THAT MAKE THIS MORE THAN AN APPEND
---------------------------------------------------
1. Per-symbol features are recomputed over FULL history every run. Expanding
   ranks depend on the whole series, and the cache's tail refetch revises
   recent bars, so recent feature values genuinely change. We recompute
   everything and rewrite only the recent window.

2. Labels arrive late. A 5-day-forward label for today is not knowable today.
   Tonight, rows from 5 sessions ago become resolvable. That is a backfill
   pass, not leakage - but it means the recent edge of the panel is always
   unlabelled, and training must treat those rows as UNKNOWN rather than as
   negatives.

3. The universe is point-in-time. Cross-sectional ranks for 2019 rank against
   the symbols that actually traded in 2019. Using today's symbol list is
   survivorship bias, and it is the easiest way to build a backtest that
   cannot be traded.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Sector-relative features. They need dated index membership, which the cache
does not have. A sector map scraped today and applied to 2019 is both
lookahead and survivorship, in the feature family most likely to carry real
cross-sectional alpha. Pass --sector-map with a dated file to enable them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features_daily as fx
from data_quality import (
    assert_panel_ready, discover_symbols, _paths, _read_ok, build_calendar,
)

# Columns ranked cross-sectionally each date. These are the ones where
# "big relative to its peers today" is the actual hypothesis; ranking
# everything would just add 160 collinear columns.
CROSS_SECTIONAL_BASE = [
    "D_ret_1d_pct", "D_range_pct", "D_atr_pct", "D_rsi14", "D_adx14",
    "D_dist_from_52wh", "D_dist_from_52wl", "D_pos_in_52w_range",
    "D_drawdown_252", "D_dvol_z20", "D_dvol_z252", "D_realvol_20",
    "D_realvol_ratio_20_60", "D_amihud_20", "D_gap_pct",
    "D_intraday_ret_pct", "D_ret_skew_60", "D_downside_dev_60",
    "D_bb_pctB_20", "D_donch_pos_20", "D_ema20_angle_deg",
]

# ----------------------------------------------------------------------
# POINT-IN-TIME LIQUIDITY FLOOR
#
# The tempting fix is to delete illiquid names from watchlist.txt. That is
# survivorship bias: the decision uses 2026 information to decide whether a
# stock belongs in the 2019 cross-section. Stocks that were liquid in 2019
# and later died are exactly the ones a model picks and loses on, and
# screening them out in hindsight makes those losses invisible.
#
# So the filter is applied PER DATE, from trailing data only. A stock is in
# the universe on date t if its median turnover over the prior LIQ_WINDOW
# sessions, measured as of t, cleared the floor. Liquid in 2019 and dead by
# 2022 means in the 2019 cross-section and out of the 2022 one - which is
# what actually happened, and what was actually tradeable.
#
# The floor is in rupees of daily turnover. It also decides what you can
# realistically enter and exit at your position size, so set it from your own
# sizing rather than from a default. 0 disables the filter entirely.
# ----------------------------------------------------------------------
LIQ_WINDOW = 60                 # trailing sessions for the turnover median
LIQ_MIN_TURNOVER = 0.0          # rupees/day; 0 = off. Try 1e7 (1 crore).
LIQ_MIN_OBSERVATIONS = 40       # need this many bars in the window to judge

MARKET_SERIES = "NIFTY50"
LABEL_HORIZON = 5
LABEL_TOUCH_PCT = 0.05

# ----------------------------------------------------------------------
# LABEL VARIANTS
#
# The panel carries SEVERAL brackets, not one. Computing them costs one extra
# pass over paths you already have, and it removes the pressure to pick the
# right bracket before you know anything - which is exactly the pressure that
# produces a choice you later want to revisit.
#
# atr=True scales the barrier to each stock's own ATR, so "1.5 ATR" means the
# same event for a large cap and a smallcap. A fixed 5% is ~1.2 median ATRs
# here, which makes the fixed label partly a volatility detector.
#
# THE DISCIPLINE THAT MAKES THIS SAFE: choose ONE variant as the evaluation
# target and freeze it BEFORE looking at model results. The rest are
# diagnostics. Picking the best-performing label after the fact is the same
# overfitting you avoided by not tuning barriers on model output.
#
# Note also that the LABEL need not equal the TRADE. A label should be dense,
# measurable and unambiguous so the model can learn; a live bracket should
# reflect slippage, GTT behaviour and capital lockup. They are allowed to
# differ, and usually should.
# ----------------------------------------------------------------------
LABEL_VARIANTS = [
    # (suffix,      tp,    sl,   atr-scaled)
    ("5p3",        0.05,  0.03,  False),   # legacy default
    ("3p2",        0.03,  0.02,  False),
    ("atr1p5_1p0", 1.50,  1.00,  True),    # best-measured on the daily scan
    ("atr2p0_1p0", 2.00,  1.00,  True),
    ("atr1p0_1p0", 1.00,  1.00,  True),
]

# The variant the model is evaluated on. FREEZE THIS before training.
PRIMARY_VARIANT = "atr1p5_1p0"

# Back-compat: the unsuffixed label columns mirror the primary variant.
LABEL_TP = 0.05
LABEL_SL = 0.03

_BASE_LABELS = (
    "label_touch", "label_fwd_ret_5d", "label_mfe_5d", "label_mae_5d",
    "label_first_touch", "label_tp_before_sl", "label_days_to_tp",
    "label_days_to_sl", "label_same_day_ambiguous",
)
_VARIANT_LABELS = tuple(
    f"label_{f}_{suf}"
    for suf, _, _, _ in LABEL_VARIANTS
    for f in ("tp_before_sl", "days_to_tp", "same_day_ambiguous", "touch")
)
PANEL_LABELS = _BASE_LABELS + _VARIANT_LABELS

# ----------------------------------------------------------------------
# WHICH UNIVERSE DOES THE PANEL REPRESENT?
#
# Two things are easy to conflate, so this states the rule explicitly:
#
#   TRADABILITY is point-in-time. A stock that had not listed in 2019 has no
#   2019 rows and cannot influence anyone's 2019 rank. That is handled by
#   construction - no bar, no row.
#
#   DATA QUALITY is a property of the WHOLE series. The gate inspects a
#   symbol's entire history: interior session gaps, OHLC violations, an
#   unresolved split. When a symbol is quarantined, the claim is not "it was
#   untradeable last week" - it is "this series' data cannot be trusted,
#   including its past".
#
# So a quarantined symbol has ALL its rows removed, history included. Keeping
# 2019 rows for a series we currently refuse to trust is the worst of both
# worlds: the panel would carry data the gate has rejected.
#
# The consequence is the important part. Cross-sectional ranks are computed
# ACROSS the universe on each date, so removing a symbol changes every other
# symbol's rank on every date it appeared. An incremental run cannot repair
# that by touching the recent tail. Therefore: if the allowed universe has
# changed since the last build, the panel is rebuilt in FULL, automatically.
# ----------------------------------------------------------------------


def _universe_hash(symbols: Sequence[str]) -> str:
    import hashlib
    return hashlib.sha256("|".join(sorted(symbols)).encode()).hexdigest()[:16]


def _build_signature(symbols: Sequence[str], min_turnover: float) -> str:
    """
    Everything that changes what a panel row MEANS.

    v28: universe_hash alone was not enough. A run with the liquidity floor
    off produced rows drawn from a different effective universe, and a run
    with new label columns produced rows with a different schema - and the
    gate-cleared symbol list was identical in both cases, so no full rebuild
    was triggered. The result was a panel whose recent tail did not match its
    own history: different universe, different columns, labels present for 20
    sessions and NaN for 2880.

    Any change here forces a FULL rebuild, because an incremental tail cannot
    make old rows agree with new ones.
    """
    import hashlib
    parts = [
        "u=" + _universe_hash(symbols),
        f"liq={float(min_turnover):.6g}",
        f"win={LIQ_WINDOW}",
        "variants=" + ";".join(f"{a}:{b}:{c}:{int(d)}"
                               for a, b, c, d in LABEL_VARIANTS),
        f"primary={PRIMARY_VARIANT}",
        f"h={LABEL_HORIZON}",
        "labels=" + ",".join(sorted(PANEL_LABELS)),
        "xs=" + ",".join(sorted(CROSS_SECTIONAL_BASE)),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------
def compute_labels(
    df: pd.DataFrame, horizon: int = LABEL_HORIZON, touch: float = LABEL_TOUCH_PCT
) -> pd.DataFrame:
    """
    Forward outcomes for one symbol. DELIBERATELY forward-looking.

    label_touch is the BigMove target: did the high reach +touch% at any point
    in the next `horizon` sessions, measured from today's close.

    Every label is NaN where the window is incomplete. That matters: an
    unresolved row is UNKNOWN, not a negative. Filling it with 0 would teach
    the model that the most recent week never moves.
    """
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")

    # Forward windows exclude today's bar: shift(-1) then roll forward.
    fwd_max = h.shift(-1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    fwd_min = l.shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    fwd_close = c.shift(-horizon)

    out = pd.DataFrame(index=df.index)
    out["label_mfe_5d"] = (fwd_max / c) - 1.0
    out["label_mae_5d"] = (fwd_min / c) - 1.0
    out["label_fwd_ret_5d"] = (fwd_close / c) - 1.0
    reached = out["label_mfe_5d"] >= touch
    out["label_touch"] = reached.astype("Int8").mask(out["label_mfe_5d"].isna())

    # ------------------------------------------------------------------
    # PATH-DEPENDENT LABELS
    #
    # MFE and MAE are both extremes over the window; neither says which came
    # FIRST. A stock that falls 3% on day 1 then rallies 6% on day 4 has the
    # same MFE/MAE as one that rallies 6% on day 1 then falls 3% - opposite
    # outcomes under a bracket. P(TP before SL) needs the order, so it needs
    # the path.
    #
    # THE HONEST LIMIT: daily bars do not record whether the high or the low
    # came first WITHIN a day. When both barriers are touched on the same
    # session the outcome is genuinely unknowable from this data. Those rows
    # are counted as SL (the conservative reading) and flagged in
    # label_same_day_ambiguous so the rate can be measured with and without
    # them. If that flag is a large share of your episodes, this label needs
    # intraday bars, not a better assumption.
    # ------------------------------------------------------------------
    n = len(df)
    c_arr = c.to_numpy(dtype="float64")
    h_arr = h.to_numpy(dtype="float64")
    l_arr = l.to_numpy(dtype="float64")

    # ATR% per bar, trailing only, for the ATR-scaled variants.
    tr = np.maximum(h_arr[1:] - l_arr[1:],
                    np.maximum(np.abs(h_arr[1:] - c_arr[:-1]),
                               np.abs(l_arr[1:] - c_arr[:-1])))
    atr14 = np.concatenate(
        [[np.nan], pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()])
    atr_pct = atr14 / np.where(c_arr == 0, np.nan, c_arr)

    # Running extremes over the forward window - the same trick barrier_scan
    # uses. Computing them once serves every variant.
    hz = horizon
    if n > hz:
        idx = np.arange(0, n - hz)
        off = np.arange(1, hz + 1)
        fh = h_arr[idx[:, None] + off]
        fl = l_arr[idx[:, None] + off]
        base = c_arr[idx][:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            run_hi = np.maximum.accumulate(fh / base - 1.0, axis=1)
            run_lo = np.minimum.accumulate(fl / base - 1.0, axis=1)
    else:
        idx = np.array([], dtype=int)
        run_hi = run_lo = np.zeros((0, hz))

    def _variant(tp, sl, use_atr):
        first = np.full(n, np.nan)
        d_tp = np.full(n, np.nan)
        d_sl = np.full(n, np.nan)
        amb = np.full(n, np.nan)
        touch = np.full(n, np.nan)
        if not len(idx):
            return first, d_tp, d_sl, amb, touch
        if use_atr:
            lvl_tp = tp * atr_pct[idx]
            lvl_sl = sl * atr_pct[idx]
        else:
            lvl_tp = np.full(len(idx), float(tp))
            lvl_sl = np.full(len(idx), float(sl))
        ok = np.isfinite(lvl_tp) & np.isfinite(lvl_sl)
        hit_tp = run_hi >= lvl_tp[:, None]
        hit_sl = run_lo <= -lvl_sl[:, None]
        k_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), hz)
        k_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), hz)
        t_tp, t_sl = k_tp < hz, k_sl < hz
        tie = t_tp & t_sl & (k_tp == k_sl)
        res = np.where(t_tp & (k_tp < k_sl), 1.0,
                       np.where(t_sl & (k_sl < k_tp), -1.0,
                                np.where(tie, -1.0, 0.0)))   # tie -> SL
        first[idx] = np.where(ok, res, np.nan)
        d_tp[idx] = np.where(ok & t_tp, k_tp + 1, np.nan)
        d_sl[idx] = np.where(ok & t_sl, k_sl + 1, np.nan)
        amb[idx] = np.where(ok, tie.astype(float), np.nan)
        touch[idx] = np.where(ok, t_tp.astype(float), np.nan)
        return first, d_tp, d_sl, amb, touch

    new = {}
    for suf, tp, sl, use_atr in LABEL_VARIANTS:
        f, dtp, dsl, amb, tch = _variant(tp, sl, use_atr)
        new[f"label_tp_before_sl_{suf}"] = pd.Series(
            np.where(np.isnan(f), np.nan, (f > 0).astype(float)),
            index=out.index).astype("Float64")
        new[f"label_days_to_tp_{suf}"] = dtp
        new[f"label_same_day_ambiguous_{suf}"] = pd.Series(
            amb, index=out.index).astype("Float64")
        new[f"label_touch_{suf}"] = pd.Series(tch, index=out.index).astype("Float64")
        if suf == PRIMARY_VARIANT:
            new["label_first_touch"] = f
            new["label_tp_before_sl"] = new[f"label_tp_before_sl_{suf}"]
            new["label_days_to_tp"] = dtp
            new["label_days_to_sl"] = dsl
            new["label_same_day_ambiguous"] = new[f"label_same_day_ambiguous_{suf}"]

    out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    return out


# ----------------------------------------------------------------------
# per-symbol stage
# ----------------------------------------------------------------------
def symbol_frame(root: Path, symbol: str, since: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Full-history features for one symbol, sliced to the rows we will write."""
    pq, _ = _paths(root, symbol)
    raw = pd.read_parquet(pq)
    if raw.empty or len(raw) < fx.WARMUP_BARS + LABEL_HORIZON:
        return pd.DataFrame()

    feat = fx.build_features(raw)          # full history: expanding ranks need it

    # Labels are computed on the RAW frame and joined ON TIMESTAMP.
    #
    # Joining on the positional index is wrong and silently so: drop_warmup()
    # calls reset_index(), so after it the row numbers no longer correspond to
    # the raw frame. An index join would have paired each feature row with a
    # label from ~312 bars away - a model that trains beautifully and means
    # nothing. Timestamp is the only key that survives reindexing.
    lab = compute_labels(raw)
    lab.insert(0, "timestamp", pd.to_datetime(raw["timestamp"]).to_numpy())

    feat = fx.drop_warmup(feat)
    if feat.empty:
        return pd.DataFrame()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])
    if getattr(feat["timestamp"].dt, "tz", None) is not None:
        feat["timestamp"] = feat["timestamp"].dt.tz_localize(None)
    lab["timestamp"] = pd.to_datetime(lab["timestamp"])
    if getattr(lab["timestamp"].dt, "tz", None) is not None:
        lab["timestamp"] = lab["timestamp"].dt.tz_localize(None)

    before = len(feat)
    feat = feat.merge(lab, on="timestamp", how="left", validate="one_to_one")
    if len(feat) != before:
        raise RuntimeError(f"{symbol}: label join changed row count")

    feat.insert(1, "symbol", symbol)
    if since is not None:
        feat = feat.loc[feat["timestamp"] >= since]
    return feat.reset_index(drop=True)


# ----------------------------------------------------------------------
# cross-sectional stage
# ----------------------------------------------------------------------
def apply_liquidity_floor(
    panel: pd.DataFrame,
    *,
    min_turnover: float = LIQ_MIN_TURNOVER,
    window: int = LIQ_WINDOW,
    min_obs: int = LIQ_MIN_OBSERVATIONS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop rows whose trailing turnover, as of that date, was below the floor.

    Runs BEFORE the cross-sectional stage on purpose. A stock that fails the
    floor should not be in the pool that ranks everyone else - otherwise an
    untradeable name still shifts every other symbol's percentile.

    D_dollar_vol is close * volume for that session, so the rolling median is
    computed from bars at or before t and never looks ahead.
    """
    if not min_turnover or "D_dollar_vol" not in panel.columns:
        if verbose:
            print("  liquidity floor: disabled (LIQ_MIN_TURNOVER=0)")
        return panel

    p = panel.sort_values(["symbol", "timestamp"])
    dv = pd.to_numeric(p["D_dollar_vol"], errors="coerce")
    g = dv.groupby(p["symbol"], sort=False)
    med = g.rolling(window, min_periods=min_obs).median().reset_index(level=0, drop=True)
    p = p.assign(X_turnover_med=med.reindex(p.index))

    keep = p["X_turnover_med"] >= min_turnover
    dropped = int((~keep).sum())
    if verbose:
        surviving = p.loc[keep, "symbol"].nunique()
        print(f"  liquidity floor: {min_turnover:,.0f}/day over {window} sessions "
              f"-> dropped {dropped:,} rows, {surviving} symbols remain "
              f"in at least one session")
        print(f"    (point-in-time: a name can enter and leave the universe "
              f"as its turnover changes)")
    return p.loc[keep].sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_cross_sectional(panel: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Rank within each date, across the symbols present ON that date.

    This is the real cross-sectional rank the per-symbol expanding rank was
    only ever standing in for. The universe per date is whoever has a row,
    which is point-in-time by construction: a symbol that had not listed has
    no row and cannot influence anyone else's rank.
    """
    have = [c for c in cols if c in panel.columns]
    if not have:
        return panel
    g = panel.groupby("timestamp", sort=False)
    ranked = {}
    for c in have:
        ranked[f"X_rank_{c}"] = g[c].rank(pct=True, method="average")
        z = (panel[c] - g[c].transform("mean")) / g[c].transform("std").replace(0, np.nan)
        ranked[f"X_z_{c}"] = z.clip(-5, 5)
    ranked["X_universe_size"] = g["symbol"].transform("size").astype("int32")
    return pd.concat([panel, pd.DataFrame(ranked, index=panel.index)], axis=1)


def add_market_relative(panel: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Return and volatility relative to the index, plus the index's own state."""
    pq, _ = _paths(root, MARKET_SERIES)
    if not pq.exists():
        print(f"  WARNING: {MARKET_SERIES} not cached; skipping market-relative")
        return panel
    mkt = pd.read_parquet(pq)
    mf = fx.drop_warmup(fx.build_features(mkt))
    keep = [c for c in ("D_ret_1d_pct", "D_realvol_20", "D_rsi14", "D_adx14",
                        "D_dist_from_52wh", "D_drawdown_252") if c in mf.columns]
    mf = mf[["timestamp"] + keep].copy()
    mf["timestamp"] = pd.to_datetime(mf["timestamp"]).dt.tz_localize(None)
    mf = mf.rename(columns={c: f"MKT_{c}" for c in keep})

    panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_localize(None)
    out = panel.merge(mf, on="timestamp", how="left")
    if "D_ret_1d_pct" in out.columns and "MKT_D_ret_1d_pct" in out.columns:
        out["X_excess_ret_1d"] = out["D_ret_1d_pct"] - out["MKT_D_ret_1d_pct"]
    if "D_realvol_20" in out.columns and "MKT_D_realvol_20" in out.columns:
        out["X_relvol_20"] = out["D_realvol_20"] / out["MKT_D_realvol_20"].replace(0, np.nan)
    return out


def add_exogenous(panel: pd.DataFrame, root: Path, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Join macro series onto the panel, already lagged by the cache's rules.

    Each exogenous series carries lag_sessions, set to 1 for anything closing
    after NSE. That lag is applied here, so a crude close that printed at
    23:30 can never appear in a same-day feature row.
    """
    import Daily_cache_v27 as dc
    added = []
    for s in dc.DEFAULT_EXOGENOUS:
        if s.name == MARKET_SERIES:
            continue
        pq, _ = _paths(root, s.name)
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        if raw.empty:
            continue
        al = dc.align_to_sessions(raw, sessions, series=s)
        if al.empty:
            continue
        keep = [c for c in al.columns if c.endswith(("_close", "_stale"))]
        al = al[keep].copy()
        al.index = pd.DatetimeIndex(al.index).tz_localize(None)
        al[f"{s.name}_ret_1d"] = pd.to_numeric(
            al.get(f"{s.name}_close"), errors="coerce"
        ).pct_change()
        al = al.reset_index().rename(columns={"index": "timestamp"})
        panel = panel.merge(al, on="timestamp", how="left")
        added.append(s.name)
    if added:
        print(f"  exogenous joined ({len(added)}): {', '.join(added[:8])}"
              + (" ..." if len(added) > 8 else ""))
    return panel


# ----------------------------------------------------------------------
# walk-forward with embargo
# ----------------------------------------------------------------------
def walk_forward_splits(
    panel: pd.DataFrame, n_splits: int = 5, embargo: int = LABEL_HORIZON
) -> List[dict]:
    """
    Expanding-window splits with an embargo between train and test.

    The embargo is not optional. A row dated t carries a label that resolves
    at t+5, so training on t and testing at t+1 lets the model see an outcome
    that overlaps the test period. `embargo` must be >= the label horizon.
    """
    if embargo < LABEL_HORIZON:
        raise ValueError(
            f"embargo {embargo} < label horizon {LABEL_HORIZON}: train labels "
            f"would overlap the test window"
        )
    # v28 fix: .unique() returns an extension array on pandas 3.x, and
    # np.searchsorted on it compared an int against a Timestamp. Work in a
    # DatetimeIndex throughout and use its own searchsorted.
    dates = pd.DatetimeIndex(pd.to_datetime(panel["timestamp"])).unique().sort_values()
    if len(dates) < (n_splits + 1) * (embargo + 10):
        raise ValueError(
            f"only {len(dates)} sessions; need "
            f"{(n_splits + 1) * (embargo + 10)} for {n_splits} splits"
        )
    folds = np.array_split(np.arange(len(dates)), n_splits + 1)
    out = []
    for k in range(1, n_splits + 1):
        test_pos = folds[k]
        train_end_pos = folds[k - 1][-1]
        # Pull the train end BACK by the embargo so no training label's
        # outcome window can reach into the test period.
        train_last = dates[max(0, train_end_pos - embargo)]
        test = dates[test_pos]
        out.append({
            "fold": k,
            "train_start": str(pd.Timestamp(dates[0]).date()),
            "train_end": str(train_last.date()),
            "embargo_sessions": embargo,
            "test_start": str(pd.Timestamp(test[0]).date()),
            "test_end": str(pd.Timestamp(test[-1]).date()),
        })
    return out


# ----------------------------------------------------------------------
# main build
# ----------------------------------------------------------------------
def build_panel(
    root,
    panel_dir,
    *,
    symbols: Optional[Sequence[str]] = None,
    full: bool = False,
    rebuild_days: int = 15,
    allow_partial: bool = False,
    min_turnover: float = LIQ_MIN_TURNOVER,
    verbose: bool = True,
) -> Path:
    root, panel_dir = Path(root), Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    out_pq = panel_dir / "panel.parquet"

    syms = list(symbols) if symbols else discover_symbols(root)
    if verbose:
        print(f"[panel] gating {len(syms)} cached series")
    allowed = assert_panel_ready(root, syms, allow_partial=allow_partial)
    if not allowed:
        raise RuntimeError("no series cleared the data-quality gate")

    # Equities only in the cross-section; indices and futures are context.
    equities = []
    for s in allowed:
        meta = _read_ok(_paths(root, s)[1]) or {}
        if meta.get("series_kind", "equity") == "equity":
            equities.append(s)
    if verbose:
        print(f"[panel] {len(equities)} equities cleared, "
              f"{len(allowed) - len(equities)} context series")

    cal = build_calendar(root, allowed)
    sessions = cal.sessions
    if not len(sessions):
        raise RuntimeError("no trading calendar; cannot build a panel")

    uni_hash = _universe_hash(equities)
    sig = _build_signature(equities, min_turnover)
    prior_meta = {}
    meta_path = panel_dir / "panel_meta.json"
    if meta_path.exists():
        try:
            prior_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            prior_meta = {}

    if (not full) and prior_meta and prior_meta.get("build_signature") != sig:
        old_liq = prior_meta.get("liquidity_floor")
        reasons = []
        if prior_meta.get("universe_hash") != uni_hash:
            reasons.append(f"universe ({prior_meta.get('symbols','?')} -> "
                           f"{len(equities)} symbols)")
        if old_liq is not None and float(old_liq) != float(min_turnover):
            reasons.append(f"liquidity floor ({old_liq:,.0f} -> "
                           f"{min_turnover:,.0f})")
        if prior_meta.get("build_signature") is None:
            reasons.append("panel predates signature tracking")
        if not reasons:
            reasons.append("labels or cross-sectional feature set")
        print(f"  BUILD SIGNATURE CHANGED: {'; '.join(reasons)}.")
        print(f"  An incremental tail cannot make old rows agree with new "
              f"ones, so this is a FULL rebuild.")
        full = True

    existing = None
    since = None
    if not full and out_pq.exists():
        existing = pd.read_parquet(out_pq)
        if not existing.empty:
            # Rewrite the recent window: features there may have been revised
            # by the cache's tail refetch, and labels there have just become
            # resolvable.
            span = rebuild_days + LABEL_HORIZON
            cutoff_idx = max(0, len(sessions) - span)
            since = pd.Timestamp(sessions[cutoff_idx]).tz_localize(None)
            if verbose:
                print(f"[panel] incremental: rewriting from {since.date()} "
                      f"({span} sessions)")
    if full and verbose:
        print("[panel] FULL rebuild")

    frames, skipped = [], []
    for i, s in enumerate(equities, 1):
        try:
            f = symbol_frame(root, s, since)
            if not f.empty:
                frames.append(f)
        except Exception as e:
            skipped.append(f"{s}: {type(e).__name__}: {e}")
        if verbose and i % 200 == 0:
            print(f"  ...{i}/{len(equities)}")
    if skipped:
        print(f"  WARNING: {len(skipped)} symbol(s) failed: {skipped[:3]}")
    if not frames:
        raise RuntimeError("no symbol produced rows")

    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_localize(None)
    if verbose:
        print(f"[panel] stacked {len(panel):,} rows x {panel.shape[1]} cols")

    # Order matters: the floor runs BEFORE ranking, so an untradeable name
    # cannot influence anyone else's cross-sectional percentile.
    panel = apply_liquidity_floor(panel, min_turnover=min_turnover,
                                  verbose=verbose)
    if panel.empty:
        raise RuntimeError(
            f"liquidity floor of {min_turnover:,.0f}/day removed every row. "
            f"Lower it or set it to 0."
        )
    panel = add_cross_sectional(panel, CROSS_SECTIONAL_BASE)
    panel = add_market_relative(panel, root)
    panel = add_exogenous(panel, root, sessions)

    # Drop dates whose cross-section is too thin to rank meaningfully.
    thin = panel["X_universe_size"] < 20
    if thin.any():
        print(f"  dropping {int(thin.sum()):,} rows on dates with <20 symbols")
        panel = panel.loc[~thin]

    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if existing is not None and since is not None:
        keep = existing.loc[
            pd.to_datetime(existing["timestamp"]).dt.tz_localize(None) < since
        ]
        # Belt and braces: even though a universe change forces a full rebuild
        # above, never carry forward rows for a symbol the gate does not
        # currently allow.
        allowed_set = set(equities)
        stale = ~keep["symbol"].isin(allowed_set)
        if stale.any():
            print(f"  dropping {int(stale.sum()):,} historical rows for "
                  f"{keep.loc[stale, 'symbol'].nunique()} now-quarantined symbol(s)")
            keep = keep.loc[~stale]
        panel = pd.concat([keep, panel], ignore_index=True)
        panel = panel.drop_duplicates(["timestamp", "symbol"], keep="last")
        panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    panel.to_parquet(out_pq, index=False)

    lab = panel["label_touch"] if "label_touch" in panel.columns else pd.Series(dtype="Int8")
    resolved = int(lab.notna().sum())
    meta = {
        "built_at": dt.datetime.now().isoformat(),
        "rows": int(len(panel)),
        "columns": int(panel.shape[1]),
        "symbols": int(panel["symbol"].nunique()),
        "first_session": str(panel["timestamp"].min().date()),
        "last_session": str(panel["timestamp"].max().date()),
        "labels_resolved": resolved,
        "labels_pending": int(len(panel) - resolved),
        "label_positive_rate": float(lab.dropna().mean()) if resolved else None,
        "calendar_trusted": bool(cal.trusted),
        "liquidity_floor": float(min_turnover),
        "liquidity_window": LIQ_WINDOW,
        "universe_hash": uni_hash,
        "build_signature": sig,
        "label_tp": LABEL_TP,
        "label_sl": LABEL_SL,
        "universe": sorted(equities),
        "feature_columns": len(panel_feature_columns(panel)),
        "mode": "full" if full else "incremental",
    }
    (panel_dir / "panel_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(f"[panel] {meta['rows']:,} rows | {meta['symbols']} symbols | "
              f"{meta['first_session']} -> {meta['last_session']}")
        print(f"[panel] labels resolved {resolved:,}, pending "
              f"{meta['labels_pending']:,} (the most recent "
              f"{LABEL_HORIZON} sessions are UNKNOWN, not negative)")
        if meta["label_positive_rate"] is not None:
            print(f"[panel] base rate: {meta['label_positive_rate']:.3%}")
        print(f"[panel] -> {out_pq}")
    return out_pq


def panel_feature_columns(panel: pd.DataFrame) -> List[str]:
    """
    Columns safe to hand a model, at panel level.

    Extends features_daily's registry with the panel's own labels. Build X
    with this; never with panel.columns.
    """
    banned = set(fx.non_feature_columns(panel)) | set(PANEL_LABELS) | {"symbol"}
    banned |= {c for c in panel.columns if c.startswith("label_")}
    return [c for c in panel.columns if c not in banned]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--full", action="store_true", help="rebuild all history")
    ap.add_argument("--rebuild-days", type=int, default=15)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--min-turnover", type=float, default=LIQ_MIN_TURNOVER,
                    help="point-in-time daily turnover floor in rupees "
                         "(e.g. 1e7 for 1 crore); 0 disables")
    ap.add_argument("--splits", type=int, default=0,
                    help="print N walk-forward folds after building")
    a = ap.parse_args()

    p = build_panel(a.root, a.panel, full=a.full, rebuild_days=a.rebuild_days,
                    allow_partial=a.allow_partial,
                    min_turnover=a.min_turnover)
    if a.splits:
        panel = pd.read_parquet(p, columns=["timestamp"])
        for f in walk_forward_splits(panel, n_splits=a.splits):
            print(f"  fold {f['fold']}: train -> {f['train_end']} "
                  f"| embargo {f['embargo_sessions']} | "
                  f"test {f['test_start']}..{f['test_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
