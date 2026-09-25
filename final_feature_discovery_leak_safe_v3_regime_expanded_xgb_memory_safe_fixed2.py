#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASTER FEATURE DISCOVERY — LEAK-SAFE V3 — XGBOOST MEMORY-SAFE
=======================================

Purpose
-------
One-time, restartable feature-discovery run for an existing daily panel.

Design goals
------------
1. Never use future/label/outcome columns as model inputs.
2. Preserve the panel's decision clock: features at date t are usable at t+1.
3. Use expanding walk-forward folds with an explicit label-horizon purge/gap.
4. Test NEW features incrementally against the EXISTING baseline model,
   rather than claiming value from standalone single-feature models.
5. Use a staged search so a 10h+ run is useful rather than an uncontrolled
   combinatorial explosion.
6. Checkpoint every target/fold/stage so an interruption does not destroy work.
7. Never modify panel.parquet or approved_features.json.
8. Sector-relative features use dated sector-index columns already present in
   the panel. This is NOT stock->sector membership mapping.

Important
---------
This script deliberately does not invent point-in-time stock-sector membership.
If sector-index columns exist in the panel, sector-relative features are safe
provided those index observations are themselves available at the same close.

The final result is a RESEARCH UNION. It is NOT automatically promoted to the
live model. A separate model tournament/backtest should validate the selected
set economically.

Recommended run
---------------
python final_feature_discovery_leak_safe_v3.py

Optional:
python final_feature_discovery_leak_safe_v3.py --panel /path/panel.parquet
python final_feature_discovery_leak_safe_v3.py --out /path/discovery_v3
python final_feature_discovery_leak_safe_v3.py --workers 1
python final_feature_discovery_leak_safe_v3.py --force

Dependencies:
pandas, numpy, scipy, scikit-learn, xgboost, pyarrow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception as exc:
    raise RuntimeError(
        "This version requires XGBoost. Install it with: pip install -U xgboost"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

VERSION = "2026-09-25.master.v3.regime_expanded.xgb_memory_safe"
SEED = 1729
XGB_N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))

N_FOLDS = 5
LABEL_HORIZON = 5
EMBARGO = 5

# Staging limits. Increase if you intentionally want a much longer run.
MAX_CANDIDATES_STAGE_A = 600
MAX_CANDIDATES_STAGE_B = 250
MAX_CANDIDATES_STAGE_C = 150
MAX_FINAL_RESEARCH = 75

# Additive regime-conditioned research layer.
REGIME_NAMES = ("TREND_UP", "RANGE", "TREND_DOWN")
MAX_REGIME_FEATURES = 15
MIN_REGIME_TRAIN_ROWS = 500
MIN_REGIME_TEST_ROWS = 50

MIN_COVERAGE = 0.70
MIN_DAILY_IC_OBS = 40
REDUNDANCY_THRESHOLD = 0.97

# Incremental model thresholds are deliberately descriptive rather than
# "magic trading thresholds". Promotion is decided later by the tournament.
MIN_DELTA_AUC = 0.002
MIN_DELTA_LOGLOSS = 0.0005

BANNED_EXACT = {
    "ret_1d_close_pct",
    "ret_3d_close_pct",
    "ret_5d_close_pct",
    "ret_1d_oc_pct",
    "ret_3d_oc_pct",
    "ret_5d_oc_pct",
    "ret_5d_open_to_close_pct",
}

BANNED_TOKENS = (
    "label",
    "target",
    "future",
    "fwd",
    "forward",
    "next_",
    "tp_hit",
    "sl_hit",
    "exit",
    "outcome",
    "lead",
)

ID_COLS = {"symbol", "timestamp", "_date", "_row_id"}

DEFAULT_TARGETS = (
    "label_tp_before_sl",
    "label_tp_before_sl_3p2",
    "label_tp_before_sl_5p3",
)

# Existing approved feature file is searched in these locations.
APPROVED_NAMES = (
    "approved_features.json",
    "approved_features_v2.json",
)

# Possible economic forward-return columns. They are NEVER used as X.
EVAL_RETURN_COLUMNS = (
    "ret_5d_open_to_close_pct",
    "ret_5d_oc_pct",
    "ret_5d_close_pct",
)


# ============================================================
# UTILITIES
# ============================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def atomic_write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(s))[:180]


def suspicious(name: str) -> bool:
    n = str(name).lower()
    if n in {x.lower() for x in BANNED_EXACT}:
        return True
    return any(tok in n for tok in BANNED_TOKENS)


def is_numeric_candidate(name: str, df: pd.DataFrame) -> bool:
    if name in ID_COLS or suspicious(name):
        return False
    return pd.api.types.is_numeric_dtype(df[name])


def median_impute_fit_transform(
    xtr: pd.DataFrame, xte: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    a = imp.fit_transform(xtr)
    b = imp.transform(xte)
    return a, b


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def stage_done(out: Path, key: str, signature: str) -> bool:
    p = out / "checkpoints" / f"{safe_name(key)}.json"
    if not p.exists():
        return False
    try:
        x = json.loads(p.read_text())
        return x.get("status") == "COMPLETE" and x.get("signature") == signature
    except Exception:
        return False


def mark_stage(out: Path, key: str, signature: str, payload=None) -> None:
    p = out / "checkpoints" / f"{safe_name(key)}.json"
    atomic_write_json(
        {
            "status": "COMPLETE",
            "key": key,
            "signature": signature,
            "version": VERSION,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "payload": payload or {},
        },
        p,
    )


def stage_start(out: Path, key: str, signature: str) -> None:
    p = out / "checkpoints" / f"{safe_name(key)}.started.json"
    atomic_write_json(
        {
            "status": "STARTED",
            "key": key,
            "signature": signature,
            "version": VERSION,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        p,
    )


# ============================================================
# DATA LOADING + STATIC LEAK GATE
# ============================================================

def memory_log(df: pd.DataFrame, label: str) -> None:
    """Log dataframe memory and block count without forcing consolidation."""
    try:
        gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
        blocks = int(getattr(df, "_mgr", getattr(df, "_data", None)).nblocks)
        log(f"Memory checkpoint [{label}]: {gb:.2f} GiB | blocks={blocks:,} | cols={len(df.columns):,}")
    except Exception:
        pass



def load_panel(path: Path) -> pd.DataFrame:
    log(f"Loading panel: {path}")
    df = pd.read_parquet(path)

    required = {"symbol", "timestamp", "close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Panel missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.loc[df["timestamp"].notna()].copy()
    df["_date"] = df["timestamp"].dt.normalize()
    df["_row_id"] = np.arange(len(df), dtype=np.int64)

    # Stable ordering is critical for all groupby/rolling operations.
    df = df.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)

    if df[["symbol", "timestamp"]].duplicated().any():
        dup = int(df[["symbol", "timestamp"]].duplicated().sum())
        raise RuntimeError(f"Duplicate symbol/timestamp rows: {dup}")

    return df


def static_leak_gate(df: pd.DataFrame) -> dict:
    """Hard gate for candidate/model input columns."""
    all_numeric = [c for c in df.columns if is_numeric_candidate(c, df)]
    forbidden = [c for c in df.columns if suspicious(c)]
    exact = [c for c in df.columns if c.lower() in {x.lower() for x in BANNED_EXACT}]

    if exact:
        # These may exist in the panel for evaluation, but can never enter X.
        log(f"Evaluation-only forward columns present: {len(exact)}")

    return {
        "numeric_columns": len(all_numeric),
        "forbidden_or_future_named_columns": sorted(set(forbidden)),
        "evaluation_forward_columns": sorted(set(exact)),
    }


def find_approved_features(panel_path: Path) -> list[str]:
    roots = [
        panel_path.parent,
        Path.cwd(),
        panel_path.parent.parent,
    ]
    for root in roots:
        for name in APPROVED_NAMES:
            p = root / name
            if not p.exists():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    for key in ("features", "approved_features", "base_features"):
                        if isinstance(obj.get(key), list):
                            return [str(x) for x in obj[key]]
                if isinstance(obj, list):
                    return [str(x) for x in obj]
            except Exception:
                continue
    return []


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def group_shift(df: pd.DataFrame, col: str, n: int = 1) -> pd.Series:
    return df.groupby("symbol", sort=False)[col].shift(n)


def roll(df: pd.DataFrame, col: str, window: int, min_periods: Optional[int] = None):
    if min_periods is None:
        min_periods = max(2, window // 2)
    return (
        df.groupby("symbol", sort=False)[col]
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_std(df: pd.DataFrame, s: pd.Series, window: int):
    # Reindex through the already aligned series; groupby rolling preserves
    # the sorted symbol/timestamp order.
    return (
        s.groupby(df["symbol"], sort=False)
        .rolling(window, min_periods=max(2, window // 2))
        .std()
        .reset_index(level=0, drop=True)
    )


def rolling_mean_series(df: pd.DataFrame, s: pd.Series, window: int):
    return (
        s.groupby(df["symbol"], sort=False)
        .rolling(window, min_periods=max(2, window // 2))
        .mean()
        .reset_index(level=0, drop=True)
    )


def add_derived_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Build derived candidates in memory-efficient batches.

    IMPORTANT: do not insert hundreds of columns one-by-one into the large
    panel.  Pandas can fragment the DataFrame when columns are repeatedly
    inserted.  We therefore collect new columns in a dict and attach them
    once with pd.concat(axis=1).
    """
    x = df
    base_id = x["_row_id"].to_numpy(copy=False)
    g = x.groupby("symbol", sort=False)
    close = x["close"].astype(float)
    opn = x["open"].astype(float) if "open" in x else close
    high = x["high"].astype(float) if "high" in x else close
    low = x["low"].astype(float) if "low" in x else close
    volume = x["volume"].astype(float) if "volume" in x else pd.Series(np.nan, index=x.index)

    new_cols: dict[str, pd.Series] = {}
    candidates: list[dict] = []

    def add(name: str, series, family: str, lookback: int, hypothesis: str):
        if name in x.columns or name in new_cols:
            return
        new_cols[name] = pd.to_numeric(series, errors="coerce").astype("float32")
        candidates.append({
            "feature": name, "family": family, "lookback": lookback,
            "hypothesis": hypothesis, "resolved_at": "t_close",
            "usable_from": "t+1", "pit_status": "causal_same_symbol",
        })

    for w, desc in ((1,"short-term price persistence"),(3,"short-term momentum"),
                    (5,"5-session momentum"),(10,"medium momentum"),
                    (20,"one-month momentum"),(60,"quarter-scale momentum")):
        add(f"FD_ret_{w}", g["close"].pct_change(w), "momentum", w, desc)

    add("FD_intraday_ret", close / opn.replace(0, np.nan) - 1.0,
        "price_structure", 1, "same-day close versus open")
    add("FD_gap", opn / group_shift(x, "close", 1).replace(0, np.nan) - 1.0,
        "price_structure", 1, "overnight gap")
    add("FD_range_pct", (high-low) / close.replace(0, np.nan),
        "volatility", 1, "daily range relative to close")
    add("FD_body_pct", (close-opn) / close.replace(0, np.nan),
        "price_structure", 1, "candle body")
    add("FD_close_pos", (close-low) / (high-low).replace(0, np.nan),
        "price_structure", 1, "close location within daily range")

    for w in (5,10,20,60):
        mean_c = roll(x, "close", w)
        std_c = rolling_std(x, close, w)
        add(f"FD_price_z_{w}", (close-mean_c)/std_c.replace(0,np.nan),
            "mean_reversion", w, "distance from rolling price mean")
        # FD_range_pct is already staged in new_cols, so reference it there.
        range_s = new_cols["FD_range_pct"]
        add(f"FD_range_mean_{w}", rolling_mean_series(x, range_s, w),
            "volatility", w, "average daily range")
        if "volume" in x:
            mean_v = rolling_mean_series(x, volume, w)
            std_v = rolling_std(x, volume, w)
            add(f"FD_vol_z_{w}", (volume-mean_v)/std_v.replace(0,np.nan),
                "volume", w, "volume surprise")
            add(f"FD_vol_ratio_{w}", volume/mean_v.replace(0,np.nan),
                "volume", w, "volume relative to rolling average")

    for w in (10,20,50,100,200):
        sma = g["close"].rolling(w, min_periods=max(2,w//2)).mean().reset_index(level=0, drop=True)
        add(f"FD_dist_sma_{w}", close/sma.replace(0,np.nan)-1.0,
            "trend", w, "distance from moving average")

    prev_close = group_shift(x, "close", 1)
    tr = pd.concat([
        (high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()
    ], axis=1).max(axis=1)
    for w in (5,14,20,60):
        atr = (x.assign(_tmp_tr=tr).groupby("symbol", sort=False)["_tmp_tr"]
               .rolling(w, min_periods=max(2,w//2)).mean()
               .reset_index(level=0, drop=True))
        add(f"FD_atr_pct_{w}", atr/close.replace(0,np.nan),
            "volatility", w, "ATR normalized by price")

    for w in (10,20,60):
        old = group_shift(x, "close", w)
        add(f"FD_slope_{w}", (close/old.replace(0,np.nan)-1.0)/max(w,1),
            "trend", w, "normalized rolling price slope")

    rank_source = [c for c in ("FD_ret_5","FD_ret_20","FD_atr_pct_14",
                               "FD_intraday_ret","FD_vol_z_20")
                   if c in new_cols or c in x.columns]
    for c in rank_source:
        name=f"FD_XRANK_{c}"
        if name not in x.columns:
            new_cols[name]=new_cols[c].groupby(x["_date"]).rank(pct=True, method="average").astype("float32")
            candidates.append({"feature":name,"family":"cross_sectional","lookback":20,
                              "hypothesis":f"cross-sectional rank of {c}",
                              "resolved_at":"t_close","usable_from":"t+1",
                              "pit_status":"date_cross_section"})

    if new_cols:
        block=pd.DataFrame(new_cols, index=x.index)
        x=pd.concat([x, block], axis=1, copy=False)

    if not np.array_equal(base_id, x["_row_id"].to_numpy()):
        raise RuntimeError("FATAL: row-id misalignment after feature construction.")
    return x, candidates


def add_existing_feature_zscores(
    df: pd.DataFrame, existing_cols: list[str], max_features: int = 100
) -> tuple[pd.DataFrame, list[dict]]:
    """Add existing-feature z-scores in one column block, avoiding fragmentation."""
    x=df
    candidates=[]; new_cols={}
    usable=[c for c in existing_cols if c in x.columns and is_numeric_candidate(c,x)][:max_features]
    for c in usable:
        s=pd.to_numeric(x[c],errors="coerce")
        mu=s.groupby(x["symbol"],sort=False).rolling(60,min_periods=20).mean().reset_index(level=0,drop=True)
        sd=s.groupby(x["symbol"],sort=False).rolling(60,min_periods=20).std().reset_index(level=0,drop=True)
        name=f"FD_Z60__{c}"
        new_cols[name]=((s-mu)/sd.replace(0,np.nan)).astype("float32")
        candidates.append({"feature":name,"family":"existing_feature_normalization","lookback":60,
                          "hypothesis":f"causal normalization of existing {c}",
                          "resolved_at":"t_close","usable_from":"t+1",
                          "pit_status":"inherits_existing_feature"})
    if new_cols:
        x=pd.concat([x,pd.DataFrame(new_cols,index=x.index)],axis=1,copy=False)
    return x,candidates

def add_existing_panel_feature_expansions(
    panel_df: pd.DataFrame,
    work_df: pd.DataFrame,
    exclude_features: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Expand original numeric panel features without repeated column insertion."""
    x=work_df
    base_id=x["_row_id"].to_numpy(copy=False)
    exclude=set(exclude_features or [])
    source_cols=[c for c in panel_df.columns
                 if c not in ID_COLS and not str(c).startswith("_")
                 and not str(c).startswith("FD_") and not str(c).startswith("REG_")
                 and not suspicious(c)
                 and c not in {"open","high","low","close","volume","adj_close"}
                 and pd.api.types.is_numeric_dtype(panel_df[c])]
    source_cols=sorted(dict.fromkeys(source_cols),key=str)
    candidates=[]; new_cols={}
    for c in source_cols:
        if c not in exclude:
            candidates.append({"feature":c,"family":"existing_panel_raw","lookback":0,
                              "hypothesis":f"raw existing panel feature {c}","source_feature":c,
                              "resolved_at":"t_close","usable_from":"t+1",
                              "pit_status":"inherits_existing_feature_static_gate"})
        s=pd.to_numeric(x[c],errors="coerce")
        lag1=s.groupby(x["symbol"],sort=False).shift(1)
        lag5=s.groupby(x["symbol"],sort=False).shift(5)
        mu20=s.groupby(x["symbol"],sort=False).rolling(20,min_periods=10).mean().reset_index(level=0,drop=True)
        sd20=s.groupby(x["symbol"],sort=False).rolling(20,min_periods=10).std().reset_index(level=0,drop=True)
        xrank=x.groupby("_date")[c].rank(pct=True,method="average")
        specs=[
            (f"FD_PX_LAG1__{safe_name(c)}",lag1,"existing_feature_lag",1,f"one-session lag of existing panel feature {c}"),
            (f"FD_PX_D5__{safe_name(c)}",s-lag5,"existing_feature_change",5,f"five-session change of existing panel feature {c}"),
            (f"FD_PX_Z20__{safe_name(c)}",(s-mu20)/sd20.replace(0,np.nan),"existing_feature_normalization",20,f"20-session causal z-score of existing panel feature {c}"),
            (f"FD_PX_XRANK__{safe_name(c)}",xrank,"existing_feature_cross_section",1,f"same-date cross-sectional rank of existing panel feature {c}"),
        ]
        for name,series,family,lb,hyp in specs:
            if name not in x.columns and name not in new_cols:
                new_cols[name]=pd.to_numeric(series,errors="coerce").astype("float32")
                candidates.append({"feature":name,"family":family,"lookback":lb,
                                  "hypothesis":hyp,"source_feature":c,"resolved_at":"t_close",
                                  "usable_from":"t+1","pit_status":"inherits_existing_feature_static_gate"})
        if len(new_cols)%100==0:
            log(f"Existing-panel feature expansion staged: {len(new_cols):,} columns")
    if new_cols:
        log(f"Attaching existing-panel expansion block: {len(new_cols):,} columns")
        x=pd.concat([x,pd.DataFrame(new_cols,index=x.index)],axis=1,copy=False)
    if not np.array_equal(base_id,x["_row_id"].to_numpy()):
        raise RuntimeError("FATAL: row-id misalignment after existing-panel feature expansion.")
    return x,candidates

def add_sector_relative_features(
    df: pd.DataFrame,
    sector_columns: Optional[dict[str,str]]=None,
) -> tuple[pd.DataFrame,list[dict]]:
    """Add sector-relative features as one column block."""
    x=df; candidates=[]; new_cols={}
    if not sector_columns or "close" not in x: return x,candidates
    stock=x["close"].astype(float)
    for sector_name,col in sector_columns.items():
        if col not in x.columns:
            log(f"Sector index column not found; skipped: {sector_name} -> {col}")
            continue
        idx=pd.to_numeric(x[col],errors="coerce")
        temp=pd.DataFrame({"_date":x["_date"],"_idx":idx})
        daily_idx=temp.groupby("_date")["_idx"].first()
        for w in (1,5,20):
            sr=stock.groupby(x["symbol"],sort=False).pct_change(w)
            ir=daily_idx.pct_change(w).reindex(x["_date"]).to_numpy()
            name=f"FD_REL_{safe_name(sector_name)}_{w}"
            if name not in x.columns:
                new_cols[name]=pd.Series(sr.to_numpy()-ir,index=x.index,dtype="float32")
                candidates.append({"feature":name,"family":"sector_relative_index","lookback":w,
                                  "hypothesis":f"stock return relative to {sector_name} sector index",
                                  "resolved_at":"t_close","usable_from":"t+1",
                                  "pit_status":"dated_index_series_required","source_column":col})
    if new_cols:
        x=pd.concat([x,pd.DataFrame(new_cols,index=x.index)],axis=1,copy=False)
    return x,candidates


# ============================================================
# WALK-FORWARD / PURGE
# ============================================================

@dataclass
class Fold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_rows: int
    test_rows: int
    purge_sessions: int
    embargo_sessions: int


def build_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> list[Fold]:
    dates = np.array(sorted(pd.to_datetime(df["_date"].dropna().unique())))
    if len(dates) < (n_folds + 1) * 20:
        raise RuntimeError("Not enough dates for requested walk-forward folds.")

    blocks = np.array_split(dates, n_folds + 1)
    folds: list[Fold] = []

    for i in range(1, n_folds + 1):
        test_dates = blocks[i]
        test_start = pd.Timestamp(test_dates[0])
        test_end = pd.Timestamp(test_dates[-1])

        # Expanding train, with LABEL_HORIZON sessions removed immediately
        # before test. EMBARGO is explicitly retained as an additional buffer.
        pre_test_cut = max(0, i * 0)
        eligible = dates[dates < test_start]

        gap = max(LABEL_HORIZON, EMBARGO)
        train_dates = eligible[:-gap] if len(eligible) > gap else np.array([], dtype="datetime64[ns]")

        if len(train_dates) == 0:
            raise RuntimeError(f"Fold {i} has no training dates after purge/embargo.")

        train_start = pd.Timestamp(train_dates[0])
        train_end = pd.Timestamp(train_dates[-1])

        tr_rows = int(df["_date"].isin(train_dates).sum())
        te_rows = int(df["_date"].isin(test_dates).sum())

        folds.append(
            Fold(
                fold=i,
                train_start=str(train_start.date()),
                train_end=str(train_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                train_rows=tr_rows,
                test_rows=te_rows,
                purge_sessions=LABEL_HORIZON,
                embargo_sessions=EMBARGO,
            )
        )

    return folds


_FOLD_INDEX_CACHE: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}


def _cached_fold_indices(df: pd.DataFrame, folds: list[Fold]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute fold row indices once per working DataFrame."""
    key = id(df)
    cached = _FOLD_INDEX_CACHE.get(key)
    if cached is not None and all(f.fold in cached for f in folds):
        return cached

    cached = {}
    for f in folds:
        cached[f.fold] = fold_indices(df, f)
    _FOLD_INDEX_CACHE[key] = cached
    return cached


def fold_indices(df: pd.DataFrame, f: Fold) -> tuple[np.ndarray, np.ndarray]:
    tr = (
        (df["_date"] >= pd.Timestamp(f.train_start))
        & (df["_date"] <= pd.Timestamp(f.train_end))
    ).to_numpy()
    te = (
        (df["_date"] >= pd.Timestamp(f.test_start))
        & (df["_date"] <= pd.Timestamp(f.test_end))
    ).to_numpy()

    if np.any(df.loc[tr, "_date"] >= pd.Timestamp(f.test_start)):
        raise RuntimeError(f"Leak gate: train reaches test start in fold {f.fold}.")

    if np.any(df.loc[tr, "_date"] > pd.Timestamp(f.test_start) - pd.Timedelta(days=1)):
        raise RuntimeError(f"Leak gate: train/test temporal overlap in fold {f.fold}.")

    return np.flatnonzero(tr), np.flatnonzero(te)


def validate_fold_purge(df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    rows = []
    dates = sorted(pd.to_datetime(df["_date"].unique()))
    date_pos = {d: i for i, d in enumerate(dates)}

    for f in folds:
        ts = pd.Timestamp(f.test_start)
        te = pd.Timestamp(f.test_end)
        tr = pd.Timestamp(f.train_end)

        distance = date_pos[ts] - date_pos[tr]
        if distance < max(LABEL_HORIZON, EMBARGO) + 1:
            raise RuntimeError(
                f"Fold {f.fold}: insufficient purge distance={distance}"
            )

        rows.append({**asdict(f), "validated": True, "date_gap": distance})

    return pd.DataFrame(rows)


# ============================================================
# MODELS
# ============================================================

def make_baseline_model() -> xgb.XGBClassifier:
    """Regularized CPU XGBoost baseline used for OOS feature discovery."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=80,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        gamma=0.0,
        max_bin=256,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=XGB_N_JOBS,
        verbosity=0,
    )


def make_screen_model() -> LogisticRegression:
    return LogisticRegression(
        C=0.5,
        max_iter=500,
        class_weight="balanced",
        random_state=SEED,
    )


def fit_predict(
    model,
    xtr: pd.DataFrame,
    ytr: pd.Series,
    xte: pd.DataFrame,
) -> np.ndarray:
    """Fit/predict without full-panel imputation copies. XGBoost handles NaNs natively."""
    xtr = xtr.replace([np.inf, -np.inf], np.nan)
    xte = xte.replace([np.inf, -np.inf], np.nan)
    model.fit(xtr, ytr.astype(int).to_numpy())
    return model.predict_proba(xte)[:, 1]


def metric_pair(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    auc = roc_auc_score(y, p)
    ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    return float(auc), float(ll)


def top1_hit_rate(test: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> float:
    z = pd.DataFrame(
        {
            "_date": test["_date"].to_numpy(),
            "y": y,
            "p": p,
        }
    )
    top = z.sort_values(["_date", "p"], ascending=[True, False]).groupby("_date").head(1)
    return float(top["y"].mean()) if len(top) else np.nan


# ============================================================
# STAGE A — CHEAP CAUSAL SCREEN
# ============================================================

def daily_ic_score(
    df: pd.DataFrame, feature: str, target: str
) -> tuple[float, float, int]:
    z = df[["_date", feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if z.empty:
        return np.nan, np.nan, 0

    vals = []
    for _, g in z.groupby("_date", sort=False):
        if len(g) < 10 or g[feature].nunique() < 2 or g[target].nunique() < 2:
            continue
        r = g[feature].corr(g[target], method="spearman")
        if pd.notna(r):
            vals.append(float(r))

    if not vals:
        return np.nan, np.nan, 0

    a = np.asarray(vals)
    return float(np.mean(a)), float(np.median(a)), int(len(a))


def stage_a_screen(
    df: pd.DataFrame,
    candidates: list[dict],
    target: str,
    out: Path,
) -> pd.DataFrame:
    rows = []
    for i, meta in enumerate(candidates, 1):
        c = meta["feature"]
        if c not in df.columns or suspicious(c):
            continue

        coverage = float(df[c].notna().mean())
        if coverage < MIN_COVERAGE:
            continue

        ic_mean, ic_med, n = daily_ic_score(df, c, target)
        if n < MIN_DAILY_IC_OBS:
            continue

        rows.append(
            {
                **meta,
                "target": target,
                "coverage": coverage,
                "ic_mean": ic_mean,
                "ic_median": ic_med,
                "ic_abs_mean": abs(ic_mean),
                "ic_obs": n,
            }
        )

    r = pd.DataFrame(rows)
    if r.empty:
        raise RuntimeError(f"Stage A produced no candidates for {target}.")

    r = r.sort_values(
        ["ic_abs_mean", "coverage"], ascending=[False, False]
    ).head(MAX_CANDIDATES_STAGE_A)

    atomic_to_csv(r, out / f"stageA_{safe_name(target)}.csv")
    return r


# ============================================================
# STAGE B — REDUNDANCY
# ============================================================

def redundancy_filter(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    baseline_features: list[str],
    out: Path,
    target: str,
) -> pd.DataFrame:
    names = [c for c in candidates["feature"].tolist() if c in df.columns]

    # Sample by rows, but retain enough observations across the full period.
    rng = np.random.default_rng(SEED)
    if len(df) > 250_000:
        idx = rng.choice(len(df), size=250_000, replace=False)
        sample = df.iloc[np.sort(idx)]
    else:
        sample = df

    cols = [c for c in names if pd.api.types.is_numeric_dtype(sample[c])]
    if not cols:
        return candidates.iloc[0:0].copy()

    # Rank correlations are used only as a redundancy screen, never as proof
    # of incremental value.
    corr = sample[cols].corr(method="spearman", min_periods=500)

    selected = []
    rejected = []
    # Strong candidates first.
    ordered = candidates.sort_values(
        ["ic_abs_mean", "coverage"], ascending=[False, False]
    )["feature"].tolist()

    for c in ordered:
        redundant = False
        for s in selected:
            v = corr.loc[c, s] if c in corr.index and s in corr.columns else np.nan
            if pd.notna(v) and abs(float(v)) >= REDUNDANCY_THRESHOLD:
                redundant = True
                rejected.append((c, s, float(v)))
                break
        if not redundant:
            selected.append(c)

    keep = candidates[candidates["feature"].isin(selected)].copy()
    keep = keep.head(MAX_CANDIDATES_STAGE_B)

    atomic_to_csv(
        pd.DataFrame(rejected, columns=["feature", "representative", "spearman"]),
        out / f"redundancy_rejected_{safe_name(target)}.csv",
    )
    atomic_to_csv(keep, out / f"stageB_{safe_name(target)}.csv")
    return keep


# ============================================================
# STAGE C — STANDALONE OOS SCREEN
# ============================================================

def standalone_oos(
    df: pd.DataFrame,
    candidate: str,
    target: str,
    folds: list[Fold],
) -> dict:
    fold_rows = []

    for f in folds:
        tr, te = fold_indices(df, f)
        trdf = df.iloc[tr]
        tedf = df.iloc[te]

        xtr = trdf[[candidate]]
        xte = tedf[[candidate]]
        ytr = trdf[target]
        yte = tedf[target]

        good_tr = ytr.notna()
        good_te = yte.notna()
        xtr = xtr.loc[good_tr]
        ytr = ytr.loc[good_tr]
        xte = xte.loc[good_te]
        yte = yte.loc[good_te]

        if len(xtr) < 1000 or len(xte) < 100:
            continue
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", make_screen_model()),
            ]
        )
        pipe.fit(xtr, ytr.astype(int))
        p = pipe.predict_proba(xte)[:, 1]

        auc, ll = metric_pair(yte.to_numpy(), p)
        base_p = np.full(len(yte), float(ytr.mean()))
        _, base_ll = metric_pair(yte.to_numpy(), base_p)

        fold_rows.append(
            {
                "fold": f.fold,
                "auc": auc,
                "logloss": ll,
                "base_logloss": base_ll,
                "delta_logloss": base_ll - ll,
                "n_test": len(yte),
            }
        )

    if not fold_rows:
        return {"folds": 0}

    r = pd.DataFrame(fold_rows)
    return {
        "folds": int(len(r)),
        "mean_auc": float(r["auc"].mean()),
        "mean_delta_logloss": float(r["delta_logloss"].mean()),
        "min_delta_logloss": float(r["delta_logloss"].min()),
        "positive_folds": int((r["delta_logloss"] > 0).sum()),
    }


def stage_c_screen(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    target: str,
    folds: list[Fold],
    out: Path,
) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(candidates["feature"], 1):
        if i % 25 == 0:
            log(f"Stage C {target}: {i}/{len(candidates)}")

        try:
            r = standalone_oos(df, c, target, folds)
            rows.append({"feature": c, **r})
        except Exception as e:
            rows.append({"feature": c, "error": repr(e), "folds": 0})

    r = pd.DataFrame(rows)
    if r.empty:
        raise RuntimeError(f"Stage C produced no results for {target}.")

    r["screen_score"] = (
        r["mean_delta_logloss"].fillna(-np.inf)
        + 0.25 * (r["mean_auc"].fillna(0.5) - 0.5)
    )
    r = r.sort_values("screen_score", ascending=False).head(MAX_CANDIDATES_STAGE_C)

    atomic_to_csv(r, out / f"stageC_{safe_name(target)}.csv")
    return r


# ============================================================
# STAGE D — INCREMENTAL OOS VS EXISTING BASELINE
# ============================================================

def choose_baseline_features(
    df: pd.DataFrame, approved: list[str]
) -> list[str]:
    approved = [
        c for c in approved
        if c in df.columns and is_numeric_candidate(c, df)
    ]

    if approved:
        return approved

    # Fallback is intentionally conservative: use the original known approved
    # feature names if present. If none are present, fail rather than silently
    # building an arbitrary "baseline".
    known = [
        "D_rsi14",
        "D_atr_pct",
        "D_ema20_angle_deg",
        "D_dvol_z20",
        "D_pos_in_52w_range",
    ]
    known = [c for c in known if c in df.columns and is_numeric_candidate(c, df)]
    if known:
        return known

    raise RuntimeError(
        "No existing baseline feature set found. Refusing to invent a baseline."
    )


def incremental_oos(
    df: pd.DataFrame,
    baseline: list[str],
    candidate: str,
    target: str,
    folds: list[Fold],
) -> dict:
    fold_rows = []

    for f in folds:
        tr, te = fold_indices(df, f)
        trdf = df.iloc[tr]
        tedf = df.iloc[te]

        cols0 = baseline
        cols1 = baseline + [candidate]

        ytr = trdf[target]
        yte = tedf[target]
        good_tr = ytr.notna()
        good_te = yte.notna()

        trdf = trdf.loc[good_tr]
        tedf = tedf.loc[good_te]
        ytr = ytr.loc[good_tr].astype(int)
        yte = yte.loc[good_te].astype(int)

        if len(trdf) < 1000 or len(tedf) < 100:
            continue
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        p0 = fit_predict(
            make_baseline_model(),
            trdf[cols0],
            ytr,
            tedf[cols0],
        )
        p1 = fit_predict(
            make_baseline_model(),
            trdf[cols1],
            ytr,
            tedf[cols1],
        )

        auc0, ll0 = metric_pair(yte.to_numpy(), p0)
        auc1, ll1 = metric_pair(yte.to_numpy(), p1)

        hit0 = top1_hit_rate(tedf, yte.to_numpy(), p0)
        hit1 = top1_hit_rate(tedf, yte.to_numpy(), p1)

        fold_rows.append(
            {
                "fold": f.fold,
                "auc_base": auc0,
                "auc_plus": auc1,
                "delta_auc": auc1 - auc0,
                "logloss_base": ll0,
                "logloss_plus": ll1,
                "delta_logloss": ll0 - ll1,
                "top1_base": hit0,
                "top1_plus": hit1,
                "delta_top1": hit1 - hit0,
                "n_test": len(tedf),
            }
        )

    if not fold_rows:
        return {"folds": 0}

    r = pd.DataFrame(fold_rows)

    return {
        "folds": int(len(r)),
        "mean_delta_auc": float(r["delta_auc"].mean()),
        "median_delta_auc": float(r["delta_auc"].median()),
        "min_delta_auc": float(r["delta_auc"].min()),
        "positive_auc_folds": int((r["delta_auc"] > 0).sum()),
        "mean_delta_logloss": float(r["delta_logloss"].mean()),
        "min_delta_logloss": float(r["delta_logloss"].min()),
        "positive_logloss_folds": int((r["delta_logloss"] > 0).sum()),
        "mean_delta_top1": float(r["delta_top1"].mean()),
        "positive_top1_folds": int((r["delta_top1"] > 0).sum()),
    }


def stage_d_incremental(
    df: pd.DataFrame,
    baseline: list[str],
    stage_c: pd.DataFrame,
    target: str,
    folds: list[Fold],
    out: Path,
) -> pd.DataFrame:
    """
    Incremental OOS vs baseline, optimized without changing the statistic.

    The baseline prediction is identical for every candidate within a fold,
    so fit it once per fold and reuse it. Candidate models are still fitted
    independently and use the exact same training/test rows and metric logic.
    """
    candidates = stage_c["feature"].tolist()
    fold_cache = _cached_fold_indices(df, folds)
    by_candidate = {c: [] for c in candidates}

    for f in folds:
        tr, te = fold_cache[f.fold]
        trdf = df.iloc[tr]
        tedf = df.iloc[te]

        ytr0 = trdf[target]
        yte0 = tedf[target]
        good_tr = ytr0.notna().to_numpy()
        good_te = yte0.notna().to_numpy()

        trbase = trdf.loc[good_tr]
        tebase = tedf.loc[good_te]
        ytr = ytr0.loc[good_tr].astype(int)
        yte = yte0.loc[good_te].astype(int)

        if len(trbase) < 1000 or len(tebase) < 100:
            continue
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        p0 = fit_predict(
            make_baseline_model(),
            trbase[baseline],
            ytr,
            tebase[baseline],
        )
        auc0, ll0 = metric_pair(yte.to_numpy(), p0)
        hit0 = top1_hit_rate(tebase, yte.to_numpy(), p0)

        for i, c in enumerate(candidates, 1):
            if i % 25 == 0:
                log(f"Stage D {target}: fold {f.fold} candidate {i}/{len(candidates)}")
            try:
                cols1 = baseline + [c]
                p1 = fit_predict(
                    make_baseline_model(),
                    trbase[cols1],
                    ytr,
                    tebase[cols1],
                )
                auc1, ll1 = metric_pair(yte.to_numpy(), p1)
                hit1 = top1_hit_rate(tebase, yte.to_numpy(), p1)

                by_candidate[c].append(
                    {
                        "fold": f.fold,
                        "auc_base": auc0,
                        "auc_plus": auc1,
                        "delta_auc": auc1 - auc0,
                        "logloss_base": ll0,
                        "logloss_plus": ll1,
                        "delta_logloss": ll0 - ll1,
                        "top1_base": hit0,
                        "top1_plus": hit1,
                        "delta_top1": hit1 - hit0,
                        "n_test": len(tebase),
                    }
                )
            except Exception as e:
                # Keep candidate-local failure isolated; do not kill the run.
                by_candidate[c].append({"fold": f.fold, "error": repr(e)})

    rows = []
    for c in candidates:
        fr = pd.DataFrame(by_candidate[c])
        if fr.empty:
            rows.append({"feature": c, "folds": 0})
            continue
        good = fr.loc[fr["delta_auc"].notna()].copy()
        if good.empty:
            rows.append({"feature": c, "folds": 0})
            continue

        rows.append(
            {
                "feature": c,
                "folds": int(len(good)),
                "mean_delta_auc": float(good["delta_auc"].mean()),
                "median_delta_auc": float(good["delta_auc"].median()),
                "min_delta_auc": float(good["delta_auc"].min()),
                "positive_auc_folds": int((good["delta_auc"] > 0).sum()),
                "mean_delta_logloss": float(good["delta_logloss"].mean()),
                "min_delta_logloss": float(good["delta_logloss"].min()),
                "positive_logloss_folds": int((good["delta_logloss"] > 0).sum()),
                "mean_delta_top1": float(good["delta_top1"].mean()),
                "positive_top1_folds": int((good["delta_top1"] > 0).sum()),
            }
        )

    r = pd.DataFrame(rows)
    if r.empty:
        raise RuntimeError(f"Stage D produced no results for {target}.")

    r["pass_incremental"] = (
        (r["folds"] >= N_FOLDS)
        & (r["mean_delta_auc"] >= MIN_DELTA_AUC)
        & (r["mean_delta_logloss"] >= MIN_DELTA_LOGLOSS)
        & (r["positive_auc_folds"] >= math.ceil(N_FOLDS * 0.60))
        & (r["positive_logloss_folds"] >= math.ceil(N_FOLDS * 0.60))
    )
    r["incremental_score"] = (
        r["mean_delta_auc"].fillna(-np.inf)
        + 0.50 * r["mean_delta_logloss"].fillna(-np.inf)
        + 0.25 * r["mean_delta_top1"].fillna(-np.inf)
    )
    r = r.sort_values("incremental_score", ascending=False)
    atomic_to_csv(r, out / f"stageD_{safe_name(target)}.csv")
    return r


# ============================================================
# STAGE E — MULTIPLE-TESTING / EMPIRICAL NULL
# ============================================================

def empirical_max_null(
    df: pd.DataFrame,
    baseline: list[str],
    candidate_names: list[str],
    target: str,
    folds: list[Fold],
    permutations: int,
    out: Path,
) -> dict:
    """
    Memory-safe empirical max-null.

    The real target and baseline remain untouched. For every permutation, each
    candidate is independently shuffled across rows WITHIN each date. The
    baseline predictions are fitted once per fold and reused across every
    permutation/candidate. No full-panel copy is ever created.
    """
    if not candidate_names or permutations <= 0:
        return {"permutations": 0}

    null_path = out / f"null_{safe_name(target)}.json"
    max_scores = []

    # Resume only from completed permutations written atomically by the prior
    # iteration. A partially-computed permutation is intentionally discarded.
    if null_path.exists():
        try:
            prior = json.loads(null_path.read_text())
            prior_scores = prior.get("max_delta_auc", [])
            candidate_sig = hashlib.sha256(
                "|".join(candidate_names).encode()
            ).hexdigest()
            if (
                prior.get("version") == VERSION
                and prior.get("target") == target
                and prior.get("candidate_signature") == candidate_sig
                and prior.get("null_method") == "candidate_only_within_date"
                and isinstance(prior_scores, list)
                and len(prior_scores) <= permutations
            ):
                max_scores = [
                    float(x) if x is not None else np.nan
                    for x in prior_scores
                ]
                if max_scores:
                    log(
                        f"Resuming null {target}: "
                        f"{len(max_scores)}/{permutations} permutations already complete"
                    )
        except Exception:
            max_scores = []

    fold_cache = _cached_fold_indices(df, folds)

    # Precompute date groups once. This preserves exactly the same within-date
    # permutation scheme without repeatedly invoking groupby/transform.
    date_values = pd.to_datetime(df["_date"]).to_numpy()
    order = np.argsort(date_values, kind="mergesort")
    sorted_dates = date_values[order]
    boundaries = np.flatnonzero(sorted_dates[1:] != sorted_dates[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(order)]
    date_groups = [(order[s:e]) for s, e in zip(starts, ends)]

    # Baseline predictions are invariant across permutations.
    base_cache = {}
    for f in folds:
        tr, te = fold_cache[f.fold]
        trdf = df.iloc[tr]
        tedf = df.iloc[te]
        ytr0 = trdf[target]
        yte0 = tedf[target]
        good_tr = ytr0.notna().to_numpy()
        good_te = yte0.notna().to_numpy()
        trbase = trdf.loc[good_tr]
        tebase = tedf.loc[good_te]
        ytr = ytr0.loc[good_tr].astype(int)
        yte = yte0.loc[good_te].astype(int)

        if len(trbase) < 1000 or len(tebase) < 100 or ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        p0 = fit_predict(make_baseline_model(), trbase[baseline], ytr, tebase[baseline])
        auc0, ll0 = metric_pair(yte.to_numpy(), p0)
        base_cache[f.fold] = (tr, te, good_tr, good_te, ytr, yte, auc0, ll0)

    start_perm = len(max_scores)
    for pno in range(start_perm, permutations):
        log(f"Null {target}: permutation {pno + 1}/{permutations}")
        perm_rng = np.random.default_rng(SEED + 1000 + pno)

        best = -np.inf

        for ci, c in enumerate(candidate_names, 1):
            if ci % 25 == 0:
                log(f"Null {target}: candidate {ci}/{len(candidate_names)}")

            # One 1-D copy only; no DataFrame copy.
            shuffled_values = pd.to_numeric(
                df[c], errors="coerce"
            ).to_numpy(dtype=np.float64, copy=True)

            # Independent within-date permutation, preserving the candidate's
            # empirical cross-sectional distribution on each date.
            for idx in date_groups:
                if len(idx) > 1:
                    local = shuffled_values[idx].copy()
                    shuffled_values[idx] = local[perm_rng.permutation(idx.size)]

            score_rows = []
            for f in folds:
                cache = base_cache.get(f.fold)
                if cache is None:
                    continue
                tr, te, good_tr, good_te, ytr, yte, auc0, ll0 = cache

                # Pull only baseline + this candidate. Build a tiny frame rather
                # than copying the entire working panel.
                tr_pos = tr[good_tr]
                te_pos = te[good_te]
                xtr = df.iloc[tr_pos][baseline].copy()
                xte = df.iloc[te_pos][baseline].copy()
                xtr[c] = shuffled_values[tr_pos]
                xte[c] = shuffled_values[te_pos]

                try:
                    p1 = fit_predict(
                        make_baseline_model(),
                        xtr,
                        ytr,
                        xte,
                    )
                    auc1, ll1 = metric_pair(yte.to_numpy(), p1)
                    if pd.notna(auc1):
                        score_rows.append(float(auc1 - auc0))
                except Exception:
                    continue

            if score_rows:
                best = max(best, float(np.mean(score_rows)))

        max_scores.append(best)

        atomic_write_json(
            {
                "version": VERSION,
                "target": target,
                "candidate_signature": hashlib.sha256(
                    "|".join(candidate_names).encode()
                ).hexdigest(),
                "null_method": "candidate_only_within_date",
                "permutations_completed": pno + 1,
                "max_delta_auc": max_scores,
            },
            null_path,
        )

    arr = np.asarray([x for x in max_scores if np.isfinite(x)])
    if len(arr) == 0:
        return {"permutations": 0}

    return {
        "permutations": int(len(arr)),
        "max_null_mean_delta_auc_p95": float(np.quantile(arr, 0.95)),
        "max_null_mean_delta_auc_p99": float(np.quantile(arr, 0.99)),
        "max_null_mean_delta_auc_mean": float(np.mean(arr)),
    }


# ============================================================
# STAGE F — FINAL RESEARCH UNION + REGIME CONDITIONALITY
# ============================================================

def add_causal_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stock-specific, mutually-exclusive regimes using information available
    at date t only. Regimes are routing/context labels, not model features.

    20-session momentum is compared with the stock's own trailing distribution
    through t-1:
      upper third -> TREND_UP
      middle third -> RANGE
      lower third -> TREND_DOWN

    A causal volatility context flag is also added for research diagnostics.
    """
    # df is the working panel. Avoid duplicating the full multi-GB DataFrame.
    # Only two small regime columns are added below.
    x = df

    mom20 = (
        pd.to_numeric(x["close"], errors="coerce")
        .groupby(x["symbol"], sort=False)
        .pct_change(20)
    )

    q_low = (
        mom20.groupby(x["symbol"], sort=False)
        .transform(
            lambda s: s.shift(1).rolling(252, min_periods=60).quantile(1.0 / 3.0)
        )
    )
    q_high = (
        mom20.groupby(x["symbol"], sort=False)
        .transform(
            lambda s: s.shift(1).rolling(252, min_periods=60).quantile(2.0 / 3.0)
        )
    )

    regime = np.full(len(x), "UNKNOWN", dtype=object)
    valid = mom20.notna() & q_low.notna() & q_high.notna()
    regime[valid & (mom20 > q_high)] = "TREND_UP"
    regime[valid & (mom20 < q_low)] = "TREND_DOWN"
    regime[valid & ~(mom20 > q_high) & ~(mom20 < q_low)] = "RANGE"

    regime_block = {"REG_STOCK_TREND": pd.Series(regime, index=x.index)}

    atr = None
    for c in ("D_atr_pct", "atr_pct_14", "FD_atr_pct_14"):
        if c in x.columns:
            atr = pd.to_numeric(x[c], errors="coerce")
            break

    if atr is not None:
        atr_q = (
            atr.groupby(x["symbol"], sort=False)
            .transform(
                lambda s: s.shift(1).rolling(252, min_periods=60).quantile(2.0 / 3.0)
            )
        )
        regime_block["REG_STOCK_VOL_HIGH"] = (
            atr.notna() & atr_q.notna() & (atr > atr_q)
        ).astype("int8")
    else:
        regime_block["REG_STOCK_VOL_HIGH"] = pd.Series(np.int8(0), index=x.index)

    return pd.concat([x, pd.DataFrame(regime_block, index=x.index)], axis=1, copy=False)


def regime_incremental_oos(
    df: pd.DataFrame,
    baseline: list[str],
    candidate: str,
    target: str,
    folds: list[Fold],
    regime: str,
) -> dict:
    """Evaluate baseline + candidate only within one stock regime."""
    fold_rows = []

    for f in folds:
        tr, te = fold_indices(df, f)
        trdf = df.iloc[tr]
        tedf = df.iloc[te]

        trdf = trdf.loc[trdf["REG_STOCK_TREND"].eq(regime)]
        tedf = tedf.loc[tedf["REG_STOCK_TREND"].eq(regime)]

        ytr = trdf[target]
        yte = tedf[target]
        good_tr = ytr.notna()
        good_te = yte.notna()

        trdf = trdf.loc[good_tr]
        tedf = tedf.loc[good_te]
        ytr = ytr.loc[good_tr].astype(int)
        yte = yte.loc[good_te].astype(int)

        if len(trdf) < MIN_REGIME_TRAIN_ROWS or len(tedf) < MIN_REGIME_TEST_ROWS:
            continue
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        p0 = fit_predict(make_baseline_model(), trdf[baseline], ytr, tedf[baseline])
        p1 = fit_predict(
            make_baseline_model(),
            trdf[baseline + [candidate]],
            ytr,
            tedf[baseline + [candidate]],
        )

        auc0, ll0 = metric_pair(yte.to_numpy(), p0)
        auc1, ll1 = metric_pair(yte.to_numpy(), p1)

        fold_rows.append(
            {
                "fold": f.fold,
                "regime": regime,
                "auc_base": auc0,
                "auc_plus": auc1,
                "delta_auc": auc1 - auc0,
                "logloss_base": ll0,
                "logloss_plus": ll1,
                "delta_logloss": ll0 - ll1,
                "n_train": len(trdf),
                "n_test": len(tedf),
            }
        )

    if not fold_rows:
        return {"folds": 0}

    r = pd.DataFrame(fold_rows)
    return {
        "folds": int(len(r)),
        "mean_delta_auc": float(r["delta_auc"].mean()),
        "median_delta_auc": float(r["delta_auc"].median()),
        "min_delta_auc": float(r["delta_auc"].min()),
        "positive_auc_folds": int((r["delta_auc"] > 0).sum()),
        "mean_delta_logloss": float(r["delta_logloss"].mean()),
        "min_delta_logloss": float(r["delta_logloss"].min()),
        "positive_logloss_folds": int((r["delta_logloss"] > 0).sum()),
        "mean_train_rows": float(r["n_train"].mean()),
        "mean_test_rows": float(r["n_test"].mean()),
    }


def stage_regime_specialists(
    df: pd.DataFrame,
    baseline: list[str],
    stage_c: pd.DataFrame,
    target: str,
    folds: list[Fold],
    out: Path,
) -> pd.DataFrame:
    """
    Regime-specific incremental discovery, optimized by fitting each regime/fold
    baseline once. Candidate models remain one-per-candidate and are unchanged.
    """
    candidates = stage_c["feature"].tolist()
    fold_cache = _cached_fold_indices(df, folds)
    rows = []

    for regime in REGIME_NAMES:
        log(f"Regime discovery {target}: {regime} | {len(candidates)} candidates")
        by_candidate = {c: [] for c in candidates}

        for f in folds:
            tr, te = fold_cache[f.fold]
            trdf = df.iloc[tr]
            tedf = df.iloc[te]

            trdf = trdf.loc[trdf["REG_STOCK_TREND"].eq(regime)]
            tedf = tedf.loc[tedf["REG_STOCK_TREND"].eq(regime)]

            ytr = trdf[target]
            yte = tedf[target]
            good_tr = ytr.notna()
            good_te = yte.notna()
            trdf = trdf.loc[good_tr]
            tedf = tedf.loc[good_te]
            ytr = ytr.loc[good_tr].astype(int)
            yte = yte.loc[good_te].astype(int)

            if len(trdf) < MIN_REGIME_TRAIN_ROWS or len(tedf) < MIN_REGIME_TEST_ROWS:
                log(f"Regime {target}/{regime} fold {f.fold}: insufficient rows; skipped")
                continue
            if ytr.nunique() < 2 or yte.nunique() < 2:
                log(f"Regime {target}/{regime} fold {f.fold}: one-class target; skipped")
                continue

            # One baseline fit per regime/fold, reused for every candidate.
            p0 = fit_predict(
                make_baseline_model(),
                trdf[baseline],
                ytr,
                tedf[baseline],
            )
            auc0, ll0 = metric_pair(yte.to_numpy(), p0)

            for i, c in enumerate(candidates, 1):
                if i % 25 == 0:
                    log(
                        f"Regime {target}/{regime}: fold {f.fold} "
                        f"candidate {i}/{len(candidates)}"
                    )
                try:
                    p1 = fit_predict(
                        make_baseline_model(),
                        trdf[baseline + [c]],
                        ytr,
                        tedf[baseline + [c]],
                    )
                    auc1, ll1 = metric_pair(yte.to_numpy(), p1)
                    by_candidate[c].append(
                        {
                            "fold": f.fold,
                            "regime": regime,
                            "auc_base": auc0,
                            "auc_plus": auc1,
                            "delta_auc": auc1 - auc0,
                            "logloss_base": ll0,
                            "logloss_plus": ll1,
                            "delta_logloss": ll0 - ll1,
                            "n_train": len(trdf),
                            "n_test": len(tedf),
                        }
                    )
                except Exception as e:
                    by_candidate[c].append({"fold": f.fold, "error": repr(e)})

        for c in candidates:
            fr = pd.DataFrame(by_candidate[c])
            good = fr.loc[fr["delta_auc"].notna()].copy() if not fr.empty else pd.DataFrame()
            if good.empty:
                rows.append({"feature": c, "regime": regime, "folds": 0})
                continue
            rows.append(
                {
                    "feature": c,
                    "regime": regime,
                    "folds": int(len(good)),
                    "mean_delta_auc": float(good["delta_auc"].mean()),
                    "median_delta_auc": float(good["delta_auc"].median()),
                    "min_delta_auc": float(good["delta_auc"].min()),
                    "positive_auc_folds": int((good["delta_auc"] > 0).sum()),
                    "mean_delta_logloss": float(good["delta_logloss"].mean()),
                    "min_delta_logloss": float(good["delta_logloss"].min()),
                    "positive_logloss_folds": int((good["delta_logloss"] > 0).sum()),
                    "mean_train_rows": float(good["n_train"].mean()),
                    "mean_test_rows": float(good["n_test"].mean()),
                }
            )

    r = pd.DataFrame(rows)
    if r.empty:
        raise RuntimeError(f"Regime discovery produced no results for {target}.")

    r["pass_regime"] = (
        (r["folds"] >= N_FOLDS)
        & (r["mean_delta_auc"] >= MIN_DELTA_AUC)
        & (r["mean_delta_logloss"] >= MIN_DELTA_LOGLOSS)
        & (r["positive_auc_folds"] >= math.ceil(N_FOLDS * 0.60))
        & (r["positive_logloss_folds"] >= math.ceil(N_FOLDS * 0.60))
    )
    r["regime_score"] = (
        r["mean_delta_auc"].fillna(-np.inf)
        + 0.50 * r["mean_delta_logloss"].fillna(-np.inf)
    )

    selected_parts = []
    for regime in REGIME_NAMES:
        selected_parts.append(
            r.loc[(r["regime"] == regime) & r["pass_regime"]]
            .sort_values("regime_score", ascending=False)
            .head(MAX_REGIME_FEATURES)
            .copy()
        )

    selected = pd.concat(selected_parts, ignore_index=True)
    atomic_to_csv(r, out / f"stageG_regime_all_{safe_name(target)}.csv")
    atomic_to_csv(selected, out / f"stageG_regime_selected_{safe_name(target)}.csv")
    return selected


def final_research_union(
    stage_d: pd.DataFrame,
    max_features: int = MAX_FINAL_RESEARCH,
) -> pd.DataFrame:
    r = stage_d.copy()

    if "pass_incremental" not in r.columns:
        return r.iloc[0:0].copy()

    keep = r.loc[r["pass_incremental"]].copy()

    # Prefer features with stable incremental evidence. Do not use a ranking
    # as a trading recommendation; this only determines which experiments
    # enter the final research union.
    keep = keep.sort_values(
        ["positive_auc_folds", "positive_logloss_folds", "incremental_score"],
        ascending=[False, False, False],
    ).head(max_features)

    keep["research_status"] = "RESEARCH_ONLY"
    return keep


# ============================================================
# FINAL INTEGRITY GATE
# ============================================================

def final_integrity_gate(
    df: pd.DataFrame,
    target: str,
    baseline: list[str],
    final_features: list[str],
    folds: list[Fold],
    out: Path,
) -> dict:
    errors = []

    if df["_row_id"].duplicated().any():
        errors.append("duplicate_row_id")

    if df[["symbol", "timestamp"]].duplicated().any():
        errors.append("duplicate_symbol_timestamp")

    for c in baseline + final_features:
        if c not in df.columns:
            errors.append(f"missing_feature:{c}")
        if suspicious(c):
            errors.append(f"future_named_feature:{c}")

    overlap = set(baseline) & set(final_features)
    if overlap:
        # Not an error: final features may include an existing feature in some
        # runs. We report it rather than silently removing it.
        pass

    for f in folds:
        tr, te = fold_indices(df, f)
        tr_dates = set(df.iloc[tr]["_date"].unique())
        te_dates = set(df.iloc[te]["_date"].unique())
        if tr_dates & te_dates:
            errors.append(f"fold_overlap:{f.fold}")

        if max(pd.to_datetime(tr_dates)) >= min(pd.to_datetime(te_dates)):
            errors.append(f"train_not_before_test:{f.fold}")

    # Candidate map must not contain suspicious model inputs.
    cand_files = list(out.glob("stageD_*.csv"))
    for p in cand_files:
        try:
            z = pd.read_csv(p)
            if "feature" in z:
                bad = [c for c in z["feature"].astype(str) if suspicious(c)]
                if bad:
                    errors.append(f"suspicious_candidates:{p.name}:{len(bad)}")
        except Exception as e:
            errors.append(f"cannot_read:{p.name}:{e!r}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "baseline_features": baseline,
        "final_features": final_features,
        "targets": [target],
        "folds": [asdict(x) for x in folds],
        "version": VERSION,
    }

    atomic_write_json(result, out / f"integrity_{safe_name(target)}.json")

    if errors:
        raise RuntimeError(f"FINAL INTEGRITY GATE FAILED: {errors}")

    return result


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel.parquet")
    ap.add_argument("--out", default="feature_discovery_v3")
    ap.add_argument("--workers", type=int, default=1,
                    help="Reserved for future parallel stage implementation. "
                         "Default 1 is deterministic.")
    ap.add_argument("--permutations", type=int, default=20,
                    help="Empirical max-null permutations on final shortlist.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run a real-panel preflight only: load, leak gate, feature construction, "
             "sector-relative construction, fold/purge validation, and a small Stage-A sample. "
             "Does not run OOS models or permutations."
    )
    return ap.parse_args()


def smoke_test(panel_path: Path, out: Path) -> None:
    """Real-panel preflight. Designed to fail fast before an overnight run."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    log("=" * 72)
    log("SMOKE TEST — REAL PANEL ONLY")
    log("=" * 72)

    if not panel_path.exists():
        raise RuntimeError(f"Panel not found: {panel_path}")

    log(f"Panel: {panel_path}")
    log(f"XGBoost version: {xgb.__version__} | n_jobs={XGB_N_JOBS}")
    log(f"Panel size: {panel_path.stat().st_size / (1024**2):.1f} MB")

    df = load_panel(panel_path)
    memory_log(df, "after_load")
    gate = static_leak_gate(df)
    atomic_write_json(gate, out / "smoke_static_leak_gate.json")

    log(
        f"Loaded {len(df):,} rows | "
        f"{df.symbol.nunique():,} symbols | "
        f"{df._date.nunique():,} dates"
    )
    log(
        f"Date range: {df._date.min().date()} -> {df._date.max().date()}"
    )

    # Hard checks before any expensive work.
    if not df["_row_id"].is_unique:
        raise RuntimeError("SMOKE FAIL: _row_id is not unique.")
    if df[["symbol", "timestamp"]].duplicated().any():
        raise RuntimeError("SMOKE FAIL: duplicate symbol/timestamp rows.")

    targets_present = [t for t in DEFAULT_TARGETS if t in df.columns]
    if not targets_present:
        raise RuntimeError(
            f"SMOKE FAIL: none of the expected targets exist. "
            f"Expected one of {DEFAULT_TARGETS}"
        )
    log(f"Targets present: {targets_present}")

    work, inv = add_derived_features(df)
    if not np.array_equal(df["_row_id"].to_numpy(), work["_row_id"].to_numpy()):
        raise RuntimeError("SMOKE FAIL: row alignment changed after derived features.")
    log(f"Derived features created: {len(inv)}")
    memory_log(work, "after_derived")

    approved = find_approved_features(panel_path)
    work, zinv = add_existing_feature_zscores(work, approved, max_features=100)
    if not np.array_equal(df["_row_id"].to_numpy(), work["_row_id"].to_numpy()):
        raise RuntimeError("SMOKE FAIL: row alignment changed after existing-feature normalization.")
    log(f"Existing-feature normalized candidates: {len(zinv)}")
    memory_log(work, "after_existing_zscores")

    work, peinv = add_existing_panel_feature_expansions(
        df, work, exclude_features=approved
    )
    if not np.array_equal(df["_row_id"].to_numpy(), work["_row_id"].to_numpy()):
        raise RuntimeError("SMOKE FAIL: row alignment changed after existing-panel feature expansion.")
    log(f"Existing-panel feature expansion candidates: {len(peinv):,}")
    memory_log(work, "after_panel_expansion")

    sector_map = {
        "AUTO": "NIFTYAUTO",
        "FMCG": "NIFTYFMCG",
        "IT": "NIFTYIT",
        "METAL": "NIFTYMETAL",
        "PHARMA": "NIFTYPHARMA",
        "REALTY": "NIFTYREALTY",
        "ENERGY": "NIFTYENERGY",
        "PSUBANK": "NIFTYPSUBANK",
        "FINSERVICE": "NIFTYFINSERVICE",
        "MEDIA": "NIFTYMEDIA",
        "CONSUMPTION": "NIFTYCONSUMPTION",
        "INFRA": "NIFTYINFRA",
    }
    work, sinv = add_sector_relative_features(work, sector_map)
    if not np.array_equal(df["_row_id"].to_numpy(), work["_row_id"].to_numpy()):
        raise RuntimeError("SMOKE FAIL: row alignment changed after sector-relative features.")
    log(f"Sector-relative candidates created: {len(sinv)}")
    memory_log(work, "after_sector_relative")

    work = add_causal_regimes(work)
    if not np.array_equal(df["_row_id"].to_numpy(), work["_row_id"].to_numpy()):
        raise RuntimeError("SMOKE FAIL: row alignment changed after stock-regime construction.")
    regime_counts = work["REG_STOCK_TREND"].value_counts(dropna=False).to_dict()
    log(f"Stock regimes created: {regime_counts}")
    memory_log(work, "after_regimes")

    folds = build_folds(work)
    fold_audit = validate_fold_purge(work, folds)
    atomic_to_csv(fold_audit, out / "smoke_fold_schedule.csv")
    log("Walk-forward/purge validation: PASS")

    # Tiny real Stage-A sample: enough to prove the candidate pipeline works,
    # but deliberately not enough to start the expensive discovery.
    all_meta = inv + zinv + peinv + sinv
    safe_meta = [
        m for m in all_meta
        if m["feature"] in work.columns and not suspicious(m["feature"])
    ]
    candidate_df = pd.DataFrame(safe_meta).drop_duplicates("feature")

    target = targets_present[0]
    sample_meta = candidate_df.head(min(25, len(candidate_df))).to_dict("records")
    stage_a = stage_a_screen(work, sample_meta, target, out)
    log(f"Stage-A smoke sample: {len(stage_a)} candidates survived.")

    # Save a small working artifact so the user can inspect it.
    work.head(1000).to_parquet(out / "smoke_working_sample.parquet", index=False)
    atomic_write_json(
        {
            "status": "SMOKE_PASS",
            "version": VERSION,
            "panel": str(panel_path),
            "rows": len(df),
            "symbols": int(df.symbol.nunique()),
            "dates": int(df._date.nunique()),
            "targets": targets_present,
            "derived_features": len(inv),
            "existing_feature_zscores": len(zinv),
            "existing_panel_feature_expansions": len(peinv),
            "sector_relative_features": len(sinv),
            "stock_regimes": {
                str(k): int(v)
                for k, v in work["REG_STOCK_TREND"].value_counts(dropna=False).items()
            },
            "stage_a_survivors": len(stage_a),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        out / "SMOKE_PASS.json",
    )

    log("=" * 72)
    log("SMOKE TEST PASSED")
    log("Safe to start the long discovery run.")
    log("=" * 72)


def main():
    args = parse_args()

    panel_path = Path(args.panel).resolve()
    out = Path(args.out).resolve()

    if args.smoke:
        smoke_test(panel_path, out / "smoke_test")
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    manifest = {
        "version": VERSION,
        "seed": SEED,
        "panel": str(panel_path),
        "panel_sha256": sha256_file(panel_path),
        "n_folds": N_FOLDS,
        "label_horizon": LABEL_HORIZON,
        "embargo": EMBARGO,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(manifest, out / "run_manifest.json")

    log(f"VERSION={VERSION}")
    log(f"XGBoost version={xgb.__version__} | n_jobs={XGB_N_JOBS}")
    log(f"Output={out}")

    df = load_panel(panel_path)
    memory_log(df, "after_load")
    gate = static_leak_gate(df)
    atomic_write_json(gate, out / "static_leak_gate.json")

    log(
        f"Panel rows={len(df):,} symbols={df.symbol.nunique():,} "
        f"dates={df._date.nunique():,}"
    )

    # --------------------------------------------------------
    # Derived candidates
    # --------------------------------------------------------
    sig = hashlib.sha256(
        (VERSION + str(len(df)) + str(df.columns.tolist())).encode()
    ).hexdigest()

    key = "feature_construction"
    if stage_done(out, key, sig) and not args.force:
        log("Feature construction checkpoint found; loading cached working panel.")
        working_path = out / "working_panel.parquet"
        if not working_path.exists():
            raise RuntimeError("Checkpoint exists but working_panel.parquet is missing.")
        work = pd.read_parquet(working_path)
        candidates = json.loads((out / "candidate_inventory.json").read_text())
    else:
        stage_start(out, key, sig)

        work, candidates = add_derived_features(df)

        approved = find_approved_features(panel_path)
        work, zc = add_existing_feature_zscores(work, approved, max_features=100)
        candidates.extend(zc)

        # FEATURE ADDITIONS ONLY: systematically expand the numeric features
        # that were already present in the original panel. This does not
        # change any existing model/stage logic.
        work, pec = add_existing_panel_feature_expansions(
            df, work, exclude_features=approved
        )
        candidates.extend(pec)
        log(f"Existing-panel feature expansion candidates: {len(pec):,}")
        memory_log(work, "after_panel_expansion")

        # Existing sector-index columns. Add/modify this mapping only when the
        # exact index columns are actually present in panel.parquet.
        sector_map = {
            "AUTO": "NIFTYAUTO",
            "FMCG": "NIFTYFMCG",
            "IT": "NIFTYIT",
            "METAL": "NIFTYMETAL",
            "PHARMA": "NIFTYPHARMA",
            "REALTY": "NIFTYREALTY",
            "ENERGY": "NIFTYENERGY",
            "PSUBANK": "NIFTYPSUBANK",
            "FINSERVICE": "NIFTYFINSERVICE",
            "MEDIA": "NIFTYMEDIA",
            "CONSUMPTION": "NIFTYCONSUMPTION",
            "INFRA": "NIFTYINFRA",
        }
        work, sc = add_sector_relative_features(work, sector_map)
        candidates.extend(sc)
        memory_log(work, "after_sector_relative")

        # Additive stock-regime context for regime-specific discovery.
        work = add_causal_regimes(work)
        memory_log(work, "after_regimes")

        # Final row-alignment gate.
        if not np.array_equal(
            work["_row_id"].to_numpy(),
            df["_row_id"].to_numpy(),
        ):
            raise RuntimeError("FATAL: working panel row alignment changed.")

        # Remove accidental temporary / future columns from candidate inventory.
        candidates = [
            m for m in candidates
            if m["feature"] in work.columns and not suspicious(m["feature"])
        ]

        # Deduplicate inventory by feature.
        seen = set()
        clean = []
        for m in candidates:
            if m["feature"] not in seen:
                clean.append(m)
                seen.add(m["feature"])
        candidates = clean

        work.to_parquet(out / "working_panel.parquet", index=False)
        # Fold-index cache is tied to this exact in-memory working DataFrame.
        _FOLD_INDEX_CACHE.pop(id(work), None)
        atomic_write_json(candidates, out / "candidate_inventory.json")

        mark_stage(
            out,
            key,
            sig,
            {"candidate_count": len(candidates)},
        )

    log(f"Candidate inventory: {len(candidates):,}")

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------
    folds = build_folds(work)
    fold_df = validate_fold_purge(work, folds)
    atomic_to_csv(fold_df, out / "fold_schedule.csv")

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------
    approved = find_approved_features(panel_path)
    baseline = choose_baseline_features(work, approved)
    atomic_write_json({"features": baseline}, out / "baseline_features.json")
    log(f"Baseline features: {baseline}")

    targets = [t for t in DEFAULT_TARGETS if t in work.columns]
    if not targets:
        raise RuntimeError("None of the configured targets exist in panel.")

    all_final = {}

    for target in targets:
        log("=" * 72)
        log(f"TARGET: {target}")
        log("=" * 72)

        # Only candidates that are present and not target-derived.
        candidate_df = pd.DataFrame(candidates)
        candidate_df = candidate_df[
            candidate_df["feature"].isin(
                [c for c in work.columns if is_numeric_candidate(c, work)]
            )
        ].copy()

        # --------------------------------------------
        # Stage A
        # --------------------------------------------
        sig_a = hashlib.sha256(
            (sig + target + "A").encode()
        ).hexdigest()
        key_a = f"{target}_stageA"

        if stage_done(out, key_a, sig_a) and not args.force:
            stage_a = pd.read_csv(out / f"stageA_{safe_name(target)}.csv")
        else:
            stage_start(out, key_a, sig_a)
            stage_a = stage_a_screen(work, candidate_df.to_dict("records"), target, out)
            mark_stage(out, key_a, sig_a, {"rows": len(stage_a)})

        # --------------------------------------------
        # Stage B
        # --------------------------------------------
        sig_b = hashlib.sha256(
            (sig_a + str(stage_a["feature"].tolist())).encode()
        ).hexdigest()
        key_b = f"{target}_stageB"

        if stage_done(out, key_b, sig_b) and not args.force:
            stage_b = pd.read_csv(out / f"stageB_{safe_name(target)}.csv")
        else:
            stage_start(out, key_b, sig_b)
            stage_b = redundancy_filter(
                work, stage_a, baseline, out, target
            )
            mark_stage(out, key_b, sig_b, {"rows": len(stage_b)})

        # --------------------------------------------
        # Stage C
        # --------------------------------------------
        sig_c = hashlib.sha256(
            (sig_b + str(stage_b["feature"].tolist())).encode()
        ).hexdigest()
        key_c = f"{target}_stageC"

        if stage_done(out, key_c, sig_c) and not args.force:
            stage_c = pd.read_csv(out / f"stageC_{safe_name(target)}.csv")
        else:
            stage_start(out, key_c, sig_c)
            stage_c = stage_c_screen(work, stage_b, target, folds, out)
            mark_stage(out, key_c, sig_c, {"rows": len(stage_c)})

        # --------------------------------------------
        # Stage G — REGIME-SPECIFIC FEATURE DISCOVERY
        # --------------------------------------------
        sig_g = hashlib.sha256(
            (sig_c + str(stage_c["feature"].tolist()) + str(baseline) + "REGIME").encode()
        ).hexdigest()
        key_g = f"{target}_stageG_regime"

        if stage_done(out, key_g, sig_g) and not args.force:
            regime_selected = pd.read_csv(
                out / f"stageG_regime_selected_{safe_name(target)}.csv"
            )
        else:
            stage_start(out, key_g, sig_g)
            regime_selected = stage_regime_specialists(
                work, baseline, stage_c, target, folds, out
            )
            mark_stage(out, key_g, sig_g, {"rows": len(regime_selected)})

        regime_map = {
            regime: regime_selected.loc[
                regime_selected["regime"].eq(regime), "feature"
            ].tolist()
            for regime in REGIME_NAMES
        }
        atomic_write_json(
            {
                "target": target,
                "baseline_features": baseline,
                "regime_feature_map": regime_map,
                "max_features_per_regime": MAX_REGIME_FEATURES,
                "status": "RESEARCH_ONLY",
            },
            out / f"REGIME_FEATURE_MAP_{safe_name(target)}.json",
        )

        # --------------------------------------------
        # Stage D
        # --------------------------------------------
        sig_d = hashlib.sha256(
            (sig_c + str(stage_c["feature"].tolist()) + str(baseline)).encode()
        ).hexdigest()
        key_d = f"{target}_stageD"

        if stage_done(out, key_d, sig_d) and not args.force:
            stage_d = pd.read_csv(out / f"stageD_{safe_name(target)}.csv")
        else:
            stage_start(out, key_d, sig_d)
            stage_d = stage_d_incremental(
                work, baseline, stage_c, target, folds, out
            )
            mark_stage(out, key_d, sig_d, {"rows": len(stage_d)})

        # --------------------------------------------
        # Stage E empirical max-null
        # --------------------------------------------
        final_shortlist = (
            stage_d.loc[
                stage_d["pass_incremental"].fillna(False)
            ]
            .sort_values("incremental_score", ascending=False)
            .head(25)["feature"]
            .tolist()
        )

        null_result = {}
        if final_shortlist and args.permutations > 0:
            sig_e = hashlib.sha256(
                (sig_d + str(final_shortlist) + str(args.permutations)).encode()
            ).hexdigest()
            key_e = f"{target}_stageE"

            if stage_done(out, key_e, sig_e) and not args.force:
                null_result = json.loads(
                    (out / f"null_summary_{safe_name(target)}.json").read_text()
                )
            else:
                stage_start(out, key_e, sig_e)
                null_result = empirical_max_null(
                    work,
                    baseline,
                    final_shortlist,
                    target,
                    folds,
                    args.permutations,
                    out,
                )
                atomic_write_json(
                    null_result,
                    out / f"null_summary_{safe_name(target)}.json",
                )
                mark_stage(out, key_e, sig_e, null_result)

        # --------------------------------------------
        # Stage F
        # --------------------------------------------
        final_r = final_research_union(stage_d)
        final_features = final_r["feature"].tolist()

        # If the empirical null is available, report the threshold alongside
        # the research set. We do NOT silently delete features based on it:
        # the null is a discovery-bias diagnostic, not a substitute for the
        # final economic tournament.
        if null_result:
            threshold = null_result.get("max_null_mean_delta_auc_p95", np.nan)
            final_r["empirical_null_p95_delta_auc"] = threshold
            final_r["above_empirical_null_p95"] = (
                final_r["mean_delta_auc"] > threshold
            )

        atomic_to_csv(
            final_r,
            out / f"FINAL_RESEARCH_UNION_{safe_name(target)}.csv",
        )

        if not regime_selected.empty:
            regime_research = regime_selected.copy()
            regime_research["research_role"] = "REGIME_SPECIALIST"
            regime_research.to_csv(
                out / f"FINAL_REGIME_SPECIALISTS_{safe_name(target)}.csv",
                index=False,
            )

        atomic_write_json(
            {
                "target": target,
                "baseline_features": baseline,
                "research_features": final_features,
                "count": len(final_features),
                "regime_specialist_features": {
                    regime: regime_map.get(regime, [])
                    for regime in REGIME_NAMES
                },
                "regime_specialist_counts": {
                    regime: len(regime_map.get(regime, []))
                    for regime in REGIME_NAMES
                },
                "null_summary": null_result,
                "status": "RESEARCH_ONLY",
            },
            out / f"deployment_map_{safe_name(target)}.json",
        )

        # --------------------------------------------
        # Final integrity gate
        # --------------------------------------------
        final_integrity_gate(
            work,
            target,
            baseline,
            final_features,
            folds,
            out,
        )

        all_final[target] = final_features

    # --------------------------------------------------------
    # Global completion gate
    # --------------------------------------------------------
    expected = []
    for t in targets:
        for s in ("stageA", "stageB", "stageC", "stageD"):
            expected.append(f"{t}_{s}")

    missing = []
    for key in expected:
        p = out / "checkpoints" / f"{safe_name(key)}.json"
        if not p.exists():
            missing.append(key)

    if missing:
        raise RuntimeError(f"GLOBAL COMPLETION GATE FAILED. Missing: {missing}")

    atomic_write_json(
        {
            "status": "COMPLETE",
            "version": VERSION,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "targets": targets,
            "baseline_features": baseline,
            "research_union": all_final,
            "warning": (
                "Research only. Do not replace the live feature set from this "
                "file without running the existing model tournament/backtest."
            ),
        },
        out / "RUN_COMPLETE.json",
    )

    log("=" * 72)
    log("FEATURE DISCOVERY COMPLETE")
    log(f"Results: {out}")
    log("Live panel/approved_features were NOT modified.")
    log("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted. Resume using the same command; completed stages are checkpointed.")
        sys.exit(130)
    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
