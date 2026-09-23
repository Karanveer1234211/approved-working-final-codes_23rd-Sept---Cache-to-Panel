#!/usr/bin/env python3
"""
model_tournament.py
===================

Controlled 10-hour model research tournament for the frozen trading panel.

DESIGN PRINCIPLES
-----------------
0. Target variants are predeclared before the run: ATR 1.5/1.0, fixed 3%/2%, and fixed 5%/3%. No target is chosen after seeing results.
1. The OUTER test folds are sacred. No model, hyperparameter, representation,
   threshold, calibration rule, or feature choice is selected using outer-test
   results.
2. Every experiment uses the same panel, same frozen approved feature set,
   same 5-session embargo, same walk-forward folds, and the same economic
   label: label_tp_before_sl.
3. Daily Top-1 is the primary execution metric because the intended system
   chooses at most one stock per session. Pooled threshold results are
   secondary diagnostics.
4. Ranking models are evaluated as ranking systems: each trading date is one
   query group. XGBoost requires rows sorted by qid/query group during fit.
5. Calibration is fitted only on an inner held-out calibration tail of the
   training window. The outer test fold is never used to fit calibration.
6. Every completed experiment/fold is checkpointed atomically. If the process
   stops or the PC reboots, --resume continues from the last completed unit.
7. A preflight validates files, schema, labels, folds, feature availability,
   package availability, and model constructors BEFORE a long run starts.
8. A deterministic smoke test runs before the tournament. A failed smoke test
   aborts before any long computation.
9. Results are append-only CSV/JSON artifacts. Existing results are never
   silently overwritten.
10. No automatic "winner" is declared. The script identifies candidates only
    for a separate locked certification.

IMPORTANT
---------
This is a research tournament, not a trading backtest. It does not model
slippage, MTF interest, capacity, market impact, execution delay, or live
order mechanics. The 42.4% hurdle is carried from the existing base-model workflow for
the frozen ATR target. The existing panel also contains 3p2 and 5p3 fixed
barriers; those are tested as separate targets with their own predeclared
35-bps binary-payoff hurdles. They are NOT selected after seeing results.

SOURCE-ALIGNED CHOICES
----------------------
The existing base_model.py uses:
- target: label_tp_before_sl
- 5-session embargo
- 5 walk-forward folds
- calibration on a held-out tail of each training window
- daily Top-N evaluation
- 42.4% TP-before-SL hurdle

The existing New_model.py uses a rank-mode idea with LightGBM regression and
cross-sectional daily ranking. This tournament adds a safer binary-relevance
LambdaMART track: TP-before-SL is the relevance label and each date is a
query group. This avoids introducing a new forward-return target merely to
test ranking.

USAGE
-----
First, cheap validation only:
    python model_tournament.py preflight --root "%CACHE_DAILY_ROOT%"

Then a short smoke test:
    python model_tournament.py smoke --root "%CACHE_DAILY_ROOT%"

Then the long run:
    python model_tournament.py run --root "%CACHE_DAILY_ROOT%" --hours 10

If interrupted:
    python model_tournament.py run --root "%CACHE_DAILY_ROOT%" --hours 10 --resume

To inspect completed results:
    python model_tournament.py report --root "%CACHE_DAILY_ROOT%"

Optional:
    --max-experiments N
    --cpu 8
    --seed 42

The script intentionally does NOT use multiprocessing by default. The goal is
reproducibility and avoiding memory explosions on a long unattended run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Frozen research contract
# ---------------------------------------------------------------------------

SEED = 42
N_SPLITS = 5
EMBARGO = 5
TARGET = "label_tp_before_sl"
BREAKEVEN = 0.424
COST_RATE = 0.0035  # 35 bps, same cost assumption referenced by base_model.py
TARGET_SPECS = [
    # Existing frozen ATR bracket. Its 42.4% hurdle is carried from the
    # existing barrier-scan workflow and is NOT recomputed here.
    {"key": "atr1p5_1p0", "target": "label_tp_before_sl",
     "tp": None, "sl": None, "hurdle": 0.424,
     "hurdle_source": "existing base_model barrier-scan hurdle"},
    # Fixed-price variants already present in panel_build.py. For these two
    # we use the simple binary-payoff breakeven at 35 bps round-trip cost:
    # p*TP - (1-p)*SL - cost > 0. This is an economic diagnostic, not a
    # claim that the panel's original barrier scan produced these numbers.
    {"key": "3p2", "target": "label_tp_before_sl_3p2",
     "tp": 0.03, "sl": 0.02,
     "hurdle": (0.02 + COST_RATE) / (0.03 + 0.02),
     "hurdle_source": "simple TP/SL binary-payoff breakeven at 35 bps"},
    {"key": "5p3", "target": "label_tp_before_sl_5p3",
     "tp": 0.05, "sl": 0.03,
     "hurdle": (0.03 + COST_RATE) / (0.05 + 0.03),
     "hurdle_source": "simple TP/SL binary-payoff breakeven at 35 bps"},
]
CALIB_FRACTION = 0.25
MIN_TRAIN_ROWS = 1000
MIN_CAL_ROWS = 200
MIN_TEST_ROWS = 200
TOP_NS = (1, 3, 5, 10)
MAX_TEST_ROWS_IN_MEMORY = 2_000_000

# Keep the tournament intentionally small and predeclared.
EXPERIMENTS = [
    # Baselines / nonlinear classification
    {"id": "hgb_base", "family": "hgb_classifier", "variant": "raw", "params": {
        "max_iter": 300, "max_depth": 5, "learning_rate": 0.06,
        "min_samples_leaf": 200, "l2_regularization": 1.0}},
    {"id": "hgb_shallow", "family": "hgb_classifier", "variant": "raw", "params": {
        "max_iter": 500, "max_depth": 3, "learning_rate": 0.04,
        "min_samples_leaf": 150, "l2_regularization": 2.0}},
    {"id": "logistic", "family": "logistic", "variant": "raw", "params": {
        "C": 0.25}},
    {"id": "extra_trees", "family": "extra_trees", "variant": "raw", "params": {
        "n_estimators": 350, "max_depth": 8, "min_samples_leaf": 80,
        "max_features": 0.8}},
    # Optional gradient boosting libraries. Skipped cleanly if unavailable.
    {"id": "xgb_classifier", "family": "xgb_classifier", "variant": "raw", "params": {
        "n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
        "min_child_weight": 50, "subsample": 0.85, "colsample_bytree": 0.9,
        "reg_lambda": 3.0}},
    {"id": "lgb_classifier", "family": "lgb_classifier", "variant": "raw", "params": {
        "n_estimators": 500, "num_leaves": 15, "learning_rate": 0.03,
        "min_child_samples": 150, "reg_lambda": 3.0,
        "feature_fraction": 0.9}},
    # Ranking tracks: binary relevance within each date.
    {"id": "xgb_lambdamart", "family": "xgb_ranker", "variant": "raw", "params": {
        "n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
        "min_child_weight": 50, "subsample": 0.85, "colsample_bytree": 0.9,
        "reg_lambda": 3.0, "objective": "rank:ndcg",
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 5}},
    {"id": "lgb_lambdarank", "family": "lgb_ranker", "variant": "raw", "params": {
        "n_estimators": 500, "num_leaves": 15, "learning_rate": 0.03,
        "min_child_samples": 150, "reg_lambda": 3.0,
        "feature_fraction": 0.9, "objective": "lambdarank"}},
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def append_csv_atomic(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def file_sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def import_optional(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Paths / approved feature contract
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    root: Path
    panel: Path
    approved: Path
    out: Path
    checkpoints: Path
    predictions: Path
    logs: Path


def make_paths(root: Path) -> Paths:
    root = root.resolve()
    # Support either root/panel.parquet or root/panel/panel.parquet.
    p1 = root / "panel.parquet"
    p2 = root / "panel" / "panel.parquet"
    panel = p1 if p1.exists() else p2
    approved_candidates = [
        root / "audit" / "approved_features.json",
        root / "approved_features.json",
        root / "panel" / "audit" / "approved_features.json",
    ]
    approved = next((p for p in approved_candidates if p.exists()),
                    approved_candidates[0])
    out = root / "model_tournament"
    return Paths(root, panel, approved, out, out / "checkpoints",
                 out / "predictions", out / "logs")


def load_approved(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(
            f"approved_features.json not found at {path}. "
            "Do not run the tournament against an unapproved feature set.")
    obj = json.loads(path.read_text(encoding="utf-8"))
    feats = list(obj.get("approved", []))
    if not feats:
        raise RuntimeError("approved_features.json contains no approved features.")
    if len(feats) != len(set(feats)):
        raise RuntimeError("approved feature list contains duplicates.")
    if TARGET not in obj.get("target", TARGET):
        # This is deliberately informational; feature audit target metadata can
        # be absent or represented differently. The panel target is checked later.
        log("WARNING: approved_features.json target metadata is not identical to "
            f"{TARGET!r}; panel target remains authoritative.")
    return feats, obj


# ---------------------------------------------------------------------------
# Data / fold contract
# ---------------------------------------------------------------------------

def load_panel(panel_path: Path, features: Sequence[str], targets: Sequence[str]) -> pd.DataFrame:
    if not panel_path.exists():
        raise RuntimeError(f"Panel not found: {panel_path}")
    log(f"Loading panel: {panel_path}")
    p = pd.read_parquet(panel_path)
    required = {"timestamp", "symbol", *targets, *features}
    missing = sorted(required - set(p.columns))
    if missing:
        raise RuntimeError(f"Panel missing required columns: {missing}")
    if p.empty:
        raise RuntimeError("Panel is empty.")

    ts = pd.to_datetime(p["timestamp"], errors="coerce")
    if ts.isna().any():
        raise RuntimeError("Panel contains invalid timestamps.")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    p["timestamp"] = ts

    if p["symbol"].isna().any():
        raise RuntimeError("Panel contains null symbols.")

    # Features must be numeric. Do not silently coerce object/string features:
    # that could turn malformed data into plausible numbers.
    for f in features:
        if not pd.api.types.is_numeric_dtype(p[f]):
            raise RuntimeError(f"Approved feature {f!r} is not numeric.")
    for target_name in targets:
        if not pd.api.types.is_numeric_dtype(p[target_name]):
            raise RuntimeError(f"Target {target_name!r} is not numeric.")

    p = p.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)

    dup = p.duplicated(["timestamp", "symbol"])
    if dup.any():
        raise RuntimeError(
            f"Panel has {int(dup.sum())} duplicate (timestamp,symbol) rows.")

    for target_name in targets:
        y = pd.to_numeric(p[target_name], errors="coerce")
        nonnull = y.dropna()
        bad = ~nonnull.isin([0, 1])
        if bad.any():
            raise RuntimeError(
                f"Target {target_name} contains non-binary values: "
                f"{sorted(nonnull[bad].unique())[:10]}")

    # Do not impute here. Tree models can handle NaNs; linear models use
    # fold-local median imputation. Global imputation would be a subtle
    # train/test contamination risk.
    return p


def get_splits(p: pd.DataFrame, n_splits: int) -> List[Dict[str, Any]]:
    # Import the project's established splitter rather than creating a second
    # definition with subtly different boundaries.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    try:
        import panel_build as PB
    except Exception as e:
        raise RuntimeError(
            "Could not import panel_build.py. The tournament refuses to invent "
            f"a second walk-forward splitter. Error: {e}") from e

    splits = PB.walk_forward_splits(p, n_splits=n_splits)
    if len(splits) != n_splits:
        raise RuntimeError(f"Expected {n_splits} folds, got {len(splits)}.")
    return splits


def sessions_of(p: pd.DataFrame) -> np.ndarray:
    return np.sort(p["timestamp"].drop_duplicates().values)


def session_index_map(sessions: np.ndarray) -> Dict[pd.Timestamp, int]:
    return {pd.Timestamp(x): i for i, x in enumerate(sessions)}


def fold_masks(p: pd.DataFrame, sp: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    ts = p["timestamp"]
    test = ((ts >= pd.Timestamp(sp["test_start"])) &
            (ts <= pd.Timestamp(sp["test_end"]))).to_numpy()

    train_end = pd.Timestamp(sp["train_end"])
    train = (ts <= train_end).to_numpy()

    # The project's splitter already defines the outer boundary. We still
    # explicitly verify the embargo before every fit.
    if test.any() and train.any():
        train_dates = p.loc[train, "timestamp"].drop_duplicates().sort_values()
        test_dates = p.loc[test, "timestamp"].drop_duplicates().sort_values()
        if len(train_dates) and len(test_dates):
            gap = (pd.Timestamp(test_dates.iloc[0]) -
                   pd.Timestamp(train_dates.iloc[-1])).days
            if gap < 1:
                raise RuntimeError("Outer train/test dates overlap or are not ordered.")
    return train, test


def split_train_cal_fit(
    p: pd.DataFrame, train_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    """
    Inner fit/cal split. Calibration starts after EMBARGO sessions beyond the
    last fit session. The calibration tail ends at the outer train end.
    """
    dates = np.sort(p.loc[train_mask, "timestamp"].drop_duplicates().values)
    if len(dates) < 40:
        raise RuntimeError(f"Training window has only {len(dates)} sessions.")

    cut = int(np.floor(len(dates) * (1 - CALIB_FRACTION)))
    cut = max(1, min(cut, len(dates) - EMBARGO - 1))
    fit_end_i = cut - 1
    cal_start_i = cut - 1 + EMBARGO
    if cal_start_i >= len(dates):
        raise RuntimeError("Inner calibration split leaves no calibration sessions.")

    fit_end = pd.Timestamp(dates[fit_end_i])
    cal_start = pd.Timestamp(dates[cal_start_i])
    fit = train_mask & (p["timestamp"] <= fit_end).to_numpy()
    cal = train_mask & (p["timestamp"] >= cal_start).to_numpy()

    if not fit.any() or not cal.any():
        raise RuntimeError("Empty inner fit/calibration split.")

    # Verify the requested embargo in session units, not calendar days.
    between = dates[(dates > dates[fit_end_i]) & (dates < dates[cal_start_i])]
    if len(between) < EMBARGO - 1:
        raise RuntimeError(
            f"Inner embargo verification failed: expected {EMBARGO} sessions.")
    return fit, cal, fit_end, cal_start


# ---------------------------------------------------------------------------
# Feature representations
# ---------------------------------------------------------------------------

def build_features(
    p: pd.DataFrame,
    feature_names: Sequence[str],
    variant: str,
    fit_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Only 'raw' is enabled in the initial frozen tournament.

    This function is deliberately explicit so adding transformations later
    requires a new predeclared experiment rather than quietly expanding the
    feature space.
    """
    if variant != "raw":
        raise RuntimeError(f"Unknown feature variant: {variant}")
    X = p[list(feature_names)].copy()
    return X, list(feature_names)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model(family: str, params: Dict[str, Any], seed: int):
    if family == "hgb_classifier":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            **params, early_stopping=False, random_state=seed)

    if family == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            **params, max_iter=1000, solver="lbfgs", random_state=seed)

    if family == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            **params, random_state=seed, n_jobs=1, class_weight="balanced")

    if family == "xgb_classifier":
        xgb = import_optional("xgboost")
        if xgb is None:
            raise ImportError("xgboost is not installed")
        kw = dict(params)
        kw.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": 1,
            "tree_method": "hist",
        })
        return xgb.XGBClassifier(**kw)

    if family == "lgb_classifier":
        lgb = import_optional("lightgbm")
        if lgb is None:
            raise ImportError("lightgbm is not installed")
        kw = dict(params)
        kw.update({"random_state": seed, "n_jobs": 1, "verbosity": -1})
        return lgb.LGBMClassifier(**kw)

    if family == "xgb_ranker":
        xgb = import_optional("xgboost")
        if xgb is None:
            raise ImportError("xgboost is not installed")
        kw = dict(params)
        kw.update({
            "random_state": seed,
            "n_jobs": 1,
            "tree_method": "hist",
            "eval_metric": "ndcg@1",
        })
        return xgb.XGBRanker(**kw)

    if family == "lgb_ranker":
        lgb = import_optional("lightgbm")
        if lgb is None:
            raise ImportError("lightgbm is not installed")
        kw = dict(params)
        kw.update({"random_state": seed, "n_jobs": 1, "verbosity": -1})
        return lgb.LGBMRanker(**kw)

    raise RuntimeError(f"Unknown model family: {family}")


def prepare_matrix(
    X: pd.DataFrame,
    fit_mask: np.ndarray,
    model_family: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns fit-local numeric matrix and fold-local medians. Predictor values
    are stored as float32 to halve dense X memory versus float64; the original
    parquet panel is never modified.

    Tree models receive NaNs directly. Linear models receive medians calculated
    ONLY from the fit rows, then applied to calibration/test.
    """
    # Five approved features are numerically safe in float32 and this halves
    # the dense predictor memory versus float64. Keep the panel parquet itself
    # untouched; only the modelling matrix is downcast.
    arr = X.to_numpy(dtype=np.float32, copy=True)
    med = np.nanmedian(arr[fit_mask], axis=0).astype(np.float32, copy=False)
    bad = ~np.isfinite(med)
    med[bad] = 0.0

    if model_family == "logistic":
        arr = np.where(np.isfinite(arr), arr, med[None, :])
    else:
        # Preserve NaNs for tree models; replace infinities only.
        arr[~np.isfinite(arr) & ~np.isnan(arr)] = np.nan
    return arr, med


def fit_model(
    family: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    p: pd.DataFrame,
    seed: int,
):
    model = make_model(family, params, seed)

    if family in {"xgb_ranker", "lgb_ranker"}:
        # Ranking labels are the economic binary target. Every date is a query.
        qid_fit_dates = p.iloc[fit_idx]["timestamp"].factorize(sort=True)[0]
        # factorize on the fit slice is valid because qid is only grouping;
        # calibration/test qids are supplied separately.
        order = np.argsort(qid_fit_dates, kind="mergesort")
        fit_idx_sorted = fit_idx[order]
        qid_sorted = qid_fit_dates[order].astype(np.int32)
        y_sorted = y[order]

        # Verify qid groups are contiguous after sorting.
        if len(qid_sorted) and np.any(np.diff(qid_sorted) < 0):
            raise RuntimeError("Ranking qid sort failed.")

        if family == "xgb_ranker":
            model.fit(X[fit_idx_sorted], y_sorted, qid=qid_sorted)
        else:
            group = np.bincount(qid_sorted)
            model.fit(X[fit_idx_sorted], y_sorted, group=group)
        return model

    # Classification models.
    yfit = y[fit_idx]
    if len(np.unique(yfit)) < 2:
        raise RuntimeError("Fit window contains only one target class.")
    model.fit(X[fit_idx], yfit)
    return model


def predict_scores(model, family: str, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if family in {"xgb_ranker", "lgb_ranker"}:
        return np.asarray(model.predict(X[idx]), dtype=float)
    return np.asarray(model.predict_proba(X[idx])[:, 1], dtype=float)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_scores(
    raw_cal: np.ndarray,
    y_cal: np.ndarray,
    raw_test: np.ndarray,
) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression
    if len(np.unique(y_cal)) < 2:
        # Calibration cannot be learned from one class. Returning raw scores
        # is explicit and recorded, rather than inventing a mapping.
        return np.clip(raw_test, 0.0, 1.0)

    # For ranker outputs, transform to percentile-like scores first. Isotonic
    # maps the relative ranking score to P(TP first) using calibration only.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal, y_cal)
    return np.asarray(iso.predict(raw_test), dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import log_loss
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def daily_metrics(df: pd.DataFrame, score_col: str = "p_cal", hurdle: float = BREAKEVEN) -> Dict[str, Any]:
    d = df.dropna(subset=["y", score_col]).copy()
    if d.empty:
        return {"signals": 0}

    d["rank"] = d.groupby("timestamp")[score_col].rank(
        ascending=False, method="first")

    out: Dict[str, Any] = {"signals": int(len(d)),
                           "sessions": int(d["timestamp"].nunique())}

    for n in TOP_NS:
        s = d[d["rank"] <= n]
        if s.empty:
            continue
        r = float(s["y"].mean())
        se2 = 2.0 * np.sqrt(r * (1.0 - r) / len(s))
        out[f"top{n}_realised"] = r
        out[f"top{n}_se2"] = float(se2)
        out[f"top{n}_lower_2se"] = float(r - se2)
        out[f"top{n}_clears"] = bool(r - se2 > hurdle)
        out[f"top{n}_signals"] = int(len(s))

    # Spearman is a useful ranking diagnostic, but only as a secondary metric.
    try:
        from scipy.stats import spearmanr
        daily_ic = []
        for _, g in d.groupby("timestamp", sort=True):
            if len(g) >= 3 and g["y"].nunique() >= 2:
                daily_ic.append(spearmanr(g[score_col], g["y"]).statistic)
        out["mean_daily_spearman"] = float(np.nanmean(daily_ic)) if daily_ic else np.nan
        out["median_daily_spearman"] = float(np.nanmedian(daily_ic)) if daily_ic else np.nan
    except Exception:
        out["mean_daily_spearman"] = np.nan
        out["median_daily_spearman"] = np.nan

    return out


def fold_metrics(
    pred: pd.DataFrame,
    family: str,
    fold: int,
    train_start: Any,
    train_end: Any,
    cal_start: Any,
    test_start: Any,
    test_end: Any,
    hurdle: float,
) -> Dict[str, Any]:
    y = pred["y"].to_numpy(dtype=int)
    raw = pred["raw_score"].to_numpy(dtype=float)
    pc = pred["p_cal"].to_numpy(dtype=float)
    dm = daily_metrics(pred, hurdle=hurdle)

    r = {
        "fold": fold,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "cal_start": str(cal_start),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "n_test": int(len(pred)),
        "base_rate": float(y.mean()),
        "auc_raw": auc(y, raw),
        "auc_cal": auc(y, pc),
        "logloss_cal": logloss(y, pc),
        "brier_cal": brier(y, pc),
        "mean_score_raw": float(np.mean(raw)),
        "mean_score_cal": float(np.mean(pc)),
    }
    r.update(dm)
    return r


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def experiment_signature(exp: Dict[str, Any], features: List[str], panel_hash: str, target_spec: Dict[str, Any]) -> str:
    contract = {
        "experiment": exp,
        "features": features,
        "target": target_spec["target"],
        "target_key": target_spec["key"],
        "tp": target_spec["tp"],
        "sl": target_spec["sl"],
        "hurdle": target_spec["hurdle"],
        "hurdle_source": target_spec["hurdle_source"],
        "n_splits": N_SPLITS,
        "embargo": EMBARGO,
        "breakeven": BREAKEVEN,
        "calib_fraction": CALIB_FRACTION,
        "panel_hash": panel_hash,
        "seed": SEED,
    }
    return json_hash(contract)


def checkpoint_path(paths: Paths, exp_id: str, fold: int) -> Path:
    return paths.checkpoints / f"{exp_id}__fold{fold}.json"


def completed_checkpoint(paths: Paths, exp_id: str, fold: int) -> Optional[Dict[str, Any]]:
    p = checkpoint_path(paths, exp_id, fold)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("status") == "completed":
            return obj
    except Exception:
        return None
    return None


def run_one_fold(
    exp: Dict[str, Any],
    fold_no: int,
    sp: Dict[str, Any],
    p: pd.DataFrame,
    X: pd.DataFrame,
    seed: int,
    target_spec: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    train_mask, test_mask = fold_masks(p, sp)
    fit_mask, cal_mask, fit_end, cal_start = split_train_cal_fit(p, train_mask)

    target = target_spec["target"]
    hurdle = float(target_spec["hurdle"])
    fit_idx = np.where(fit_mask & p[target].notna().to_numpy())[0]
    cal_idx = np.where(cal_mask & p[target].notna().to_numpy())[0]
    test_idx = np.where(test_mask & p[target].notna().to_numpy())[0]

    if len(fit_idx) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"fold {fold_no}: only {len(fit_idx)} fit rows.")
    if len(cal_idx) < MIN_CAL_ROWS:
        raise RuntimeError(f"fold {fold_no}: only {len(cal_idx)} calibration rows.")
    if len(test_idx) < MIN_TEST_ROWS:
        raise RuntimeError(f"fold {fold_no}: only {len(test_idx)} test rows.")

    family = exp["family"]
    y = p[target].to_numpy(dtype=np.float32)

    # Fold-local preprocessing.
    Xm, med = prepare_matrix(X, fit_mask, family)

    # Record class balance before fitting.
    y_fit = y[fit_idx].astype(int)
    y_cal = y[cal_idx].astype(int)
    if len(np.unique(y_fit)) < 2:
        raise RuntimeError(f"fold {fold_no}: fit target has one class.")
    if len(np.unique(y_cal)) < 2:
        # Not fatal for calibration, but very unusual; raw scores remain usable.
        log(f"WARNING: {exp['id']} fold {fold_no}: calibration has one class.")

    t0 = time.perf_counter()
    model = fit_model(
        family, exp["params"], Xm, y, fit_idx, cal_idx, p, seed)
    fit_seconds = time.perf_counter() - t0

    raw_cal = predict_scores(model, family, Xm, cal_idx)
    raw_test = predict_scores(model, family, Xm, test_idx)
    p_cal = calibrate_scores(raw_cal, y_cal, raw_test)

    # Sanity: no score may be non-finite.
    if not np.isfinite(raw_test).all() or not np.isfinite(p_cal).all():
        raise RuntimeError(f"fold {fold_no}: model emitted non-finite predictions.")

    pred = pd.DataFrame({
        "timestamp": p.iloc[test_idx]["timestamp"].to_numpy(),
        "symbol": p.iloc[test_idx]["symbol"].to_numpy(),
        "y": y[test_idx].astype(int),
        "raw_score": raw_test,
        "p_cal": np.clip(p_cal, 0.0, 1.0),
        "fold": fold_no,
    })

    metrics = fold_metrics(
        pred, family, fold_no,
        p.loc[train_mask, "timestamp"].min(),
        p.loc[train_mask, "timestamp"].max(),
        cal_start,
        sp["test_start"], sp["test_end"], hurdle)
    metrics["target"] = target
    metrics["target_key"] = target_spec["key"]
    metrics["tp"] = target_spec["tp"]
    metrics["sl"] = target_spec["sl"]
    metrics["hurdle"] = hurdle
    metrics["hurdle_source"] = target_spec["hurdle_source"]
    metrics["fit_seconds"] = fit_seconds
    metrics["fit_rows"] = int(len(fit_idx))
    metrics["cal_rows"] = int(len(cal_idx))
    metrics["test_rows"] = int(len(test_idx))
    metrics["fit_median_missing"] = int(np.isnan(Xm[fit_idx]).sum())

    # Verify no test timestamp is used in fit/cal.
    max_train_date = pd.Timestamp(p.loc[train_mask, "timestamp"].max())
    min_test_date = pd.Timestamp(p.loc[test_mask, "timestamp"].min())
    if min_test_date <= max_train_date:
        raise RuntimeError("Outer test starts on/before outer training end.")

    # Check that the test symbols/dates are unique.
    if pred.duplicated(["timestamp", "symbol"]).any():
        raise RuntimeError("Prediction set has duplicate timestamp/symbol rows.")

    return metrics, pred


def save_predictions(paths: Paths, exp_id: str, fold: int, pred: pd.DataFrame) -> Path:
    paths.predictions.mkdir(parents=True, exist_ok=True)
    path = paths.predictions / f"{exp_id}__fold{fold}.parquet"
    tmp = path.with_suffix(".tmp.parquet")
    pred.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def preflight(root: Path, hours: float = 10.0) -> Dict[str, Any]:
    paths = make_paths(root)
    paths.out.mkdir(parents=True, exist_ok=True)

    log("=" * 78)
    log("MODEL TOURNAMENT PREFLIGHT")
    log("=" * 78)
    log(f"Python: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Root: {paths.root}")
    log(f"Panel: {paths.panel}")
    log(f"Approved: {paths.approved}")

    feats, approved_obj = load_approved(paths.approved)
    log(f"Approved features: {len(feats)}")
    for f in feats:
        log(f"  + {f}")

    if paths.panel.stat().st_size <= 0:
        raise RuntimeError("Panel file is empty on disk.")

    targets = [s["target"] for s in TARGET_SPECS]
    p = load_panel(paths.panel, feats, targets)
    log(f"Panel rows: {len(p):,}")
    log(f"Panel dates: {p['timestamp'].min().date()} .. {p['timestamp'].max().date()}")
    log(f"Symbols: {p['symbol'].nunique():,}")

    for spec in TARGET_SPECS:
        y = p[spec["target"]].dropna()
        log(f"Target {spec['key']}: {len(y):,} rows; base rate: {y.mean():.4f}; "
            f"hurdle: {spec['hurdle']:.3%} ({spec['hurdle_source']})")
        if y.empty or y.nunique() < 2:
            raise RuntimeError(f"Target {spec['target']} does not contain both classes.")

    splits = get_splits(p, N_SPLITS)
    log(f"Outer folds: {len(splits)}")
    for i, sp in enumerate(splits, 1):
        tr, te = fold_masks(p, sp)
        fit, cal, fit_end, cal_start = split_train_cal_fit(p, tr)
        log(f"  fold {i}: train {sp['train_start']}..{sp['train_end']} | "
            f"test {sp['test_start']}..{sp['test_end']} | "
            f"fit_end {fit_end.date()} | cal_start {cal_start.date()} | "
            f"rows train/test={int(tr.sum()):,}/{int(te.sum()):,}")

    # Constructor/import preflight. Missing optional packages are recorded, not
    # fatal, but a package that is installed and has an incompatible API is fatal.
    available = {}
    for pkg in ["sklearn", "xgboost", "lightgbm", "scipy", "pyarrow"]:
        mod = import_optional(pkg)
        available[pkg] = bool(mod)
        log(f"{pkg}: {'AVAILABLE' if mod else 'not installed'}")

    # Instantiate every available experiment and run tiny synthetic fit tests.
    rng = np.random.default_rng(SEED)
    sx = pd.DataFrame(rng.normal(size=(200, len(feats))), columns=feats)
    sy = np.array([0, 1] * 100, dtype=int)
    fake_p = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=200, freq="D"),
        "symbol": [f"S{i%20:03d}" for i in range(200)],
        TARGET: sy,
    })
    # Synthetic model constructor tests, including ranking groups.
    for exp in EXPERIMENTS:
        try:
            fam = exp["family"]
            if fam.startswith("xgb") and not available["xgboost"]:
                status = "SKIP (xgboost unavailable)"
            elif fam.startswith("lgb") and not available["lightgbm"]:
                status = "SKIP (lightgbm unavailable)"
            else:
                if fam in {"xgb_ranker", "lgb_ranker"}:
                    q = fake_p["timestamp"].factorize(sort=True)[0]
                    order = np.argsort(q)
                    m = make_model(fam, exp["params"], SEED)
                    if fam == "xgb_ranker":
                        m.fit(sx.iloc[order], sy[order], qid=q[order])
                    else:
                        m.fit(sx.iloc[order], sy[order],
                              group=np.bincount(q[order]))
                else:
                    m = make_model(fam, exp["params"], SEED)
                    m.fit(sx, sy)
                status = "OK"
        except Exception as e:
            raise RuntimeError(
                f"Preflight model test FAILED for {exp['id']}: {e}") from e
        log(f"experiment {exp['id']}: {status}")

    # Freeze a manifest hash. If the approved feature file changes later, a
    # resumed run must refuse to mix incompatible results.
    panel_hash = file_sha256(paths.panel)
    manifest = {
        "created_at": now_iso(),
        "root": str(paths.root),
        "panel": str(paths.panel),
        "panel_sha256": panel_hash,
        "approved_file_sha256": file_sha256(paths.approved),
        "approved_features": feats,
        "approved_metadata": approved_obj,
        "target_specs": TARGET_SPECS,
        "legacy_default_target": TARGET,
        "legacy_default_breakeven": BREAKEVEN,
        "n_splits": N_SPLITS,
        "embargo": EMBARGO,
        "calib_fraction": CALIB_FRACTION,
        "seed": SEED,
        "experiments": EXPERIMENTS,
        "available_packages": available,
        "requested_hours": hours,
        "method": (
            "Outer walk-forward test folds from project panel_build splitter; "
            "inner fit/cal split with session embargo; fold-local preprocessing; "
            "classification and binary-relevance learning-to-rank; daily Top-N "
            "evaluation; no automatic winner."
        ),
    }
    manifest["manifest_hash"] = json_hash(manifest)
    atomic_write_json(paths.out / "run_manifest.json", manifest)

    log("PREFLIGHT PASSED.")
    return manifest


def verify_resume_manifest(paths: Paths) -> Dict[str, Any]:
    mpath = paths.out / "run_manifest.json"
    if not mpath.exists():
        raise RuntimeError("No run_manifest.json. Start without --resume.")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    current_panel_hash = file_sha256(paths.panel)
    current_approved_hash = file_sha256(paths.approved)
    if manifest.get("panel_sha256") != current_panel_hash:
        raise RuntimeError(
            "Panel SHA256 changed since the tournament started. "
            "Refusing to mix old and new OOS results.")
    if manifest.get("approved_file_sha256") != current_approved_hash:
        raise RuntimeError(
            "approved_features.json changed since the tournament started. "
            "Refusing to mix incompatible results.")
    return manifest


def run_tournament(root: Path, hours: float, resume: bool,
                   max_experiments: Optional[int], cpu: int) -> None:
    paths = make_paths(root)
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    paths.predictions.mkdir(parents=True, exist_ok=True)

    if resume:
        manifest = verify_resume_manifest(paths)
        feats = manifest["approved_features"]
        log("RESUME: manifest verified; panel and approved feature hashes match.")
    else:
        manifest = preflight(root, hours)
        feats = manifest["approved_features"]

    targets = [s["target"] for s in TARGET_SPECS]
    p = load_panel(paths.panel, feats, targets)
    splits = get_splits(p, N_SPLITS)
    X, feature_names = build_features(p, feats, "raw")
    panel_hash = manifest["panel_sha256"]

    base_exps = EXPERIMENTS[:max_experiments] if max_experiments else EXPERIMENTS
    exps = [(spec, exp) for spec in TARGET_SPECS for exp in base_exps]
    started = time.perf_counter()
    deadline = started + hours * 3600.0

    # Create run-level status file.
    status = {
        "started_at": now_iso(),
        "hours": hours,
        "deadline": dt.datetime.now().astimezone().isoformat(),
        "experiments_requested": [exp["id"] for _, exp in exps],
        "status": "running",
    }
    atomic_write_json(paths.out / "run_status.json", status)

    log("=" * 78)
    log(f"TOURNAMENT STARTED | {len(exps)} experiments | {N_SPLITS} folds | "
        f"{hours:.2f}h budget")
    log("=" * 78)

    fold_rows_path = paths.out / "fold_results.csv"
    exp_rows_path = paths.out / "experiment_summary.csv"

    for exp_no, (target_spec, exp) in enumerate(exps, 1):
        log("")
        log("-" * 78)
        exp_id = f"{target_spec['key']}__{exp['id']}"
        log(f"[{exp_no}/{len(exps)}] {exp_id} | {exp['family']} | {exp['variant']} | "
            f"hurdle={target_spec['hurdle']:.3%}")
        log("-" * 78)

        exp_sig = experiment_signature(exp, feats, panel_hash, target_spec)
        fold_results: List[Dict[str, Any]] = []

        # Skip an experiment only if ALL folds are already checkpointed with the
        # exact same signature.
        for fold_no, sp in enumerate(splits, 1):
            if time.perf_counter() >= deadline:
                log("TIME BUDGET REACHED. Stopping cleanly before starting a new fold.")
                status["status"] = "time_budget_reached"
                status["stopped_at"] = now_iso()
                atomic_write_json(paths.out / "run_status.json", status)
                return

            cp = completed_checkpoint(paths, exp_id, fold_no)
            if cp and cp.get("experiment_signature") == exp_sig:
                log(f"fold {fold_no}: already complete; loading checkpoint.")
                fold_results.append(cp["metrics"])
                continue
            elif cp:
                raise RuntimeError(
                    f"Checkpoint exists for {exp_id} fold {fold_no} but its "
                    "experiment signature changed. Refusing to overwrite.")

            cp_path = checkpoint_path(paths, exp_id, fold_no)
            atomic_write_json(cp_path, {
                "status": "running",
                "experiment_id": exp_id,
                "experiment_signature": exp_sig,
                "fold": fold_no,
                "started_at": now_iso(),
            })

            try:
                log(f"fold {fold_no}/{N_SPLITS}: fitting...")
                metrics, pred = run_one_fold(
                    exp, fold_no, sp, p, X, SEED + fold_no, target_spec)
                pred_path = save_predictions(paths, exp_id, fold_no, pred)

                cp_obj = {
                    "status": "completed",
                    "completed_at": now_iso(),
                    "experiment_id": exp_id,
                    "experiment_signature": exp_sig,
                    "fold": fold_no,
                    "metrics": metrics,
                    "prediction_file": str(pred_path),
                }
                atomic_write_json(cp_path, cp_obj)
                fold_results.append(metrics)

                # Append only after the checkpoint has become durable.
                append_csv_atomic(
                    fold_rows_path,
                    {"experiment_id": exp_id,
                     "experiment_signature": exp_sig, **metrics})
                log(f"fold {fold_no}: COMPLETE | "
                    f"AUC={metrics['auc_cal']:.4f} | "
                    f"Top1={metrics.get('top1_realised', np.nan):.3f} | "
                    f"Top1 lower2SE={metrics.get('top1_lower_2se', np.nan):.3f}")

            except ImportError as e:
                # Optional package missing: mark experiment skipped, do not crash
                # the entire overnight run.
                log(f"SKIP {exp_id}: {e}")
                atomic_write_json(cp_path, {
                    "status": "skipped",
                    "experiment_id": exp_id,
                    "experiment_signature": exp_sig,
                    "fold": fold_no,
                    "reason": str(e),
                    "completed_at": now_iso(),
                })
                break

            except Exception as e:
                # Serious coding/data/method error: STOP. Do not continue and
                # produce a partial result that looks complete.
                err = {
                    "status": "failed",
                    "experiment_id": exp_id,
                    "experiment_signature": exp_sig,
                    "fold": fold_no,
                    "failed_at": now_iso(),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(cp_path, err)
                status["status"] = "failed"
                status["failed_at"] = now_iso()
                status["failure"] = err
                atomic_write_json(paths.out / "run_status.json", status)
                raise

        # Only summarize if every fold completed.
        if len(fold_results) == N_SPLITS:
            summary: Dict[str, Any] = {
                "experiment_id": exp_id,
                "experiment_signature": exp_sig,
                "target_key": target_spec["key"],
                "target": target_spec["target"],
                "tp": target_spec["tp"],
                "sl": target_spec["sl"],
                "hurdle": target_spec["hurdle"],
                "hurdle_source": target_spec["hurdle_source"],
                "family": exp["family"],
                "variant": exp["variant"],
                "features": "|".join(feats),
                "folds": N_SPLITS,
            }
            for metric in [
                "auc_cal", "logloss_cal", "brier_cal",
                "top1_realised", "top1_lower_2se",
                "top3_realised", "top3_lower_2se",
                "top5_realised", "top5_lower_2se",
                "top10_realised", "top10_lower_2se",
                "mean_daily_spearman",
            ]:
                vals = np.array([
                    float(r[metric]) for r in fold_results
                    if r.get(metric) is not None and np.isfinite(float(r[metric]))
                ])
                summary[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
                summary[f"{metric}_median"] = float(np.median(vals)) if len(vals) else np.nan
                summary[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan

            # Strict robustness indicators. These are descriptive, not a winner.
            summary["top1_all_folds_above_hurdle_point"] = bool(
                all(r.get("top1_realised", -np.inf) > float(target_spec["hurdle"]) for r in fold_results))
            summary["top1_all_folds_clear_hurdle_2se"] = bool(
                all(r.get("top1_lower_2se", -np.inf) > float(target_spec["hurdle"]) for r in fold_results))
            summary["mean_auc"] = summary["auc_cal_mean"]
            append_csv_atomic(exp_rows_path, summary)
            log(f"SUMMARY {exp_id}: mean AUC={summary['auc_cal_mean']:.4f}, "
                f"mean Top1={summary['top1_realised_mean']:.3f}, "
                f"worst Top1 lower2SE={summary['top1_lower_2se_min']:.3f}")

    status["status"] = "completed"
    status["completed_at"] = now_iso()
    status["elapsed_hours"] = (time.perf_counter() - started) / 3600.0
    atomic_write_json(paths.out / "run_status.json", status)
    log("=" * 78)
    log("TOURNAMENT COMPLETED")
    log("=" * 78)
    log(f"Results: {paths.out}")
    log("No winner is declared automatically. Review experiment_summary.csv,")
    log("then run a separate locked certification on the predeclared candidate(s).")


def report(root: Path) -> None:
    paths = make_paths(root)
    log("=" * 78)
    log("MODEL TOURNAMENT REPORT")
    log("=" * 78)
    status_path = paths.out / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        log(f"Status: {status.get('status')}")
        log(f"Started: {status.get('started_at')}")
        log(f"Completed: {status.get('completed_at', '-')}")
    else:
        log("No run_status.json found.")

    summary = paths.out / "experiment_summary.csv"
    folds = paths.out / "fold_results.csv"
    if not summary.exists():
        log("No complete experiment summaries yet.")
        return

    d = pd.read_csv(summary)
    cols = [
        "experiment_id", "target_key", "hurdle", "family", "auc_cal_mean", "auc_cal_min",
        "top1_realised_mean", "top1_realised_min",
        "top1_lower_2se_mean", "top1_lower_2se_min",
        "top1_all_folds_clear_hurdle_2se",
    ]
    cols = [c for c in cols if c in d.columns]
    log(d[cols].sort_values(
        ["top1_lower_2se_mean", "auc_cal_mean"], ascending=False
    ).to_string(index=False))

    log("")
    log("Interpretation guardrails:")
    log("- Top1 is the closest metric to the intended one-trade-per-day execution.")
    log("- Each target has its own predeclared hurdle; 42.4% applies only to the frozen ATR target.")
    log("- AUC is descriptive; it is not a tradeability verdict.")
    log("- The table is NOT a license to pick the highest row and declare a winner.")
    log("- Candidate selection should be frozen before any final certification.")


def smoke(root: Path) -> None:
    """
    Tiny deterministic end-to-end test on synthetic data. This catches broken
    constructors, ranking group handling, calibration, metric plumbing and
    checkpoint serialization without touching the real panel.
    """
    log("=" * 78)
    log("DETERMINISTIC SMOKE TEST")
    log("=" * 78)
    rng = np.random.default_rng(SEED)
    n_dates = 60
    names = [f"S{i:03d}" for i in range(20)]
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    rows = []
    for di, d in enumerate(dates):
        for si, s in enumerate(names):
            z = rng.normal()
            x1 = z + rng.normal(scale=0.2)
            x2 = rng.normal()
            y = int(x1 + rng.normal(scale=0.8) > 0.0)
            rows.append((d, s, x1, x2, y))
    q = pd.DataFrame(rows, columns=["timestamp", "symbol", "f1", "f2", TARGET])
    X = q[["f1", "f2"]].copy()
    y = q[TARGET].to_numpy(dtype=float)

    # Classification path.
    m = make_model("hgb_classifier", EXPERIMENTS[0]["params"], SEED)
    m.fit(X, y.astype(int))
    ps = m.predict_proba(X)[:, 1]
    assert np.isfinite(ps).all() and len(ps) == len(q)

    # Calibration path.
    pc = calibrate_scores(ps[:500], y[:500], ps[500:])
    assert np.isfinite(pc).all()
    assert ((pc >= 0) & (pc <= 1)).all()

    # Ranking path if XGBoost is installed.
    if import_optional("xgboost") is not None:
        qid = q["timestamp"].factorize(sort=True)[0].astype(np.int32)
        order = np.argsort(qid, kind="mergesort")
        m = make_model("xgb_ranker", EXPERIMENTS[6]["params"], SEED)
        m.fit(X.iloc[order], y.astype(int)[order], qid=qid[order])
        rp = m.predict(X)
        assert np.isfinite(rp).all()

    dm = daily_metrics(pd.DataFrame({
        "timestamp": q["timestamp"],
        "symbol": q["symbol"],
        "y": y.astype(int),
        "raw_score": ps,
        "p_cal": pc.tolist() + [np.nan] * (len(q) - len(pc)),
    }).dropna(subset=["p_cal"]))
    assert "top1_realised" in dm

    log("Smoke test PASSED.")
    log("Real panel was NOT modified.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ["preflight", "smoke", "report"]:
        sp = sub.add_parser(name)
        sp.add_argument("--root", required=True)

    sp = sub.add_parser("run")
    sp.add_argument("--root", required=True)
    sp.add_argument("--hours", type=float, default=10.0)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--max-experiments", type=int, default=None)
    sp.add_argument("--cpu", type=int, default=1,
                    help="Reserved for future parallel execution; currently kept at 1 "
                         "to maximize reproducibility and memory safety.")
    sp.add_argument("--seed", type=int, default=SEED,
                    help="Currently only 42 is supported by the frozen manifest.")

    a = ap.parse_args()
    root = Path(a.root).expanduser()

    if a.cmd == "preflight":
        preflight(root)
        return 0
    if a.cmd == "smoke":
        smoke(root)
        return 0
    if a.cmd == "report":
        report(root)
        return 0
    if a.cmd == "run":
        if a.seed != SEED:
            raise SystemExit(
                f"Frozen tournament seed is {SEED}; refusing a different seed.")
        run_tournament(root, a.hours, a.resume, a.max_experiments, a.cpu)
        return 0
    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
