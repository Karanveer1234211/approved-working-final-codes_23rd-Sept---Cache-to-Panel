#!/usr/bin/env python3
"""
feature_audit.py - decide what deserves to enter the model. Standalone.

    python feature_audit.py run   --root <cache_root>
    python feature_audit.py run   --root <cache_root> --mode greedy
    python feature_audit.py derive --root <cache_root>   # just build derived cols
    python feature_audit.py report --root <cache_root>

THE ONE RULE THAT MAKES THIS HONEST
===================================
FEATURE SELECTION IS MODEL FITTING.

Choosing features by looking at 2016-2026 and then "testing" on 2024-2026 is
not an out-of-sample test - the selection already saw the answers. Every
number in the incremental-value stage is therefore produced inside a fold:
selected on that fold's TRAIN window, scored on its TEST window, with the
same purge and embargo the model evaluation uses. Labels span five sessions,
so without the embargo a training row's outcome window overlaps the first
test rows and information crosses the boundary.

The cheap diagnostics (stages 1-5) run on all data by design - they describe
the data rather than choosing from it. The moment a decision is made, it is
made per fold.

WHAT IT PRODUCES
----------------
    feature_audit.parquet     every candidate, every metric
    approved_features.json    the set that earned its place
    feature_clusters.csv      redundancy groups and their representatives
    interaction_candidates.csv
    feature_stability.csv     IC per era, sign consistency
    feature_rejections.csv    what was dropped and WHY - kept deliberately,
                              because a feature that is redundant against
                              today's model may not be against tomorrow's

APPROVAL RULE
--------------
In greedy mode a candidate must actually be selected (win a greedy step) in
at least 80% of folds, AND its selected-step incremental AUC must clear the
measured permutation-null cut in at least 80% of folds. With the frozen
5-fold audit this is 4/5 folds for both conditions. Merely being evaluated
in a fold does not count as selection.

WHAT IT CANNOT TELL YOU
-----------------------
Whether the approved set makes money. It ranks information content against
one target, on daily bars, gross of costs. A feature can carry real signal
and still be untradeable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_TARGET = "label_tp_before_sl"
SECONDARY_TARGETS = ["label_touch", "label_fwd_ret_5d",
                     "label_mfe_5d", "label_mae_5d"]

MIN_COVERAGE = 0.60        # drop features missing on >40% of rows
NEAR_CONST_TOL = 1e-10
DUP_CORR = 0.999           # |r| above this = the same feature twice
CLUSTER_CORR = 0.85        # redundancy grouping threshold
# A fixed "meaningful AUC gain" is guesswork, and guessing low is fatal: a
# threshold under the noise floor approves whatever the ranking stage feeds
# it. The floor is MEASURED instead, by a PERMUTATION null.
#
# Sentinels are real candidate columns with their values SHUFFLED. Gaussian
# noise was tried first and was too well behaved - real features have fatter
# tails and more internal structure, and a flexible model overfits that
# structure more than it overfits clean noise, so a gaussian floor sat below
# what junk features actually achieved. Shuffling preserves the marginal
# distribution and every quirk of the column while destroying only its
# relationship to the target, which is precisely the null being tested.
N_SENTINELS = 10           # permuted columns per fold
SENTINEL_SIGMAS = 3.0      # candidate must beat mean + k*sd of the null
MIN_INCREMENTAL = 0.0015   # absolute floor, applied on top of the null
TRAIN_SAMPLE = 150_000     # rows per fold fit; audit is exploratory
ERAS = 3                   # equal-width time blocks for stability


# ----------------------------------------------------------------------
# STAGE 3 - DERIVED FEATURES
#
# Generated from an explicit spec, not combinatorially. Normalised by ATR
# wherever a raw difference would otherwise encode price level or volatility
# rather than the relationship being measured: (Close - EMA20) is a bigger
# number for an expensive, volatile stock regardless of what the gap means.
# ----------------------------------------------------------------------
DISTANCE_SPECS = [
    ("close", "D_ema20", "dist_close_ema20"),
    ("close", "D_ema50", "dist_close_ema50"),
    ("D_ema20", "D_ema50", "dist_ema20_ema50"),
    ("D_ema50", "D_ema200", "dist_ema50_ema200"),
    ("D_close", "D_vwap20", "dist_close_vwap20"),
]
DIFF_SPECS = [
    ("D_rsi7", "D_rsi14", "diff_rsi7_rsi14"),
    ("D_rsi14", "D_rsi21", "diff_rsi14_rsi21"),
    ("D_adx14", "D_adx28", "diff_adx14_adx28"),
]
RATIO_SPECS = [
    ("D_atr14", "D_atr50", "ratio_atr14_atr50"),
    ("D_realvol_20", "D_realvol_60", "ratio_rvol20_60"),
    ("D_bb_width_20", "D_atr_pct", "ratio_bbwidth_atr"),
    ("D_volume", "D_vol_sma20", "ratio_vol_vol20"),
]
SLOPE_SPECS = [                     # (column, lookback sessions)
    ("D_ema20", 3), ("D_ema20", 5), ("D_ema50", 10),
    ("D_rsi14", 5), ("D_obv", 5), ("D_macd", 5),
]

# Stage 4 - interactions, by CATEGORY rather than by column. Every pair of
# categories is crossed once using each category's representative, which
# keeps the candidate count linear instead of quadratic in feature count.
CATEGORIES = {
    "trend": ["D_ema20_angle_deg", "dist_ema20_ema50_atr", "D_adx14"],
    "momentum": ["D_rsi14", "D_macd_hist", "D_roc10"],
    "volatility": ["D_atr_pct", "ratio_atr14_atr50", "D_realvol_ratio_20_60"],
    "volume": ["D_dvol_z20", "ratio_vol_vol20", "D_obv_slope5"],
    "location": ["D_pos_in_52w_range", "D_bb_pctB_20", "D_drawdown_252"],
}


def _atr_norm(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in ("D_atr14", "D_atr_pct", "D_realvol_20"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if c == "D_atr_pct" and "close" in df.columns:
                v = v / 100.0 * pd.to_numeric(df["close"], errors="coerce")
            if v.abs().sum() > 0:
                return v.replace(0, np.nan)
    return None


def derive(panel: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Build the controlled derived set. Returns (new columns, their names)."""
    out: Dict[str, pd.Series] = {}
    atr = _atr_norm(panel)
    made, skipped = [], []

    def num(c):
        return pd.to_numeric(panel[c], errors="coerce") if c in panel.columns else None

    for a, b, name in DISTANCE_SPECS:
        x, y = num(a), num(b)
        if x is None or y is None:
            skipped.append(name); continue
        out[f"{name}_atr"] = (x - y) / atr if atr is not None else (x - y) / y.abs()
        made.append(f"{name}_atr")

    for a, b, name in DIFF_SPECS:
        x, y = num(a), num(b)
        if x is None or y is None:
            skipped.append(name); continue
        out[name] = x - y
        made.append(name)

    for a, b, name in RATIO_SPECS:
        x, y = num(a), num(b)
        if x is None or y is None:
            skipped.append(name); continue
        out[name] = x / y.replace(0, np.nan)
        made.append(name)

    # Slopes normalised by ATR: a "10 rupee rise over 5 days" means something
    # different for every stock; "0.4 ATR over 5 days" is comparable.
    g = panel.groupby("symbol", sort=False)
    for c, lb in SLOPE_SPECS:
        if c not in panel.columns:
            skipped.append(f"{c}_slope{lb}"); continue
        v = pd.to_numeric(panel[c], errors="coerce")
        sl = v - g[c].shift(lb)
        name = f"{c.replace('D_', '')}_slope{lb}"
        out[name] = sl / atr if atr is not None else sl
        made.append(name)

    new = pd.DataFrame(out, index=panel.index)
    if verbose:
        print(f"  derived {len(made)} features"
              + (f", skipped {len(skipped)} (inputs absent)" if skipped else ""))
    return new, made


def build_interactions(df: pd.DataFrame, verbose: bool = True
                       ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Cross categories, not columns.

    Crossing every feature with every other turns 200 features into ~20,000
    candidates, and at that count something will look predictive by chance no
    matter what the data says. One representative per category keeps the
    candidate list small enough that a surviving interaction means something.
    """
    present = {k: [c for c in v if c in df.columns] for k, v in CATEGORIES.items()}
    present = {k: v for k, v in present.items() if v}
    out, made, rows = {}, [], []
    cats = sorted(present)
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            ca, cb = present[a][0], present[b][0]
            x = pd.to_numeric(df[ca], errors="coerce")
            y = pd.to_numeric(df[cb], errors="coerce")
            # Rank-transform before multiplying so the product is not
            # dominated by whichever input has the fatter tail.
            xr = x.rank(pct=True) - 0.5
            yr = y.rank(pct=True) - 0.5
            name = f"ix_{a}_{b}"
            out[name] = xr * yr
            made.append(name)
            rows.append({"interaction": name, "cat_a": a, "cat_b": b,
                         "col_a": ca, "col_b": cb})
    if verbose:
        print(f"  {len(made)} interactions from {len(cats)} categories "
              f"(vs {len(df.columns)**2:,} if crossing every column)")
    return pd.DataFrame(out, index=df.index), made, pd.DataFrame(rows)


# ----------------------------------------------------------------------
# STAGE 1 - HYGIENE
# ----------------------------------------------------------------------
def hygiene(df: pd.DataFrame, feats: List[str], verbose: bool = True) -> pd.DataFrame:
    recs = []
    for c in feats:
        v = pd.to_numeric(df[c], errors="coerce")
        cov = float(v.notna().mean())
        nun = int(v.nunique(dropna=True))
        sd = float(v.std()) if cov > 0 else np.nan
        q1, q99 = (v.quantile(0.01), v.quantile(0.99)) if cov > 0 else (np.nan, np.nan)
        iqr = v.quantile(0.75) - v.quantile(0.25) if cov > 0 else np.nan
        tail = float(((v < q1 - 10 * iqr) | (v > q99 + 10 * iqr)).mean()) \
            if pd.notna(iqr) and iqr > 0 else 0.0
        reason = None
        if cov < MIN_COVERAGE:
            reason = f"coverage {cov:.1%} < {MIN_COVERAGE:.0%}"
        elif nun <= 1 or (pd.notna(sd) and sd < NEAR_CONST_TOL):
            reason = "constant or near-constant"
        elif not np.isfinite(v.replace([np.inf, -np.inf], np.nan)).any():
            reason = "no finite values"
        recs.append({"feature": c, "coverage": round(cov, 4), "n_unique": nun,
                     "std": sd, "extreme_tail_frac": round(tail, 5),
                     "hygiene_fail": reason})
    h = pd.DataFrame(recs)
    if verbose:
        bad = h["hygiene_fail"].notna().sum()
        print(f"  hygiene: {len(h) - bad}/{len(h)} pass, {bad} rejected")
    return h


def find_duplicates(df: pd.DataFrame, feats: List[str], verbose: bool = True
                    ) -> Dict[str, str]:
    """Pairs correlating above DUP_CORR are the same feature under two names."""
    sub = df[feats].apply(pd.to_numeric, errors="coerce")
    sub = sub.sample(min(len(sub), 60_000), random_state=0)
    corr = sub.corr(method="pearson", min_periods=200).abs()
    dupe: Dict[str, str] = {}
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        if a in dupe:
            continue
        for b in cols[i + 1:]:
            if b in dupe:
                continue
            r = corr.loc[a, b]
            if pd.notna(r) and r >= DUP_CORR:
                dupe[b] = a
    if verbose and dupe:
        print(f"  {len(dupe)} duplicate features (|r| >= {DUP_CORR})")
    return dupe


# ----------------------------------------------------------------------
# STAGE 2 - STANDALONE INFORMATION
# ----------------------------------------------------------------------
def rank_ic(df: pd.DataFrame, feats: List[str], target: str,
            date_col: str = "timestamp", min_names: int = 20) -> pd.Series:
    """
    Cross-sectional rank IC: correlate feature and target WITHIN each date,
    then average across dates.

    Pooling every row instead would mix two different things - whether the
    feature ranks names correctly on a given day, and whether its level drifts
    with the market over years. A cross-sectional model only ever uses the
    first, so the second is noise dressed as signal.
    """
    y = pd.to_numeric(df[target], errors="coerce")
    ok = y.notna()
    d = df.loc[ok, [date_col] + feats].copy()
    d["_y"] = y[ok]
    res = {}
    for c in feats:
        vals = []
        for _, grp in d.groupby(date_col, sort=False):
            v = pd.to_numeric(grp[c], errors="coerce")
            m = v.notna() & grp["_y"].notna()
            if int(m.sum()) < min_names or v[m].nunique() < 3:
                continue
            vals.append(v[m].rank().corr(grp["_y"][m].rank()))
        res[c] = float(np.nanmean(vals)) if vals else np.nan
    return pd.Series(res)


def stability(df: pd.DataFrame, feats: List[str], target: str,
              eras: int = ERAS, verbose: bool = True) -> pd.DataFrame:
    """IC per era plus sign consistency - a feature that flips sign is noise."""
    d = df.copy()
    d["_era"] = pd.qcut(d["timestamp"].rank(method="first"), eras,
                        labels=[f"era{i+1}" for i in range(eras)])
    out = {}
    for e, grp in d.groupby("_era", observed=True):
        out[str(e)] = rank_ic(grp, feats, target)
        if verbose:
            print(f"    {e}: {grp['timestamp'].min().date()} -> "
                  f"{grp['timestamp'].max().date()} ({len(grp):,} rows)")
    s = pd.DataFrame(out)
    s["ic_mean"] = s.mean(axis=1)
    s["ic_std"] = s.std(axis=1)
    sign = np.sign(s[[c for c in s.columns if c.startswith("era")]])
    s["sign_stable"] = (sign.abs().sum(axis=1) > 0) & \
                       (sign.sum(axis=1).abs() == sign.abs().sum(axis=1))
    return s.reset_index().rename(columns={"index": "feature"})


# ----------------------------------------------------------------------
# STAGE 5 - REDUNDANCY
# ----------------------------------------------------------------------
def cluster(df: pd.DataFrame, feats: List[str], ic: pd.Series,
            thresh: float = CLUSTER_CORR, verbose: bool = True) -> pd.DataFrame:
    """
    Group features that move together; keep the highest-|IC| one per group.

    Greedy single-link on |correlation|. Crude compared with a proper
    hierarchical clustering, but the decision it drives - which of five
    momentum variants to carry - is not sensitive to the difference.
    """
    sub = df[feats].apply(pd.to_numeric, errors="coerce")
    sub = sub.sample(min(len(sub), 60_000), random_state=0)
    corr = sub.corr().abs()
    unassigned = set(feats)
    rows, cid = [], 0
    order = ic.reindex(feats).abs().sort_values(ascending=False).index
    for f in order:
        if f not in unassigned:
            continue
        members = [g for g in unassigned
                   if g == f or (pd.notna(corr.loc[f, g]) and corr.loc[f, g] >= thresh)]
        cid += 1
        for g in members:
            rows.append({"cluster": cid, "feature": g, "representative": f,
                         "is_representative": g == f,
                         "abs_ic": float(abs(ic.get(g, np.nan)))})
        unassigned -= set(members)
    c = pd.DataFrame(rows)
    if verbose:
        print(f"  {cid} clusters from {len(feats)} features "
              f"(|r| >= {thresh})")
    return c


# ----------------------------------------------------------------------
# STAGE 6 - INCREMENTAL OOS VALUE, WALK-FORWARD
# ----------------------------------------------------------------------
def _fit_score(tr_X, tr_y, te_X, te_y, seed: int = 0) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    if tr_y.nunique() < 2 or te_y.nunique() < 2:
        return np.nan
    m = HistGradientBoostingClassifier(
        max_iter=120, max_depth=4, learning_rate=0.08,
        early_stopping=False, random_state=seed)
    m.fit(tr_X, tr_y)
    return float(roc_auc_score(te_y, m.predict_proba(te_X)[:, 1]))


def incremental(df: pd.DataFrame, base: List[str], candidates: List[str],
                target: str, splits, *, mode: str = "marginal",
                max_steps: int = 10, verbose: bool = True) -> pd.DataFrame:
    """
    How much does each candidate add OOS, ON TOP OF the base set?

    mode="marginal": each candidate is added to base on its own. Answers
        "does this carry information the base lacks", costs one fit per
        candidate per fold, and cannot tell you whether two candidates are
        substitutes for each other.
    mode="greedy": the winner is added and the search repeats. Answers the
        sharper question - the smallest set that works - at roughly
        max_steps times the cost.

    EVERY FIT IS INSIDE A FOLD. Selection on train, score on test, embargo
    already applied by walk_forward_splits. A candidate that only looks good
    when selection sees the test window will not look good here.
    """
    y = pd.to_numeric(df[target], errors="coerce")
    ts = pd.to_datetime(df["timestamp"])
    have = y.notna().to_numpy()
    recs = []
    rng = np.random.default_rng(0)

    # Permutation null: real columns, shuffled. Whatever AUC these earn is
    # what a feature with this shape and NO relationship to the target earns.
    pool = candidates if len(candidates) >= N_SENTINELS else candidates * 4
    picks = list(rng.choice(pool, size=min(N_SENTINELS, len(pool)),
                            replace=len(pool) < N_SENTINELS))
    sent = {}
    for i, c in enumerate(picks):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        sent[f"__perm{i}_{c}"[:60]] = rng.permutation(v)
    sent = pd.DataFrame(sent, index=df.index)
    df = pd.concat([df, sent], axis=1)
    sentinels = list(sent.columns)
    for fi, sp in enumerate(splits, 1):
        # walk_forward_splits returns date boundaries with the embargo already
        # subtracted from train_end, so masks built from them inherit it.
        tr_m = (ts <= pd.Timestamp(sp["train_end"])).to_numpy()
        te_m = ((ts >= pd.Timestamp(sp["test_start"])) &
                (ts <= pd.Timestamp(sp["test_end"]))).to_numpy()
        tr = np.where(tr_m & have)[0]
        te = np.where(te_m & have)[0]
        if len(tr) > TRAIN_SAMPLE:
            tr = rng.choice(tr, TRAIN_SAMPLE, replace=False)
        if not len(tr) or not len(te):
            continue
        ytr, yte = y.iloc[tr].astype(int), y.iloc[te].astype(int)

        def sc(cols):
            return _fit_score(df.iloc[tr][cols], ytr, df.iloc[te][cols], yte)

        base_auc = sc(base)
        if verbose:
            print(f"    fold {fi}: train -> {sp['train_end']} | embargo "
                  f"{sp['embargo_sessions']} | test {sp['test_start']}.."
                  f"{sp['test_end']} | base AUC {base_auc:.4f} "
                  f"({len(tr):,}/{len(te):,})")
        for c in sentinels:
            a = sc(base + [c])
            recs.append({"fold": fi, "feature": c, "base_auc": base_auc,
                         "with_auc": a, "delta": a - base_auc, "step": 0,
                         "is_sentinel": True})

        if mode == "marginal":
            for c in candidates:
                a = sc(base + [c])
                recs.append({"fold": fi, "feature": c, "base_auc": base_auc,
                             "with_auc": a, "delta": a - base_auc, "step": 1,
                             "is_sentinel": False})
        else:
            cur, pool = list(base), list(candidates)
            cur_auc = base_auc
            for step in range(1, max_steps + 1):
                best, best_a = None, cur_auc
                step_records = {}
                for c in pool:
                    a = sc(cur + [c])
                    rec = {"fold": fi, "feature": c, "base_auc": cur_auc,
                           "with_auc": a, "delta": a - cur_auc,
                           "step": step, "is_sentinel": False,
                           "selected": False}
                    recs.append(rec)
                    step_records[c] = rec
                    if a > best_a:
                        best, best_a = c, a
                if best is None or best_a - cur_auc < MIN_INCREMENTAL:
                    break
                step_records[best]["selected"] = True
                cur.append(best); pool.remove(best); cur_auc = best_a
                if verbose:
                    print(f"      step {step}: +{best} -> {cur_auc:.4f}")
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------
def run(panel_path, *, target: str = DEFAULT_TARGET, mode: str = "marginal",
        top_candidates: int = 25, n_splits: int = 3,
        max_rows: int = 400_000, verbose: bool = True) -> Path:
    import panel_build as PB

    panel_path = Path(panel_path)
    out_dir = panel_path.parent / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    p = pd.read_parquet(panel_path)
    p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None)
    p = p.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if len(p) > max_rows:
        # Subsample by DATE, never by row: dropping random rows from a date
        # would corrupt every cross-sectional statistic computed below.
        keep_dates = pd.Index(sorted(p["timestamp"].unique()))
        stride = max(1, int(len(p) / max_rows))
        keep_dates = keep_dates[::stride]
        p = p[p["timestamp"].isin(keep_dates)].reset_index(drop=True)
        if verbose:
            print(f"  subsampled to {len(p):,} rows across "
                  f"{len(keep_dates):,} dates (whole dates kept intact)")

    base_feats = [c for c in PB.panel_feature_columns(p)
                  if pd.api.types.is_numeric_dtype(p[c])]
    if verbose:
        print(f"\n[1/7] derived features")
    der, der_names = derive(p, verbose)
    p = pd.concat([p, der], axis=1)

    if verbose:
        print(f"\n[2/7] interactions")
    ix, ix_names, ix_map = build_interactions(p, verbose)
    p = pd.concat([p, ix], axis=1)
    ix_map.to_csv(out_dir / "interaction_candidates.csv", index=False)

    all_feats = base_feats + der_names + ix_names
    if verbose:
        print(f"\n[3/7] hygiene on {len(all_feats)} candidates")
    hy = hygiene(p, all_feats, verbose)
    alive = hy.loc[hy["hygiene_fail"].isna(), "feature"].tolist()

    dupes = find_duplicates(p, alive, verbose)
    alive = [f for f in alive if f not in dupes]

    if verbose:
        print(f"\n[4/7] standalone rank IC on {len(alive)} features")
    ic = rank_ic(p, alive, target)

    if verbose:
        print(f"\n[5/7] stability across {ERAS} eras")
    st = stability(p, alive, target, verbose=verbose)
    st.to_csv(out_dir / "feature_stability.csv", index=False)

    if verbose:
        print(f"\n[6/7] redundancy clustering")
    cl = cluster(p, alive, ic, verbose=verbose)
    cl.to_csv(out_dir / "feature_clusters.csv", index=False)
    reps = cl.loc[cl["is_representative"], "feature"].tolist()

    st_i = st.set_index("feature")
    ranked = (ic.reindex(reps).abs()
              * st_i.reindex(reps)["sign_stable"].map({True: 1.0, False: 0.5})
              ).sort_values(ascending=False)
    cands = ranked.head(top_candidates).index.tolist()
    # A deliberately small, uncontroversial base so incremental value is
    # measured against something, not against nothing.
    base = [c for c in ("D_rsi14", "D_atr_pct", "D_ema20_angle_deg",
                        "D_dvol_z20", "D_pos_in_52w_range") if c in p.columns]
    cands = [c for c in cands if c not in base]

    if verbose:
        print(f"\n[7/7] incremental OOS ({mode}) - {len(cands)} candidates, "
              f"{n_splits} folds")
        print(f"  base: {base}")
    splits = PB.walk_forward_splits(p, n_splits=n_splits)
    inc = incremental(p, base, cands, target, splits, mode=mode, verbose=verbose)

    if len(inc):
        sen = inc[inc["is_sentinel"]]["delta"].dropna()
        if len(sen) > 1:
            null_mu, null_sd = float(sen.mean()), float(sen.std())
            null_floor = null_mu + SENTINEL_SIGMAS * null_sd
        else:
            null_mu = null_sd = 0.0
            null_floor = 0.0
        cut = max(null_floor, MIN_INCREMENTAL)
        if verbose:
            print(f"\n  PERMUTATION NULL ({len(sen)} shuffled-column fits): "
                  f"mean {null_mu:+.4f}  sd {null_sd:.4f}")
            print(f"  cut = mean + {SENTINEL_SIGMAS:.0f}sd = {null_floor:+.4f}"
                  f"  ->  a candidate must beat {cut:+.4f} AUC")
        # In greedy mode, a feature is considered "selected" only when it
        # actually wins a step in a fold. A feature merely being evaluated
        # (possibly at several steps) does NOT count as fold selection.
        # This prevents "tested in 5 folds" from being confused with
        # "selected in 5 folds".
        nonsent = inc[~inc["is_sentinel"]].copy()
        if mode == "greedy":
            selected_rows = nonsent[nonsent["selected"]].copy()
        else:
            # Marginal mode has no greedy winner. Treat a candidate as selected
            # in a fold only when its measured incremental AUC clears the
            # measured null cut in that fold.
            selected_rows = nonsent[nonsent["delta"] >= cut].copy()

        if len(selected_rows):
            agg = (selected_rows.groupby("feature")
                   .agg(
                       selected_folds=("fold", "nunique"),
                       selected_delta_mean=("delta", "mean"),
                       selected_delta_min=("delta", "min"),
                   )
                   .reset_index())
            null_counts = (selected_rows.assign(null_clear=selected_rows["delta"] >= cut)
                           .groupby("feature")["null_clear"].sum()
                           .rename("null_clear_folds")
                           .reset_index())
            agg = agg.merge(null_counts, on="feature", how="left")
        else:
            agg = pd.DataFrame(columns=[
                "feature", "selected_folds", "selected_delta_mean",
                "selected_delta_min", "null_clear_folds"
            ])
    else:
        null_floor, cut = 0.0, MIN_INCREMENTAL
        agg = pd.DataFrame(columns=["feature", "delta_mean", "delta_min", "folds"])

    audit = (hy.merge(ic.rename("ic").reset_index()
                      .rename(columns={"index": "feature"}), on="feature", how="left")
             .merge(st_i[["ic_std", "sign_stable"]].reset_index(), on="feature", how="left")
             .merge(cl[["feature", "cluster", "is_representative", "representative"]],
                    on="feature", how="left")
             .merge(agg, on="feature", how="left"))
    audit["kind"] = np.where(audit["feature"].isin(ix_names), "interaction",
                             np.where(audit["feature"].isin(der_names),
                                      "derived", "base"))
    audit["duplicate_of"] = audit["feature"].map(dupes)

    def verdict(r):
        if pd.notna(r["hygiene_fail"]):
            return "REJECT_HYGIENE"
        if pd.notna(r["duplicate_of"]):
            return "DUPLICATE"
        if r["is_representative"] is False:
            return "REDUNDANT"

        sf = r.get("selected_folds")
        nf = r.get("null_clear_folds")
        if pd.notna(sf):
            # New approval rule:
            #   1) selected in at least 4 of 5 folds
            #   2) the selected-step incremental AUC clears the measured
            #      permutation-null cut in at least 4 of 5 folds
            #
            # With a different number of folds, use ceil(80% of folds).
            required = int(np.ceil(0.80 * n_splits))
            if int(sf) >= required and pd.notna(nf) and int(nf) >= required:
                return "KEEP"
            if pd.notna(r.get("selected_delta_mean")):
                return "UNSTABLE"
            return "NO_INCREMENTAL_VALUE"

        return "NOT_SELECTED_ANY_FOLD"

    audit["status"] = audit.apply(verdict, axis=1)
    audit.to_parquet(out_dir / "feature_audit.parquet", index=False)
    audit[audit["status"].str.startswith(("REJECT", "DUPLICATE", "REDUNDANT",
                                          "NO_INCREMENTAL", "UNSTABLE"))] \
        .to_csv(out_dir / "feature_rejections.csv", index=False)

    approved = base + audit.loc[audit["status"] == "KEEP", "feature"].tolist()
    (out_dir / "approved_features.json").write_text(json.dumps({
        "built_at": dt.datetime.now().isoformat(),
        "target": target, "mode": mode, "n_splits": n_splits,
        "null_method": "permutation (shuffled real columns)",
        "null_floor_auc": round(null_floor, 5),
        "incremental_cut": round(cut, 5),
        "n_sentinels": N_SENTINELS,
        "base": base, "approved": approved,
        "approval_rule": f"selected_in >= {int(np.ceil(0.80 * n_splits))}/{n_splits} folds AND null_clear >= {int(np.ceil(0.80 * n_splits))}/{n_splits} folds",
        "note": "Selected per fold on train windows only. Rejected features "
                "are retained in feature_rejections.csv - redundancy is "
                "relative to today's model, not permanent.",
    }, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n  {out_dir}")
        print(f"  approved {len(approved)} features in "
              f"{(time.perf_counter()-t0)/60:.1f} min")
    return out_dir


def report(panel_path) -> None:
    d = Path(panel_path).parent / "audit"
    a = pd.read_parquet(d / "feature_audit.parquet")
    print(f"\n  {len(a)} candidates audited\n")
    print(f"  {'status':<24}{'n':>6}")
    for s, n in a["status"].value_counts().items():
        print(f"  {s:<24}{n:>6}")
    k = a[a["status"] == "KEEP"].sort_values("selected_delta_mean", ascending=False)
    if len(k):
        print(f"\n  {'feature':<32}{'kind':>12}{'IC':>8}{'stable':>8}"
              f"{'sel':>6}{'null':>6}{'dAUC':>9}{'worst':>9}")
        for _, r in k.head(25).iterrows():
            print(f"  {r['feature']:<32}{r['kind']:>12}{r['ic']:>8.4f}"
                  f"{str(r['sign_stable']):>8}{int(r['selected_folds']):>6}"
                  f"{int(r['null_clear_folds']):>6}"
                  f"{r['selected_delta_mean']:>+9.4f}"
                  f"{r['selected_delta_min']:>+9.4f}")
    print("\n  'worst' is the weakest fold. A feature with a good mean and a")
    print("  negative worst fold is not stable - it is averaging over a period")
    print("  where it actively hurt.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "derive", "report"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--mode", default="marginal", choices=["marginal", "greedy"])
    ap.add_argument("--top-candidates", type=int, default=25)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=400_000)
    a = ap.parse_args()

    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT", "."))
    pdir = Path(a.panel) if a.panel else root / "panel"
    ppq = pdir / "panel.parquet" if pdir.is_dir() else pdir

    if a.cmd == "run":
        run(ppq, target=a.target, mode=a.mode,
            top_candidates=a.top_candidates, n_splits=a.splits,
            max_rows=a.max_rows)
    elif a.cmd == "derive":
        p = pd.read_parquet(ppq)
        _, names = derive(p)
        print("  " + ", ".join(names))
    else:
        report(ppq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
