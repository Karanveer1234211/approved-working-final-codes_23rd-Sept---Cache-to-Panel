#!/usr/bin/env python3
"""
base_model.py - train the base model and find out whether it is USABLE.

    python base_model.py train  --root <cache_root>
    python base_model.py train  --root <cache_root> --breakeven 0.424
    python base_model.py report --root <cache_root>

WHY THIS IS NOT JUST A TRAINING SCRIPT
======================================
AUC 0.57 means the ranking beats chance. It says nothing about whether any
probability this model emits is economically usable, and those are different
questions:

    AUC             does a higher score mean a higher chance of TP first?
    CALIBRATION     when it says 45%, does 45% of that bucket actually hit?
    THRESHOLD       is there ANY cut-off where the realised rate clears
                    breakeven after costs, on enough trades to matter?

A model can rank well and still never produce a single tradeable bucket. The
barrier scan says ~42.4% P(TP first) is breakeven at 1.5a/1.0a net of 35 bps.
The base rate is ~28.4%. So the only question that matters here is whether
the top of the score distribution reaches 42.4% out of sample.

THE CALIBRATION TRAP
--------------------
A calibrator fitted on the test set makes any model look perfectly
calibrated - it has seen the answers. So each fold's train window is split:
the model fits on the earlier part, the calibrator fits on a held-out tail,
with the same embargo used everywhere else. The test window is touched once,
for scoring, and never for fitting anything.

WHAT THIS STILL CANNOT TELL YOU
-------------------------------
Whether it makes money. There is no slippage model here, no position sizing,
no capacity limit, no borrow cost, and it assumes you can take every signal
at the close. Treat a positive result as "worth a paper-traded forward test",
not as a strategy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_TARGET = "label_tp_before_sl"
CALIB_FRACTION = 0.25       # tail of each train window held out for calibration
EMBARGO = 5                 # sessions, matches the label horizon
TRAIN_CAP = 400_000
DEFAULT_BREAKEVEN = 0.424   # from barrier_scan at 1.5a/1.0a, 35 bps round trip
TOP_N_DAILY = (1, 3, 5, 10)


def _model(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.06,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def _metrics(y, p) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"auc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "mean_pred": float(p.mean()), "base_rate": float(y.mean()),
            "n": int(len(y))}


def train_walk_forward(panel_path, *, features: List[str], target: str,
                       n_splits: int = 5, verbose: bool = True) -> pd.DataFrame:
    """
    Fit per fold, calibrate on a held-out tail of train, predict test.

    Returns every out-of-sample prediction, pooled across folds, with the
    columns needed for the economic evaluation that follows.
    """
    import panel_build as PB
    from sklearn.isotonic import IsotonicRegression

    p = pd.read_parquet(panel_path)
    p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None)
    p = p.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    missing = [f for f in features if f not in p.columns]
    if missing:
        raise SystemExit(f"approved features missing from the panel: {missing}")

    y_all = pd.to_numeric(p[target], errors="coerce")
    ts = p["timestamp"]
    sessions = np.sort(ts.unique())
    splits = PB.walk_forward_splits(p, n_splits=n_splits)
    rng = np.random.default_rng(0)
    out = []

    for fi, sp in enumerate(splits, 1):
        tr_end = pd.Timestamp(sp["train_end"])
        te_m = ((ts >= pd.Timestamp(sp["test_start"])) &
                (ts <= pd.Timestamp(sp["test_end"]))).to_numpy()

        # Split the TRAIN window into fit / calibrate, with an embargo
        # between them - the calibration tail must not overlap the outcome
        # windows of the rows the model fitted on.
        tr_sess = sessions[sessions <= np.datetime64(tr_end)]
        cut = max(1, int(len(tr_sess) * (1 - CALIB_FRACTION)))
        fit_end = pd.Timestamp(tr_sess[cut - 1])
        cal_start_i = min(cut - 1 + EMBARGO, len(tr_sess) - 1)
        cal_start = pd.Timestamp(tr_sess[cal_start_i])

        fit_m = (ts <= fit_end).to_numpy()
        cal_m = ((ts >= cal_start) & (ts <= tr_end)).to_numpy()
        have = y_all.notna().to_numpy()
        fit_i = np.where(fit_m & have)[0]
        cal_i = np.where(cal_m & have)[0]
        te_i = np.where(te_m & have)[0]
        if len(fit_i) > TRAIN_CAP:
            fit_i = np.sort(rng.choice(fit_i, TRAIN_CAP, replace=False))
        if not len(fit_i) or not len(cal_i) or not len(te_i):
            continue

        ytr = y_all.iloc[fit_i].astype(int)
        ycal = y_all.iloc[cal_i].astype(int)
        yte = y_all.iloc[te_i].astype(int)

        m = _model()
        m.fit(p.iloc[fit_i][features], ytr)
        raw_cal = m.predict_proba(p.iloc[cal_i][features])[:, 1]
        raw_te = m.predict_proba(p.iloc[te_i][features])[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_cal, ycal)
        cal_te = iso.predict(raw_te)

        if verbose:
            r = _metrics(yte, raw_te)
            c = _metrics(yte, cal_te)
            print(f"  fold {fi}: fit<={fit_end.date()} | cal "
                  f"{cal_start.date()}..{tr_end.date()} | test "
                  f"{sp['test_start']}..{sp['test_end']}")
            print(f"     n_fit {len(fit_i):,}  n_cal {len(cal_i):,}  "
                  f"n_test {len(te_i):,}")
            print(f"     AUC {r['auc']:.4f} | logloss {r['logloss']:.4f} "
                  f"-> {c['logloss']:.4f} | brier {r['brier']:.4f} "
                  f"-> {c['brier']:.4f}", flush=True)

        out.append(pd.DataFrame({
            "fold": fi, "timestamp": ts.iloc[te_i].to_numpy(),
            "symbol": p["symbol"].iloc[te_i].to_numpy(),
            "y": yte.to_numpy(), "p_raw": raw_te, "p_cal": cal_te}))

    if not out:
        raise SystemExit("no folds produced predictions")
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
def calibration_table(pred: pd.DataFrame, col: str = "p_cal",
                      bins: int = 10) -> pd.DataFrame:
    """Predicted versus realised, by decile of the score."""
    d = pred.dropna(subset=[col, "y"]).copy()
    d["bucket"] = pd.qcut(d[col].rank(method="first"), bins, labels=False)
    g = d.groupby("bucket").agg(
        n=("y", "size"), predicted=(col, "mean"), realised=("y", "mean"),
        lo=(col, "min"), hi=(col, "max")).reset_index()
    # 2 standard errors on the realised rate: a bucket whose realised rate is
    # within noise of its prediction is calibrated as far as this data shows.
    g["se2"] = 2 * np.sqrt(g["realised"] * (1 - g["realised"]) / g["n"])
    return g


def threshold_sweep(pred: pd.DataFrame, breakeven: float,
                    col: str = "p_cal") -> pd.DataFrame:
    """
    For each cut-off: how many signals, and does the realised rate clear
    breakeven?

    This is the question AUC cannot answer. A model can order names correctly
    and still never put enough of them above the line to trade.
    """
    d = pred.dropna(subset=[col, "y"])
    rows = []
    for t in np.arange(0.20, 0.71, 0.02):
        s = d[d[col] >= t]
        if len(s) < 50:
            continue
        r = float(s["y"].mean())
        se2 = 2 * np.sqrt(r * (1 - r) / len(s))
        rows.append({"threshold": round(float(t), 2), "n": int(len(s)),
                     "pct_of_rows": len(s) / len(d),
                     "realised": r, "se2": se2,
                     "clears_breakeven": bool(r - se2 > breakeven),
                     "margin": r - breakeven})
    return pd.DataFrame(rows)


def daily_topn(pred: pd.DataFrame, col: str = "p_cal",
               ns=TOP_N_DAILY) -> pd.DataFrame:
    """
    Take the top N scores each session - which is how the system would
    actually be used - and measure the realised rate.

    A threshold sweep answers "is any bucket good enough". This answers "is
    the thing I would actually do good enough", and they can differ: on some
    days the top name scores 0.60, on others the best available is 0.31.
    """
    d = pred.dropna(subset=[col, "y"]).copy()
    d["rk"] = d.groupby("timestamp")[col].rank(ascending=False, method="first")
    rows = []
    for n in ns:
        s = d[d["rk"] <= n]
        if s.empty:
            continue
        r = float(s["y"].mean())
        se2 = 2 * np.sqrt(r * (1 - r) / len(s))
        rows.append({"top_n": n, "signals": int(len(s)),
                     "sessions": int(s["timestamp"].nunique()),
                     "realised": r, "se2": se2,
                     "mean_score": float(s[col].mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def train(panel_path, *, breakeven: float = DEFAULT_BREAKEVEN,
          n_splits: int = 5, target: str = DEFAULT_TARGET,
          verbose: bool = True) -> Path:
    panel_path = Path(panel_path)
    audit_dir = panel_path.parent / "audit"
    out_dir = panel_path.parent / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    af = audit_dir / "approved_features.json"
    if not af.exists():
        raise SystemExit(
            f"{af} not found - run feature_audit.py first. Training on an "
            f"unaudited feature set is how a bloated model gets mistaken "
            f"for a good one.")
    ap = json.loads(af.read_text(encoding="utf-8"))
    features = ap["approved"]
    if verbose:
        print(f"\n  target:   {target}")
        print(f"  features: {len(features)} approved")
        for f in features:
            print(f"      {f}")
        print(f"  breakeven: {breakeven:.1%} P(TP first) "
              f"(barrier_scan, net of costs)")
        print(f"\n[1/3] walk-forward training ({n_splits} folds)", flush=True)

    pred = train_walk_forward(panel_path, features=features, target=target,
                              n_splits=n_splits, verbose=verbose)
    pred.to_parquet(out_dir / "oos_predictions.parquet", index=False)

    if verbose:
        print(f"\n[2/3] pooled out-of-sample metrics", flush=True)
    m_raw = _metrics(pred["y"], pred["p_raw"])
    m_cal = _metrics(pred["y"], pred["p_cal"])
    if verbose:
        print(f"  {'':<12}{'AUC':>9}{'logloss':>10}{'brier':>9}"
              f"{'mean p':>9}")
        print(f"  {'raw':<12}{m_raw['auc']:>9.4f}{m_raw['logloss']:>10.4f}"
              f"{m_raw['brier']:>9.4f}{m_raw['mean_pred']:>9.3f}")
        print(f"  {'calibrated':<12}{m_cal['auc']:>9.4f}"
              f"{m_cal['logloss']:>10.4f}{m_cal['brier']:>9.4f}"
              f"{m_cal['mean_pred']:>9.3f}")
        print(f"  base rate {m_cal['base_rate']:.4f} on "
              f"{m_cal['n']:,} OOS predictions")

    cal = calibration_table(pred)
    cal.to_csv(out_dir / "calibration.csv", index=False)
    sw = threshold_sweep(pred, breakeven)
    sw.to_csv(out_dir / "threshold_sweep.csv", index=False)
    tn = daily_topn(pred)
    tn.to_csv(out_dir / "daily_topn.csv", index=False)

    if verbose:
        print(f"\n[3/3] economic evaluation", flush=True)
        _print_eval(cal, sw, tn, breakeven, m_cal)

    usable = bool(sw["clears_breakeven"].any()) if len(sw) else False
    (out_dir / "model_meta.json").write_text(json.dumps({
        "built_at": dt.datetime.now().isoformat(),
        "target": target, "features": features, "n_splits": n_splits,
        "breakeven": breakeven,
        "oos_auc_raw": m_raw["auc"], "oos_auc_cal": m_cal["auc"],
        "oos_logloss_cal": m_cal["logloss"], "oos_brier_cal": m_cal["brier"],
        "base_rate": m_cal["base_rate"],
        "any_threshold_clears_breakeven": usable,
        "note": "Calibrator fitted on a held-out tail of each train window "
                "with an embargo; the test window is used for scoring only. "
                "No costs beyond the breakeven figure, no slippage model, no "
                "capacity limit.",
    }, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n  {out_dir}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return out_dir


def _print_eval(cal, sw, tn, breakeven, m) -> None:
    print("\n  CALIBRATION - does a stated probability mean what it says?")
    print(f"  {'decile':>7}{'n':>9}{'range':>16}{'predicted':>11}"
          f"{'realised':>10}{'2se':>8}")
    for _, r in cal.iterrows():
        flag = "" if abs(r["predicted"] - r["realised"]) <= r["se2"] else "  off"
        print(f"  {int(r['bucket'])+1:>7}{int(r['n']):>9,}"
              f"{f'{r.lo:.3f}-{r.hi:.3f}':>16}{r['predicted']:>11.3f}"
              f"{r['realised']:>10.3f}{r['se2']:>8.3f}{flag}")

    print(f"\n  THRESHOLD SWEEP - breakeven is {breakeven:.1%}")
    print(f"  {'cut':>6}{'n':>10}{'% rows':>9}{'realised':>10}{'2se':>8}"
          f"{'margin':>9}")
    for _, r in sw.iterrows():
        mark = "  CLEARS" if r["clears_breakeven"] else ""
        print(f"  {r['threshold']:>6.2f}{int(r['n']):>10,}"
              f"{r['pct_of_rows']:>8.1%}{r['realised']:>10.3f}"
              f"{r['se2']:>8.3f}{r['margin']:>+9.3f}{mark}")

    print(f"\n  DAILY TOP-N - what the system would actually do")
    print(f"  {'top':>5}{'signals':>10}{'sessions':>10}{'realised':>10}"
          f"{'2se':>8}{'mean score':>12}")
    for _, r in tn.iterrows():
        print(f"  {int(r['top_n']):>5}{int(r['signals']):>10,}"
              f"{int(r['sessions']):>10,}{r['realised']:>10.3f}"
              f"{r['se2']:>8.3f}{r['mean_score']:>12.3f}")

    any_clear = bool(sw["clears_breakeven"].any()) if len(sw) else False
    print("\n" + "=" * 68)
    if any_clear:
        best = sw[sw["clears_breakeven"]].iloc[0]
        print(f"  A threshold clears breakeven: at p>={best['threshold']:.2f}, "
              f"{int(best['n']):,} signals")
        print(f"  realised {best['realised']:.1%} vs breakeven "
              f"{breakeven:.1%} (margin {best['margin']:+.1%})")
        print("  'Clears' means realised MINUS two standard errors is still")
        print("  above breakeven - not merely that the point estimate is.")
    else:
        print("  NO threshold clears breakeven out of sample.")
        print(f"  Best realised rate is "
              f"{sw['realised'].max():.1%} against a {breakeven:.1%} bar.")
        print("  The model ranks better than chance and is still not")
        print("  tradeable at this bracket and cost assumption. Options:")
        print("    - a wider bracket (barrier_scan showed lower breakevens)")
        print("    - reduce costs")
        print("    - find more signal (this is where Soul gets tested)")
    print("=" * 68)
    print("  NOT A BACKTEST. No slippage, no sizing, no capacity limit, and")
    print("  it assumes every signal is taken at the close.")
    print("=" * 68)


def report(panel_path) -> None:
    d = Path(panel_path).parent / "model"
    meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
    cal = pd.read_csv(d / "calibration.csv")
    sw = pd.read_csv(d / "threshold_sweep.csv")
    tn = pd.read_csv(d / "daily_topn.csv")
    print(f"\n  built {meta['built_at'][:19]} | {len(meta['features'])} features")
    print(f"  OOS AUC {meta['oos_auc_cal']:.4f} | brier "
          f"{meta['oos_brier_cal']:.4f} | base rate {meta['base_rate']:.4f}")
    _print_eval(cal, sw, tn, meta["breakeven"],
                {"base_rate": meta["base_rate"]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["train", "report"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--breakeven", type=float, default=DEFAULT_BREAKEVEN)
    a = ap.parse_args()

    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT", "."))
    pdir = Path(a.panel) if a.panel else root / "panel"
    ppq = pdir / "panel.parquet" if pdir.is_dir() else pdir

    if a.cmd == "train":
        train(ppq, breakeven=a.breakeven, n_splits=a.splits, target=a.target)
    else:
        report(ppq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
