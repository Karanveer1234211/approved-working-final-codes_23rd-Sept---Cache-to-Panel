#!/usr/bin/env python3
"""
label_audit.py - prove the target is what the contract says it is.

    python label_audit.py --root %CACHE_DAILY_ROOT%
    python label_audit.py --root %CACHE_DAILY_ROOT% --symbols 120

Cheap and standalone. Run it after every panel rebuild, BEFORE any research.

INDEPENDENCE
============
Checking a label with the code that produced it proves nothing. This file
imports nothing from panel_build's label code: it re-derives every sampled
label from the raw cache OHLCV with a plain loop written from the contract,
then compares with what the panel stored. panel_build uses vectorised running
maxima; this uses explicit session-by-session checks. Agreement between two
unrelated implementations is evidence; agreement with itself is not.

THE CONTRACT (primary variant)
------------------------------
    ATR14      SIMPLE 14-day mean of true range, / close at T
               (NOT D_atr14 - that feature is Wilder-smoothed)
    TP level   +1.5 x ATR14      SL level  -1.0 x ATR14
    path       sessions T+1..T+5: high vs TP, low vs SL, from close[T]
    tie        both crossed in one session -> SL
    exit_ret   +TP | -SL | close[T+5]/close[T]-1
    pending    last 5 sessions of each symbol -> NaN, never negative

Two checks go beyond restating the contract:
  * a 'neither' outcome must lie STRICTLY between -SL and +TP. If the close
    at T+5 were outside, the high or low on that session would have crossed
    a barrier. A violation is an impossible outcome.
  * label_tp_before_sl, label_first_touch and label_exit_ret must tell the
    same story on every row of the full panel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TP_MULT, SL_MULT, H, ATR_N = 1.5, 1.0, 5, 14
TOL = 1e-9
LABELS = ["label_tp_before_sl", "label_first_touch", "label_exit_ret",
          "label_same_day_ambiguous", "label_fwd_ret_5d"]


class Report:
    def __init__(self):
        self.fail, self.lines = [], []

    def ok(self, msg):
        self.lines.append(("ok  ", msg)); print(f"  [ok  ] {msg}", flush=True)

    def warn(self, msg):
        self.lines.append(("WARN", msg)); print(f"  [WARN] {msg}", flush=True)

    def bad(self, msg):
        self.fail.append(msg); self.lines.append(("FAIL", msg))
        print(f"  [FAIL] {msg}", flush=True)


def _dates(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return t.dt.normalize()


# ----------------------------------------------------------------------
# independent reference implementation - written from the contract
# ----------------------------------------------------------------------
def reference_labels(o_h_l_c: pd.DataFrame) -> pd.DataFrame:
    h = o_h_l_c["high"].to_numpy(float)
    l = o_h_l_c["low"].to_numpy(float)
    c = o_h_l_c["close"].to_numpy(float)
    n = len(c)
    tr = np.full(n, np.nan)
    for k in range(1, n):
        tr[k] = max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1]))
    atr = np.full(n, np.nan)
    for k in range(ATR_N, n):
        w = tr[k - ATR_N + 1:k + 1]
        if np.isfinite(w).all():
            atr[k] = w.mean()
    first = np.full(n, np.nan); exit_r = np.full(n, np.nan)
    tie = np.full(n, np.nan); tp_l = np.full(n, np.nan); sl_l = np.full(n, np.nan)
    for t in range(0, n - H):
        if not (np.isfinite(atr[t]) and np.isfinite(c[t]) and c[t] > 0):
            continue
        a = atr[t] / c[t]
        tp, sl = TP_MULT * a, SL_MULT * a
        tp_l[t], sl_l[t] = tp, sl
        res, tt = 0.0, 0.0
        for k in range(1, H + 1):
            up = h[t + k] / c[t] - 1 >= tp
            dn = l[t + k] / c[t] - 1 <= -sl
            if up and dn:
                res, tt = -1.0, 1.0; break
            if up:
                res = 1.0; break
            if dn:
                res = -1.0; break
        first[t], tie[t] = res, tt
        exit_r[t] = tp if res == 1 else (-sl if res == -1 else c[t + H] / c[t] - 1)
    return pd.DataFrame({"date": _dates(o_h_l_c["timestamp"]).to_numpy(),
                         "r_first": first, "r_exit": exit_r, "r_tie": tie,
                         "r_tp": tp_l, "r_sl": sl_l})


# ----------------------------------------------------------------------
def audit(root: Path, n_symbols: int = 60, seed: int = 0) -> Report:
    import pyarrow.parquet as pq
    import research_common as RC
    from data_quality import _paths
    pp = root / "panel" / "panel.parquet"
    R = Report()

    print("\n=== 1. PANEL STRUCTURE ===")
    schema = pq.ParquetFile(pp).schema_arrow.names
    meta_p = pp.parent / "panel_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    lc = meta.get("label_contract")
    if not lc:
        R.bad("panel_meta.json has no label_contract - panel built with an old "
              "panel_build.py; rebuild")
    elif lc.get("primary_variant") != "atr1p5_1p0":
        R.bad(f"primary variant is {lc.get('primary_variant')}, expected atr1p5_1p0")
    else:
        R.ok(f"label contract: {lc['take_profit']} TP / {lc['stop_loss']} SL, "
             f"{lc['horizon_sessions']} sessions, ties -> SL")
    if "label_tp" in meta or "label_sl" in meta:
        R.bad("stale label_tp/label_sl still in panel_meta.json")
    missing = [c for c in LABELS if c not in schema]
    if missing:
        R.bad(f"label columns absent: {missing} - rebuild the panel")
        return R
    fwd = [c for c in schema if c in RC.FORWARD_COLUMNS]
    (R.bad if fwd else R.ok)(f"forward-looking columns stored as non-labels: {fwd or 'none'}")

    P = pd.read_parquet(pp, columns=["timestamp", "symbol"] + LABELS)
    P["date"] = _dates(P["timestamp"])
    import panel_build as PB
    n_feat = len(PB.panel_feature_columns(pd.DataFrame(columns=schema)))
    R.ok(f"{len(P):,} rows | {P['symbol'].nunique():,} symbols | "
         f"{P['date'].min().date()} -> {P['date'].max().date()} | {n_feat} feature columns")
    dup = int(P.duplicated(["date", "symbol"]).sum())
    (R.bad if dup else R.ok)(f"duplicate (date, symbol) rows: {dup}")
    for c in LABELS:
        v = pd.to_numeric(P[c], errors="coerce").to_numpy(float)
        ninf = int(np.isinf(v).sum())
        if ninf:
            R.bad(f"{c}: {ninf} infinite values")
    R.ok("no infinite values in any label column") if not any(
        "infinite" in f for f in R.fail) else None

    print("\n=== 2. INTERNAL CONSISTENCY, FULL PANEL ===")
    ft = pd.to_numeric(P["label_first_touch"], errors="coerce").to_numpy(float)
    tb = pd.to_numeric(P["label_tp_before_sl"], errors="coerce").to_numpy(float)
    er = pd.to_numeric(P["label_exit_ret"], errors="coerce").to_numpy(float)
    amb = pd.to_numeric(P["label_same_day_ambiguous"], errors="coerce").to_numpy(float)
    f5 = pd.to_numeric(P["label_fwd_ret_5d"], errors="coerce").to_numpy(float)
    res = np.isfinite(ft)
    c1 = int(((tb == 1) != (ft == 1))[res].sum())
    (R.bad if c1 else R.ok)(f"label_tp_before_sl disagrees with first_touch on {c1:,} rows")
    c2 = int((np.isfinite(ft) != np.isfinite(er)).sum())
    (R.bad if c2 else R.ok)(f"exit_ret missing where outcome is known (or vice versa): {c2:,}")
    c3 = int(((ft == 1) & ~(er > 0)).sum())
    (R.bad if c3 else R.ok)(f"TP-first rows with exit_ret <= 0: {c3:,}")
    c4 = int(((ft == -1) & ~(er < 0)).sum())
    (R.bad if c4 else R.ok)(f"SL-first rows with exit_ret >= 0: {c4:,}")
    nm = (ft == 0) & np.isfinite(f5)
    c5 = int((np.abs(er[nm] - f5[nm]) > TOL).sum())
    (R.bad if c5 else R.ok)(f"'neither' rows where exit_ret != 5-session return: {c5:,}")
    c6 = int(((amb == 1) & (ft != -1)).sum())
    (R.bad if c6 else R.ok)(f"same-day ties NOT counted as SL: {c6:,}")
    n_res = int(res.sum())
    counts = {"TP_first": float((ft == 1).sum() / n_res), "SL_first": float((ft == -1).sum() / n_res),
              "neither": float((ft == 0).sum() / n_res), "tie_as_SL": float((amb == 1).sum() / n_res)}
    R.ok("outcomes: " + " | ".join(f"{k} {v:.1%}" for k, v in counts.items()))
    q = np.nanpercentile(er, [1, 5, 25, 50, 75, 95, 99])
    R.ok("exit_ret percentiles 1/5/25/50/75/95/99: "
         + " ".join(f"{x:+.2%}" for x in q))

    print("\n=== 3. INDEPENDENT RECOMPUTATION FROM THE RAW CACHE ===")
    rng = np.random.default_rng(seed)
    sizes = P.groupby("symbol").size().sort_values()
    systematic = list(sizes.index[:5]) + list(sizes.index[-5:])
    others = [s for s in sizes.index if s not in systematic]
    pick = systematic + list(rng.choice(others, min(n_symbols, len(others)), replace=False))
    tot = {"rows": 0, "outcome": 0, "exit": 0, "tie": 0, "impossible": 0,
           "pending_bad": 0, "no_cache": 0}
    worst = []
    for sym in pick:
        cp, _ = _paths(root, sym)
        if not Path(cp).exists():
            tot["no_cache"] += 1; continue
        raw = pd.read_parquet(cp, columns=["timestamp", "high", "low", "close"])
        raw = raw.sort_values("timestamp").reset_index(drop=True)
        ref = reference_labels(raw)
        pan = P[P["symbol"] == sym][["date", "label_first_touch", "label_exit_ret",
                                     "label_same_day_ambiguous"]]
        m = pan.merge(ref, on="date", how="left")
        pf = pd.to_numeric(m["label_first_touch"], errors="coerce").to_numpy(float)
        pe = pd.to_numeric(m["label_exit_ret"], errors="coerce").to_numpy(float)
        pa = pd.to_numeric(m["label_same_day_ambiguous"], errors="coerce").to_numpy(float)
        rf, re_, rt = m["r_first"].to_numpy(), m["r_exit"].to_numpy(), m["r_tie"].to_numpy()
        both = np.isfinite(pf) & np.isfinite(rf)
        tot["rows"] += int(both.sum())
        tot["outcome"] += int((pf[both] != rf[both]).sum())
        tot["exit"] += int((np.abs(pe[both] - re_[both]) > TOL).sum())
        tot["tie"] += int((pa[both] != rt[both]).sum())
        nei = both & (pf == 0)
        tp_, sl_ = m["r_tp"].to_numpy(), m["r_sl"].to_numpy()
        tot["impossible"] += int((~((pe[nei] > -sl_[nei]) & (pe[nei] < tp_[nei]))).sum())
        # pending: the last H cache sessions must be NaN in the panel
        tail = set(_dates(raw["timestamp"]).iloc[-H:])
        tail_rows = pan[pan["date"].isin(tail)]
        tot["pending_bad"] += int(pd.to_numeric(tail_rows["label_first_touch"],
                                                errors="coerce").notna().sum())
        if both.any():
            d = np.abs(pe[both] - re_[both])
            worst.append((sym, float(np.nanmax(d))))
    checked = len(pick) - tot["no_cache"]
    R.ok(f"{checked} symbols recomputed ({len(systematic)} systematic: fewest/most "
         f"rows; rest random) | {tot['rows']:,} labelled rows compared")
    if tot["no_cache"]:
        R.warn(f"{tot['no_cache']} sampled symbols had no cache file")
    for k, lab in (("outcome", "TP/SL/neither outcome mismatches"),
                   ("exit", f"exit_ret mismatches (> {TOL:g})"),
                   ("tie", "tie-flag mismatches"),
                   ("impossible", "'neither' outcomes outside (-SL, +TP)"),
                   ("pending_bad", "last-5-session rows carrying a label")):
        (R.bad if tot[k] else R.ok)(f"{lab}: {tot[k]:,}")
    if worst:
        s_, w_ = max(worst, key=lambda x: x[1])
        R.ok(f"largest exit_ret difference: {w_:.2e} ({s_})")

    out = {"passed": not R.fail, "failures": R.fail, "outcome_mix": counts,
           "exit_ret_pct": dict(zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"],
                                    map(float, q))),
           "recomputation": tot, "symbols_checked": checked}
    (pp.parent / "label_audit.json").write_text(json.dumps(out, indent=2, default=str),
                                                encoding="utf-8")
    return R


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--symbols", type=int, default=60)
    a = ap.parse_args()
    root = a.root or os.environ.get("CACHE_DAILY_ROOT")
    if not root:
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    R = audit(Path(root), a.symbols)
    print("\n" + "=" * 64)
    if R.fail:
        print("  LABEL AUDIT FAILED - do not run any research on this panel:")
        for f in R.fail:
            print(f"    - {f}")
        return 1
    print("  LABEL AUDIT PASSED - the target matches its contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
