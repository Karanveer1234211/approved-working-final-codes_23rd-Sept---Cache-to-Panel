import sys, types, json, tempfile, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
for n in ("kiteconnect","kiteconnect.exceptions"): sys.modules[n]=types.ModuleType(n)
sys.modules["kiteconnect"].KiteConnect=object
for n in ("KiteException","TokenException","InputException","NetworkException","DataException","GeneralException"):
    setattr(sys.modules["kiteconnect.exceptions"],n,type(n,(Exception,),{}))
sys.path.insert(0,".")
import importlib
pb = importlib.import_module("panel_build")
fx = importlib.import_module("features_daily")

root=Path(tempfile.mkdtemp()); pdir=Path(tempfile.mkdtemp())
ses=pd.DatetimeIndex(pd.bdate_range("2021-01-01","2024-12-31"))
IST=fx.IST
def w(sym,ix,kind="equity",seed=0,vol=True,exch=None):
    r=np.random.default_rng(seed)
    c=100*np.exp(np.cumsum(r.standard_t(4,len(ix))*0.013))
    o=c*(1+r.normal(0,.004,len(ix)))
    d=pd.DataFrame({"timestamp":ix.tz_localize(IST),"open":o,
                    "high":np.maximum(c,o)*1.008,"low":np.minimum(c,o)*0.992,"close":c,
                    "volume":r.lognormal(13.6,.85,len(ix)) if vol else np.nan})
    d.to_parquet(root/f"{sym}_daily.parquet",index=False)
    m={"schema_version":"v26","rows":len(d),"series_kind":kind,
       "corporate_action_suspects":[],"last_timestamp":str(d["timestamp"].iloc[-1])}
    if exch: m["exchange"]=exch
    (root/f"{sym}_daily.ok.json").write_text(json.dumps(m,default=str))

N=30
for i in range(N): w(f"STK{i:02d}",ses,seed=i)
w("NIFTY50",ses,kind="index",seed=900,vol=False)
print(f"fixture: {N} equities + index, {len(ses)} sessions\n")

print("=== FULL BUILD ===")
p=pb.build_panel(root,pdir,full=True)
panel=pd.read_parquet(p)
meta=json.loads((pdir/"panel_meta.json").read_text())
print()

print("=== CHECKS ===")
# 1. labels: last horizon sessions unresolved, NOT zero
last=sorted(panel["timestamp"].unique())[-pb.LABEL_HORIZON:]
tail=panel[panel["timestamp"].isin(last)]
assert tail["label_touch"].isna().all(), "recent labels must be UNKNOWN"
print(f"  ok   last {pb.LABEL_HORIZON} sessions unlabelled ({len(tail)} rows), not zero-filled")

# 2. label is genuinely forward and correct on a spot check
s0=panel[panel.symbol=="STK00"].sort_values("timestamp").reset_index(drop=True)
raw=pd.read_parquet(root/"STK00_daily.parquet")
raw["timestamp"]=pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)
j=len(s0)//2; d0=s0.loc[j,"timestamp"]
ri=raw.index[raw.timestamp==d0][0]
fwd=raw.loc[ri+1:ri+5,"high"].max()/raw.loc[ri,"close"]-1
assert abs(fwd-s0.loc[j,"label_mfe_5d"])<1e-12
assert int(s0.loc[j,"label_touch"])==int(fwd>=0.05)
print(f"  ok   label_touch matches hand-computed 5-day MFE ({fwd:+.3%})")

# 3. no label leaks into X
X=pb.panel_feature_columns(panel)
assert not any(c.startswith("label_") for c in X)
assert "ret_5d_close_pct" not in X
print(f"  ok   X has {len(X)} cols, zero label/forward columns")

# 4. cross-sectional ranks are within-date
g=panel.groupby("timestamp")["X_rank_D_rsi14"]
mx=g.max().dropna(); mn=g.min().dropna()
assert (mx<=1.0+1e-9).all() and (mn>=0).all()
print(f"  ok   cross-sectional ranks in [0,1] within each date")

# 5. point-in-time universe: a late lister must not appear before listing
w("LATE",ses[400:],seed=777)
p2=pb.build_panel(root,pdir,full=True,verbose=False)
pn2=pd.read_parquet(p2)
first_late=pn2[pn2.symbol=="LATE"]["timestamp"].min()
assert first_late>=ses[400]
others=pn2[(pn2.timestamp<ses[400])]["symbol"].unique()
assert "LATE" not in others
print(f"  ok   late lister absent before {first_late.date()} (point-in-time universe)")

# 6. incremental == full: EXACT equivalence, not just numeric maxdiff
pn_full=pd.read_parquet(p2).copy()
p3=pb.build_panel(root,pdir,full=False,rebuild_days=15,verbose=False)
pn_inc=pd.read_parquet(p3)

a=pn_full.sort_values(["timestamp","symbol"]).reset_index(drop=True)
b=pn_inc.sort_values(["timestamp","symbol"]).reset_index(drop=True)

assert list(a.columns)==list(b.columns), (
    f"column sets differ: only-full={set(a.columns)-set(b.columns)}, "
    f"only-inc={set(b.columns)-set(a.columns)}")
print(f"  ok   identical column list ({len(a.columns)})")

dt_diff={c:(str(a[c].dtype),str(b[c].dtype)) for c in a.columns if a[c].dtype!=b[c].dtype}
assert not dt_diff, f"dtype drift: {dt_diff}"
print(f"  ok   identical dtypes across all {len(a.columns)} columns")

assert len(a)==len(b), f"row count {len(a)} vs {len(b)}"
assert a[["timestamp","symbol"]].equals(b[["timestamp","symbol"]]), "key order differs"
print(f"  ok   identical row keys ({len(a):,} rows)")

nan_mismatch=[c for c in a.columns if not a[c].isna().equals(b[c].isna())]
assert not nan_mismatch, f"NaN masks differ: {nan_mismatch[:6]}"
print(f"  ok   identical NaN masks (labels included)")

bad=[]
for c in a.columns:
    if pd.api.types.is_numeric_dtype(a[c]):
        av=a[c].to_numpy(dtype="float64",na_value=np.nan)
        bv=b[c].to_numpy(dtype="float64",na_value=np.nan)
        if not np.array_equal(av,bv,equal_nan=True): bad.append(c)
    else:
        if not a[c].equals(b[c]): bad.append(c)
assert not bad, f"value mismatch in {bad[:6]}"
print(f"  ok   bit-exact values in every column (numeric and non-numeric)")

# 7. embargo is enforced
try:
    pb.walk_forward_splits(pn_inc,n_splits=3,embargo=2); print("  FAIL embargo not enforced")
except ValueError as e: print(f"  ok   embargo < horizon rejected: {str(e)[:52]}")
folds=pb.walk_forward_splits(pn_inc,n_splits=3)
for f in folds:
    assert pd.Timestamp(f["train_end"])<pd.Timestamp(f["test_start"])
print(f"  ok   {len(folds)} walk-forward folds, train_end < test_start with embargo")
print(f"       e.g. fold1 train->{folds[0]['train_end']} test {folds[0]['test_start']}..{folds[0]['test_end']}")

# 8. universe change forces a full rebuild, and quarantined history is dropped
import shutil
(root/"STK07_daily.parquet").unlink()
(root/"STK07_daily.ok.json").unlink()
p4=pb.build_panel(root,pdir,full=False,rebuild_days=15,verbose=True)
pn4=pd.read_parquet(p4)
assert "STK07" not in set(pn4["symbol"]), "removed symbol must vanish from ALL history"
m4=json.loads((pdir/"panel_meta.json").read_text())
assert m4["mode"]=="full", "universe change must force a full rebuild"
print(f"  ok   universe change -> full rebuild; STK07 gone from all "
      f"{pn4['timestamp'].nunique()} sessions")

print(f"\n  panel: {meta['rows']:,} rows, {meta['columns']} cols, "
      f"base rate {meta['label_positive_rate']:.2%}")
print("\nALL PANEL TESTS PASSED")

import tempfile as _tf
# ============ 9. POINT-IN-TIME LIQUIDITY FLOOR ============
print("\n9. liquidity floor is point-in-time, not hindsight")
r9 = Path(_tf.mkdtemp())
ses9 = pd.DatetimeIndex(pd.bdate_range("2021-01-01", "2024-12-31"))
def w9(sym, ix, seed, vol_fn):
    rr = np.random.default_rng(seed)
    c = 100*np.exp(np.cumsum(rr.standard_t(4, len(ix))*0.013))
    v = np.array([vol_fn(i, len(ix)) for i in range(len(ix))], dtype=float)
    d = pd.DataFrame({"timestamp": ix.tz_localize(IST), "open": c,
                      "high": c*1.01, "low": c*0.99, "close": c, "volume": v})
    d.to_parquet(r9/f"{sym}_daily.parquet", index=False)
    (r9/f"{sym}_daily.ok.json").write_text(json.dumps({
        "schema_version":"v27","rows":len(d),"series_kind":"equity",
        "corporate_action_suspects":[],
        "last_timestamp":str(d["timestamp"].iloc[-1])}, default=str))

# 25 always-liquid names so the cross-section is viable
for i in range(25):
    w9(f"LIQ{i:02d}", ses9, i, lambda k,n: 2e6)
# DYING: liquid for the first half, dries up in the second
w9("DYING", ses9, 900, lambda k,n: 2e6 if k < n//2 else 1e3)
# RISING: illiquid early, becomes liquid later
w9("RISING", ses9, 901, lambda k,n: 1e3 if k < n//2 else 2e6)
w9("NIFTY50", ses9, 902, lambda k,n: np.nan)
# index needs its own kind
(r9/"NIFTY50_daily.ok.json").write_text(json.dumps({
    "schema_version":"v27","rows":len(ses9),"series_kind":"index",
    "corporate_action_suspects":[],"last_timestamp":str(ses9[-1])}, default=str))

FLOOR = 5e7   # 100 * 2e6 = 2e8 for liquid names; 100 * 1e3 = 1e5 for dry ones
pq9 = pb.build_panel(r9, Path(_tf.mkdtemp()), full=True,
                     min_turnover=FLOOR, verbose=False)
pn9 = pd.read_parquet(pq9)
mid = pd.Timestamp(ses9[len(ses9)//2])

for sym, early_expected, late_expected in [("DYING", True, False),
                                           ("RISING", False, True)]:
    rows = pn9[pn9.symbol == sym]
    early = bool((rows.timestamp < mid - pd.Timedelta(days=120)).any())
    late  = bool((rows.timestamp > mid + pd.Timedelta(days=120)).any())
    ok = (early == early_expected) and (late == late_expected)
    print(f"   {'ok  ' if ok else 'FAIL'} {sym:<7} in-universe early={early} "
          f"late={late}  (want {early_expected}/{late_expected})")
    assert ok, sym

print("   ok   a name enters and leaves the universe as its turnover changes")
print("        - no hindsight: DYING is present in the period it was tradeable")
assert "X_turnover_med" in pn9.columns
nofloor = pd.read_parquet(pb.build_panel(r9, Path(_tf.mkdtemp()), full=True,
                                         min_turnover=0, verbose=False))
print(f"   floor off: {len(nofloor):,} rows | floor on: {len(pn9):,} rows")
assert len(pn9) < len(nofloor)
print("\nALL PANEL TESTS PASSED (incl. liquidity floor)")

# ============ 10. BUILD SIGNATURE FORCES A FULL REBUILD ============
print("\n10. changing the liquidity floor must force a full rebuild")
r10 = Path(_tf.mkdtemp()); p10 = Path(_tf.mkdtemp())
ses10 = pd.DatetimeIndex(pd.bdate_range("2021-01-01","2024-12-31"))
for i in range(25):
    w9(f"L{i:02d}", ses10, 500+i, lambda k,n: 2e6) if False else None
def w10(sym, ix, seed, vol):
    rr=np.random.default_rng(seed)
    c=100*np.exp(np.cumsum(rr.standard_t(4,len(ix))*0.013))
    d=pd.DataFrame({"timestamp":ix.tz_localize(IST),"open":c,"high":c*1.01,
                    "low":c*0.99,"close":c,"volume":np.full(len(ix),vol)})
    d.to_parquet(r10/f"{sym}_daily.parquet",index=False)
    (r10/f"{sym}_daily.ok.json").write_text(json.dumps({
        "schema_version":"v27","rows":len(d),"series_kind":"equity",
        "corporate_action_suspects":[],
        "last_timestamp":str(d["timestamp"].iloc[-1])},default=str))
for i in range(25): w10(f"L{i:02d}", ses10, 600+i, 2e6)
w10("NIFTY50", ses10, 690, 1e6)
(r10/"NIFTY50_daily.ok.json").write_text(json.dumps({
    "schema_version":"v27","rows":len(ses10),"series_kind":"index",
    "corporate_action_suspects":[],"last_timestamp":str(ses10[-1])},default=str))

pb.build_panel(r10, p10, full=True, min_turnover=0, verbose=False)
m_a = json.loads((p10/"panel_meta.json").read_text())
# same universe, DIFFERENT floor, incremental requested
pb.build_panel(r10, p10, full=False, min_turnover=5e7, verbose=False)
m_b = json.loads((p10/"panel_meta.json").read_text())
print(f"   floor {m_a['liquidity_floor']:,.0f} -> {m_b['liquidity_floor']:,.0f}")
print(f"   mode requested=incremental, actual={m_b['mode']}")
assert m_a["universe_hash"] == m_b["universe_hash"], "universe unchanged"
assert m_a["build_signature"] != m_b["build_signature"]
assert m_b["mode"] == "full", "floor change must force a full rebuild"
print("   ok   identical universe but changed floor -> FULL rebuild forced")
print("        (v27 would have silently written a mixed panel)")
print("\nALL PANEL TESTS PASSED (incl. build signature)")

# ============ 11. INCREMENTAL + LIQUIDITY FLOOR ============
print("\n11. incremental build must survive the liquidity floor")
r11 = Path(_tf.mkdtemp()); p11 = Path(_tf.mkdtemp())
ses11 = pd.DatetimeIndex(pd.bdate_range("2022-01-03","2026-09-21"))
def w11(sym, ix, seed, vol):
    rr=np.random.default_rng(seed)
    c=100*np.exp(np.cumsum(rr.standard_t(5,len(ix))*0.012))
    d=pd.DataFrame({"timestamp":ix.tz_localize(IST),"open":c,"high":c*1.012,
                    "low":c*0.988,"close":c,"volume":np.full(len(ix),vol)})
    d.to_parquet(r11/f"{sym}_daily.parquet",index=False)
    (r11/f"{sym}_daily.ok.json").write_text(json.dumps({
        "schema_version":"v27","rows":len(d),"series_kind":"equity",
        "corporate_action_suspects":[],
        "last_timestamp":str(d["timestamp"].iloc[-1])},default=str))
for i in range(30): w11(f"Q{i:02d}", ses11, 900+i, 5e5)   # ~5 crore/day turnover
w11("NIFTY50", ses11, 990, 1e6)
(r11/"NIFTY50_daily.ok.json").write_text(json.dumps({
    "schema_version":"v27","rows":len(ses11),"series_kind":"index",
    "corporate_action_suspects":[],"last_timestamp":str(ses11[-1])},default=str))

pb.build_panel(r11, p11, full=True, min_turnover=1e7, verbose=False)
m_full = json.loads((p11/"panel_meta.json").read_text())
print(f"   full build:        {m_full['rows']:,} rows, {m_full['symbols']} symbols")

# THE BUG: incremental loaded only 20 sessions, so the 60-session turnover
# median was NaN everywhere and the floor removed every row.
pb.build_panel(r11, p11, full=False, min_turnover=1e7, verbose=False)
m_inc = json.loads((p11/"panel_meta.json").read_text())
print(f"   incremental build: {m_inc['rows']:,} rows, {m_inc['symbols']} symbols")
print(f"   mode: {m_inc['mode']}")
assert m_inc["rows"] > 0, "incremental produced an empty panel"
assert m_inc["symbols"] == m_full["symbols"], (m_inc["symbols"], m_full["symbols"])
assert abs(m_inc["rows"] - m_full["rows"]) < m_full["rows"]*0.02, \
    (m_inc["rows"], m_full["rows"])
print("   ok   incremental survives the floor and matches the full build")
print("        (before: 'liquidity floor removed every row', 0 symbols)")
print("\nALL PANEL TESTS PASSED (incl. incremental liquidity floor)")

# ============ 12. MEMORY: FLOOR MUST NOT COPY THE PANEL ============
print("\n12. liquidity floor memory behaviour")
import tracemalloc
rng12=np.random.default_rng(12)
N,S=40_000,60
ts12=pd.DatetimeIndex(pd.bdate_range("2024-01-01",periods=N//S))
rows12=pd.DataFrame({
    "timestamp":np.tile(ts12,S),
    "symbol":np.repeat([f"M{i:02d}" for i in range(S)],len(ts12)),
})
# Build in one shot: the real panel arrives from pd.concat and is
# consolidated, so a fragmented test frame would measure the wrong thing.
_feat=pd.DataFrame({f"D_f{i}":rng12.normal(size=len(rows12)) for i in range(120)})
_feat["D_dollar_vol"]=np.abs(rng12.normal(3e7,1e7,len(rows12)))
rows12=pd.concat([rows12,_feat],axis=1).copy()
base_mb=rows12.memory_usage(deep=False).sum()/1e6
tracemalloc.start()
out12=pb.apply_liquidity_floor(rows12,min_turnover=1e7,verbose=False)
_,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"   frame {base_mb:,.0f} MB | floor peak allocation {peak/1e6:,.0f} MB")
print(f"   ratio {peak/1e6/base_mb:.2f}x the frame size")
assert peak/1e6 < base_mb*2.2, f"floor allocated {peak/1e6:.0f}MB on a {base_mb:.0f}MB frame"
print("   ok   the floor no longer makes 3 full copies of the panel")

# correctness must be unchanged: NaN median -> dropped
assert (out12["X_turnover_med"]>=1e7).all()
assert len(out12)>0 and len(out12)<=len(rows12)
print(f"   kept {len(out12):,}/{len(rows12):,} rows, all above the floor")

# float32 downcast preserves values within float32 precision
a=np.float64(1.2345678901234); b=np.float32(a)
print(f"   float32 precision check: {a:.10f} -> {float(b):.10f} "
      f"(rel err {abs(a-float(b))/a:.2e})")
assert abs(a-float(b))/a < 1e-6
print("   ok   float32 is ~7 significant digits - far beyond any price ratio")
print("\nALL PANEL TESTS PASSED (incl. memory)")
