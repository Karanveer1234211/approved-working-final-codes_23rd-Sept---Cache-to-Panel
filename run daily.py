#!/usr/bin/env python3
"""
run_daily.py - the daily pipeline. Cache -> gate -> leak check -> panel.

    python run_daily.py --symbols-file watchlist.txt --only-stale
    python run_daily.py --symbols-file watchlist.txt --full --min-turnover 1e7

WHAT CHANGED IN v2, AND WHY IT MATTERS
======================================
The previous version did not do what its name said.

  * run_panel() was a STUB. It printed "not wired up yet" and returned, so
    the scheduled job fetched data, gated it, built NO panel - and exited 0,
    so nothing ever complained. Every panel to date was built by hand.
  * It imported Daily_cache_v24, two versions behind. None of the orphan-bar
    trimming, session clamping or root-guard work was in the scheduled path.
  * features_daily.assert_leak_free() appeared only inside a DOCSTRING, as
    example text. The leak gate has never run in the daily job.

A pipeline that exits 0 without doing its job is worse than one that fails,
because nothing prompts anyone to look. Every stage below does its work or
raises.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CACHE_SCRIPT = HERE / "Daily_cache_v27.py"


def _hdr(title: str) -> None:
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62, flush=True)


def _stage(n: int, total: int, name: str) -> float:
    print(f"\n[{n}/{total}] {name}", flush=True)
    return time.perf_counter()


def _done(t0: float, msg: str) -> str:
    el = time.perf_counter() - t0
    m, s = divmod(int(el), 60)
    print(f"      OK  {msg}  ({m:02d}:{s:02d})", flush=True)
    return f"{m:02d}:{s:02d}"


def _fail(msg: str, detail: str = "") -> None:
    print(f"\n      FAILED  {msg}", flush=True)
    if detail:
        print(detail, flush=True)


def run_cache(symbols_file: str, extra: List[str]) -> None:
    t0 = _stage(1, 4, "CACHE")
    if not CACHE_SCRIPT.exists():
        raise SystemExit(f"cache script missing: {CACHE_SCRIPT}")
    cmd = [sys.executable, str(CACHE_SCRIPT), "--symbols-file", symbols_file] + extra
    print(f"      {' '.join(cmd[1:])}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        _fail(f"cache build exited {r.returncode}")
        raise SystemExit(r.returncode)
    _done(t0, "cache updated")


def run_gate(root: Path, symbols: List[str], *, allow_partial: bool) -> List[str]:
    t0 = _stage(2, 4, "DATA QUALITY GATE")
    from data_quality import assert_panel_ready, DataQualityError
    print(f"      auditing {len(symbols)} series", flush=True)
    try:
        allowed = assert_panel_ready(root, symbols, allow_partial=allow_partial)
    except DataQualityError as e:
        _fail("quarantine - not building a panel on this data", str(e))
        raise SystemExit(
            f"\nInspect {root / 'quarantine.json'}, then either repair the "
            f"cache (--force on the affected symbols) or rerun with "
            f"--allow-partial if a thinner universe is acceptable today.")
    _done(t0, f"{len(allowed)}/{len(symbols)} cleared")
    return allowed


def run_leak_gate(root: Path) -> None:
    """
    The static feature-contract check, ONCE per process, before any feature
    is computed.

    It proves no feature reads a bar it could not have seen. It existed and
    was documented; it was never called outside a docstring. Running it here
    means a contract violation stops the pipeline instead of silently
    producing a leaky panel.
    """
    t0 = _stage(3, 4, "FEATURE LEAK GATE")
    import features_daily as fx
    rec = {"checked_at": dt.datetime.now().isoformat()}
    try:
        fx.assert_leak_free()
        rec.update(passed=True, detail="static feature contract verified")
    except Exception as e:
        rec.update(passed=False, detail=str(e)[:400])
        (root / "feature_leak_gate.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8")
        _fail("feature contract violated - NOT building a panel", str(e))
        raise SystemExit(1)
    # The report reads this file. Without it the dashboard says NOT RECORDED
    # rather than claiming a PASS it cannot substantiate.
    (root / "feature_leak_gate.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    _done(t0, "feature contract verified")


def run_panel(root: Path, *, full: bool, min_turnover: float,
              allow_partial: bool) -> Dict:
    """Actually build the panel. Previously a stub that printed and returned."""
    t0 = _stage(4, 4, "PANEL")
    import panel_build as PB
    panel_dir = root / "panel"
    PB.build_panel(root, panel_dir, full=full, min_turnover=min_turnover,
                   allow_partial=allow_partial, verbose=True)
    meta_p = panel_dir / "panel_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    _done(t0, f"{meta.get('rows', '?')} rows | {meta.get('symbols', '?')} symbols")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-file", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="force a full panel rebuild")
    ap.add_argument("--min-turnover", type=float, default=1e7)
    ap.add_argument("--no-report", action="store_true")
    args, extra = ap.parse_known_args()

    started = dt.datetime.now()
    root_env = os.environ.get("CACHE_DAILY_ROOT")
    root = Path(args.root) if args.root else (Path(root_env) if root_env else None)
    if root is None:
        raise SystemExit(
            "CACHE_DAILY_ROOT is not set and --root was not given.\n"
            "  Run:  call env_daily.bat\n"
            "There is deliberately no default - guessing a cache path "
            "silently corrupts every downstream artefact.")

    _hdr("NSE QUANT DAILY PIPELINE")
    print(f"  Run:   {started:%d-%b-%Y %H:%M}")
    print(f"  Root:  {root}")
    print(f"  Mode:  {'FULL REBUILD' if args.full else 'INCREMENTAL'}")
    print(f"  Floor: {args.min_turnover:,.0f}/day", flush=True)

    if not args.skip_cache:
        run_cache(args.symbols_file, extra)
    else:
        print("\n[1/4] CACHE  skipped (--skip-cache)", flush=True)

    from data_quality import discover_symbols
    symbols = discover_symbols(root)
    if not symbols:
        raise SystemExit(f"no cached symbols under {root}")

    run_gate(root, symbols, allow_partial=args.allow_partial)
    run_leak_gate(root)
    run_panel(root, full=args.full, min_turnover=args.min_turnover,
              allow_partial=args.allow_partial)

    if not args.no_report:
        try:
            import daily_report
            daily_report.build(root, started=started)
        except Exception as e:
            print(f"\n  report generation failed ({e}); the pipeline itself "
                  f"is unaffected", flush=True)

    _hdr("RUN COMPLETE")
    el = (dt.datetime.now() - started).total_seconds()
    print(f"  Total: {int(el // 60):02d}:{int(el % 60):02d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
