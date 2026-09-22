#!/usr/bin/env python3
"""
daily_report.py - aggregate what the pipeline already wrote into one verdict.

    python daily_report.py --root <cache_root>

DESIGN RULE: THIS FILE COMPUTES NOTHING
=======================================
Every number here is read from an artefact another stage already produced:

    cache_coverage_report.json   stale, failed, unresolved, CA suspects
    quarantine.json              requested / allowed / quarantined + reasons
    panel_meta.json              rows, symbols, dates, labels, base rate,
                                 build signature, liquidity floor

Duplicating a check here would create a second implementation that can
disagree with the first, and then nobody knows which is right. data_quality
is the GATE; this is the dashboard that reads it.

Writes:
    reports/<date>_daily_report.json   machine-readable
    reports/<date>_daily_report.html   human dashboard
    reports/latest_daily_report.html   double-click this one
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OK, WARN, BAD = "OK", "WARN", "FAIL"


def _read(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _num(d: Optional[dict], *keys, default=None):
    """First present key, tolerating the different shapes these files use."""
    if not d:
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default


def collect(root: Path) -> dict:
    root = Path(root)
    cov = _read(root / "cache_coverage_report.json")
    qua = _read(root / "quarantine.json")
    pmeta = _read(root / "panel" / "panel_meta.json")

    checks: List[Tuple[str, object, str, str]] = []

    def add(name, value, status, note=""):
        checks.append((name, value, status, note))

    # ---- cache ----
    if cov is None:
        add("Cache coverage report", "missing", BAD,
            "cache_coverage_report.json not found - did the cache stage run?")
    else:
        for label, keys in (("Stale symbols", ("stale", "stale_symbols")),
                            ("Failed fetches", ("failed", "failures")),
                            ("Unresolved symbols", ("unresolved",
                                                    "unresolved_symbols")),
                            ("Corporate-action suspects",
                             ("corporate_action_suspects", "ca_suspects"))):
            v = _num(cov, *keys, default=None)
            n = len(v) if isinstance(v, (list, dict)) else (v or 0)
            add(label, n, OK if not n else WARN)

    # ---- gate ----
    if qua is None:
        add("Quality gate report", "missing", BAD,
            "quarantine.json not found - the gate did not run")
    else:
        req = _num(qua, "requested", default=None)
        allowed = _num(qua, "allowed", default=None)
        quar = _num(qua, "quarantined", default=None)
        unver = _num(qua, "unverified", default=None)
        n_req = len(req) if isinstance(req, list) else (req or 0)
        n_all = len(allowed) if isinstance(allowed, list) else (allowed or 0)
        n_q = len(quar) if isinstance(quar, (list, dict)) else (quar or 0)
        n_u = len(unver) if isinstance(unver, (list, dict)) else (unver or 0)
        add("Symbols requested", n_req, OK)
        add("Symbols cleared", n_all, OK if n_all else BAD)
        add("Quarantined", n_q, OK if not n_q else WARN,
            "see quarantine.json for per-symbol reasons")
        add("Unverified", n_u, OK if not n_u else WARN)

    # ---- panel ----
    if pmeta is None:
        add("Panel", "NOT BUILT", BAD,
            "panel_meta.json not found - the panel stage did not complete")
    else:
        rows = _num(pmeta, "rows", default=0)
        syms = _num(pmeta, "symbols", default=0)
        add("Panel rows", f"{rows:,}" if isinstance(rows, int) else rows,
            OK if rows else BAD)
        add("Panel symbols", syms, OK if syms else BAD)
        add("Panel range",
            f"{_num(pmeta, 'start', 'first_date', default='?')} -> "
            f"{_num(pmeta, 'end', 'last_date', default='?')}", OK)
        add("Build mode", _num(pmeta, "mode", default="?"), OK)
        add("Liquidity floor",
            f"{_num(pmeta, 'liquidity_floor', default=0):,.0f}/day", OK)
        sig = _num(pmeta, "build_signature", default=None)
        add("Build signature", sig or "absent", OK if sig else WARN,
            "" if sig else "panel predates signature tracking - rebuild full")
        br = _num(pmeta, "base_rate", default=None)
        if br is not None:
            add("Base rate", f"{float(br):.3%}", OK)
        pend = _num(pmeta, "labels_pending", default=None)
        if pend is not None:
            add("Labels pending", pend, OK,
                "the most recent sessions are UNKNOWN, not negative")

    # ---- leak gate: report only what we can actually verify ----
    lg = root / "feature_leak_gate.json"
    lgd = _read(lg)
    if lgd is None:
        add("Feature leak gate", "NOT RECORDED", WARN,
            "run_daily writes this; a manual panel build does not")
    else:
        passed = bool(lgd.get("passed"))
        add("Feature leak gate", "PASS" if passed else "FAIL",
            OK if passed else BAD, lgd.get("detail", ""))

    worst = BAD if any(c[2] == BAD for c in checks) else (
        WARN if any(c[2] == WARN for c in checks) else OK)
    return {"generated": dt.datetime.now().isoformat(), "root": str(root),
            "verdict": worst, "checks": [
                {"name": n, "value": v, "status": s, "note": note}
                for n, v, s, note in checks]}


_CSS = """
body{background:#11161d;color:#d6dde6;font:14px/1.6 ui-monospace,Menlo,Consolas,monospace;margin:0;padding:28px}
h1{font-size:18px;letter-spacing:.08em;margin:0 0 4px}
.sub{color:#7d8896;margin-bottom:22px}
table{border-collapse:collapse;width:100%;max-width:820px}
td{padding:7px 12px;border-bottom:1px solid #1e2732}
td.v{text-align:right;color:#fff}
.note{color:#6e7a8a;font-size:12px}
.badge{display:inline-block;min-width:52px;text-align:center;padding:2px 8px;border-radius:3px;font-size:12px}
.OK{background:#14361f;color:#5fd68a}.WARN{background:#3a3212;color:#e0c24a}.FAIL{background:#3d1717;color:#ef6d6d}
.verdict{margin:26px 0;padding:16px 20px;border-radius:5px;max-width:820px;font-size:16px}
.vOK{background:#14361f;color:#7ae5a3}.vWARN{background:#3a3212;color:#f0d878}.vFAIL{background:#3d1717;color:#ff8686}
"""

_VERDICT = {
    OK: "RUN COMPLETE - DATA SAFE",
    WARN: "RUN COMPLETE WITH WARNINGS - inspect before trusting the panel",
    BAD: "RUN BLOCKED - the panel is not safe to use",
}


def to_html(rep: dict) -> str:
    rows = []
    for c in rep["checks"]:
        note = (f'<div class="note">{html.escape(str(c["note"]))}</div>'
                if c["note"] else "")
        rows.append(
            f'<tr><td>{html.escape(c["name"])}{note}</td>'
            f'<td class="v">{html.escape(str(c["value"]))}</td>'
            f'<td><span class="badge {c["status"]}">{c["status"]}</span></td></tr>')
    v = rep["verdict"]
    return (f'<!doctype html><meta charset="utf-8">'
            f'<title>Daily pipeline report</title><style>{_CSS}</style>'
            f'<h1>NSE QUANT - DAILY PIPELINE REPORT</h1>'
            f'<div class="sub">{html.escape(rep["generated"][:19])} &nbsp;|&nbsp; '
            f'{html.escape(rep["root"])}</div>'
            f'<div class="verdict v{v}">{_VERDICT[v]}</div>'
            f'<table>{"".join(rows)}</table>'
            f'<div class="sub" style="margin-top:24px">This report reads '
            f'artefacts written by the cache, gate and panel stages. It '
            f'recomputes nothing - data_quality.py is the gate, this is the '
            f'dashboard.</div>')


def build(root, *, started=None, verbose: bool = True) -> Path:
    root = Path(root)
    rep = collect(root)
    d = root / "reports"
    d.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().isoformat()
    (d / f"{day}_daily_report.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    h = to_html(rep)
    (d / f"{day}_daily_report.html").write_text(h, encoding="utf-8")
    latest = d / "latest_daily_report.html"
    latest.write_text(h, encoding="utf-8")

    if verbose:
        print("\n" + "=" * 62)
        print("  HYGIENE SUMMARY")
        print("=" * 62)
        for c in rep["checks"]:
            mark = {"OK": "  ok ", "WARN": "  !! ", "FAIL": "  XX "}[c["status"]]
            print(f"{mark}{c['name']:<34}{str(c['value']):>18}")
        print("=" * 62)
        print(f"  {_VERDICT[rep['verdict']]}")
        print(f"  {latest}")
        print("=" * 62, flush=True)
    return latest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = a.root or os.environ.get("CACHE_DAILY_ROOT")
    if not root:
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    rep = collect(Path(root))
    build(Path(root))
    return 0 if rep["verdict"] != BAD else 1


if __name__ == "__main__":
    raise SystemExit(main())
