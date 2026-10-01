"""
audit.py
----------
Phase 1 + Phase 83: an audit that RUNS rather than describes.

Two halves:

  STATIC   scans every module for the patterns that cause the bugs this project
           actually suffers from -- data fetched outside the loader, caches
           whose keys cannot distinguish two different requests, live quotes
           touching historical frames, naive/aware timestamp mixing.

  RUNTIME  asks the live code the questions a user would: does a historical
           date return the price of that date, do two different dates return
           different prices, do two tickers stay separate, does a timeframe
           change the data.

Anything it finds is printed as a finding with a severity, and the exit code is
non-zero when a CRITICAL finding exists, so it can gate a release.

Run:  python audit.py
"""

import ast
import datetime
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
findings = []


def finding(severity, phase, where, what, detail=""):
    findings.append({"severity": severity, "phase": phase, "where": where,
                     "what": what, "detail": detail})


# ---------------------------------------------------------------- static scan
SKIP_FILES = {"audit.py", "test_timeframes.py", "model_lab.py", "lab_configs.py"}
DATA_FETCH = re.compile(r"yf\.(download|Ticker)\s*\(")
# A price-bar fetch is the thing Phase 3 cares about, because bars have a timeframe and
# an as-of moment. Fetching .info / .news / earnings dates is metadata: it has no
# timeframe, so routing it through the price engine would buy nothing. The two are
# reported separately rather than lumped together as one number.
BAR_FETCH = re.compile(r"yf\.download\s*\(|\.history\s*\(")
CACHE_DECOR = re.compile(r"@st\.cache_(data|resource)")


def static_scan():
    print("=" * 78)
    print("STATIC SCAN")
    print("=" * 78)
    py = sorted(f for f in os.listdir(HERE) if f.endswith(".py") and f not in SKIP_FILES)
    fetchers = {}
    for fname in py:
        src = open(os.path.join(HERE, fname), encoding="utf-8", errors="replace").read()
        lines = src.split("\n")

        # 1. who fetches market data directly?
        hits = [i + 1 for i, ln in enumerate(lines) if DATA_FETCH.search(ln)]
        bars = [i + 1 for i, ln in enumerate(lines) if BAR_FETCH.search(ln)]
        if hits:
            fetchers[fname] = hits
            if fname not in ("data_loader.py", "market_data.py"):
                if bars:
                    finding("MEDIUM", "3", f"{fname}:{bars[0]}",
                            "fetches PRICE BARS directly instead of going through the data engine",
                            f"{len(bars)} bar call site(s) -- these carry a timeframe and an "
                            f"as-of moment, so they belong behind market_data.get_market_data()")
                meta = [h for h in hits if h not in bars]
                if meta:
                    finding("INFO", "3", f"{fname}:{meta[0]}",
                            "fetches metadata directly (.info / .news / earnings / fundamentals)",
                            f"{len(meta)} call site(s); metadata has no timeframe, so the price "
                            f"engine would add nothing -- listed for completeness only")

        # 2. cached functions whose key cannot separate two different requests
        for i, ln in enumerate(lines):
            if CACHE_DECOR.search(ln):
                sig = ""
                for j in range(i + 1, min(i + 6, len(lines))):
                    sig += lines[j]
                    if "):" in lines[j]:
                        break
                if "def " not in sig:
                    continue
                name = sig.split("def ")[1].split("(")[0]
                args = sig[sig.index("(") + 1:sig.rindex(")")] if "(" in sig and ")" in sig else ""
                if re.search(r"\b(ticker|symbol)\b", args) and "interval" not in args \
                        and "tf" not in args and "timeframe" not in args:
                    if any(k in name.lower() for k in ("load", "bars", "ohlc", "data", "engine",
                                                       "quote", "candle")):
                        # An exemption is allowed, but only where it is written down at the
                        # definition itself, and it is still reported so it stays visible.
                        exempt = "audit-exempt" in sig
                        finding("INFO" if exempt else "HIGH", "8", f"{fname}:{i + 1}",
                                f"cache key for {name}() has no timeframe component"
                                + (" (exempt: declared timeframe-free)" if exempt else ""),
                                f"args: {args.strip()[:90]}")

        # 3. live quote touching anything that looks historical
        for i, ln in enumerate(lines):
            if "apply_live_quote" in ln and "def " not in ln:
                finding("INFO", "2", f"{fname}:{i + 1}",
                        "live quote is overlaid onto a price frame here",
                        "must never happen when the frame is historical")

        # 4. naive/aware mixing
        if "tz_localize" in src and "tz_convert" in src and fname != "data_loader.py":
            finding("LOW", "7", fname, "handles timezones locally",
                    "Phase 7 wants one central timezone layer")
    print(f"scanned {len(py)} modules; direct data fetchers: {', '.join(fetchers) or 'none'}")


# ---------------------------------------------------------------- runtime
def runtime_checks():
    print("\n" + "=" * 78)
    print("RUNTIME CHECKS")
    print("=" * 78)
    import data_loader as dl

    ticker = "HDFCBANK.NS"
    today = datetime.date.today()

    # --- TEST 1: a historical end date must not return today's price --------
    hist_end = (today - datetime.timedelta(days=45))
    while hist_end.weekday() >= 5:
        hist_end -= datetime.timedelta(days=1)
    hist = dl.load_stock_data(ticker, "2024-01-01", hist_end.strftime("%Y-%m-%d"))
    live = dl.load_stock_data(ticker, "2024-01-01", today.strftime("%Y-%m-%d"))
    h_last, h_px = hist.index[-1], float(hist["Close"].iloc[-1])
    l_last, l_px = live.index[-1], float(live["Close"].iloc[-1])
    print(f"\nTEST 1 historical vs current frame")
    print(f"  history ending {hist_end}: last candle {h_last.date()} close {h_px:.2f}")
    print(f"  history ending today     : last candle {l_last.date()} close {l_px:.2f}")
    ok = h_last.date() <= hist_end and abs(h_px - l_px) > 1e-9
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        finding("CRITICAL", "2", "data_loader.load_stock_data",
                "a historical end date returns today's last candle/price", f"{h_last} vs {l_last}")

    # --- TEST 2: different historical dates give different prices ----------
    print(f"\nTEST 2 two different historical dates")
    d1 = (today - datetime.timedelta(days=120))
    d2 = (today - datetime.timedelta(days=60))
    p1 = dl.load_stock_data(ticker, "2024-01-01", d1.strftime("%Y-%m-%d"))
    p2 = dl.load_stock_data(ticker, "2024-01-01", d2.strftime("%Y-%m-%d"))
    v1, v2 = float(p1["Close"].iloc[-1]), float(p2["Close"].iloc[-1])
    print(f"  ...{d1}: {v1:.2f}   ...{d2}: {v2:.2f}")
    ok = abs(v1 - v2) > 1e-9 and p1.index[-1] < p2.index[-1]
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        finding("CRITICAL", "2", "data_loader.load_stock_data",
                "different historical end dates return the same last price", f"{v1} vs {v2}")

    # --- TEST 3: the dashboard's live overlay must refuse historical frames -
    print(f"\nTEST 3 live-quote overlay against a historical frame")
    try:
        sys.path.insert(0, HERE)
        from market_data import apply_live_quote_safe
        quote = {"price": 99999.0, "time": int(datetime.datetime.now().timestamp()),
                 "tz": "Asia/Kolkata", "prev_close": 1.0, "day_high": 1.0, "day_low": 1.0,
                 "volume": 1, "open": 1.0}
        out, _, used = apply_live_quote_safe(hist, quote, is_historical=True)
        ok = (not used) and float(out["Close"].iloc[-1]) == h_px
        print(f"  overlay applied to historical frame: {used} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            finding("CRITICAL", "2", "market_data.apply_live_quote_safe",
                    "live quote mutated a historical frame")
    except Exception as e:
        print(f"  guard not present yet ({type(e).__name__})")
        finding("CRITICAL", "2", "dashboard.apply_live_quote",
                "no guard preventing the live quote from being written into a historical frame",
                str(e)[:80])

    # --- TEST 4: no cross-ticker contamination -----------------------------
    print(f"\nTEST 4 cross-ticker separation")
    a = dl.load_stock_data("HDFCBANK.NS", "2025-01-01", today.strftime("%Y-%m-%d"))
    b = dl.load_stock_data("ICICIBANK.NS", "2025-01-01", today.strftime("%Y-%m-%d"))
    ok = abs(float(a["Close"].iloc[-1]) - float(b["Close"].iloc[-1])) > 1e-9
    print(f"  HDFCBANK {float(a['Close'].iloc[-1]):.2f} vs ICICIBANK {float(b['Close'].iloc[-1]):.2f}"
          f" -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        finding("CRITICAL", "8", "cache", "two tickers returned the same prices")

    # --- TEST 5: timeframe actually changes the data -----------------------
    print(f"\nTEST 5 timeframe separation")
    try:
        d15 = dl.load_ohlcv(ticker, "15m")
        d1d = dl.load_ohlcv(ticker, "1d")
        ok = len(d15) != len(d1d)
        print(f"  15m {len(d15):,} bars vs 1d {len(d1d):,} bars -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            finding("CRITICAL", "5", "data_loader.load_ohlcv", "timeframes return identical data")
    except Exception as e:
        print(f"  FAIL ({type(e).__name__}: {e})")
        finding("HIGH", "5", "data_loader.load_ohlcv", "interval loading failed", str(e)[:80])

    # --- TEST 6: targets must not see their own future ---------------------
    print(f"\nTEST 6 target construction")
    import forecast_engine as fe
    table, groups = fe.build_feature_table(live, ticker)
    feat_cols = sum(groups.values(), [])
    worst = None
    for h in (5, 21):
        tcol = f"Fwd_Ret_{h}"
        if tcol not in table:
            continue
        corr = table[feat_cols].corrwith(table[tcol]).abs().sort_values(ascending=False)
        top, val = corr.index[0], float(corr.iloc[0])
        print(f"  h={h:>3} strongest feature/target correlation: {top} = {val:.3f}")
        if val > 0.5:
            worst = (h, top, val)
    if worst:
        finding("CRITICAL", "83", "feature table",
                "a feature is almost perfectly correlated with the target (leakage)",
                f"h={worst[0]} {worst[1]} r={worst[2]:.3f}")
    print(f"  -> {'FAIL' if worst else 'PASS'}")

    # --- TEST 7: the macro basket must obey the as-of moment too -------------
    print(f"\nTEST 7 macro/market data respects the as-of date")
    cutoff = pd.Timestamp(hist_end)
    try:
        extra = fe.load_extra_data(ticker, "2026-01-01", hist_end.strftime("%Y-%m-%d"))
        ok = True
        for key in ("macro", "market"):
            frame = extra.get(key)
            if frame is None or len(frame) == 0:
                print(f"  {key}: unavailable (not fatal)")
                continue
            last = pd.Timestamp(frame.index[-1])
            good = last.normalize() <= cutoff.normalize()
            ok = ok and good
            print(f"  {key}: last observation {last.date()} vs cutoff {cutoff.date()}"
                  f" -> {'PASS' if good else 'FAIL'}")
            if not good:
                finding("CRITICAL", "3", f"forecast_engine.load_extra_data[{key}]",
                        "external data contains observations after the as-of moment",
                        f"last {last} > cutoff {cutoff}")
    except Exception as e:
        print(f"  could not check ({type(e).__name__}: {e})")
        finding("LOW", "3", "forecast_engine.load_extra_data",
                "as-of check could not run", str(e)[:80])


def report():
    print("\n" + "=" * 78)
    print("FINDINGS")
    print("=" * 78)
    if not findings:
        print("none")
        return 0
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    for f in sorted(findings, key=lambda x: order.get(x["severity"], 9)):
        print(f"[{f['severity']:8}] phase {f['phase']:>3} | {f['where']}")
        print(f"            {f['what']}")
        if f["detail"]:
            print(f"            {f['detail']}")
    crit = sum(1 for f in findings if f["severity"] == "CRITICAL")
    print(f"\n{len(findings)} finding(s), {crit} critical")
    return 1 if crit else 0


if __name__ == "__main__":
    static_scan()
    try:
        runtime_checks()
    except Exception as e:
        print(f"\nruntime checks aborted: {type(e).__name__}: {e}")
        finding("HIGH", "1", "audit", "runtime checks could not complete", str(e)[:120])
    sys.exit(report())
