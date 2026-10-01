"""
exp_selectivity.py
------------------
All 24 paper trades lost. Can that be fixed?

Three things are already measured and constrain what "fixing" can mean:

  * a blind same-day bracket has expectancy -0.00% at ZERO cost -- the entry has
    no edge, the round trip is the whole loss
  * conditioning on the model's direction signal made it WORSE (0 of 6 symbols
    profitable)
  * no target setting beats simply holding to the close

So more modelling on every day is not the lever. The lever, if one exists, is
TRADING LESS: taking only the days where the signal is strong enough to pay for
the round trip, and standing aside on the rest. That is the abstention idea the
spec calls meta-labelling, and it has never been tested here.

This measures it honestly:

  * out-of-fold predictions only -- every prediction is made by a model that
    never saw that day
  * signal strength is bucketed by decile, and expectancy measured per bucket
  * the cost is corrected to a realistic INTRADAY round trip (0.082% in charges
    plus slippage = 0.12-0.18% all-in) instead of the 0.30% delivery rate used
    before, which over-charged every trade by roughly 2x
  * exits compared: hold to close vs stop-only vs the full bracket
  * the decile threshold is chosen on TRAINING folds and applied to a held-out
    tail, so the "only trade the best days" rule is not fitted on its own answer

If no bucket is positive, that is the answer and it gets reported as one.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data_loader as dl
import forecast_engine as fe
import market_data as md
import models_tabular as mt
import options_plan as opl

UNIVERSE = ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "ITC.NS", "SBIN.NS",
            "MSFT", "AAPL", "JPM", "KO"]
COST_INTRADAY = 0.15          # realistic all-in, see the cost breakdown
N_FOLDS = 5
STOP_ATR = 0.8


def oof_frame(ticker):
    """Out-of-fold direction predictions plus each session's outcomes."""
    raw = dl.load_stock_data(ticker, "2014-01-01", "2026-09-29")
    raw, _ = md.repair_market_data(raw, ticker)
    table, groups = fe.build_feature_table(raw, ticker)
    tcol = "Fwd_Ret_1"
    if tcol not in table:
        return None
    d = table.dropna(subset=[tcol])
    cols = [c for c in sum(groups.values(), []) if c in d.columns and c != tcol]
    if len(d) < 800:
        return None

    n = len(d)
    start = int(n * 0.4)
    size = (n - start) // N_FOLDS
    preds = pd.Series(np.nan, index=d.index)
    for k in range(N_FOLDS):
        a = start + k * size
        b = start + (k + 1) * size if k < N_FOLDS - 1 else n
        tr, te = d.iloc[:a], d.iloc[a:b]
        if len(tr) < 400 or len(te) < 20:
            continue
        sc = StandardScaler().fit(tr[cols].values)
        m = mt.build_random_forest_regularized()
        m.fit(sc.transform(tr[cols].values), tr[tcol].values)
        preds.iloc[a:b] = m.predict(sc.transform(te[cols].values))

    atr_prev = opl.atr(raw).shift(1)
    o, h, l, c = (raw[x].astype(float) for x in ("Open", "High", "Low", "Close"))
    out = pd.DataFrame({"pred": preds}).dropna()
    ix = out.index
    out["open"], out["high"], out["low"], out["close"] = o[ix], h[ix], l[ix], c[ix]
    out["atr"] = atr_prev[ix]
    out = out[out["atr"].notna() & (out["atr"] > 0)]
    out["ticker"] = ticker

    long = out["pred"] > 0
    # hold to close, taking the side the signal points
    oc = (out["close"] / out["open"] - 1) * 100
    out["close_only"] = np.where(long, oc, -oc)
    # stop-only: same, but cut at the stop if the session hit it
    stop = np.where(long, out["open"] - STOP_ATR * out["atr"],
                    out["open"] + STOP_ATR * out["atr"])
    hit_s = np.where(long, out["low"] <= stop, out["high"] >= stop)
    ex = np.where(hit_s, stop, out["close"])
    out["stop_only"] = np.where(long, (ex / out["open"] - 1), (1 - ex / out["open"])) * 100
    out["abs_signal"] = out["pred"].abs()
    return out


def main():
    frames = []
    for t in UNIVERSE:
        try:
            f = oof_frame(t)
            if f is not None and len(f):
                frames.append(f)
                print(f"  {t:<13} {len(f):,} out-of-fold sessions")
        except Exception as e:
            print(f"  {t:<13} skipped ({type(e).__name__}: {str(e)[:45]})")
    if not frames:
        print("no data")
        return 1
    df = pd.concat(frames, ignore_index=True)

    print(f"\n{'=' * 90}")
    print(f"EXPECTANCY BY SIGNAL STRENGTH  (out-of-fold, {len(df):,} sessions, "
          f"{COST_INTRADAY:.2f}% round trip)")
    print("=" * 90)
    df["decile"] = pd.qcut(df["abs_signal"], 10, labels=False, duplicates="drop")
    print(f"{'decile':>7}{'sessions':>10}{'avg |signal|':>14}"
          f"{'hold-to-close':>16}{'stop-only':>12}{'win rate':>11}")
    best = None
    for dcl, g in df.groupby("decile"):
        cc = g["close_only"].mean() - COST_INTRADAY
        so = g["stop_only"].mean() - COST_INTRADAY
        wr = ((g["close_only"] - COST_INTRADAY) > 0).mean() * 100
        print(f"{int(dcl) + 1:>7}{len(g):>10,}{g['abs_signal'].mean() * 100:>13.3f}%"
              f"{cc:>+15.3f}%{so:>+11.3f}%{wr:>10.1f}%")
        if best is None or cc > best[1]:
            best = (int(dcl) + 1, cc)
    print(f"\n  best decile: {best[0]} at {best[1]:+.3f}% per trade")
    print(f"  all sessions: {df['close_only'].mean() - COST_INTRADAY:+.3f}% per trade")

    print(f"\n{'=' * 90}\nCOST SENSITIVITY — top-decile days only, hold to close\n{'=' * 90}")
    top = df[df["decile"] == df["decile"].max()]
    for c in (0.0, 0.08, 0.12, 0.15, 0.18, 0.30):
        e = top["close_only"].mean() - c
        print(f"  cost {c:.2f}%  ->  {e:+.3f}% per trade   "
              f"({'PROFITABLE' if e > 0 else 'loses'})")

    print(f"\n{'=' * 90}\nHELD-OUT TEST — threshold picked on the first 70%, applied to the last 30%\n{'=' * 90}")
    rows = []
    for t, g in df.groupby("ticker"):
        g = g.sort_index()
        cut = int(len(g) * 0.7)
        tr, te = g.iloc[:cut], g.iloc[cut:]
        if len(te) < 100:
            continue
        thr = tr["abs_signal"].quantile(0.9)          # chosen on training only
        sel = te[te["abs_signal"] >= thr]
        if len(sel) < 20:
            continue
        rows.append({
            "ticker": t, "held_out": len(te), "taken": len(sel),
            "taken_pct": len(sel) / len(te) * 100,
            "selective": sel["close_only"].mean() - COST_INTRADAY,
            "everything": te["close_only"].mean() - COST_INTRADAY,
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        print(f"{'ticker':<13}{'held-out':>10}{'taken':>8}{'% taken':>9}"
              f"{'selective':>12}{'trade all':>12}{'edge':>10}")
        for _, r in res.iterrows():
            print(f"{r.ticker:<13}{r.held_out:>10,}{r.taken:>8}{r.taken_pct:>8.0f}%"
                  f"{r.selective:>+11.3f}%{r.everything:>+11.3f}%"
                  f"{r.selective - r.everything:>+9.3f}%")
        print(f"\n  selective {res.selective.mean():+.3f}%/trade   "
              f"trade-everything {res.everything.mean():+.3f}%/trade")
        print(f"  selective is profitable in {(res.selective > 0).sum()}/{len(res)} symbols")
        print(f"  selective beats trading everything in "
              f"{(res.selective > res.everything).sum()}/{len(res)} symbols")
    df.to_csv("exp_selectivity.csv", index=False)
    print("\nwrote exp_selectivity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
