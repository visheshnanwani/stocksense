"""
exp_anomalies.py
----------------
WHERE ARE WE LAGGING?

Everything built so far tries to answer one question: "which way does THIS stock
move TOMORROW?" Every measurement says that question has no answer here:

    raw ensemble, ungated        loses to "no change" on every symbol
    same-day bracket, zero cost  -0.00% expectancy, 48.9% win rate
    stronger conviction          monotonically WORSE (-0.109% -> -0.621%)

That is not a bug in the features or the learner. Next-day direction on a liquid
large cap is about the most efficient thing in markets: thousands of professionals
with faster data are competing on exactly it. Out-predicting them from daily Yahoo
bars was never likely.

The documented, repeatedly replicated equity edges do not live at one day on one
stock. They live CROSS-SECTIONALLY (ranking many stocks against each other) and at
LONGER horizons:

    short-term reversal   past-week losers beat past-week winners (Jegadeesh 1990)
    momentum 12-1         past-year winners keep winning, skipping last month
                          (Jegadeesh & Titman 1993)
    low volatility        low-vol stocks earn more per unit of risk (Haugen 1991)

This tests those on the NSE universe, long/short and long-only, with real costs,
and splits the history so the answer is out-of-sample.

HONEST CAVEAT, STATED UP FRONT: the universe is today's list of large caps, so it
carries survivorship bias -- companies that failed are not in it. That inflates
LONG-ONLY results. Long/short is far less affected because the bias hits both
legs, so the long/short numbers are the ones to trust.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import data_loader as dl
import market_data as md
import stock_search as ss

COST_LEG = 0.15          # per rebalance, per leg
START = "2014-01-01"
END = "2026-09-29"


def build_panel():
    syms = [r[0] for r in getattr(ss, "POPULAR_STOCKS", []) if r[0].endswith(".NS")]
    closes = {}
    for t in syms:
        try:
            d = dl.load_stock_data(t, START, END)
            d, _ = md.repair_market_data(d, t)
            if len(d) > 1500:
                closes[t] = d["Close"].astype(float)
        except Exception:
            continue
    px = pd.DataFrame(closes).sort_index()
    px = px.dropna(axis=1, thresh=int(len(px) * 0.9)).ffill()
    return px


def backtest_rank(px, signal, hold, n_bucket=5, long_short=True, cost_leg=COST_LEG):
    """
    Rank every symbol by `signal` each rebalance, hold `hold` sessions.
    Long the best bucket, short the worst (or long-only), equal weight.
    """
    rets = px.pct_change()
    dates = px.index[::hold]
    legs = []
    for i, dt in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        s = signal.loc[dt].dropna()
        if len(s) < 10:
            continue
        k = max(len(s) // n_bucket, 2)
        top = s.nlargest(k).index
        bot = s.nsmallest(k).index
        fwd = (px.loc[nxt] / px.loc[dt] - 1) * 100
        long_leg = fwd[bot].mean()          # convention: LOW signal = long bucket
        short_leg = fwd[top].mean()
        if long_short:
            legs.append({"date": nxt, "ret": long_leg - short_leg - 2 * cost_leg})
        else:
            legs.append({"date": nxt, "ret": long_leg - cost_leg})
    if not legs:
        return None
    df = pd.DataFrame(legs).set_index("date")
    per_year = 252 / hold
    mu = df["ret"].mean()
    sd = df["ret"].std()
    return {"periods": len(df), "mean_pct": mu, "ann_pct": mu * per_year,
            "sharpe": (mu / sd * np.sqrt(per_year)) if sd > 0 else 0.0,
            "win_rate": float((df["ret"] > 0).mean() * 100),
            "total_pct": float(df["ret"].sum()), "series": df}


def main():
    print("building panel...")
    px = build_panel()
    print(f"  {px.shape[1]} symbols x {px.shape[0]:,} sessions "
          f"({px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d})\n")
    if px.shape[1] < 15:
        print("not enough symbols")
        return 1

    rets = px.pct_change()
    signals = {
        "short-term reversal (5d)": (rets.rolling(5).sum(), 5),
        "short-term reversal (10d)": (rets.rolling(10).sum(), 10),
        "momentum 12-1": (-(px.shift(21) / px.shift(252) - 1), 21),
        "low volatility (60d)": (rets.rolling(60).std(), 21),
    }

    print("=" * 96)
    print(f"FULL HISTORY  (cost {COST_LEG:.2f}% per leg per rebalance)")
    print("=" * 96)
    print(f"{'signal':<28}{'hold':>6}{'mode':>12}{'periods':>9}"
          f"{'per period':>12}{'annualised':>12}{'Sharpe':>9}{'win':>8}")
    keep = {}
    for name, (sig, hold) in signals.items():
        for ls, label in ((True, "long/short"), (False, "long-only")):
            r = backtest_rank(px, sig, hold, long_short=ls)
            if not r:
                continue
            keep[(name, label)] = r
            print(f"{name:<28}{hold:>6}{label:>12}{r['periods']:>9}"
                  f"{r['mean_pct']:>+11.3f}%{r['ann_pct']:>+11.1f}%"
                  f"{r['sharpe']:>9.2f}{r['win_rate']:>7.0f}%")

    # ---------------- out of sample ----------------
    split = px.index[int(len(px) * 0.6)]
    print(f"\n{'=' * 96}")
    print(f"OUT OF SAMPLE  —  first 60% ({px.index[0]:%Y-%m} to {split:%Y-%m}) "
          f"vs last 40% ({split:%Y-%m} onwards)")
    print("=" * 96)
    print(f"{'signal':<28}{'mode':>12}{'in-sample ann':>16}{'out-of-sample ann':>20}{'holds?':>9}")
    survivors = []
    for name, (sig, hold) in signals.items():
        for ls, label in ((True, "long/short"), (False, "long-only")):
            a = backtest_rank(px.loc[:split], sig.loc[:split], hold, long_short=ls)
            b = backtest_rank(px.loc[split:], sig.loc[split:], hold, long_short=ls)
            if not a or not b:
                continue
            ok = a["ann_pct"] > 0 and b["ann_pct"] > 0
            if ok:
                survivors.append((name, label, b["ann_pct"], b["sharpe"]))
            print(f"{name:<28}{label:>12}{a['ann_pct']:>+15.1f}%{b['ann_pct']:>+19.1f}%"
                  f"{'yes' if ok else 'no':>9}")

    print(f"\n{'=' * 96}\nVERDICT\n{'=' * 96}")
    if survivors:
        survivors.sort(key=lambda x: -x[3])
        print("  These worked in BOTH halves:")
        for n, l, ann, sh in survivors:
            print(f"    {n:<28} {l:<12} {ann:+.1f}%/yr out-of-sample, Sharpe {sh:.2f}")
        print("\n  Long/short results are the trustworthy ones: the universe is today's")
        print("  large caps, so long-only is flattered by survivorship bias.")
    else:
        print("  Nothing survived both halves. The documented anomalies do not show up")
        print("  in this universe at these costs either.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
