"""
paper_trade_intraday.py
-----------------------
Intraday paper trade of last week, with the rule chosen BEFORE the week.

  TRAIN  the 53 sessions before last week. Every rule and parameter in
         intraday_strategies.strategy_grid() is scored here. The winner is
         picked here, on training expectancy, and then frozen.
  TEST   last week. The frozen rule trades it once, blind.

This ordering is the whole point. Searching ~25 rule/parameter combinations and
then reporting the best one's performance ON THE SEARCH DATA would produce a
beautiful number that means nothing. The number below is what the winner did on
days that played no part in choosing it.

Costs 0.15% round trip. Equal weight across the universe, one trade per symbol
per session at most.

Run:  python paper_trade_intraday.py
      python paper_trade_intraday.py --test-days 5 --capital 100000
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import data_loader as dl
import intraday_strategies as istr

UNIVERSE = ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "ITC.NS",
            "SBIN.NS", "ICICIBANK.NS", "AXISBANK.NS"]
COST = istr.COST


def load_all(interval="5m"):
    out = {}
    for t in UNIVERSE:
        try:
            d = dl.load_ohlcv(t, interval)
            if d is not None and len(d) > 500:
                out[t] = d
        except Exception:
            continue
    return out


def score(strategy, data, days, capital_per_symbol):
    """Run one rule over a set of sessions. Returns the trade list."""
    trades = []
    for t, df in data.items():
        prev_close = None
        for day, g in istr.sessions(df):
            if day not in days:
                prev_close = float(g["Close"].iloc[-1])
                continue
            r = strategy(g, prev_close)
            prev_close = float(g["Close"].iloc[-1])
            if r is None:
                continue
            net = r["gross"] - COST
            trades.append({"date": day, "ticker": t, "side": r["side"],
                           "entry": r["entry"], "exit": r["exit"], "how": r["how"],
                           "gross_pct": r["gross"], "net_pct": net,
                           "pnl": capital_per_symbol * net / 100})
    return trades


def summarise(trades):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    return {"n": len(df), "expectancy": df.net_pct.mean(),
            "win_rate": (df.net_pct > 0).mean() * 100,
            "total_pct": df.net_pct.sum(), "pnl": df.pnl.sum(),
            "best": df.net_pct.max(), "worst": df.net_pct.min(), "df": df}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-days", type=int, default=5)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--interval", default="5m")
    args = ap.parse_args()

    print("loading intraday data...")
    data = load_all(args.interval)
    if len(data) < 3:
        print("not enough intraday data")
        return 1
    all_days = sorted({d for df in data.values() for d in df.index.normalize().unique()})
    test_days = set(all_days[-args.test_days:])
    train_days = set(all_days[:-args.test_days])
    cap = args.capital / len(data)

    print(f"  {len(data)} symbols, {args.interval} bars")
    print(f"  TRAIN {len(train_days)} sessions "
          f"({min(train_days):%d %b} to {max(train_days):%d %b})")
    print(f"  TEST  {len(test_days)} sessions "
          f"({min(test_days):%d %b} to {max(test_days):%d %b})  <- traded blind\n")

    # ---------------- choose on training only ----------------
    print("=" * 92)
    print("STEP 1 — score every rule on the TRAINING sessions (the test week is untouched)")
    print("=" * 92)
    print(f"{'rule':<34}{'trades':>8}{'expectancy':>13}{'win rate':>11}{'total':>10}")
    results = []
    for name, fn in istr.strategy_grid():
        tr = score(fn, data, train_days, cap)
        s = summarise(tr)
        if not s or s["n"] < 25:
            continue
        results.append((name, fn, s))
        print(f"{name:<34}{s['n']:>8}{s['expectancy']:>+12.3f}%"
              f"{s['win_rate']:>10.0f}%{s['total_pct']:>+9.1f}%")
    if not results:
        print("no rule produced enough trades")
        return 1

    results.sort(key=lambda x: -x[2]["expectancy"])
    best_name, best_fn, best_train = results[0]
    print(f"\n  WINNER ON TRAINING: {best_name}")
    print(f"    {best_train['n']} trades, {best_train['expectancy']:+.3f}% per trade, "
          f"{best_train['win_rate']:.0f}% win rate")
    pos = sum(1 for _, _, s in results if s["expectancy"] > 0)
    print(f"    {pos} of {len(results)} rules were positive in training")

    # ---------------- apply to the untouched week ----------------
    print(f"\n{'=' * 92}")
    print(f"STEP 2 — trade LAST WEEK with that rule, frozen. Rs {args.capital:,.0f}, "
          f"{COST:.2f}% round trip")
    print("=" * 92)
    tr = score(best_fn, data, test_days, cap)
    s = summarise(tr)
    if not s:
        print("  the rule produced no trades last week")
        return 0
    df = s["df"].sort_values(["date", "ticker"])
    print(f"{'date':<12}{'ticker':<14}{'side':<7}{'exit':>9}{'gross':>9}{'net':>9}{'P&L':>11}")
    for _, r in df.iterrows():
        print(f"{r['date']:%Y-%m-%d}  {r['ticker']:<14}{r['side']:<7}{r['how']:>9}"
              f"{r['gross_pct']:>+8.2f}%{r['net_pct']:>+8.2f}%{r['pnl']:>+11,.0f}")

    day = df.groupby("date").agg(n=("net_pct", "size"), pnl=("pnl", "sum"))
    day["equity"] = args.capital + day["pnl"].cumsum()
    print(f"\n{'date':<12}{'trades':>8}{'P&L':>12}{'equity':>14}")
    for dt, r in day.iterrows():
        print(f"{dt:%Y-%m-%d}  {int(r['n']):>7}{r['pnl']:>+12,.0f}{r['equity']:>14,.0f}")

    print(f"\n{'=' * 92}\nRESULT — LAST WEEK, OUT OF SAMPLE\n{'=' * 92}")
    print(f"  rule                  {best_name}")
    print(f"  trades                {s['n']}  ({int((df.net_pct > 0).sum())} winners, "
          f"{int((df.net_pct <= 0).sum())} losers, {s['win_rate']:.0f}% win rate)")
    print(f"  expectancy            {s['expectancy']:+.3f}% per trade")
    print(f"  P&L                   {s['pnl']:+,.0f} on {args.capital:,.0f} "
          f"({s['pnl'] / args.capital * 100:+.2f}%)")
    print(f"  best / worst trade    {s['best']:+.2f}% / {s['worst']:+.2f}%")
    print(f"  training expectancy   {best_train['expectancy']:+.3f}%  "
          f"-> test {s['expectancy']:+.3f}%  "
          f"({'held up' if s['expectancy'] > 0 else 'did not hold up'})")

    # ---------------- is one week enough? ----------------
    print(f"\n{'=' * 92}\nHOW MUCH DOES ONE WEEK TELL YOU?\n{'=' * 92}")
    tdf = best_train["df"].copy()
    tdf["week"] = pd.to_datetime(tdf["date"]).dt.to_period("W")
    wk = tdf.groupby("week")["net_pct"].sum()
    if len(wk) >= 4:
        print(f"  the same rule over {len(wk)} training weeks:")
        print(f"    mean week {wk.mean():+.2f}%   sd {wk.std():.2f}%   "
              f"best {wk.max():+.2f}%   worst {wk.min():+.2f}%")
        print(f"    weeks profitable: {(wk > 0).mean() * 100:.0f}%")
        print(f"  last week came in at {s['total_pct']:+.2f}% "
              f"({float((wk < s['total_pct']).mean() * 100):.0f}th percentile)")
        if wk.std() > 0:
            need = int((1.96 * wk.std() / max(abs(wk.mean()), 1e-9)) ** 2)
            print(f"    separating a {wk.mean():+.2f}%/week effect from noise needs "
                  f"~{need:,} weeks")
    print(f"\n  Only {len(results)} rules were tried, on {len(train_days)} sessions of a")
    print(f"  60-day data limit. Even a positive week here is a small sample on a short")
    print(f"  history — it is evidence, not proof.")
    df.to_csv("paper_trade_intraday_log.csv", index=False)
    print(f"\n  wrote paper_trade_intraday_log.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
