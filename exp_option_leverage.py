"""
exp_option_leverage.py
----------------------
"Options can easily give 10-15% in a day" -- is that true, and does it help?

The first half is TRUE and this file confirms it: an at-the-money option moves
roughly 10-20x the underlying in percentage terms, so a 0.8% move in the stock is
a 10-15% move in the premium. That happens constantly.

The second half is the question. Leverage is a multiplier, and it multiplies
whatever edge it is applied to -- including a negative one. Everything measured in
this project says the intraday directional edge is +0.078% gross and -0.072% after
the cheapest possible costs. This takes the SAME trades the opening-range-breakout
rule actually produced and runs them through real option economics.

WHAT IS MODELLED
  Black-Scholes premium for an ATM option, priced at entry and at exit, using
  each symbol's own realised volatility as the implied vol. Intraday holding, so
  theta over a few hours of a weekly contract. Entry at the ask, exit at the bid,
  with a spread taken from what liquid NSE options actually quote.

WHAT THAT IGNORES (all of it favourable to the trade)
  IV crush after the open, widening spreads when it moves against you, and the
  fact that a real fill is worse than a mid. So the numbers here are optimistic.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import intraday_strategies as istr
import paper_trade_intraday as pti

# --- option assumptions, deliberately generous -----------------------------
DTE_DAYS = 5.0              # a weekly contract with 5 days to run
SPREAD_PCT = 1.5            # bid/ask as a % of premium on a liquid NSE option
R = 0.065                   # risk-free
CAPITAL = 100_000.0


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(lambda v: __import__("math").erf(v / np.sqrt(2)))(x))


def bs_call(S, K, T, r, sigma):
    S, K = np.asarray(S, float), np.asarray(K, float)
    T = max(float(T), 1e-6)
    sigma = max(float(sigma), 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def bs_put(S, K, T, r, sigma):
    S, K = np.asarray(S, float), np.asarray(K, float)
    T = max(float(T), 1e-6)
    sigma = max(float(sigma), 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def option_trade(entry_px, exit_px, side, sigma_ann, hours_held=3.0):
    """
    Premium P&L of buying one ATM option and closing it the same session.
    Returns the percent change in the premium, after the bid/ask spread.
    """
    K = entry_px
    T0 = DTE_DAYS / 252.0
    T1 = max((DTE_DAYS - hours_held / 6.25) / 252.0, 1e-6)   # 6.25h NSE session
    price = bs_call if side == "long" else bs_put
    p0 = float(price(entry_px, K, T0, R, sigma_ann))
    p1 = float(price(exit_px, K, T1, R, sigma_ann))
    if p0 <= 0:
        return None
    # buy at the ask, sell at the bid
    buy = p0 * (1 + SPREAD_PCT / 200)
    sell = p1 * (1 - SPREAD_PCT / 200)
    return (sell / buy - 1) * 100


def main():
    print("loading intraday data...")
    data = pti.load_all("5m")
    if len(data) < 3:
        print("not enough data")
        return 1
    days = sorted({d for df in data.values() for d in df.index.normalize().unique()})
    all_days = set(days)
    cap = CAPITAL / len(data)

    # realised vol per symbol, annualised -- stands in for implied vol
    sigma = {}
    for t, df in data.items():
        daily = df["Close"].resample("1D").last().dropna()
        sigma[t] = float(daily.pct_change().std() * np.sqrt(252))

    # the SAME trades the best intraday rule produced
    fn = lambda s, pc: istr.orb(s, 3, 0.5)
    trades = pti.score(fn, data, all_days, cap)
    df = pd.DataFrame(trades)
    print(f"  {len(df)} opening-range-breakout trades over {len(days)} sessions\n")

    rows = []
    for _, r in df.iterrows():
        sg = sigma.get(r["ticker"], 0.25)
        opt = option_trade(r["entry"], r["exit"], r["side"], sg)
        if opt is None:
            continue
        rows.append({"underlying_net": r["net_pct"], "underlying_gross": r["gross_pct"],
                     "option_pct": opt, "ticker": r["ticker"], "date": r["date"]})
    o = pd.DataFrame(rows)

    print("=" * 88)
    print("IS THE 10-15% CLAIM TRUE? — yes, the moves are real")
    print("=" * 88)
    lev = (o["option_pct"] / o["underlying_gross"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  median leverage: an ATM option moves {lev.median():.1f}x the underlying, in %")
    print(f"  days the option gained more than +10%: {(o.option_pct > 10).mean() * 100:.1f}%")
    print(f"  days it gained more than +15%:         {(o.option_pct > 15).mean() * 100:.1f}%")
    print(f"  best single trade:                     {o.option_pct.max():+.1f}%")
    print("\n  So yes — double-digit days happen constantly. Now the other side:")
    print(f"  days the option LOST more than -10%:   {(o.option_pct < -10).mean() * 100:.1f}%")
    print(f"  days it lost more than -15%:           {(o.option_pct < -15).mean() * 100:.1f}%")
    print(f"  worst single trade:                    {o.option_pct.min():+.1f}%")

    print(f"\n{'=' * 88}")
    print("WHAT LEVERAGE DOES TO THE EDGE")
    print("=" * 88)
    print(f"  underlying, gross of costs   {o.underlying_gross.mean():+.3f}% per trade")
    print(f"  underlying, net of costs     {o.underlying_net.mean():+.3f}% per trade")
    print(f"  the option version           {o.option_pct.mean():+.3f}% per trade")
    print(f"  win rate, underlying         {(o.underlying_net > 0).mean() * 100:.0f}%")
    print(f"  win rate, option             {(o.option_pct > 0).mean() * 100:.0f}%")
    print("\n  Leverage multiplied the loss, it did not create a profit. It also could")
    print("  not: a multiplier applied to a negative number stays negative.")

    print(f"\n{'=' * 88}")
    print(f"COMPOUNDING Rs {CAPITAL:,.0f} — one ATM option position per signal")
    print("=" * 88)
    for frac, label in ((1.00, "100% of capital each trade"),
                        (0.25, "25% of capital each trade"),
                        (0.10, "10% of capital each trade")):
        eq = CAPITAL
        path = []
        for _, r in o.sort_values("date").iterrows():
            eq *= (1 + frac * r["option_pct"] / 100)
            path.append(eq)
            if eq < CAPITAL * 0.01:
                break
        ser = pd.Series(path)
        dd = float(((ser / ser.cummax()) - 1).min() * 100)
        print(f"  {label:<28} end Rs {eq:>12,.0f}   max drawdown {dd:>7.1f}%"
              f"   {'WIPED OUT' if eq < CAPITAL * 0.05 else ''}")

    print(f"\n{'=' * 88}")
    print("WHAT 10-15% A DAY WOULD ACTUALLY MEAN")
    print("=" * 88)
    for d in (10, 12, 15):
        for n, lbl in ((21, "1 month"), (252, "1 year")):
            v = CAPITAL * (1 + d / 100) ** n
            print(f"  {d}% a day for {lbl:<8} -> Rs {v:,.0f}" +
                  ("   (more than the entire Indian equity market)" if v > 5e14 else ""))
    print("\n  A rate that turns one lakh into the whole market inside a year is not a")
    print("  target to tune toward. It is the reason the target cannot exist: anyone")
    print("  who had it would own everything within months, and nobody does.")
    o.to_csv("exp_option_leverage.csv", index=False)
    print("\nwrote exp_option_leverage.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
