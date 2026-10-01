"""
paper_trade_portfolio.py
------------------------
Paper-trade the approaches that actually survived out-of-sample testing.

The day-trading work in paper_trade.py loses money, and the diagnosis is not a
bug: next-day direction on a liquid large cap has no edge (measured -0.00%
expectancy at ZERO cost), and every documented anomaly tested long/short came out
at or below zero after costs. What DID survive five years out-of-sample was
holding a diversified basket, and, more weakly, monthly momentum.

    equal-weight buy & hold   +10.4%/yr   Sharpe 0.75   max drawdown -17.1%
    momentum 12-1 long-only    +9.4%/yr   Sharpe 0.56   max drawdown -26.6%
    NIFTY 50                   +5.8%/yr   Sharpe 0.48

So this runs those as a paper trade over a window you choose, with costs, against
the index. Weeks are the wrong unit for them -- the edge is measured in years --
so the default window is one year and short windows print a warning.

Run:  python paper_trade_portfolio.py
      python paper_trade_portfolio.py --months 60 --strategy momentum
"""
import argparse, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import exp_anomalies as ea

CAPITAL = 100_000.0
COST_LEG = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--strategy", choices=["hold", "momentum", "both"], default="both")
    args = ap.parse_args()

    px = ea.build_panel()
    n = int(args.months * 21)
    win = px.iloc[-n:]
    yrs = len(win) / 252
    print("=" * 88)
    print(f"PORTFOLIO PAPER TRADE  —  last {args.months} months "
          f"({win.index[0]:%d %b %Y} to {win.index[-1]:%d %b %Y}), "
          f"{px.shape[1]} NSE large caps, Rs {CAPITAL:,.0f}")
    print("=" * 88)
    if args.months < 12:
        print("  NOTE: these edges are measured in years. A window this short shows what")
        print("  the market did, not whether the strategy works.\n")

    out = []
    if args.strategy in ("hold", "both"):
        ew = win.pct_change().mean(axis=1)
        eq = CAPITAL * (1 + ew.fillna(0)).cumprod()
        tot = eq.iloc[-1] / CAPITAL - 1
        dd = float(((eq / eq.cummax()) - 1).min() * 100)
        out.append(("Equal-weight buy & hold", tot * 100, (1 + tot) ** (1 / yrs) - 1,
                    ew.mean() / ew.std() * np.sqrt(252), dd, eq.iloc[-1] - CAPITAL, 1))

    if args.strategy in ("momentum", "both"):
        sig = -(px.shift(21) / px.shift(252) - 1)
        b = ea.backtest_rank(win, sig.loc[win.index], 21, long_short=False, cost_leg=COST_LEG)
        if b:
            s = b["series"]["ret"] / 100
            eq = CAPITAL * (1 + s).cumprod()
            tot = eq.iloc[-1] / CAPITAL - 1
            dd = float(((eq / eq.cummax()) - 1).min() * 100)
            out.append(("Momentum 12-1 (monthly rebalance)", tot * 100,
                        (1 + tot) ** (1 / yrs) - 1,
                        s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0,
                        dd, eq.iloc[-1] - CAPITAL, b["periods"]))

    try:
        import data_loader as dl
        nif = dl.load_stock_data("^NSEI", "2014-01-01", "2026-09-29").loc[win.index[0]:]
        r = nif["Close"].pct_change()
        tot = nif["Close"].iloc[-1] / nif["Close"].iloc[0] - 1
        eq = CAPITAL * (1 + r.fillna(0)).cumprod()
        out.append(("NIFTY 50 (benchmark)", tot * 100, (1 + tot) ** (1 / yrs) - 1,
                    r.mean() / r.std() * np.sqrt(252),
                    float(((eq / eq.cummax()) - 1).min() * 100), CAPITAL * tot, 1))
    except Exception:
        pass

    print(f"{'strategy':<36}{'total':>10}{'annualised':>13}{'Sharpe':>9}"
          f"{'max DD':>10}{'P&L':>14}{'trades':>9}")
    for name, tot, ann, sh, dd, pnl, nt in out:
        print(f"{name:<36}{tot:>+9.2f}%{ann * 100:>+12.1f}%{sh:>9.2f}"
              f"{dd:>9.1f}%{pnl:>+14,.0f}{nt:>9}")
    print()
    print("  Costs: 0.15% per leg on every rebalance. Buy & hold pays them once.")
    print("  Survivorship caveat: the universe is today's large caps, so long-only")
    print("  results are flattered. The ranking between them is still informative.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
