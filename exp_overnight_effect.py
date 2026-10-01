"""
exp_overnight_effect.py
-----------------------
The overnight/intraday decomposition, and why it was NOT shipped.

THE LEAD
  Published research (Lou, Polk & Skouras 2019; NY Fed "Overnight Drift") finds
  that nearly all long-run equity return accrues overnight, while the intraday
  session drifts around zero. Every strategy in this project traded INTRADAY, so
  if that holds on NSE it would explain years of negative results at a stroke.

WHAT THE DAILY BARS SAID
  29 NSE large caps, 12.5 years, from daily Open/Close:

      overnight  +39.9%/yr   positive in 29/29 symbols   Sharpe 1.84
      intraday   -17.6%/yr   positive in  1/29 symbols   Sharpe -0.59
      NIFTY overnight +31.1%/yr at Sharpe 2.80 vs +11.0% buy & hold

  NIFTY futures cost 0.0348% round trip, so 252 round trips a year is ~13.9% of
  capital -- leaving roughly +17%/yr net. That would have been the best result in
  the entire project by a wide margin.

WHY IT IS FALSE
  The whole decomposition rests on the daily Open being the first traded price.
  Checked against the first 5-minute bar of the same session:

      symbol         sessions   mean |diff|   Open == first trade
      HDFCBANK.NS          58        0.128%                   38%
      RELIANCE.NS          58        0.082%                   43%
      TCS.NS               58        0.145%                   29%
      INFY.NS              58        0.155%                   28%

  Yahoo's daily Open for NSE symbols is not the opening trade. It disagrees with
  the tape on roughly two sessions in three, by about 0.11% on average -- and
  0.11% x 252 sessions is ~25%/yr, which is the size of the "effect".

  Recomputed from intraday bars alone, with no daily Open involved, it collapses:

      HDFCBANK.NS   overnight -28.6%/yr   intraday -22.3%/yr
      RELIANCE.NS   overnight -15.0%/yr   intraday -15.7%/yr
      TCS.NS        overnight +12.1%/yr   intraday -13.2%/yr

  The US indices, whose Open field is reliable, show no such split either
  (^GSPC +7.4% overnight vs +4.3% intraday -- balanced, as expected).

CONCLUSION
  A Sharpe-2.8 strategy that exists only in one vendor's Open column is a bug in
  the data, not an edge in the market. Not shipped. This file is kept so the
  finding cannot be rediscovered and believed.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import data_loader as dl, market_data as md, stock_search as ss


def decomposition(symbols, start="2014-01-01", end="2026-09-29"):
    rows = []
    for t in symbols:
        try:
            d = dl.load_stock_data(t, start, end); d, _ = md.repair_market_data(d, t)
            if len(d) < 1500: continue
            on = (d["Open"] / d["Close"].shift(1) - 1).dropna()
            idd = (d["Close"] / d["Open"] - 1).dropna()
            n = min(len(on), len(idd)); yrs = n / 252
            f = lambda s: ((1 + s.iloc[-n:]).prod() ** (1 / yrs) - 1) * 100
            rows.append({"ticker": t, "overnight": f(on), "intraday": f(idd)})
        except Exception:
            continue
    return pd.DataFrame(rows)


def validate_open(symbols):
    """The check that killed it: is the daily Open the first traded price?"""
    rows = []
    for t in symbols:
        try:
            d5 = dl.load_ohlcv(t, "5m")
            dd = dl.load_stock_data(t, "2014-01-01", "2026-09-29")
            first = d5.groupby(d5.index.normalize())["Open"].first()
            j = pd.DataFrame({"intraday_open": first}).join(dd[["Open"]], how="inner")
            if not len(j): continue
            diff = (j["intraday_open"] / j["Open"] - 1).abs() * 100
            rows.append({"ticker": t, "sessions": len(j), "mean_diff_pct": diff.mean(),
                         "max_diff_pct": diff.max(), "match_pct": (diff < 0.05).mean() * 100})
        except Exception:
            continue
    return pd.DataFrame(rows)


if __name__ == "__main__":
    syms = [r[0] for r in ss.POPULAR_STOCKS if r[0].endswith(".NS")][:30]
    print("apparent decomposition from DAILY bars:")
    d = decomposition(syms)
    print(f"  overnight {d.overnight.mean():+.1f}%/yr   intraday {d.intraday.mean():+.1f}%/yr")
    print(f"  overnight positive in {(d.overnight > 0).sum()}/{len(d)} symbols\n")
    print("is the daily Open actually the first trade?")
    v = validate_open(syms[:6])
    print(v.to_string(index=False))
    print(f"\n  Open matches the tape on only {v.match_pct.mean():.0f}% of sessions "
          f"-> the decomposition above is an artifact, not an edge.")
