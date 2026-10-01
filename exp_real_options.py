"""
exp_real_options.py
-------------------
Selling NSE index options, tested against REAL exchange prices.

Every options result before this used Black-Scholes premiums computed from
realised vol, because no source served actual quotes. NSE's daily F&O bhavcopy
does (nse_fo_data.py), so this uses what actually traded.

THREE LOOK-AHEAD LEAKS WERE FOUND AND FIXED HERE. They are documented because
each one produced a spectacular, completely false result:

  LEAK 1 - selecting contracts by that day's VOLUME.
      Volume is only known at the close. Sorting on it picks the contracts that
      turned out to be busy, and busyness is not neutral: the seller's P&L by
      volume quintile ran +0.48, -12.62, -8.18, +7.29, +12.26 points. A 25x
      spread, handed over for free by hindsight.
      FIX: require liquidity on the PREVIOUS session.

  LEAK 2 - the evaluation window was the calmest in years.
      NIFTY realised 8.4% annualised over Aug-Sep 2026 against a 15.9% long-run
      average -- 0.53x. Largest daily move in the window 1.64%; largest in twelve
      years 12.98%. Short volatility always looks brilliant in a calm tape.
      FIX: eight months instead of one, and the monthly table is printed so bad
      months cannot hide inside a good total.

  LEAK 3 - the worst one. `UndrlygPric` in the bhavcopy is the CLOSING spot,
      verified at 0.0000% mean difference against the NIFTY close. Selecting
      "at the money" with it means choosing the strike that ENDED at the money,
      which is precisely the strike that decayed most that day.
      FIX: choose the strike from the PREVIOUS close, which a trader actually
      knows at the open.

    result with leaks 1+3 present   Sharpe 17.51, 96% of sessions positive
    result with all three fixed     Sharpe  1.27, 68% of sessions positive

  A Sharpe of 17 is not a discovery, it is a receipt for a bug.

THE STRATEGY THAT SURVIVES
  At the open, short the ATM call and ATM put on NIFTY and BANKNIFTY (strike
  chosen from the prior close), on contracts that were liquid yesterday, with
  7 or fewer days to expiry. Cover at the close. Margin ~12% of notional,
  Rs 60 per lot round trip.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import nse_fo_data as fo

LOT = {"NIFTY": 75, "BANKNIFTY": 35}
MARGIN_FRAC = 0.12
COST_RS = 60.0
MAX_DTE = 7
ATM_BAND = 0.005
MIN_PREV_VOL = 1000

# Per-leg stop: cover a leg once its intraday high reaches this multiple of the
# price it was sold at. 2.5x was chosen on the first half of the history and held
# on the untouched second half (Sharpe 1.32 -> 2.27). Tighter settings score
# better at the trigger price and fall apart on slippage, because they fire far
# more often: at a 5% worse fill 2.0x drops to +1,547 while 2.5x keeps +42,266.
# See exp_straddle_improvements.py and exp_stop_validation.py.
STOP_MULT = 2.5
STOP_SLIP_PCT = 5.0        # assume the stop fills 5% worse than the trigger


def build(start="2026-02-01", end="2026-09-30"):
    df = fo.load_range(start, end, symbols=list(LOT))
    if df.empty:
        return df
    d = df[(df.open > 1.0) & (df.dte >= 0) & (df.dte <= MAX_DTE)].copy()
    d["lot"] = d.symbol.map(LOT)
    d = d[d.lot.notna()].sort_values("date")

    # `spot` is the CLOSE -- shift it so selection uses only the prior session
    sp = d.groupby(["symbol", "date"])["spot"].first().reset_index()
    sp["prev_spot"] = sp.groupby("symbol")["spot"].shift(1)
    d = d.merge(sp[["symbol", "date", "prev_spot"]], on=["symbol", "date"], how="left")
    d = d[d.prev_spot.notna()]

    d["key"] = (d.symbol + "_" + d.expiry.astype(str) + "_"
                + d.strike.astype(str) + "_" + d.cp)
    d["prev_vol"] = d.groupby("key")["volume"].shift(1)
    d = d[d.prev_vol >= MIN_PREV_VOL]

    d["absm"] = (d.strike / d.prev_spot - 1).abs()

    # exit: the stop if the session's high reached it, else the close
    trigger = d.open * STOP_MULT
    hit = d.high >= trigger
    d["stopped"] = hit
    d["exit_px"] = np.where(hit, trigger * (1 + STOP_SLIP_PCT / 100), d.close)
    d["pnl"] = (d.open - d.exit_px) * d.lot - COST_RS
    d["margin"] = d.prev_spot * d.lot * MARGIN_FRAC
    d["ret"] = d.pnl / d.margin * 100
    return d


def report(d):
    pick = (d[d.absm <= ATM_BAND].sort_values("absm")
            .groupby(["date", "symbol", "cp"]).head(1))
    day = pick.groupby("date").agg(n=("pnl", "size"), pnl=("pnl", "sum"),
                                   margin=("margin", "sum"))
    day["ret"] = day.pnl / day.margin * 100
    eq = day.pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    sharpe = day.ret.mean() / day.ret.std() * np.sqrt(252) if day.ret.std() > 0 else 0

    print("=" * 78)
    print("SHORT ATM STRADDLE, NIFTY + BANKNIFTY, intraday, real exchange prices")
    print("=" * 78)
    print(f"  sessions {len(day)}   positions {len(pick)}")
    print(f"  total P&L           Rs {day.pnl.sum():+,.0f}")
    print(f"  mean per session    Rs {day.pnl.mean():+,.0f}  "
          f"({day.ret.mean():+.2f}% of margin)")
    print(f"  winning sessions    {(day.pnl > 0).mean() * 100:.0f}%")
    print(f"  worst session       Rs {day.pnl.min():+,.0f} ({day.ret.min():.1f}%)")
    print(f"  best session        Rs {day.pnl.max():+,.0f}")
    print(f"  Sharpe              {sharpe:.2f}")
    print(f"  max drawdown        Rs {dd:+,.0f}")
    print(f"  monthly (approx)    {day.ret.mean() * 21:+.1f}% of margin")
    print(f"  stop fired on       {int(pick.stopped.sum())} of {len(pick)} legs "
          f"({STOP_MULT:g}x, filled {STOP_SLIP_PCT:g}% worse)")
    print()
    print("  RISK: the maximum drawdown is larger than the total profit. That is")
    print("  the short-volatility signature -- many small wins, occasional large")
    print("  losses -- and eight months is not long enough to have met a real one.")
    print()
    m = day.groupby(day.index.to_period("M")).agg(
        sessions=("pnl", "size"), pnl=("pnl", "sum"), mean_ret=("ret", "mean"))
    print(m.round(2).to_string())
    return day


if __name__ == "__main__":
    d = build()
    if d.empty:
        print("no data -- run nse_fo_data.py first")
    else:
        report(d)
