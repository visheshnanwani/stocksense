"""
exp_straddle_improvements.py
----------------------------
Can the September loss be prevented without giving up the August gain?

The base strategy -- short the ATM straddle at the open, cover at the close --
made +Rs 38,860 in August and lost Rs 19,861 in September. The whole September
loss came from two sessions (24 and 28 Sep) where the market moved hard and one
leg tripled: the BANKNIFTY 55600 put went 311 -> 1,058.70 in a single day.

So the question is narrow and testable: what caps that tail without eating the
premium that pays for everything else? Six variants, all on real NSE bhavcopy
prices, all with the same point-in-time discipline as the base (strike from the
PRIOR close, liquidity from the PRIOR session, <= 7 DTE):

  base        short ATM call + put, hold to close
  stop_Nx     same, but exit the day a leg's intraday HIGH reaches N x its open
  strangle_N  sell N% out of the money instead of at the money
  condor_N    sell the ATM straddle AND buy wings N% out -- defined risk
  calm_only   skip the session if yesterday moved more than the recent norm
  lowvol_skip skip when the premium on offer is small relative to recent movement

A stop is measured against the option's own HIGH that session. The bhavcopy does
not record WHEN the high happened, so a session that both stopped out and would
have recovered is counted as stopped -- the pessimistic reading, same convention
used throughout this project.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import nse_fo_data as fo

LOT = {"NIFTY": 75, "BANKNIFTY": 35}
MARGIN_FRAC = 0.12
COST_RS = 60.0          # per lot, round trip
MAX_DTE = 7
MIN_PREV_VOL = 1000
START, END = "2026-02-01", "2026-09-30"


def load():
    df = fo.load_range(START, END, symbols=list(LOT))
    d = df[(df.open > 1.0) & (df.dte >= 0) & (df.dte <= MAX_DTE)].copy()
    d["lot"] = d.symbol.map(LOT)
    d = d[d.lot.notna()].sort_values("date")

    sp = d.groupby(["symbol", "date"])["spot"].first().reset_index()
    sp["prev_spot"] = sp.groupby("symbol")["spot"].shift(1)
    d = d.merge(sp[["symbol", "date", "prev_spot"]], on=["symbol", "date"], how="left")
    d = d[d.prev_spot.notna()]

    d["key"] = d.symbol + "_" + d.expiry.astype(str) + "_" + d.strike.astype(str) + "_" + d.cp
    d["prev_vol"] = d.groupby("key")["volume"].shift(1)
    d = d[d.prev_vol >= MIN_PREV_VOL]
    d["mny"] = d.strike / d.prev_spot - 1          # from the PRIOR close
    return d


def pick_leg(day_df, cp, target_mny):
    """The contract nearest a target moneyness, chosen on prior-close data only."""
    c = day_df[day_df.cp == cp]
    if c.empty:
        return None
    i = (c.mny - target_mny).abs().idxmin()
    return c.loc[i]


def run(d, kind="base", stop_mult=None, otm=0.0, wing=0.02, calm=False, min_prem=None):
    """One variant. Returns a per-session frame."""
    rows = []
    for (dt, sym), g in d.groupby(["date", "symbol"]):
        lot = int(g.lot.iloc[0])
        prev_spot = float(g.prev_spot.iloc[0])
        margin = prev_spot * lot * MARGIN_FRAC * 2

        # --- regime filters, both using only data known before the open -------
        if calm or min_prem is not None:
            hist = d[(d.symbol == sym) & (d.date < dt)]
            if hist.empty:
                continue
            spots = hist.groupby("date")["prev_spot"].first().tail(21)
            if len(spots) < 10:
                continue
            norm = float(spots.pct_change().abs().mean())
            last_move = abs(prev_spot / float(spots.iloc[-1]) - 1) if len(spots) else 0.0
            if calm and norm > 0 and last_move > 1.5 * norm:
                continue

        ce = pick_leg(g, "CE", otm)
        pe = pick_leg(g, "PE", -otm)
        if ce is None or pe is None:
            continue

        collected = float(ce.open) + float(pe.open)
        if min_prem is not None and collected / prev_spot * 100 < min_prem:
            continue

        legs, cost_lots = 2, 2

        # short legs: cover at the close, or at the stop if one was reached
        def leg_exit(leg):
            o, c_, h = float(leg.open), float(leg.close), float(leg.high)
            if stop_mult is not None and h >= o * stop_mult:
                return o * stop_mult, True
            return c_, False

        ce_exit, ce_stopped = leg_exit(ce)
        pe_exit, pe_stopped = leg_exit(pe)
        short_pnl = (float(ce.open) - ce_exit) + (float(pe.open) - pe_exit)
        stopped = ce_stopped or pe_stopped

        long_pnl = 0.0
        if kind == "condor":
            lc = pick_leg(g, "CE", wing)
            lp = pick_leg(g, "PE", -wing)
            if lc is None or lp is None:
                continue
            # long wings: pay the open, sell at the close
            long_pnl = (float(lc.close) - float(lc.open)) + (float(lp.close) - float(lp.open))
            legs, cost_lots = 4, 4
            margin *= 0.45          # defined risk needs far less margin

        pts = short_pnl + long_pnl
        pnl = pts * lot - COST_RS * cost_lots
        rows.append({"date": dt, "symbol": sym, "pnl": pnl, "margin": margin,
                     "collected_pct": collected / prev_spot * 100,
                     "stopped": stopped, "legs": legs})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    return t.groupby("date").agg(n=("pnl", "size"), pnl=("pnl", "sum"),
                                 margin=("margin", "sum"),
                                 stopped=("stopped", "sum")).reset_index()


def summarise(name, day):
    if day.empty:
        return None
    eq = day.pnl.cumsum()
    dd = float((eq - eq.cummax()).min())
    ret = day.pnl / day.margin * 100
    sep = day[day.date >= "2026-09-01"]
    aug = day[(day.date >= "2026-08-01") & (day.date < "2026-09-01")]
    return {"variant": name, "sessions": len(day), "total": day.pnl.sum(),
            "per_session": day.pnl.mean(), "win_pct": (day.pnl > 0).mean() * 100,
            "worst_day": day.pnl.min(), "max_dd": dd,
            "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0,
            "aug": aug.pnl.sum() if len(aug) else np.nan,
            "sep": sep.pnl.sum() if len(sep) else np.nan}


def main():
    print("loading real NSE option data...")
    d = load()
    print(f"  {len(d):,} contract-rows, {d.date.nunique()} sessions\n")

    variants = [
        ("base (current)",        dict(kind="base")),
        ("stop at 2.0x",          dict(kind="base", stop_mult=2.0)),
        ("stop at 2.5x",          dict(kind="base", stop_mult=2.5)),
        ("stop at 3.0x",          dict(kind="base", stop_mult=3.0)),
        ("strangle 1% OTM",       dict(kind="base", otm=0.01)),
        ("strangle 2% OTM",       dict(kind="base", otm=0.02)),
        ("iron condor 2% wings",  dict(kind="condor", wing=0.02)),
        ("iron condor 3% wings",  dict(kind="condor", wing=0.03)),
        ("skip after a big move", dict(kind="base", calm=True)),
        ("stop 2x + calm filter", dict(kind="base", stop_mult=2.0, calm=True)),
        ("condor 2% + stop 2x",   dict(kind="condor", wing=0.02, stop_mult=2.0)),
    ]

    out = []
    for name, kw in variants:
        s = summarise(name, run(d, **kw))
        if s:
            out.append(s)

    r = pd.DataFrame(out)
    print("=" * 108)
    print("EVERY VARIANT, Feb-Sep 2026, real exchange prices")
    print("=" * 108)
    print(f"{'variant':<24}{'total':>11}{'/session':>10}{'win%':>7}{'worst day':>12}"
          f"{'max DD':>11}{'Sharpe':>8}{'August':>11}{'Sept':>11}")
    for _, x in r.iterrows():
        print(f"{x.variant:<24}{x.total:>+11,.0f}{x.per_session:>+10,.0f}{x.win_pct:>6.0f}%"
              f"{x.worst_day:>+12,.0f}{x.max_dd:>+11,.0f}{x.sharpe:>8.2f}"
              f"{x.aug:>+11,.0f}{x.sep:>+11,.0f}")

    base = r[r.variant == "base (current)"].iloc[0]
    print()
    print("=" * 108)
    print("WHAT ACTUALLY HELPS  (vs the current strategy)")
    print("=" * 108)
    better = r[(r.sep > base.sep) & (r.total > 0)].sort_values("sharpe", ascending=False)
    if better.empty:
        print("  Nothing improved September while staying profitable overall.")
    else:
        for _, x in better.iterrows():
            if x.variant == "base (current)":
                continue
            print(f"  {x.variant:<24} Sept {x.sep:+,.0f} (vs {base.sep:+,.0f})   "
                  f"total {x.total:+,.0f} (vs {base.total:+,.0f})   "
                  f"Sharpe {x.sharpe:.2f} (vs {base.sharpe:.2f})   "
                  f"worst day {x.worst_day:+,.0f} (vs {base.worst_day:+,.0f})")
    r.to_csv("exp_straddle_improvements.csv", index=False)
    print("\nwrote exp_straddle_improvements.csv")


if __name__ == "__main__":
    main()
