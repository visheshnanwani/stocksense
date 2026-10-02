"""
exp_factor_search.py
--------------------
Every factor worth testing on the short straddle, over a full year of real NSE
option prices, with the discipline that makes a search like this mean anything.

THE DANGER, STATED FIRST
    Trying hundreds of combinations on one year and keeping the best is how
    people manufacture strategies that die on contact with money. With ~240
    sessions, the best of 300 combinations will look excellent by luck alone.

    So nothing is chosen on the whole year. The year is cut in half by DATE:
    every combination is ranked on the FIRST half and then applied, untouched,
    to the SECOND. Only a combination that works in both is reported as working,
    and the count of combinations tried is printed so the reader can discount
    accordingly.

THE FACTORS
  entry filters, each computed from data available BEFORE the open
    dte           days to expiry bucket (0-1, 2-3, 4-7)
    dow           day of the week
    prem_rich     premium collected vs the symbol's recent typical daily move --
                  the variance-risk-premium idea, the one with theory behind it
    vol_regime    trailing 21-day realised vol against its own 6-month norm
    prior_move    size of yesterday's move
    gap           overnight gap into the open
  construction
    moneyness     ATM, 0.5% / 1% / 2% out
    legs          both, calls only, puts only
    stop          none, 2.0x, 2.5x, 3.0x per leg
    profit_take   close early once a set share of the premium is captured
"""

import itertools
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import nse_fo_data as fo

LOT = {"NIFTY": 75, "BANKNIFTY": 35}
MARGIN_FRAC = 0.12
COST_RS = 60.0
STOP_SLIP = 5.0
START, END = "2025-10-01", "2026-09-30"


# ----------------------------------------------------------------- data
def build():
    df = fo.load_range(START, END, symbols=list(LOT))
    d = df[(df.open > 1.0) & (df.dte >= 0) & (df.dte <= 7)].copy()
    d["lot"] = d.symbol.map(LOT)
    d = d[d.lot.notna()].sort_values("date")

    sp = d.groupby(["symbol", "date"])["spot"].first().reset_index()
    sp["prev_spot"] = sp.groupby("symbol")["spot"].shift(1)
    sp["prev_move"] = (sp.prev_spot / sp.groupby("symbol")["spot"].shift(2) - 1).abs() * 100
    r = sp.groupby("symbol")["spot"].pct_change()
    sp["vol21"] = r.groupby(sp.symbol).transform(lambda x: x.rolling(21).std()).shift(1) * 100
    sp["vol126"] = r.groupby(sp.symbol).transform(lambda x: x.rolling(126).std()).shift(1) * 100
    d = d.merge(sp[["symbol", "date", "prev_spot", "prev_move", "vol21", "vol126"]],
                on=["symbol", "date"], how="left")
    d = d[d.prev_spot.notna()]

    d["key"] = d.symbol + "_" + d.expiry.astype(str) + "_" + d.strike.astype(str) + "_" + d.cp
    d["prev_vol"] = d.groupby("key")["volume"].shift(1)
    d = d[d.prev_vol >= 1000]
    d["mny"] = d.strike / d.prev_spot - 1
    d["dow"] = d.date.dt.dayofweek
    d["gap"] = (d.spot / d.prev_spot - 1).abs() * 100   # same-session only as a REGIME label
    return d


def legs_for(d, otm):
    """One call and one put per symbol-session at the chosen moneyness."""
    out = []
    for (dt, sym), g in d.groupby(["date", "symbol"]):
        for cp, tgt in (("CE", otm), ("PE", -otm)):
            c = g[g.cp == cp]
            if c.empty:
                continue
            out.append(c.loc[(c.mny - tgt).abs().idxmin()])
    return pd.DataFrame(out)


def pnl_for(legs, stop=None, take=None):
    o, c_, h, l = legs.open.values, legs.close.values, legs.high.values, legs.low.values
    exit_px = c_.copy()
    if stop is not None:
        trig = o * stop
        hit = h >= trig
        exit_px = np.where(hit, trig * (1 + STOP_SLIP / 100), exit_px)
    if take is not None:
        # cover once the option has lost `take` of its value (a seller's profit)
        tgt = o * (1 - take)
        got = l <= tgt
        if stop is not None:
            got = got & ~(h >= o * stop)
        exit_px = np.where(got, tgt, exit_px)
    return (o - exit_px) * legs.lot.values - COST_RS


def sessionise(legs, pnl):
    t = pd.DataFrame({"date": legs.date.values, "pnl": pnl,
                      "margin": legs.prev_spot.values * legs.lot.values * MARGIN_FRAC})
    return t.groupby("date").agg(pnl=("pnl", "sum"), margin=("margin", "sum")).reset_index()


def score(day):
    if len(day) < 25:
        return None
    r = day.pnl / day.margin * 100
    if r.std() == 0:
        return None
    eq = day.pnl.cumsum()
    return {"n": len(day), "total": day.pnl.sum(),
            "sharpe": r.mean() / r.std() * np.sqrt(252),
            "dd": float((eq - eq.cummax()).min()), "worst": day.pnl.min(),
            "win": (day.pnl > 0).mean() * 100}


# ----------------------------------------------------------------- filters
def apply_filter(legs, name, d_sp):
    if name == "all":
        return legs
    if name == "dte 0-1":
        return legs[legs.dte <= 1]
    if name == "dte 2-3":
        return legs[(legs.dte >= 2) & (legs.dte <= 3)]
    if name == "dte 4-7":
        return legs[legs.dte >= 4]
    if name == "Mon-Wed":
        return legs[legs.dow <= 2]
    if name == "Thu-Fri":
        return legs[legs.dow >= 3]
    if name == "calm prior day":
        return legs[legs.prev_move <= legs.prev_move.median()]
    if name == "wild prior day":
        return legs[legs.prev_move > legs.prev_move.median()]
    if name == "low vol regime":
        return legs[legs.vol21 <= legs.vol126]
    if name == "high vol regime":
        return legs[legs.vol21 > legs.vol126]
    if name == "premium rich":
        rich = legs.open / legs.prev_spot * 100 / legs.vol21.replace(0, np.nan)
        return legs[rich >= rich.median()]
    if name == "premium cheap":
        rich = legs.open / legs.prev_spot * 100 / legs.vol21.replace(0, np.nan)
        return legs[rich < rich.median()]
    return legs


FILTERS = ["all", "dte 0-1", "dte 2-3", "dte 4-7", "Mon-Wed", "Thu-Fri",
           "calm prior day", "wild prior day", "low vol regime", "high vol regime",
           "premium rich", "premium cheap"]
OTMS = [(0.0, "ATM"), (0.005, "0.5% OTM"), (0.01, "1% OTM"), (0.02, "2% OTM")]
SIDES = [("both", None), ("calls only", "CE"), ("puts only", "PE")]
STOPS = [(None, "no stop"), (2.0, "stop 2.0x"), (2.5, "stop 2.5x"), (3.0, "stop 3.0x")]
TAKES = [(None, "hold to close"), (0.5, "take 50%"), (0.7, "take 70%")]


def main():
    print("loading a full year of real NSE option prices...")
    d = build()
    days = sorted(d.date.unique())
    mid = days[len(days) // 2]
    print(f"  {len(d):,} contract-rows, {len(days)} sessions "
          f"({days[0]:%d %b %Y} to {days[-1]:%d %b %Y})")
    print(f"  TRAIN {days[0]:%d %b} - {mid:%d %b}   TEST {mid:%d %b} - {days[-1]:%d %b}\n")

    leg_cache = {lbl: legs_for(d, o) for o, lbl in OTMS}

    results = []
    for (otm, olbl), (slbl, side), (stop, stlbl), (take, tklbl), flt in itertools.product(
            OTMS, SIDES, STOPS, TAKES, FILTERS):
        legs = leg_cache[olbl]
        if side:
            legs = legs[legs.cp == side]
        legs = apply_filter(legs, flt, d)
        if len(legs) < 60:
            continue
        pnl = pnl_for(legs, stop, take)
        day = sessionise(legs, pnl)
        tr = score(day[day.date < mid])
        te = score(day[day.date >= mid])
        if not tr or not te:
            continue
        results.append({"otm": olbl, "side": slbl, "stop": stlbl, "take": tklbl,
                        "filter": flt,
                        "tr_sharpe": tr["sharpe"], "tr_total": tr["total"],
                        "te_sharpe": te["sharpe"], "te_total": te["total"],
                        "te_dd": te["dd"], "te_worst": te["worst"], "te_win": te["win"],
                        "n": tr["n"] + te["n"]})

    r = pd.DataFrame(results)
    print(f"{len(r):,} combinations tested\n")

    base = r[(r.otm == "ATM") & (r.side == "both") & (r.stop == "no stop")
             & (r.take == "hold to close") & (r["filter"] == "all")]
    b = base.iloc[0] if len(base) else None
    if b is not None:
        print(f"BASELINE (ATM straddle, no stop, no filter):")
        print(f"  train Sharpe {b['tr_sharpe']:.2f}  total {b['tr_total']:+,.0f}   |   "
              f"test Sharpe {b['te_sharpe']:.2f}  total {b['te_total']:+,.0f}\n")

    print("=" * 112)
    print("TOP 15 BY TRAINING SHARPE -- and what each then did on the untouched half")
    print("=" * 112)
    print(f"{'moneyness':<11}{'side':<11}{'stop':<11}{'exit':<15}{'filter':<17}"
          f"{'TRAIN Sh':>9}{'TEST Sh':>9}{'TEST total':>12}{'TEST dd':>11}")
    top = r.sort_values("tr_sharpe", ascending=False).head(15)
    for _, x in top.iterrows():
        # bracket access throughout: x.take would hit Series.take, not the column
        print(f"{x['otm']:<11}{x['side']:<11}{x['stop']:<11}{x['take']:<15}{x['filter']:<17}"
              f"{x['tr_sharpe']:>9.2f}{x['te_sharpe']:>9.2f}{x['te_total']:>+12,.0f}"
              f"{x['te_dd']:>+11,.0f}")

    print(f"\n  of the top 15 on training, {int((top['te_sharpe'] > 0).sum())} stayed positive on test, "
          f"{int((top['te_sharpe'] > (b['te_sharpe'] if b is not None else 0)).sum())} beat the baseline")

    print("\n" + "=" * 112)
    print("HOW WELL DOES TRAINING RANK PREDICT TEST? (if this is ~0, the search found noise)")
    print("=" * 112)
    c = r[["tr_sharpe", "te_sharpe"]].corr().iloc[0, 1]
    print(f"  correlation between train Sharpe and test Sharpe across all "
          f"{len(r):,} combinations: {c:+.3f}")
    decile = r.assign(dq=pd.qcut(r.tr_sharpe.rank(method="first"), 10, labels=False))
    g = decile.groupby("dq").agg(train=("tr_sharpe", "mean"), test=("te_sharpe", "mean"),
                                 n=("te_sharpe", "size"))
    print(f"\n  {'train decile':<14}{'mean TRAIN Sharpe':>20}{'mean TEST Sharpe':>20}")
    for i, row in g.iterrows():
        print(f"  {int(i) + 1:<14}{row.train:>20.2f}{row.test:>20.2f}")

    r.to_csv("exp_factor_search.csv", index=False)
    print("\nwrote exp_factor_search.csv")


if __name__ == "__main__":
    main()
