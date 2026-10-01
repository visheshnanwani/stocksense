"""
exp_stop_validation.py
----------------------
Does the stop-loss survive honest testing, or did I just pick the best of eleven?

The sweep found that exiting a leg when it doubles turns the base strategy's
Sharpe from 1.10 to 1.45 and September from -Rs 19,861 to +Rs 1,789. That is
exactly the shape of a result that evaporates: eleven variants were tried and the
winner was read off the same data it was chosen on.

Three things are checked here before recommending anything.

  1. OUT OF SAMPLE -- choose the stop multiple on the first half of the history,
     apply it blind to the second half.
  2. SLIPPAGE -- the backtest assumes a fill exactly at the stop price. A real
     stop is a market order into a moving book, and a leg that doubles is moving
     fast. Every multiple is retested with 2%, 5% and 10% worse fills.
  3. SHAPE -- is the gain broad, or one lucky session? Reported as the number of
     sessions the stop actually fired and what it changed on each.

The stop is measured against the option's own intraday HIGH from the bhavcopy,
which does not say WHEN the high happened. A session that touched the stop and
then recovered is still counted as stopped -- the pessimistic reading.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import exp_straddle_improvements as base

LOT = base.LOT
COST_RS = base.COST_RS


def run_stop(d, stop_mult=None, slip_pct=0.0):
    """Base straddle with an optional stop; `slip_pct` worsens the stop fill."""
    rows = []
    for (dt, sym), g in d.groupby(["date", "symbol"]):
        lot = int(g.lot.iloc[0])
        prev_spot = float(g.prev_spot.iloc[0])
        ce = base.pick_leg(g, "CE", 0.0)
        pe = base.pick_leg(g, "PE", 0.0)
        if ce is None or pe is None:
            continue

        def leg(l):
            o, c_, h = float(l.open), float(l.close), float(l.high)
            if stop_mult is not None and h >= o * stop_mult:
                return o * stop_mult * (1 + slip_pct / 100), True
            return c_, False

        ce_x, ce_s = leg(ce)
        pe_x, pe_s = leg(pe)
        pts = (float(ce.open) - ce_x) + (float(pe.open) - pe_x)
        rows.append({"date": dt, "symbol": sym,
                     "pnl": pts * lot - COST_RS * 2,
                     "margin": prev_spot * lot * base.MARGIN_FRAC * 2,
                     "stopped": ce_s or pe_s})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    return t.groupby("date").agg(n=("pnl", "size"), pnl=("pnl", "sum"),
                                 margin=("margin", "sum"),
                                 stopped=("stopped", "sum")).reset_index()


def stats(day):
    if day.empty:
        return None
    ret = day.pnl / day.margin * 100
    eq = day.pnl.cumsum()
    return {"total": day.pnl.sum(), "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0,
            "worst": day.pnl.min(), "dd": float((eq - eq.cummax()).min()),
            "win": (day.pnl > 0).mean() * 100, "n": len(day),
            "fired": int(day.stopped.sum())}


def main():
    d = base.load()
    days = sorted(d.date.unique())
    mid = days[len(days) // 2]
    print(f"{len(days)} sessions  |  train {days[0]:%d %b} - {mid:%d %b}  |  "
          f"test {mid:%d %b} - {days[-1]:%d %b}\n")
    tr = d[d.date < mid]
    te = d[d.date >= mid]

    print("=" * 94)
    print("1. CHOOSE ON THE FIRST HALF, APPLY TO THE SECOND")
    print("=" * 94)
    print(f"{'stop':<12}{'TRAIN total':>14}{'TRAIN Sharpe':>14}{'TEST total':>14}"
          f"{'TEST Sharpe':>13}{'TEST worst':>13}")
    best, rows = None, []
    for m in (None, 1.75, 2.0, 2.25, 2.5, 3.0):
        a, b = stats(run_stop(tr, m)), stats(run_stop(te, m))
        if not a or not b:
            continue
        lbl = "none" if m is None else f"{m:g}x"
        rows.append((lbl, m, a, b))
        print(f"{lbl:<12}{a['total']:>+14,.0f}{a['sharpe']:>14.2f}"
              f"{b['total']:>+14,.0f}{b['sharpe']:>13.2f}{b['worst']:>+13,.0f}")
        if m is not None and (best is None or a["sharpe"] > best[2]["sharpe"]):
            best = (lbl, m, a, b)
    none_row = next(r for r in rows if r[1] is None)
    print(f"\n  training picks {best[0]}.  On the untouched half it scores "
          f"{best[3]['total']:+,.0f} at Sharpe {best[3]['sharpe']:.2f}, "
          f"against {none_row[3]['total']:+,.0f} / {none_row[3]['sharpe']:.2f} with no stop.")
    verdict_oos = best[3]["sharpe"] > none_row[3]["sharpe"]
    print(f"  -> the stop {'HELD UP' if verdict_oos else 'did NOT hold up'} out of sample.")

    print(f"\n{'=' * 94}")
    print("2. WHAT IF THE STOP FILLS WORSE THAN THE TRIGGER PRICE?")
    print("=" * 94)
    print(f"{'stop':<12}{'fill at trigger':>18}{'+2% worse':>13}{'+5% worse':>13}{'+10% worse':>13}")
    for m in (1.75, 2.0, 2.25, 2.5, 3.0):
        cells = []
        for sl in (0.0, 2.0, 5.0, 10.0):
            s = stats(run_stop(d, m, sl))
            cells.append(s["total"])
        print(f"{m:g}x{'':<10}{cells[0]:>+18,.0f}{cells[1]:>+13,.0f}"
              f"{cells[2]:>+13,.0f}{cells[3]:>+13,.0f}")
    s0 = stats(run_stop(d, None))
    print(f"\n  no stop at all: {s0['total']:+,.0f}")

    print(f"\n{'=' * 94}")
    print("3. IS THE GAIN BROAD, OR ONE LUCKY DAY?")
    print("=" * 94)
    full_none = run_stop(d, None)
    full_stop = run_stop(d, 2.0)
    j = full_none.merge(full_stop, on="date", suffixes=("_none", "_stop"))
    j["delta"] = j.pnl_stop - j.pnl_none
    fired = j[j.stopped_stop > 0]
    print(f"  the stop fired on {len(fired)} of {len(j)} sessions ({len(fired)/len(j)*100:.0f}%)")
    print(f"  on those sessions it changed P&L by {fired.delta.sum():+,.0f} in total")
    print(f"  it helped on {int((fired.delta > 0).sum())} and hurt on {int((fired.delta < 0).sum())}")
    top = fired.reindex(fired.delta.abs().sort_values(ascending=False).index).head(6)
    print(f"\n  {'date':<12}{'without stop':>15}{'with stop':>13}{'difference':>14}")
    for _, x in top.iterrows():
        print(f"  {x.date:%Y-%m-%d}{x.pnl_none:>+15,.0f}{x.pnl_stop:>+13,.0f}{x.delta:>+14,.0f}")
    share = abs(top.delta.iloc[0]) / abs(fired.delta.sum()) * 100 if fired.delta.sum() else 0
    print(f"\n  the single biggest session is {share:.0f}% of the stop's total effect")


if __name__ == "__main__":
    main()
