"""
test_intraday_target.py
-----------------------
The target-move finder answers "I want 1-2%, when do I get it?" from measured
history rather than from a daily forecast. These checks keep it honest.

  1. hit rates fall as the target grows, and are bounded 0..100
  2. the one-directional rate never exceeds the either-direction rate
     (you cannot buy the low and sell the high -- the whole point)
  3. the daily-derived hit rate agrees with an independent intraday-derived one
     on the same sessions, which is what justifies using daily bars for the
     full history
  4. time-to-hit grows with the target
  5. within-1-day <= within-3-days <= within-5-days
  6. costs are subtracted, and a target below cost is refused
  7. a synthetic symbol that never moves reports 0% rather than a number

Run:  python test_intraday_target.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import data_loader as dl
import intraday_target as itg

TICKER = "HDFCBANK.NS"
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main():
    daily = dl.load_stock_data(TICKER, "2014-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))

    print(f"\n{'=' * 72}\n1-2. hit rates behave like probabilities\n{'=' * 72}")
    stats = {t: itg.session_touch_stats(daily, t) for t in (0.5, 1.0, 1.5, 2.0, 3.0)}
    ups = [stats[t]["p_up"] for t in sorted(stats)]
    check("one-directional hit rate falls as the target grows",
          all(a >= b for a, b in zip(ups, ups[1:])), " -> ".join(f"{u:.0f}%" for u in ups))
    check("all rates are within 0..100",
          all(0 <= v <= 100 for s in stats.values()
              for k, v in s.items() if k.startswith("p_")))
    check("one direction never beats either direction",
          all(s["p_up"] <= s["p_either"] + 1e-9 and s["p_down"] <= s["p_either"] + 1e-9
              for s in stats.values()),
          f"at 1.0%: up {stats[1.0]['p_up']:.0f}% vs either {stats[1.0]['p_either']:.0f}%")

    print(f"\n{'=' * 72}\n3. daily-derived agrees with intraday-derived\n{'=' * 72}")
    bars = dl.load_ohlcv(TICKER, "15m")
    first_day = bars.index.normalize().min()
    same = daily.loc[daily.index >= first_day]
    for t in (0.5, 1.0):
        d_rate = float(((same["High"] / same["Open"] - 1) * 100 >= t).mean() * 100)
        i_rate = itg.time_to_touch(bars, t, "up").get("hit_rate", float("nan"))
        check(f"+{t:.1f}% hit rate agrees within 10 pts",
              abs(d_rate - i_rate) <= 10,
              f"daily {d_rate:.0f}% vs 15m {i_rate:.0f}% over ~{len(same)} sessions")

    print(f"\n{'=' * 72}\n4-5. time grows with the target; horizons nest\n{'=' * 72}")
    mins = []
    for t in (0.5, 1.0, 1.5):
        tm = itg.time_to_touch(bars, t, "up")
        mins.append(tm.get("median_minutes"))
    got = [m for m in mins if m is not None]
    check("median time to hit grows with the target",
          all(a <= b + 1e-9 for a, b in zip(got, got[1:])),
          " -> ".join(f"{m:.0f}min" for m in got))

    md = itg.days_to_touch(daily, 1.0)
    check("within-1 <= within-3 <= within-5",
          md["within_1_day"] <= md["within_3_days"] <= md["within_5_days"],
          f"{md['within_1_day']:.0f}% / {md['within_3_days']:.0f}% / {md['within_5_days']:.0f}%")
    md2 = itg.days_to_touch(daily, 2.0)
    check("a bigger target takes at least as long",
          md2["median_days"] >= md["median_days"],
          f"1.0% -> {md['median_days']:.0f}d, 2.0% -> {md2['median_days']:.0f}d")

    print(f"\n{'=' * 72}\n6. costs\n{'=' * 72}")
    w = itg.find_target_window(TICKER, 1.0, daily=daily, cost_pct=0.30)
    check("net = target - cost", abs(w["net_pct"] - 0.70) < 1e-9, f"{w['net_pct']:.2f}%")
    cheap = itg.find_target_window(TICKER, 0.20, daily=daily, cost_pct=0.30)
    check("a target below cost is refused",
          cheap["verdict"] == "costs_exceed_target", f"verdict={cheap['verdict']}")

    print(f"\n{'=' * 72}\n7. a symbol that never moves\n{'=' * 72}")
    idx = pd.bdate_range("2020-01-01", periods=400)
    flat = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                         "Close": 100.0, "Volume": 1000}, index=idx)
    fs = itg.session_touch_stats(flat, 1.0)
    check("flat symbol reports 0% rather than a number",
          fs["p_up"] == 0.0 and fs["p_either"] == 0.0,
          f"up {fs['p_up']:.0f}%, either {fs['p_either']:.0f}%")
    fm = itg.days_to_touch(flat, 1.0)
    check("flat symbol reports no multi-day path either", fm == {} or fm.get("never_pct") == 100.0,
          f"{fm.get('never_pct', 'empty')}")

    print(f"\n{'=' * 72}")
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: " + "; ".join(fails))
        return 1
    print("ALL CHECKS PASSED — the target finder reports measured history, not a guess.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
