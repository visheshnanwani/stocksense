"""
test_window_scan.py
-------------------
The buy/sell scan used to recommend a holding period that was really just the
slider value: it bought on day 1 and sold on the last day of whatever window you
asked for, because the forecast path is near-monotonic so argmin is the first day
and argmax the last. Measured on HDFCBANK.NS before the fix:

    slider  5 -> hold   6d, "gain" 0.14%
    slider 30 -> hold  41d, "gain" 0.64%
    slider 60 -> hold  83d, "gain" 2.10%
    slider 90 -> hold 125d, "gain" 3.23%

Same buy date every time. That is the bug this file exists to prevent.

CHECKS
  1. the recommendation does not simply track the slider
  2. costs are subtracted -- net is always below gross
  3. a window is never recommended below the probability floor
  4. raising the cost can only reduce the number of tradeable windows
  5. the hold's uncertainty is used, not the price level's
  6. a flat forecast with no edge returns "no trade" rather than inventing one

Run:  python test_window_scan.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import data_loader as dl
import forecast_engine as fe
import market_data as md

TICKER = "HDFCBANK.NS"
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main():
    raw = dl.load_stock_data(TICKER, "2014-01-01", None or pd.Timestamp.today().strftime("%Y-%m-%d"))
    raw, _ = md.repair_market_data(raw, TICKER)
    eng = fe.ForecastEngine(raw, TICKER).fit()

    print(f"\n{'=' * 70}\n1. the recommendation must not track the slider\n{'=' * 70}")
    holds, picks = {}, {}
    for h in (30, 45, 60, 90):
        s = eng.scan_best_window(h)
        b = s.get("best_short")
        holds[h] = b["hold_days"] if b else None
        picks[h] = (b["buy_date"], b["sell_date"]) if b else None
        print(f"  slider {h:>3}: verdict={s['verdict']:<10} "
              f"hold={holds[h]}d  qualifying={s['qualifying']:,}/{s['candidates']:,}")
    tradeable = [h for h in holds if holds[h] is not None]
    # the old code produced a strictly increasing hold; the new one must not
    seq = [holds[h] for h in sorted(tradeable)]
    strictly_tracking = len(seq) > 2 and all(b > a for a, b in zip(seq, seq[1:]))
    check("hold length does not rise with every slider step", not strictly_tracking,
          f"holds={seq}")
    big = [picks[h] for h in (60, 90) if picks[h]]
    if len(big) == 2:
        check("sliders 60 and 90 agree on the same window", big[0] == big[1],
              f"{big[0][0]:%d %b}->{big[0][1]:%d %b} vs {big[1][0]:%d %b}->{big[1][1]:%d %b}")

    print(f"\n{'=' * 70}\n2-5. costs, probability floor, monotonicity\n{'=' * 70}")
    s = eng.scan_best_window(60, cost_pct=0.30, min_prob=0.55)
    fr = s["frontier"]
    check("net is always below gross", all(c["net_pct"] < c["gross_pct"] for c in fr),
          f"across {len(fr)} holding lengths")
    worst_gap = max(c["gross_pct"] - c["net_pct"] for c in fr)
    best_gap = min(c["gross_pct"] - c["net_pct"] for c in fr)
    check("the cost subtracted equals the round trip", abs(worst_gap - 0.30) < 1e-6 and abs(best_gap - 0.30) < 1e-6,
          f"gap {best_gap:.4f}..{worst_gap:.4f} vs 0.30")
    if s["verdict"] == "tradeable":
        check("recommended window clears the probability floor",
              s["best_short"]["prob_profit"] >= 55.0,
              f"{s['best_short']['prob_profit']:.1f}% >= 55%")
        check("recommended window is profitable after costs",
              s["best_short"]["net_pct"] > 0, f"{s['best_short']['net_pct']:+.2f}%")

    cheap = eng.scan_best_window(60, cost_pct=0.05, min_prob=0.55)["qualifying"]
    dear = eng.scan_best_window(60, cost_pct=0.90, min_prob=0.55)["qualifying"]
    check("higher costs can only reduce tradeable windows", dear <= cheap,
          f"0.05% -> {cheap:,} windows, 0.90% -> {dear:,}")
    strict = eng.scan_best_window(60, cost_pct=0.30, min_prob=0.75)["qualifying"]
    check("a stricter probability floor can only reduce them too", strict <= s["qualifying"],
          f"55% -> {s['qualifying']:,}, 75% -> {strict:,}")

    # the uncertainty used must be the HOLD's, not the level's: a 5-day hold
    # starting 50 days out must not inherit 55 days of accumulated uncertainty
    path = s["forecast_path"]
    sig = (path["Upper_80"] - path["Lower_80"]).to_numpy() / (2 * 1.2816)
    late = [c for c in s["frontier"] if c["hold_steps"] == 5]
    if late and len(sig) > 55:
        lvl = sig[55] / float(path["Predicted_Close"].iloc[55])
        c = late[0]
        implied = abs(c["net_pct"] / 100) / max(abs(c["t_stat"]), 1e-9)
        check("hold uncertainty is smaller than level uncertainty", implied < lvl,
              f"hold sd {implied * 100:.2f}% vs level sd {lvl * 100:.2f}%")

    print(f"\n{'=' * 70}\n6. a forecast with no edge must return 'no trade'\n{'=' * 70}")

    class FlatEngine(fe.ForecastEngine):
        def forecast_path(self, n):
            idx = pd.bdate_range(pd.Timestamp.today().normalize(), periods=n)
            px = np.full(n, 100.0)                       # dead flat: no edge at all
            half = 1.2816 * 2.0 * np.sqrt(np.arange(1, n + 1))
            return pd.DataFrame({"Predicted_Close": px,
                                 "Lower_80": px - half, "Upper_80": px + half}, index=idx)

    flat = FlatEngine.__new__(FlatEngine)
    out = fe.ForecastEngine.scan_best_window(flat, 60)
    check("flat forecast -> no_trade", out["verdict"] == "no_trade",
          f"verdict={out['verdict']}, qualifying={out['qualifying']}")

    print(f"\n{'=' * 70}")
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: " + "; ".join(fails))
        return 1
    print("ALL CHECKS PASSED — the recommendation is a finding, not the slider.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
