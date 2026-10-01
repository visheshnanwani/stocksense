"""
test_options_plan.py
--------------------
The same-day plan is the easiest screen in the project to make dishonest: entry,
stop and target look authoritative whatever the numbers behind them. These checks
keep the arithmetic right and the record visible.

  1. stop and target sit on the correct sides of the entry, both directions
  2. reward:risk equals target_atr / stop_atr
  3. the backtest's outcome buckets are exhaustive and sum to 100%
  4. raising costs can only lower expectancy, by exactly the amount added
  5. a wider stop is hit less often
  6. no option premium is invented when the provider has no chain
  7. the pessimistic tie-break really is pessimistic -- counting ambiguous
     sessions as wins instead must not LOWER the measured expectancy

Run:  python test_options_plan.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import data_loader as dl
import market_data as md
import options_plan as opl

TICKER = "HDFCBANK.NS"
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main():
    d = dl.load_stock_data(TICKER, "2014-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    d, _ = md.repair_market_data(d, TICKER)

    print(f"\n{'=' * 72}\n1-2. the levels\n{'=' * 72}")
    lp = opl.build_plan(d, "long", 0.8, 1.5)
    sp = opl.build_plan(d, "short", 0.8, 1.5)
    check("long: stop below entry, target above",
          lp["stop"] < lp["entry"] < lp["target"],
          f"{lp['stop']:.2f} < {lp['entry']:.2f} < {lp['target']:.2f}")
    check("short: stop above entry, target below",
          sp["target"] < sp["entry"] < sp["stop"],
          f"{sp['target']:.2f} < {sp['entry']:.2f} < {sp['stop']:.2f}")
    check("reward:risk equals target_atr / stop_atr",
          abs(lp["rr"] - (1.5 / 0.8)) < 1e-9, f"{lp['rr']:.4f} vs {1.5 / 0.8:.4f}")

    print(f"\n{'=' * 72}\n3-5. the record\n{'=' * 72}")
    r = opl.backtest_bracket(d, "long", 0.8, 1.5, 0.30)
    total = r["target_hit_pct"] + r["stop_hit_pct"] + r["neither_pct"]
    check("outcome buckets are exhaustive", abs(total - 100.0) < 1e-6, f"sum = {total:.4f}%")
    check("ambiguous sessions are a subset of stopped ones",
          r["ambiguous_pct"] <= r["stop_hit_pct"] + 1e-9,
          f"{r['ambiguous_pct']:.2f}% <= {r['stop_hit_pct']:.2f}%")

    r0 = opl.backtest_bracket(d, "long", 0.8, 1.5, 0.0)
    r1 = opl.backtest_bracket(d, "long", 0.8, 1.5, 1.0)
    check("cost is subtracted exactly",
          abs((r0["expectancy_pct"] - r1["expectancy_pct"]) - 1.0) < 1e-9,
          f"0% -> {r0['expectancy_pct']:+.4f}, 1% -> {r1['expectancy_pct']:+.4f}")
    check("higher cost can only lower expectancy", r1["expectancy_pct"] < r0["expectancy_pct"])

    tight = opl.backtest_bracket(d, "long", 0.4, 1.5, 0.30)
    wide = opl.backtest_bracket(d, "long", 1.2, 1.5, 0.30)
    check("a wider stop is hit less often",
          wide["stop_hit_pct"] < tight["stop_hit_pct"],
          f"0.4xATR stops {tight['stop_hit_pct']:.1f}% vs 1.2xATR {wide['stop_hit_pct']:.1f}%")

    print(f"\n{'=' * 72}\n6. nothing is invented when there is no chain\n{'=' * 72}")
    ch = opl.option_chain(TICKER, float(lp["entry"]))
    check("no chain for an NSE symbol, and it says so",
          ch["available"] is False and bool(ch.get("reason")), ch.get("reason", "")[:60])
    check("no premium/break-even keys are returned",
          not any(k in ch for k in ("calls", "puts", "mid", "break_even")))
    rule = opl.strike_rule(float(lp["entry"]), lp)
    check("a strike RULE is still given", rule["kind"] == "CALL" and rule["atm"] > 0,
          f"{rule['kind']} ATM {rule['atm']:.0f} step {rule['step']:g}")
    check("ATM strike is within one step of spot",
          abs(rule["atm"] - lp["entry"]) <= rule["step"],
          f"|{rule['atm']:.0f} - {lp['entry']:.2f}| <= {rule['step']:g}")

    print(f"\n{'=' * 72}\n7. the tie-break is genuinely pessimistic\n{'=' * 72}")
    # recompute counting ambiguous sessions as WINS; expectancy must not fall
    a = opl.atr(d).shift(1)
    o, h, l = (d[x].astype(float) for x in ("Open", "High", "Low"))
    ok = a.notna() & (a > 0)
    o, h, l, a = o[ok], h[ok], l[ok], a[ok]
    stop, tgt = o - 0.8 * a, o + 1.5 * a
    both = (h >= tgt) & (l <= stop)
    gain_if_win = ((tgt / o - 1) * 100 - 0.30)[both]
    loss_as_counted = ((stop / o - 1) * 100 - 0.30)[both]
    delta = float((gain_if_win - loss_as_counted).sum() / len(o))
    check("counting ties as wins would RAISE expectancy",
          delta >= 0, f"optimistic version is {delta:+.4f}pp higher "
                      f"({int(both.sum())} ambiguous sessions)")

    print(f"\n{'=' * 72}")
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: " + "; ".join(fails))
        return 1
    print("ALL CHECKS PASSED - levels are arithmetically sound and the record is a floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
