"""
exp_contrarian.py
-----------------
The selectivity run found something uncomfortable: expectancy falls MONOTONICALLY
as the model's conviction rises.

    decile  1 (|signal| 0.009%)   -0.109%/trade   44.5% win
    decile 10 (|signal| 0.531%)   -0.621%/trade   32.5% win

At zero cost the top decile still loses 0.471%. The model is not merely useless
when confident, it is anti-predictive: the confident calls are wrong more often
than a coin.

The obvious move is to fade it -- take the opposite side when conviction is high.
The obvious move is also exactly how people fool themselves, because the pattern
was FOUND in this data. So this file does not ask "does fading work on the data
that suggested it". It asks:

    decide on the TRAINING half -- follow or fade, and at what conviction
    apply that decision to a held-out half the decision never saw
    then apply it again to a second, later, untouched period

A finding that survives only the first test is a finding about one split. The
verdict at the bottom is whatever the numbers say, including "this is noise".

Costs are the corrected intraday round trip (0.15% all-in), not the 0.30%
delivery rate used earlier.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

COST = 0.15


def load():
    df = pd.read_csv("exp_selectivity.csv")
    # close_only is already signed by the model's side; raw = what the day did
    # if you went long. Recover it so the side can be flipped cleanly.
    df["raw_long"] = np.where(df["pred"] > 0, df["close_only"], -df["close_only"])
    return df


def evaluate(train, test, q):
    """Decide follow-or-fade + threshold on `train`, score on `test`."""
    if len(train) < 200 or len(test) < 60:
        return None
    thr = train["abs_signal"].quantile(q)
    tr_sel = train[train["abs_signal"] >= thr]
    if len(tr_sel) < 40:
        return None
    # what the training half says about following the signal
    follow_tr = np.where(tr_sel["pred"] > 0, tr_sel["raw_long"], -tr_sel["raw_long"]).mean()
    decision = "fade" if follow_tr < 0 else "follow"

    te_sel = test[test["abs_signal"] >= thr]
    if len(te_sel) < 20:
        return None
    side = np.sign(te_sel["pred"]) * (-1 if decision == "fade" else 1)
    ret = side * te_sel["raw_long"] - COST
    return {"decision": decision, "thr": thr, "n_test": len(te_sel),
            "taken_pct": len(te_sel) / len(test) * 100,
            "expectancy": float(ret.mean()), "win_rate": float((ret > 0).mean() * 100),
            "total": float(ret.sum()),
            "follow_on_train": float(follow_tr)}


def main():
    df = load()
    print(f"{len(df):,} out-of-fold sessions, {df.ticker.nunique()} symbols, "
          f"{COST:.2f}% round trip\n")

    for q, label in ((0.90, "top 10% conviction"), (0.75, "top 25% conviction")):
        print("=" * 92)
        print(f"{label.upper()}  —  decide on the first half, apply to the second")
        print("=" * 92)
        print(f"{'ticker':<13}{'decision':>9}{'train says':>12}{'n':>6}{'% taken':>9}"
              f"{'expectancy':>13}{'win rate':>10}")
        rows = []
        for t, g in df.groupby("ticker"):
            g = g.reset_index(drop=True)
            mid = len(g) // 2
            r = evaluate(g.iloc[:mid], g.iloc[mid:], q)
            if not r:
                continue
            r["ticker"] = t
            rows.append(r)
            print(f"{t:<13}{r['decision']:>9}{r['follow_on_train']:>+11.3f}%"
                  f"{r['n_test']:>6}{r['taken_pct']:>8.0f}%"
                  f"{r['expectancy']:>+12.3f}%{r['win_rate']:>9.1f}%")
        res = pd.DataFrame(rows)
        if res.empty:
            print("  no usable splits")
            continue
        print(f"\n  pooled expectancy {res.expectancy.mean():+.3f}%/trade")
        print(f"  profitable in {(res.expectancy > 0).sum()}/{len(res)} symbols")
        print(f"  training half said FADE for {(res.decision == 'fade').sum()}/{len(res)}")
        print()

    # ---------------- second, independent period -------------------------
    print("=" * 92)
    print("SECOND TEST — three consecutive periods: decide on A, apply to B, then apply to C")
    print("=" * 92)
    print(f"{'ticker':<13}{'decision(A)':>12}{'B expectancy':>15}{'C expectancy':>15}"
          f"{'both +?':>9}")
    rows = []
    for t, g in df.groupby("ticker"):
        g = g.reset_index(drop=True)
        a, b = len(g) // 3, 2 * len(g) // 3
        A, B, C = g.iloc[:a], g.iloc[a:b], g.iloc[b:]
        rb = evaluate(A, B, 0.90)
        rc = evaluate(A, C, 0.90)
        if not rb or not rc:
            continue
        both = rb["expectancy"] > 0 and rc["expectancy"] > 0
        rows.append({"ticker": t, "decision": rb["decision"],
                     "B": rb["expectancy"], "C": rc["expectancy"], "both": both})
        print(f"{t:<13}{rb['decision']:>12}{rb['expectancy']:>+14.3f}%"
              f"{rc['expectancy']:>+14.3f}%{'yes' if both else 'no':>9}")
    r2 = pd.DataFrame(rows)
    if not r2.empty:
        print(f"\n  period B pooled {r2.B.mean():+.3f}%   period C pooled {r2.C.mean():+.3f}%")
        print(f"  positive in BOTH later periods: {int(r2.both.sum())}/{len(r2)} symbols")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    if not r2.empty and r2.B.mean() > 0 and r2.C.mean() > 0 and r2.both.sum() >= len(r2) * 0.6:
        print("  Fading high-conviction signals survived two independent out-of-sample")
        print("  periods. Worth shipping, with the caveat that it was found by looking.")
    else:
        print("  It does NOT hold up. The monotone decile pattern is real in-sample but")
        print("  does not survive an honest out-of-sample test, which is the signature of")
        print("  a pattern mined from noise rather than an edge. Not shipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
