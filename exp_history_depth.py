"""
exp_history_depth.py
--------------------
The other way to "train on all the data we have": not more INSTRUMENTS, more
YEARS of each one.

The dashboard loads from 2014 by default -- about twelve years. Yahoo will serve
far more than that for most listed names (AAPL back to 1980). So: does starting
earlier make the forecast better?

It is not obvious that it should. More rows cut variance, which helps. But a
2008-era relationship between, say, volatility and next-month return may simply
not be the relationship that holds now, and training on it biases the model
toward a market that no longer exists. Those two effects pull in opposite
directions and only measurement settles which wins.

The test holds everything fixed except the training START date, scores on the
SAME held-out dates for every variant, and reports per asset class.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import exp_all_assets as E
import pooled_model as pm

STARTS = ["2004-01-01", "2010-01-01", "2014-01-01", "2018-01-01", "2021-01-01"]
HORIZONS = (5, 21)
PANEL = "panel_deep_history.parquet"


def build():
    import os
    uni = E.universe()
    if os.path.exists(PANEL):
        panel = pd.read_parquet(PANEL)
        print(f"loaded cached deep panel: {len(panel):,} rows")
    else:
        print(f"building a DEEP panel from {STARTS[0]} over {len(uni)} instruments...")
        panel = pm.build_panel(sorted(uni), start=STARTS[0], horizons=HORIZONS, verbose=True)
        panel.to_parquet(PANEL, index=False)
        print(f"cached {len(panel):,} rows -> {PANEL}")
    panel["cls"] = panel["symbol"].map(uni)
    return panel.dropna(subset=["cls"])


def main():
    panel = build()
    cols = [c for c in panel.columns
            if c not in ("symbol", "date", "cls") and not c.startswith("y_")]
    cols = [c for c in cols if panel[c].notna().mean() > 0.6]

    span = panel.groupby("cls")["date"].agg(["min", "max"])
    print("\nhistory actually available, by class:")
    for cls, r in span.iterrows():
        print(f"  {cls:10} {r['min']:%Y-%m-%d} -> {r['max']:%Y-%m-%d}")

    rows = []
    for h in HORIZONS:
        target = f"y_{h}"
        _, test = pm.split_by_date(panel, E.TEST_FRACTION, horizon=h)
        cut = test.date.min()
        print(f"\n=== horizon {h}d ===  test from {cut:%d %b %Y} "
              f"({len(test):,} rows) -- identical for every variant")
        for start in STARTS:
            tr = panel[(panel.date >= start) &
                       (panel.date < cut - pd.Timedelta(days=int(h * 1.6)))]
            model = E.fit(tr, cols, target)
            rec = {"h": h, "start": start[:4], "train_rows": len(tr),
                   "train_years": round((cut - pd.Timestamp(start)).days / 365.25, 1)}
            for cls in sorted(panel.cls.unique()):
                te = test[test.cls == cls]
                if len(te) < E.MIN_TEST_ROWS:
                    continue
                r2, _hit, _n = E.skill(te[target].values, E.predict(model, te, cols))
                rec[cls] = r2
            rows.append(rec)
            print(f"  from {start[:4]}  {len(tr):>9,} rows  " + "  ".join(
                f"{c} {rec.get(c, float('nan')):+.2f}%" for c in sorted(panel.cls.unique())
                if c in rec), flush=True)

    r = pd.DataFrame(rows)
    r.to_csv("exp_history_depth.csv", index=False)

    classes = [c for c in sorted(panel.cls.unique()) if c in r.columns]
    print("\n" + "=" * 104)
    print("DOES STARTING EARLIER HELP?  (skill vs doing nothing, averaged over horizons)")
    print("=" * 104)
    g = r.groupby("start")[classes].mean()
    print(f"{'train from':<12}" + "".join(f"{c:>13}" for c in classes))
    for start, row in g.iterrows():
        print(f"{start:<12}" + "".join(f"{row[c]:>13.2f}" for c in classes))
    print(f"\n{'best start':<12}" + "".join(f"{g[c].idxmax():>13}" for c in classes))
    print(f"{'vs 2014':<12}" + "".join(
        f"{g[c].max() - g.loc['2014', c]:>+13.2f}" for c in classes))
    print("\nwrote exp_history_depth.csv")
    return r


if __name__ == "__main__":
    main()
