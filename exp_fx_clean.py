"""
exp_fx_clean.py
---------------
Does rebuilding the FX daily bar from hourly data actually buy any accuracy?

The rebuilt bars are unquestionably better DATA -- every pair goes from
"leaking" to "ok" on the range check, with the close outside its own high/low
falling from 2-14% to 0.0%. That is a fact about the data and it is already
settled (fx_clean_bars.py).

Whether it makes the FORECAST better is a different question, and the answer is
not obviously yes. Clean high/low data restores 34 features, but those features
have to earn their place against the 116 close-only ones that were working
fine, and the rebuilt history is only ~2.8 years deep against the 12.7 the
daily feed gives.

THE COMPARISON  (all three on the SAME dates, so span cannot explain a result)
    close-only        what ships after the leak guard: the 34 range features
                      dropped, everything else kept
    clean-range       the same, plus the 34 features rebuilt from hourly bars
    yahoo-range       the same, plus the 34 features as Yahoo publishes them --
                      the leaking version, included ONLY as a control. If it
                      "wins", that is the leak talking and proves the point.

The window is restricted to where the rebuilt bars exist, for every variant
equally. Otherwise close-only would get twelve years and clean-range three, and
the comparison would measure sample size instead of data quality.
"""

import warnings

warnings.filterwarnings("ignore")

import datetime

import numpy as np
import pandas as pd

import data_loader as dl
import exp_all_assets as E
import forecast_engine as fe
import fx_clean_bars as fx
from features import build_stationary_features, STATIONARY_FEATURE_COLUMNS
from features_indicators import indicator_features

HORIZONS = (1, 5, 21)
TEST_FROM = "2024-12-20"        # the same boundary the main experiment uses


def featurise(raw, symbol, horizons=HORIZONS):
    base = build_stationary_features(raw, keep_latest=False)
    ind = indicator_features(raw).reindex(base.index)
    f = base[[c for c in STATIONARY_FEATURE_COLUMNS if c in base.columns]].join(ind)
    close = raw["Close"].reindex(f.index)
    for h in horizons:
        f[f"y_{h}"] = close.shift(-h) / close - 1
    f["symbol"] = symbol
    f["date"] = f.index
    return f.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)


def build():
    """Three panels over the same pairs and the same dates."""
    end = str(datetime.date.today())
    rows = {"close-only": [], "clean-range": [], "yahoo-range": []}
    for sym in fx.FX_PAIRS:
        y = dl.load_stock_data(sym, "2014-01-01", end)
        clean = fx.clean_daily(sym)
        if clean is None or clean.empty:
            print(f"  {sym:10} no hourly rebuild -- skipped")
            continue
        # The rebuilt bar is used WHOLE -- its own open, high, low and close.
        # An earlier version kept Yahoo's close and took only the rebuilt
        # high/low, which put two clocks in one bar and leaked worse than the
        # original (next-in-range 90% vs 75%). See fx_clean_bars.splice.
        rows["yahoo-range"].append(featurise(y, sym))
        rows["clean-range"].append(featurise(clean, sym))
        co = featurise(y, sym)
        for c in [c for c in co.columns if c in fe.RANGE_DERIVED_FEATURES]:
            co[c] = np.nan
        rows["close-only"].append(co)
        print(f"  {sym:10} rebuilt {len(clean):4} daily bars "
              f"({clean.index.min():%b %Y} -> {clean.index.max():%b %Y})", flush=True)

    panels = {k: pd.concat(v, ignore_index=True) for k, v in rows.items() if v}
    # restrict every panel to where the rebuilt bars exist
    first = max(p[[c for c in p.columns if c in fe.RANGE_DERIVED_FEATURES]]
                .notna().any(axis=1).pipe(lambda m: p.loc[m, "date"].min())
                for k, p in panels.items() if k == "clean-range")
    for k in panels:
        panels[k] = panels[k][panels[k]["date"] >= first]
    print(f"\n  all variants restricted to {first:%d %b %Y} onward "
          f"({len(panels['clean-range']):,} rows each)")
    return panels


def main():
    print("rebuilding FX bars from hourly data...")
    panels = build()
    cut = pd.Timestamp(TEST_FROM)

    results = []
    for h in HORIZONS:
        target = f"y_{h}"
        for name, p in panels.items():
            cols = [c for c in p.columns
                    if c not in ("symbol", "date") and not c.startswith("y_")]
            cols = [c for c in cols if p[c].notna().mean() > 0.3]
            tr = p[p.date < cut - pd.Timedelta(days=int(h * 1.6))]
            te = p[p.date >= cut]
            m = E.fit(tr, cols, target)
            r2, hit, n = E.skill(te[target].values, E.predict(m, te, cols))
            results.append({"h": h, "variant": name, "feats": len(cols),
                            "train": len(tr), "test": n, "r2": r2, "hit": hit})
            print(f"  h={h:<3} {name:<13} {len(cols):>4} features  "
                  f"train {len(tr):>6,}  R2 {r2:>+8.2f}%  direction {hit:>5.1f}%", flush=True)

    r = pd.DataFrame(results)
    r.to_csv("exp_fx_clean.csv", index=False)

    print("\n" + "=" * 86)
    print("DOES THE CLEAN REBUILD HELP THE FORECAST?")
    print("=" * 86)
    piv = r.pivot(index="h", columns="variant", values="r2")
    hitp = r.pivot(index="h", columns="variant", values="hit")
    order = ["close-only", "clean-range", "yahoo-range"]
    order = [c for c in order if c in piv.columns]
    print(f"{'horizon':<9}" + "".join(f"{c:>16}" for c in order) + "     (R^2 %)")
    for h_, row in piv.iterrows():
        print(f"{h_:<9}" + "".join(f"{row[c]:>16.2f}" for c in order))
    print(f"\n{'horizon':<9}" + "".join(f"{c:>16}" for c in order) + "     (direction %)")
    for h_, row in hitp.iterrows():
        print(f"{h_:<9}" + "".join(f"{row[c]:>15.1f}%" for c in order))

    if {"close-only", "clean-range"} <= set(piv.columns):
        d = (piv["clean-range"] - piv["close-only"]).mean()
        dh = (hitp["clean-range"] - hitp["close-only"]).mean()
        print(f"\n  clean rebuild vs close-only:  R^2 {d:+.2f} points, "
              f"direction {dh:+.1f} points")
        print("  -> " + ("the rebuilt range features EARN their place."
                         if d > 0.25 or dh > 0.5 else
                         "the rebuilt range features do NOT add forecasting value; "
                         "they are better data that the model cannot use."))
    if "yahoo-range" in piv.columns:
        print(f"\n  control: Yahoo's leaking range scores "
              f"{piv['yahoo-range'].mean():+.2f} vs clean {piv['clean-range'].mean():+.2f}. "
              f"\n  A big edge for the leaking version is the look-ahead, not skill.")
    return r


if __name__ == "__main__":
    main()
