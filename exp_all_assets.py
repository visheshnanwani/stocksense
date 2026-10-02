"""
exp_all_assets.py
-----------------
Does training on EVERYTHING -- stocks, metals, energy, forex, crypto, indices,
ETFs -- forecast better than training on one symbol at a time?

THE REASON TO EXPECT A GAIN
    Ten years of daily data is ~2,500 rows per symbol, and at a 21-day horizon
    only ~120 of those outcomes are independent. That is almost nothing, and it
    is why the per-symbol engine so often shrinks itself to near zero: it cannot
    tell signal from noise on that little data. Pool 131 instruments and the
    same model sees ~300,000 rows. Every feature here is stationary by
    construction -- ratios, z-scores, percentile ranks -- so a row from NIFTY
    and a row from copper are on the same scale and can sit in one table.

THE REASON TO DOUBT IT
    Copper is not a bank stock. Pooling assumes the mapping from features to
    next-month return is the SAME across everything in the pool; where it is
    not, the extra rows are not extra information, they are contamination. An
    earlier equities-only version of this was tested and rejected for exactly
    that reason. So the question is not "is more data better" -- it is "WHICH
    pool, for WHICH asset class", and that is settled per class, on held-out
    dates, against the per-symbol model it would replace.

WHAT IS COMPARED  (same features, same test dates, same metric -- only the
training set changes, so nothing else can explain a difference)
    naive         predict zero. The bar every forecast must clear.
    per-symbol    trained on that symbol's own history alone. What ships today.
    pooled-equity the current pooled universe: stocks only.
    pooled-all    every asset class at once. What was asked for.
    pooled-class  one pooled model per asset class (metals with metals, FX
                  with FX). The middle option: more data than one symbol, less
                  contamination than one global model.
    + xrank       pooled-all plus cross-sectional rank features, which only
                  exist in a pooled setting.

HONESTY RULES
    * Split is by DATE, never by row, with the horizon purged at the boundary.
    * Scored per ASSET CLASS, because a gain on 60 US stocks must not be
      allowed to hide a loss on 11 currency pairs.
    * Skill is reported as R^2 against the naive forecast. Negative means the
      model is worse than predicting no change -- which is a normal outcome
      here and is printed, not buried.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import pooled_model as pm
import stock_search as ss

PANEL_CACHE = "panel_all_assets.parquet"
HORIZONS = (1, 5, 21)
START = "2014-01-01"
TEST_FRACTION = 0.15
MIN_TEST_ROWS = 150


# --------------------------------------------------------------- universe
def universe():
    """Every instrument the dashboard can analyse, tagged with its asset class."""
    out = {}
    for row in ss.POPULAR_STOCKS:          # (symbol, name, exchange) tuples
        sym = row[0] if isinstance(row, (tuple, list)) else row
        out[sym] = "stock_in" if sym.endswith(".NS") else "stock_us"
    for sym, cls in ss.ASSET_CLASSES.items():
        out[sym] = cls
    return out


def build_or_load():
    uni = universe()
    if os.path.exists(PANEL_CACHE):
        panel = pd.read_parquet(PANEL_CACHE)
        print(f"loaded cached panel: {len(panel):,} rows, "
              f"{panel.symbol.nunique()} symbols")
    else:
        print(f"building panel over {len(uni)} instruments (this takes a while)...")
        panel = pm.build_panel(sorted(uni), start=START, horizons=HORIZONS, verbose=True)
        panel.to_parquet(PANEL_CACHE, index=False)
        print(f"cached {len(panel):,} rows -> {PANEL_CACHE}")
    panel["cls"] = panel["symbol"].map(uni)
    panel = panel.dropna(subset=["cls"])
    return drop_untrustworthy_ranges(panel)


def drop_untrustworthy_ranges(panel):
    """
    Blank the High/Low-derived features for any symbol whose bars fail the
    range check.

    The first version of this experiment reported 80%+ directional accuracy on
    every freely-floating currency pair. That was not skill: Yahoo's daily FX
    bar has a range that extends past its close, so the next day's price is
    already inside today's high/low (75-85% of the time, against ~45% on a
    well-formed bar) and the wick features carry the answer. S_Lower_Wick_Pct
    correlated 0.490 with the NEXT day's FX return. See market_data.py.
    """
    import data_loader as dl
    import forecast_engine as fe
    import market_data as md

    cols = [c for c in panel.columns if c in fe.RANGE_DERIVED_FEATURES]
    if not cols:
        return panel
    # end=None raises inside the loader ("NaTType does not support strftime").
    # The first version of this swallowed that in a bare except and checked
    # exactly zero symbols, silently, leaving the FX leak in the results. So
    # the end date is explicit now and failures are counted and raised on,
    # because a quality gate that quietly checks nothing is worse than none.
    end = str(pd.Timestamp(panel["date"].max()).date())
    hit, failed = {}, []
    for sym in sorted(panel.symbol.unique()):
        try:
            q = md.ohlc_range_quality(dl.load_stock_data(sym, START, end), sym)
        except Exception as ex:
            failed.append(f"{sym} ({type(ex).__name__})")
            continue
        if q["verdict"] != "ok":
            hit[sym] = q["verdict"]
    checked = panel.symbol.nunique() - len(failed)
    print(f"  range check: {checked}/{panel.symbol.nunique()} symbols checked, "
          f"{len(hit)} failed the check")
    if failed:
        print(f"  COULD NOT CHECK {len(failed)}: {', '.join(failed[:10])}")
    if checked < panel.symbol.nunique() * 0.9:
        raise RuntimeError(f"only {checked} symbols could be range-checked -- "
                           f"refusing to report numbers that may still contain the leak")
    if hit:
        mask = panel.symbol.isin(hit)
        panel.loc[mask, cols] = np.nan
        by = {}
        for sym, v in hit.items():
            by.setdefault(v, []).append(sym)
        for v, syms in sorted(by.items()):
            print(f"  range features blanked ({v}) on {len(syms)}: {', '.join(sorted(syms))}")
    return panel


# --------------------------------------------------------------- scoring
def skill(y_true, y_pred):
    """R^2 against the naive zero forecast, in percent. >0 beats doing nothing."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 30:
        return np.nan, np.nan, 0
    y_true, y_pred = y_true[ok], y_pred[ok]
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum(y_true ** 2)          # naive = 0
    r2 = (1 - sse / sst) * 100 if sst > 0 else np.nan
    moved = np.abs(y_pred) > 1e-6
    hit = (np.sign(y_pred[moved]) == np.sign(y_true[moved])).mean() * 100 if moved.sum() > 20 else np.nan
    return r2, hit, int(ok.sum())


def fit(train, cols, target, kind="gbm", cap=250_000):
    # Only the TARGET must be present. Gradient boosting handles missing
    # features natively, and requiring all of them silently deleted every FX
    # row from this experiment: Yahoo reports no volume for currency pairs, so
    # all six volume indicators are NaN for all 34,310 of them. Dropping those
    # rows left nothing to train or score on, which is why forex came back
    # blank. Ridge cannot cope with NaN, so it gets the columns mean-filled.
    d = train[cols + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    d = d.loc[:, [c for c in d.columns if d[c].notna().any()]]
    cols = [c for c in cols if c in d.columns]
    if len(d) < 400 or not cols:
        return None
    if len(d) > cap:
        d = d.sample(cap, random_state=0)
    y = d[target].values
    if kind == "ridge":
        X = d[cols].fillna(d[cols].mean()).fillna(0.0).values
        m = Ridge(alpha=10.0).fit(X, y)
    else:
        m = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.03, max_iter=350, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0).fit(d[cols].values, y)
    m._cols = cols            # the columns that survived, for predict()
    return m


def predict(model, frame, cols):
    if model is None:
        return np.full(len(frame), np.nan)
    use = getattr(model, "_cols", cols)
    X = frame.reindex(columns=use).replace([np.inf, -np.inf], np.nan)
    if isinstance(model, Ridge):
        X = X.fillna(0.0)
    return model.predict(X.values)


# --------------------------------------------------------------- the run
def main():
    panel = build_or_load()
    base_cols = [c for c in panel.columns
                 if c not in ("symbol", "date", "cls") and not c.startswith("y_")]
    base_cols = [c for c in base_cols if panel[c].notna().mean() > 0.6]
    print(f"{len(base_cols)} features, classes: "
          f"{dict(panel.groupby('cls')['symbol'].nunique())}\n")

    xpanel = pm.add_cross_section(panel, base_cols)
    xcols = base_cols + [c + pm.CROSS_SECTION_SUFFIX for c in base_cols]
    xcols = [c for c in xcols if c in xpanel.columns and xpanel[c].notna().mean() > 0.5]

    rows = []
    for h in HORIZONS:
        target = f"y_{h}"
        train, test = pm.split_by_date(panel, TEST_FRACTION, horizon=h)
        xtrain, xtest = pm.split_by_date(xpanel, TEST_FRACTION, horizon=h)
        print(f"=== horizon {h}d ===  train {len(train):,} rows "
              f"(to {train.date.max():%d %b %Y}), test {len(test):,} rows "
              f"(from {test.date.min():%d %b %Y})", flush=True)

        # Pools grouped for an economic reason, not just "everything at once":
        # instruments driven by the same forces belong in the same pool.
        EQUITYLIKE = ["stock_us", "stock_in", "etf", "index"]
        MACRO = ["commodity", "forex", "index"]
        models = {
            "pooled-all": (fit(train, base_cols, target), base_cols, panel),
            "pooled-equity": (fit(train[train.cls.isin(["stock_us", "stock_in"])],
                                  base_cols, target), base_cols, panel),
            "pooled-equitylike": (fit(train[train.cls.isin(EQUITYLIKE)], base_cols, target),
                                  base_cols, panel),
            "pooled-macro": (fit(train[train.cls.isin(MACRO)], base_cols, target),
                             base_cols, panel),
            "pooled-all+xrank": (fit(xtrain, xcols, target), xcols, xpanel),
        }
        per_class = {c: fit(train[train.cls == c], base_cols, target)
                     for c in sorted(panel.cls.unique())}

        for cls in sorted(panel.cls.unique()):
            te = test[test.cls == cls]
            if len(te) < MIN_TEST_ROWS:
                continue
            rec = {"h": h, "class": cls, "symbols": te.symbol.nunique(), "rows": len(te)}

            # per-symbol: one model per symbol, trained on that symbol only
            ps_pred, ps_true = [], []
            for sym, g in te.groupby("symbol"):
                m = fit(train[train.symbol == sym], base_cols, target)
                if m is None:
                    continue
                ps_pred.append(predict(m, g, base_cols))
                ps_true.append(g[target].values)
            if ps_pred:
                r2, hit, n = skill(np.concatenate(ps_true), np.concatenate(ps_pred))
                rec["per-symbol"], rec["per-symbol_hit"] = r2, hit

            r2, hit, _ = skill(te[target].values, predict(per_class[cls], te, base_cols))
            rec["pooled-class"], rec["pooled-class_hit"] = r2, hit

            for name, (m, cols, src) in models.items():
                frame = src[(src.cls == cls) & (src.date >= te.date.min())]
                frame = frame[frame.date.isin(te.date.unique())]
                r2, hit, _ = skill(frame[target].values, predict(m, frame, cols))
                rec[name], rec[name + "_hit"] = r2, hit

            rows.append(rec)
            print(f"  {cls:10}  " + "  ".join(
                f"{k} {rec.get(k, float('nan')):+.2f}%"
                for k in ("per-symbol", "pooled-class", "pooled-equitylike",
                          "pooled-macro", "pooled-all")), flush=True)
        print()

    r = pd.DataFrame(rows)
    r.to_csv("exp_all_assets.csv", index=False)

    variants = ["per-symbol", "pooled-class", "pooled-equity", "pooled-equitylike",
                "pooled-macro", "pooled-all", "pooled-all+xrank"]
    print("=" * 100)
    print("SKILL vs DOING NOTHING  (R^2 %, positive = beats predicting no change)")
    print("=" * 100)
    print(f"{'horizon':<9}{'class':<12}" + "".join(f"{v:>19}" for v in variants))
    for _, x in r.iterrows():
        print(f"{x['h']:<9}{x['class']:<12}" +
              "".join(f"{x.get(v, float('nan')):>19.2f}" for v in variants))

    print("\n" + "=" * 100)
    print("WHICH TRAINING SET WINS, PER ASSET CLASS  (averaged over horizons)")
    print("=" * 100)
    g = r.groupby("class")[variants].mean()
    print(f"{'class':<12}" + "".join(f"{v:>19}" for v in variants) + "    best")
    for cls, row in g.iterrows():
        best = row.idxmax() if row.notna().any() else "n/a"
        print(f"{cls:<12}" + "".join(f"{row[v]:>19.2f}" for v in variants) + f"    {best}")

    print("\n" + "=" * 100)
    print("DIRECTIONAL ACCURACY  (% of non-flat calls with the right sign)")
    print("=" * 100)
    hv = [v + "_hit" for v in variants]
    gh = r.groupby("class")[hv].mean()
    print(f"{'class':<12}" + "".join(f"{v:>19}" for v in variants))
    for cls, row in gh.iterrows():
        print(f"{cls:<12}" + "".join(f"{row[v + '_hit']:>18.1f}%" for v in variants))

    print("\nwrote exp_all_assets.csv")
    return r


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
