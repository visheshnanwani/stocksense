"""
pooled_model.py
-----------------
Learning across a whole universe at once, instead of one symbol at a time.

The single biggest limitation of the per-symbol engine is sample size: ten years
of daily data is about 2,500 rows, and with a 21-day horizon only ~120 of those
outcomes are independent. No learner can extract much from that, which is most of
why the per-symbol models end up shrinking themselves to near-zero.

A pooled model sidesteps it. Forty symbols give ~100,000 rows, and because every
feature in this project is stationary by construction (ratios, z-scores,
percentile ranks), a row from RELIANCE and a row from MSFT are directly
comparable. The model learns "what a stock that looks like THIS tends to do
next", pooled across the universe, which is how cross-sectional equity models are
actually built.

Two targets are supported, because they answer different questions:

  absolute : the h-day return itself. Directly comparable to the per-symbol
             engine, and what the dashboard needs for a price forecast.
  relative : the h-day return minus the universe median that day. This strips
             out "the whole market went up", leaving the part a stock-picker can
             actually act on -- historically far more learnable than absolute.

Cross-sectional context is added per date (each feature's rank within the
universe that day), which only exists in a pooled setting and is where much of
the edge in this style of model comes from.

Everything is split by DATE, never by row, so the test period is strictly after
everything the model saw.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

EPS = 1e-12

# A deliberately broad universe: US mega-caps, Indian large-caps, and a few
# non-equities so the model sees more than one market's behaviour.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "WMT",
    "XOM", "JNJ", "PG", "KO", "DIS", "INTC", "AMD", "ORCL", "CSCO", "PFE",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "ITC.NS", "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "WIPRO.NS", "HCLTECH.NS", "NTPC.NS", "ONGC.NS", "TATASTEEL.NS", "BHARTIARTL.NS",
]

CROSS_SECTION_SUFFIX = "_xrank"


def build_panel(symbols, start="2014-01-01", end=None, horizons=(1, 5, 21), verbose=True):
    """
    One long DataFrame: (date, symbol) rows, stationary features, forward returns.

    Every feature comes from features_indicators (already stationary), plus the
    engine's own stationary price/volume block, so nothing symbol-specific like a
    price level can leak in.
    """
    import datetime

    import data_loader as dl
    from features import build_stationary_features, STATIONARY_FEATURE_COLUMNS
    from features_indicators import indicator_features

    end = end or datetime.date.today().strftime("%Y-%m-%d")
    frames = []
    for sym in symbols:
        try:
            raw = dl.load_stock_data(sym, start, end)
            base = build_stationary_features(raw, keep_latest=False)
            ind = indicator_features(raw).reindex(base.index)
            f = base[[c for c in STATIONARY_FEATURE_COLUMNS if c in base.columns]].join(ind)
            close = raw["Close"].reindex(f.index)
            for h in horizons:
                f[f"y_{h}"] = close.shift(-h) / close - 1
            f["symbol"] = sym
            f["date"] = f.index
            frames.append(f.reset_index(drop=True))
            if verbose:
                print(f"  {sym:14} {len(f):5} rows", flush=True)
        except Exception as e:
            if verbose:
                print(f"  {sym:14} skipped ({type(e).__name__})", flush=True)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    return panel.replace([np.inf, -np.inf], np.nan)


def add_cross_section(panel: pd.DataFrame, feature_cols, min_names=8) -> pd.DataFrame:
    """
    Each feature's percentile rank WITHIN the universe on that date.

    This is the information a single-symbol model can never see: not "RSI is 62"
    but "RSI is higher than 80% of the market today".
    """
    out = panel.copy()
    counts = out.groupby("date")["symbol"].transform("size")
    usable = counts >= min_names
    for c in feature_cols:
        ranks = out.groupby("date")[c].rank(pct=True)
        out[c + CROSS_SECTION_SUFFIX] = ranks.where(usable)
    return out


def split_by_date(panel: pd.DataFrame, test_fraction=0.15, horizon=21):
    """Train on the earliest dates, test on the last `test_fraction`, with the
    horizon purged in between so no outcome straddles the boundary."""
    dates = np.sort(panel["date"].unique())
    cut = dates[int(len(dates) * (1 - test_fraction))]
    purge_start = pd.Timestamp(cut) - pd.Timedelta(days=int(horizon * 1.6))
    train = panel[panel["date"] < purge_start]
    test = panel[panel["date"] >= cut]
    return train, test


def fit_pooled(train, feature_cols, target, model="gbm", sample_cap=200_000):
    """One model for the whole universe."""
    d = train[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) > sample_cap:
        d = d.sample(sample_cap, random_state=0).sort_index()
    if len(d) < 2000:
        return None
    X, y = d[feature_cols].values, d[target].values
    if model == "ridge":
        return Ridge(alpha=1.0).fit(X, y)
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0).fit(X, y)


def shrink_to_validation(model, valid, feature_cols, target):
    """
    The same discipline the per-symbol engine uses: scale predictions by the
    coefficient that a validation slice says they deserve, and zero them out if
    they cannot beat predicting no change.
    """
    d = valid[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 500:
        return 0.0
    pred = model.predict(d[feature_cols].values)
    y = d[target].values
    denom = float(pred @ pred)
    if denom <= EPS:
        return 0.0
    alpha = float(np.clip((pred @ y) / denom, 0.0, 1.0))
    if np.sqrt(np.mean((y - alpha * pred) ** 2)) >= np.sqrt(np.mean(y ** 2)):
        return 0.0
    return alpha


# ------------------------------------------------------------------------
# Out-of-fold predictions and the deployable forecaster
# ------------------------------------------------------------------------
import os
import pickle
from datetime import date

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_cache")
OOF_FILE = os.path.join(CACHE_DIR, "pooled_oof.pkl")
MODEL_FILE = os.path.join(CACHE_DIR, "pooled_model.pkl")
POOLED_COLUMN = "Pooled_View"


def build_oof_predictions(panel: pd.DataFrame, feature_cols, horizons=(1, 5, 21, 63),
                          n_folds: int = 6, verbose: bool = True) -> pd.DataFrame:
    """
    (date, symbol) -> the universe model's view, made by a model that never saw
    that date. Walk-forward: each fold trains on everything before it.
    """
    dates = np.sort(panel["date"].unique())
    if len(dates) < 1000:
        return pd.DataFrame()
    # start predicting only once there is a decent training base
    starts = np.linspace(int(len(dates) * 0.40), len(dates), n_folds + 1).astype(int)
    out = []
    for h in horizons:
        target = f"y_{h}"
        if target not in panel.columns:
            continue
        for i in range(n_folds):
            lo, hi = starts[i], starts[i + 1]
            if hi <= lo:
                continue
            fold_start, fold_end = dates[lo], dates[min(hi, len(dates) - 1)]
            # purge: training outcomes must finish before the fold begins
            train = panel[panel["date"] < pd.Timestamp(fold_start) - pd.Timedelta(days=int(h * 1.6))]
            last_fold = (i == n_folds - 1)
            fold = panel[(panel["date"] >= fold_start) &
                         ((panel["date"] <= fold_end) if last_fold else (panel["date"] < fold_end))]
            if len(train) < 5000 or fold.empty:
                continue
            model = fit_pooled(train, feature_cols, target)
            if model is None:
                continue
            d = fold[feature_cols].replace([np.inf, -np.inf], np.nan)
            ok = d.notna().all(axis=1)
            if not ok.any():
                continue
            pred = model.predict(d[ok].values)
            out.append(pd.DataFrame({"date": fold.loc[ok, "date"].values,
                                     "symbol": fold.loc[ok, "symbol"].values,
                                     "h": h, "pooled": pred}))
            if verbose:
                print(f"  h={h:>3} fold {i + 1}/{n_folds}: trained on {len(train):,}, "
                      f"predicted {int(ok.sum()):,} rows to {pd.Timestamp(fold_end):%Y-%m}", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def train_and_cache(universe=None, horizons=(1, 5, 21, 63), start="2014-01-01", verbose=True):
    """Build the panel, cross-fit the out-of-fold views, fit the final live model,
    and cache all of it. Run once a day at most."""
    universe = universe or DEFAULT_UNIVERSE
    os.makedirs(CACHE_DIR, exist_ok=True)
    panel = build_panel(universe, start=start, horizons=horizons, verbose=verbose)
    if panel.empty:
        return None
    feats = [c for c in panel.columns if c not in ("symbol", "date") and not c.startswith("y_")]

    oof = build_oof_predictions(panel, feats, horizons=horizons, verbose=verbose)
    if not oof.empty:
        with open(OOF_FILE, "wb") as fh:
            pickle.dump({"built": date.today().isoformat(), "oof": oof, "features": feats}, fh)

    live = {}
    for h in horizons:
        target = f"y_{h}"
        if target in panel.columns:
            m = fit_pooled(panel, feats, target)
            if m is not None:
                live[h] = m
    with open(MODEL_FILE, "wb") as fh:
        pickle.dump({"built": date.today().isoformat(), "models": live, "features": feats,
                     "universe": universe}, fh)
    return {"panel_rows": len(panel), "oof_rows": len(oof), "horizons": sorted(live)}


def load_oof():
    try:
        with open(OOF_FILE, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def load_live_models():
    try:
        with open(MODEL_FILE, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def pooled_view_for(raw: pd.DataFrame, ticker: str, horizons=(1, 5, 21, 63)) -> pd.DataFrame:
    """
    The universe model's view for one symbol, aligned to its price index:
    out-of-fold history where available, and today's row from the live model.

    Returns an empty frame when the cache is missing, in which case the engine
    simply carries on without the group.
    """
    store, live = load_oof(), load_live_models()
    if store is None and live is None:
        return pd.DataFrame(index=raw.index)

    cols = {}
    if store is not None:
        oof = store["oof"]
        mine = oof[oof["symbol"] == ticker]
        for h in horizons:
            s = (mine[mine["h"] == h]
                 .drop_duplicates(subset="date", keep="last")   # later fold = more training data
                 .set_index("date")["pooled"])
            if len(s):
                cols[f"{POOLED_COLUMN}_{h}"] = s.reindex(raw.index)

    if live is not None and live.get("models"):
        feats = live["features"]
        try:
            from features import build_stationary_features, STATIONARY_FEATURE_COLUMNS
            from features_indicators import indicator_features
            base = build_stationary_features(raw, keep_latest=True)
            ind = indicator_features(raw).reindex(base.index)
            X = base[[c for c in STATIONARY_FEATURE_COLUMNS if c in base.columns]].join(ind)
            X = X.reindex(columns=feats)
            recent = X.tail(260).replace([np.inf, -np.inf], np.nan)
            ok = recent.notna().all(axis=1)
            for h, model in live["models"].items():
                key = f"{POOLED_COLUMN}_{h}"
                pred = pd.Series(index=raw.index, dtype=float)
                if ok.any():
                    pred.loc[recent.index[ok]] = model.predict(recent[ok].values)
                if key in cols:
                    cols[key] = cols[key].combine_first(pred)   # keep out-of-fold where it exists
                else:
                    cols[key] = pred
        except Exception:
            pass

    return pd.DataFrame(cols, index=raw.index) if cols else pd.DataFrame(index=raw.index)
