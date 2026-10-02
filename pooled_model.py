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
    """
    One model for one pool.

    Only the TARGET has to be present. Requiring every feature instead silently
    trained NOTHING for forex: those symbols have no exchange volume, and their
    high/low features are removed by the range guard, so a row with all columns
    present does not exist and dropna() emptied the frame. The forex pool came
    back with zero models and the failure looked like success. Gradient boosting
    handles missing values natively; Ridge cannot, so it gets a mean fill.
    """
    d = train[feature_cols + [target]].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[target])
    cols = [c for c in feature_cols if c in d.columns and d[c].notna().any()]
    if len(d) > sample_cap:
        d = d.sample(sample_cap, random_state=0).sort_index()
    if len(d) < 2000 or not cols:
        return None
    y = d[target].values
    if model == "ridge":
        m = Ridge(alpha=1.0).fit(d[cols].fillna(d[cols].mean()).fillna(0.0).values, y)
    else:
        m = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0).fit(d[cols].values, y)
    m._cols = cols
    return m


def predict_pooled(model, frame, feature_cols):
    """Score rows with the columns the model was actually fitted on."""
    use = getattr(model, "_cols", feature_cols)
    X = frame.reindex(columns=use).replace([np.inf, -np.inf], np.nan)
    if isinstance(model, Ridge):
        X = X.fillna(0.0)
    return model.predict(X.values)


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

# ------------------------------------------------------------------------
# WHICH POOL EACH ASSET CLASS LEARNS FROM
# ------------------------------------------------------------------------
# Measured, not assumed. exp_all_assets.py compared seven training sets over
# 129 instruments and 374,891 rows, split by date, scored per asset class
# against the naive "no change" forecast:
#
#   class        per-symbol   best pool            winner
#   stock_in        -22.19      -0.62              equity
#   stock_us         -9.32      +0.05              equity
#   etf            -233.20      -0.33              equity
#   crypto          -57.28     -11.40              equity
#   commodity        -7.91      -0.19              own class
#   forex            -8.31      +1.32              own class
#   index            -2.42      +2.63              macro
#
# Two things that table settles:
#   * Pooling beats one-symbol-at-a-time in EVERY class, usually by a lot. A
#     single symbol gives ~2,500 rows and the model learns noise from it; that
#     is why the per-symbol engine shrinks itself to nothing and the forecast
#     comes out flat.
#   * Pooling EVERYTHING together wins nothing. It scored -34.79 on Indian
#     stocks where a stocks-only pool scored -0.62. Copper does not behave like
#     a bank stock, and one model forced to explain both explains neither.
#
# Note these are still mostly negative: beating "no change" on return MAGNITUDE
# is close to impossible at these horizons. Direction is the usable part --
# 57.3% on ETFs, 56.5% on indices, 54.0% on US stocks.
POOL_MEMBERS = {
    "equity": ("stock_us", "stock_in"),
    "commodity": ("commodity",),
    "forex": ("forex",),
    "macro": ("commodity", "forex", "index"),
}
POOL_FOR_CLASS = {
    "stock_us": "equity", "stock_in": "equity", "etf": "equity", "crypto": "equity",
    "commodity": "commodity", "forex": "forex", "index": "macro",
}


def asset_class(ticker: str) -> str:
    """The asset class of a ticker, for choosing its pool."""
    try:
        import stock_search as ss
        cls = ss.ASSET_CLASSES.get(ticker)
        if cls:
            return cls
    except Exception:
        pass
    t = (ticker or "").upper()
    if t.endswith("=X"):
        return "forex"
    if t.endswith("=F"):
        return "commodity"
    if t.startswith("^"):
        return "index"
    if "-USD" in t or "-INR" in t:
        return "crypto"
    return "stock_in" if t.endswith((".NS", ".BO")) else "stock_us"


def pool_for(ticker: str) -> str:
    return POOL_FOR_CLASS.get(asset_class(ticker), "equity")


def full_universe():
    """Every instrument the dashboard can analyse, tagged with its class."""
    out = {}
    try:
        import stock_search as ss
        for row in ss.POPULAR_STOCKS:
            sym = row[0] if isinstance(row, (tuple, list)) else row
            out[sym] = "stock_in" if sym.endswith(".NS") else "stock_us"
        out.update(ss.ASSET_CLASSES)
    except Exception:
        for sym in DEFAULT_UNIVERSE:
            out[sym] = asset_class(sym)
    return out


def blank_untrustworthy_ranges(panel, verbose=True):
    """
    Remove the High/Low-derived features for feeds whose bar range cannot be
    trusted, before anything trains on them.

    Yahoo's daily FX bar has a range that runs past its close, so the next
    day's price is already inside it (75-85% of the time against ~45% on a
    well-formed bar). Trained on raw, a pooled model reads the future and
    reports 80%+ directional accuracy on currencies. See
    market_data.ohlc_range_quality and the note in fx_clean_bars.py.
    """
    import data_loader as dl
    import forecast_engine as fe
    import market_data as md

    cols = [c for c in panel.columns if c in fe.RANGE_DERIVED_FEATURES]
    if not cols:
        return panel
    end = str(pd.Timestamp(panel["date"].max()).date())
    start = str(pd.Timestamp(panel["date"].min()).date())
    bad, failed = [], []
    for sym in sorted(panel["symbol"].unique()):
        try:
            q = md.ohlc_range_quality(dl.load_stock_data(sym, start, end), sym)
        except Exception:
            failed.append(sym)
            continue
        if q["verdict"] != "ok":
            bad.append(sym)
    if bad:
        panel.loc[panel["symbol"].isin(bad), cols] = np.nan
    if verbose:
        print(f"  range check: {panel['symbol'].nunique() - len(failed)} checked, "
              f"{len(bad)} blanked" + (f", {len(failed)} unreadable" if failed else ""))
    return panel


def build_oof_predictions(panel: pd.DataFrame, feature_cols, horizons=(1, 5, 21, 63),
                          n_folds: int = 6, verbose: bool = True,
                          serve: pd.DataFrame = None) -> pd.DataFrame:
    """
    (date, symbol) -> the universe model's view, made by a model that never saw
    that date. Walk-forward: each fold trains on everything before it.

    `serve` is who gets SCORED, when that differs from who gets trained on.
    ETFs and crypto learn from the equity pool but are not in it -- an ETF is
    not a stock and would distort what the pool learns, yet the pool predicts
    it well. Without this they were scored only by the live model's 260-row
    tail, fell under the engine's coverage filter and were dropped: BTC-USD got
    no pooled feature at all. 35 of 129 instruments were in that position.
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
            scored = panel if serve is None else serve
            fold = scored[(scored["date"] >= fold_start) &
                          ((scored["date"] <= fold_end) if last_fold else (scored["date"] < fold_end))]
            if len(train) < 5000 or fold.empty:
                continue
            model = fit_pooled(train, feature_cols, target)
            if model is None:
                continue
            # a row is scorable if the model's own columns are not all missing
            use = getattr(model, "_cols", feature_cols)
            d = fold.reindex(columns=use).replace([np.inf, -np.inf], np.nan)
            ok = d.notna().any(axis=1)
            if not ok.any():
                continue
            pred = predict_pooled(model, fold[ok], feature_cols)
            out.append(pd.DataFrame({"date": fold.loc[ok, "date"].values,
                                     "symbol": fold.loc[ok, "symbol"].values,
                                     "h": h, "pooled": pred}))
            if verbose:
                print(f"  h={h:>3} fold {i + 1}/{n_folds}: trained on {len(train):,}, "
                      f"predicted {int(ok.sum()):,} rows to {pd.Timestamp(fold_end):%Y-%m}", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def train_and_cache(universe=None, horizons=(1, 5, 21, 63), start="2014-01-01",
                    verbose=True, panel=None):
    """
    Train the pooled models the dashboard actually uses, one per POOL.

    The universe defaults to every instrument the dashboard can analyse -- all
    asset classes, not just equities -- because the comparison showed each class
    learns best from a different slice of it (see POOL_FOR_CLASS). Pass `panel`
    to reuse an already-built one.
    """
    uni = full_universe() if universe is None else {s: asset_class(s) for s in universe}
    os.makedirs(CACHE_DIR, exist_ok=True)
    if panel is None:
        panel = build_panel(sorted(uni), start=start, horizons=horizons, verbose=verbose)
    if panel is None or panel.empty:
        return None
    panel = panel.copy()
    panel["cls"] = panel["symbol"].map(uni)
    panel = panel.dropna(subset=["cls"])
    panel = blank_untrustworthy_ranges(panel, verbose=verbose)

    feats = [c for c in panel.columns
             if c not in ("symbol", "date", "cls") and not c.startswith("y_")]

    oof_all, live = [], {}
    for pool, classes in POOL_MEMBERS.items():
        sub = panel[panel["cls"].isin(classes)]
        if sub.empty:
            continue
        if verbose:
            print(f"\n--- pool '{pool}' ({', '.join(classes)}): "
                  f"{len(sub):,} rows, {sub.symbol.nunique()} symbols ---", flush=True)
        served = [c for c, pl in POOL_FOR_CLASS.items() if pl == pool]
        serve = panel[panel["cls"].isin(served)]
        if verbose and set(served) - set(classes):
            print(f"    also scoring {', '.join(sorted(set(served) - set(classes)))} "
                  f"from this pool ({len(serve):,} rows)", flush=True)
        o = build_oof_predictions(sub, feats, horizons=horizons, verbose=verbose,
                                  serve=serve)
        if not o.empty:
            o["pool"] = pool
            oof_all.append(o)
        for h in horizons:
            if f"y_{h}" in sub.columns:
                m = fit_pooled(sub, feats, f"y_{h}")
                if m is not None:
                    live[(pool, h)] = m

    oof = pd.concat(oof_all, ignore_index=True) if oof_all else pd.DataFrame()
    if not oof.empty:
        with open(OOF_FILE, "wb") as fh:
            pickle.dump({"built": date.today().isoformat(), "oof": oof,
                         "features": feats, "pooled_by_class": True}, fh)
    with open(MODEL_FILE, "wb") as fh:
        pickle.dump({"built": date.today().isoformat(), "models": live, "features": feats,
                     "universe": sorted(uni), "pools": POOL_MEMBERS,
                     "pool_for_class": POOL_FOR_CLASS, "pooled_by_class": True}, fh)
    return {"panel_rows": len(panel), "oof_rows": len(oof),
            "pools": sorted({p for p, _ in live}), "models": len(live)}


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
        want = pool_for(ticker)
        try:
            from features import build_stationary_features, STATIONARY_FEATURE_COLUMNS
            from features_indicators import indicator_features
            base = build_stationary_features(raw, keep_latest=True)
            ind = indicator_features(raw).reindex(base.index)
            X = base[[c for c in STATIONARY_FEATURE_COLUMNS if c in base.columns]].join(ind)
            X = X.reindex(columns=feats)
            recent = X.tail(260).replace([np.inf, -np.inf], np.nan)
            ok = recent.notna().any(axis=1)
            for mk, model in live["models"].items():
                # keys are (pool, horizon) once the models are trained per pool;
                # a plain horizon means an old single-pool cache, still usable
                if isinstance(mk, tuple):
                    pool, h = mk
                    if pool != want:
                        continue
                else:
                    h = mk
                key = f"{POOLED_COLUMN}_{h}"
                pred = pd.Series(index=raw.index, dtype=float)
                if ok.any():
                    pred.loc[recent.index[ok]] = predict_pooled(model, recent[ok], feats)
                if key in cols:
                    cols[key] = cols[key].combine_first(pred)   # keep out-of-fold where it exists
                else:
                    cols[key] = pred
        except Exception:
            pass

    return pd.DataFrame(cols, index=raw.index) if cols else pd.DataFrame(index=raw.index)
