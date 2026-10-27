"""
forecast_engine.py
--------------------
The dashboard's price-forecasting engine (v12). It uses EVERYTHING that
might carry signal, and keeps only what measurably helps:

  Feature groups (each built with no look-ahead):
    base       -- 39 scale-free price/volume indicators (features.py)
    market     -- index returns, VIX level/changes, relative strength vs
                  market & sector, 60-day beta/correlation
    technical  -- the technical scorecard's votes, for every day
    candles    -- candlestick pattern bias and bullish/bearish counts
    smoothing  -- exponential smoothing with a FIXED alpha = 0.3 (model_smoothing.py):
                  price vs its smoothed level, and the level's 5/20-day slope
    calendar   -- month, turn-of-month, days to month end
    macro      -- US 10Y yield, dollar index, oil, gold (+ USD/INR and the
                  previous US session for Indian stocks)
    events     -- days to/since earnings, last earnings surprise, analyst
                  up/downgrades, days since dividend, trailing dividend yield
    news       -- GDELT tone/attention, Alpha Vantage, own daily headlines

  Group selection: starting from `base`, each group is added only if it
  lowers error in walk-forward cross-validation (3 consecutive time
  periods): by >= 0.3% on average AND in at least 2 of the 3 periods
  (greedy forward selection, per horizon). A group that doesn't help a
  stock consistently is left out -- this guards against "helped once by
  luck" groups, which in testing badly hurt some stocks (e.g. TCS).

  Safety gate: if the final model doesn't beat the naive "no change"
  forecast in at least 2 of the 3 periods, its signal strength is set to 0
  and the forecast falls back to naive.

  Direct multi-horizon models: instead of stacking 1-day guesses on top
  of each other (errors compound), separate shrunk ensembles (Ridge +
  Random Forest + XGBoost) predict the return 1, 5, 10, 21 and 63
  trading days ahead. The forecast for any date is interpolated between
  them. Each horizon also gets an 80% likely range from its validation
  errors, so the dashboard can show uncertainty honestly.

  Shrinkage: each horizon's prediction is scaled toward "no change" by
  alpha in [0, 1], fitted on the pooled out-of-fold predictions.
"""

import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from candlestick_patterns import detect_patterns, PATTERN_INFO, PATTERN_NAMES
from features import build_stationary_features, STATIONARY_FEATURE_COLUMNS
from models_tabular import ENSEMBLE_MODEL_BUILDERS
from technical_analysis import technical_score_series
from model_smoothing import smoothing_features, SMOOTHING_ALPHA

HORIZONS = [1, 5, 10, 21, 63]
# The indicator bank enters as six families, each gated on its own.
INDICATOR_GROUPS = {
    "ind_trend": "Trend_", "ind_momentum": "Mom_", "ind_volatility": "Vol_",
    "ind_volume": "Volm_", "ind_structure": "Struct_", "ind_season": "Seas_",
}

GROUP_ORDER = ["market", "technical", "candles", "smoothing", "calendar", "macro", "events", "news",
               "social", "universe", "arima"] + list(INDICATOR_GROUPS)

# These describe a DAY: a news tone, a macro series, an earnings date, the daily
# social reading, the universe model's daily view. Spreading them across 25
# intraday candles would repeat one number as if it were 25 observations, so on
# an intraday timeframe they are left out entirely.
DAILY_ONLY_GROUPS = {"market", "macro", "events", "news", "social", "universe", "calendar"}
GROUP_LABELS = {"base": "Price & volume", "market": "Market & VIX", "technical": "Technical rating",
                "candles": "Candlesticks", "smoothing": f"Exp. smoothing (α={SMOOTHING_ALPHA})", "calendar": "Calendar", "macro": "Macro (rates, $, oil, gold)",
                "events": "Earnings & dividends", "news": "News", "social": "Social buzz",
                "universe": "Universe model (38 symbols)",
                "arima": "ARIMA baseline",
                "ind_trend": "Trend indicators (ADX, Aroon, Ichimoku…)",
                "ind_momentum": "Momentum indicators (RSI, Stoch, CCI, TSI…)",
                "ind_volatility": "Volatility indicators (Bollinger, Keltner, Ulcer…)",
                "ind_volume": "Volume indicators (OBV, CMF, MFI, VWAP…)",
                "ind_structure": "Market structure (52w position, Hurst, entropy…)",
                "ind_season": "Seasonality (day, month, turn of month)"}
Z80 = 1.2816  # 80% two-sided normal interval


def _is_indian(ticker: str) -> bool:
    return ticker.upper().endswith((".NS", ".BO"))


# ------------------------------------------------------------------------
# External data (all optional -- anything that fails is simply left out)
# ------------------------------------------------------------------------
def _close_frame(symbols: dict, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(list(symbols.values()), start=start, end=end_excl, auto_adjust=False, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    close = close.rename(columns={v: k for k, v in symbols.items()})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # Phase 3: never trust the provider to respect the end date. Truncating here means a
    # historical replay cannot pick up a macro observation dated after its as-of moment.
    close = close.loc[close.index <= pd.Timestamp(end).normalize() + pd.Timedelta(hours=23, minutes=59)]
    return close


def load_extra_data(ticker: str, start: str, end: str) -> dict:
    """Download market, macro and event data for the engine. Never raises."""
    from data_loader import load_market_context, market_symbols_for
    out = {"market": None, "macro": None, "events": {}, "notes": []}
    try:
        out["market"] = load_market_context(start, end, **market_symbols_for(ticker))
    except Exception as e:
        out["notes"].append(f"market data unavailable ({e})")
    try:
        symbols = {"US10Y": "^TNX", "Dollar": "DX-Y.NYB", "Oil": "CL=F", "Gold": "GC=F"}
        if _is_indian(ticker):
            symbols.update({"USDINR": "USDINR=X", "US_Mkt": "^GSPC"})
        macro = _close_frame(symbols, start, end)
        out["macro"] = macro if not macro.empty else None
    except Exception as e:
        out["notes"].append(f"macro data unavailable ({e})")
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        ev = {}
        try:
            e = tk.get_earnings_dates(limit=80)
            if e is not None and not e.empty:
                e = e.copy()
                e.index = pd.to_datetime(e.index).tz_localize(None).normalize()
                ev["earnings"] = e
        except Exception:
            pass
        try:
            u = tk.upgrades_downgrades
            if u is not None and len(u):
                u = u.copy()
                u.index = pd.to_datetime(u.index).tz_localize(None).normalize()
                ev["upgrades"] = u
        except Exception:
            pass
        try:
            d = tk.dividends
            if d is not None and len(d):
                d = d.copy()
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                ev["dividends"] = d
        except Exception:
            pass
        out["events"] = ev
    except Exception as e:
        out["notes"].append(f"event data unavailable ({e})")
    return out


# ------------------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------------------
def _market_features(raw, market_df):
    if market_df is None or market_df.empty:
        return pd.DataFrame(index=raw.index)
    m = market_df.reindex(raw.index).ffill(limit=5)
    r = raw["Close"].pct_change()
    idx_r = m["SP500_Close"].pct_change()
    f = pd.DataFrame(index=raw.index)
    for w in (1, 5, 20):
        f[f"Mkt_Ret_{w}d"] = m["SP500_Close"].pct_change(w)
    for w in (5, 20):
        f[f"Sector_Ret_{w}d"] = m["Sector_Close"].pct_change(w)
        f[f"RelStr_Mkt_{w}d"] = raw["Close"].pct_change(w) - f[f"Mkt_Ret_{w}d"]
        f[f"RelStr_Sector_{w}d"] = raw["Close"].pct_change(w) - f[f"Sector_Ret_{w}d"]
    vix = m["VIX_Close"]
    f["VIX_Z_1y"] = (vix - vix.rolling(252, min_periods=60).mean()) / vix.rolling(252, min_periods=60).std()
    f["VIX_Chg_5d"] = vix.pct_change(5)
    cov = r.rolling(60).cov(idx_r)
    f["Beta_60d"] = cov / idx_r.rolling(60).var()
    f["Corr_60d"] = r.rolling(60).corr(idx_r)
    return f


def _technical_features(raw):
    return technical_score_series(raw)


def _candle_features(raw):
    flags = detect_patterns(raw)
    bias = pd.Series(0.0, index=raw.index)
    bull = pd.Series(0.0, index=raw.index)
    bear = pd.Series(0.0, index=raw.index)
    for name in PATTERN_NAMES:
        b, strength, _ = PATTERN_INFO[name]
        fl = flags[name].astype(float)
        bias += fl * b * strength
        if b > 0:
            bull += fl * strength
        elif b < 0:
            bear += fl * strength
    return pd.DataFrame({"Candle_Bias_3d": bias.rolling(3, min_periods=1).sum(),
                         "Candle_Bull_10d": bull.rolling(10, min_periods=1).sum(),
                         "Candle_Bear_10d": bear.rolling(10, min_periods=1).sum()}, index=raw.index)


def _calendar_features(index):
    idx = pd.DatetimeIndex(index)
    f = pd.DataFrame(index=idx)
    f["Cal_Month_Sin"] = np.sin(2 * np.pi * idx.month / 12)
    f["Cal_Month_Cos"] = np.cos(2 * np.pi * idx.month / 12)
    f["Cal_Day_Of_Month"] = idx.day / 31
    month_end = idx + pd.offsets.BMonthEnd(0)
    f["Cal_BDays_To_Month_End"] = [len(pd.bdate_range(d, e)) - 1 for d, e in zip(idx, month_end)]
    month_start_rank = pd.Series(1, index=idx).groupby([idx.year, idx.month]).cumsum().values
    f["Cal_Turn_Of_Month"] = ((f["Cal_BDays_To_Month_End"] <= 1) | (month_start_rank <= 3)).astype(float)
    f["Cal_BDays_To_Month_End"] = f["Cal_BDays_To_Month_End"] / 22
    return f


def _macro_features(raw, macro_df, ticker):
    if macro_df is None or macro_df.empty:
        return pd.DataFrame(index=raw.index)
    m = macro_df.reindex(raw.index.union(macro_df.index)).ffill(limit=5).reindex(raw.index)
    if _is_indian(ticker):
        # US/FX/commodity daily closes happen AFTER the Indian close, so only
        # the previous day's values are known when an NSE/BSE day ends.
        m = m.shift(1)
    f = pd.DataFrame(index=raw.index)
    for col in m.columns:
        s = m[col]
        if col == "US10Y":  # a yield: use changes in percentage points
            f["Macro_US10Y_Chg_5d"] = s.diff(5) / 10
            f["Macro_US10Y_Chg_20d"] = s.diff(20) / 10
        else:
            f[f"Macro_{col}_Ret_5d"] = s.pct_change(5)
            f[f"Macro_{col}_Ret_20d"] = s.pct_change(20)
    if "US_Mkt" in m.columns:
        f["Macro_US_Mkt_Ret_1d"] = m["US_Mkt"].pct_change()
    return f


def _event_features(raw, events):
    idx = raw.index
    f = pd.DataFrame(index=idx)
    e = events.get("earnings")
    if e is not None and len(e):
        # Treat an earnings release as known from the NEXT trading day (many
        # companies report after the close), so there's no look-ahead.
        dates = np.array(sorted(set(e.index)), dtype="datetime64[ns]")
        pos_prev = np.searchsorted(dates, idx.values, side="left") - 1   # last release strictly before today
        pos_next = np.searchsorted(dates, idx.values, side="left")        # next release today or later
        days_since = np.where(pos_prev >= 0, (idx.values - dates[np.clip(pos_prev, 0, None)]).astype("timedelta64[D]").astype(float), np.nan)
        days_to = np.where(pos_next < len(dates), (dates[np.clip(pos_next, None, len(dates) - 1)] - idx.values).astype("timedelta64[D]").astype(float), np.nan)
        f["Earn_Days_Since"] = np.clip(days_since, 0, 90) / 90
        f["Earn_Days_To"] = np.where(np.isnan(days_to) | (days_to > 30), 1.0, days_to / 30)  # dates are announced ~2-4 weeks ahead
        f["Earn_Recent"] = ((days_since >= 0) & (days_since <= 3)).astype(float)
        if "Surprise(%)" in e.columns:
            surprise = e["Surprise(%)"].groupby(level=0).last().sort_index()
            s = surprise.reindex(surprise.index.union(idx)).shift(1).ffill().reindex(idx)  # known from the next day
            f["Earn_Last_Surprise"] = (s.clip(-50, 50) / 100)
    u = events.get("upgrades")
    if u is not None and len(u) and "Action" in u.columns:
        act = u["Action"].astype(str).str.lower()
        net = (act.eq("up").astype(int) - act.eq("down").astype(int)).groupby(level=0).sum()
        daily = net.reindex(pd.date_range(min(idx.min(), net.index.min()), idx.max()), fill_value=0)
        f["Analyst_Net_30d"] = daily.shift(1).rolling(30, min_periods=1).sum().reindex(idx)
    d = events.get("dividends")
    if d is not None and len(d):
        ex = d.index.values.astype("datetime64[ns]")
        pos = np.searchsorted(ex, idx.values, side="right") - 1
        since = np.where(pos >= 0, (idx.values - ex[np.clip(pos, 0, None)]).astype("timedelta64[D]").astype(float), 365)
        f["Div_Days_Since"] = np.clip(since, 0, 365) / 365
        daily_div = d.reindex(pd.date_range(min(idx.min(), d.index.min()), idx.max()), fill_value=0.0)
        f["Div_Yield_TTM"] = (daily_div.rolling(365, min_periods=1).sum().reindex(idx) / raw["Close"]).clip(0, 0.3)
    return f


# Every feature that reads High or Low rather than just Close. Found by
# perturbing the high/low of a real frame and recording which columns moved --
# 34 of 150 -- rather than by guessing from the names, which misses things like
# Tech_Score and Ichimoku that aggregate a range term several layers down.
#
# These are dropped for any symbol whose bars fail market_data.ohlc_range_quality.
# On Yahoo's FX feed the bar's range extends past its close, so these features
# can see the next day's price; see the note in market_data.py. The remaining
# 116 close-only features are unaffected and still train.
RANGE_DERIVED_FEATURES = frozenset({
    "Candle_Bear_10d", "Candle_Bias_3d", "Candle_Bull_10d", "High", "Low",
    "Mom_CCI20", "Mom_Stoch_D", "Mom_Stoch_K", "Mom_Williams_R",
    "S_ATR_Pct", "S_Body_Pct", "S_Dist_52w_High", "S_Dist_52w_Low", "S_HL_Range",
    "S_Lower_Wick_Pct", "S_Stoch_K", "S_Upper_Wick_Pct",
    "Struct_Close_Location", "Tech_Score", "Tech_Score_Chg_5d",
    "Trend_ADX14", "Trend_ATR_Channel_Pos", "Trend_Aroon", "Trend_DI_Diff",
    "Trend_Ichimoku_ConvBase", "Trend_Ichimoku_PxCloud",
    "Vol_ATR_Pct", "Vol_Donchian_Pos", "Vol_GarmanKlass", "Vol_Keltner_Pos",
    "Vol_Parkinson", "Volm_CMF20", "Volm_MFI14", "Volm_VWAP_Dist",
})


def build_feature_table(raw, ticker, extra=None, news_df=None, social_df=None, pooled_df=None,
                       horizons=None, intraday=False, on_progress=None):
    """All candidate features + forward-return targets for every horizon.
    Returns (table, groups) where groups maps group name -> list of columns."""
    extra = extra or {}
    base = build_stationary_features(raw, keep_latest=True)
    idx = base.index
    try:
        from features_indicators import indicator_features
        _bank = indicator_features(raw)
    except Exception:
        _bank = None

    parts = {
        "market": _market_features(raw, extra.get("market")),
        "technical": _technical_features(raw),
        "candles": _candle_features(raw),
        "smoothing": smoothing_features(raw["Close"]),
        # ARIMA's own forecast, gated like any other group: it is kept only if
        # it cuts validation error by the usual margin. A classic baseline does
        # not get a free pass.
        "arima": _arima_features(raw, horizons or HORIZONS, on_progress=on_progress),
        "calendar": _calendar_features(raw.index),
        "macro": _macro_features(raw, extra.get("macro"), ticker),
        "events": _event_features(raw, extra.get("events", {})),
    }
    if _bank is not None and not _bank.empty:
        for gname, prefix in INDICATOR_GROUPS.items():
            cols_ = [c for c in _bank.columns if c.startswith(prefix)]
            if cols_:
                parts[gname] = _bank[cols_]
    if news_df is not None and news_df.shape[1] > 0:
        parts["news"] = news_df.reindex(raw.index)
    # Social platforms publish no free archive, so this history only exists
    # from the day the dashboard first collected it; the coverage filter below
    # drops the group until there is enough of it to learn anything from.
    if social_df is not None and social_df.shape[1] > 0:
        parts["social"] = social_df.reindex(raw.index)
    # What a model trained across the whole universe expects from a stock that
    # looks like this one today. Out-of-fold, so it never saw the row it scores.
    if pooled_df is not None and pooled_df.shape[1] > 0:
        parts["universe"] = pooled_df.reindex(raw.index)
    groups = {"base": list(STATIONARY_FEATURE_COLUMNS)}
    table = base.copy()
    for name, frame in parts.items():
        frame = frame.replace([np.inf, -np.inf], np.nan)
        cols = [c for c in frame.columns if frame[c].reindex(idx).notna().mean() > 0.5]
        if not cols:
            continue
        # Missing early values (warm-up windows, data gaps) -> neutral 0.
        table = table.join(frame[cols].reindex(idx).fillna(0.0))
        groups[name] = cols
    if intraday:      # drop the groups that only mean something once a day
        for g in list(groups):
            if g in DAILY_ONLY_GROUPS:
                table = table.drop(columns=[c for c in groups[g] if c in table.columns])
                groups.pop(g)
    # Feeds whose bar range cannot be trusted lose every range-derived feature.
    try:
        import market_data as _md
        q = _md.ohlc_range_quality(raw, ticker)
    except Exception:
        q = {"verdict": "ok"}
    if q.get("verdict") != "ok":
        drop = [c for c in table.columns if c in RANGE_DERIVED_FEATURES]
        if drop:
            table = table.drop(columns=drop)
            for g in list(groups):
                groups[g] = [c for c in groups[g] if c not in RANGE_DERIVED_FEATURES]
                if not groups[g]:
                    groups.pop(g)
        table.attrs["range_quality"] = q
        table.attrs["range_features_dropped"] = sorted(drop)

    close = raw["Close"]
    for h in (horizons or HORIZONS):
        table[f"Fwd_Ret_{h}"] = (close.shift(-h) / close - 1).reindex(idx)
    table["Fwd_Open_1"] = (raw["Open"].shift(-1) / close - 1).reindex(idx)
    return table, groups


# ------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------
ARIMA_ALWAYS_ON = True   # ARIMA bypasses the feature-group gate -- see select_groups
ARIMA_PREFIX = "Arima_Fwd_"
_ARIMA_WARMUP = 250     # rows before the first forecast
_ARIMA_REFIT = 63       # refit the order's parameters once a quarter


def _arima_features(raw, horizons, on_progress=None):
    """
    ARIMA's own forward-return forecast, as a feature the ensemble can vote on.

    ARIMA is univariate and forecasts h steps from the price series, so it
    cannot implement fit(X, y)/predict(X) like the tabular members. It enters
    here instead: one column per horizon, each value computed from prices
    STRICTLY BEFORE that date, so it carries no more information than the
    model would have had on the day.

    Parameters are refit quarterly rather than daily. A daily refit is ~40x
    the cost for a model whose coefficients barely move, and this already
    adds the better part of a minute to a first load.
    """
    from model_arima import select_best_order
    from statsmodels.tsa.arima.model import ARIMA

    close = raw["Close"].astype(float)
    n = len(close)
    cols = {h: np.full(n, np.nan) for h in horizons}
    if n < _ARIMA_WARMUP + max(horizons) + 10:
        return pd.DataFrame(cols, index=raw.index).rename(
            columns=lambda h: f"{ARIMA_PREFIX}{h}")

    # one order for the whole series, chosen on the warm-up window only
    try:
        order = select_best_order(close.iloc[:_ARIMA_WARMUP])
    except Exception:
        order = (2, 1, 0)

    hmax, fitted, last_fit, processed = max(horizons), None, -10 ** 9, 0
    for i in range(_ARIMA_WARMUP, n):
        hist = close.iloc[:i]                       # strictly before row i
        if on_progress and (i - _ARIMA_WARMUP) % _ARIMA_REFIT == 0:
            on_progress(f"ARIMA history: {i:,}/{n:,} sessions")
        try:
            if fitted is None or (i - last_fit) >= _ARIMA_REFIT:
                fitted = ARIMA(hist.values, order=order).fit()
                last_fit = i
            else:
                # Carry the Kalman state forward using only new observations.
                # apply(full_history) redundantly filters every old row again.
                fitted = fitted.extend(close.iloc[processed:i].values)
            processed = i
            path = np.asarray(fitted.forecast(hmax), dtype=float)
        except Exception:
            continue
        p0 = hist.iloc[-1]
        if not np.isfinite(p0) or p0 <= 0:
            continue
        for h in horizons:
            if h <= len(path) and np.isfinite(path[h - 1]):
                cols[h][i] = path[h - 1] / p0 - 1.0   # forecast as a RETURN

    out = pd.DataFrame(cols, index=raw.index)
    out.columns = [f"{ARIMA_PREFIX}{h}" for h in out.columns]
    return out


class ArimaMember:
    """
    Ensemble member whose only input is ARIMA's forecast for this horizon.

    It fits a 1-D least-squares calibration of that forecast onto the target,
    which undoes the feature scaling and lets ARIMA be over- or under-confident
    without distorting the blend. It then competes for weight on out-of-fold
    RMSE exactly like Ridge, Random Forest and XGBoost.
    """

    def __init__(self, col_idx):
        self.j, self.a, self.b = col_idx, 0.0, 0.0

    def fit(self, X, y):
        x = np.asarray(X)[:, self.j]
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 30 and np.std(x[ok]) > 1e-12:
            self.b, self.a = np.polyfit(x[ok], y[ok], 1)
        else:                                   # nothing usable -> predict the mean
            self.a, self.b = (float(np.mean(y[ok])) if ok.any() else 0.0), 0.0
        return self

    def predict(self, X):
        x = np.nan_to_num(np.asarray(X)[:, self.j], nan=0.0, posinf=0.0, neginf=0.0)
        return self.a + self.b * x


def _fast_members():
    return [Ridge(alpha=10.0),
            XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=20, reg_lambda=5.0, n_jobs=1, random_state=42)]


def _folds(frame, n_folds=3, min_train_frac=0.55, gap=0):
    """Expanding-window walk-forward folds: train on everything before a cut,
    validate on the next block. Several time periods instead of one, so a
    group/signal has to help consistently, not just in one lucky window."""
    n = len(frame)
    start = int(n * min_train_frac)
    size = (n - start) // n_folds
    if gap < 0 or start <= gap or size < 1:
        raise ValueError("Not enough observations for purged walk-forward validation.")
    return [(frame.iloc[:start + k * size - gap], frame.iloc[start + k * size: start + (k + 1) * size if k < n_folds - 1 else n])
            for k in range(n_folds)]


def _fit_predict(train, val, cols, target, members_factory):
    scaler = StandardScaler().fit(train[cols].values)
    Xtr, Xva = scaler.transform(train[cols].values), scaler.transform(val[cols].values)
    members = members_factory()
    if isinstance(members, dict):
        names, models = list(members.keys()), list(members.values())
    else:
        names, models = [type(m).__name__ for m in members], members
    preds = []
    for m in models:
        m.fit(Xtr, train[target].values)
        preds.append(m.predict(Xva))
    return names, preds


def forecast_vol_series(raw, h, vol_models=None):
    """The volatility series the bands are calibrated against: the HAR forecast
    where it earned its place, the trailing window otherwise."""
    if vol_models and h in vol_models:
        s_ = vol_models[h].predict_series(raw) * np.sqrt(h / 252.0)
        if len(s_.dropna()) > 200:
            return s_.reindex(raw.index).ffill()
    return trailing_vol(raw, h)


def trailing_vol(raw, h, window=21):
    """Volatility of an h-day move as known on each day (no look-ahead)."""
    log_ret = np.log(raw["Close"].clip(lower=1e-12)).diff()
    v = log_ret.rolling(window).std() * np.sqrt(h)
    return v.replace(0, np.nan).ffill()


def _cv_score(frame, cols, target, members_factory, n_folds=3, vol=None):
    """
    Walk-forward CV. Member weights (inverse RMSE) and the shrinkage alpha
    are fitted on all out-of-fold predictions pooled together. Safety gate:
    alpha is forced to 0 unless the shrunk model beats the naive forecast in
    at least 2 of the folds -- so an unstable 'signal' falls back to naive.
    """
    # A training label at t contains the close at t+h. Remove overlapping
    # labels so every training outcome is known before validation starts.
    horizon = int(target.rsplit("_", 1)[-1])
    folds = _folds(frame, n_folds, gap=horizon)
    fold_preds, fold_y, names = [], [], None
    for tr, va in folds:
        names, preds = _fit_predict(tr, va, cols, target, members_factory)
        fold_preds.append(np.vstack(preds))
        fold_y.append(va[target].values)
    P, y = np.hstack(fold_preds), np.concatenate(fold_y)
    rmses = np.sqrt(((P - y) ** 2).mean(axis=1))
    w = (1 / np.maximum(rmses, 1e-12)); w = w / w.sum()
    blended = w @ P
    alpha = float(np.clip((blended @ y) / (blended @ blended + 1e-12), 0.0, 1.0))
    fold_rmse, fold_naive = [], []
    for fp, fy in zip(fold_preds, fold_y):
        fold_rmse.append(np.sqrt(np.mean((fy - alpha * (w @ fp)) ** 2)))
        fold_naive.append(np.sqrt(np.mean(fy ** 2)))
    wins = sum(r < nv for r, nv in zip(fold_rmse, fold_naive))
    if wins < 2:
        alpha = 0.0
        fold_rmse = fold_naive

    # ---- alternative combination: fit the weights instead of averaging ----
    # Worth trying whenever the members disagree, which is exactly when a plain
    # average throws information away.
    stack_coef, stack_rmse = None, None
    try:
        from sklearn.linear_model import LinearRegression
        sk = LinearRegression(positive=True, fit_intercept=False).fit(P.T, y)
        coef = np.asarray(sk.coef_, dtype=float)
        if np.isfinite(coef).all() and coef.sum() > 1e-9:
            cand = [np.sqrt(np.mean((fy - (coef @ fp)) ** 2)) for fp, fy in zip(fold_preds, fold_y)]
            beats_blend = sum(c < r for c, r in zip(cand, fold_rmse))
            beats_naive = sum(c < nv for c, nv in zip(cand, fold_naive))
            gain = 1 - (np.mean(cand) / max(np.mean(fold_rmse), 1e-12))
            # must beat the blend in most folds, beat naive in most folds, and
            # by a real margin -- the same bar a feature group has to clear
            if beats_blend >= 3 and beats_naive >= 2 and gain >= STRICT_GAIN:
                stack_coef, stack_rmse = coef, cand
                fold_rmse = cand
    except Exception:
        stack_coef = None
    resid = (y - (stack_coef @ P)) if stack_coef is not None else (y - alpha * blended)
    out = {"rmse": float(np.mean(fold_rmse)), "fold_rmse": fold_rmse, "alpha": alpha,
           "weights": dict(zip(names, w)), "sigma": float(np.std(resid)),
           "naive_rmse": float(np.mean(fold_naive)), "wins": wins, "sigma_ratio": None,
           "stack_coef": (dict(zip(names, stack_coef)) if stack_coef is not None else None),
           "combination": "stacked" if stack_coef is not None else "blend"}
    if vol is not None:
        # how big the model's misses were RELATIVE to the volatility of the day:
        # a single number that travels across regimes, unlike a fixed sigma
        v = np.concatenate([vol.reindex(va.index).values for _, va in folds])
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() > 50:
            scaled = resid[ok] / v[ok]
            out["sigma_ratio"] = float(np.std(scaled))
            # empirical 10th/90th percentiles of those scaled misses: keeps fat
            # tails and any asymmetry that a Gaussian band would flatten away
            out["q_lo_ratio"] = float(np.quantile(scaled, 0.10))
            out["q_hi_ratio"] = float(np.quantile(scaled, 0.90))
    return out


STRICT_GAIN = 0.010   # wide, high-variance groups must clear this instead
MIN_GAIN = 0.003  # a group must cut CV error by >= 0.3% ...


def select_groups(frame, groups, target, on_progress=None):
    """Greedy forward selection of feature groups on walk-forward CV error.
    A group is kept only if it lowers mean CV error by >= 0.3% AND lowers it
    in at least 2 of the 3 time folds."""
    chosen = ["base"]
    cols = list(groups["base"])
    # ARIMA is admitted unconditionally, by explicit choice, so the classic
    # baseline always has a vote in the ensemble and is always visible in the
    # Model Insights tab. It is the ONE group exempt from the gate below.
    # Measured cost of that exemption on MSFT: skill +0.060% -> +0.062%, and
    # ~19s added to a first load. Set ARIMA_ALWAYS_ON = False to gate it like
    # everything else.
    if ARIMA_ALWAYS_ON and "arima" in groups and len(groups["arima"]):
        chosen.append("arima")
        cols += list(groups["arima"])
    if on_progress:
        on_progress(f"{target}: validating base features")
    best = _cv_score(frame, cols, target, _fast_members)
    trail = [("base", best["rmse"])]
    for g in GROUP_ORDER:
        if g not in groups or g in chosen:
            continue
        if on_progress:
            on_progress(f"{target}: checking {g}")
        trial = _cv_score(frame, cols + groups[g], target, _fast_members)
        fold_better = sum(t < b for t, b in zip(trial["fold_rmse"], best["fold_rmse"]))
        # The indicator families are wide (10-20 columns each) and were measured
        # to help some symbols and badly hurt others, so they must clear a
        # higher bar: every fold must improve, and by 1% rather than 0.3%.
        strict = g.startswith("ind_")
        need_folds = 3 if strict else 2
        need_gain = STRICT_GAIN if strict else MIN_GAIN
        if trial["rmse"] < best["rmse"] * (1 - need_gain) and fold_better >= need_folds:
            chosen.append(g)
            cols += groups[g]
            best = trial
        trail.append((g, trial["rmse"]))
    return chosen, cols, trail


def _full_members(cols=None):
    m = {name: builder() for name, builder in ENSEMBLE_MODEL_BUILDERS.items()}
    if cols:
        for j, c in enumerate(cols):
            if str(c).startswith(ARIMA_PREFIX):
                m["ARIMA"] = ArimaMember(j)
                break
    return m


class HorizonModel:
    """Shrunk Ridge + Random Forest + XGBoost + ARIMA ensemble for one horizon."""

    def __init__(self, cols, weights, alpha, sigma, groups_used, trail, val_rmse, val_naive_rmse,
                 sigma_ratio=None, q_lo_ratio=None, q_hi_ratio=None, stack_coef=None,
                 combination="blend"):
        self.cols, self.weights, self.alpha, self.sigma = cols, weights, alpha, sigma
        self.groups_used, self.trail = groups_used, trail
        self.val_rmse, self.val_naive_rmse = val_rmse, val_naive_rmse
        # sigma_ratio: typical miss expressed in units of that day's volatility,
        # so the band can be rebuilt from TODAY's volatility instead of an average
        self.sigma_ratio = sigma_ratio
        # empirical quantiles of the scaled residuals -> asymmetric, fat-tailed band
        self.q_lo_ratio, self.q_hi_ratio = q_lo_ratio, q_hi_ratio
        # how this horizon combines its members: the shrunk average, or fitted
        # non-negative weights that won on the same folds
        self.stack_coef, self.combination = stack_coef, combination
        self.scaler, self.members = None, {}

    def band_offsets(self, current_vol=None):
        """(lower, upper) offsets around the forecast for an 80% band."""
        if (self.q_lo_ratio is not None and self.q_hi_ratio is not None
                and current_vol and np.isfinite(current_vol) and current_vol > 0):
            lo, hi = self.q_lo_ratio * current_vol, self.q_hi_ratio * current_vol
            cap = 3.5 * self.sigma
            return float(np.clip(lo, -cap, -0.15 * self.sigma)), float(np.clip(hi, 0.15 * self.sigma, cap))
        sd = self.band_sigma(current_vol)
        return -Z80 * sd, Z80 * sd

    def band_sigma(self, current_vol=None):
        """Uncertainty for a forecast made today."""
        if self.sigma_ratio and current_vol and np.isfinite(current_vol) and current_vol > 0:
            # keep it sane if today is a once-in-a-decade reading
            return float(np.clip(self.sigma_ratio * current_vol, 0.3 * self.sigma, 3.0 * self.sigma))
        return self.sigma

    def fit_final(self, labeled, target):
        self.scaler = StandardScaler().fit(labeled[self.cols].values)
        X = self.scaler.transform(labeled[self.cols].values)
        self.members = {n: m.fit(X, labeled[target].values)
                        for n, m in _full_members(self.cols).items()}
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(rows[self.cols].values)
        if self.stack_coef:          # fitted weights won on this symbol's own folds
            return sum(self.stack_coef.get(n, 0.0) * m.predict(X) for n, m in self.members.items())
        return self.alpha * sum(self.weights[n] * m.predict(X) for n, m in self.members.items())


def _train_horizon(labeled, groups, target, vol=None, on_progress=None):
    chosen, cols, trail = select_groups(labeled, groups, target, on_progress=on_progress)
    if on_progress:
        on_progress(f"{target}: validating final ensemble")
    s = _cv_score(labeled, cols, target, lambda: _full_members(cols), vol=vol)
    return HorizonModel(cols, s["weights"], s["alpha"], s["sigma"], chosen, trail, s["rmse"],
                        s["naive_rmse"], sigma_ratio=s.get("sigma_ratio"),
                        q_lo_ratio=s.get("q_lo_ratio"), q_hi_ratio=s.get("q_hi_ratio"),
                        stack_coef=s.get("stack_coef"), combination=s.get("combination", "blend"))


class ForecastEngine:
    """
    Usage:
        engine = ForecastEngine(raw, ticker, extra=load_extra_data(...), news_df=...).fit()
        engine.forecast_path(30)          # DataFrame of daily forecasts + 80% range
        engine.predict_date(some_date)    # dict for the dashboard
    """

    def __init__(self, raw, ticker, extra=None, news_df=None, horizons=HORIZONS, social_df=None,
                 pooled_df=None, timeframe="1d", bars_per_year=252.0, on_progress=None):
        self.on_progress = on_progress
        self.raw, self.ticker, self.horizons = raw, ticker, list(horizons)
        self.feature_inputs = dict(extra=extra, news_df=news_df, social_df=social_df, pooled_df=pooled_df)
        self.timeframe = timeframe
        self.bars_per_year = bars_per_year          # for anything that annualises
        self.intraday = timeframe not in ("1d", "1wk")
        # HAR-RV volatility forecasts, one per horizon; each refuses to fit
        # unless it beats "the next h days look like the last 21"
        try:
            if self.intraday:
                self.vol_models = {}      # HAR-RV is fitted on daily realised volatility
            else:
                from volatility_model import fit_horizon_vols
                self.vol_models = fit_horizon_vols(raw, self.horizons)
        except Exception:
            self.vol_models = {}
        # Crypto (and anything else quoted 7 days a week) has weekend rows, so
        # its forecast steps are calendar days rather than business days.
        self.trades_weekends = bool((raw.index.dayofweek >= 5).mean() > 0.10)
        self.table, self.groups = build_feature_table(raw, ticker, extra, news_df, social_df, pooled_df,
                                                      horizons=self.horizons, intraday=self.intraday,
                                                      on_progress=on_progress)
        self.models, self.open_model = {}, None

    def fit(self):
        for h in self.horizons:
            target = f"Fwd_Ret_{h}"
            labeled = self.table[self.table[target].notna()]
            if len(labeled) < 300:
                continue
            groups_h = dict(self.groups)
            if "universe" in groups_h:      # keep only this horizon's pooled view
                same_h = [c for c in groups_h["universe"] if c.endswith(f"_{h}")]
                if same_h:
                    groups_h["universe"] = same_h
                else:
                    groups_h.pop("universe")
            # bands: calibrated against trailing volatility, which measured better
            # than the HAR forecast for coverage (3.64 vs 5.18 pts off target)
            self.models[h] = _train_horizon(labeled, groups_h, target,
                                            vol=trailing_vol(self.raw, h),
                                            on_progress=self.on_progress).fit_final(labeled, target)
        # Next-day OPEN (overnight gap), using the 1-day model's chosen features
        if 1 in self.models:
            if self.on_progress:
                self.on_progress("Validating next-session opening price")
            lab = self.table[self.table["Fwd_Open_1"].notna()]
            m1 = self.models[1]
            # must use the same member set fit_final will build from these cols,
            # or weights and members disagree and predict() raises KeyError
            s = _cv_score(lab, m1.cols, "Fwd_Open_1", lambda: _full_members(m1.cols))
            self.open_model = HorizonModel(m1.cols, s["weights"], s["alpha"], s["sigma"], m1.groups_used, [],
                                           s["rmse"], s["naive_rmse"], stack_coef=s.get("stack_coef"),
                                           combination=s.get("combination", "blend")).fit_final(lab, "Fwd_Open_1")
        return self

    # ---- forecasting ----
    def _future_index(self, origin, n):
        """
        Timestamps for the next n candles. Daily and weekly step through the
        calendar; intraday walks forward inside trading sessions, so a 15-minute
        forecast lands on 15-minute marks during market hours rather than at
        03:00 in the morning.
        """
        tf = getattr(self, "timeframe", "1d")
        if tf == "1wk":
            return pd.date_range(origin + pd.Timedelta(weeks=1), periods=n, freq="W-MON")
        if tf == "1d" or not getattr(self, "intraday", False):
            step = pd.date_range if getattr(self, "trades_weekends", False) else pd.bdate_range
            return step(origin + pd.Timedelta(days=1), periods=n)

        # intraday: reuse the session shape the data itself shows
        idx = self.raw.index
        times = pd.Series(idx.time).value_counts().index.sort_values()
        per_session = max(len(times), 1)
        delta = (idx[-1] - idx[-2]) if len(idx) > 1 else pd.Timedelta(minutes=5)
        out, cursor = [], origin
        while len(out) < n:
            cursor = cursor + delta
            t = cursor.time()
            if t > max(times):                       # past the close -> next session
                nxt = cursor.normalize() + pd.Timedelta(days=1)
                while nxt.weekday() >= 5:
                    nxt += pd.Timedelta(days=1)
                cursor = pd.Timestamp.combine(nxt.date(), min(times))
            out.append(cursor)
        return pd.DatetimeIndex(out[:n])

    def _curve(self, row):
        hs = sorted(self.models)
        rets = {h: float(self.models[h].predict(row)[0]) for h in hs}
        day = row.index[-1]
        sig, lo_off, hi_off = {}, {}, {}
        for h in hs:
            v = trailing_vol(self.raw, h).reindex([day]).iloc[0]
            vv = float(v) if pd.notna(v) else None
            sig[h] = self.models[h].band_sigma(vv)
            lo_off[h], hi_off[h] = self.models[h].band_offsets(vv)
        self._last_offsets = (lo_off, hi_off)
        return hs, rets, sig

    def forecast_path(self, n_days: int, origin_date=None) -> pd.DataFrame:
        """Daily forecast for the next n business days after `origin_date`
        (default: last available day), with an 80% likely range."""
        origin_date = pd.Timestamp(origin_date) if origin_date is not None else self.table.index[-1]
        row = self.table.loc[[origin_date]]
        base_price = float(self.raw.loc[origin_date, "Close"])
        hs, rets, sig = self._curve(row)
        xs = [0] + hs
        dates = self._future_index(origin_date, n_days)
        days = np.arange(1, n_days + 1)
        max_h = hs[-1]
        r = np.interp(np.minimum(days, max_h), xs, [0] + [rets[h] for h in hs])
        s = np.interp(np.minimum(days, max_h), xs, [0] + [sig[h] for h in hs])
        lo_off, hi_off = getattr(self, "_last_offsets", ({}, {}))
        lo_i = np.interp(np.minimum(days, max_h), xs, [0] + [lo_off.get(h, -Z80 * sig[h]) for h in hs])
        hi_i = np.interp(np.minimum(days, max_h), xs, [0] + [hi_off.get(h, Z80 * sig[h]) for h in hs])
        beyond = days > max_h
        r[beyond] = rets[max_h] * days[beyond] / max_h            # extend the longest-horizon drift
        s[beyond] = sig[max_h] * np.sqrt(days[beyond] / max_h)    # uncertainty keeps growing
        grow = np.sqrt(np.maximum(days, 1) / max_h)
        lo_i[beyond] = lo_off.get(max_h, -Z80 * sig[max_h]) * grow[beyond]
        hi_i[beyond] = hi_off.get(max_h, Z80 * sig[max_h]) * grow[beyond]
        close = base_price * (1 + r)
        out = pd.DataFrame({"Predicted_Close": close,
                            "Lower_80": base_price * (1 + r + lo_i),
                            "Upper_80": base_price * (1 + r + hi_i)}, index=dates)
        opens = np.r_[np.nan, close[:-1]]
        if self.open_model is not None:
            opens[0] = base_price * (1 + float(self.open_model.predict(row)[0]))
        else:
            opens[0] = base_price
        out["Predicted_Open"] = opens
        return out

    def predict_date(self, target_date, buy_threshold=0.01, sell_threshold=0.01):
        target_date = pd.Timestamp(target_date)
        last = self.table.index[-1]

        def signal_for(pred, ref):
            pct = (pred - ref) / ref
            sig = "BUY" if pct > buy_threshold else "SELL" if pct < -sell_threshold else "HOLD"
            return sig, pct

        if target_date <= last:
            if target_date not in self.raw.index:
                nearest = self.raw.index[self.raw.index.get_indexer([target_date], method="nearest")[0]]
                raise ValueError(f"{target_date.date()} is not a trading day in the dataset (nearest: {nearest.date()}).")
            pos = self.raw.index.get_loc(target_date)
            prev = self.raw.index[pos - 1] if pos > 0 else None
            if prev is None or prev not in self.table.index:
                raise ValueError("Not enough prior history to predict this date.")
            # Today's fitted models have seen this historical outcome. Refit
            # selection, weights and calibration using only outcomes known at
            # the previous close. The row at prev has an unknown next-day label.
            historical_raw = self.raw.loc[:prev]
            historical_table, historical_groups = build_feature_table(
                historical_raw, self.ticker, **self.feature_inputs, horizons=[1], intraday=self.intraday)
            row = historical_table.loc[[prev]]
            past = historical_table.loc[historical_table.index < prev]
            labeled = past.dropna(subset=["Fwd_Ret_1"])
            if len(labeled) < 300:
                raise ValueError("At least 300 prior feature rows are needed for an honest historical prediction.")
            hist_model = _train_horizon(labeled, historical_groups, "Fwd_Ret_1",
                                       vol=trailing_vol(self.raw.loc[:prev], 1)).fit_final(labeled, "Fwd_Ret_1")
            open_lab = past.dropna(subset=["Fwd_Open_1"])
            open_stats = _cv_score(open_lab, hist_model.cols, "Fwd_Open_1", lambda: _full_members(hist_model.cols))
            hist_open = HorizonModel(hist_model.cols, open_stats["weights"], open_stats["alpha"],
                                     open_stats["sigma"], hist_model.groups_used, [], open_stats["rmse"],
                                     open_stats["naive_rmse"], stack_coef=open_stats.get("stack_coef"),
                                     combination=open_stats.get("combination", "blend")).fit_final(open_lab, "Fwd_Open_1")
            ref = float(self.raw.loc[prev, "Close"])
            pred = ref * (1 + float(hist_model.predict(row)[0]))
            pred_open = ref * (1 + float(hist_open.predict(row)[0]))
            sig, pct = signal_for(pred, ref)
            v1 = trailing_vol(self.raw, 1).reindex([prev]).iloc[0]
            lo, hi = hist_model.band_offsets(float(v1) if pd.notna(v1) else None)
            return {"mode": "historical", "target_date": target_date, "predicted_price": pred,
                    "actual_price": float(self.raw.loc[target_date, "Close"]), "predicted_open": pred_open,
                    "actual_open": float(self.raw.loc[target_date, "Open"]), "reference_price": ref,
                    "reference_date": prev, "signal": sig, "pct_change": pct,
                    "lower_80": pred + lo * ref, "upper_80": pred + hi * ref}

        if getattr(self, "trades_weekends", False):
            n = (target_date.normalize() - last.normalize()).days
        else:
            n = int(np.busday_count(last.date() + datetime.timedelta(days=1),
                                    target_date.date() + datetime.timedelta(days=1)))
        path = self.forecast_path(max(n, 1))
        if target_date not in path.index:
            nearest = path.index[path.index.get_indexer([target_date], method="nearest")[0]]
            raise ValueError(f"{target_date.date()} is not a forecast day (nearest: {nearest.date()}).")
        p = path.loc[target_date]
        ref = float(self.raw["Close"].iloc[-1])
        sig, pct = signal_for(p["Predicted_Close"], ref)
        return {"mode": "future", "target_date": target_date, "predicted_price": float(p["Predicted_Close"]),
                "actual_price": None, "predicted_open": float(p["Predicted_Open"]), "actual_open": None,
                "reference_price": ref, "reference_date": last, "signal": sig, "pct_change": pct,
                "lower_80": float(p["Lower_80"]), "upper_80": float(p["Upper_80"]), "forecast_path": path}

    def scan_best_window(self, horizon_days=30, cost_pct=0.30, min_prob=0.55):
        """
        Score every (buy, sell) pair on net-of-cost return, the uncertainty of that
        HOLD, and how long the money is tied up.

        cost_pct : round-trip cost in percent. 0.30 is a realistic Indian equity
                   round trip (brokerage + STT + exchange fees + slippage).
        min_prob : the minimum modelled probability of finishing in profit before a
                   window may be recommended at all.

        Returns the shortest qualifying window (the question is how to stop waiting
        two months for 1.4%), plus the best risk-adjusted and best annualised
        windows, a frontier by holding length, and an explicit verdict when nothing
        is worth trading.
        """
        from math import erf, sqrt

        path = self.forecast_path(horizon_days)
        px = path["Predicted_Close"].to_numpy(dtype=float)
        idx = path.index
        n = len(px)
        # sigma of the LEVEL at each step, read back out of the model's own 80% band
        sig = (path["Upper_80"].to_numpy(dtype=float)
               - path["Lower_80"].to_numpy(dtype=float)) / (2 * 1.2816)
        cost = cost_pct / 100.0

        def norm_cdf(z):
            return 0.5 * (1 + erf(z / sqrt(2)))

        cands = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                days = max((idx[j] - idx[i]).days, 1)
                gross = px[j] / px[i] - 1.0
                net = gross - cost
                # the hold's own uncertainty: independent increments between i and j
                var = max(sig[j] ** 2 - sig[i] ** 2, 1e-12)
                sd_ret = sqrt(var) / px[i]
                t = net / sd_ret if sd_ret > 0 else 0.0
                cands.append({
                    "buy_date": idx[i], "buy_price": float(px[i]),
                    "sell_date": idx[j], "sell_price": float(px[j]),
                    "hold_days": days, "hold_steps": j - i,
                    "gross_pct": gross * 100, "net_pct": net * 100,
                    "prob_profit": norm_cdf(t) * 100, "t_stat": t,
                    "annualised_pct": ((1 + net) ** (365.0 / days) - 1) * 100 if net > -1 else float("nan"),
                })

        res = {"forecast_path": path, "cost_pct": cost_pct, "min_prob": min_prob,
               "candidates": len(cands)}
        good = [c for c in cands if c["net_pct"] > 0 and c["prob_profit"] >= min_prob * 100]
        res["qualifying"] = len(good)

        # Is there any TIMING edge, or is this just buy-and-hold riding the drift?
        bh = next((c for c in cands if c["buy_date"] == idx[0] and c["sell_date"] == idx[-1]), None)
        res["buy_hold"] = bh
        if good:
            best_short = min(good, key=lambda c: (c["hold_days"], -c["t_stat"]))
            res.update(best_short)
            res["best_short"] = best_short
            res["best_risk_adj"] = max(good, key=lambda c: c["t_stat"])
            res["best_annual"] = max(good, key=lambda c: c["annualised_pct"])
            res["verdict"] = "tradeable"
            # timing edge = does the chosen window beat simply holding the whole window?
            res["beats_buy_hold"] = bool(bh and best_short["annualised_pct"] > bh["annualised_pct"])
        else:
            best_any = max(cands, key=lambda c: c["t_stat"]) if cands else None
            res.update(best_any or {})
            res["best_short"] = res["best_risk_adj"] = res["best_annual"] = best_any
            res["verdict"] = "no_trade"
            res["beats_buy_hold"] = False

        # frontier: the best net return achievable for each holding length
        by_len = {}
        for c in cands:
            k = c["hold_steps"]
            if k not in by_len or c["t_stat"] > by_len[k]["t_stat"]:
                by_len[k] = c
        res["frontier"] = [by_len[k] for k in sorted(by_len)]

        # kept so older callers and the CSV download keep working
        res["potential_gain_pct"] = res.get("gross_pct")
        return res

    def _today_vol(self, h):
        v = trailing_vol(self.raw, h).iloc[-1]
        return float(v) if pd.notna(v) else None

    def vol_outlook(self, h=21):
        """HAR-RV's view of the volatility of the next h days, and the volatility
        realised over the last 21 days, both annualised. Kept out of the bands
        (it calibrated worse there) but it forecasts volatility itself 12.1%
        better than the trailing window, so it is worth reporting."""
        m = getattr(self, "vol_models", {}).get(h)
        realised = float(trailing_vol(self.raw, 1).iloc[-1] * np.sqrt(252)) \
            if pd.notna(trailing_vol(self.raw, 1).iloc[-1]) else None
        if m is None:
            return {"forecast": None, "realised": realised, "model": "unavailable"}
        return {"forecast": m.predict_latest(self.raw), "realised": realised,
                "model": "HAR-RV", "skill_vs_trailing": m.skill}

    def summary(self):
        """Per-horizon: groups used, signal strength, validation skill."""
        rows = []
        for h, m in sorted(self.models.items()):
            rows.append({"Horizon (trading days)": h,
                         # .get, not [] -- a group added without a label is a
                         # cosmetic gap, not a reason to crash the whole tab
                         "Feature groups used": ", ".join(GROUP_LABELS.get(g, g)
                                                          for g in m.groups_used),
                         "Combination": getattr(m, "combination", "blend"),
                         "Signal strength (α)": round(m.alpha, 3),
                         "Val skill vs naive (%)": round((1 - m.val_rmse / m.val_naive_rmse) * 100, 2),
                         "80% range (±%)": round(Z80 * m.band_sigma(self._today_vol(h)) * 100, 2)})
        return pd.DataFrame(rows)


# ------------------------------------------------------------------------
# Honest held-out evaluation
# ------------------------------------------------------------------------
def evaluate_engine(raw, ticker, extra=None, news_df=None, horizons=(1, 5, 21), social_df=None,
                    pooled_df=None):
    """
    Held-out test (last 15% of days, never used for selection or fitting),
    for each horizon: the full engine (all groups, validation-selected) vs.
    the price/volume-only model vs. the naive "no change" forecast.
    Training rows whose target window overlaps the test period are purged.
    """
    table, groups = build_feature_table(raw, ticker, extra, news_df, social_df, pooled_df,
                                       horizons=sorted(set(horizons) | {1}))
    labeled1 = table[table["Fwd_Ret_1"].notna()]
    n = len(labeled1)
    if n < 400:
        return pd.DataFrame()
    test_start = labeled1.index[int(n * 0.85)]
    rows = []
    for h in horizons:
        target = f"Fwd_Ret_{h}"
        lab = table[table[target].notna()]
        pos = np.arange(len(table))
        pos_map = pd.Series(pos, index=table.index)
        test_start_pos = pos_map[test_start]
        trainval = lab[(pos_map.reindex(lab.index) + h) < test_start_pos]
        test = lab[lab.index >= test_start]
        if len(test) < 30 or len(trainval) < 300:
            continue
        result = {"Horizon (days)": h, "Test days": len(test)}
        close_test = raw["Close"].reindex(test.index).values
        actual = close_test * (1 + test[target].values)
        naive_rmse = np.sqrt(np.mean((actual - close_test) ** 2))
        for label, grp_filter in (("All groups (selected)", None), ("Price/volume only", ["base"])):
            g = groups if grp_filter is None else {k: v for k, v in groups.items() if k in grp_filter}
            model = _train_horizon(trainval, g, target, vol=trailing_vol(raw, h))
            model.fit_final(trainval, target)
            pr = model.predict(test)
            pred = close_test * (1 + pr)
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            moving = pr != 0
            diracc = np.mean(np.sign(pr[moving]) == np.sign(test[target].values[moving])) * 100 if moving.any() else np.nan
            result[f"{label}: skill vs naive (%)"] = round((1 - rmse / naive_rmse) * 100, 2)
            result[f"{label}: direction acc (%)"] = round(diracc, 1) if moving.any() else None
            if grp_filter is None:
                result["Groups chosen"] = ", ".join(model.groups_used)
                result["Price RMSE"] = round(float(rmse), 4)
                result["Price MAE"] = round(float(np.mean(np.abs(actual - pred))), 4)
                test_vol = trailing_vol(raw, h).reindex(test.index)
                offsets = np.asarray([model.band_offsets(v if pd.notna(v) else None) for v in test_vol])
                actual_return = test[target].to_numpy()
                covered = (actual_return >= pr + offsets[:, 0]) & (actual_return <= pr + offsets[:, 1])
                result["80% interval coverage (%)"] = round(float(covered.mean() * 100), 1)
        rows.append(result)
    return pd.DataFrame(rows)
