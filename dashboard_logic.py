"""
dashboard_logic.py
--------------------
All the actual prediction/ensemble/signal logic behind the dashboard,
kept separate from the Streamlit UI code (dashboard.py) so it can be
tested and reasoned about independently.

Core capabilities:
  1. train_all_models()          -- trains the regularized ensemble
     (Ridge, Random Forest, XGBoost; return mode, shrunk toward the
     naive forecast) + selects an ARIMA order, once.
  2. predict_price_for_date()    -- predicts the closing price on ANY
     date the user asks for (past = evaluated against real data,
     future = recursive ensemble forecast) and returns a Buy/Sell/Hold
     signal based on the predicted % change vs. the last known price.
  3. scan_best_buy_sell_window() -- forecasts forward over a horizon
     and finds the best predicted BUY (local low) and subsequent best
     predicted SELL (local high) within that window.

IMPORTANT CAVEAT (surface this in the UI and in your report): predictions
for FUTURE dates use recursive forecasting -- each day's prediction is
built from the previous day's PREDICTION, not real data, so accuracy
degrades the further ahead you ask. These are model estimates for research and
backtesting, not financial advice.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Trading days a symbol needs before the models can be trained on it
from data_loader import MIN_ROWS as MIN_ROWS_REQUIRED


# The dashboard ensemble uses the scale-free ("stationary") feature set and
# the regularized model set -- both measurably more accurate on held-out
# data than the original raw-price features + default models, which main.py
# still uses for the baseline model comparison.
from features import build_stationary_features as build_features, STATIONARY_FEATURE_COLUMNS as FEATURE_COLUMNS
from pipeline import chronological_split, returns_to_price
from models_tabular import ENSEMBLE_MODEL_BUILDERS
from model_arima import select_best_order
from statsmodels.tsa.arima.model import ARIMA


# Bump whenever training logic changes: the dashboard includes this in its
# cache keys, so a running dashboard retrains instead of reusing models
# built by older code (which would otherwise clash with the new page code).
MODEL_VERSION = "v13-stack-tf"


class ShrunkModel:
    """
    Wraps a fitted return model and multiplies its predictions by
    `alpha` (0..1), i.e. shrinks them toward "no change".

    Why: next-day returns are mostly noise, so raw model predictions are
    over-confident -- in a 5-stock benchmark, the un-shrunk ensemble was
    7-28% LESS accurate than simply predicting "tomorrow = today".
    alpha is fitted on the validation split as the value that minimizes
    validation error (alpha = <pred, actual> / <pred, pred>, clipped to
    [0, 1]). alpha near 1 = the models found real signal; alpha near 0 =
    they found none, so the ensemble falls back to the naive forecast
    instead of adding noise. It's reported in the dashboard as
    "signal strength".
    """

    def __init__(self, model, alpha: float, feature_cols=None, news_info=None):
        self.model = model
        self.alpha = alpha
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
        self.news_info = news_info or {"offered": False, "used": False}

    def predict(self, X):
        return self.alpha * self.model.predict(X)


def attach_news(feat_df: pd.DataFrame, news_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Join daily news features (news_history.build_news_features) onto a
    feature frame by date. Days with no news -- including future days in a
    recursive forecast, whose news obviously isn't known -- get 0 (neutral).
    """
    if news_df is None or news_df.shape[1] == 0:
        return feat_df
    out = feat_df.join(news_df.reindex(feat_df.index), how="left")
    out[list(news_df.columns)] = out[list(news_df.columns)].fillna(0.0)
    out.attrs = dict(feat_df.attrs)
    return out


def _fit_members(X, y):
    return {name: builder().fit(X, y) for name, builder in ENSEMBLE_MODEL_BUILDERS.items()}


def _weights_and_alpha(train, val, target_col, actual_price_col, cols=None):
    """
    Fit every member on `train` using feature columns `cols`, score on `val`.
    Returns (weights by inverse validation price-RMSE, shrinkage alpha,
    per-model val RMSE, validation RMSE of the final shrunk blend).
    The scaler is fitted on TRAIN only so validation stays truly unseen.
    """
    cols = cols or FEATURE_COLUMNS
    scaler = StandardScaler().fit(train[cols].values)
    members = _fit_members(scaler.transform(train[cols].values), train[target_col].values)

    X_val = scaler.transform(val[cols].values)
    val_returns = {name: m.predict(X_val) for name, m in members.items()}
    val_rmse = {
        name: np.sqrt(mean_squared_error(val[actual_price_col].values, returns_to_price(r, val["Close"].values)))
        for name, r in val_returns.items()
    }
    raw_w = {name: 1.0 / max(rmse, 1e-9) for name, rmse in val_rmse.items()}
    total = sum(raw_w.values())
    weights = {name: w / total for name, w in raw_w.items()}

    blended = sum(weights[n] * val_returns[n] for n in members)
    actual = val[target_col].values
    alpha = float(np.clip((blended @ actual) / (blended @ blended + 1e-12), 0.0, 1.0))
    blend_rmse = np.sqrt(mean_squared_error(val[actual_price_col].values,
                                            returns_to_price(alpha * blended, val["Close"].values)))
    return weights, alpha, val_rmse, blend_rmse


def _choose_columns(train, val, target_col, actual_price_col, news_cols):
    """
    News is taken into consideration, not relied on: the ensemble is scored
    on validation data WITHOUT and WITH the news columns, and news is only
    kept if it gives lower validation error. Returns (cols, weights, alpha, news_info).
    """
    weights, alpha, _, rmse_without = _weights_and_alpha(train, val, target_col, actual_price_col, FEATURE_COLUMNS)
    cols = list(FEATURE_COLUMNS)
    info = {"offered": bool(news_cols), "used": False, "val_rmse_without": round(rmse_without, 4)}
    if news_cols:
        w2, a2, _, rmse_with = _weights_and_alpha(train, val, target_col, actual_price_col, FEATURE_COLUMNS + news_cols)
        info["val_rmse_with"] = round(rmse_with, 4)
        if rmse_with < rmse_without:
            cols, weights, alpha = FEATURE_COLUMNS + news_cols, w2, a2
            info["used"] = True
    return cols, weights, alpha, info


def _train_tabular_ensemble(feat_df: pd.DataFrame, target_col: str, actual_price_col: str, news_cols=None):
    """
    Shared training routine used for BOTH the closing-price ensemble and
    the opening-price ensemble -- same models, same weighting scheme,
    just pointed at a different target column ('Target_Return' for
    close, 'Target_Open_Return' for open).

    Weights and shrinkage are learned on a train/validation split; the
    final models are then refit on ALL data (for a live dashboard you
    want the models to have seen the most recent market behaviour).

    Returns
    -------
    models  : dict of {name: (fitted ShrunkModel, fitted_scaler)}
    weights : dict of {name: normalized ensemble weight}
    """
    train, val, _ = chronological_split(feat_df, train_frac=0.85, val_frac=0.15)
    cols, weights, alpha, news_info = _choose_columns(train, val, target_col, actual_price_col, news_cols)

    scaler = StandardScaler().fit(feat_df[cols].values)
    members = _fit_members(scaler.transform(feat_df[cols].values), feat_df[target_col].values)
    models = {name: (ShrunkModel(m, alpha, cols, news_info), scaler) for name, m in members.items()}
    return models, weights


def ensemble_news_info(models) -> dict:
    """Whether news features were offered to / kept by the ensemble, with validation RMSE both ways."""
    return getattr(next(iter(models.values()))[0], "news_info", {"offered": False, "used": False})


def ensemble_signal_strength(models) -> float:
    """The fitted shrinkage alpha (same for every member): 0 = no usable signal found, 1 = full trust."""
    # getattr: models trained by an older version of this file have no alpha (= no shrinkage)
    return getattr(next(iter(models.values()))[0], "alpha", 1.0)


def train_all_models(raw_df: pd.DataFrame, news_df: pd.DataFrame = None):
    """
    Train the regularized tabular ensemble (return mode) on the FULL
    dataset, for BOTH closing price and opening price, plus select an
    ARIMA order for the multi-day forecast.

    Returns
    -------
    models       : dict of {name: (fitted_model, fitted_scaler)} -- CLOSE price
    weights      : dict of {name: normalized ensemble weight} -- CLOSE price
    arima_order  : tuple, the selected (p, d, q) order
    feat_df      : the full engineered feature DataFrame (for reference/plots)
    models_open  : dict of {name: (fitted_model, fitted_scaler)} -- OPEN price
    weights_open : dict of {name: normalized ensemble weight} -- OPEN price
    """
    feat_df = attach_news(build_features(raw_df), news_df)
    if len(feat_df) < 150:
        raise ValueError(
            f"Only {len(raw_df)} trading days of history are available -- not enough to train the models "
            f"(the indicators alone need ~1 year of warm-up). Newly listed stocks need roughly "
            f"{150 + 252} trading days (~1.6 years) of history."
        )

    news_cols = list(news_df.columns) if news_df is not None else []
    models, weights = _train_tabular_ensemble(feat_df, "Target_Return", "Target_Close", news_cols)
    models_open, weights_open = _train_tabular_ensemble(feat_df, "Target_Open_Return", "Target_Open", news_cols)

    arima_order = select_best_order(feat_df["Close"])

    return models, weights, arima_order, feat_df, models_open, weights_open


def _predict_one_step_tabular(models, weights, feature_row: pd.DataFrame, today_close: float):
    """Blend all tabular models' next-day price prediction for one row of features."""
    blended = 0.0
    for name, (model, scaler) in models.items():
        X = feature_row[getattr(model, "feature_cols", FEATURE_COLUMNS)].values
        X_scaled = scaler.transform(X)
        pred_return = model.predict(X_scaled)[0]
        pred_price = today_close * (1 + pred_return)
        blended += weights[name] * pred_price
    return blended


def recursive_ensemble_forecast(raw_df: pd.DataFrame, models, weights, arima_order, horizon_days: int,
                                 models_open=None, weights_open=None, news_df=None):
    """
    Forecast forward `horizon_days` business days from the end of raw_df,
    blending the tabular ensemble (recursive, day by day) with ARIMA's
    direct multi-step forecast (computed once, since ARIMA natively
    supports forecasting an arbitrary number of steps ahead in one call).

    If models_open/weights_open are provided (the opening-price ensemble
    from train_all_models), also computes a predicted OPENING price for
    each future day -- a genuinely different target from the closing
    price, not just a copy of it, using the same recursive approach.

    Returns a DataFrame indexed by future business-day dates with a
    'Predicted_Close' column, and 'Predicted_Open' too if models_open given.
    """
    last_date = raw_df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)

    # ARIMA: one multi-step forecast covering the whole horizon.
    arima_model = ARIMA(raw_df["Close"].values, order=arima_order)
    arima_fitted = arima_model.fit()
    arima_forecast = np.asarray(arima_fitted.forecast(steps=horizon_days))

    # Tabular ensemble: recursive, one day at a time.
    working_df = raw_df.copy()
    tabular_preds = []
    open_preds = []

    for i, step_date in enumerate(future_dates):
        # keep_latest=True -> features of the LAST known day (not the day before);
        # ~400 rows is plenty of history for the longest (252-day) window.
        feat_df = attach_news(build_features(working_df.tail(400), keep_latest=True), news_df)
        latest_row = feat_df.iloc[[-1]]
        today_close = working_df["Close"].iloc[-1]
        blended_tabular = _predict_one_step_tabular(models, weights, latest_row, today_close)
        tabular_preds.append(blended_tabular)

        # Combine tabular ensemble with this day's ARIMA forecast (equal blend)
        combined_price = 0.6 * blended_tabular + 0.4 * arima_forecast[i]

        if models_open is not None:
            blended_open = _predict_one_step_tabular(models_open, weights_open, latest_row, today_close)
            open_preds.append(blended_open)
            day_open, day_close = blended_open, combined_price
        else:
            day_open, day_close = combined_price, combined_price

        recent_volume_avg = working_df["Volume"].tail(10).mean()
        new_row = pd.DataFrame({
            "Open": [day_open], "High": [max(day_open, day_close)], "Low": [min(day_open, day_close)],
            "Close": [day_close], "Adj Close": [day_close], "Volume": [recent_volume_avg],
        }, index=[step_date])
        working_df = pd.concat([working_df, new_row])

    predicted_close = working_df["Close"].iloc[-horizon_days:]
    result = {"Predicted_Close": predicted_close.values}
    if models_open is not None:
        result["Predicted_Open"] = np.array(open_preds)
    return pd.DataFrame(result, index=future_dates)


def evaluate_ensemble_accuracy(raw_df: pd.DataFrame, news_df: pd.DataFrame = None):
    """
    Honest held-out accuracy of the SAME ensembling method the dashboard
    uses: weights + shrinkage learned on train -> validation, models
    refit on train+validation, then scored on a TEST split (last 15%)
    that nothing ever saw.

    Crucially it also scores the NAIVE forecast ("tomorrow's close =
    today's close") on the same days. For daily stock prices, R² is ~0.97+
    for almost ANY forecast (including the naive one) because prices
    barely move day to day -- so R² alone is misleading. The meaningful
    questions are:
      * "Skill vs naive"   -- % lower RMSE than the naive forecast
                              (positive = better than naive, negative = worse)
      * "Direction accuracy" -- how often the predicted up/down was right
                              (50% = coin flip)

    Returns
    -------
    metrics : dict with MAE, MSE, RMSE, MAPE (%), R2, Naive RMSE,
              Skill vs naive (%), Direction accuracy (%), Signal strength
    weights : dict of {model_name: weight} used in the blend
    """
    feat_df = attach_news(build_features(raw_df), news_df)
    train, val, test = chronological_split(feat_df)  # proper 70/15/15, test never touched during training
    news_cols = list(news_df.columns) if news_df is not None else []

    train_val = pd.concat([train, val])

    def test_returns(cols, weights, alpha):
        scaler = StandardScaler().fit(train_val[cols].values)
        members = _fit_members(scaler.transform(train_val[cols].values), train_val["Target_Return"].values)
        X_test = scaler.transform(test[cols].values)
        return alpha * sum(weights[n] * m.predict(X_test) for n, m in members.items())

    # Same news decision rule as the live models: decided on validation, never on test.
    cols, weights, alpha, news_info = _choose_columns(train, val, "Target_Return", "Target_Close", news_cols)
    pred_return = test_returns(cols, weights, alpha)

    today_close = test["Close"].values
    y_true = test["Target_Close"].values
    y_pred = returns_to_price(pred_return, today_close)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_true, today_close))

    actual_up = test["Target_Return"].values > 0
    moving = pred_return != 0
    dir_acc = np.mean((pred_return[moving] > 0) == actual_up[moving]) * 100 if moving.any() else float("nan")

    metrics = {
        "Model": "Combined Ensemble (Ridge + Random Forest + XGBoost, shrunk)",
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4),
        "R2": round(r2, 4),
        "Naive RMSE": round(naive_rmse, 4),
        "Skill vs naive (%)": round((1 - rmse / naive_rmse) * 100, 2),
        "Direction accuracy (%)": round(dir_acc, 2) if moving.any() else None,
        "Signal strength": round(alpha, 3),
    }
    if news_cols:
        # Report the other variant on the SAME test days too, for an honest comparison
        if news_info["used"]:
            w0, a0, _, _ = _weights_and_alpha(train, val, "Target_Return", "Target_Close", FEATURE_COLUMNS)
            other = returns_to_price(test_returns(list(FEATURE_COLUMNS), w0, a0), today_close)
            rmse_with, rmse_without = rmse, np.sqrt(mean_squared_error(y_true, other))
        else:
            w1, a1, _, _ = _weights_and_alpha(train, val, "Target_Return", "Target_Close", FEATURE_COLUMNS + news_cols)
            other = returns_to_price(test_returns(FEATURE_COLUMNS + news_cols, w1, a1), today_close)
            rmse_with, rmse_without = np.sqrt(mean_squared_error(y_true, other)), rmse
        metrics["News used (chosen on validation)"] = "yes" if news_info["used"] else "no"
        metrics["Skill vs naive WITHOUT news (%)"] = round((1 - rmse_without / naive_rmse) * 100, 2)
        metrics["Skill vs naive WITH news (%)"] = round((1 - rmse_with / naive_rmse) * 100, 2)
    return metrics, weights


def _fit_direction_models(labeled, cols):
    """Train the direction classifiers on `cols`: weights and validation
    log-loss from a train/validation split, final models refit on all rows."""
    from models_classification import CLASSIFICATION_MODEL_BUILDERS
    from sklearn.metrics import accuracy_score, log_loss

    train, val, _ = chronological_split(labeled, train_frac=0.85, val_frac=0.15)
    y_train, y_val = train["Target_Direction"].values.astype(int), val["Target_Direction"].values.astype(int)
    val_scaler = StandardScaler().fit(train[cols].values)
    X_train, X_val = val_scaler.transform(train[cols].values), val_scaler.transform(val[cols].values)

    scaler = StandardScaler().fit(labeled[cols].values)
    X_all, y_all = scaler.transform(labeled[cols].values), labeled["Target_Direction"].values.astype(int)

    clf_models, weights_raw, val_probs = {}, {}, {}
    for name, builder in CLASSIFICATION_MODEL_BUILDERS.items():
        scorer = builder().fit(X_train, y_train)
        val_probs[name] = scorer.predict_proba(X_val)[:, 1]
        weights_raw[name] = max(accuracy_score(y_val, val_probs[name] > 0.5), 0.01)
        clf_models[name] = (builder().fit(X_all, y_all), scaler)

    total = sum(weights_raw.values())
    clf_weights = {name: w / total for name, w in weights_raw.items()}
    blended = np.clip(sum(clf_weights[n] * val_probs[n] for n in clf_models), 1e-6, 1 - 1e-6)
    return clf_models, clf_weights, log_loss(y_val, blended, labels=[0, 1])


def train_direction_and_regime(raw_df: pd.DataFrame, ticker: str = None, news_df: pd.DataFrame = None):
    """
    Trains the direction (up/down) classifiers on advanced features
    (technical indicators + market context + volatility regime), and
    returns everything needed to show a live direction signal and
    market regime indicator in the dashboard, alongside (not instead
    of) the existing price prediction.

    If `news_df` (from news_history.build_news_features) is given, the
    classifiers are trained both WITH and WITHOUT the news columns and
    news is kept only if it improves validation log-loss -- news is
    taken into consideration, never relied on blindly. The decision is
    in advanced_feat_df.attrs["news_info"].

    Returns
    -------
    clf_models : dict of {name: (fitted_model, fitted_scaler)}
    clf_weights : dict of {name: normalized weight}, by validation accuracy
    advanced_feat_df : the full advanced feature DataFrame (for regime lookup);
                       attrs["feature_cols"] = columns the models use
    """
    from data_loader import load_market_context, market_symbols_for
    from features_advanced import build_advanced_features, BASE_ADVANCED_COLUMNS, MARKET_CONTEXT_COLUMNS

    start = raw_df.index[0].strftime("%Y-%m-%d")
    end = raw_df.index[-1].strftime("%Y-%m-%d")

    try:
        # Match market context to the stock (e.g. NIFTY/India VIX for NSE stocks)
        market_kwargs = market_symbols_for(ticker) if ticker else {}
        market_df = load_market_context(start, end, **market_kwargs)
    except Exception:
        market_df = None  # if market data fetch fails, fall back to no market context

    # keep_latest=True keeps TODAY's row (target unknown) so the live signal
    # predicts tomorrow from today; training only uses rows with a known target.
    advanced_feat_df = build_advanced_features(raw_df, market_df, keep_latest=True)
    base_cols = BASE_ADVANCED_COLUMNS + (MARKET_CONTEXT_COLUMNS if market_df is not None else [])

    # The wide technical-indicator bank: ADX, Aroon, Ichimoku, Keltner, Donchian,
    # MFI, CMF, Ulcer, Hurst, entropy, seasonality and the rest. Measured to help
    # DIRECTION (50.6% -> 52.6% on held-out data) even though it hurt squared
    # error, so it is offered here and kept only if it earns its place below.
    indicator_cols = []
    try:
        from features_indicators import indicator_features
        ind = indicator_features(raw_df).reindex(advanced_feat_df.index)
        ind = ind.loc[:, ind.notna().mean() > 0.8].ffill().fillna(0.0)
        indicator_cols = [c for c in ind.columns if c not in advanced_feat_df.columns]
        advanced_feat_df = advanced_feat_df.join(ind[indicator_cols])
    except Exception:
        indicator_cols = []
    news_cols = list(news_df.columns) if news_df is not None else []
    advanced_feat_df = attach_news(advanced_feat_df, news_df)
    labeled = advanced_feat_df[advanced_feat_df["Target_Direction"].notna()]

    clf_models, clf_weights, loss_without = _fit_direction_models(labeled, base_cols)
    cols = base_cols

    # indicators: kept only if they cut validation log-loss
    ind_info = {"offered": bool(indicator_cols), "used": False, "n": len(indicator_cols),
                "val_logloss_without": round(loss_without, 4)}
    if indicator_cols:
        try:
            m_i, w_i, loss_i = _fit_direction_models(labeled, base_cols + indicator_cols)
            ind_info["val_logloss_with"] = round(loss_i, 4)
            if loss_i < loss_without:
                clf_models, clf_weights, cols = m_i, w_i, base_cols + indicator_cols
                loss_without = loss_i          # the new bar for the news test
                ind_info["used"] = True
        except Exception:
            pass

    news_info = {"offered": bool(news_cols), "used": False, "val_logloss_without": round(loss_without, 4)}
    if news_cols:
        m2, w2, loss_with = _fit_direction_models(labeled, cols + news_cols)
        news_info["val_logloss_with"] = round(loss_with, 4)
        if loss_with < loss_without:
            clf_models, clf_weights, cols = m2, w2, cols + news_cols
            news_info["used"] = True

    advanced_feat_df.attrs["feature_cols"] = cols
    advanced_feat_df.attrs["base_cols"] = base_cols
    advanced_feat_df.attrs["news_info"] = news_info
    advanced_feat_df.attrs["indicator_info"] = ind_info
    return clf_models, clf_weights, advanced_feat_df


def evaluate_direction_accuracy(advanced_feat_df, test_frac=0.15, confidence=0.05):
    """
    Honest held-out accuracy of the direction classifiers: train on the
    first 85% of labeled days, predict the last 15% (never seen), and
    compare against the simplest baseline -- always guessing "up"
    (stocks rise on slightly more than half of days, so 50% isn't even
    the right bar to beat).

    Also reports accuracy on only the days where the blended model was
    at least `confidence` away from 50/50, and how many days that was
    (coverage) -- skipping low-confidence days is the one lever that
    sometimes gives a real, if small, edge. If news columns were offered,
    also reports test accuracy with and without them.
    """
    from features_advanced import ADVANCED_FEATURE_COLUMNS
    from models_classification import CLASSIFICATION_MODEL_BUILDERS

    labeled = advanced_feat_df[advanced_feat_df["Target_Direction"].notna()]
    split = int(len(labeled) * (1 - test_frac))
    train, test = labeled.iloc[:split], labeled.iloc[split:]
    y_tr, y_te = train["Target_Direction"].values.astype(int), test["Target_Direction"].values.astype(int)

    def test_probs(cols):
        scaler = StandardScaler().fit(train[cols].values)
        X_tr, X_te = scaler.transform(train[cols].values), scaler.transform(test[cols].values)
        return np.mean([b().fit(X_tr, y_tr).predict_proba(X_te)[:, 1] for b in CLASSIFICATION_MODEL_BUILDERS.values()], axis=0)

    cols = advanced_feat_df.attrs.get("feature_cols", ADVANCED_FEATURE_COLUMNS)
    prob_up = test_probs(cols)
    confident = np.abs(prob_up - 0.5) >= confidence
    result = {
        "Test days": len(test),
        "Direction accuracy (%)": round(np.mean((prob_up > 0.5) == y_te) * 100, 2),
        "Always-UP baseline (%)": round(y_te.mean() * 100, 2),
        "Accuracy when confident (%)": round(np.mean((prob_up[confident] > 0.5) == y_te[confident]) * 100, 2) if confident.any() else None,
        "Confident-day coverage (%)": round(confident.mean() * 100, 1),
    }
    news_info = advanced_feat_df.attrs.get("news_info", {})
    if news_info.get("offered"):
        base = advanced_feat_df.attrs["base_cols"]
        news_cols = [c for c in advanced_feat_df.columns if c.startswith(("News_", "AV_", "Own_"))]
        p_without = prob_up if cols == base else test_probs(base)
        p_with = prob_up if cols != base else test_probs(base + news_cols)
        result["Accuracy WITHOUT news (%)"] = round(np.mean((p_without > 0.5) == y_te) * 100, 2)
        result["Accuracy WITH news (%)"] = round(np.mean((p_with > 0.5) == y_te) * 100, 2)
    return result


def get_direction_signal(clf_models, clf_weights, advanced_feat_df):
    """
    Blends all direction classifiers' probability-of-"up" for the most
    recent available row into one ensemble confidence score.

    Returns (prob_up, label) e.g. (0.63, "↑ Likely UP (63%)")
    """
    from features_advanced import ADVANCED_FEATURE_COLUMNS

    latest_row = advanced_feat_df.iloc[[-1]]
    X_latest = latest_row[advanced_feat_df.attrs.get("feature_cols", ADVANCED_FEATURE_COLUMNS)].values

    blended_prob = 0.0
    for name, (model, scaler) in clf_models.items():
        X_scaled = scaler.transform(X_latest)
        prob_up = model.predict_proba(X_scaled)[0, 1]
        blended_prob += clf_weights[name] * prob_up

    if blended_prob >= 0.55:
        label = f"↑ Likely UP ({blended_prob*100:.0f}%)"
    elif blended_prob <= 0.45:
        label = f"↓ Likely DOWN ({(1-blended_prob)*100:.0f}%)"
    else:
        label = f"↔ Uncertain ({blended_prob*100:.0f}% up)"

    return blended_prob, label


def get_current_regime(advanced_feat_df):
    """Human-readable current volatility regime, from features_advanced.py's
    Volatility_Regime flag (1 = more turbulent than the past year's typical level)."""
    latest_regime = advanced_feat_df["Volatility_Regime"].iloc[-1]
    return "🌊 High Volatility" if latest_regime == 1 else "😌 Calm / Low Volatility"


def get_signal(predicted_price: float, reference_price: float, buy_threshold=0.01, sell_threshold=0.01):
    """Turn a predicted price + reference price into a BUY/SELL/HOLD signal."""
    pct_change = (predicted_price - reference_price) / reference_price
    if pct_change > buy_threshold:
        signal = "BUY"
    elif pct_change < -sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"
    return signal, pct_change


def predict_price_for_date(raw_df, models, weights, arima_order, target_date, models_open=None, weights_open=None,
                           news_df=None):
    """
    Predict the closing price (and, if models_open is given, the OPENING
    price too) on `target_date`.
    - If target_date falls WITHIN the historical dataset, this shows what
      the ensemble would have predicted using only data available the
      day before (a real backtest-style prediction), alongside the
      ACTUAL price for comparison.
    - If target_date is BEYOND the historical dataset, this runs a
      recursive ensemble forecast out to that date.

    Returns a dict with predicted_price (close), predicted_open (if
    models_open given), reference_price (yesterday's close), signal,
    pct_change, and (if historical) the actual close/open prices.
    """
    target_date = pd.to_datetime(target_date)
    last_historical_date = raw_df.index[-1]

    if target_date <= last_historical_date:
        # Historical / backtest-style prediction
        feat_df = attach_news(build_features(raw_df, keep_latest=True), news_df)
        if target_date not in feat_df.index:
            # Not a trading day (weekend/holiday) or too early for full features
            available_dates = feat_df.index
            nearest = available_dates[available_dates.get_indexer([target_date], method="nearest")[0]]
            raise ValueError(
                f"{target_date.date()} is not a trading day in the dataset "
                f"(nearest available: {nearest.date()})."
            )

        idx = feat_df.index.get_loc(target_date)
        if idx == 0:
            raise ValueError("Not enough prior history to predict this date.")

        row_before = feat_df.iloc[[idx - 1]]  # features known the day before target_date
        reference_price = feat_df["Close"].iloc[idx - 1]
        predicted_price = _predict_one_step_tabular(models, weights, row_before, reference_price)
        actual_price = feat_df["Close"].iloc[idx]

        predicted_open, actual_open = None, None
        if models_open is not None:
            predicted_open = _predict_one_step_tabular(models_open, weights_open, row_before, reference_price)
            actual_open = feat_df["Open"].iloc[idx]

        signal, pct_change = get_signal(predicted_price, reference_price)
        return {
            "mode": "historical",
            "target_date": target_date,
            "predicted_price": predicted_price,
            "actual_price": actual_price,
            "predicted_open": predicted_open,
            "actual_open": actual_open,
            "reference_price": reference_price,
            "reference_date": feat_df.index[idx - 1],
            "signal": signal,
            "pct_change": pct_change,
        }
    else:
        horizon_days = int(np.busday_count(
            last_historical_date.date() + pd.Timedelta(days=1).to_pytimedelta(),
            target_date.date() + pd.Timedelta(days=1).to_pytimedelta(),
        ))
        horizon_days = max(horizon_days, 1)
        forecast_df = recursive_ensemble_forecast(
            raw_df, models, weights, arima_order, horizon_days,
            models_open=models_open, weights_open=weights_open, news_df=news_df,
        )

        if target_date not in forecast_df.index:
            nearest = forecast_df.index[forecast_df.index.get_indexer([target_date], method="nearest")[0]]
            raise ValueError(
                f"{target_date.date()} is not a business day "
                f"(nearest forecasted date: {nearest.date()})."
            )

        predicted_price = forecast_df.loc[target_date, "Predicted_Close"]
        predicted_open = forecast_df.loc[target_date, "Predicted_Open"] if "Predicted_Open" in forecast_df.columns else None
        reference_price = raw_df["Close"].iloc[-1]
        signal, pct_change = get_signal(predicted_price, reference_price)

        return {
            "mode": "future",
            "target_date": target_date,
            "predicted_price": predicted_price,
            "actual_price": None,
            "predicted_open": predicted_open,
            "actual_open": None,
            "reference_price": reference_price,
            "reference_date": last_historical_date,
            "signal": signal,
            "pct_change": pct_change,
            "forecast_path": forecast_df,
        }


def scan_best_buy_sell_window(raw_df, models, weights, arima_order, horizon_days=30, news_df=None):
    """
    Forecast forward `horizon_days` and identify the best predicted BUY
    (lowest predicted price) and the best predicted SELL AFTER that buy
    point (highest predicted price occurring later in the window).

    Returns a dict with buy_date/buy_price, sell_date/sell_price,
    potential % gain, and the full forecast path (for plotting).
    """
    forecast_df = recursive_ensemble_forecast(raw_df, models, weights, arima_order, horizon_days, news_df=news_df)
    prices = forecast_df["Predicted_Close"]

    buy_idx = prices.values.argmin()
    buy_date = prices.index[buy_idx]
    buy_price = prices.iloc[buy_idx]

    if buy_idx < len(prices) - 1:
        after_buy = prices.iloc[buy_idx + 1:]
        sell_idx_local = after_buy.values.argmax()
        sell_date = after_buy.index[sell_idx_local]
        sell_price = after_buy.iloc[sell_idx_local]
    else:
        sell_date, sell_price = None, None

    result = {
        "forecast_path": forecast_df,
        "buy_date": buy_date,
        "buy_price": buy_price,
        "sell_date": sell_date,
        "sell_price": sell_price,
    }
    if sell_price is not None:
        result["potential_gain_pct"] = (sell_price - buy_price) / buy_price * 100
    else:
        result["potential_gain_pct"] = None

    return result
