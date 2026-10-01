"""
model_arima.py
--------------
ARIMA is univariate (uses only past Close prices, no other features),
so it doesn't fit the same fit/predict interface as the tabular models.
It's a classic statistical baseline -- fast to run, good to include as
a reference point in your comparison table.

Improvement: instead of a fixed (5,1,0) order, we do a small grid search
over a handful of (p,d,q) combinations and pick whichever gives the best
AIC on the training data. This is a lightweight version of what
auto_arima does, without needing an extra dependency.

Usage:
    order = select_best_order(train_close)
    preds = run_arima_forecast(train_close, n_test_steps, order=order)
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def select_best_order(train_close: pd.Series, p_values=(1, 2, 3, 5), d_values=(1,), q_values=(0, 1, 2)):
    """
    Small grid search over ARIMA orders, scored by AIC (lower is better).
    Skips any combination that fails to converge.
    """
    best_aic = np.inf
    best_order = (5, 1, 0)  # fallback default

    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train_close.values, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except Exception:
            continue

    return best_order


def run_arima_forecast(train_close: pd.Series, n_test_steps: int, order=None):
    """
    Fit ARIMA on the training closing-price series and forecast the
    next `n_test_steps` values (one-shot forecast, matching the length
    of your test set).

    Parameters
    ----------
    train_close : pd.Series
        Closing prices from the TRAIN split only (chronological).
    n_test_steps : int
        Number of steps to forecast = len(test set).
    order : tuple or None
        (p, d, q) ARIMA order. If None, automatically selects the best
        order via select_best_order().

    Returns
    -------
    np.ndarray of forecasted closing prices, length n_test_steps.
    """
    if order is None:
        order = select_best_order(train_close)

    model = ARIMA(train_close.values, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=n_test_steps)
    return np.asarray(forecast), order


def one_step_arima_forecast(train_close: pd.Series, full_close: pd.Series, test_start_idx: int, order=None):
    """
    FAIR one-day-ahead ARIMA predictions for the test period -- the same
    task every other model is scored on -- without re-fitting per day.

    Fits ARIMA once on the TRAIN closes, then re-runs those fixed
    parameters over the full real series (statsmodels' .apply()), so
    each day's prediction uses all REAL prices up to the day before.
    The one-shot run_arima_forecast() instead forecasts the entire test
    period (often 1-2 years) from the last training day, which turns
    into a nearly flat line and makes ARIMA look far worse than it is.

    Parameters
    ----------
    full_close : closes for train + val + test (contiguous, chronological)
    test_start_idx : position of the first test row in full_close

    Returns
    -------
    np.ndarray, where element i predicts full_close[test_start_idx + i + 1]
    (the NEXT day's close, matching Target_Close), and the order used.
    """
    if order is None:
        order = select_best_order(train_close)
    fitted = ARIMA(train_close.values, order=order).fit()
    applied = fitted.apply(full_close.values)          # same params, all real data
    one_step = np.asarray(applied.predict())           # one_step[i] predicts close[i] from close[:i]
    next_day = np.append(one_step[test_start_idx + 1:], applied.forecast(steps=1))
    return next_day, order


def rolling_arima_forecast(full_close: pd.Series, test_start_idx: int, order=(5, 1, 0)):
    """
    OPTIONAL / more rigorous alternative: re-fit ARIMA one step at a time,
    using all real data up to t-1 to predict t (walk-forward validation).
    This is slower but more realistic than a single one-shot forecast,
    since ARIMA's one-shot multi-step forecasts degrade quickly.

    Use this if you have time -- mention in your report if you use the
    faster one-shot version instead due to time constraints.
    """
    history = list(full_close.values[:test_start_idx])
    preds = []
    for t in range(test_start_idx, len(full_close)):
        model = ARIMA(history, order=order)
        fitted = model.fit()
        next_pred = fitted.forecast(steps=1)[0]
        preds.append(next_pred)
        history.append(full_close.values[t])  # add the REAL value, not the prediction
    return np.asarray(preds)
