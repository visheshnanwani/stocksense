"""
model_smoothing.py
--------------------
Simple Exponential Smoothing (SES) with a FIXED smoothing factor.

    level_t = alpha * price_t + (1 - alpha) * level_{t-1}
    forecast for t+1 = level_t

alpha is fixed at SMOOTHING_ALPHA = 0.3 (not fitted), as requested:
  * alpha close to 1 -> follows the latest price closely (little smoothing)
  * alpha close to 0 -> very smooth, reacts slowly to new prices
0.3 is a common textbook default: each day's price gets 30% weight and the
previous smoothed level 70%, so the weight of a price from k days ago is
0.3 * 0.7^k (about half of the total weight sits on the last ~2 days).

It's used in three places:
  1. main.py -- as a forecasting model in the model comparison
     ("Exponential Smoothing (alpha=0.3)").
  2. forecast_engine.py -- as the "smoothing" feature group (distance of the
     price from its smoothed level, and the level's recent slope), which the
     engine keeps only if it improves accuracy for that stock.
  3. dashboard.py -- as a chart overlay line.

The recursion only ever uses past and current prices, so there's no look-ahead.
"""

import pandas as pd

SMOOTHING_ALPHA = 0.3


def exponential_smoothing(series: pd.Series, alpha: float = SMOOTHING_ALPHA) -> pd.Series:
    """Smoothed level for every day (same as SES; starts from the first price)."""
    return series.ewm(alpha=alpha, adjust=False).mean()


def ses_one_step_forecast(close: pd.Series, alpha: float = SMOOTHING_ALPHA) -> pd.Series:
    """
    One-day-ahead SES forecasts: the value at date t is the forecast for the
    NEXT trading day's close, made with prices up to and including t.
    (Equivalent to statsmodels' SimpleExpSmoothing with smoothing_level=alpha,
    optimized=False, initialized at the first observation.)
    """
    return exponential_smoothing(close, alpha)


def smoothing_features(close: pd.Series, alpha: float = SMOOTHING_ALPHA) -> pd.DataFrame:
    """Scale-free features from the smoothed level, for the forecast engine."""
    level = exponential_smoothing(close, alpha)
    return pd.DataFrame({
        "SES_Dist": close / level - 1,                 # price vs its smoothed level
        "SES_Slope_5d": level / level.shift(5) - 1,    # direction of the smoothed trend
        "SES_Slope_20d": level / level.shift(20) - 1,
    }, index=close.index)
