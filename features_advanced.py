"""
features_advanced.py
----------------------
Extends the base feature set (features.py) with things that matter for
real-world use:

  1. MARKET CONTEXT -- VIX (volatility index) and S&P 500 returns. A
     stock rarely moves independent of the broader market; these give
     models signal that pure price/volume technical indicators can't
     capture on their own.
  2. VOLATILITY REGIME -- a simple "is the market currently calm or
     turbulent" flag (rolling 20-day volatility vs. its own recent
     history), since models often behave very differently in each.
  3. Target_Direction -- a binary up/down target for classification,
     alongside the existing Target_Close / Target_Return regression
     targets. Predicting direction is a fundamentally easier, more
     robust problem than predicting exact price -- useful as a second
     opinion alongside your price predictions, not a replacement.

This is a SEPARATE module from features.py on purpose: your existing
pipeline (main.py, dashboard.py, ensemble.py) keeps working exactly as
it did, untouched. This module is for the additional scripts
(train_classification.py, walk_forward.py) that build on top of it.

Usage:
    from data_loader import load_stock_data, load_market_context
    from features_advanced import build_advanced_features, ADVANCED_FEATURE_COLUMNS

    raw = load_stock_data("MSFT", "2014-01-01", "2025-01-01")
    market = load_market_context("2014-01-01", "2025-01-01")
    feat_df = build_advanced_features(raw, market)
"""

import numpy as np
import pandas as pd

from features import build_features, FEATURE_COLUMNS  # reuse all the base technical indicators


def build_advanced_features(stock_df: pd.DataFrame, market_df: pd.DataFrame = None, keep_latest: bool = False) -> pd.DataFrame:
    """
    Builds everything features.build_features() does, PLUS market
    context and regime features, PLUS a direction classification target.

    Parameters
    ----------
    stock_df : pd.DataFrame
        Raw OHLCV data for the stock (from load_stock_data).
    market_df : pd.DataFrame or None
        Output of load_market_context() -- VIX_Close, SP500_Close,
        indexed by Date. If None, market features are skipped (useful
        for recursive future-forecasting where forward VIX/S&P500
        values aren't available).

    keep_latest : bool
        True keeps the newest day(s), whose target isn't known yet
        (Target_Direction is NaN there) -- needed to predict the NEXT
        day from TODAY's features instead of yesterday's.

    Returns
    -------
    pd.DataFrame with all base features + (if market_df given)
    VIX_Close, VIX_Change, SP500_Return, Relative_Strength +
    Volatility_Regime + Target_Direction, with NaN rows dropped.
    """
    data = build_features(stock_df, keep_latest=keep_latest)  # base technical indicators + targets

    # --- Volatility regime: is the last 20 days calmer or more turbulent
    # than the preceding year's typical volatility? 1 = high-vol regime.
    vol_20d = data["Close"].pct_change().rolling(20).std()
    vol_median_1y = vol_20d.rolling(252, min_periods=60).median()
    data["Volatility_Regime"] = (vol_20d > vol_median_1y).astype(float)

    # --- Candlestick reading as a feature: rolling 3-day sum of detected
    # pattern bias x strength (candlestick_patterns.py). Computed on the
    # raw OHLC so it only uses candles up to and including each day.
    from candlestick_patterns import candle_bias_series
    data["Candle_Bias_3d"] = candle_bias_series(stock_df).reindex(data.index).fillna(0.0)

    # --- Market context (optional) ---
    if market_df is not None:
        merged = data.join(market_df, how="left")
        merged["VIX_Close"] = merged["VIX_Close"].ffill()
        merged["SP500_Close"] = merged["SP500_Close"].ffill()
        merged["VIX_Change"] = merged["VIX_Close"].pct_change()
        merged["SP500_Return"] = merged["SP500_Close"].pct_change()
        merged["Relative_Strength"] = merged["Return_1d"] - merged["SP500_Return"]
        data = merged

    # --- Direction target: 1 if next day's close is higher, else 0 ---
    data["Target_Direction"] = (data["Target_Return"] > 0).astype(float).where(data["Target_Return"].notna())

    feature_cols = BASE_ADVANCED_COLUMNS + (MARKET_CONTEXT_COLUMNS if market_df is not None else [])
    data = data.dropna(subset=feature_cols)
    if not keep_latest:
        data = data.dropna()
        data["Target_Direction"] = data["Target_Direction"].astype(int)
    return data


BASE_ADVANCED_COLUMNS = FEATURE_COLUMNS + ["Volatility_Regime", "Candle_Bias_3d"]
MARKET_CONTEXT_COLUMNS = ["VIX_Close", "VIX_Change", "SP500_Return", "Relative_Strength"]

# Use this when you built features WITH market_df:
ADVANCED_FEATURE_COLUMNS = BASE_ADVANCED_COLUMNS + MARKET_CONTEXT_COLUMNS
# Use this when you built features WITHOUT market_df (e.g. recursive forecasting):
ADVANCED_FEATURE_COLUMNS_NO_MARKET = BASE_ADVANCED_COLUMNS
