"""
predict_future.py
------------------
Predicts the stock's closing price on specific FUTURE dates (dates
beyond your historical dataset) that you choose.

HOW THIS WORKS (important to understand and explain in your report):
Your trained models only ever see real historical data during training.
To predict several days into the future, this script does RECURSIVE
forecasting:
  1. Predict tomorrow's price using the last known real data.
  2. Append that prediction to the price history as if it were real.
  3. Recompute technical indicators (moving averages, RSI, etc.) using
     this extended series.
  4. Predict the day after using the newly extended series.
  5. Repeat until reaching your target date.

This means prediction error COMPOUNDS the further into the future you
go -- a forecast for tomorrow is far more reliable than one for 3
months from now, since later predictions are built on earlier
predictions rather than real data. This is a genuine, well-known
limitation of multi-step time-series forecasting, not a flaw specific
to this code -- it's worth explicitly discussing in your report.

Because future Open/High/Low/Volume are unknown, this script
approximates them (Open=High=Low=predicted Close, Volume=recent
average) purely so the feature-engineering pipeline has values to work
with. This is a simplification -- flag it as such in your report.

Usage:
    Edit FUTURE_DATES below, then run:
        python predict_future.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_stock_data
from features import build_features, FEATURE_COLUMNS
from models_tabular import TABULAR_MODEL_BUILDERS
from pipeline import returns_to_price
from stock_search import resolve_ticker

# ----------------------------- CONFIG ---------------------------------
TICKER = None                  # None = ask for a stock name when you run the script (e.g. set "MSFT" to skip the prompt)
HISTORY_START = "2014-01-01"
HISTORY_END = "2025-01-01"          # last date of real historical data used for training
MODEL_NAME = "Linear Regression"     # any key in TABULAR_MODEL_BUILDERS
TARGET_MODE = "return"               # "price" or "return" -- "return" is recommended
                                      # for multi-step recursive forecasting since it
                                      # doesn't have the extrapolation problem described
                                      # in the README.

# Dates you want a predicted price for. Must be actual business days
# (weekdays) -- weekends/holidays aren't trading days, so there's no
# price to predict for them.
FUTURE_DATES = [
    "2025-01-15",
    "2025-02-01",
    "2025-03-01",
]
# ------------------------------------------------------------------------

TARGET_COL_MAP = {"price": "Target_Close", "return": "Target_Return"}


def train_full_model():
    """Train on ALL available historical data (no held-out test set) --
    for future forecasting you want the model to have learned from as
    much real data as possible, unlike the train/val/test split used
    for model evaluation in main.py."""
    print(f"Loading historical data for {TICKER}...")
    raw = load_stock_data(TICKER, HISTORY_START, HISTORY_END)
    feat_df = build_features(raw)

    target_col = TARGET_COL_MAP[TARGET_MODE]
    X = feat_df[FEATURE_COLUMNS].values
    y = feat_df[target_col].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Training {MODEL_NAME} ({TARGET_MODE} mode) on {len(feat_df)} rows...")
    model = TABULAR_MODEL_BUILDERS[MODEL_NAME]()
    if MODEL_NAME == "XGBoost":
        model.fit(X_scaled, y)  # no eval_set available here; early stopping disabled
    else:
        model.fit(X_scaled, y)

    return model, scaler, raw


def recursive_forecast(model, scaler, raw_df, target_dates):
    """
    Iteratively forecasts forward, one business day at a time, from the
    end of raw_df up to the furthest requested date, then returns the
    predicted price on each specifically requested date.
    """
    target_dates = pd.to_datetime(sorted(target_dates))
    last_known_date = raw_df.index[-1]

    if target_dates[0] <= last_known_date:
        raise ValueError(
            f"Requested date {target_dates[0].date()} is not after the last "
            f"historical date {last_known_date.date()}. Future prediction "
            "only works for dates beyond your historical dataset."
        )

    working_df = raw_df.copy()
    results = {}

    # Generate the full sequence of business days we need to step through
    all_steps = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), end=target_dates[-1])

    for step_date in all_steps:
        feat_df = build_features(working_df, keep_latest=True)  # keep today's row (its target is unknown)
        latest_row = feat_df.iloc[[-1]]  # most recent fully-featured row

        X_latest = latest_row[FEATURE_COLUMNS].values
        X_latest_scaled = scaler.transform(X_latest)
        pred_native = model.predict(X_latest_scaled)[0]

        today_close = working_df["Close"].iloc[-1]
        if TARGET_MODE == "return":
            pred_price = today_close * (1 + pred_native)
        else:
            pred_price = pred_native

        # Approximate OHLCV for the new synthetic day so feature
        # engineering has something to work with on the next iteration.
        recent_volume_avg = working_df["Volume"].tail(10).mean()
        new_row = pd.DataFrame({
            "Open": [pred_price], "High": [pred_price], "Low": [pred_price],
            "Close": [pred_price], "Adj Close": [pred_price], "Volume": [recent_volume_avg],
        }, index=[step_date])
        working_df = pd.concat([working_df, new_row])

        if step_date in target_dates:
            results[step_date] = pred_price

    return results


def main():
    global TICKER
    TICKER = resolve_ticker(TICKER)
    model, scaler, raw = train_full_model()
    predictions = recursive_forecast(model, scaler, raw, FUTURE_DATES)

    print(f"\n===== PREDICTED {TICKER} CLOSING PRICES =====")
    print(f"(Model: {MODEL_NAME}, target mode: {TARGET_MODE})\n")
    for date in sorted(predictions.keys()):
        days_ahead = (date - raw.index[-1]).days
        print(f"  {date.date()}  ->  ${predictions[date]:.2f}   ({days_ahead} calendar days ahead)")

    print(
        "\nNote: accuracy degrades the further ahead the date is, since each "
        "step is forecast from the previous step's PREDICTION rather than "
        "real data (recursive forecasting -- see the module docstring). "
        "These are model estimates for research and backtesting, not "
        "financial advice."
    )


if __name__ == "__main__":
    main()
