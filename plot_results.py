"""
plot_results.py
----------------
Generates the two plots you'll want for your report:
  1. actual_vs_predicted.png  -- actual test-period closing price vs.
     your model's predictions, as a line chart.
  2. portfolio_comparison.png -- portfolio value over time under your
     model's trading strategy vs. Buy-and-Hold.

Uses Linear Regression by default (it was the strongest performer in
both target modes), but you can swap MODEL_NAME / TARGET_MODE below to
plot any of the tabular models instead. This script only supports the
tabular models (Linear Regression, Random Forest, SVR, XGBoost) since
they train in seconds -- no need to wait through LSTM/GRU/Transformer
training again just to make a plot.

Run:
    python plot_results.py
"""

import matplotlib
matplotlib.use("Agg")  # safe for headless / notebook use; plots are saved to file
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_stock_data
from features import build_features, FEATURE_COLUMNS
from pipeline import chronological_split, returns_to_price
from models_tabular import TABULAR_MODEL_BUILDERS
from backtest import simulate_strategy, buy_and_hold_baseline
from stock_search import resolve_ticker

# ----------------------------- CONFIG ---------------------------------
TICKER = None                  # None = ask for a stock name when you run the script (e.g. set "MSFT" to skip the prompt)
START_DATE = "2014-01-01"
END_DATE = "2025-01-01"
MODEL_NAME = "Linear Regression"   # any key in TABULAR_MODEL_BUILDERS
TARGET_MODE = "return"             # "price" or "return"
INITIAL_INVESTMENT = 10_000.0
# ------------------------------------------------------------------------

TARGET_COL_MAP = {"price": "Target_Close", "return": "Target_Return"}


def main():
    global TICKER
    TICKER = resolve_ticker(TICKER)
    print(f"Loading data for {TICKER}...")
    raw = load_stock_data(TICKER, START_DATE, END_DATE)
    feat_df = build_features(raw)
    train, val, test = chronological_split(feat_df)

    target_col = TARGET_COL_MAP[TARGET_MODE]
    X_train = train[FEATURE_COLUMNS].values
    y_train = train[target_col].values
    X_test = test[FEATURE_COLUMNS].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training {MODEL_NAME} ({TARGET_MODE} mode)...")
    model = TABULAR_MODEL_BUILDERS[MODEL_NAME]()
    if MODEL_NAME == "XGBoost":
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(scaler.transform(val[FEATURE_COLUMNS].values), val[target_col].values)],
            verbose=False,
        )
    else:
        model.fit(X_train_scaled, y_train)

    preds_native = model.predict(X_test_scaled)
    today_close_test = test["Close"].values
    if TARGET_MODE == "return":
        preds_price = returns_to_price(preds_native, today_close_test)
    else:
        preds_price = preds_native

    actual_price = test["Target_Close"].values
    test_dates = test.index

    # ---- PLOT 1: Actual vs Predicted ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates, actual_price, label="Actual Close Price", color="#1f77b4", linewidth=1.8)
    ax.plot(test_dates, preds_price, label=f"{MODEL_NAME} Predicted ({TARGET_MODE})",
            color="#d62728", linewidth=1.5, linestyle="--")
    ax.set_title(f"{TICKER}: Actual vs Predicted Closing Price (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("actual_vs_predicted.png", dpi=150)
    print("Saved actual_vs_predicted.png")
    plt.close(fig)

    # ---- PLOT 2: Portfolio value over time (strategy vs Buy-and-Hold) ----
    summary, daily_log = simulate_strategy(
        test["Close"], pd.Series(preds_price), initial_investment=INITIAL_INVESTMENT
    )
    baseline_summary = buy_and_hold_baseline(test["Close"], initial_investment=INITIAL_INVESTMENT)

    # Build a day-by-day Buy-and-Hold portfolio value series for the same dates
    close_vals = test["Close"].reset_index(drop=True)
    shares = INITIAL_INVESTMENT / close_vals.iloc[0]
    baseline_portfolio = shares * close_vals

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates[:len(daily_log)], daily_log["Portfolio_Value"],
            label=f"{MODEL_NAME} Strategy ({TARGET_MODE})", color="#2ca02c", linewidth=1.8)
    ax.plot(test_dates, baseline_portfolio.values, label="Buy-and-Hold", color="#7f7f7f",
            linewidth=1.8, linestyle="--")
    ax.axhline(INITIAL_INVESTMENT, color="black", linewidth=0.8, linestyle=":", label="Initial Investment")
    ax.set_title(f"{TICKER}: Portfolio Value Over Time -- Strategy vs Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("portfolio_comparison.png", dpi=150)
    print("Saved portfolio_comparison.png")
    plt.close(fig)

    print("\nSummary:")
    print(f"  {MODEL_NAME} ({TARGET_MODE}) strategy: {summary}")
    print(f"  Buy-and-Hold baseline:               {baseline_summary}")


if __name__ == "__main__":
    main()
