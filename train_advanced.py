"""
train_advanced.py
--------------------
The "real use" upgrade: runs your EXISTING price-prediction models
(unchanged, still available) ALONGSIDE a direction (up/down)
classification track, using market context (VIX, S&P 500) and a
volatility regime feature, then backtests with realistic transaction
costs instead of a frictionless assumption.

This does NOT touch or replace main.py / dashboard.py -- your original
price-prediction pipeline still works exactly as before. This is an
additional, more realistic layer on top, for your own exploration
beyond the basic requirements.

What this adds over main.py:
  1. Market context features (VIX, S&P 500) -- see features_advanced.py
  2. A volatility regime feature
  3. Direction classification models (Logistic Regression, Random
     Forest, XGBoost) reported with accuracy/precision/recall/F1 --
     alongside your existing price-prediction metrics
  4. Realistic transaction costs (commission + slippage) in the
     backtest, so you see how much a strategy's paper profit survives
     real-world trading frictions

Run:
    python train_advanced.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
)

from data_loader import load_stock_data, load_market_context, market_symbols_for
from features_advanced import build_advanced_features, ADVANCED_FEATURE_COLUMNS
from pipeline import chronological_split, evaluate_predictions, returns_to_price
from models_tabular import TABULAR_MODEL_BUILDERS
from models_classification import CLASSIFICATION_MODEL_BUILDERS
from backtest import simulate_strategy, buy_and_hold_baseline
from stock_search import resolve_ticker

# ----------------------------- CONFIG ---------------------------------
TICKER = None                  # None = ask for a stock name when you run the script (e.g. set "MSFT" to skip the prompt)
START_DATE = "2014-01-01"
END_DATE = "2025-01-01"
INITIAL_INVESTMENT = 10_000.0
BUY_THRESHOLD = 0.01
SELL_THRESHOLD = 0.01

# Realistic trading frictions -- set both to 0.0 to reproduce the
# frictionless backtest used elsewhere in this project.
COMMISSION_PCT = 0.001   # 0.1%, a typical discount broker
SLIPPAGE_PCT = 0.0005    # 0.05%

# For classification: only trade when the model is at least this
# confident (probability of "up"), not just barely over 50%.
CLASSIFICATION_CONFIDENCE = 0.55
# ------------------------------------------------------------------------


def run_price_prediction(train, val, test):
    """Your existing price-prediction models (return mode), unchanged
    from models_tabular.py -- 'keeping the predict price option'."""
    X_train = train[ADVANCED_FEATURE_COLUMNS].values
    y_train = train["Target_Return"].values
    X_val = val[ADVANCED_FEATURE_COLUMNS].values
    y_val = val["Target_Return"].values
    X_test = test[ADVANCED_FEATURE_COLUMNS].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    today_close_test = test["Close"].values
    y_test_price = test["Target_Close"].values

    metrics_list = []
    price_preds = {}

    for name, builder in TABULAR_MODEL_BUILDERS.items():
        print(f"  [Price] Training {name}...")
        model = builder()
        if name == "XGBoost":
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        else:
            model.fit(X_train_scaled, y_train)

        preds_return = model.predict(X_test_scaled)
        preds_price = returns_to_price(preds_return, today_close_test)

        metrics_list.append(evaluate_predictions(y_test_price, preds_price, model_name=f"{name} (price)"))
        price_preds[name] = preds_price

    return metrics_list, price_preds


def run_direction_classification(train, val, test):
    """Predict UP/DOWN instead of exact price -- a second, more robust
    signal to cross-check against the price predictions above."""
    X_train = train[ADVANCED_FEATURE_COLUMNS].values
    y_train = train["Target_Direction"].values
    X_test = test[ADVANCED_FEATURE_COLUMNS].values
    y_test = test["Target_Direction"].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metrics_list = []
    direction_probs = {}

    for name, builder in CLASSIFICATION_MODEL_BUILDERS.items():
        print(f"  [Direction] Training {name}...")
        model = builder()
        model.fit(X_train_scaled, y_train)

        probs_up = model.predict_proba(X_test_scaled)[:, 1]
        preds = (probs_up >= 0.5).astype(int)

        metrics_list.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "F1": round(f1_score(y_test, preds, zero_division=0), 4),
        })
        direction_probs[name] = probs_up

    return metrics_list, direction_probs


def backtest_with_costs(test_df, price_preds, direction_probs):
    """Backtest BOTH the price-based and direction-based signals, with
    realistic transaction costs, compared to Buy-and-Hold."""
    results = []
    actual_close = test_df["Close"]

    # Price-prediction-based strategies (existing approach)
    for name, preds in price_preds.items():
        summary, _ = simulate_strategy(
            actual_close, pd.Series(preds), initial_investment=INITIAL_INVESTMENT,
            buy_threshold=BUY_THRESHOLD, sell_threshold=SELL_THRESHOLD,
            commission_pct=COMMISSION_PCT, slippage_pct=SLIPPAGE_PCT,
        )
        summary["Model"] = f"{name} (price-based)"
        results.append(summary)

    # Direction-based strategies: convert probability-of-up into an
    # implied "predicted price" so the same simulate_strategy function
    # can be reused -- a confident UP becomes a price prediction above
    # today's close, a confident DOWN becomes one below it.
    today_close = test_df["Close"].values
    for name, probs in direction_probs.items():
        implied_pct_change = (probs - 0.5) * 2 * BUY_THRESHOLD * 2  # scale prob into a % move
        implied_price = today_close * (1 + implied_pct_change)
        # Only trade when confident (probability far enough from 50/50)
        confident_mask = np.abs(probs - 0.5) >= (CLASSIFICATION_CONFIDENCE - 0.5)
        implied_price_confident = np.where(confident_mask, implied_price, today_close)

        summary, _ = simulate_strategy(
            actual_close, pd.Series(implied_price_confident), initial_investment=INITIAL_INVESTMENT,
            buy_threshold=BUY_THRESHOLD, sell_threshold=SELL_THRESHOLD,
            commission_pct=COMMISSION_PCT, slippage_pct=SLIPPAGE_PCT,
        )
        summary["Model"] = f"{name} (direction-based)"
        results.append(summary)

    baseline = buy_and_hold_baseline(actual_close, initial_investment=INITIAL_INVESTMENT)
    baseline["Model"] = "Buy-and-Hold (Baseline)"
    results.append(baseline)

    df = pd.DataFrame(results)
    cols = ["Model", "Initial_Investment", "Final_Value", "Total_Profit_Loss",
            "Percent_Return", "Num_Trades", "Total_Costs_Paid"]
    return df[cols].sort_values("Percent_Return", ascending=False).reset_index(drop=True)


def main():
    global TICKER
    TICKER = resolve_ticker(TICKER)
    print(f"Loading {TICKER} data + market context (volatility index, broad market index, sector)...")
    raw = load_stock_data(TICKER, START_DATE, END_DATE)
    market = load_market_context(START_DATE, END_DATE, **market_symbols_for(TICKER))

    feat_df = build_advanced_features(raw, market)
    print(f"{len(feat_df)} rows after advanced feature engineering "
          f"({len(ADVANCED_FEATURE_COLUMNS)} features, including market context + regime).")

    train, val, test = chronological_split(feat_df)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    print("\n--- Price prediction (your existing approach, keeping this option) ---")
    price_metrics, price_preds = run_price_prediction(train, val, test)
    price_df = pd.DataFrame(price_metrics).sort_values("RMSE")
    print(price_df.to_string(index=False))

    print("\n--- Direction classification (new, additional signal) ---")
    dir_metrics, dir_probs = run_direction_classification(train, val, test)
    dir_df = pd.DataFrame(dir_metrics).sort_values("Accuracy", ascending=False)
    print(dir_df.to_string(index=False))
    print(
        "\nNote: 50% accuracy is what a coin flip achieves. Anything consistently "
        "above ~52-55% on a liquid large-cap stock is considered meaningful in "
        "real quant finance -- treat very high numbers with suspicion (possible "
        "overfitting) rather than celebration."
    )

    print(f"\n--- Backtest with realistic costs (commission={COMMISSION_PCT*100}%, slippage={SLIPPAGE_PCT*100}%) ---")
    profit_df = backtest_with_costs(test, price_preds, dir_probs)
    print(profit_df.to_string(index=False))

    price_df.to_csv("advanced_price_metrics.csv", index=False)
    dir_df.to_csv("advanced_direction_metrics.csv", index=False)
    profit_df.to_csv("advanced_profitability_with_costs.csv", index=False)
    print("\nSaved advanced_price_metrics.csv, advanced_direction_metrics.csv, advanced_profitability_with_costs.csv")


if __name__ == "__main__":
    main()
