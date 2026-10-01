"""
main.py
-------
Runs the full comparison across all 9 models (the 8 below trained TWICE, plus
Exponential Smoothing with a fixed alpha = 0.3):
  (a) predicting next-day CLOSING PRICE directly
  (b) predicting next-day % RETURN (converted back to price for comparison)

Why both: tree-based models (Random Forest, XGBoost) and SVR cannot
extrapolate beyond the price range seen during training. On a stock
like MSFT that trended strongly upward, the test period contains
prices higher than anything seen in training, so "price mode" makes
these models look artificially terrible. "Return mode" fixes this,
since % returns don't have the same extrapolation problem. Comparing
both modes side by side is more rigorous and makes for a stronger
report than picking one.

ARIMA is univariate and forecasts price natively (a return-differenced
model is what ARIMA(p,1,q) already does internally), so it only appears
once, tagged accordingly.

Outputs:
  1. forecasting_metrics_comparison.csv  -- MAE/MSE/RMSE/MAPE/R2 for all
     16 (model x target-mode) combinations + ARIMA, all in PRICE terms
     so they're directly comparable.
  2. profitability_comparison.csv        -- backtest results for all of
     the above vs a Buy-and-Hold baseline.

Edit the CONFIG section below, then run:
    python main.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_stock_data
from features import build_features, FEATURE_COLUMNS, build_stationary_features, STATIONARY_FEATURE_COLUMNS
from pipeline import chronological_split, evaluate_predictions, results_table, returns_to_price
from models_tabular import TABULAR_MODEL_BUILDERS
from model_arima import one_step_arima_forecast
from model_smoothing import ses_one_step_forecast, SMOOTHING_ALPHA
from models_sequential import make_sequences, get_callbacks, SEQUENTIAL_MODEL_BUILDERS
from backtest import simulate_strategy, buy_and_hold_baseline
from stock_search import resolve_ticker

# ----------------------------- CONFIG ---------------------------------
TICKER = None                  # None = ask for a stock name when you run the script (e.g. set "MSFT" to skip the prompt)
START_DATE = "2014-01-01"
END_DATE = "2025-01-01"
SEQ_WINDOW = 30                # lookback window for LSTM/GRU/Transformer
INITIAL_INVESTMENT = 10_000.0
EPOCHS = 100                   # generous budget; EarlyStopping cuts it short automatically
BATCH_SIZE = 32
TARGET_MODES = ["price", "return"]   # run every model in both modes
USE_STATIONARY_FEATURES = True       # True = scale-free features (more accurate on held-out data, see
                                     # README "Accuracy improvements"); False = original raw-price features
# ------------------------------------------------------------------------

TARGET_COL_MAP = {"price": "Target_Close", "return": "Target_Return"}


def run_tabular_models(train, val, test, target_mode):
    """Fit + evaluate Linear Regression, Random Forest, SVR, XGBoost.
    Returns predictions already converted to PRICE terms."""
    target_col = TARGET_COL_MAP[target_mode]

    X_train = train[FEATURE_COLUMNS].values
    y_train = train[target_col].values
    X_val = val[FEATURE_COLUMNS].values
    y_val = val[target_col].values
    X_test = test[FEATURE_COLUMNS].values
    y_test_native = test[target_col].values  # native scale (price or return)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    today_close_test = test["Close"].values  # for return -> price conversion

    metrics_list = []
    predictions_price = {}

    for name, builder in TABULAR_MODEL_BUILDERS.items():
        print(f"  Training {name} [{target_mode}]...")
        model = builder()

        if name == "XGBoost":
            # early stopping needs a validation set
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train_scaled, y_train)

        preds_native = model.predict(X_test_scaled)

        if target_mode == "return":
            preds_price = returns_to_price(preds_native, today_close_test)
        else:
            preds_price = preds_native

        y_test_price = test["Target_Close"].values
        label = f"{name} ({target_mode})"
        metrics_list.append(evaluate_predictions(y_test_price, preds_price, model_name=label,
                                                 today_close=today_close_test))
        predictions_price[label] = preds_price

    return metrics_list, predictions_price


def run_arima_model(train, val, test):
    """Fit + evaluate ARIMA (univariate, Close price only, native price forecast).
    Scored on one-day-ahead predictions like every other model (see
    model_arima.one_step_arima_forecast for why not a one-shot forecast)."""
    print("  Training ARIMA...")
    train_close = train["Close"]
    y_test_price = test["Target_Close"].values

    full_close = pd.concat([train["Close"], val["Close"], test["Close"]])
    preds, order = one_step_arima_forecast(train_close, full_close, test_start_idx=len(train) + len(val))
    print(f"    -> best order found: {order}")

    metrics = evaluate_predictions(y_test_price, preds, model_name="ARIMA (price)", today_close=test["Close"].values)
    return metrics, preds


def run_smoothing_model(train, val, test):
    """Simple Exponential Smoothing with a FIXED alpha (model_smoothing.SMOOTHING_ALPHA = 0.3).
    The smoothed level at day t is the forecast for day t+1's close."""
    print(f"  Running Exponential Smoothing (alpha={SMOOTHING_ALPHA})...")
    full_close = pd.concat([train["Close"], val["Close"], test["Close"]])
    preds = ses_one_step_forecast(full_close).loc[test.index].values
    label = f"Exponential Smoothing (alpha={SMOOTHING_ALPHA})"
    metrics = evaluate_predictions(test["Target_Close"].values, preds, model_name=label,
                                   today_close=test["Close"].values)
    return metrics, preds, label


def run_sequential_models(train, val, test, target_mode):
    """Fit + evaluate LSTM, GRU, Transformer. Returns predictions in PRICE terms."""
    target_col = TARGET_COL_MAP[target_mode]

    full = pd.concat([train, val, test])
    X_all = full[FEATURE_COLUMNS].values
    y_all = full[target_col].values
    close_all = full["Close"].values  # for return -> price conversion

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_all)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

    X_seq, y_seq = make_sequences(X_scaled, y_scaled, window=SEQ_WINDOW)
    today_close_seq = close_all[SEQ_WINDOW:]  # aligned with y_seq

    n_train = len(train) - SEQ_WINDOW
    n_val = len(val)

    X_train_seq, y_train_seq = X_seq[:n_train], y_seq[:n_train]
    X_val_seq, y_val_seq = X_seq[n_train:n_train + n_val], y_seq[n_train:n_train + n_val]
    X_test_seq, y_test_seq = X_seq[n_train + n_val:], y_seq[n_train + n_val:]
    today_close_test_seq = today_close_seq[n_train + n_val:]

    metrics_list = []
    predictions_price = {}

    for name, builder in SEQUENTIAL_MODEL_BUILDERS.items():
        print(f"  Training {name} [{target_mode}]...")
        model = builder(input_shape=(SEQ_WINDOW, X_seq.shape[-1]))
        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
            callbacks=get_callbacks(patience=10),
        )
        preds_scaled = model.predict(X_test_seq, verbose=0).flatten()
        preds_native = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        if target_mode == "return":
            preds_price = returns_to_price(preds_native, today_close_test_seq)
        else:
            preds_price = preds_native

        # actual next-day close price for this same slice, for evaluation
        y_test_actual_native = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
        if target_mode == "return":
            y_test_actual_price = returns_to_price(y_test_actual_native, today_close_test_seq)
        else:
            y_test_actual_price = y_test_actual_native

        label = f"{name} ({target_mode})"
        metrics_list.append(evaluate_predictions(y_test_actual_price, preds_price, model_name=label,
                                                 today_close=today_close_test_seq))
        predictions_price[label] = preds_price

    return metrics_list, predictions_price, today_close_test_seq, len(y_test_seq)


def run_backtests(test_df, all_price_preds, arima_preds, seq_pred_lengths):
    """Run the trading-strategy simulation for every model + baseline."""
    results = []
    actual_close_full = test_df["Close"]

    for label, preds in all_price_preds.items():
        n = len(preds)
        # Sequential model predictions are shorter (lost SEQ_WINDOW rows) --
        # align against the tail of test_df's Close prices.
        actual_close_slice = actual_close_full.iloc[-n:]
        summary, _ = simulate_strategy(
            actual_close_slice, pd.Series(preds), initial_investment=INITIAL_INVESTMENT
        )
        summary["Model"] = label
        results.append(summary)

    summary, _ = simulate_strategy(
        actual_close_full, pd.Series(arima_preds), initial_investment=INITIAL_INVESTMENT
    )
    summary["Model"] = "ARIMA (price)"
    results.append(summary)

    baseline = buy_and_hold_baseline(actual_close_full, initial_investment=INITIAL_INVESTMENT)
    baseline["Model"] = "Buy-and-Hold (Baseline)"
    results.append(baseline)

    df = pd.DataFrame(results)
    cols = ["Model", "Initial_Investment", "Final_Value", "Total_Profit_Loss", "Percent_Return", "Num_Trades"]
    return df[cols].sort_values("Percent_Return", ascending=False).reset_index(drop=True)


def main():
    global TICKER
    TICKER = resolve_ticker(TICKER)
    print(f"Loading data for {TICKER}...")
    raw = load_stock_data(TICKER, START_DATE, END_DATE)
    print(f"Loaded {len(raw)} rows.")

    global FEATURE_COLUMNS
    if USE_STATIONARY_FEATURES:
        FEATURE_COLUMNS = STATIONARY_FEATURE_COLUMNS
        feat_df = build_stationary_features(raw)
    else:
        feat_df = build_features(raw)
    print(f"{len(feat_df)} rows remain after feature engineering "
          f"({'stationary' if USE_STATIONARY_FEATURES else 'original'} feature set, {len(FEATURE_COLUMNS)} features).")

    train, val, test = chronological_split(feat_df)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    all_metrics = []
    all_price_preds = {}

    for mode in TARGET_MODES:
        print(f"\n--- Target mode: {mode.upper()} ---")
        tab_metrics, tab_preds = run_tabular_models(train, val, test, mode)
        all_metrics += tab_metrics
        all_price_preds.update(tab_preds)

        seq_metrics, seq_preds, _, seq_len = run_sequential_models(train, val, test, mode)
        all_metrics += seq_metrics
        all_price_preds.update(seq_preds)

    arima_metrics, arima_preds = run_arima_model(train, val, test)
    all_metrics.append(arima_metrics)

    ses_metrics, ses_preds, ses_label = run_smoothing_model(train, val, test)
    all_metrics.append(ses_metrics)
    all_price_preds[ses_label] = ses_preds

    # The bar every model has to clear: "tomorrow's close = today's close".
    all_metrics.append(evaluate_predictions(test["Target_Close"].values, test["Close"].values,
                                            model_name="Naive baseline (tomorrow = today)",
                                            today_close=test["Close"].values))

    metrics_df = results_table(all_metrics)
    print("\n===== FORECASTING METRICS COMPARISON (all in price terms) =====")
    print(metrics_df.to_string(index=False))
    print("\nRead 'Skill vs naive (%)' first: positive = more accurate than just predicting "
          "tomorrow = today. R² is ~0.97+ even for the naive guess, so it can't tell models apart.")
    metrics_df.to_csv("forecasting_metrics_comparison.csv", index=False)

    profit_df = run_backtests(test, all_price_preds, arima_preds, None)
    print("\n===== TRADING STRATEGY PROFITABILITY COMPARISON =====")
    print(profit_df.to_string(index=False))
    profit_df.to_csv("profitability_comparison.csv", index=False)

    print("\nDone. Results saved to forecasting_metrics_comparison.csv and profitability_comparison.csv")


if __name__ == "__main__":
    main()
