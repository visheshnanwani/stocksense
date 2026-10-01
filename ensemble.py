"""
ensemble.py
-----------
Combines predictions from ALL models into a single blended forecast,
then runs ONE trading strategy backtest on that combined prediction --
instead of treating each model separately.

Why this helps: individual models often make different kinds of errors.
Averaging (or weighted-averaging) their predictions tends to cancel out
some of that noise, which is why ensembling is a very standard technique
in both forecasting and Kaggle-style ML competitions. It also naturally
answers "why do so many models make 0 trades" -- a blended signal is
smoother and more consistently timed than any single volatile model
(like SVR) or any single overly-flat model.

Two weighting schemes are supported:
  - "equal"        : simple average across all included models.
  - "inverse_rmse"  : models that were more accurate on the VALIDATION
                      set get proportionally more weight in the blend
                      (a model with half the validation RMSE of another
                      gets roughly twice the vote).

By default this includes: Linear Regression, Random Forest, SVR,
XGBoost (all in "return" mode, which avoids the extrapolation problem
described in README.md) plus ARIMA (price-native). LSTM/GRU/Transformer
can optionally be included too (INCLUDE_SEQUENTIAL_MODELS = True) but
this takes much longer to run since those need real training time.

Run:
    python ensemble.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from data_loader import load_stock_data
from features import build_features, FEATURE_COLUMNS
from pipeline import chronological_split, evaluate_predictions, returns_to_price
from models_tabular import TABULAR_MODEL_BUILDERS
from model_arima import run_arima_forecast
from models_sequential import make_sequences, get_callbacks, SEQUENTIAL_MODEL_BUILDERS
from backtest import simulate_strategy, buy_and_hold_baseline
from stock_search import resolve_ticker

# ----------------------------- CONFIG ---------------------------------
TICKER = None                  # None = ask for a stock name when you run the script (e.g. set "MSFT" to skip the prompt)
START_DATE = "2014-01-01"
END_DATE = "2025-01-01"
INITIAL_INVESTMENT = 10_000.0
BUY_THRESHOLD = 0.01
SELL_THRESHOLD = 0.01

WEIGHTING = "inverse_rmse"          # "equal" or "inverse_rmse"
INCLUDE_SEQUENTIAL_MODELS = False   # set True to also include LSTM/GRU/Transformer (slower)
SEQ_WINDOW = 30
EPOCHS = 100
BATCH_SIZE = 32
# ------------------------------------------------------------------------


def train_tabular_return_models(train, val, test):
    """Train all 4 tabular models in return mode. Returns price-space
    predictions for both val and test, plus each model's val RMSE."""
    X_train = train[FEATURE_COLUMNS].values
    y_train = train["Target_Return"].values
    X_val = val[FEATURE_COLUMNS].values
    y_val = val["Target_Return"].values
    X_test = test[FEATURE_COLUMNS].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    today_close_val = val["Close"].values
    today_close_test = test["Close"].values
    y_val_price = val["Target_Close"].values

    model_val_preds = {}
    model_test_preds = {}
    model_val_rmse = {}

    for name, builder in TABULAR_MODEL_BUILDERS.items():
        print(f"  Training {name} (return)...")
        model = builder()
        if name == "XGBoost":
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        else:
            model.fit(X_train_scaled, y_train)

        val_preds_native = model.predict(X_val_scaled)
        test_preds_native = model.predict(X_test_scaled)

        val_preds_price = returns_to_price(val_preds_native, today_close_val)
        test_preds_price = returns_to_price(test_preds_native, today_close_test)

        rmse = np.sqrt(mean_squared_error(y_val_price, val_preds_price))

        label = f"{name} (return)"
        model_val_preds[label] = val_preds_price
        model_test_preds[label] = test_preds_price
        model_val_rmse[label] = rmse

    return model_val_preds, model_test_preds, model_val_rmse


def train_arima(train, val, test):
    """ARIMA is price-native; forecast covering val+test length, split accordingly."""
    print("  Training ARIMA...")
    train_close = train["Close"]
    n_val, n_test = len(val), len(test)
    preds, order = run_arima_forecast(train_close, n_test_steps=n_val + n_test)
    print(f"    -> best order: {order}")
    val_preds_price = preds[:n_val]
    test_preds_price = preds[n_val:]

    y_val_price = val["Target_Close"].values
    rmse = np.sqrt(mean_squared_error(y_val_price, val_preds_price))

    label = "ARIMA (price)"
    return {label: val_preds_price}, {label: test_preds_price}, {label: rmse}


def train_sequential_return_models(train, val, test):
    """Optional: LSTM, GRU, Transformer in return mode. Predictions are
    shorter (lose SEQ_WINDOW rows) so they get aligned/truncated when combined."""
    full = pd.concat([train, val, test])
    X_all = full[FEATURE_COLUMNS].values
    y_all = full["Target_Return"].values
    close_all = full["Close"].values

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_all)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

    X_seq, y_seq = make_sequences(X_scaled, y_scaled, window=SEQ_WINDOW)
    today_close_seq = close_all[SEQ_WINDOW:]

    n_train = len(train) - SEQ_WINDOW
    n_val = len(val)

    X_train_seq, y_train_seq = X_seq[:n_train], y_seq[:n_train]
    X_val_seq = X_seq[n_train:n_train + n_val]
    X_test_seq = X_seq[n_train + n_val:]
    today_close_val_seq = today_close_seq[n_train:n_train + n_val]
    today_close_test_seq = today_close_seq[n_train + n_val:]
    y_val_price_seq = returns_to_price(y_seq[n_train:n_train + n_val], today_close_val_seq)

    model_val_preds, model_test_preds, model_val_rmse = {}, {}, {}

    for name, builder in SEQUENTIAL_MODEL_BUILDERS.items():
        print(f"  Training {name} (return)...")
        model = builder(input_shape=(SEQ_WINDOW, X_seq.shape[-1]))
        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_seq[n_train:n_train + n_val]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
            callbacks=get_callbacks(patience=10),
        )
        val_preds_native = scaler_y.inverse_transform(
            model.predict(X_val_seq, verbose=0).reshape(-1, 1)
        ).flatten()
        test_preds_native = scaler_y.inverse_transform(
            model.predict(X_test_seq, verbose=0).reshape(-1, 1)
        ).flatten()

        val_preds_price = returns_to_price(val_preds_native, today_close_val_seq)
        test_preds_price = returns_to_price(test_preds_native, today_close_test_seq)

        rmse = np.sqrt(mean_squared_error(y_val_price_seq, val_preds_price))

        label = f"{name} (return)"
        model_val_preds[label] = val_preds_price
        model_test_preds[label] = test_preds_price
        model_val_rmse[label] = rmse

    return model_val_preds, model_test_preds, model_val_rmse


def build_ensemble_prediction(model_test_preds, model_val_rmse, weighting="inverse_rmse"):
    """
    Combine multiple models' test-period predictions into one series.
    Models with shorter prediction arrays (sequential models) force
    everyone else to be truncated to that same shorter, most-recent
    length so they can be averaged together fairly.
    """
    min_len = min(len(p) for p in model_test_preds.values())
    aligned = {name: preds[-min_len:] for name, preds in model_test_preds.items()}

    if weighting == "equal":
        weights = {name: 1.0 for name in aligned}
    elif weighting == "inverse_rmse":
        weights = {name: 1.0 / max(model_val_rmse.get(name, 1.0), 1e-6) for name in aligned}
    else:
        raise ValueError("weighting must be 'equal' or 'inverse_rmse'")

    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}

    print("\nEnsemble weights:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.3f}")

    ensemble_preds = np.zeros(min_len)
    for name, preds in aligned.items():
        ensemble_preds += weights[name] * preds

    return ensemble_preds, min_len


def main():
    global TICKER
    TICKER = resolve_ticker(TICKER)
    print(f"Loading data for {TICKER}...")
    raw = load_stock_data(TICKER, START_DATE, END_DATE)
    feat_df = build_features(raw)
    train, val, test = chronological_split(feat_df)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    all_val_preds, all_test_preds, all_val_rmse = {}, {}, {}

    print("\n--- Tabular models ---")
    v, t, r = train_tabular_return_models(train, val, test)
    all_val_preds.update(v); all_test_preds.update(t); all_val_rmse.update(r)

    print("\n--- ARIMA ---")
    v, t, r = train_arima(train, val, test)
    all_val_preds.update(v); all_test_preds.update(t); all_val_rmse.update(r)

    if INCLUDE_SEQUENTIAL_MODELS:
        print("\n--- Sequential models ---")
        v, t, r = train_sequential_return_models(train, val, test)
        all_val_preds.update(v); all_test_preds.update(t); all_val_rmse.update(r)

    print("\nValidation RMSE per model:")
    for name, rmse in sorted(all_val_rmse.items(), key=lambda x: x[1]):
        print(f"  {name}: {rmse:.4f}")

    ensemble_preds, min_len = build_ensemble_prediction(all_test_preds, all_val_rmse, WEIGHTING)

    actual_close_aligned = test["Close"].iloc[-min_len:]
    actual_target_aligned = test["Target_Close"].iloc[-min_len:].values

    metrics = evaluate_predictions(actual_target_aligned, ensemble_preds, model_name="Ensemble")
    print("\n===== ENSEMBLE FORECASTING METRICS =====")
    print(pd.DataFrame([metrics]).to_string(index=False))

    summary, daily_log = simulate_strategy(
        actual_close_aligned, pd.Series(ensemble_preds),
        initial_investment=INITIAL_INVESTMENT,
        buy_threshold=BUY_THRESHOLD, sell_threshold=SELL_THRESHOLD,
    )
    baseline = buy_and_hold_baseline(actual_close_aligned, initial_investment=INITIAL_INVESTMENT)

    print("\n===== ENSEMBLE TRADING STRATEGY RESULTS =====")
    print(pd.DataFrame([
        {**summary, "Model": "Ensemble"},
        {**baseline, "Model": "Buy-and-Hold (Baseline)"},
    ])[["Model", "Initial_Investment", "Final_Value", "Total_Profit_Loss", "Percent_Return", "Num_Trades"]]
        .to_string(index=False))

    pd.DataFrame([metrics]).to_csv("ensemble_forecasting_metrics.csv", index=False)
    daily_log.to_csv("ensemble_daily_log.csv", index=False)
    print("\nSaved ensemble_forecasting_metrics.csv and ensemble_daily_log.csv")


if __name__ == "__main__":
    main()
