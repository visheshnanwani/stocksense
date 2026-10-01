"""
tune_hyperparameters.py
-------------------------
Searches for better hyperparameters for Random Forest, SVR, and XGBoost
using time-series-aware cross-validation (TimeSeriesSplit -- unlike
regular k-fold CV, this never trains on future data to predict the
past, which would be a form of data leakage on time series).

Run this once to find better settings, then manually copy the winning
parameters into the relevant build_xxx() function in models_tabular.py.

Run:
    python tune_hyperparameters.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from data_loader import load_stock_data
from features import build_features, FEATURE_COLUMNS
from pipeline import chronological_split
from stock_search import resolve_ticker

TICKER = None                  # None = ask for a stock name when you run the script (e.g. set "MSFT" to skip the prompt)
START_DATE = "2014-01-01"
END_DATE = "2025-01-01"
N_SPLITS = 5          # number of time-series CV folds
N_ITER = 25            # number of random hyperparameter combinations to try per model


def main():
    global TICKER
    TICKER = resolve_ticker(TICKER)
    print(f"Loading {TICKER} data...")
    raw = load_stock_data(TICKER, START_DATE, END_DATE)
    feat_df = build_features(raw)
    train, val, test = chronological_split(feat_df)

    # Tune using train+val combined (test stays untouched, same as everywhere else in this project)
    tune_df = pd.concat([train, val])
    X = tune_df[FEATURE_COLUMNS].values
    y = tune_df["Target_Return"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # ---- Random Forest ----
    print("\nTuning Random Forest...")
    rf_param_dist = {
        "n_estimators": [200, 300, 500, 700, 1000],
        "max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_param_dist, n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_squared_error", random_state=42, n_jobs=-1,
    )
    rf_search.fit(X_scaled, y)
    print(f"  Best params: {rf_search.best_params_}")
    print(f"  Best CV RMSE: {np.sqrt(-rf_search.best_score_):.5f}")

    # ---- SVR ----
    print("\nTuning SVR...")
    svr_param_dist = {
        "C": [1, 10, 100, 500, 1000, 5000],
        "epsilon": [0.001, 0.005, 0.01, 0.05, 0.1],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    }
    svr_search = RandomizedSearchCV(
        SVR(kernel="rbf"),
        svr_param_dist, n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_squared_error", random_state=42, n_jobs=-1,
    )
    svr_search.fit(X_scaled, y)
    print(f"  Best params: {svr_search.best_params_}")
    print(f"  Best CV RMSE: {np.sqrt(-svr_search.best_score_):.5f}")

    # ---- XGBoost ----
    print("\nTuning XGBoost...")
    xgb_param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [2, 3, 4, 5, 6, 8],
        "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1),
        xgb_param_dist, n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_squared_error", random_state=42, n_jobs=-1,
    )
    xgb_search.fit(X_scaled, y)
    print(f"  Best params: {xgb_search.best_params_}")
    print(f"  Best CV RMSE: {np.sqrt(-xgb_search.best_score_):.5f}")

    print("\n" + "=" * 60)
    print("Copy the winning params above into models_tabular.py's")
    print("build_random_forest(), build_svr(), and build_xgboost()")
    print("default arguments, then re-run main.py / ensemble.py to")
    print("see the accuracy improvement.")
    print("=" * 60)


if __name__ == "__main__":
    main()
