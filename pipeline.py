"""
pipeline.py
-----------
Shared utilities every model uses: chronological train/val/test split
and evaluation metrics. Keeping this identical across models is what
makes the comparison fair.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """
    Split a time-ordered DataFrame into train/val/test WITHOUT shuffling.
    This is critical for time series -- random shuffling leaks future
    information into training and gives falsely optimistic results.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def evaluate_predictions(y_true, y_pred, model_name="Model", today_close=None) -> dict:
    """
    Compute standard forecasting metrics. Returns a dict so results
    can be collected into a comparison table across all models.

    If `today_close` (the last known close before each prediction) is
    given, also reports:
      * Skill vs naive (%) -- % lower RMSE than predicting "tomorrow =
        today". Daily prices barely move, so R² is ~0.97+ even for that
        naive guess; this is the fair test of whether a model adds anything.
      * DirAcc (%)         -- how often the predicted up/down move was
        right (50% = coin flip).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    result = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4),
        "R2": round(r2, 4),
    }
    if today_close is not None:
        today_close = np.asarray(today_close).flatten()
        naive_rmse = np.sqrt(mean_squared_error(y_true, today_close))
        result["Skill vs naive (%)"] = round((1 - rmse / naive_rmse) * 100, 2)
        pred_move, true_move = y_pred - today_close, y_true - today_close
        moving = pred_move != 0
        result["DirAcc (%)"] = round(np.mean(np.sign(pred_move[moving]) == np.sign(true_move[moving])) * 100, 2)             if moving.any() else np.nan
    return result


def results_table(list_of_result_dicts) -> pd.DataFrame:
    """Combine multiple evaluate_predictions() outputs into one table."""
    return pd.DataFrame(list_of_result_dicts).sort_values("RMSE").reset_index(drop=True)


def returns_to_price(predicted_returns, today_close_prices):
    """
    Convert predicted next-day % returns back into predicted price levels,
    so return-based models can be evaluated/backtested on the same
    price scale as price-based models -- this is what makes the final
    comparison table fair across both target modes.

    predicted_price[t] = today_close[t] * (1 + predicted_return[t])
    """
    predicted_returns = np.asarray(predicted_returns).flatten()
    today_close_prices = np.asarray(today_close_prices).flatten()
    return today_close_prices * (1 + predicted_returns)
