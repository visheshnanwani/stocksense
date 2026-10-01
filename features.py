"""
features.py
------------
Builds technical-indicator features from raw OHLCV data and creates
the prediction target (next-day closing price / return).

Usage:
    feat_df = build_features(raw_df)
"""

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def build_features(df: pd.DataFrame, target_horizon: int = 1, keep_latest: bool = False) -> pd.DataFrame:
    """
    Given raw OHLCV data, engineer technical indicator features and a
    next-period target column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume
    target_horizon : int
        How many days ahead to predict (1 = next day).
    keep_latest : bool
        False (default, for TRAINING): drop every row with a NaN,
        including the newest day(s), whose target isn't known yet.
        True (for PREDICTING): keep the newest day(s) with NaN targets.
        Without this, "predict tomorrow" would silently use YESTERDAY's
        features, since the newest row is always dropped for having no
        target.

    Returns
    -------
    pd.DataFrame
        Feature matrix + target column 'Target_Close', with NaN rows dropped.
    """
    data = df.copy()

    close = data["Close"]

    # --- Lagged returns ---
    for lag in [1, 2, 3, 5, 10, 30]:
        data[f"Return_{lag}d"] = close.pct_change(lag)

    # --- Moving averages ---
    for window in [5, 10, 20, 50]:
        data[f"SMA_{window}"] = close.rolling(window).mean()
        data[f"EMA_{window}"] = close.ewm(span=window, adjust=False).mean()

    # --- Volatility ---
    data["Volatility_10d"] = close.pct_change().rolling(10).std()
    data["Volatility_30d"] = close.pct_change().rolling(30).std()

    # --- RSI ---
    data["RSI_14"] = _rsi(close, 14)

    # --- MACD ---
    macd_line, signal_line = _macd(close)
    data["MACD"] = macd_line
    data["MACD_Signal"] = signal_line

    # --- Bollinger Bands ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data["BB_Upper"] = sma20 + 2 * std20
    data["BB_Lower"] = sma20 - 2 * std20
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / sma20

    # --- Volume features ---
    # Some markets have zero-volume sessions (e.g. NSE special/holiday
    # sessions) and indices report volume as 0 -- a raw pct_change would
    # divide by zero and produce inf, which crashes every sklearn model.
    data["Volume_Change"] = data["Volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["Volume_SMA_10"] = data["Volume"].rolling(10).mean()

    # --- Target: next `target_horizon`-day closing price ---
    data["Target_Close"] = close.shift(-target_horizon)
    # Also keep target as a % return, useful for the trading strategy later
    data["Target_Return"] = data["Target_Close"] / close - 1

    # --- Target: next `target_horizon`-day OPENING price ---
    # Expressed as a % move relative to TODAY'S close (the overnight gap),
    # the same convention as Target_Return, so both can share the same
    # ensemble machinery and be converted back to price the same way.
    data["Target_Open"] = data["Open"].shift(-target_horizon)
    data["Target_Open_Return"] = data["Target_Open"] / close - 1

    data = data.replace([np.inf, -np.inf], np.nan)
    if keep_latest:
        return data.dropna(subset=FEATURE_COLUMNS)
    return data.dropna()


def build_stationary_features(df: pd.DataFrame, target_horizon: int = 1, keep_latest: bool = False) -> pd.DataFrame:
    """
    Improved feature set, used by the dashboard ensemble. Every feature is
    SCALE-FREE (a ratio, % distance, or bounded oscillator) instead of a
    raw price level like SMA_50 = 412.3.

    Why this matters: raw price levels keep drifting to values the model
    has never seen (a stock that trended from $40 to $400), so tree
    models (Random Forest, XGBoost) can't extrapolate and linear models
    get badly conditioned. "Price is 3% above its 50-day average" means
    the same thing at $40 or $400. In a benchmark on 5 stocks
    (MSFT, AAPL, JPM, RELIANCE.NS, TCS.NS), switching to these features
    lowered test error on 4 of the 5.

    Returns build_features()'s columns (targets, OHLCV, ...) plus
    STATIONARY_FEATURE_COLUMNS. Needs ~1 year of warm-up rows (52-week
    high/low and 200-day average).
    """
    base = build_features(df, target_horizon, keep_latest=True)
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)

    for lag in [1, 2, 3, 5, 10, 20, 60]:
        f[f"S_Ret_{lag}d"] = c.pct_change(lag)
    for w in [5, 10, 20, 50, 200]:
        f[f"S_Dist_SMA_{w}"] = c / c.rolling(w).mean() - 1
    f["S_SMA5_vs_20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1
    f["S_SMA20_vs_50"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    f["S_SMA50_vs_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    for w in [5, 10, 20, 60]:
        f[f"S_Vol_{w}d"] = r.rolling(w).std()
    f["S_Vol_Ratio"] = f["S_Vol_5d"] / f["S_Vol_60d"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["S_ATR_Pct"] = tr.rolling(14).mean() / c
    f["S_RSI_14"] = _rsi(c, 14) / 100
    f["S_RSI_5"] = _rsi(c, 5) / 100
    macd_line, signal_line = _macd(c)
    f["S_MACD_Pct"] = macd_line / c
    f["S_MACD_Hist_Pct"] = (macd_line - signal_line) / c
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    f["S_BB_PctB"] = (c - (sma20 - 2 * std20)) / (4 * std20)
    f["S_BB_Width"] = 4 * std20 / sma20
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["S_Stoch_K"] = (c - lo14) / (hi14 - lo14)
    f["S_Dist_52w_High"] = c / h.rolling(252, min_periods=120).max() - 1
    f["S_Dist_52w_Low"] = c / l.rolling(252, min_periods=120).min() - 1
    vv = v.replace(0, np.nan)  # zero-volume sessions / indices
    f["S_Rel_Volume"] = (vv / vv.rolling(20).mean()).fillna(1.0)
    f["S_OBV_Slope"] = ((np.sign(r) * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)).fillna(0.0)
    rng = (h - l).replace(0, np.nan)
    f["S_Body_Pct"] = ((c - o) / rng).fillna(0.0)
    f["S_Upper_Wick_Pct"] = ((h - np.maximum(o, c)) / rng).fillna(0.0)
    f["S_Lower_Wick_Pct"] = ((np.minimum(o, c) - l) / rng).fillna(0.0)
    f["S_Gap"] = o / c.shift() - 1
    f["S_Intraday_Ret"] = c / o - 1
    f["S_HL_Range"] = (h - l) / c
    f["S_Day_Of_Week"] = df.index.dayofweek / 4

    f = f.replace([np.inf, -np.inf], np.nan)
    data = base.join(f, how="inner")
    data = data.dropna(subset=STATIONARY_FEATURE_COLUMNS)
    if not keep_latest:
        data = data.dropna()
    return data


STATIONARY_FEATURE_COLUMNS = (
    [f"S_Ret_{lag}d" for lag in [1, 2, 3, 5, 10, 20, 60]]
    + [f"S_Dist_SMA_{w}" for w in [5, 10, 20, 50, 200]]
    + ["S_SMA5_vs_20", "S_SMA20_vs_50", "S_SMA50_vs_200"]
    + [f"S_Vol_{w}d" for w in [5, 10, 20, 60]]
    + ["S_Vol_Ratio", "S_ATR_Pct", "S_RSI_14", "S_RSI_5", "S_MACD_Pct", "S_MACD_Hist_Pct",
       "S_BB_PctB", "S_BB_Width", "S_Stoch_K", "S_Dist_52w_High", "S_Dist_52w_Low",
       "S_Rel_Volume", "S_OBV_Slope", "S_Body_Pct", "S_Upper_Wick_Pct", "S_Lower_Wick_Pct",
       "S_Gap", "S_Intraday_Ret", "S_HL_Range", "S_Day_Of_Week"]
)


FEATURE_COLUMNS = [
    "Return_1d", "Return_2d", "Return_3d", "Return_5d", "Return_10d", "Return_30d",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_5", "EMA_10", "EMA_20", "EMA_50",
    "Volatility_10d", "Volatility_30d",
    "RSI_14", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "BB_Width",
    "Volume_Change", "Volume_SMA_10",
]

TARGET_COLUMN = "Target_Close"


if __name__ == "__main__":
    raw = pd.read_csv("raw_stock_data.csv", index_col="Date", parse_dates=True)
    feat = build_features(raw)
    print(feat[FEATURE_COLUMNS + [TARGET_COLUMN]].head())
    print(f"\nFeature rows after cleaning: {len(feat)}")
    feat.to_csv("features.csv")
