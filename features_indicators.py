"""
features_indicators.py
------------------------
A wide, standard technical-indicator bank, written from the source formulas so
the project has no extra dependency.

Everything here is **stationary by construction**: an indicator is only useful to
a model if its value means the same thing at Rs.100 and at Rs.100,000, so prices
enter as ratios, z-scores or percentile ranks, never as levels. Every column uses
data up to and including day t only, so there is no look-ahead.

Families included
  trend        moving-average stacks, ADX/DMI, Aroon, Ichimoku, Supertrend-style
               ATR channel position, linear-regression slope and R^2
  momentum     RSI (Wilder), Stochastic %K/%D, Williams %R, CCI, ROC, TSI, KAMA
               efficiency ratio, MACD histogram, PPO
  volatility   ATR%, Bollinger width and %B, Keltner and Donchian position,
               Ulcer index, Parkinson and Garman-Klass estimators, vol-of-vol,
               realised-vol ratios (short vs long)
  volume       OBV slope, Chaikin money flow, money-flow index, volume z-score,
               Amihud illiquidity, VWAP distance
  structure    distance to 52-week high/low, drawdown, Hurst exponent, sample
               entropy of returns, run length, gap statistics
  seasonality  day-of-week, month, turn-of-month, days to quarter end

Usage:
    from features_indicators import indicator_features
    feats = indicator_features(raw_ohlcv)          # DataFrame aligned to raw
"""

import numpy as np
import pandas as pd

EPS = 1e-12


# ------------------------------------------------------------------ helpers
def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing -- what RSI/ADX/ATR are actually defined with."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def _true_range(high, low, close):
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / (s + EPS)


def _pct_rank(series: pd.Series, window: int) -> pd.Series:
    """Where today sits in its own recent distribution, 0..1."""
    return series.rolling(window).apply(lambda w: (w[-1] > w[:-1]).mean() if len(w) > 1 else np.nan, raw=True)


def _slope_r2(series: pd.Series, window: int):
    """Linear-regression slope (per day, in % of price) and its R^2."""
    x = np.arange(window)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()

    def _slope(w):
        y = w - w.mean()
        return float((x_c * y).sum() / denom)

    def _r2(w):
        y = w - w.mean()
        b = (x_c * y).sum() / denom
        ss_res = float(((y - b * x_c) ** 2).sum())
        ss_tot = float((y ** 2).sum())
        return 1 - ss_res / (ss_tot + EPS)

    log_p = np.log(series.clip(lower=EPS))
    return (log_p.rolling(window).apply(_slope, raw=True) * 100,
            log_p.rolling(window).apply(_r2, raw=True))


def _hurst(series: pd.Series, window: int = 120) -> pd.Series:
    """Rescaled-range Hurst exponent: >0.5 trending, <0.5 mean-reverting."""
    def _h(w):
        r = np.diff(np.log(np.maximum(w, EPS)))
        if len(r) < 20 or np.std(r) < EPS:
            return np.nan
        lags = [2, 4, 8, 16, 32]
        taus = []
        for lag in lags:
            if lag >= len(r):
                return np.nan
            taus.append(np.sqrt(np.mean((r[lag:] - r[:-lag]) ** 2)))
        taus = np.array(taus)
        if np.any(taus <= 0):
            return np.nan
        return float(np.polyfit(np.log(lags), np.log(taus), 1)[0])
    return series.rolling(window).apply(_h, raw=True)


def _entropy(series: pd.Series, window: int = 60, bins: int = 8) -> pd.Series:
    """Shannon entropy of recent returns: high = noisy, low = one-directional."""
    def _e(w):
        hist, _ = np.histogram(w, bins=bins)
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(bins))
    return series.rolling(window).apply(_e, raw=True)


# ------------------------------------------------------------------ the bank
def indicator_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Every indicator, as stationary columns aligned to `raw`."""
    o, h, l, c = raw["Open"], raw["High"], raw["Low"], raw["Close"]
    v = raw["Volume"].replace(0, np.nan)
    ret = c.pct_change()
    log_ret = np.log(c.clip(lower=EPS)).diff()
    f = pd.DataFrame(index=raw.index)

    # ---------------- trend ----------------
    for n in (10, 20, 50, 100, 200):
        f[f"Trend_Px_vs_SMA{n}"] = c / c.rolling(n).mean() - 1
    f["Trend_SMA20_vs_SMA50"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    f["Trend_SMA50_vs_SMA200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    f["Trend_EMA12_vs_EMA26"] = c.ewm(span=12).mean() / c.ewm(span=26).mean() - 1
    for n in (20, 60):
        slope, r2 = _slope_r2(c, n)
        f[f"Trend_Slope{n}"], f[f"Trend_SlopeR2_{n}"] = slope, r2

    tr = _true_range(h, l, c)
    atr14 = _wilder(tr, 14)
    up_move, down_move = h.diff(), -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * _wilder(pd.Series(plus_dm, index=raw.index), 14) / (atr14 + EPS)
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=raw.index), 14) / (atr14 + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    f["Trend_ADX14"] = _wilder(dx, 14) / 100
    f["Trend_DI_Diff"] = (plus_di - minus_di) / 100

    n = 25
    f["Trend_Aroon"] = ((h.rolling(n + 1).apply(np.argmax, raw=True)
                         - l.rolling(n + 1).apply(np.argmin, raw=True)) / n)

    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    f["Trend_Ichimoku_ConvBase"] = conv / (base + EPS) - 1
    f["Trend_Ichimoku_PxCloud"] = c / (((conv + base) / 2).shift(26) + EPS) - 1

    f["Trend_ATR_Channel_Pos"] = (c - c.rolling(20).mean()) / (2 * atr14 + EPS)

    # ---------------- momentum ----------------
    delta = c.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    for n in (7, 14, 28):
        rs = _wilder(gain, n) / (_wilder(loss, n) + EPS)
        f[f"Mom_RSI{n}"] = (100 - 100 / (1 + rs)) / 100 - 0.5
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = (c - low14) / (high14 - low14 + EPS)
    f["Mom_Stoch_K"] = stoch_k - 0.5
    f["Mom_Stoch_D"] = stoch_k.rolling(3).mean() - 0.5
    f["Mom_Williams_R"] = (high14 - c) / (high14 - low14 + EPS) - 0.5
    tp = (h + l + c) / 3
    f["Mom_CCI20"] = ((tp - tp.rolling(20).mean())
                      / (0.015 * tp.rolling(20).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True) + EPS)) / 100
    for n in (5, 10, 21, 63, 126, 252):
        f[f"Mom_ROC{n}"] = c.pct_change(n)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    f["Mom_MACD_Hist"] = (macd - macd.ewm(span=9).mean()) / (c + EPS)
    f["Mom_PPO"] = macd / (c.ewm(span=26).mean() + EPS)
    mom = c.diff(25)
    f["Mom_TSI"] = (mom.ewm(span=13).mean().ewm(span=13).mean()
                    / (mom.abs().ewm(span=13).mean().ewm(span=13).mean() + EPS))
    change, volatility = (c - c.shift(10)).abs(), c.diff().abs().rolling(10).sum()
    f["Mom_Efficiency"] = change / (volatility + EPS)          # Kaufman efficiency ratio
    f["Mom_12_1"] = c.pct_change(252) - c.pct_change(21)       # classic momentum, last month skipped

    # ---------------- volatility ----------------
    f["Vol_ATR_Pct"] = atr14 / (c + EPS)
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["Vol_BB_Width"] = (4 * sd20) / (sma20 + EPS)
    f["Vol_BB_PctB"] = (c - (sma20 - 2 * sd20)) / (4 * sd20 + EPS) - 0.5
    kc_mid = c.ewm(span=20).mean()
    f["Vol_Keltner_Pos"] = (c - kc_mid) / (2 * atr14 + EPS)
    dc_hi, dc_lo = h.rolling(20).max(), l.rolling(20).min()
    f["Vol_Donchian_Pos"] = (c - dc_lo) / (dc_hi - dc_lo + EPS) - 0.5
    for n in (10, 21, 63):
        f[f"Vol_Realised{n}"] = log_ret.rolling(n).std() * np.sqrt(252)
    f["Vol_Ratio_Short_Long"] = f["Vol_Realised10"] / (f["Vol_Realised63"] + EPS) - 1
    f["Vol_Of_Vol"] = f["Vol_Realised21"].rolling(63).std()
    f["Vol_Parkinson"] = (np.log(h / (l + EPS)) ** 2).rolling(21).mean() ** 0.5 * np.sqrt(252 / (4 * np.log(2)))
    gk = 0.5 * np.log(h / (l + EPS)) ** 2 - (2 * np.log(2) - 1) * np.log(c / (o + EPS)) ** 2
    f["Vol_GarmanKlass"] = gk.rolling(21).mean().clip(lower=0) ** 0.5 * np.sqrt(252)
    dd = c / c.cummax() - 1
    f["Vol_Ulcer"] = (dd.rolling(63).apply(lambda w: np.sqrt(np.mean(w ** 2)), raw=True))

    # ---------------- volume ----------------
    obv = (np.sign(delta.fillna(0)) * v.fillna(0)).cumsum()
    f["Volm_OBV_Slope20"] = obv.diff(20) / (v.rolling(20).mean() * 20 + EPS)
    mfm = ((c - l) - (h - c)) / (h - l + EPS)
    f["Volm_CMF20"] = (mfm * v).rolling(20).sum() / (v.rolling(20).sum() + EPS)
    pos_flow = (tp * v).where(tp > tp.shift(1), 0.0)
    neg_flow = (tp * v).where(tp < tp.shift(1), 0.0)
    f["Volm_MFI14"] = (pos_flow.rolling(14).sum()
                       / (pos_flow.rolling(14).sum() + neg_flow.rolling(14).sum() + EPS)) - 0.5
    f["Volm_Z20"] = _zscore(np.log(v.fillna(method="ffill") + 1), 20)
    f["Volm_Amihud"] = (ret.abs() / (v * c + EPS)).rolling(21).mean() * 1e9
    vwap = (tp * v).rolling(21).sum() / (v.rolling(21).sum() + EPS)
    f["Volm_VWAP_Dist"] = c / (vwap + EPS) - 1

    # ---------------- structure ----------------
    f["Struct_From_52w_High"] = c / (c.rolling(252).max() + EPS) - 1
    f["Struct_From_52w_Low"] = c / (c.rolling(252).min() + EPS) - 1
    f["Struct_Drawdown"] = dd
    f["Struct_Hurst"] = _hurst(c, 120)
    f["Struct_Entropy"] = _entropy(ret, 60)
    f["Struct_Up_Days_20"] = (ret > 0).rolling(20).mean() - 0.5
    f["Struct_Gap"] = o / (c.shift(1) + EPS) - 1
    f["Struct_Close_Location"] = (c - l) / (h - l + EPS) - 0.5
    f["Struct_Ret_Skew63"] = ret.rolling(63).skew()
    f["Struct_Ret_Kurt63"] = ret.rolling(63).kurt()
    f["Struct_Px_Rank252"] = _pct_rank(c, 252) - 0.5

    # ---------------- seasonality ----------------
    idx = raw.index
    f["Seas_DayOfWeek"] = idx.dayofweek / 4 - 0.5
    f["Seas_Month_Sin"] = np.sin(2 * np.pi * idx.month / 12)
    f["Seas_Month_Cos"] = np.cos(2 * np.pi * idx.month / 12)
    f["Seas_Turn_Of_Month"] = ((idx.day >= 28) | (idx.day <= 3)).astype(float)
    f["Seas_Days_To_QEnd"] = (idx.to_period("Q").end_time - idx).days / 92

    return f.replace([np.inf, -np.inf], np.nan)


INDICATOR_COLUMNS = None      # filled on first use, for the engine's group map


def indicator_columns(raw: pd.DataFrame = None):
    global INDICATOR_COLUMNS
    if INDICATOR_COLUMNS is None and raw is not None:
        INDICATOR_COLUMNS = list(indicator_features(raw).columns)
    return INDICATOR_COLUMNS
