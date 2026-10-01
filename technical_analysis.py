"""
technical_analysis.py
-----------------------
Classical chart-reading tools that complement the ML predictions:

  1. support_resistance_levels() -- finds horizontal price levels where
     the stock has repeatedly turned (swing highs/lows clustered
     together), i.e. the lines a trader would draw on a chart.
  2. technical_scorecard()       -- a TradingView-style summary that
     tallies ~10 classical indicators (moving averages, RSI, MACD,
     Bollinger Bands, Stochastic, golden/death cross, candlestick
     reading) into one Strong Buy / Buy / Neutral / Sell / Strong Sell
     rating, with every individual vote shown so it's explainable.
  3. add_overlay_indicators()    -- extra indicator columns used by the
     dashboard charts (SMA 200, Stochastic, ATR, Bollinger mid).

Like the candlestick module, these are rule-of-thumb heuristics shown
as CONTEXT next to the model predictions -- not signals on their own.
"""

import numpy as np
import pandas as pd

from features import _rsi, _macd


def add_overlay_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    for w in (5, 20, 50, 200):
        d[f"SMA_{w}"] = c.rolling(w).mean()
    std20 = c.rolling(20).std()
    d["BB_Mid"] = d["SMA_20"]
    d["BB_Upper"] = d["SMA_20"] + 2 * std20
    d["BB_Lower"] = d["SMA_20"] - 2 * std20
    d["RSI_14"] = _rsi(c, 14)
    d["MACD"], d["MACD_Signal"] = _macd(c)
    low14, high14 = d["Low"].rolling(14).min(), d["High"].rolling(14).max()
    d["Stoch_K"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()
    return d


def support_resistance_levels(df: pd.DataFrame, lookback: int = 250, swing_window: int = 5,
                              tolerance: float = 0.015, max_levels: int = 6):
    """
    Returns a list of dicts {level, touches, kind} sorted by price, where
    kind is "support" (below current price) or "resistance" (above).

    Method: a swing high/low is a bar whose high/low is the extreme of
    the surrounding `swing_window` bars on each side. Swing points
    within `tolerance` (1.5%) of each other are merged into one level;
    levels touched more often are considered stronger.
    """
    d = df.tail(lookback)
    if len(d) < 2 * swing_window + 1:
        return []
    win = 2 * swing_window + 1
    highs = d["High"][d["High"] == d["High"].rolling(win, center=True).max()]
    lows = d["Low"][d["Low"] == d["Low"].rolling(win, center=True).min()]
    points = sorted(list(highs.values) + list(lows.values))

    clusters = []
    for p in points:
        if clusters and abs(p - np.mean(clusters[-1])) / np.mean(clusters[-1]) <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    price = df["Close"].iloc[-1]
    levels = [{"level": float(np.mean(c)), "touches": len(c)} for c in clusters]
    # Keep the strongest levels, but always keep the nearest one on each side.
    levels.sort(key=lambda x: (-x["touches"], abs(x["level"] - price)))
    chosen = levels[:max_levels]
    below = [l for l in levels if l["level"] < price]
    above = [l for l in levels if l["level"] >= price]
    for nearest in (max(below, key=lambda x: x["level"]) if below else None,
                    min(above, key=lambda x: x["level"]) if above else None):
        if nearest and nearest not in chosen:
            chosen.append(nearest)
    for l in chosen:
        l["kind"] = "support" if l["level"] < price else "resistance"
    return sorted(chosen, key=lambda x: x["level"])


def technical_scorecard(df: pd.DataFrame, candle_reading: dict = None):
    """
    Tally classical indicator votes into one rating.

    Returns (rating, score, votes_df) where score is in [-1, 1] and
    votes_df lists each indicator, its value, and its Buy/Sell/Neutral vote.
    """
    d = add_overlay_indicators(df)
    last = d.iloc[-1]
    price = last["Close"]
    votes = []

    def vote(name, value, v, why):
        votes.append({"Indicator": name, "Value": value, "Vote": {1: "Buy", -1: "Sell", 0: "Neutral"}[v],
                      "_v": v, "Why": why})

    for w in (20, 50, 200):
        sma = last[f"SMA_{w}"]
        if pd.notna(sma):
            v = 1 if price > sma else -1
            vote(f"Price vs SMA {w}", f"{sma:,.2f}", v, f"Price is {'above' if v > 0 else 'below'} the {w}-day average")

    if pd.notna(last["SMA_50"]) and pd.notna(last["SMA_200"]):
        v = 1 if last["SMA_50"] > last["SMA_200"] else -1
        vote("SMA 50 vs SMA 200", "Golden cross" if v > 0 else "Death cross", v,
             "Long-term trend is up" if v > 0 else "Long-term trend is down")

    rsi = last["RSI_14"]
    if pd.notna(rsi):
        v = 1 if rsi < 30 else (-1 if rsi > 70 else 0)
        vote("RSI (14)", f"{rsi:.1f}", v, "Oversold (<30)" if v > 0 else ("Overbought (>70)" if v < 0 else "Between 30 and 70"))

    if pd.notna(last["MACD"]):
        v = 1 if last["MACD"] > last["MACD_Signal"] else -1
        vote("MACD vs signal", f"{last['MACD'] - last['MACD_Signal']:+.2f}", v,
             "Bullish momentum" if v > 0 else "Bearish momentum")

    if pd.notna(last["BB_Upper"]):
        pct_b = (price - last["BB_Lower"]) / (last["BB_Upper"] - last["BB_Lower"])
        v = 1 if pct_b < 0.05 else (-1 if pct_b > 0.95 else 0)
        vote("Bollinger %B", f"{pct_b:.2f}", v,
             "At/below lower band" if v > 0 else ("At/above upper band" if v < 0 else "Inside the bands"))

    k = last["Stoch_K"]
    if pd.notna(k):
        v = 1 if k < 20 else (-1 if k > 80 else 0)
        vote("Stochastic %K", f"{k:.1f}", v, "Oversold (<20)" if v > 0 else ("Overbought (>80)" if v < 0 else "Between 20 and 80"))

    if len(d) > 21:
        roc = price / d["Close"].iloc[-21] - 1
        v = 1 if roc > 0.02 else (-1 if roc < -0.02 else 0)
        vote("1-month momentum", f"{roc*100:+.1f}%", v, "Rising" if v > 0 else ("Falling" if v < 0 else "Flat"))

    if candle_reading is not None:
        v = 1 if candle_reading["verdict"] == "Bullish" else (-1 if candle_reading["verdict"] == "Bearish" else 0)
        vote("Candlestick reading", candle_reading["verdict"], v, f"Patterns over the last {candle_reading['lookback']} candles")

    votes_df = pd.DataFrame(votes)
    score = votes_df["_v"].mean() if not votes_df.empty else 0.0
    if score >= 0.5:
        rating = "Strong Buy"
    elif score >= 0.15:
        rating = "Buy"
    elif score <= -0.5:
        rating = "Strong Sell"
    elif score <= -0.15:
        rating = "Sell"
    else:
        rating = "Neutral"
    return rating, float(score), votes_df.drop(columns="_v") if not votes_df.empty else votes_df


def technical_score_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    The same indicator votes as technical_scorecard(), computed for EVERY
    day (vectorized, no look-ahead), so the rating can be used as a model
    input and learned from history instead of only being displayed.

    Returns DataFrame with Tech_Score (mean vote, -1..+1), Tech_Trend_Votes
    (moving-average votes only) and Tech_Score_Chg_5d.
    """
    d = add_overlay_indicators(df)
    c = d["Close"]
    trend_votes = [np.sign(c - d[f"SMA_{w}"]) for w in (20, 50, 200)]
    trend_votes.append(np.sign(d["SMA_50"] - d["SMA_200"]))
    rsi = d["RSI_14"]
    osc_votes = [
        pd.Series(np.select([rsi < 30, rsi > 70], [1, -1], 0), index=d.index),
        np.sign(d["MACD"] - d["MACD_Signal"]),
    ]
    pct_b = (c - d["BB_Lower"]) / (d["BB_Upper"] - d["BB_Lower"])
    osc_votes.append(pd.Series(np.select([pct_b < 0.05, pct_b > 0.95], [1, -1], 0), index=d.index))
    k = d["Stoch_K"]
    osc_votes.append(pd.Series(np.select([k < 20, k > 80], [1, -1], 0), index=d.index))
    roc = c / c.shift(21) - 1
    osc_votes.append(pd.Series(np.select([roc > 0.02, roc < -0.02], [1, -1], 0), index=d.index))

    trend = pd.concat(trend_votes, axis=1).mean(axis=1)
    score = pd.concat(trend_votes + osc_votes, axis=1).mean(axis=1)
    out = pd.DataFrame({"Tech_Score": score, "Tech_Trend_Votes": trend}, index=d.index)
    out["Tech_Score_Chg_5d"] = out["Tech_Score"] - out["Tech_Score"].shift(5)
    return out
