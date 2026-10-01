"""
candlestick_patterns.py
-------------------------
Reads candlestick charts the way a technical analyst would: detects
classic Japanese candlestick patterns on OHLC data, turns the most
recent ones into a plain-English "reading", and -- importantly --
checks how RELIABLE each pattern has actually been on THIS stock's own
history, instead of trusting textbook claims blindly.

Patterns detected (22):
  Single-candle : Doji, Dragonfly Doji, Gravestone Doji, Hammer,
                  Hanging Man, Inverted Hammer, Shooting Star,
                  Bullish/Bearish Marubozu, Spinning Top
  Two-candle    : Bullish/Bearish Engulfing, Bullish/Bearish Harami,
                  Piercing Line, Dark Cloud Cover, Tweezer Bottom/Top
  Three-candle  : Morning Star, Evening Star, Three White Soldiers,
                  Three Black Crows

Most reversal patterns only mean something in the right CONTEXT (a
hammer after a downtrend is bullish; the exact same candle after an
uptrend is a bearish "hanging man"), so a simple prior-trend filter is
applied, the same way a human chart reader would.

Usage:
    from candlestick_patterns import detect_patterns, pattern_events, read_candles, pattern_reliability
    events = pattern_events(raw_df)            # every pattern occurrence
    reading = read_candles(raw_df)             # summary of the last few candles
    stats = pattern_reliability(raw_df)        # how well each pattern worked on this stock

HONEST CAVEAT: candlestick patterns are widely used but their standalone
predictive power in academic studies is weak-to-modest, and varies a
lot by stock and market. That's exactly why pattern_reliability()
exists -- use it to see whether a pattern has any edge on the stock
you're analyzing before trusting it.
"""

import numpy as np
import pandas as pd

# name -> (bias, strength 1-3, description)
#   bias: +1 bullish, -1 bearish, 0 neutral/indecision
PATTERN_INFO = {
    "Doji": (0, 1, "Open ≈ close: buyers and sellers are in balance. Signals indecision; often precedes a turn when it follows a strong move."),
    "Dragonfly Doji": (1, 1, "Doji with a long lower shadow after a decline: sellers pushed price down but buyers drove it all the way back up."),
    "Gravestone Doji": (-1, 1, "Doji with a long upper shadow after a rise: buyers pushed price up but sellers drove it all the way back down."),
    "Hammer": (1, 2, "Small body at the top with a long lower shadow after a downtrend: strong intraday rejection of lower prices. Bullish reversal."),
    "Hanging Man": (-1, 1, "Hammer-shaped candle after an uptrend: shows selling pressure appearing. Bearish warning, needs confirmation."),
    "Inverted Hammer": (1, 1, "Small body at the bottom with a long upper shadow after a downtrend: buyers are starting to test higher. Needs confirmation."),
    "Shooting Star": (-1, 2, "Small body at the bottom with a long upper shadow after an uptrend: rally was rejected. Bearish reversal."),
    "Bullish Marubozu": (1, 2, "Big green candle with almost no shadows: buyers were in control from open to close."),
    "Bearish Marubozu": (-1, 2, "Big red candle with almost no shadows: sellers were in control from open to close."),
    "Spinning Top": (0, 1, "Small body with shadows on both sides: indecision, momentum is weakening."),
    "Bullish Engulfing": (1, 2, "Green candle whose body completely engulfs the previous red body after a decline. Classic bullish reversal."),
    "Bearish Engulfing": (-1, 2, "Red candle whose body completely engulfs the previous green body after a rise. Classic bearish reversal."),
    "Bullish Harami": (1, 1, "Small green candle inside the previous large red body: downtrend is losing steam."),
    "Bearish Harami": (-1, 1, "Small red candle inside the previous large green body: uptrend is losing steam."),
    "Piercing Line": (1, 2, "After a red candle, price opens lower but closes above the midpoint of the prior body. Bullish reversal."),
    "Dark Cloud Cover": (-1, 2, "After a green candle, price opens higher but closes below the midpoint of the prior body. Bearish reversal."),
    "Tweezer Bottom": (1, 1, "Two candles with matching lows after a decline: a support level held twice."),
    "Tweezer Top": (-1, 1, "Two candles with matching highs after a rise: a resistance level held twice."),
    "Morning Star": (1, 3, "Red candle, small 'star' candle, then a strong green candle: a three-step bullish reversal. One of the more reliable patterns."),
    "Evening Star": (-1, 3, "Green candle, small 'star' candle, then a strong red candle: a three-step bearish reversal. One of the more reliable patterns."),
    "Three White Soldiers": (1, 3, "Three consecutive strong green candles, each closing higher: sustained buying pressure."),
    "Three Black Crows": (-1, 3, "Three consecutive strong red candles, each closing lower: sustained selling pressure."),
}

PATTERN_NAMES = list(PATTERN_INFO.keys())
BIAS_LABEL = {1: "Bullish", -1: "Bearish", 0: "Neutral"}


def _candle_parts(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    p = pd.DataFrame(index=df.index)
    p["O"], p["H"], p["L"], p["C"] = o, h, l, c
    p["body"] = (c - o).abs()
    p["range"] = (h - l).replace(0, np.nan)
    p["upper"] = h - np.maximum(o, c)
    p["lower"] = np.minimum(o, c) - l
    p["green"] = c > o
    p["red"] = c < o
    p["avg_body"] = p["body"].rolling(14, min_periods=5).mean()
    p["mid"] = (o + c) / 2
    return p


def _trend(close: pd.Series, lookback: int = 5):
    """Prior-trend filter, measured on the candles BEFORE the pattern:
    downtrend = yesterday's close is below both its level `lookback`
    days earlier and its 10-day average (uptrend is the mirror image)."""
    prev = close.shift(1)
    sma = prev.rolling(10, min_periods=5).mean()
    down = (prev < close.shift(1 + lookback)) & (prev < sma)
    up = (prev > close.shift(1 + lookback)) & (prev > sma)
    return up, down


def detect_patterns(df: pd.DataFrame, trend_lookback: int = 5) -> pd.DataFrame:
    """
    Returns a boolean DataFrame (same index as df, one column per
    pattern in PATTERN_NAMES). A pattern is flagged on the date of its
    LAST candle, i.e. the day it becomes visible on the chart -- so it
    never uses future information.
    """
    p = _candle_parts(df)
    up, down = _trend(p["C"], trend_lookback)
    body, rng, upper, lower = p["body"], p["range"], p["upper"], p["lower"]
    avg = p["avg_body"]
    O, C, H, L = p["O"], p["C"], p["H"], p["L"]

    # previous candles
    O1, C1, H1, L1, body1 = O.shift(1), C.shift(1), H.shift(1), L.shift(1), body.shift(1)
    green1, red1 = p["green"].shift(1, fill_value=False), p["red"].shift(1, fill_value=False)
    O2, C2, body2 = O.shift(2), C.shift(2), body.shift(2)
    green2, red2 = p["green"].shift(2, fill_value=False), p["red"].shift(2, fill_value=False)
    green, red = p["green"], p["red"]

    out = pd.DataFrame(False, index=df.index, columns=PATTERN_NAMES)

    # ---------- single-candle ----------
    is_doji = body <= 0.1 * rng
    out["Dragonfly Doji"] = is_doji & (upper <= 0.1 * rng) & (lower >= 0.6 * rng) & down
    out["Gravestone Doji"] = is_doji & (lower <= 0.1 * rng) & (upper >= 0.6 * rng) & up
    out["Doji"] = is_doji & ~out["Dragonfly Doji"] & ~out["Gravestone Doji"]

    hammer_shape = (body > 0.1 * rng) & (lower >= 2 * body) & (upper <= 0.25 * rng)
    out["Hammer"] = hammer_shape & down
    out["Hanging Man"] = hammer_shape & up

    inv_shape = (body > 0.1 * rng) & (upper >= 2 * body) & (lower <= 0.25 * rng)
    out["Inverted Hammer"] = inv_shape & down
    out["Shooting Star"] = inv_shape & up

    marubozu = (body >= 0.9 * rng) & (body > 1.3 * avg)
    out["Bullish Marubozu"] = marubozu & green
    out["Bearish Marubozu"] = marubozu & red

    out["Spinning Top"] = (body > 0.1 * rng) & (body <= 0.3 * rng) & (upper > body) & (lower > body) \
        & ~hammer_shape & ~inv_shape

    # ---------- two-candle ----------
    out["Bullish Engulfing"] = red1 & green & (O <= C1) & (C >= O1) & (body > body1) & down
    out["Bearish Engulfing"] = green1 & red & (O >= C1) & (C <= O1) & (body > body1) & up

    big1 = body1 > avg.shift(1)
    out["Bullish Harami"] = red1 & big1 & green & (O > C1) & (C < O1) & (body < 0.6 * body1) & down
    out["Bearish Harami"] = green1 & big1 & red & (O < C1) & (C > O1) & (body < 0.6 * body1) & up

    mid1 = (O1 + C1) / 2
    out["Piercing Line"] = red1 & big1 & green & (O < C1) & (C > mid1) & (C < O1) & down
    out["Dark Cloud Cover"] = green1 & big1 & red & (O > C1) & (C < mid1) & (C > O1) & up

    tol = 0.002 * C
    out["Tweezer Bottom"] = red1 & green & ((L - L1).abs() <= tol) & down
    out["Tweezer Top"] = green1 & red & ((H - H1).abs() <= tol) & up

    # ---------- three-candle ----------
    # Trend is judged before the FIRST candle of the pattern.
    up3, down3 = up.shift(2, fill_value=False), down.shift(2, fill_value=False)
    big2 = body2 > avg.shift(2)
    small_star = body1 <= 0.5 * body2
    mid2 = (O2 + C2) / 2
    out["Morning Star"] = red2 & big2 & small_star & (np.maximum(O1, C1) <= C2 * 1.002) \
        & green & (C > mid2) & down3
    out["Evening Star"] = green2 & big2 & small_star & (np.minimum(O1, C1) >= C2 * 0.998) \
        & red & (C < mid2) & up3

    strong = body > 0.6 * avg
    strong1, strong2 = strong.shift(1, fill_value=False), strong.shift(2, fill_value=False)
    out["Three White Soldiers"] = green & green1 & green2 & strong & strong1 & strong2 \
        & (C > C1) & (C1 > C2) & (O > O1) & (O < C1) & (O1 > O2) & (O1 < C2) \
        & (upper <= 0.3 * body) & ~up3
    out["Three Black Crows"] = red & red1 & red2 & strong & strong1 & strong2 \
        & (C < C1) & (C1 < C2) & (O < O1) & (O > C1) & (O1 < O2) & (O1 > C2) \
        & (lower <= 0.3 * body) & ~down3

    # When a strong multi-candle pattern fires, don't also report the
    # weaker single-candle pattern it contains on the same day.
    multi = out[["Morning Star", "Evening Star", "Three White Soldiers", "Three Black Crows",
                 "Bullish Engulfing", "Bearish Engulfing"]].any(axis=1)
    for weak in ["Doji", "Spinning Top", "Bullish Marubozu", "Bearish Marubozu"]:
        out[weak] &= ~multi

    return out.fillna(False).astype(bool)


def pattern_events(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Long-format list of every detected pattern: one row per occurrence
    with Date, Pattern, Bias, Strength, Close, Volume_Confirmed, and the
    realized forward return over `forward_days` (NaN for recent
    occurrences that don't have enough future data yet).
    """
    flags = detect_patterns(df)
    vol_ok = df["Volume"] > 1.2 * df["Volume"].rolling(20, min_periods=5).mean() if "Volume" in df else False
    fwd = df["Close"].shift(-forward_days) / df["Close"] - 1

    rows = []
    for name in PATTERN_NAMES:
        bias, strength, _ = PATTERN_INFO[name]
        for d in flags.index[flags[name]]:
            rows.append({
                "Date": d, "Pattern": name, "Bias": BIAS_LABEL[bias], "Bias_Num": bias,
                "Strength": strength, "Close": float(df.at[d, "Close"]),
                "Volume_Confirmed": bool(vol_ok.loc[d]) if isinstance(vol_ok, pd.Series) else False,
                f"Fwd_{forward_days}d_Return": float(fwd.loc[d]) if pd.notna(fwd.loc[d]) else np.nan,
            })
    if not rows:
        return pd.DataFrame(columns=["Date", "Pattern", "Bias", "Bias_Num", "Strength", "Close",
                                     "Volume_Confirmed", f"Fwd_{forward_days}d_Return"])
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def pattern_reliability(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    How well has each pattern ACTUALLY worked on this stock?

    For every pattern with a directional bias, reports how often the
    price moved in the expected direction over the next `forward_days`
    trading days (hit rate), and the average forward return. Compare the
    hit rate against the stock's base rate (how often it goes up in ANY
    5-day window) -- a pattern is only useful if it beats that.
    """
    ev = pattern_events(df, forward_days)
    col = f"Fwd_{forward_days}d_Return"
    ev = ev.dropna(subset=[col])
    base_up = (df["Close"].shift(-forward_days) / df["Close"] - 1).dropna().gt(0).mean()

    rows = []
    for name, g in ev.groupby("Pattern"):
        bias = PATTERN_INFO[name][0]
        r = g[col]
        if bias > 0:
            hit, base = (r > 0).mean(), base_up
        elif bias < 0:
            hit, base = (r < 0).mean(), 1 - base_up
        else:
            hit, base = np.nan, np.nan
        rows.append({
            "Pattern": name, "Bias": BIAS_LABEL[bias], "Occurrences": len(g),
            "Hit rate": hit, "Base rate": base,
            "Edge vs base": (hit - base) if bias != 0 else np.nan,
            f"Avg {forward_days}d return": r.mean(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Occurrences", ascending=False).reset_index(drop=True)


def read_candles(df: pd.DataFrame, lookback: int = 5, reliability: pd.DataFrame = None) -> dict:
    """
    Plain-English reading of the most recent `lookback` candles.

    Recent patterns are scored (bias x strength, newer candles weigh
    more, volume-confirmed patterns weigh 1.5x) into a net score:
      > +1.5 bullish, < -1.5 bearish, otherwise neutral / mixed.
    If `reliability` is given, patterns with a negative historical edge
    on this stock are down-weighted, and the reading notes it.
    """
    recent_df = df.tail(lookback + 30)  # extra history so trend/averages are warm
    ev = pattern_events(recent_df)
    cutoff = df.index[-lookback] if len(df) >= lookback else df.index[0]
    ev = ev[ev["Date"] >= cutoff]

    edge_map = {}
    if reliability is not None and not reliability.empty:
        edge_map = dict(zip(reliability["Pattern"], reliability["Edge vs base"]))

    score = 0.0
    notes = []
    last = df.index[-1]
    positions = {d: i for i, d in enumerate(df.index[-lookback:])}
    for _, e in ev.iterrows():
        age = lookback - 1 - positions.get(e["Date"], 0)  # 0 = today
        w = e["Bias_Num"] * e["Strength"] * (0.8 ** age)
        if e["Volume_Confirmed"]:
            w *= 1.5
        edge = edge_map.get(e["Pattern"])
        if edge is not None and pd.notna(edge) and edge < 0:
            w *= 0.5
        score += w

        when = "today" if e["Date"] == last else f"{age} bar{'s' if age != 1 else ''} ago"
        extra = " (on above-average volume)" if e["Volume_Confirmed"] else ""
        hist = ""
        if edge is not None and pd.notna(edge):
            hist = f" — historically {'beat' if edge > 0 else 'lagged'} the base rate on this stock by {abs(edge)*100:.0f} pts"
        notes.append(f"**{e['Pattern']}** ({e['Bias'].lower()}) {when}{extra}{hist}. {PATTERN_INFO[e['Pattern']][2]}")

    if score > 1.5:
        verdict, emoji = "Bullish", "🟢"
    elif score < -1.5:
        verdict, emoji = "Bearish", "🔴"
    elif ev.empty:
        verdict, emoji = "No clear pattern", "⚪"
    else:
        verdict, emoji = "Neutral / Mixed", "⚪"

    # Describe today's candle even if it isn't a named pattern.
    p = _candle_parts(df.tail(20)).iloc[-1]
    rng = p["range"] if pd.notna(p["range"]) else 0
    if rng == 0:
        today = "Today's candle has no range (flat)."
    else:
        color = "green (closed up)" if p["green"] else ("red (closed down)" if p["red"] else "flat")
        size = "large" if p["body"] > 1.3 * p["avg_body"] else ("small" if p["body"] < 0.6 * p["avg_body"] else "average-sized")
        today = (f"Today's candle is a {size} {color} body; body is {p['body']/rng*100:.0f}% of the day's range, "
                 f"upper shadow {p['upper']/rng*100:.0f}%, lower shadow {p['lower']/rng*100:.0f}%.")

    return {"verdict": verdict, "emoji": emoji, "score": score, "events": ev,
            "notes": notes, "today": today, "lookback": lookback}


def candle_bias_series(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Rolling sum of (bias x strength) of patterns over the last `window`
    days -- a compact numeric feature for the ML models (no look-ahead)."""
    flags = detect_patterns(df)
    daily = pd.Series(0.0, index=df.index)
    for name in PATTERN_NAMES:
        bias, strength, _ = PATTERN_INFO[name]
        daily += flags[name].astype(float) * bias * strength
    return daily.rolling(window, min_periods=1).sum()


if __name__ == "__main__":
    from data_loader import load_stock_data
    raw = load_stock_data("MSFT", "2018-01-01", "2026-01-01")
    print(pattern_events(raw).tail(15))
    print(pattern_reliability(raw))
    r = read_candles(raw)
    print(r["verdict"], r["score"], r["today"])
    for n in r["notes"]:
        print(" -", n)
