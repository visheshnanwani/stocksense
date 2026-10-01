"""
timeframes.py
---------------
One place that knows what a "candle" means, so nothing in the project has to
hardcode "daily" ever again.

Every limit below was MEASURED against the data source (Yahoo Finance) rather
than copied from documentation, because the documented limits and the real ones
differ. For HDFCBANK.NS on 28 Sep 2026:

    interval   most history actually returned      rows
    1m         7 days                              2,542
    5m         60 days                             4,274
    15m        60 days                             1,450
    30m        60 days                               756
    1h         730 days                            5,034
    4h (native) 60 days                              117   <-- unusable
    1d         full history                        7,720
    1wk        full history                        1,604

So 2-hour and 4-hour candles are NOT fetched natively (Yahoo only serves 60 days
of them, which is 117 bars -- far too few to model). They are built by resampling
1-hour bars, which gives ~2,500 and ~1,250 bars respectively. That resampling is
explicit, documented, and reported in the UI; it is never silently substituted.

Sessions are market-specific: NSE trades 09:15-15:30 (375 minutes), US markets
09:30-16:00 (390 minutes), so "one trading day" is a different number of candles
depending on both the timeframe and where the symbol trades.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- definitions
TIMEFRAMES = {
    "1m":  dict(label="1 minute",   yf="1m",  minutes=1,    max_days=7,    resample=None, intraday=True),
    "5m":  dict(label="5 minutes",  yf="5m",  minutes=5,    max_days=59,   resample=None, intraday=True),
    "15m": dict(label="15 minutes", yf="15m", minutes=15,   max_days=59,   resample=None, intraday=True),
    "30m": dict(label="30 minutes", yf="30m", minutes=30,   max_days=59,   resample=None, intraday=True),
    "1h":  dict(label="1 hour",     yf="60m", minutes=60,   max_days=729,  resample=None, intraday=True),
    "2h":  dict(label="2 hours",    yf="60m", minutes=120,  max_days=729,  resample="2h", intraday=True),
    "4h":  dict(label="4 hours",    yf="60m", minutes=240,  max_days=729,  resample="4h", intraday=True),
    "1d":  dict(label="1 day",      yf="1d",  minutes=None, max_days=None, resample=None, intraday=False),
    "1wk": dict(label="1 week",     yf="1wk", minutes=None, max_days=None, resample=None, intraday=False),
}

ORDER = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1wk"]
DEFAULT = "1d"

# Minutes in one trading session, by exchange timezone.
SESSION_MINUTES = {
    "Asia/Kolkata": 375,        # 09:15 - 15:30
    "America/New_York": 390,    # 09:30 - 16:00
    "Europe/London": 510,       # 08:00 - 16:30
    "Asia/Tokyo": 300,
    "Asia/Hong_Kong": 330,
}
DEFAULT_SESSION_MINUTES = 390
CRYPTO_SESSION_MINUTES = 24 * 60          # crypto never closes

# Models need enough labelled rows; below this a timeframe cannot be analysed.
MIN_BARS_FOR_MODEL = 300
# A forecast horizon longer than this share of available history has almost no
# independent examples behind it, so it is refused rather than faked.
MAX_HORIZON_SHARE = 0.05


def get(tf: str) -> dict:
    return TIMEFRAMES[tf if tf in TIMEFRAMES else DEFAULT]


def label(tf: str) -> str:
    return get(tf)["label"]


def is_intraday(tf: str) -> bool:
    return get(tf)["intraday"]


def session_minutes(timezone: str = None, asset_class: str = None) -> int:
    if asset_class in ("crypto", "forex"):
        return CRYPTO_SESSION_MINUTES
    return SESSION_MINUTES.get(timezone, DEFAULT_SESSION_MINUTES)


def candles_per_session(tf: str, timezone: str = None, asset_class: str = None) -> float:
    """How many candles of this size fit in one trading day."""
    spec = get(tf)
    if tf == "1d":
        return 1.0
    if tf == "1wk":
        return 0.2                     # five trading days per weekly candle
    return max(session_minutes(timezone, asset_class) / spec["minutes"], 1.0)


def horizon_to_candles(tf: str, trading_days: float, timezone: str = None,
                       asset_class: str = None) -> int:
    """'21 trading days' -> how many candles of the selected size that is."""
    return max(int(round(trading_days * candles_per_session(tf, timezone, asset_class))), 1)


def candles_to_trading_days(tf: str, candles: int, timezone: str = None,
                            asset_class: str = None) -> float:
    return candles / candles_per_session(tf, timezone, asset_class)


def max_history_days(tf: str) -> int:
    """How far back the provider will actually go for this candle size."""
    spec = get(tf)
    return spec["max_days"] if spec["max_days"] else 365 * 25


def expected_bars(tf: str, timezone: str = None, asset_class: str = None) -> int:
    """Roughly how many bars the provider can return, used to filter the menu."""
    spec = get(tf)
    days = spec["max_days"]
    if days is None:
        return 5000
    sessions = days * (5 / 7) if asset_class not in ("crypto",) else days
    return int(sessions * candles_per_session(tf, timezone, asset_class))


def available(timezone: str = None, asset_class: str = None, min_bars: int = MIN_BARS_FOR_MODEL):
    """
    The timeframes worth offering for this symbol: only those where the provider
    can return enough bars to analyse. Nothing is listed that cannot be served.
    """
    out = []
    for tf in ORDER:
        if expected_bars(tf, timezone, asset_class) >= min_bars:
            out.append(tf)
    return out or [DEFAULT]


def horizon_options(tf: str, timezone: str = None, asset_class: str = None):
    """
    Sensible forecast horizons for a given candle size, as
    [(label, trading_days), ...]. A 15-minute chart has no business offering a
    6-month forecast, and a weekly chart has no business offering "next day".
    """
    if tf == "1wk":
        return [("4 weeks", 20), ("3 months", 63), ("6 months", 126), ("1 year", 252)]
    if tf == "1d":
        return [("1 day", 1), ("1 week", 5), ("2 weeks", 10), ("1 month", 21), ("3 months", 63)]
    if tf in ("2h", "4h", "1h"):
        return [("Next candle", 1 / candles_per_session(tf, timezone, asset_class)),
                ("Half a session", 0.5), ("1 session", 1), ("2 sessions", 2), ("1 week", 5)]
    return [("Next candle", 1 / candles_per_session(tf, timezone, asset_class)),
            ("30 minutes", 30 / session_minutes(timezone, asset_class)),
            ("2 hours", 120 / session_minutes(timezone, asset_class)),
            ("Half a session", 0.5), ("1 session", 1)]


def model_horizons(tf: str, n_bars: int, timezone: str = None, asset_class: str = None):
    """
    The horizons (in CANDLES) the engine should train models for, scaled to the
    timeframe and to how much history exists. Never longer than
    MAX_HORIZON_SHARE of the data, because a horizon with only a handful of
    independent examples behind it cannot be validated.
    """
    per_session = candles_per_session(tf, timezone, asset_class)
    if tf == "1d":
        wanted = [1, 5, 10, 21, 63]
    elif tf == "1wk":
        wanted = [1, 2, 4, 13, 26]
    else:
        wanted = sorted({1,
                         max(int(round(per_session * 0.25)), 2),
                         max(int(round(per_session * 0.5)), 3),
                         max(int(round(per_session)), 4),
                         max(int(round(per_session * 2)), 5)})
    cap = max(int(n_bars * MAX_HORIZON_SHARE), 1)
    return [h for h in wanted if h <= cap] or [1]


def describe(tf: str, n_bars: int, timezone: str = None, asset_class: str = None) -> dict:
    """Everything the UI needs to state plainly what it is working with."""
    spec = get(tf)
    return {
        "code": tf,
        "label": spec["label"],
        "intraday": spec["intraday"],
        "resampled_from": "1 hour" if spec["resample"] else None,
        "bars": n_bars,
        "candles_per_session": round(candles_per_session(tf, timezone, asset_class), 2),
        "max_history_days": spec["max_days"],
        "max_horizon_candles": max(int(n_bars * MAX_HORIZON_SHARE), 1),
    }


def annualisation_factor(tf: str, timezone: str = None, asset_class: str = None) -> float:
    """Bars per year, for anything that needs to annualise (volatility)."""
    if tf == "1d":
        return 252.0
    if tf == "1wk":
        return 52.0
    return 252.0 * candles_per_session(tf, timezone, asset_class)
