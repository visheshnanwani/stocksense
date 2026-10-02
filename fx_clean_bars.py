"""
fx_clean_bars.py
----------------
Daily FX bars with a High and Low that actually belong to the day.

THE PROBLEM THIS SOLVES
    Yahoo's daily FX bar is malformed. Spot FX trades 24 hours and the high/low
    it publishes span a window that does not end where the close is stamped, so
    the next day's price is already inside today's range:

        EURUSD=X   close outside its own high/low   2.9% of bars
                   TOMORROW's close inside TODAY's range   74.6%   (AAPL: 45%)

    Anything built from that high/low can see ahead. Measured directly:
    S_Lower_Wick_Pct correlated 0.490 with the NEXT day's FX return, against
    0.047 for the strongest equity feature. See market_data.ohlc_range_quality.

THE FIX
    Yahoo's HOURLY FX data is fine -- it is the daily aggregation that is
    broken. So the daily bar is rebuilt here from hourly bars, which gives a
    high and low that are genuinely the extremes of that calendar day:

        rebuilt EURUSD   close outside high/low  0.0%   next-in-range 55.0%
        yahoo   EURUSD   close outside high/low  2.9%   next-in-range 74.6%

WHAT IT COSTS
    Yahoo serves at most 730 days of hourly data, so this covers roughly the
    last two years, not the twelve the daily feed covers. That is a real limit
    and the reason this is a supplement rather than a replacement: the long
    close-only history still comes from the daily feed, and the range features
    simply do not exist before the hourly window starts. Gradient boosting
    treats those as missing, which is what they are.

    Sources that would give the full history were checked and rejected:
      stooq            behind a JavaScript bot check
      Dukascopy        one file per day -- ~31,000 requests for 11 pairs
      ECB/Frankfurter  12 years, clean, but close only -- no high or low
"""

import datetime
import os

import numpy as np
import pandas as pd

CACHE_DIR = "fx_hourly_cache"
MAX_HOURLY_DAYS = 730          # Yahoo's hard limit for 1h bars


def _cache_path(symbol):
    return os.path.join(CACHE_DIR, symbol.replace("=", "_").replace("/", "_") + ".parquet")


def fetch_hourly(symbol: str, refresh=False) -> pd.DataFrame:
    """Hourly bars for one pair, cached to disk (Yahoo serves ~730 days)."""
    path = _cache_path(symbol)
    if not refresh and os.path.exists(path):
        try:
            age = datetime.date.today() - datetime.date.fromtimestamp(os.path.getmtime(path))
            if age.days < 1:
                return pd.read_parquet(path)
        except Exception:
            pass
    import yfinance as yf
    df = yf.download(symbol, period=f"{MAX_HOURLY_DAYS}d", interval="1h",
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception:
        pass
    return df


def daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly bars into a daily bar whose high/low are that day's."""
    if hourly is None or hourly.empty:
        return pd.DataFrame()
    h = hourly.copy()
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_convert("UTC").tz_localize(None)
    d = h.resample("1D").agg(Open=("Open", "first"), High=("High", "max"),
                             Low=("Low", "min"), Close=("Close", "last")).dropna()
    d["Volume"] = 0.0          # FX has no exchange volume; kept for shape
    return d


def clean_daily(symbol: str, refresh=False) -> pd.DataFrame:
    return daily_from_hourly(fetch_hourly(symbol, refresh=refresh))


def splice(daily_long: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """
    DO NOT USE -- kept only so the mistake is on the record.

    This grafted the rebuilt high/low onto Yahoo's daily close, on the reasoning
    that keeping the original close left the target comparable. It is wrong, and
    badly so. Yahoo's daily FX close is stamped on one clock and the rebuilt
    range spans the UTC calendar day on another, so the range runs past the
    close and the next day's price falls inside it MORE often than before the
    "fix":

        EURUSD   pure rebuilt  next-in-range 50.2%   (ok)
                 SPLICED       next-in-range 90.1%   (worse than Yahoo's 74.7%)
        GBPUSD   pure rebuilt  next-in-range 49.0%
                 SPLICED       next-in-range 94.4%

    It produced 82.5% directional accuracy on one-day FX, which is how it was
    caught. A bar must come from one clock: use clean_daily() whole, or use the
    daily feed with close-only features. Never half of each.
    """
    raise NotImplementedError(
        "splice() mixes two clocks in one bar and reintroduces the look-ahead; "
        "use clean_daily() for a consistent rebuilt bar instead")


def report(symbols):
    import data_loader as dl
    import market_data as md
    print(f"{'pair':<12}{'yahoo daily':>28}{'rebuilt from hourly':>32}")
    print(f"{'':<12}{'out':>8}{'next-in':>10}{'verdict':>10}{'days':>8}{'out':>8}{'next-in':>10}{'verdict':>10}")
    for s in symbols:
        try:
            y = dl.load_stock_data(s, "2014-01-01", str(datetime.date.today()))
            qy = md.ohlc_range_quality(y, s)
            c = clean_daily(s)
            qc = md.ohlc_range_quality(c, s) if len(c) else {"verdict": "no data",
                                                             "close_outside_share": np.nan,
                                                             "next_in_range_share": np.nan}
            print(f"{s:<12}{qy['close_outside_share'] * 100:>7.1f}%{qy['next_in_range_share'] * 100:>9.1f}%"
                  f"{qy['verdict']:>10}{len(c):>8}"
                  f"{qc['close_outside_share'] * 100:>7.1f}%{qc['next_in_range_share'] * 100:>9.1f}%"
                  f"{qc['verdict']:>10}")
        except Exception as e:
            print(f"{s:<12} failed: {type(e).__name__}")


FX_PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X", "EURINR=X", "GBPINR=X",
            "JPYINR=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "USDCNY=X"]

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    report(FX_PAIRS)
