"""
data_loader.py
----------------
Fetches historical OHLCV data for a chosen stock using yfinance.

Usage:
    df = load_stock_data("RELIANCE.NS", start="2015-01-01", end="2025-01-01")
"""

import datetime
import re

import pandas as pd
import yfinance as yf

# Below this many trading days there isn't enough history for the
# indicators (up to a 200/252-day window) plus a train/validation/test split.
MIN_ROWS = 300


def _exclusive_end(end: str) -> str:
    """yfinance treats `end` as EXCLUSIVE (end="2026-09-21" stops at the 18th, the
    previous trading day). Callers here pass an inclusive date, e.g. today,
    so shift it by one day -- otherwise today's prices never show up."""
    return (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def candidate_symbols(ticker: str):
    """
    Symbols to try, in order, when `ticker` has little or no data on Yahoo.

    Main case: Indian SME-platform symbols like "OWAIS-SM.NS". Yahoo's
    search still lists them, but once a company migrates to NSE's main
    board its history lives under the plain symbol ("OWAIS.NS") and the
    -SM symbol has almost no data. Also tries the other Indian exchange
    (.NS <-> .BO), since some stocks only have history on one of them.
    """
    t = ticker.strip().upper()
    out = [t]
    m = re.match(r"^(.+?)-(SM|ST|BE|BZ)\.(NS|BO)$", t)
    if m:
        base = m.group(1)
        out += [f"{base}.NS", f"{base}.BO"]
    elif t.endswith(".NS"):
        out.append(t[:-3] + ".BO")
    elif t.endswith(".BO"):
        out.append(t[:-3] + ".NS")
    return list(dict.fromkeys(out))


def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical daily OHLCV data for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL", "RELIANCE.NS", "TCS.NS", "MSFT".
        For NSE (India) stocks, append ".NS" (e.g. "RELIANCE.NS").
        For BSE, append ".BO".
    start : str
        Start date "YYYY-MM-DD".
    end : str
        End date "YYYY-MM-DD", INCLUSIVE (data for this day is included
        if it exists -- during market hours that's today's live, still-forming candle).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with columns:
        Open, High, Low, Close, Adj Close, Volume

    If `ticker` has less than MIN_ROWS days of data, related symbols
    from candidate_symbols() are tried too (e.g. "OWAIS-SM.NS" ->
    "OWAIS.NS"). The symbol actually used is stored in df.attrs["ticker"].
    """
    best = None
    for symbol in candidate_symbols(ticker):
        df = yf.download(symbol, start=start, end=_exclusive_end(end), auto_adjust=False, progress=False)
        # yfinance sometimes returns MultiIndex columns when a single ticker
        # is passed as a list; flatten just in case.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if best is None or len(df) > len(best[1]):
            best = (symbol, df)
        if len(df) >= MIN_ROWS:
            break

    symbol, df = best
    if df.empty:
        raise ValueError(
            f"No price data found for '{ticker}' on Yahoo Finance. The symbol may be delisted or "
            "renamed. Try searching for the company again and pick a different listing (e.g. .NS vs .BO)."
        )
    df.index.name = "Date"
    df.attrs["ticker"] = symbol
    return df


def load_market_context(start: str, end: str, sector_etf: str = "XLK",
                        vix_symbol: str = "^VIX", index_symbol: str = "^GSPC") -> pd.DataFrame:
    """
    Download broader market context data: VIX (volatility index), S&P 500
    index closes, and a sector ETF. A stock rarely moves independent of
    the broader market -- these give models signal that pure price/volume
    technical indicators can't capture (e.g. "the whole market is
    panicking" vs "this stock specifically is being sold off").

    Parameters
    ----------
    sector_etf : str
        Ticker for a sector ETF relevant to the stock being analyzed.
        Default "XLK" (Technology Select Sector SPDR) fits tech stocks
        like MSFT/AAPL/GOOGL. Use "XLF" for financials, "XLE" for energy,
        "XLV" for healthcare, etc.

    vix_symbol / index_symbol : str
        Volatility index and broad market index. Defaults are the US ones
        (^VIX, ^GSPC); use market_symbols_for(ticker) to get the right
        set automatically (e.g. ^INDIAVIX / ^NSEI for NSE stocks). The
        output column names stay VIX_Close / SP500_Close either way so
        the feature code doesn't need to change.

    Returns
    -------
    pd.DataFrame indexed by Date with columns: VIX_Close, SP500_Close, Sector_Close
    """
    end = _exclusive_end(end)
    vix = yf.download(vix_symbol, start=start, end=end, auto_adjust=False, progress=False)
    sp500 = yf.download(index_symbol, start=start, end=end, auto_adjust=False, progress=False)
    sector = yf.download(sector_etf, start=start, end=end, auto_adjust=False, progress=False)

    for d in (vix, sp500, sector):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    if sp500.empty:
        raise ValueError(f"No market index data for '{index_symbol}'.")
    # Sector/volatility series are nice-to-have: if Yahoo has no data for
    # one (some sector indices aren't published), fall back instead of
    # letting dropna() wipe out the whole frame.
    sector_close = sector["Close"] if not sector.empty else sp500["Close"]
    market_df = pd.DataFrame({
        "VIX_Close": vix["Close"] if not vix.empty else sp500["Close"].pct_change().rolling(20).std() * 100 * (252 ** 0.5),
        "SP500_Close": sp500["Close"],
        "Sector_Close": sector_close,
    })
    market_df = market_df.ffill().dropna()
    market_df.index.name = "Date"
    return market_df


# Indian sector indices on NSE, keyed by Yahoo's sector name.
# Only indices Yahoo actually publishes daily data for; other sectors
# fall back to NIFTY 50.
_INDIA_SECTOR_INDEX = {
    "Technology": "^CNXIT", "Financial Services": "^NSEBANK", "Healthcare": "^CNXPHARMA",
}


def market_symbols_for(ticker: str) -> dict:
    """
    Pick market-context symbols that actually match the stock being
    analyzed, instead of always using US tech (VIX / S&P 500 / XLK):
      - NSE/BSE stocks (.NS / .BO) -> India VIX, NIFTY 50, NSE sector index
      - everything else            -> VIX, S&P 500, matching SPDR sector ETF
    Returns kwargs for load_market_context().
    """
    from stock_search import get_company_profile, SECTOR_ETFS, asset_class

    try:
        profile = get_company_profile(ticker)
        sector, cls = profile.get("sector"), profile.get("asset_class")
    except Exception:
        sector, cls = None, asset_class(ticker)

    # Non-equities have their own drivers: the dollar moves metals and oil,
    # US yields move currencies, and alt-coins follow bitcoin & the Nasdaq.
    if cls == "commodity":
        return {"vix_symbol": "^VIX", "index_symbol": "^GSPC", "sector_etf": "DX-Y.NYB"}
    if cls == "forex":
        return {"vix_symbol": "^VIX", "index_symbol": "DX-Y.NYB", "sector_etf": "^TNX"}
    if cls == "crypto":
        peer = "ETH-USD" if ticker.upper().startswith("BTC") else "BTC-USD"
        return {"vix_symbol": "^VIX", "index_symbol": "^IXIC", "sector_etf": peer}

    # A gold/silver ETF tracks the metal, not its listing's sector.
    if cls == "etf":
        _n = (profile.get("name") or "").lower() if isinstance(locals().get("profile"), dict) else ""
        metal = "GC=F" if "gold" in _n else "SI=F" if "silver" in _n else None
        if metal:
            local = ticker.upper().endswith((".NS", ".BO"))
            return {"vix_symbol": "^INDIAVIX" if local else "^VIX",
                    "index_symbol": "^NSEI" if local else "^GSPC", "sector_etf": metal}

    if ticker.upper().endswith((".NS", ".BO")):
        return {"vix_symbol": "^INDIAVIX", "index_symbol": "^NSEI",
                "sector_etf": _INDIA_SECTOR_INDEX.get(sector, "^NSEI")}
    return {"vix_symbol": "^VIX", "index_symbol": "^GSPC", "sector_etf": SECTOR_ETFS.get(sector, "XLK")}


if __name__ == "__main__":
    # Quick manual test
    data = load_stock_data("AAPL", "2015-01-01", "2025-01-01")
    print(data.head())
    print(f"\nTotal records: {len(data)}")
    data.to_csv("raw_stock_data.csv")
    print("Saved to raw_stock_data.csv")


# ------------------------------------------------------------------------
# Interval-aware loading (intraday, daily, weekly)
# ------------------------------------------------------------------------
_OHLC_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
             "Adj Close": "last", "Volume": "sum"}


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _resample_within_sessions(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate bars to a larger candle WITHOUT letting one candle straddle two
    trading days: each session is resampled from its own first bar, so a 4-hour
    candle on the NSE starts at 09:15, not at midnight.
    """
    if df.empty:
        return df
    pieces = []
    for _, session in df.groupby(df.index.normalize()):
        agg = {k: v for k, v in _OHLC_AGG.items() if k in session.columns}
        r = session.resample(rule, origin=session.index[0]).agg(agg)
        pieces.append(r.dropna(subset=["Close"]))
    out = pd.concat(pieces).sort_index() if pieces else df.iloc[0:0]
    out.attrs = dict(df.attrs)
    return out


def load_ohlcv(ticker: str, interval: str = "1d", start: str = None, end: str = None,
               lookback_days: int = None) -> pd.DataFrame:
    """
    OHLCV at the requested candle size.

    interval : any key of timeframes.TIMEFRAMES ("1m", "5m", "15m", "30m",
               "1h", "2h", "4h", "1d", "1wk")
    start/end: used for daily and weekly only -- intraday history is limited to
               a recent window by the provider, so it is fetched by period.

    The frame carries what it actually is in `attrs`: interval, whether it was
    resampled, the true first/last timestamp and the bar count, so nothing
    downstream has to guess.
    """
    import timeframes as tfm

    spec = tfm.get(interval)
    yf_interval = spec["yf"]
    symbol = ticker

    if spec["intraday"]:
        days = min(lookback_days or tfm.max_history_days(interval), tfm.max_history_days(interval))
        df = yf.download(symbol, period=f"{int(days)}d", interval=yf_interval,
                         progress=False, auto_adjust=False)
        df = _flatten(df).dropna(subset=["Close"])
        if df.empty:
            raise ValueError(
                f"Yahoo Finance returned no {tfm.label(interval)} candles for '{ticker}'. "
                f"Intraday history is limited to about {tfm.max_history_days(interval)} days and is "
                f"not available for every symbol.")
        if df.index.tz is not None:
            df.index = df.index.tz_convert(
                df.index.tz).tz_localize(None)        # keep exchange local time, drop the tz object
        if spec["resample"]:
            df = _resample_within_sessions(df, spec["resample"])
    else:
        end = end or datetime.date.today().strftime("%Y-%m-%d")
        start = start or "2014-01-01"
        df = yf.download(symbol, start=start, end=_exclusive_end(end), interval=yf_interval,
                         progress=False, auto_adjust=False)
        df = _flatten(df).dropna(subset=["Close"])
        if df.empty:
            raise ValueError(f"No {tfm.label(interval)} data found for '{ticker}' on Yahoo Finance.")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    df.index.name = "Date"
    df.attrs.update({
        "ticker": symbol,
        "interval": interval,
        "interval_label": tfm.label(interval),
        "resampled_from": "1h" if spec["resample"] else None,
        "bars": len(df),
        "first": df.index[0],
        "last": df.index[-1],
        "intraday": spec["intraday"],
    })
    return df


def interval_availability(ticker: str, timezone: str = None, asset_class: str = None,
                          probe: bool = False) -> dict:
    """
    Which candle sizes can actually be served for this symbol.

    Without `probe` this answers from the measured provider limits, which is
    what the UI uses to build its menu. With `probe=True` it really downloads a
    small sample of each -- slow, used by the test suite.
    """
    import timeframes as tfm

    out = {}
    for tf in tfm.ORDER:
        if not probe:
            bars = tfm.expected_bars(tf, timezone, asset_class)
            out[tf] = {"ok": bars >= tfm.MIN_BARS_FOR_MODEL, "bars": bars, "probed": False}
            continue
        try:
            df = load_ohlcv(ticker, tf)
            out[tf] = {"ok": len(df) >= tfm.MIN_BARS_FOR_MODEL, "bars": len(df), "probed": True,
                       "first": df.index[0], "last": df.index[-1]}
        except Exception as e:
            out[tf] = {"ok": False, "bars": 0, "probed": True, "error": str(e)[:80]}
    return out
