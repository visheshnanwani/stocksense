"""
nse_fo_data.py
--------------
Real NSE F&O history, straight from the exchange.

Every options conclusion in this project until now rested on Black-Scholes
prices computed from realised volatility, because no source served actual
premiums: Yahoo has no NSE options at all, and NSE's own option-chain API
returns an empty body to anything that is not its own page.

The daily F&O bhavcopy is a different door and it is open. It is the exchange's
own settlement file, one row per contract per session:

    TradDt TckrSymb XpryDt StrkPric OptnTp
    OpnPric HghPric LwPric ClsPric LastPric PrvsClsgPric
    UndrlygPric OpnInterest ChngInOpnInterest TtlTradgVol TtlTrfVal

So the modelled premiums can be replaced with what actually traded. Files are
cached under fo_cache/ so the archive is fetched once.

    https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip
"""

import io
import os
import time
import zipfile

import pandas as pd
import requests

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fo_cache")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
BASE = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"

_session = None


def session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
                          "Referer": "https://www.nseindia.com/all-reports-derivatives"})
        try:
            s.get("https://www.nseindia.com/all-reports-derivatives", timeout=30)
        except Exception:
            pass
        _session = s
    return _session


def fetch_day(day: pd.Timestamp, retries: int = 2) -> pd.DataFrame:
    """One session's F&O bhavcopy. Empty frame on a holiday or a missing file."""
    os.makedirs(CACHE, exist_ok=True)
    key = day.strftime("%Y%m%d")
    path = os.path.join(CACHE, f"fo_{key}.parquet")
    miss = os.path.join(CACHE, f"fo_{key}.missing")
    if os.path.exists(path):
        return pd.read_parquet(path)
    if os.path.exists(miss):
        return pd.DataFrame()

    for attempt in range(retries + 1):
        try:
            r = session().get(BASE.format(d=key), timeout=60)
            if r.status_code == 200 and len(r.content) > 5000:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
                df.to_parquet(path, index=False)
                return df
            if r.status_code in (403, 404):
                open(miss, "w").close()          # holiday / not published
                return pd.DataFrame()
        except Exception:
            if attempt < retries:
                time.sleep(2)
    return pd.DataFrame()


def options(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Just the option rows, with tidy names and numeric types."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[df["OptnTp"].isin(["CE", "PE"])].copy()
    if symbol:
        d = d[d["TckrSymb"] == symbol]
    if d.empty:
        return d
    keep = {"TradDt": "date", "TckrSymb": "symbol", "XpryDt": "expiry",
            "StrkPric": "strike", "OptnTp": "cp", "OpnPric": "open",
            "HghPric": "high", "LwPric": "low", "ClsPric": "close",
            "LastPric": "last", "PrvsClsgPric": "prev_close",
            "UndrlygPric": "spot", "OpnIntrst": "oi",
            "ChngInOpnIntrst": "oi_change", "TtlTradgVol": "volume",
            "TtlTrfVal": "turnover", "TtlNbOfTxsExctd": "trades"}
    d = d.rename(columns={k: v for k, v in keep.items() if k in d.columns})
    d = d[[c for c in keep.values() if c in d.columns]]
    for c in ("strike", "open", "high", "low", "close", "last", "prev_close",
              "spot", "oi", "oi_change", "volume", "turnover", "trades"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("date", "expiry"):
        if c in d:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    d["dte"] = (d["expiry"] - d["date"]).dt.days
    d["moneyness"] = d["strike"] / d["spot"] - 1
    return d


def load_range(start, end, symbols=None, min_volume=0) -> pd.DataFrame:
    """Every option row between two dates. Cached, so the second call is fast."""
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    frames, got, missing = [], 0, 0
    for day in days:
        raw = fetch_day(day)
        if raw.empty:
            missing += 1
            continue
        o = options(raw)
        if o.empty:
            continue
        if symbols:
            o = o[o["symbol"].isin(symbols)]
        if min_volume:
            o = o[o["volume"] >= min_volume]
        if not o.empty:
            frames.append(o)
            got += 1
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.attrs["sessions"] = got
    out.attrs["missing"] = missing
    return out


if __name__ == "__main__":
    import sys
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=int(sys.argv[1]) if len(sys.argv) > 1 else 45)
    df = load_range(start, end, symbols=["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK"])
    print(f"sessions loaded: {df.attrs.get('sessions')}  rows: {len(df):,}")
    if len(df):
        print(df.head(4).to_string(index=False))
        print("\nby symbol:")
        print(df.groupby("symbol").agg(rows=("close", "size"),
                                       sessions=("date", "nunique"),
                                       volume=("volume", "sum")).to_string())
