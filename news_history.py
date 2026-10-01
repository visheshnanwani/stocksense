"""
news_history.py
-----------------
HISTORICAL news signals that the models can actually learn from, combining
three sources (the dashboard's live headline box is news_sentiment.py):

  1. GDELT (free, no API key)      -- daily average news TONE and article
     VOLUME for the company's name, across worldwide online news, back to
     2017. This is the main training source.
  2. Alpha Vantage (optional key)  -- per-ticker sentiment scores from its
     NEWS_SENTIMENT endpoint (history from ~2022, mainly US stocks). Free key:
     https://www.alphavantage.co/support/#api-key  -> set the environment
     variable ALPHAVANTAGE_API_KEY, or paste it in the dashboard sidebar.
  3. Own daily collection          -- every time the dashboard (or
     collect_news.py) runs, today's headlines from Yahoo Finance + Google
     News are scored and saved, building your own history going forward.

Everything is cached under news_cache/ so each source is only downloaded
once; later runs fetch just the new days.

NO LOOK-AHEAD: news dates are calendar days in UTC, while markets close at
different UTC times (NSE 10:00 UTC, NYSE ~20:00 UTC). To be safe regardless
of time zone, a trading day's features only use news dated up to the
PREVIOUS calendar day.

HOW THE MODELS USE IT (dashboard_logic.py): news columns are an OPTIONAL
extra input next to ~40 price features. Each model is trained with and
without them, and news is only kept if it lowers error on validation data
-- so it's taken into consideration, never relied on blindly.
"""

import datetime
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "news_cache"
_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-prediction research project)"}

GDELT_URL = "http://api.gdeltproject.org/api/v2/doc/doc"
GDELT_START = datetime.date(2017, 1, 1)  # earliest date the GDELT DOC API serves
_GDELT_MIN_GAP = 6.0                     # GDELT asks for at most one request per 5 s
_last_gdelt_call = [0.0]

AV_URL = "https://www.alphavantage.co/query"
AV_START = datetime.date(2022, 3, 1)     # Alpha Vantage news history starts around here

# Minimum days of self-collected history before it's offered to the models
OWN_MIN_DAYS = 120

NEWS_FEATURE_COLUMNS = [
    "News_Tone_3d",      # volume-weighted GDELT tone, last 3 days
    "News_Tone_14d",     # ... last 14 days
    "News_Tone_Shift",   # 3d minus 14d: is coverage getting more positive/negative?
    "News_Attention",    # log ratio of recent article share vs. its 60-day norm (unusual attention)
    "News_Has_Coverage", # 1 if any articles in the last 3 days
]
AV_FEATURE_COLUMNS = ["AV_Sentiment_3d", "AV_Articles_3d"]
OWN_FEATURE_COLUMNS = ["Own_Sentiment_3d"]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()[:60]


_NAME_SUFFIXES = r"\b(limited|ltd|inc|incorporated|corporation|corp|company|co|plc|holdings?|group|the)\b\.?"


def company_query(name: str) -> str:
    """'Microsoft Corporation' -> 'Microsoft', 'Reliance Industries Limited' ->
    'Reliance Industries'. Legal suffixes add nothing to a news search."""
    cleaned = re.sub(_NAME_SUFFIXES, " ", name, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w&' .-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-")
    return cleaned or name


# --------------------------------------------------------------------------
# 1. GDELT
# --------------------------------------------------------------------------
class GdeltBusy(ConnectionError):
    """GDELT is rate-limiting or unreachable right now -- try again later."""


def _gdelt_request(params: dict, deadline: float, max_attempts: int = 5):
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        wait = _GDELT_MIN_GAP - (time.time() - _last_gdelt_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_gdelt_call[0] = time.time()
        try:
            r = requests.get(GDELT_URL, params=params, headers=_HEADERS, timeout=60)
        except requests.RequestException:
            time.sleep(10)
            continue
        if r.status_code == 429:           # rate limited: back off and retry (within the time budget)
            time.sleep(min(20 * (attempt + 1), max(0.0, deadline - time.time())))
            continue
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:                 # GDELT returns plain text for bad queries / no results
            return {}
    return None


def _gdelt_chunk(query: str, mode: str, start: datetime.date, end: datetime.date, deadline: float) -> pd.Series:
    params = {"query": f'"{query}"', "mode": mode, "format": "json",
              "startdatetime": start.strftime("%Y%m%d000000"),
              "enddatetime": end.strftime("%Y%m%d235959")}
    data = _gdelt_request(params, deadline)
    if data is None:
        raise GdeltBusy("GDELT is busy or rate-limiting right now -- news will be retried on the next run.")
    timeline = (data.get("timeline") or [{}])[0].get("data", [])
    if not timeline:
        return pd.Series(dtype=float)
    if (data.get("query_details") or {}).get("date_resolution", "day") != "day" and (end - start).days > 60:
        # Long spans can come back at coarser resolution -- split and retry.
        mid = start + (end - start) / 2
        return pd.concat([_gdelt_chunk(query, mode, start, mid, deadline),
                          _gdelt_chunk(query, mode, mid + datetime.timedelta(days=1), end, deadline)])
    idx = pd.to_datetime([p["date"][:8] for p in timeline], format="%Y%m%d")
    if mode == "timelinevolraw":
        vol = pd.Series([p["value"] for p in timeline], index=idx, dtype=float)
        norm = pd.Series([p.get("norm", np.nan) for p in timeline], index=idx, dtype=float)
        return pd.DataFrame({"volume": vol, "norm": norm})
    return pd.Series([p["value"] for p in timeline], index=idx, dtype=float, name="tone")


def fetch_gdelt_daily(query: str, start: datetime.date, end: datetime.date = None, chunk_years: int = 10,
                      time_budget: float = 90.0) -> pd.DataFrame:
    """
    Daily GDELT tone (average sentiment of articles, roughly -10..+10) and
    article volume for `query`, cached in news_cache/. Only days after the
    cached range are downloaded on later runs.

    Gives up after `time_budget` seconds (GDELT's free API rate-limits
    hard), saving whatever was downloaded so the next run continues from
    there, then raises GdeltBusy.
    Returns DataFrame indexed by date with columns: tone, volume, norm.
    """
    deadline = time.time() + time_budget
    end = end or datetime.date.today()
    start = max(start, GDELT_START)
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"gdelt_{_slug(query)}.csv"

    cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    if not cached.empty:
        # re-fetch the last few days (GDELT keeps updating recent days) plus anything new
        fetch_ranges = [(max(cached.index.max().date() - datetime.timedelta(days=3), start), end)]
        if cached.index.min().date() > start + datetime.timedelta(days=7):
            fetch_ranges.insert(0, (start, cached.index.min().date()))
    else:
        fetch_ranges = [(start, end)]

    parts, error = [], None
    try:
        for a, b in fetch_ranges:
            cursor = a
            while cursor <= b:
                chunk_end = min(datetime.date(cursor.year + chunk_years, cursor.month, 1) - datetime.timedelta(days=1), b)
                tone = _gdelt_chunk(query, "timelinetone", cursor, chunk_end, deadline)
                vol = _gdelt_chunk(query, "timelinevolraw", cursor, chunk_end, deadline)
                frame = vol if isinstance(vol, pd.DataFrame) else pd.DataFrame(columns=["volume", "norm"])
                frame = frame.join(tone.rename("tone"), how="outer") if len(tone) else frame.assign(tone=np.nan)
                parts.append(frame)
                cursor = chunk_end + datetime.timedelta(days=1)
    except GdeltBusy as e:
        error = e  # keep what we have; save it below, then report

    new = pd.concat(parts) if parts else pd.DataFrame(columns=["tone", "volume", "norm"])
    combined = pd.concat([cached, new]) if not cached.empty else new
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index.name = "date"
    if not combined.empty:
        combined[["tone", "volume", "norm"]].to_csv(path)
    if error is not None and combined.empty:
        raise error
    return combined[["tone", "volume", "norm"]]


# --------------------------------------------------------------------------
# 2. Alpha Vantage (optional)
# --------------------------------------------------------------------------
def alpha_vantage_key(explicit: str = None) -> str:
    return (explicit or os.environ.get("ALPHAVANTAGE_API_KEY") or "").strip()


def _av_symbol(ticker: str):
    """Alpha Vantage news covers US-style tickers; NSE/BSE symbols aren't supported."""
    t = ticker.upper()
    if t.endswith((".NS", ".BO")) or t.startswith("^"):
        return None
    return t.replace("-", ".")


def fetch_alpha_vantage_daily(ticker: str, api_key: str, start: datetime.date, max_requests: int = 12) -> pd.DataFrame:
    """
    Daily relevance-weighted sentiment for `ticker` from Alpha Vantage.
    Fetched quarter by quarter (newest first) and cached per quarter, so the
    free tier's 25-requests/day limit is spread over several runs if needed.
    Returns DataFrame indexed by date: av_sentiment, av_articles. Empty if
    the ticker isn't covered or no key.
    """
    symbol = _av_symbol(ticker)
    if not api_key or not symbol:
        return pd.DataFrame(columns=["av_sentiment", "av_articles"])
    CACHE_DIR.mkdir(exist_ok=True)
    start = max(start, AV_START)
    today = datetime.date.today()

    quarters = pd.period_range(start, today, freq="Q")[::-1]
    frames, requests_made = [], 0
    for q in quarters:
        path = CACHE_DIR / f"av_{_slug(symbol)}_{q}.csv"
        is_current = q.end_time.date() >= today - datetime.timedelta(days=1)
        fresh = path.exists() and (not is_current or
                                   datetime.datetime.fromtimestamp(path.stat().st_mtime).date() == today)
        if fresh:
            frames.append(pd.read_csv(path, index_col=0, parse_dates=True))
            continue
        if requests_made >= max_requests:
            continue
        params = {"function": "NEWS_SENTIMENT", "tickers": symbol, "limit": 1000, "sort": "EARLIEST",
                  "time_from": q.start_time.strftime("%Y%m%dT0000"),
                  "time_to": min(q.end_time.date(), today).strftime("%Y%m%dT2359"), "apikey": api_key}
        try:
            data = requests.get(AV_URL, params=params, headers=_HEADERS, timeout=60).json()
        except (requests.RequestException, ValueError):
            break
        requests_made += 1
        if "feed" not in data:  # rate limit / invalid key messages come back as {"Information": ...} or {"Note": ...}
            msg = data.get("Information") or data.get("Note") or data.get("Error Message") or ""
            if "invalid" in msg.lower() or "api key" in msg.lower():
                raise ValueError(f"Alpha Vantage rejected the API key: {msg[:200]}")
            break
        rows = []
        for art in data["feed"]:
            for ts in art.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol:
                    rel = float(ts.get("relevance_score", 0) or 0)
                    rows.append({"date": pd.to_datetime(art["time_published"][:8], format="%Y%m%d"),
                                 "score": float(ts.get("ticker_sentiment_score", 0) or 0), "rel": rel})
        df = pd.DataFrame(rows, columns=["date", "score", "rel"])
        if df.empty:
            daily = pd.DataFrame(columns=["av_sentiment", "av_articles"])
        else:
            df["w"] = df["rel"].clip(lower=0.05)
            g = df.groupby("date")
            daily = pd.DataFrame({"av_sentiment": g.apply(lambda x: np.average(x["score"], weights=x["w"])),
                                  "av_articles": g.size()})
        daily.index.name = "date"
        daily.to_csv(path)
        frames.append(daily)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["av_sentiment", "av_articles"])
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


# --------------------------------------------------------------------------
# 3. Own daily collection (option 1)
# --------------------------------------------------------------------------
def _own_path(ticker: str) -> Path:
    return CACHE_DIR / f"daily_{_slug(ticker)}.csv"


def collect_today(ticker: str, summary: dict) -> None:
    """
    Save today's headline sentiment (from news_sentiment.get_news_sentiment_summary)
    so a history builds up day by day. Re-running on the same day overwrites
    that day's row with the latest numbers.
    """
    if not summary or summary.get("article_count", 0) == 0:
        return
    CACHE_DIR.mkdir(exist_ok=True)
    path = _own_path(ticker)
    df = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    today = pd.Timestamp(datetime.date.today())
    df.loc[today, "own_sentiment"] = summary["average_score"]
    df.loc[today, "own_articles"] = summary["article_count"]
    df.index.name = "date"
    df.sort_index().to_csv(path)


def load_own_history(ticker: str) -> pd.DataFrame:
    path = _own_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=["own_sentiment", "own_articles"])
    return pd.read_csv(path, index_col=0, parse_dates=True)


# --------------------------------------------------------------------------
# Features for the models
# --------------------------------------------------------------------------
def _prior_days_mean(series: pd.Series, days: int, weights: pd.Series = None) -> pd.Series:
    """Mean over the `days` calendar days BEFORE each day (never the same day)."""
    s = series.shift(1, freq="D")
    if weights is None:
        return s.rolling(f"{days}D", min_periods=1).mean()
    w = weights.shift(1, freq="D")
    num = (s * w).rolling(f"{days}D", min_periods=1).sum()
    den = w.rolling(f"{days}D", min_periods=1).sum()
    return num / den.replace(0, np.nan)


def build_news_features(ticker: str, company_name: str, trading_dates: pd.DatetimeIndex,
                        av_key: str = None, time_budget: float = 90.0):
    """
    Build model-ready daily news features aligned to `trading_dates`
    (plus any later dates, which get neutral values).

    Returns (features DataFrame, info dict). The info dict lists which
    sources worked, coverage, and which feature columns are available.
    Never raises for a missing source -- it just leaves it out.
    """
    info = {"sources": [], "columns": [], "coverage_pct": 0.0, "notes": []}
    cal = pd.date_range(min(trading_dates.min(), pd.Timestamp(GDELT_START)), pd.Timestamp(datetime.date.today()) +
                        pd.Timedelta(days=1), freq="D")
    feats = pd.DataFrame(index=cal)

    query = company_query(company_name or ticker)
    try:
        g = fetch_gdelt_daily(query, trading_dates.min().date(), time_budget=time_budget)
        g = g.reindex(cal)
        # Days missing from GDELT (its own outages) stay NaN = "unknown",
        # which the rolling means skip -- they are NOT treated as zero coverage.
        vol = g["volume"]
        share = vol / g["norm"].replace(0, np.nan)
        tone = g["tone"].where(vol > 0)
        feats["News_Tone_3d"] = _prior_days_mean(tone, 3, vol.fillna(0))
        feats["News_Tone_14d"] = _prior_days_mean(tone, 14, vol.fillna(0))
        feats["News_Tone_Shift"] = feats["News_Tone_3d"] - feats["News_Tone_14d"]
        recent_share = _prior_days_mean(share, 3)
        normal_share = _prior_days_mean(share, 60)
        feats["News_Attention"] = np.log((recent_share + 1e-9) / (normal_share + 1e-9)).clip(-5, 5)
        feats["News_Has_Coverage"] = (vol.shift(1, freq="D").reindex(cal).rolling("3D", min_periods=1).sum() > 0).astype(float)
        known = vol.loc[trading_dates.min():].dropna()
        covered = (known > 0).mean() * 100 if len(known) else 0.0
        info["coverage_pct"] = round(float(covered), 1)
        if covered >= 20:
            info["sources"].append(f'GDELT ("{query}")')
            info["columns"] += NEWS_FEATURE_COLUMNS
        else:
            info["notes"].append(f'GDELT has too little coverage of "{query}" ({covered:.0f}% of days) -- not used.')
    except Exception as e:
        info["notes"].append(f"GDELT unavailable: {e}")

    key = alpha_vantage_key(av_key)
    if key:
        try:
            av = fetch_alpha_vantage_daily(ticker, key, trading_dates.min().date()).reindex(cal)
            if av["av_articles"].notna().sum() >= 60:
                arts = av["av_articles"].fillna(0)
                feats["AV_Sentiment_3d"] = _prior_days_mean(av["av_sentiment"], 3, arts)
                feats["AV_Articles_3d"] = np.log1p(_prior_days_mean(arts, 3) * 3)
                info["sources"].append("Alpha Vantage")
                info["columns"] += AV_FEATURE_COLUMNS
            elif _av_symbol(ticker) is None:
                info["notes"].append("Alpha Vantage news doesn't cover NSE/BSE symbols -- using GDELT only.")
            else:
                info["notes"].append("Alpha Vantage returned too little history yet (free tier fetches ~12 quarters per run).")
        except Exception as e:
            info["notes"].append(f"Alpha Vantage unavailable: {e}")

    own = load_own_history(ticker)
    if len(own) >= OWN_MIN_DAYS:
        feats["Own_Sentiment_3d"] = _prior_days_mean(own["own_sentiment"].reindex(cal), 3)
        info["sources"].append(f"own collected headlines ({len(own)} days)")
        info["columns"] += OWN_FEATURE_COLUMNS
    else:
        info["notes"].append(f"Own daily headline collection: {len(own)}/{OWN_MIN_DAYS} days saved so far "
                             "(used automatically once there's enough history).")

    feats = feats.fillna(0.0)  # no news = neutral
    return feats[info["columns"]] if info["columns"] else feats.iloc[:, :0], info
