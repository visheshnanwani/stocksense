"""
social_media.py
-----------------
What people are saying about a stock RIGHT NOW on social platforms, and
how that feeds the forecasting models.

Sources (all free, no paid API needed):

  * StockTwits  -- the retail-trader message board. Every message can carry
                   an explicit Bullish / Bearish tag chosen by its author,
                   which is a much cleaner signal than guessing sentiment
                   from text. ~30 latest messages per symbol.
  * Reddit      -- r/stocks, r/wallstreetbets, r/investing and the rest, via
                   Reddit's public search RSS feed (the .json endpoints are
                   blocked for unauthenticated clients, the RSS one is not).
  * YouTube     -- the most recent videos matching the stock, read from the
                   `ytInitialData` blob on the search results page: title,
                   channel, age and view count.
  * X / Twitter -- needs a free developer Bearer token (X shut down all
                   unauthenticated access). Pass one and it is used; leave it
                   empty and the platform is simply reported as "not
                   connected" rather than faked.

Text is scored with VADER plus the finance-specific lexicon already used for
news (news_sentiment), so "beat estimates" or "downgrade" count properly.

TWO SEPARATE JOBS -- worth being clear about, because they have very
different guarantees:

  1. LIVE VIEW (social_snapshot)  -- fetched on demand, seconds old, shown on
     the dashboard. Fully real-time.
  2. MODEL INPUT (build_social_features) -- social platforms do NOT publish a
     free historical archive, so there is nothing to backfill. The dashboard
     therefore saves one snapshot per day (collect_today) and a history
     accumulates from the day you start using it. Once MIN_HISTORY_DAYS days
     exist, the forecast engine tests a "social" feature group exactly like
     every other group and keeps it only where it measurably helps. Until
     then the group is reported as having no data -- it is never faked.

Usage:
    from social_media import social_snapshot, collect_today
    snap = social_snapshot("TSLA", "Tesla Inc")
    print(snap["mood"], snap["platforms"]["stocktwits"]["bull_pct"])
"""

import datetime
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from news_sentiment import score_sentiment

CACHE_DIR = Path(__file__).parent / "news_cache"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept-Language": "en-US,en;q=0.9"}

PLATFORMS = ["stocktwits", "reddit", "youtube", "x"]
PLATFORM_LABELS = {"stocktwits": "StockTwits", "reddit": "Reddit", "youtube": "YouTube", "x": "X / Twitter"}

# Model-side settings
MIN_HISTORY_DAYS = 40          # below this there is nothing for the models to learn from
SOCIAL_FEATURE_COLUMNS = ["Soc_Mood_1d", "Soc_Mood_3d", "Soc_Mood_Shift", "Soc_Buzz", "Soc_Bull_Ratio"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ticker_root(ticker: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE', 'BTC-USD' -> 'BTC', 'GC=F' -> 'GC'."""
    t = (ticker or "").upper().strip()
    t = t.split(".")[0].split("=")[0]
    if t.endswith(("-USD", "-INR", "-EUR")):
        t = t.rsplit("-", 1)[0]
    return t.lstrip("^")


def _clean(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _age_hours(ts) -> float:
    if ts is None:
        return np.nan
    return (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(ts)).total_seconds() / 3600


def _parse_relative_age(text: str) -> float:
    """'12h ago' / '1 day ago' / 'Streamed 2 days ago' -> hours."""
    if not text:
        return np.nan
    m = re.search(r"(\d+)\s*(second|minute|hour|day|week|month|year)s?", text, re.I)
    if not m:
        return np.nan
    n, unit = int(m.group(1)), m.group(2).lower()
    per = {"second": 1 / 3600, "minute": 1 / 60, "hour": 1, "day": 24,
           "week": 168, "month": 730, "year": 8760}[unit]
    return n * per


def _views_to_int(text: str) -> float:
    if not text:
        return np.nan
    m = re.search(r"([\d,.]+)\s*([KMB])?", text.replace(" ", " "))
    if not m:
        return np.nan
    val = float(m.group(1).replace(",", ""))
    return val * {"K": 1e3, "M": 1e6, "B": 1e9}.get((m.group(2) or "").upper(), 1)


# --------------------------------------------------------------------------
# fetchers -- each returns a list of posts and never raises
# --------------------------------------------------------------------------
def fetch_stocktwits(ticker: str, limit: int = 30) -> list:
    """Latest StockTwits messages, with the author's own Bullish/Bearish tag."""
    sym = _ticker_root(ticker)
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
    r = requests.get(url, headers=_UA, timeout=12)
    if r.status_code == 404:
        raise RuntimeError("no StockTwits board for this symbol (it lists US-traded symbols)")
    if r.status_code != 200:
        raise RuntimeError(f"StockTwits HTTP {r.status_code}")
    out = []
    for m in (r.json().get("messages") or [])[:limit]:
        body = _clean(m.get("body"))
        tag = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        user = m.get("user") or {}
        out.append({
            "platform": "stocktwits",
            "text": body,
            "url": f"https://stocktwits.com/message/{m.get('id')}",
            "author": user.get("username"),
            "created": pd.to_datetime(m.get("created_at"), errors="coerce", utc=True).tz_localize(None)
            if m.get("created_at") else None,
            "engagement": float((user.get("followers") or 0)),
            "label": tag,                                  # 'Bullish' / 'Bearish' / None
            "score": 0.6 if tag == "Bullish" else -0.6 if tag == "Bearish" else score_sentiment(body),
        })
    return out


# Reddit allows roughly one anonymous request at a time and answers 429 to
# anything faster, so space the calls out and retry once.
_REDDIT_MIN_GAP = 4.0
_reddit_last_call = [0.0]


def fetch_reddit(query: str, limit: int = 25) -> list:
    """Recent Reddit posts about the stock (public search RSS feed -- the
    .json endpoints block anonymous clients)."""
    import xml.etree.ElementTree as ET
    url = ("https://www.reddit.com/search.rss?q=" + requests.utils.quote(query) +
           "&sort=new&t=month&limit=" + str(limit))
    r = None
    for attempt in range(2):
        wait = _REDDIT_MIN_GAP - (time.time() - _reddit_last_call[0])
        if wait > 0:
            time.sleep(min(wait, _REDDIT_MIN_GAP))
        r = requests.get(url, headers=_UA, timeout=15)
        _reddit_last_call[0] = time.time()
        if r.status_code == 200:
            break
        if r.status_code != 429 or attempt == 1:
            raise RuntimeError(f"Reddit HTTP {r.status_code}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.content)
    out = []
    for e in root.findall("a:entry", ns)[:limit]:
        title = _clean(e.findtext("a:title", default="", namespaces=ns))
        link_el = e.find("a:link", ns)
        cat = e.find("a:category", ns)
        when = e.findtext("a:updated", default=None, namespaces=ns)
        out.append({
            "platform": "reddit",
            "text": title,
            "url": link_el.get("href") if link_el is not None else None,
            "author": (cat.get("label") if cat is not None else None)
                      or e.findtext("a:author/a:name", default="", namespaces=ns) or None,
            "created": pd.to_datetime(when, errors="coerce", utc=True).tz_localize(None) if when else None,
            "engagement": np.nan,
            "label": None,
            "score": score_sentiment(title),
        })
    return out


def _walk_videos(node, out):
    if isinstance(node, dict):
        v = node.get("videoRenderer")
        if v:
            title = "".join(r.get("text", "") for r in (v.get("title", {}).get("runs") or [])) \
                    or v.get("title", {}).get("simpleText", "")
            out.append({
                "id": v.get("videoId"), "title": title,
                "channel": ((v.get("ownerText", {}).get("runs") or [{}])[0]).get("text"),
                "published": v.get("publishedTimeText", {}).get("simpleText"),
                "views": v.get("viewCountText", {}).get("simpleText")
                         or v.get("shortViewCountText", {}).get("simpleText"),
            })
        for child in node.values():
            _walk_videos(child, out)
    elif isinstance(node, list):
        for child in node:
            _walk_videos(child, out)


def fetch_youtube(query: str, limit: int = 15) -> list:
    """Most recent YouTube videos about the stock (search page, sorted by upload date)."""
    url = "https://www.youtube.com/results?search_query=" + requests.utils.quote(query) + "&sp=CAI%3D"
    r = requests.get(url, headers=_UA, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"YouTube HTTP {r.status_code}")
    html = r.text
    key = "var ytInitialData = "
    i = html.find(key)
    if i < 0:
        raise RuntimeError("YouTube page layout changed")
    start = i + len(key)
    depth, j = 0, start
    while j < len(html):                       # walk to the matching closing brace
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                j += 1
                break
        j += 1
    vids = []
    _walk_videos(json.loads(html[start:j]), vids)
    out = []
    for v in vids[:limit]:
        title = _clean(v["title"])
        age = _parse_relative_age(v.get("published"))
        out.append({
            "platform": "youtube",
            "text": title,
            "url": f"https://www.youtube.com/watch?v={v['id']}" if v.get("id") else None,
            "author": v.get("channel"),
            "created": (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=age))
            if not np.isnan(age) else None,
            "engagement": _views_to_int(v.get("views")),
            "label": None,
            "score": score_sentiment(title),
        })
    return out


# X charges per post read (about $0.005 each) and has had no free tier since
# February 2026, so results are reused for X_CACHE_MINUTES instead of being
# re-fetched on every refresh of the panel.
X_CACHE_MINUTES = 60
X_DEFAULT_LIMIT = 10
X_COST_PER_READ = 0.005          # USD, X's published pay-per-use rate
_x_cache = {"key": None, "ts": 0.0, "posts": []}


def reset_x_cache():
    """Force the next snapshot to pay for a fresh X fetch."""
    _x_cache.update(key=None, ts=0.0, posts=[])


def x_cost_note(limit: int = X_DEFAULT_LIMIT) -> str:
    """Plain-English running cost, so nobody is surprised by their X bill."""
    per_day = (24 * 60 / X_CACHE_MINUTES) * limit * X_COST_PER_READ
    return (f"X bills about ${X_COST_PER_READ:.3f} per post read and has no free tier. This dashboard asks "
            f"for {limit} posts at most once an hour, so a full day of it costs roughly ${per_day:.2f} — "
            f"the other three platforms stay free.")


def fetch_x(query: str, bearer: str, limit: int = X_DEFAULT_LIMIT) -> list:
    """Recent posts from X. Needs a developer Bearer token from console.x.com --
    X removed free API access in February 2026, so reads are billed per post."""
    if not bearer:
        raise RuntimeError("no token")
    key = (query, bearer[-8:], limit)
    fresh = time.time() - _x_cache["ts"] < X_CACHE_MINUTES * 60
    if _x_cache["key"] == key and fresh:
        return list(_x_cache["posts"])     # reuse: every extra call is real money
    url = "https://api.x.com/2/tweets/search/recent"
    params = {"query": f"({query}) -is:retweet lang:en", "max_results": max(10, min(limit, 100)),
              "tweet.fields": "created_at,public_metrics"}
    r = requests.get(url, headers={"Authorization": f"Bearer {bearer}", **_UA}, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"X HTTP {r.status_code}")
    out = []
    for t in (r.json().get("data") or [])[:limit]:
        text = _clean(t.get("text"))
        pm = t.get("public_metrics") or {}
        out.append({
            "platform": "x",
            "text": text,
            "url": f"https://x.com/i/web/status/{t.get('id')}",
            "author": None,
            "created": pd.to_datetime(t.get("created_at"), errors="coerce", utc=True).tz_localize(None)
            if t.get("created_at") else None,
            "engagement": float(pm.get("like_count", 0) + pm.get("retweet_count", 0)),
            "label": None,
            "score": score_sentiment(text),
        })
    _x_cache.update(key=key, ts=time.time(), posts=list(out))
    return out


def x_bearer_token(explicit: str = None) -> str:
    """Token from the dashboard field, or the X_BEARER_TOKEN environment variable."""
    import os
    return (explicit or "").strip() or os.environ.get("X_BEARER_TOKEN", "").strip()


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def social_query(ticker: str, company_name: str = None) -> str:
    """What to search for: the plain company name plus the cashtag."""
    root = _ticker_root(ticker)
    name = re.sub(r"\b(limited|ltd|inc|incorporated|corporation|corp|company|co|plc|holdings?|group)\b\.?",
                  "", (company_name or ""), flags=re.I).strip(" .,-")
    name = re.sub(r"\s+", " ", name)
    return f"{name} {root} stock".strip() if name and name.lower() != root.lower() else f"{root} stock"


# Words that show a post is about the market rather than the plain English
# word in a company's name ("self-reliance", "apple pie", "ITC hotel review").
_MARKET_CUES = ("stock", "share", "shares", "buy", "sell", "sold", "bought", "market", "price",
                "target", "invest", "investor", "investing", "portfolio", "rally", "crash", "dip",
                "earnings", "results", "quarter", "profit", "loss", "nse", "bse", "nifty", "sensex",
                "nasdaq", "trading", "trade", "bullish", "bearish", "calls", "puts", "chart",
                "analysis", "forecast", "valuation", "dividend", "ipo", "breakout")


def _is_relevant(text: str, ticker: str, company_name: str = None) -> bool:
    """Reddit and YouTube search are fuzzy, so keep only posts that actually
    discuss this stock: the symbol itself, or the company name together with
    a market word."""
    t = (text or "").lower()
    if not t:
        return False
    root = _ticker_root(ticker).lower()
    # (?<![\w-]) so "self-reliance" does not count as a RELIANCE mention
    if len(root) >= 3 and re.search(r"(?<![\w-])\$?" + re.escape(root) + r"(?![\w-])", t):
        return True
    words = [w for w in re.split(r"[^a-z0-9]+", (company_name or "").lower())
             if len(w) > 3 and w not in ("limited", "corporation", "company", "holdings", "group",
                                         "industries", "technologies", "international", "motors")]
    named = any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in words[:3])
    return named and any(c in t for c in _MARKET_CUES)


def _platform_metrics(posts: list) -> dict:
    """Mood, volume and bull/bear split for one platform's posts."""
    if not posts:
        return {"count": 0, "mood": None, "bull_pct": None, "bear_pct": None, "recent_24h": 0,
                "engagement": 0.0}
    scores = [p["score"] for p in posts if p.get("score") is not None]
    labelled = [p["label"] for p in posts if p.get("label")]
    bull = sum(1 for l in labelled if l == "Bullish")
    bear = sum(1 for l in labelled if l == "Bearish")
    # when authors don't tag their posts, fall back to the text score
    if not labelled:
        bull = sum(1 for s in scores if s > 0.05)
        bear = sum(1 for s in scores if s < -0.05)
    total_tagged = max(bull + bear, 1)
    ages = [_age_hours(p.get("created")) for p in posts]
    return {
        "count": len(posts),
        "mood": float(np.mean(scores)) if scores else None,
        "bull_pct": 100.0 * bull / total_tagged,
        "bear_pct": 100.0 * bear / total_tagged,
        "recent_24h": int(sum(1 for a in ages if not np.isnan(a) and a <= 24)),
        "engagement": float(np.nansum([p.get("engagement", np.nan) for p in posts])),
        "newest": min([a for a in ages if not np.isnan(a)], default=None),
    }


def mood_label(score) -> str:
    if score is None:
        return "No data"
    if score >= 0.25:
        return "Very bullish"
    if score >= 0.05:
        return "Bullish"
    if score <= -0.25:
        return "Very bearish"
    if score <= -0.05:
        return "Bearish"
    return "Neutral / mixed"


def social_snapshot(ticker: str, company_name: str = None, x_bearer: str = None, limit: int = 25) -> dict:
    """
    Live picture of the social conversation about `ticker`.

    Returns {platforms: {name: metrics}, posts: [...], mood, mood_label,
             total_posts, sources, errors, fetched_at}.
    Every platform fails soft: a dead source is reported in `errors`,
    it never breaks the others.
    """
    query = social_query(ticker, company_name)
    jobs = {
        "stocktwits": lambda: fetch_stocktwits(ticker, limit),
        "reddit": lambda: fetch_reddit(query, limit),
        "youtube": lambda: fetch_youtube(query, min(limit, 15)),
        "x": lambda: fetch_x(query, x_bearer_token(x_bearer), X_DEFAULT_LIMIT),
    }
    posts, platforms, errors, sources = [], {}, {}, []
    for name, job in jobs.items():
        try:
            got = job()
        except Exception as e:
            platforms[name] = _platform_metrics([])
            errors[name] = str(e)
            continue
        if name in ("reddit", "youtube"):
            got = [p for p in got if _is_relevant(p["text"], ticker, company_name)]
        posts += got
        platforms[name] = _platform_metrics(got)
        if got:
            sources.append(name)

    scored = [p["score"] for p in posts if p.get("score") is not None]
    # Platforms are averaged with equal weight so one chatty source can't
    # drown out the others, then weighted by how many posts each carries.
    per_platform = [(platforms[p]["mood"], platforms[p]["count"]) for p in platforms
                    if platforms[p]["mood"] is not None]
    mood = (float(np.average([m for m, _ in per_platform],
                             weights=[np.sqrt(c) for _, c in per_platform]))
            if per_platform else None)
    bulls = [platforms[p]["bull_pct"] for p in platforms if platforms[p]["bull_pct"] is not None]

    return {
        "ticker": ticker,
        "query": query,
        "platforms": platforms,
        "posts": sorted(posts, key=lambda p: (p.get("created") is not None, p.get("created")), reverse=True),
        "mood": mood,
        "mood_label": mood_label(mood),
        "bull_pct": float(np.mean(bulls)) if bulls else None,
        "total_posts": len(posts),
        "scored_posts": len(scored),
        "sources": sources,
        "errors": errors,
        "fetched_at": datetime.datetime.now(),
    }


# --------------------------------------------------------------------------
# daily history -> model features
# --------------------------------------------------------------------------
def _hist_path(ticker: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ticker.upper()).strip("_")
    return CACHE_DIR / f"social_{slug}.csv"


def collect_today(ticker: str, snap: dict) -> None:
    """
    Save one row per day so a social history builds up from today onwards.
    Re-running on the same day refreshes that day's row.
    """
    if not snap or not snap.get("total_posts"):
        return
    CACHE_DIR.mkdir(exist_ok=True)
    path = _hist_path(ticker)
    df = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    today = pd.Timestamp(datetime.date.today())
    df.loc[today, "mood"] = snap.get("mood")
    df.loc[today, "posts"] = snap.get("total_posts")
    df.loc[today, "bull_pct"] = snap.get("bull_pct")
    for p in PLATFORMS:
        df.loc[today, f"{p}_posts"] = (snap.get("platforms", {}).get(p) or {}).get("count", 0)
    df.index.name = "date"
    df.sort_index().to_csv(path)


def load_history(ticker: str) -> pd.DataFrame:
    path = _hist_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=["mood", "posts", "bull_pct"])
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def build_social_features(ticker: str, trading_dates: pd.DatetimeIndex):
    """
    Model-ready daily social features, or (None, info) when too little history
    has been collected. Values use only data from BEFORE each trading day, so
    there is no look-ahead.
    """
    hist = load_history(ticker)
    info = {"days": int(len(hist)), "min_days": MIN_HISTORY_DAYS, "columns": [], "usable": False,
            "first_day": hist.index.min() if len(hist) else None}
    if len(hist) < MIN_HISTORY_DAYS:
        info["note"] = (f"{len(hist)} day(s) of social history saved so far; the models start testing it "
                        f"at {MIN_HISTORY_DAYS} days. Social platforms publish no free archive, so this "
                        f"history can only grow forwards from the first day you ran the dashboard.")
        return None, info

    cal = pd.date_range(min(trading_dates.min(), hist.index.min()),
                        max(trading_dates.max(), hist.index.max()) + pd.Timedelta(days=1), freq="D")
    h = hist.reindex(cal)
    mood, posts, bull = h["mood"], h["posts"], h["bull_pct"]

    feats = pd.DataFrame(index=cal)
    prior = mood.shift(1)                                   # yesterday's reading, never today's
    feats["Soc_Mood_1d"] = prior
    feats["Soc_Mood_3d"] = prior.rolling(3, min_periods=1).mean()
    feats["Soc_Mood_Shift"] = feats["Soc_Mood_3d"] - prior.rolling(14, min_periods=3).mean()
    vol = posts.shift(1)
    feats["Soc_Buzz"] = np.log((vol + 1) / (vol.rolling(30, min_periods=5).mean() + 1)).clip(-3, 3)
    feats["Soc_Bull_Ratio"] = (bull.shift(1) - 50.0) / 50.0

    feats = feats.ffill(limit=5)                            # a missed day keeps the last reading briefly
    out = feats.reindex(trading_dates)
    keep = [c for c in SOCIAL_FEATURE_COLUMNS if out[c].notna().mean() > 0.25]
    if not keep:
        info["note"] = "social history is too sparse to use"
        return None, info
    info.update({"columns": keep, "usable": True,
                 "coverage_pct": round(100 * float(out[keep].notna().all(axis=1).mean()), 1),
                 "note": f"{len(hist)} days of social history collected since "
                         f"{hist.index.min():%d %b %Y}."})
    return out[keep], info


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(errors="replace")
    t = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    snap = social_snapshot(t, sys.argv[2] if len(sys.argv) > 2 else None)
    mood = "n/a" if snap["mood"] is None else f"{snap['mood']:+.3f}"
    print(f"{t}: {snap['total_posts']} posts, mood {mood} ({snap['mood_label']})")
    for name, m in snap["platforms"].items():
        pm = "n/a" if m["mood"] is None else f"{m['mood']:+.2f}"
        bull = "n/a" if m["bull_pct"] is None else f"{m['bull_pct']:.0f}% bull"
        print(f"  {PLATFORM_LABELS[name]:12} {m['count']:3} posts  mood={pm}  {bull}")
    for post in snap["posts"][:6]:
        print(" -", post["platform"], "|", (post["text"] or "")[:78])
    if snap["errors"]:
        print("errors:", snap["errors"])
