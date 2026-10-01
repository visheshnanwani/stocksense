"""
news_sentiment.py
--------------------
Fetches CURRENT news headlines for a stock (via yfinance's built-in
news feed -- free, no API key needed) and scores their sentiment using
VADER (a lightweight, rule-based sentiment analyzer well-suited to
short text like headlines, with no GPU or model download required).

IMPORTANT LIMITATION -- read this before using it for training:
yfinance's news feed only returns CURRENT/recent articles, not a
historical archive. This means news sentiment can be shown as a LIVE,
real-time signal in the dashboard (genuinely useful for "what's the
sentiment right now"), but it CANNOT be backfilled into years of
historical training data for main.py/ensemble.py/train_advanced.py --
there's no free source for that. If you wanted true historical news
sentiment as a training feature, you'd need a paid/rate-limited API
(e.g. Alpha Vantage's NEWS_SENTIMENT endpoint has a free tier with a
public API key and some historical coverage, or a paid provider like
Benzinga/NewsAPI). This module deliberately keeps things free and
simple by treating news sentiment as a live overlay, not a trained
feature -- documented clearly so you understand the boundary rather
than assuming it's doing more than it is.

Usage:
    from news_sentiment import get_news_sentiment_summary
    summary = get_news_sentiment_summary("MSFT")
"""

import datetime
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# VADER is tuned for general/social-media text and misses common finance
# vocabulary (e.g. it doesn't know "tumble" or "antitrust" are negative
# in a stock-news context). This small boost lexicon nudges scores for
# words that recur constantly in financial headlines. It's a heuristic
# patch, not a substitute for a proper finance-tuned model (like FinBERT)
# -- worth mentioning as a limitation if you go deeper on this.
_FINANCE_LEXICON_BOOST = {
    "surge": 2.5, "surges": 2.5, "soar": 2.7, "soars": 2.7, "rally": 2.0, "rallies": 2.0,
    "beat": 1.8, "beats": 1.8, "outperform": 2.0, "upgrade": 2.2, "upgraded": 2.2,
    "bullish": 2.3, "buyback": 1.2, "record high": 2.0, "profit": 1.3, "growth": 1.3,
    "tumble": -2.5, "tumbles": -2.5, "plunge": -3.0, "plunges": -3.0, "slump": -2.3,
    "miss": -1.8, "misses": -1.8, "downgrade": -2.2, "downgraded": -2.2,
    "bearish": -2.3, "lawsuit": -1.7, "antitrust": -1.5, "layoffs": -2.2, "layoff": -2.2,
    "bankruptcy": -3.5, "recall": -1.5, "probe": -1.5, "investigation": -1.5,
    "fraud": -3.0, "scandal": -2.7, "loss": -1.5, "losses": -1.5, "decline": -1.5,
    "cut": -1.2, "cuts": -1.2, "warns": -1.3, "warning": -1.3,
}
_analyzer.lexicon.update(_FINANCE_LEXICON_BOOST)


def fetch_latest_news(ticker: str, max_articles: int = 15):
    """
    Fetch recent news headlines for a ticker via yfinance.

    Returns a list of dicts: {title, publisher, link, publish_time}
    Returns an empty list (not an error) if no news is available or
    the fetch fails -- callers should handle an empty list gracefully,
    since news availability varies a lot by ticker and isn't guaranteed.
    """
    try:
        raw_news = yf.Ticker(ticker).news
    except Exception:
        return []

    if not raw_news:
        return []

    articles = []
    for item in raw_news[:max_articles]:
        # yfinance's news schema has shifted between versions; handle both
        # a flat dict and a nested {"content": {...}} shape defensively.
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else item.get("publisher")
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else item.get("link")
        )
        pub_date = content.get("pubDate") or item.get("providerPublishTime")

        if title:
            articles.append({
                "title": title,
                "publisher": publisher or "Unknown",
                "link": link or "",
                "publish_time": pub_date,
            })

    return articles


def fetch_google_news(query: str, max_articles: int = 25, indian: bool = False):
    """
    Headlines from Google News RSS for `query` -- one free feed that
    aggregates hundreds of sites (Economic Times, Moneycontrol, Reuters,
    CNBC, Business Standard, ...), so the live sentiment isn't limited to
    Yahoo's handful of articles. Current headlines only (no archive).
    Returns the same dict shape as fetch_latest_news(); [] on any failure.
    """
    region = "hl=en-IN&gl=IN&ceid=IN:en" if indian else "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={quote(f'{chr(34)}{query}{chr(34)} stock')}&{region}"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
    except Exception:
        return []
    articles = []
    for item in root.iter("item"):
        title = unescape(item.findtext("title") or "")
        source = item.findtext("source") or ""
        # Google appends " - Publisher" to titles; strip it so it doesn't skew sentiment
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        try:
            published = parsedate_to_datetime(item.findtext("pubDate")).isoformat()
        except Exception:
            published = None
        if title:
            articles.append({"title": title, "publisher": source or "Google News",
                             "link": item.findtext("link") or "", "publish_time": published})
        if len(articles) >= max_articles:
            break
    return articles


def score_sentiment(text: str) -> float:
    """VADER compound sentiment score for one piece of text, from -1 (very
    negative) to +1 (very positive)."""
    return _analyzer.polarity_scores(text)["compound"]


def label_for_score(score: float) -> str:
    """Human-readable label matching VADER's own conventional thresholds."""
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def get_news_sentiment_summary(ticker: str, max_articles: int = 15, company_query: str = None):
    """
    Fetches recent news and returns an aggregated sentiment summary.

    Returns
    -------
    dict with:
        average_score   : float, mean VADER compound score across articles
        label           : str, "Positive"/"Negative"/"Neutral" for the average
        article_count   : int, how many articles were scored
        articles        : list of dicts, each with title/publisher/link/score/label,
                          sorted most recent first
        fetched_at      : timestamp of when this summary was generated
    Returns article_count=0 and a neutral/unavailable summary if no
    news could be fetched -- this is a normal, expected outcome for
    some tickers, not a failure to surface as an error to the user.
    """
    articles = fetch_latest_news(ticker, max_articles=max_articles)
    if company_query:  # add many more sources via Google News, de-duplicated by headline
        seen = {a["title"].strip().lower() for a in articles}
        indian = ticker.upper().endswith((".NS", ".BO"))
        for a in fetch_google_news(company_query, max_articles=25, indian=indian):
            key = a["title"].strip().lower()
            if key not in seen:
                articles.append(a)
                seen.add(key)

    if not articles:
        return {
            "average_score": 0.0,
            "label": "Unavailable",
            "article_count": 0,
            "articles": [],
            "fetched_at": datetime.datetime.now(),
        }

    scored_articles = []
    for a in articles:
        score = score_sentiment(a["title"])
        scored_articles.append({**a, "score": score, "label": label_for_score(score)})

    average_score = sum(a["score"] for a in scored_articles) / len(scored_articles)

    return {
        "average_score": round(average_score, 3),
        "label": label_for_score(average_score),
        "article_count": len(scored_articles),
        "articles": scored_articles,
        "fetched_at": datetime.datetime.now(),
    }


if __name__ == "__main__":
    summary = get_news_sentiment_summary("MSFT")
    print(f"Average sentiment: {summary['average_score']} ({summary['label']})")
    print(f"Based on {summary['article_count']} recent articles\n")
    for a in summary["articles"]:
        print(f"  [{a['label']:>8} {a['score']:+.2f}] {a['title']} ({a['publisher']})")
