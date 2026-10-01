"""
collect_news.py
-----------------
Daily news collector (run once a day, e.g. with Windows Task Scheduler).
For each stock in your watchlist it:
  1. scores today's headlines (Yahoo Finance + Google News) and saves the
     day's sentiment to news_cache/daily_<ticker>.csv -- your OWN news
     history, which the models start using automatically once ~120 days
     have been collected;
  2. tops up the cached GDELT history with the latest days, so the
     dashboard doesn't have to download it when you open it.

The dashboard also saves today's headlines whenever you open a stock; this
script just makes sure it happens every day, even when you don't.

Run:
    python collect_news.py                      # uses WATCHLIST below
    python collect_news.py MSFT RELIANCE.NS     # or pass tickers

Schedule it daily on Windows (run once in a terminal, adjust the paths):
    schtasks /Create /SC DAILY /ST 18:30 /TN "StockNewsCollector" ^
      /TR "python C:\\path\\to\\stock_project\\collect_news.py"
"""

import datetime
import sys

import news_history as nh
import news_sentiment as ns
from stock_search import get_company_profile

WATCHLIST = ["MSFT", "AAPL", "RELIANCE.NS", "TCS.NS"]


def collect(ticker: str) -> None:
    name = get_company_profile(ticker).get("name", ticker)
    query = nh.company_query(name)
    summary = ns.get_news_sentiment_summary(ticker, company_query=query)
    nh.collect_today(ticker, summary)
    days = len(nh.load_own_history(ticker))
    print(f"{ticker:<14} {summary['article_count']:>3} headlines, sentiment {summary['average_score']:+.3f} "
          f"({summary['label']}) -> {days} day(s) collected")
    try:
        g = nh.fetch_gdelt_daily(query, datetime.date(2017, 1, 1), time_budget=120)
        print(f"{'':<14} GDELT history cached through {g.index.max().date()}")
    except Exception as e:
        print(f"{'':<14} GDELT not updated this time ({e})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")  # Windows consoles vs. non-ASCII headlines
    except Exception:
        pass
    for t in (sys.argv[1:] or WATCHLIST):
        collect(t)
