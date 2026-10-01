"""
stock_search.py
-----------------
Lets the user type a company NAME (e.g. "reliance", "micro", "tata
motors") instead of having to know the exact ticker symbol, and
suggests real, matching stocks as they type.

Search order:
  1. Yahoo Finance's live symbol search (via yfinance.Search) -- covers
     every exchange yfinance can download from (NYSE, NASDAQ, NSE, BSE,
     LSE, ...), so the suggestions are always real, tradable tickers.
  2. A small built-in list of popular US + Indian stocks, used as an
     OFFLINE fallback (no internet / Yahoo rate-limits) and to make the
     first few keystrokes feel instant.

Used by:
  - dashboard.py  -> live type-ahead search box in the sidebar
  - every CLI script (main.py, ensemble.py, ...) via prompt_for_ticker(),
    which asks the user for a stock name in the terminal.

Usage:
    from stock_search import search_stocks, prompt_for_ticker
    search_stocks("reliance")      # -> list of dicts
    ticker = prompt_for_ticker()   # interactive terminal prompt
"""

import functools

import yfinance as yf

# Instruments we can meaningfully run the price models on. Anything with a
# daily OHLC history works: shares, funds, indices, commodity futures,
# currency pairs and crypto all behave the same way for the models.
_ALLOWED_TYPES = {"EQUITY", "ETF", "INDEX", "CURRENCY", "FUTURE", "COMMODITY",
                  "CRYPTOCURRENCY", "MUTUALFUND"}

# quoteType (Yahoo) / symbol shape -> the asset class this project uses.
_QUOTE_TYPE_CLASS = {"EQUITY": "stock", "ETF": "etf", "MUTUALFUND": "fund", "INDEX": "index",
                     "CURRENCY": "forex", "FUTURE": "commodity", "COMMODITY": "commodity",
                     "CRYPTOCURRENCY": "crypto"}

# Metals quoted as FX pairs (XAUUSD=X) are commodities, not currencies.
_METAL_CODES = ("XAU", "XAG", "XPT", "XPD")
_CRYPTO_QUOTES = ("-USD", "-INR", "-EUR", "-GBP", "-USDT", "-BTC")

ASSET_META = {
    "stock":     ("Stock", "🏢"),
    "etf":       ("ETF", "📦"),
    "fund":      ("Fund", "🏦"),
    "index":     ("Index", "📊"),
    "commodity": ("Commodity", "🥇"),
    "forex":     ("Forex", "💱"),
    "crypto":    ("Crypto", "🪙"),
}


def asset_class(symbol: str, quote_type: str = None) -> str:
    """
    Classify an instrument: stock / etf / fund / index / commodity / forex / crypto.

    Yahoo's quoteType is used when we have it; otherwise the symbol shape
    tells us ("GC=F" future, "EURUSD=X" pair, "^NSEI" index, "BTC-USD" crypto).
    """
    q = (quote_type or "").upper()
    cls = _QUOTE_TYPE_CLASS.get(q)
    s = (symbol or "").strip().upper()
    # Yahoo labels Indian ETFs (GOLDBEES.NS, NIFTYBEES.NS, ...) as plain
    # EQUITY, so our own catalog wins over quoteType where we know better.
    if s in ASSET_CLASSES:
        return ASSET_CLASSES[s]
    if s.endswith("=X"):
        return "commodity" if s.startswith(_METAL_CODES) else "forex"
    if s.endswith("=F"):
        return "commodity"
    if s.startswith("^"):
        return "index"
    if any(s.endswith(suf) for suf in _CRYPTO_QUOTES) and len(s.split("-")[0]) >= 2 and s.split("-")[0].isalpha():
        # BTC-USD yes, BRK-B no (B is not a quote currency)
        return "crypto"
    return cls or "stock"


def asset_label(symbol: str, quote_type: str = None) -> str:
    """'🥇 Commodity' style label for the header chip."""
    name, icon = ASSET_META.get(asset_class(symbol, quote_type), ASSET_META["stock"])
    return f"{icon} {name}"


def is_company(symbol: str, quote_type: str = None) -> bool:
    """True only for shares of an actual company (fundamentals exist)."""
    return asset_class(symbol, quote_type) == "stock"

# Offline fallback / instant suggestions. (symbol, name, exchange)
POPULAR_STOCKS = [
    # --- US ---
    ("AAPL", "Apple Inc.", "NASDAQ"),
    ("MSFT", "Microsoft Corporation", "NASDAQ"),
    ("GOOGL", "Alphabet Inc. (Google) Class A", "NASDAQ"),
    ("AMZN", "Amazon.com, Inc.", "NASDAQ"),
    ("META", "Meta Platforms, Inc. (Facebook)", "NASDAQ"),
    ("NVDA", "NVIDIA Corporation", "NASDAQ"),
    ("TSLA", "Tesla, Inc.", "NASDAQ"),
    ("NFLX", "Netflix, Inc.", "NASDAQ"),
    ("AMD", "Advanced Micro Devices, Inc.", "NASDAQ"),
    ("INTC", "Intel Corporation", "NASDAQ"),
    ("ORCL", "Oracle Corporation", "NYSE"),
    ("IBM", "International Business Machines", "NYSE"),
    ("JPM", "JPMorgan Chase & Co.", "NYSE"),
    ("BAC", "Bank of America Corporation", "NYSE"),
    ("V", "Visa Inc.", "NYSE"),
    ("MA", "Mastercard Incorporated", "NYSE"),
    ("WMT", "Walmart Inc.", "NYSE"),
    ("KO", "The Coca-Cola Company", "NYSE"),
    ("PEP", "PepsiCo, Inc.", "NASDAQ"),
    ("DIS", "The Walt Disney Company", "NYSE"),
    ("XOM", "Exxon Mobil Corporation", "NYSE"),
    ("JNJ", "Johnson & Johnson", "NYSE"),
    ("PFE", "Pfizer Inc.", "NYSE"),
    ("BRK-B", "Berkshire Hathaway Inc. Class B", "NYSE"),
    ("SPY", "SPDR S&P 500 ETF Trust", "NYSE Arca"),
    ("QQQ", "Invesco QQQ Trust (Nasdaq-100)", "NASDAQ"),
    # --- India (NSE) ---
    ("RELIANCE.NS", "Reliance Industries Ltd", "NSE"),
    ("TCS.NS", "Tata Consultancy Services Ltd", "NSE"),
    ("INFY.NS", "Infosys Ltd", "NSE"),
    ("HDFCBANK.NS", "HDFC Bank Ltd", "NSE"),
    ("ICICIBANK.NS", "ICICI Bank Ltd", "NSE"),
    ("SBIN.NS", "State Bank of India", "NSE"),
    ("HINDUNILVR.NS", "Hindustan Unilever Ltd", "NSE"),
    ("ITC.NS", "ITC Ltd", "NSE"),
    ("BHARTIARTL.NS", "Bharti Airtel Ltd", "NSE"),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank Ltd", "NSE"),
    ("LT.NS", "Larsen & Toubro Ltd", "NSE"),
    ("AXISBANK.NS", "Axis Bank Ltd", "NSE"),
    ("WIPRO.NS", "Wipro Ltd", "NSE"),
    ("HCLTECH.NS", "HCL Technologies Ltd", "NSE"),
    ("TECHM.NS", "Tech Mahindra Ltd", "NSE"),
    ("MARUTI.NS", "Maruti Suzuki India Ltd", "NSE"),
    ("TMPV.NS", "Tata Motors Passenger Vehicles Ltd", "NSE"),
    ("TMCV.NS", "Tata Motors Ltd (Commercial Vehicles)", "NSE"),
    ("TATASTEEL.NS", "Tata Steel Ltd", "NSE"),
    ("TATAPOWER.NS", "Tata Power Company Ltd", "NSE"),
    ("M&M.NS", "Mahindra & Mahindra Ltd", "NSE"),
    ("BAJFINANCE.NS", "Bajaj Finance Ltd", "NSE"),
    ("ASIANPAINT.NS", "Asian Paints Ltd", "NSE"),
    ("SUNPHARMA.NS", "Sun Pharmaceutical Industries Ltd", "NSE"),
    ("ADANIENT.NS", "Adani Enterprises Ltd", "NSE"),
    ("ADANIPORTS.NS", "Adani Ports and SEZ Ltd", "NSE"),
    ("ONGC.NS", "Oil and Natural Gas Corporation Ltd", "NSE"),
    ("NTPC.NS", "NTPC Ltd", "NSE"),
    ("POWERGRID.NS", "Power Grid Corporation of India Ltd", "NSE"),
    ("TITAN.NS", "Titan Company Ltd", "NSE"),
    ("ULTRACEMCO.NS", "UltraTech Cement Ltd", "NSE"),
    ("ETERNAL.NS", "Eternal Ltd (Zomato)", "NSE"),
    ("^NSEI", "NIFTY 50 Index", "NSE"),
    ("^BSESN", "S&P BSE SENSEX Index", "BSE"),
]

# Everything that is not a share: metals, energy, agri, currency pairs,
# crypto, ETFs and indices. Same (symbol, name, exchange) shape, plus the
# asset class so the offline suggestions are labelled correctly.
POPULAR_ASSETS = [
    # --- precious & industrial metals (futures + spot) ---
    # XAUUSD=X and XAGUSD=X (gold/silver spot) were removed: the provider
    # delisted them and they return no data at all. GC=F and SI=F cover the same
    # markets. Verified by test_catalog.py.
    ("GC=F", "Gold (COMEX futures, $/oz)", "COMEX", "commodity"),
    ("SI=F", "Silver (COMEX futures, $/oz)", "COMEX", "commodity"),
    ("PL=F", "Platinum (futures, $/oz)", "NYMEX", "commodity"),
    ("PA=F", "Palladium (futures, $/oz)", "NYMEX", "commodity"),
    ("HG=F", "Copper (futures, $/lb)", "COMEX", "commodity"),
    ("ALI=F", "Aluminium (futures)", "COMEX", "commodity"),
    # --- energy ---
    ("CL=F", "Crude Oil WTI (futures, $/bbl)", "NYMEX", "commodity"),
    ("BZ=F", "Brent Crude Oil (futures, $/bbl)", "ICE", "commodity"),
    ("NG=F", "Natural Gas (futures)", "NYMEX", "commodity"),
    ("RB=F", "RBOB Gasoline (futures)", "NYMEX", "commodity"),
    # --- agriculture ---
    ("ZC=F", "Corn (futures)", "CBOT", "commodity"),
    ("ZW=F", "Wheat (futures)", "CBOT", "commodity"),
    ("ZS=F", "Soybeans (futures)", "CBOT", "commodity"),
    ("KC=F", "Coffee (futures)", "ICE", "commodity"),
    ("SB=F", "Sugar (futures)", "ICE", "commodity"),
    ("CT=F", "Cotton (futures)", "ICE", "commodity"),
    # --- currency pairs ---
    ("EURUSD=X", "Euro / US Dollar", "FX", "forex"),
    ("GBPUSD=X", "British Pound / US Dollar", "FX", "forex"),
    ("USDJPY=X", "US Dollar / Japanese Yen", "FX", "forex"),
    ("USDINR=X", "US Dollar / Indian Rupee", "FX", "forex"),
    ("EURINR=X", "Euro / Indian Rupee", "FX", "forex"),
    ("GBPINR=X", "British Pound / Indian Rupee", "FX", "forex"),
    ("JPYINR=X", "Japanese Yen / Indian Rupee", "FX", "forex"),
    ("AUDUSD=X", "Australian Dollar / US Dollar", "FX", "forex"),
    ("USDCAD=X", "US Dollar / Canadian Dollar", "FX", "forex"),
    ("USDCHF=X", "US Dollar / Swiss Franc", "FX", "forex"),
    ("USDCNY=X", "US Dollar / Chinese Yuan", "FX", "forex"),
    ("DX-Y.NYB", "US Dollar Index (DXY)", "ICE", "index"),
    # --- crypto ---
    ("BTC-USD", "Bitcoin / USD", "CCC", "crypto"),
    ("ETH-USD", "Ethereum / USD", "CCC", "crypto"),
    ("BNB-USD", "BNB / USD", "CCC", "crypto"),
    ("XRP-USD", "XRP / USD", "CCC", "crypto"),
    ("SOL-USD", "Solana / USD", "CCC", "crypto"),
    ("ADA-USD", "Cardano / USD", "CCC", "crypto"),
    ("DOGE-USD", "Dogecoin / USD", "CCC", "crypto"),
    ("BTC-INR", "Bitcoin / INR", "CCC", "crypto"),
    # --- US / global ETFs ---
    ("SPY", "SPDR S&P 500 ETF Trust", "NYSE Arca", "etf"),
    ("QQQ", "Invesco QQQ Trust (Nasdaq-100)", "NASDAQ", "etf"),
    ("VOO", "Vanguard S&P 500 ETF", "NYSE Arca", "etf"),
    ("VTI", "Vanguard Total Stock Market ETF", "NYSE Arca", "etf"),
    ("IWM", "iShares Russell 2000 ETF", "NYSE Arca", "etf"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF", "NYSE Arca", "etf"),
    ("GLD", "SPDR Gold Shares (gold ETF)", "NYSE Arca", "etf"),
    ("IAU", "iShares Gold Trust", "NYSE Arca", "etf"),
    ("SLV", "iShares Silver Trust (silver ETF)", "NYSE Arca", "etf"),
    ("USO", "United States Oil Fund (crude ETF)", "NYSE Arca", "etf"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "NASDAQ", "etf"),
    ("ARKK", "ARK Innovation ETF", "NYSE Arca", "etf"),
    ("EEM", "iShares MSCI Emerging Markets ETF", "NYSE Arca", "etf"),
    ("INDA", "iShares MSCI India ETF", "NYSE Arca", "etf"),
    ("XLK", "Technology Select Sector SPDR", "NYSE Arca", "etf"),
    ("XLF", "Financial Select Sector SPDR", "NYSE Arca", "etf"),
    ("XLE", "Energy Select Sector SPDR", "NYSE Arca", "etf"),
    # --- Indian ETFs (NSE) ---
    ("NIFTYBEES.NS", "Nippon India NIFTY 50 ETF", "NSE", "etf"),
    ("GOLDBEES.NS", "Nippon India Gold ETF", "NSE", "etf"),
    ("SILVERBEES.NS", "Nippon India Silver ETF", "NSE", "etf"),
    ("BANKBEES.NS", "Nippon India Nifty Bank ETF", "NSE", "etf"),
    ("JUNIORBEES.NS", "Nippon India Nifty Next 50 ETF", "NSE", "etf"),
    ("ITBEES.NS", "Nippon India Nifty IT ETF", "NSE", "etf"),
    ("MON100.NS", "Motilal Oswal Nasdaq 100 ETF", "NSE", "etf"),
    ("MAFANG.NS", "Mirae Asset NYSE FANG+ ETF", "NSE", "etf"),
    ("HDFCGOLD.NS", "HDFC Gold ETF", "NSE", "etf"),
    ("SETFGOLD.NS", "SBI Gold ETF", "NSE", "etf"),
    # --- global indices ---
    ("^GSPC", "S&P 500 Index", "US", "index"),
    ("^IXIC", "NASDAQ Composite Index", "US", "index"),
    ("^DJI", "Dow Jones Industrial Average", "US", "index"),
    ("^NSEBANK", "NIFTY Bank Index", "NSE", "index"),
    ("^VIX", "CBOE Volatility Index (VIX)", "CBOE", "index"),
    ("^FTSE", "FTSE 100 Index", "LSE", "index"),
    ("^N225", "Nikkei 225 Index", "Tokyo", "index"),
    ("^HSI", "Hang Seng Index", "Hong Kong", "index"),
]

# Ready-made shortcuts for the dashboard sidebar ("quick picks").
QUICK_PICKS = {
    "Metals": ["GC=F", "SI=F", "HG=F", "GOLDBEES.NS", "SILVERBEES.NS", "GLD", "SLV"],
    "Energy": ["CL=F", "BZ=F", "NG=F", "USO"],
    "Forex": ["USDINR=X", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "DX-Y.NYB"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
    "ETFs": ["NIFTYBEES.NS", "BANKBEES.NS", "SPY", "QQQ", "MON100.NS"],
    "Indices": ["^NSEI", "^BSESN", "^GSPC", "^IXIC", "^VIX"],
}

# Button-sized names for QUICK_PICKS (the catalog names are too long).
QUICK_LABELS = {
    "GC=F": "Gold", "SI=F": "Silver", "HG=F": "Copper", "GLD": "Gold ETF (US)", "SLV": "Silver ETF (US)",
    "GOLDBEES.NS": "Gold ETF (IN)", "SILVERBEES.NS": "Silver ETF (IN)",
    "CL=F": "WTI crude", "BZ=F": "Brent crude", "NG=F": "Natural gas", "USO": "Oil ETF (US)",
    "USDINR=X": "USD / INR", "EURUSD=X": "EUR / USD", "GBPUSD=X": "GBP / USD", "USDJPY=X": "USD / JPY",
    "DX-Y.NYB": "Dollar index",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "XRP-USD": "XRP",
    "NIFTYBEES.NS": "Nifty 50 ETF", "BANKBEES.NS": "Bank Nifty ETF", "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF", "MON100.NS": "Nasdaq ETF (IN)",
    "^NSEI": "NIFTY 50", "^BSESN": "SENSEX", "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^VIX": "VIX",
}

ASSET_NAMES = {sym: name for sym, name, _e, _c in POPULAR_ASSETS}
ASSET_CLASSES = {sym: cls for sym, _n, _e, cls in POPULAR_ASSETS}


def catalog_entries():
    """All offline entries as (symbol, name, exchange, asset_class)."""
    return ([(sym, name, exch, "stock") for sym, name, exch in POPULAR_STOCKS]
            + [tuple(row) for row in POPULAR_ASSETS])


def _match_offline(query: str, limit: int):
    q = query.strip().lower()
    if not q:
        return []
    scored = []
    for symbol, name, exchange, cls in catalog_entries():
        s, n = symbol.lower(), name.lower()
        if s == q or s.split(".")[0] == q:
            score = 0
        elif s.startswith(q):
            score = 1
        elif n.startswith(q) or any(w.startswith(q) for w in n.split()):
            score = 2
        elif q in n or q in s:
            score = 3
        else:
            continue
        scored.append((score, {"symbol": symbol, "name": name, "exchange": exchange,
                               "type": cls.upper(), "asset_class": cls}))
    scored.sort(key=lambda t: t[0])
    return [r for _, r in scored[:limit]]


@functools.lru_cache(maxsize=512)
def _search_online(query: str, limit: int):
    """Live Yahoo Finance symbol search. Cached, since the dashboard calls
    this on every keystroke and users often retype the same prefixes."""
    res = yf.Search(query, max_results=max(limit * 2, 10), news_count=0, enable_fuzzy_query=True)
    out = []
    for q in res.quotes or []:
        symbol = q.get("symbol")
        qtype = (q.get("quoteType") or "").upper()
        if not symbol or qtype not in _ALLOWED_TYPES:
            continue
        out.append({
            "symbol": symbol,
            "name": q.get("longname") or q.get("shortname") or ASSET_NAMES.get(symbol) or symbol,
            "exchange": q.get("exchDisp") or q.get("exchange") or "",
            "type": qtype,
            "asset_class": asset_class(symbol, qtype),
        })
        if len(out) >= limit:
            break
    return tuple(tuple(sorted(d.items())) for d in out)  # hashable for lru_cache


def search_stocks(query: str, limit: int = 10):
    """
    Return up to `limit` real stocks matching `query` (company name OR
    ticker, partial is fine), best matches first.

    Each result is a dict: {symbol, name, exchange, type}
    """
    query = (query or "").strip()
    if not query:
        return []

    results = []
    try:
        results = [dict(t) for t in _search_online(query.lower(), limit)]
    except Exception:
        results = []  # offline / rate-limited -> fall back to the built-in list

    # Merge in offline matches (deduped) so popular names always show up.
    seen = {r["symbol"] for r in results}
    for r in _match_offline(query, limit):
        if r["symbol"] not in seen:
            results.append(r)
            seen.add(r["symbol"])
    return results[:limit]


def format_suggestion(r: dict) -> str:
    """Human-readable label, e.g. 'RELIANCE.NS — Reliance Industries Ltd (NSE)'."""
    exch = f" ({r['exchange']})" if r.get("exchange") else ""
    return f"{r['symbol']} — {r['name']}{exch}"


def prompt_for_ticker(default: str = None) -> str:
    """
    Interactive terminal prompt used by the CLI scripts: asks for a
    stock name, shows matching real stocks, and lets the user pick one
    by number. Typing an exact ticker and choosing it works too.
    """
    import sys
    try:  # Windows consoles default to cp1252 -- don't crash on non-ASCII company names
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    while True:
        hint = f" [Enter = {default}]" if default else ""
        query = input(f"\nEnter a stock name or ticker (e.g. 'apple', 'reliance', 'TCS'){hint}: ").strip()
        if not query:
            if default:
                return default
            continue

        matches = search_stocks(query, limit=10)
        if not matches:
            print(f"  No stocks found matching '{query}'. Try a different spelling or the company's full name.")
            continue

        print(f"\n  Suggestions for '{query}':")
        for i, r in enumerate(matches, 1):
            print(f"   {i:>2}. {format_suggestion(r).replace('—', '-')}")
        choice = input("  Pick a number (Enter = 1, 's' = search again): ").strip().lower()
        if choice == "s":
            continue
        if choice == "":
            return matches[0]["symbol"]
        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            return matches[int(choice) - 1]["symbol"]
        print("  Invalid choice, let's try again.")


def resolve_ticker(configured):
    """CLI helper: use the CONFIG value if one is set, otherwise ask the user."""
    return configured if configured else prompt_for_ticker()


@functools.lru_cache(maxsize=128)
def get_company_profile(ticker: str) -> dict:
    """
    Basic company profile for the header card (name, sector, currency,
    market cap, P/E, ...). Every field is optional -- Yahoo doesn't
    return all of them for every instrument, so callers must handle None.
    """
    profile = {"symbol": ticker, "name": ASSET_NAMES.get(ticker.upper(), ticker), "currency": "USD",
               "asset_class": asset_class(ticker), "quote_type": None}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return profile
    qtype = info.get("quoteType")
    cls = asset_class(ticker, qtype)
    profile.update({
        "quote_type": qtype,
        "asset_class": cls,
        "asset_label": asset_label(ticker, qtype),
        # funds: what the ETF holds and what it costs to hold it
        "category": info.get("category") or info.get("fundFamily"),
        "fund_family": info.get("fundFamily"),
        "total_assets": info.get("totalAssets") or info.get("netAssets"),
        "expense_ratio": info.get("netExpenseRatio") or info.get("annualReportExpenseRatio"),
        "nav": info.get("navPrice"),
        "ytd_return": info.get("ytdReturn"),
        "three_year_return": info.get("threeYearAverageReturn"),
        "yield": info.get("yield"),
    })
    profile.update({
        "name": (info.get("longName") or ASSET_NAMES.get(ticker.upper())
                 or info.get("shortName") or ticker),
        "currency": info.get("currency") or "USD",
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "summary": info.get("longBusinessSummary"),
        "website": info.get("website"),
        "analyst_rating": info.get("recommendationKey"),
        "timezone": info.get("exchangeTimezoneName"),
        "target_price": info.get("targetMeanPrice"),
    })
    # Yahoo calls many non-US ETFs "EQUITY"; their names give them away.
    if profile["asset_class"] == "stock" and not info.get("sector"):
        _n = (profile.get("name") or "").lower()
        if " etf" in _n or "bees" in _n or _n.endswith("etf") or "exchange traded fund" in _n:
            profile["asset_class"] = cls = "etf"
            profile["asset_label"] = f"{ASSET_META['etf'][1]} {ASSET_META['etf'][0]}"
    if not profile.get("exchange"):
        profile["exchange"] = {"commodity": "Futures / spot", "forex": "FX market",
                               "crypto": "Crypto (24/7)", "index": "Index"}.get(cls)
    if cls in ("forex", "crypto"):
        profile["timezone"] = profile.get("timezone") or "UTC"   # these trade around the clock
    return profile


CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "GBp": "p",
                    "CNY": "¥", "HKD": "HK$", "AUD": "A$", "CAD": "C$", "CHF": "CHF ", "SGD": "S$",
                    "BTC": "₿", "USd": "¢"}


def currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOLS.get(code or "USD", f"{code} ")


# Sector -> sector ETF, so market-context features use the right sector
# instead of always assuming tech (XLK). US SPDR sector funds.
SECTOR_ETFS = {
    "Technology": "XLK", "Financial Services": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Industrials": "XLI", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Utilities": "XLU", "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Communication Services": "XLC",
}


if __name__ == "__main__":
    t = prompt_for_ticker()
    print(f"\nYou picked: {t}")
    print(get_company_profile(t))
