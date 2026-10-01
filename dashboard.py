"""
dashboard.py
------------
Interactive Streamlit dashboard: search any stock by company name (with
live suggestions), pick a date, get a predicted price and a
Buy/Sell/Hold recommendation from the combined ensemble of all models,
scan for the best buy/sell window, read candlestick patterns (with
their historical reliability on that stock), view a technical-analysis
scorecard with support/resistance levels, and see live model-accuracy/
weight breakdowns -- in a dark "trading terminal" design (ui_theme.py) with downloadable forecast data.

All prediction logic lives in dashboard_logic.py -- this file is just
the UI wiring on top of it.

Run:
    streamlit run dashboard.py

The first run will take a minute or two (training all models). After
that, results are cached so the app stays fast as you try different
dates.
"""

import datetime
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_loader import load_stock_data
import dashboard_logic as dl
import news_sentiment as ns
import news_history as nh
import social_media as sm
import stock_search as ss
import candlestick_patterns as cp
import technical_analysis as ta
import market_data as md
import timeframes as tfm
import ui_theme as ui
import forecast_engine as fe
import research_analyst as ra
import intraday_target as itg
import options_plan as opl
import scenarios as sx
import html as _html

try:
    from streamlit_searchbox import st_searchbox  # live type-ahead suggestions
    HAS_SEARCHBOX = True
except ImportError:  # falls back to text box + dropdown of matches
    HAS_SEARCHBOX = False

st.set_page_config(page_title=f"{ui.APP_NAME} · Stock Prediction Dashboard", page_icon="📈", layout="wide")
ui.inject_css()
ui.boot_fx()          # count-up numbers + cursor glow (JS half of the effects)

# ---- top loading bar -------------------------------------------------------
# Streamlit renders top to bottom, so the page arrives in stages. This advances a
# bar pinned to the viewport as each one lands, and fades it out when done.
_load_slot = st.empty()


def _load(pct, label, done=False):
    try:
        _load_slot.markdown(ui.progress_html(pct, label, done), unsafe_allow_html=True)
    except Exception:
        pass


_load(6, "Starting up")
PLOTLY_TEMPLATE = ui.PLOTLY_TEMPLATE
SIGNAL_STYLE = ui.SIGNAL_STYLE
signal_badge_html = ui.signal_badge

# ----------------------------- STOCK SEARCH ---------------------------------
if "ticker" not in st.session_state:
    st.session_state.ticker = "MSFT"
    st.session_state.recent = ["MSFT"]


def _set_ticker(symbol):
    if symbol:
        st.session_state.ticker = symbol
        recent = [t for t in st.session_state.recent if t != symbol]
        st.session_state.recent = ([symbol] + recent)[:6]


def _search_suggestions(term: str):
    """Called by the search box on every keystroke -> [(label, symbol), ...]."""
    if not term or not term.strip():
        return []
    return [(ss.format_suggestion(r), r["symbol"]) for r in ss.search_stocks(term, limit=10)]


with st.sidebar:
    ui.brand()
    app_mode = st.segmented_control("Mode", ["📈 Stock analysis", "🚀 IPO Center"], default="📈 Stock analysis",
                                    key="app_mode", label_visibility="collapsed")
if app_mode == "🚀 IPO Center":
    import ipo_page
    ipo_page.render(on_progress=_load)   # the IPO page drives the same top bar
    _load(100, "Done", done=True)
    st.stop()
st.sidebar.title("🔎 Search any market")
with st.sidebar:
    if HAS_SEARCHBOX:
        picked = st_searchbox(
            _search_suggestions,
            placeholder="Company, metal, currency, ETF… e.g. gold, usd inr, bitcoin",
            key="stock_searchbox",
            debounce=250,
            help="Shares, ETFs, indices, commodities (gold, silver, crude), currency pairs and crypto. "
                 "Suggestions come from Yahoo Finance's live symbol search, so they're real tradable "
                 "symbols on any exchange (NYSE, NASDAQ, NSE, BSE, LSE, COMEX, FX…).",
        )
    else:
        query = st.text_input("Name or symbol", placeholder="e.g. apple, gold, usd inr, bitcoin")
        matches = ss.search_stocks(query, limit=10) if query else []
        picked = None
        if query and not matches:
            st.warning("No matching stocks found.")
        elif matches:
            choice = st.selectbox("Did you mean…", matches, format_func=ss.format_suggestion)
            if st.button("Analyze this stock", type="primary"):
                picked = choice["symbol"]
        st.caption("Tip: `pip install streamlit-searchbox` for live suggestions as you type.")

    # The search box keeps returning its last pick on every rerun, so only
    # react to a NEW pick (otherwise it would override the Recent buttons).
    if not picked:
        st.session_state.last_pick = None  # box cleared: picking the same stock again should work
    elif picked != st.session_state.get("last_pick"):
        st.session_state.last_pick = picked
        if picked != st.session_state.ticker:
            _set_ticker(picked)
            st.rerun()

    # One-click shortcuts to the markets people usually want to look at.
    st.caption("Quick picks")
    _qp_icons = {"Metals": "🥇 Metals", "Energy": "🛢️ Energy", "Forex": "💱 Forex",
                 "Crypto": "🪙 Crypto", "ETFs": "📦 ETFs", "Indices": "📊 Indices"}
    _qp_cat = st.selectbox("Quick picks", list(ss.QUICK_PICKS), label_visibility="collapsed",
                           format_func=lambda c: _qp_icons.get(c, c), key="quick_pick_cat")
    _qcols = st.columns(2)
    for _i, _sym in enumerate(ss.QUICK_PICKS[_qp_cat]):
        _nm = ss.QUICK_LABELS.get(_sym) or ss.ASSET_NAMES.get(_sym, _sym)
        if _qcols[_i % 2].button(_nm, key=f"qp_{_sym}", width="stretch",
                                 help=f"{_sym} - {ss.ASSET_NAMES.get(_sym, _sym)}"):
            _set_ticker(_sym)
            st.rerun()

    with st.expander("ℹ️ What are these?"):
        st.caption({
            "Metals": "Gold, silver and copper — traded as COMEX futures (the world price in $/oz) or as "
                      "ETFs that hold the metal for you in ₹ or $.",
            "Energy": "Crude oil (WTI is the US benchmark, Brent the global one), natural gas, and an oil ETF.",
            "Forex": "Currency pairs. USD/INR is how many rupees one dollar buys; the dollar index (DXY) "
                     "measures the dollar against a basket of currencies.",
            "Crypto": "Crypto assets priced in dollars. They trade 24/7, so forecasts here use calendar days.",
            "ETFs": "Funds you buy like a share; each one tracks an index (Nifty 50, Bank Nifty, S&P 500, "
                    "Nasdaq 100) so one trade gives you the whole basket.",
            "Indices": "The market barometers themselves. You can't buy an index directly — buy its ETF "
                       "instead — but they're the benchmark everything is measured against.",
        }.get(_qp_cat, ""))
        for _sym in ss.QUICK_PICKS[_qp_cat]:
            st.caption(f"**{ss.QUICK_LABELS.get(_sym, _sym)}** · `{_sym}` — {ss.ASSET_NAMES.get(_sym, _sym)}")

    if len(st.session_state.recent) > 1:
        st.caption("Recent")
        rcols = st.columns(2)
        for i, t in enumerate(st.session_state.recent):
            if rcols[i % 2].button(t, key=f"recent_{t}", width="stretch"):
                _set_ticker(t)
                st.rerun()

ticker = st.session_state.ticker

# The candle/horizon controls need the symbol's exchange (an NSE session is 375
# minutes, a US one 390), so their place is reserved here and filled once the
# profile has loaded, a few lines below.
tf_slot = st.sidebar.container()

with st.sidebar:
    ui.render(f'<div style="margin-top:6px"><span class="side-ticker">● ANALYZING&nbsp;&nbsp;{ticker}</span></div>')
st.sidebar.markdown("---")

# ----------------------------- SIDEBAR CONFIG ---------------------------------
st.sidebar.title("⚙️ Settings")
history_start = st.sidebar.text_input("History start date", value="2014-01-01")
history_end = st.sidebar.text_input("History end date (train up to)", value=str(datetime.date.today()))
buy_threshold = st.sidebar.slider("Buy threshold (%)", 0.1, 5.0, 1.0, 0.1) / 100
sell_threshold = st.sidebar.slider("Sell threshold (%)", 0.1, 5.0, 1.0, 0.1) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("**📰 News in the models**")
use_news = st.sidebar.checkbox(
    "Consider news history", value=True,
    help="Adds daily news tone/attention (GDELT, free) as EXTRA inputs. The models are trained with and "
         "without them and news is only kept if it improves accuracy on validation data.")
av_key = st.sidebar.text_input(
    "Alpha Vantage API key (optional)", value="", type="password",
    help="Adds per-stock sentiment scores for US stocks. Free key: alphavantage.co/support/#api-key. "
         "You can also set the ALPHAVANTAGE_API_KEY environment variable instead of typing it here.")
av_key = nh.alpha_vantage_key(av_key)

use_universe = st.sidebar.checkbox(
    "Universe model (experimental)", value=False,
    help="Offers the engine a view from a model pooled across 38 symbols. Measured honestly it is unstable: "
         "rank-IC +0.038 over the full out-of-fold history at one month, positive in only 5 of 8 years and "
         "driven mostly by 2023, with no edge at all at 5 days. It is left switchable because the engine "
         "gates it per symbol and simply drops it when it does not help — but it is off by default.")

st.sidebar.markdown("**💬 Social media**")
use_social = st.sidebar.checkbox(
    "Track social buzz", value=True,
    help="Reads what people are posting right now on StockTwits, Reddit and YouTube (and X if you add a "
         "token below), scores the mood, and saves one reading a day so the models can test it as an input.")
x_token = st.sidebar.text_input(
    "X (Twitter) Bearer token - optional", value="", type="password",
    help="Optional. X has no free API tier any more (since Feb 2026): reads are billed per post at "
         "console.x.com. Paste a Bearer token here to include X posts; leave it empty and StockTwits, "
         "Reddit and YouTube are still used for free. You can also set the X_BEARER_TOKEN "
         "environment variable.")
if x_token:
    st.sidebar.caption("💳 " + sm.x_cost_note())

st.sidebar.markdown("---")
st.sidebar.caption(
    "🧠 Forecasts come from **Ridge + Random Forest + XGBoost** ensembles, one per horizon "
    "(1 day to 3 months). Market, technical, candlestick, calendar, macro, earnings and news data "
    "are tested for every symbol - shares, ETFs, metals, currencies or crypto - and kept only "
    "where they consistently improve accuracy.\n\n"
    "⚠️ Uncertainty grows the further ahead you ask -- watch the 80% likely range. "
    "Research and backtesting tool -- not financial advice."
)


# Yahoo's chart endpoint returns the intraday bars AND a live quote that ticks
# every ~5-15 s, which is much fresher than downloading 1-minute bars.
_SESSION_HOURS = {"Asia/Kolkata": (9.25, 15.5), "America/New_York": (9.5, 16.0), "Europe/London": (8.0, 16.5),
                  "Asia/Tokyo": (9.0, 15.0), "Asia/Hong_Kong": (9.5, 16.0)}
_YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval={iv}&includePrePost=false"


def fetch_live_chart(ticker, rng="1d", iv="1m"):
    """(DataFrame of intraday bars, meta dict) straight from Yahoo's chart API. No caching: this is the live feed."""
    import requests
    r = requests.get(_YAHOO_CHART.format(sym=ticker, rng=rng, iv=iv),
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                            "(KHTML, like Gecko) Chrome/124 Safari/537.36"}, timeout=12)
    j = r.json()["chart"]["result"][0]
    meta, ts = j["meta"], j.get("timestamp") or []
    q = (j.get("indicators", {}).get("quote") or [{}])[0]
    tz = meta.get("exchangeTimezoneName") or "UTC"
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None)
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")}, index=idx)
    df.columns = [c.capitalize() for c in df.columns]
    df = df.dropna(subset=["Close"])
    return df, meta




@st.cache_data(ttl=10, show_spinner=False)  # the live quote itself refreshes every ~5-15 s
def cached_quote(ticker):  # audit-exempt: timeframe-free (a quote is one instant, not a bar)
    """Live price + today's open/high/low/volume from Yahoo's quote feed ({} if unavailable)."""
    try:
        bars, meta = fetch_live_chart(ticker, "1d", "1m")
    except Exception:
        return {}
    if not meta.get("regularMarketPrice"):
        return {}
    return {"price": float(meta["regularMarketPrice"]), "time": meta.get("regularMarketTime"),
            "tz": meta.get("exchangeTimezoneName") or "UTC", "prev_close": meta.get("chartPreviousClose") or meta.get("previousClose"),
            "day_high": meta.get("regularMarketDayHigh"), "day_low": meta.get("regularMarketDayLow"),
            "volume": meta.get("regularMarketVolume"),
            "open": float(bars["Open"].iloc[0]) if len(bars) else None}


@st.cache_data(show_spinner=False, ttl=15 * 60)  # re-download daily history every 15 min
def cached_load_data(ticker, start, end, interval="1d"):
    # Phase 8: interval is part of the cache key AND it changes what is loaded, so a
    # 15-minute request can never be served a daily frame out of the cache.
    if interval in ("1d", "1wk"):
        return load_stock_data(ticker, start, end)
    return dl.load_ohlcv(ticker, interval, start=start, end=end)


def apply_live_quote(raw_base, quote):
    """
    Deprecated shim. The real implementation lives in market_data.apply_live_quote_safe,
    which refuses to touch a frame marked historical.

    This function used to do the overlay itself, and that is what wrote today's price
    into a historical frame: it appended a row for today whenever the frame ended
    earlier. It is kept only so any remaining caller inherits the guard instead of the
    bug -- new code should call market_data.apply_live_quote_safe directly.
    """
    return md.apply_live_quote_safe(raw_base, quote)


def _header_body(ticker, raw_base, cur, profile, market_open_now, tz_name):
    """Hero price header + KPI cards. Runs inside a fragment so the headline price
    follows the live quote instead of the 15-minute-old daily download."""
    # Phase 2: the header used the unguarded overlay, so in historical mode it put
    # today's price back on top of a frame the rest of the page had already truncated.
    _hist = bool(globals().get("IS_HISTORICAL", False))
    raw, quote, quote_used = md.apply_live_quote_safe(
        raw_base, None if _hist else cached_quote(ticker), is_historical=_hist)
    last_date = raw.index[-1]
    last_price = raw["Close"].iloc[-1]
    prior_price = float(quote["prev_close"]) if (quote_used and quote.get("prev_close")) else raw["Close"].iloc[-2]
    day_change_pct = (last_price - prior_price) / prior_price * 100
    year_high = raw["High"].tail(252).max()
    year_low = raw["Low"].tail(252).min()
    if quote_used and market_open_now and quote.get("time"):
        qt = pd.Timestamp(quote["time"], unit="s", tz="UTC").tz_convert(quote["tz"]).strftime("%H:%M:%S")
        st.caption(f"🟢 Live quote · updated {qt} exchange time · re-checked every "
                   f"{st.session_state.get('live_refresh_s', 2)}s")
    if _hist:
        st.caption(f"🕰️ Historical header · every price below is as of {last_date:%d %b %Y}. "
                   f"Market cap, P/E, beta, the analyst target and dividend yield are the "
                   f"latest-known values -- the provider publishes no history for them, so "
                   f"they cannot be rewound to that date.")
    cls = profile.get("asset_class", "stock")
    if cls == "stock":
        chips = [profile.get("exchange"), profile.get("sector"), profile.get("industry"), profile.get("currency")]
    else:  # gold, EURUSD, bitcoin, an ETF... have no sector/industry
        chips = [profile.get("asset_label") or ss.asset_label(ticker), profile.get("exchange"),
                 profile.get("category"), profile.get("currency")]
    ui.hero(
        name=profile.get("name", ticker), ticker=ticker.upper(), chips=chips,
        price=last_price, change_abs=last_price - prior_price, change_pct=day_change_pct,
        market_open=market_open_now, as_of=last_date.strftime("%d %b %Y"), cur=cur,
    )

    _avg_vol = raw["Volume"].tail(21).iloc[:-1].mean()
    rel_vol = raw["Volume"].iloc[-1] / _avg_vol if _avg_vol else None
    ret_1m = (last_price / raw["Close"].iloc[-22] - 1) * 100 if len(raw) > 22 else 0.0
    _ytd_base = raw.loc[raw.index < pd.Timestamp(last_date.year, 1, 1), "Close"]
    ret_ytd = (last_price / _ytd_base.iloc[-1] - 1) * 100 if len(_ytd_base) else 0.0
    # Yahoo's dividendYield is already a percent in recent yfinance versions (e.g. 0.75 = 0.75%)
    div_yield = profile.get("dividend_yield")

    target = profile.get("target_price")
    upside = (target / last_price - 1) * 100 if target else None
    rating_txt = (profile.get("analyst_rating") or "").replace("_", " ").title() or "N/A"

    _day_word = "days" if bool((raw.index.dayofweek >= 5).mean() > 0.10) else "trading days"

    def _ret_kpi(title, rows, sub):
        """A return-over-N-sessions KPI card."""
        sub = sub.replace("trading days", _day_word)
        if len(raw) <= rows:
            return ui.kpi_html(title, "N/A", sub)
        v = (last_price / raw["Close"].iloc[-rows - 1] - 1) * 100
        return ui.kpi_html(title, f"{'▲' if v >= 0 else '▼'} {v:+.2f}%", sub, "up" if v >= 0 else "down")

    _vol_ann = float(raw["Close"].pct_change().tail(252).std() * (252 ** 0.5) * 100)
    cards = [ui.range_bar_html("Day range", raw["Low"].iloc[-1], raw["High"].iloc[-1], last_price, cur),
             ui.range_bar_html("52-week range", year_low, year_high, last_price, cur)]
    ytd_card = ui.kpi_html("YTD return", f"{'▲' if ret_ytd >= 0 else '▼'} {ret_ytd:+.2f}%",
                           f"since 1 Jan {last_date.year}", "up" if ret_ytd >= 0 else "down")
    vol_card = ui.kpi_html("Volatility (1y)", f"{_vol_ann:.1f}%", "annualised, from daily moves")

    if cls == "stock":
        cards += [
            ui.kpi_html("Market cap", f"{cur}{fmt_big(profile.get('market_cap'))}" if profile.get("market_cap") else "N/A"),
            ui.kpi_html("P/E (trailing)", f"{profile['pe_ratio']:.1f}" if profile.get("pe_ratio") else "N/A"),
            ui.kpi_html("Beta", f"{profile['beta']:.2f}" if profile.get("beta") else "N/A", "vs. market volatility"),
            ui.kpi_html("Analyst target", f"{cur}{target:,.2f}" if target else "N/A",
                        ("latest known, not as of this date" if _hist else
                         (f"{'▲' if upside >= 0 else '▼'} {upside:+.1f}% · {rating_txt}" if target else rating_txt)),
                        "" if _hist else (("up" if upside and upside >= 0 else "down") if target else "")),
            ui.kpi_html("Volume", fmt_big(raw["Volume"].iloc[-1]),
                        f"{rel_vol:.1f}× 20-day average" if rel_vol else None),
            ui.kpi_html("1-month return", f"{'▲' if ret_1m >= 0 else '▼'} {ret_1m:+.2f}%", "last 21 trading days",
                        "up" if ret_1m >= 0 else "down"),
            ytd_card,
            ui.kpi_html("Dividend yield", f"{div_yield:.2f}%" if div_yield is not None else "N/A", "trailing, per Yahoo"),
        ]
    elif cls == "etf":
        _aum = profile.get("total_assets")
        _er = profile.get("expense_ratio")
        cards += [
            ui.kpi_html("Fund size", f"{cur}{fmt_big(_aum)}" if _aum else "N/A", "assets under management"),
            ui.kpi_html("Expense ratio", f"{_er:.2f}%" if _er else "N/A", "yearly cost of holding it"),
            ui.kpi_html("Tracks", profile.get("category") or profile.get("fund_family") or "Index / commodity",
                        small=True),
            ui.kpi_html("Volume", fmt_big(raw["Volume"].iloc[-1]),
                        f"{rel_vol:.1f}× 20-day average" if rel_vol else None),
            _ret_kpi("1-month return", 21, "last 21 trading days"),
            _ret_kpi("1-year return", 252, "last 252 trading days"),
            ytd_card, vol_card,
        ]
    else:  # commodity, currency pair, crypto, index
        _extra = (ui.kpi_html("Volume", fmt_big(raw["Volume"].iloc[-1]),
                              f"{rel_vol:.1f}× 20-day average" if rel_vol else None)
                  if float(raw["Volume"].tail(21).sum()) > 0 else
                  ui.kpi_html("Trades", "24 × 7" if cls == "crypto" else "24 × 5",
                              "no exchange volume is published"))
        cards += [
            ui.kpi_html("Asset class", profile.get("asset_label") or ss.asset_label(ticker),
                        profile.get("exchange"), small=True),
            _ret_kpi("1-week return", 5, "last 5 trading days"),
            _ret_kpi("1-month return", 21, "last 21 trading days"),
            _ret_kpi("3-month return", 63, "last 63 trading days"),
            _ret_kpi("1-year return", 252, "last 252 trading days"),
            ytd_card, vol_card, _extra,
        ]
    ui.kpi_grid(cards)




# `news_sig` / `extra_sig` identify the data in cache keys; the objects themselves are
# passed as `_news_df` / `_extra` (leading underscore = not hashed by Streamlit).
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_extra_data(ticker, start, end, interval="1d"):
    return fe.load_extra_data(ticker, start, end)


@st.cache_resource(show_spinner=False)
def cached_engine(ticker, start, end, data_sig, news_sig=None, _news_df=None, _extra=None,
                  model_version=dl.MODEL_VERSION + "-engine-v17", social_sig=None, _social_df=None,
                  pooled_sig=False, _pooled_df=None, interval="1d"):
    raw = cached_load_data(ticker, start, end, interval)
    return fe.ForecastEngine(raw, ticker, extra=_extra, news_df=_news_df, social_df=_social_df,
                             pooled_df=_pooled_df).fit()


@st.cache_data(show_spinner=False)
def cached_engine_eval(ticker, start, end, data_sig, news_sig=None, _news_df=None, _extra=None,
                       model_version=dl.MODEL_VERSION + "-engine-v17", social_sig=None, _social_df=None,
                       interval="1d"):
    raw = cached_load_data(ticker, start, end, interval)
    return fe.evaluate_engine(raw, ticker, extra=_extra, news_df=_news_df, social_df=_social_df)


@st.cache_resource(show_spinner=False)
def cached_train_direction(ticker, start, end, news_sig=None, _news_df=None, model_version=dl.MODEL_VERSION):
    raw = cached_load_data(ticker, start, end)
    return dl.train_direction_and_regime(raw, ticker=ticker, news_df=_news_df)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_news_features(ticker, start, end, company_name, has_av_key, _av_key):
    raw = cached_load_data(ticker, start, end)
    return nh.build_news_features(ticker, company_name, raw.index, av_key=_av_key, time_budget=45)


@st.cache_data(show_spinner=False)
def cached_direction_accuracy(ticker, start, end, _adv_df, news_sig=None, model_version=dl.MODEL_VERSION):
    return dl.evaluate_direction_accuracy(_adv_df)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_profile(ticker):
    return ss.get_company_profile(ticker)


@st.cache_data(show_spinner=False)
def cached_candle_analysis(ticker, start, end, forward_days, interval="1d"):
    raw = cached_load_data(ticker, start, end, interval)
    return cp.pattern_events(raw, forward_days), cp.pattern_reliability(raw, forward_days)


def fmt_big(n):
    if n is None:
        return "N/A"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(n) >= div:
            return f"{n/div:.2f}{suf}"
    return f"{n:,.0f}"


@st.cache_data(show_spinner=False, ttl=120)  # the social feed is live -- 2 minutes old at most
def cached_social(ticker, company_name, x_key, bucket):
    """Live social snapshot. `bucket` is a time bucket so the auto-refreshing
    panel gets fresh data instead of the cached copy."""
    snap = sm.social_snapshot(ticker, company_name, x_bearer=x_key)
    try:
        sm.collect_today(ticker, snap)   # one row per day -> history the models can learn from
    except Exception:
        pass
    return snap


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_social_features(ticker, dates_sig, _dates):
    """Daily social history turned into model inputs (None until enough days exist)."""
    return sm.build_social_features(ticker, _dates)


@st.cache_data(show_spinner=False, ttl=900)  # refresh every 15 minutes -- news changes fast
def cached_news_sentiment(ticker, company_query=None):
    summary = ns.get_news_sentiment_summary(ticker, company_query=company_query)
    nh.collect_today(ticker, summary)  # save today's score -> builds your own news history over time
    return summary


def _live_body(ticker, daily, cur, tz_name):
    """One refresh of the live panel (wrapped in st.fragment below)."""
    c1, c2, c3 = st.columns([2, 1, 1.4])
    with c1:
        rng = st.segmented_control("Live range", ["1D · 1 min", "5D · 5 min"], default="1D · 1 min",
                                   key="live_range", label_visibility="collapsed")
    with c2:
        st.button("🔄 Refresh now", key="live_refresh_btn", width="stretch")  # any click re-runs this fragment
    with c3:
        st.caption(f"⟳ auto-refresh every {st.session_state.get('live_refresh_s', 2)}s · "
                   f"last refresh **{datetime.datetime.now().strftime('%H:%M:%S')}**")
    period, interval = ("5d", "5m") if rng == "5D · 5 min" else ("1d", "1m")
    try:
        h, meta = fetch_live_chart(ticker, period, interval)
    except Exception as e:
        st.info(f"Live feed unavailable right now ({e}). It will retry on the next refresh.")
        return
    if h.empty:
        st.info("Yahoo Finance has no intraday data for this stock right now.")
        return

    live_price = meta.get("regularMarketPrice", float(h["Close"].iloc[-1]))
    quote_time = pd.to_datetime(meta.get("regularMarketTime", 0), unit="s", utc=True).tz_convert(
        meta.get("exchangeTimezoneName") or "UTC").tz_localize(None)
    first_day = pd.Timestamp(h.index[0].date())
    prior = daily.loc[daily.index < first_day, "Close"]
    base = float(meta.get("chartPreviousClose") or (prior.iloc[-1] if len(prior) else h["Open"].iloc[0]))
    session = h[h.index.date == h.index[-1].date()]
    try:
        from zoneinfo import ZoneInfo
        now_local = datetime.datetime.now(ZoneInfo(tz_name)).replace(tzinfo=None)
        open_h, close_h = _SESSION_HOURS.get(tz_name, (9.5, 16.0))
        hour = now_local.hour + now_local.minute / 60
        _acls = profile.get("asset_class", "stock")
        if _acls == "crypto":
            is_live = True
        elif _acls == "forex":
            is_live = now_local.weekday() < 5
        else:
            is_live = (now_local.weekday() < 5 and open_h <= hour < close_h
                       and now_local.date() == h.index[-1].date())
    except Exception:
        is_live = False
    base_label = "Prev close" if period == "1d" else "5 days ago"
    stats = [("Open", f"{cur}{session['Open'].iloc[0]:,.2f}"), ("Day high", f"{cur}{session['High'].max():,.2f}"),
             ("Day low", f"{cur}{session['Low'].min():,.2f}"), ("Volume", fmt_big(session['Volume'].sum())),
             (base_label, f"{cur}{base:,.2f}")]
    ui.live_header(is_live, live_price, live_price - base, live_price / base - 1, stats, cur,
                   quote_time.strftime("%d %b %H:%M:%S"))

    color = ui.UP if live_price >= base else ui.DOWN
    fill = "rgba(12,163,12,0.12)" if live_price >= base else "rgba(208,59,59,0.12)"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
    fig.add_trace(go.Scatter(x=h.index, y=h["Close"], mode="lines", name="Price", line=dict(color=color, width=2),
                             fill="tozeroy", fillcolor=fill), row=1, col=1)
    fig.add_trace(go.Scatter(x=[h.index[-1]], y=[live_price], mode="markers", name="Live",
                             marker=dict(color=color, size=11, line=dict(color="white", width=1.5)),
                             hovertemplate=f"Live {cur}%{{y:,.2f}}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=base, line_dash="dot", line_width=1, line_color=ui.NEUTRAL, row=1, col=1,
                  annotation_text=f"{base_label} {cur}{base:,.2f}", annotation_position="bottom left",
                  annotation_font_size=10)
    up_bar = h["Close"] >= h["Open"]
    fig.add_trace(go.Bar(x=h.index, y=h["Volume"], name="Volume", showlegend=False, marker_line_width=0, opacity=0.45,
                         marker_color=[ui.UP if u else ui.DOWN for u in up_bar]), row=2, col=1)
    lo = min(h["Low"].min(), base, live_price) * 0.998
    hi = max(h["High"].max(), base, live_price) * 1.002
    open_h, close_h = _SESSION_HOURS.get(tz_name, (9.5, 16.0))
    breaks = list(WEEKEND_BREAK)
    if not TRADES_WEEKENDS:  # crypto also trades overnight, so no session gap either
        breaks.append(dict(bounds=[close_h, open_h], pattern="hour"))
    fig.update_xaxes(rangebreaks=breaks)
    fig.update_yaxes(range=[lo, hi], side="right", row=1, col=1)
    fig.update_layout(height=420, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=64, t=10, b=10), showlegend=False,
                      uirevision="live")  # keep zoom/pan across refreshes
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="live_chart")
    st.caption("Live price straight from Yahoo's quote feed (updates every ~5-15 seconds); the minute bars behind it "
               "are about a minute behind the exchange. Only this panel reloads, not the whole page — but the timer "
               "starts only once the page below has finished loading (the first load of a stock trains the models, "
               "which takes 1-3 minutes). Use 🔄 Refresh now any time.")


# ----------------------------- HEADER ---------------------------------
# Some search results point at symbols with (almost) no price history, e.g.
# Indian SME listings like "OWAIS-SM.NS" whose data moved to "OWAIS.NS" after
# migrating to the main board. load_stock_data() falls back to the symbol that
# has the data; switch the whole dashboard to it and say so.
try:
    resolved = cached_load_data(ticker, history_start, history_end).attrs.get("ticker", ticker)
except Exception as e:
    st.error(f"Could not load '{ticker}': {e}")
    st.stop()
if resolved != ticker:
    st.session_state.recent = list(dict.fromkeys(resolved if t == ticker else t for t in st.session_state.recent))
    st.session_state.ticker = resolved
    st.session_state.resolved_note = (ticker, resolved)
    st.rerun()  # redraw everything (incl. the sidebar) with the working symbol
if "resolved_note" in st.session_state:
    original, used = st.session_state.pop("resolved_note")
    st.info(f"ℹ️ **{original}** has little or no price history on Yahoo Finance, so this analysis uses "
            f"**{used}** (same company, different listing) instead.")

profile = cached_profile(ticker)
cur = ss.currency_symbol(profile.get("currency"))

# ---- candle timeframe & forecast horizon: two separate controls ----------
with tf_slot:
    st.markdown("**⏱️ Candle & horizon**")
    _tz, _acls = profile.get("timezone"), profile.get("asset_class")
    _tf_options = tfm.available(_tz, _acls)
    tf_code = st.selectbox(
        "Candle / timeframe", _tf_options,
        index=_tf_options.index(tfm.DEFAULT) if tfm.DEFAULT in _tf_options else len(_tf_options) - 1,
        format_func=lambda c: tfm.label(c) + (" · resampled" if tfm.get(c)["resample"] else ""),
        key="tf_code",
        help="Only intervals this data source can actually serve with enough history are listed. "
             "Everything — data, indicators, support/resistance, models, forecast and chart — is "
             "computed on the candles you choose.")
    _h_options = tfm.horizon_options(tf_code, _tz, _acls)
    _h_labels = [o[0] for o in _h_options]
    _default_h = "1 month" if "1 month" in _h_labels else _h_labels[min(2, len(_h_labels) - 1)]
    horizon_label = st.selectbox("Forecast horizon", _h_labels, index=_h_labels.index(_default_h),
                                 key="tf_horizon",
                                 help="Chosen separately from the candle size: a 15-minute chart can "
                                      "forecast one session, a daily chart a month.")
    horizon_days = dict(_h_options)[horizon_label]
    _cands = tfm.horizon_to_candles(tf_code, horizon_days, _tz, _acls)
    st.caption(f"**{tfm.label(tf_code)}** candles · **{horizon_label}** ahead = **{_cands:,} candles** "
               f"({tfm.candles_per_session(tf_code, _tz, _acls):g} per session)")

# Anything other than daily candles is analysed end-to-end on its own page:
# its own data, indicators, support/resistance, models, forecast and chart.
if tf_code != "1d":
    import timeframe_page
    timeframe_page.render(ticker, profile, tf_code, horizon_label, horizon_days, cur,
                          buy_threshold, sell_threshold)
    st.stop()


# Prices refresh every 15 minutes, while models are retrained at most once per
# day (their cache key is the date range) -- use the freshest data for
# everything shown and for predictions (the trained models just apply to the newest rows).
raw = cached_load_data(ticker, history_start, history_end)

# During market hours the daily download is up to 15 minutes old, so overlay the
# live quote (refreshed every ~10 s) onto today's row -- header, KPIs, charts and
# the forecast's reference price then all match the live market.
# ---- CURRENT vs HISTORICAL (Phase 2) ------------------------------------
# The history end date decides which world we are in. In the past, the frame is
# immutable: no live quote, no auto-refresh, nothing dated after the as-of day.
try:
    AS_OF = pd.Timestamp(history_end)
except Exception:
    AS_OF = pd.Timestamp(datetime.date.today())
IS_HISTORICAL = AS_OF.normalize() < pd.Timestamp(datetime.date.today())
raw = raw.loc[raw.index <= AS_OF] if IS_HISTORICAL else raw
raw.attrs.update({"ticker": ticker, "interval": "1d", "as_of": AS_OF,
                  "mode": md.MODE_HISTORICAL if IS_HISTORICAL else md.MODE_CURRENT,
                  "immutable": IS_HISTORICAL})

# Repair the handful of malformed candles a provider emits, THEN validate. A page
# that refuses to load over 1 bad candle in 3,203 is worse than useless -- it teaches
# you to distrust a tool that was right. Anything that would make the answer wrong
# still stops the page.
raw, _repairs = md.repair_market_data(raw, ticker)
_validation = None
try:
    _validation = md.validate_market_data(raw, ticker, "1d", as_of=AS_OF,
                                          mode=raw.attrs["mode"])
except md.MarketDataError as _e:
    st.error(f"**Data validation failed — analysis stopped.** {_e}")
    st.info("Nothing is substituted when validation fails: showing a different date, ticker or "
            "timeframe instead would be worse than showing nothing.")
    st.stop()
if _repairs:
    st.caption("🧹 Data quality: " + "; ".join(_repairs) +
               ". A candle's high was only ever raised to a price that candle already "
               "contained, and its low lowered the same way — nothing was invented.")

quote = None if IS_HISTORICAL else cached_quote(ticker)
raw, quote, quote_used = md.apply_live_quote_safe(raw, quote, is_historical=IS_HISTORICAL)

last_date = raw.index[-1]
last_price = raw["Close"].iloc[-1]
prior_price = float(quote["prev_close"]) if (quote_used and quote.get("prev_close")) else raw["Close"].iloc[-2]
day_change_pct = (last_price - prior_price) / prior_price * 100
# 52-week range from INTRADAY highs/lows (what Google/Yahoo/brokers show), not closes
year_high = raw["High"].tail(252).max()
year_low = raw["Low"].tail(252).min()

# Is today's candle still forming? (market open right now on the stock's exchange)
_CLOSE_TIMES = {"Asia/Kolkata": (15, 30), "America/New_York": (16, 0), "Europe/London": (16, 30),
                "Asia/Tokyo": (15, 0), "Asia/Hong_Kong": (16, 0)}
market_open_now = False
try:
    from zoneinfo import ZoneInfo
    tz_name = profile.get("timezone") or ("Asia/Kolkata" if ticker.upper().endswith((".NS", ".BO")) else "America/New_York")
    now_local = datetime.datetime.now(ZoneInfo(tz_name))
    close_h, close_m = _CLOSE_TIMES.get(tz_name, (16, 0))
    _acls = profile.get("asset_class", "stock")
    if _acls == "crypto":
        market_open_now = True                                   # never closes
    elif _acls == "forex":
        market_open_now = now_local.weekday() < 5                # 24 hours, Monday to Friday
    else:
        market_open_now = (last_date.date() == now_local.date()
                           and (now_local.hour, now_local.minute) < (close_h, close_m))
except Exception:
    pass

# Crypto has Saturday/Sunday candles; every other market has weekend gaps that
# would otherwise leave flat spots on the charts.
TRADES_WEEKENDS = bool((raw.index.dayofweek >= 5).mean() > 0.10)
WEEKEND_BREAK = [] if TRADES_WEEKENDS else [dict(bounds=["sat", "mon"])]
STEP_WORD = "calendar days" if TRADES_WEEKENDS else "business days"

# ---- freshly listed stocks (IPOs) ----------------------------------------
# The indicators need a 200-day average, and each horizon's model needs 300+
# labelled rows spread over three walk-forward folds. A stock that listed this
# week simply doesn't have that, so show what DOES exist and stop here rather
# than printing numbers with nothing behind them.
if len(raw) < dl.MIN_ROWS_REQUIRED:
    _sessions = len(raw)
    _row = raw.iloc[-1]
    _open = float(_row["Open"]) or last_price
    ui.hero(
        name=profile.get("name", ticker), ticker=ticker.upper(),
        chips=[profile.get("exchange"), profile.get("sector"), profile.get("industry"), profile.get("currency")],
        price=last_price, change_abs=last_price - _open,
        change_pct=(last_price / _open - 1) * 100 if _open else 0.0,
        market_open=market_open_now, as_of=last_date.strftime("%d %b %Y"), cur=cur,
    )
    ui.kpi_grid([
        ui.kpi_html("Sessions of history", f"{_sessions}", f"first close {raw.index[0]:%d %b %Y}"),
        ui.kpi_html("Listing-day open", f"{cur}{_open:,.2f}"),
        ui.kpi_html("Day range", f"{cur}{float(_row['Low']):,.2f} – {cur}{float(_row['High']):,.2f}", None, "", True),
        ui.kpi_html("Volume", fmt_big(float(_row["Volume"]))),
        ui.kpi_html("Needed to forecast", f"{dl.MIN_ROWS_REQUIRED} sessions", "about 14 months of trading",
                    "", True),
    ])
    st.warning(
        f"**{profile.get('name', ticker)} has only {_sessions} trading session"
        f"{'s' if _sessions != 1 else ''} of price history — it has just listed.**\n\n"
        f"Every forecast here is trained on the symbol's own past: the indicators need a 200-day average, "
        f"and each horizon's model is validated across three separate past periods "
        f"({dl.MIN_ROWS_REQUIRED}+ trading days in total). With this little data there is nothing to learn "
        f"from, so a price target would be a made-up number rather than a prediction."
    )
    st.info(
        "**What you can do today**\n\n"
        "• Open **🚀 IPO Center** in the sidebar — newly listed companies are covered there with their issue "
        "price, subscription numbers, grey-market premium and listing-day performance.\n\n"
        "• The price above is live, so you can still watch it move.\n\n"
        "• Once the stock has roughly 14 months of trading, the forecasts, candlestick reading and technical "
        "analysis start working here automatically."
    )

    # The models can't run, but the market is open -- show the live tape anyway.
    ui.section("01", "Live market", "intraday price · live quote · auto-refresh")
    _refresh_new = st.select_slider("Refresh every", options=[1, 2, 3, 5, 10, 30, 60], value=2,
                                    format_func=lambda v: f"{v}s", key="live_refresh_s",
                                    help="How often the live panel re-fetches the price.")
    st.fragment(run_every=_refresh_new)(_live_body)(
        ticker, raw, cur, tz_name if "tz_name" in globals() else "America/New_York")
    st.caption("This panel is live even though the forecasts are not: the price, day range and volume come "
               "straight from the exchange feed, which a brand-new listing has from its first minute.")
    st.stop()

tldr_slot = st.empty()  # 10-second summary, written once every section has run
if IS_HISTORICAL:
    _cur_px = None
    try:
        _q_now = cached_quote(ticker)
        _cur_px = float(_q_now["price"]) if _q_now and _q_now.get("price") else None
    except Exception:
        _cur_px = None
    _hist_px = float(raw["Close"].iloc[-1])
    _delta = ((_cur_px / _hist_px - 1) * 100) if _cur_px else None
    st.warning(
        f"🕰️ **HISTORICAL MODE — as of {AS_OF:%d %b %Y}.** Every number below is built only from data "
        f"available up to that date: the live quote is switched off, the auto-refreshing panels are "
        f"disabled, and the models see nothing later. Clear the *History end date* in the sidebar to "
        f"return to live mode.")
    ui.kpi_grid([
        ui.kpi_html(f"Close on {raw.index[-1]:%d %b %Y}", f"{cur}{_hist_px:,.2f}",
                    "the selected historical date", "", True),
        ui.kpi_html("Price right now", f"{cur}{_cur_px:,.2f}" if _cur_px else "n/a",
                    "live market, shown for reference only", "", True),
        ui.kpi_html("Moved since then", f"{_delta:+.2f}%" if _delta is not None else "n/a",
                    "not used by anything below",
                    "up" if (_delta or 0) >= 0 else "down"),
        ui.kpi_html("Candles available", f"{len(raw):,}", f"{raw.index[0]:%d %b %Y} → "
                                                          f"{raw.index[-1]:%d %b %Y}", "", True),
    ])

tape_slot = st.empty()  # market ticker tape, filled once market/macro data has loaded (below)
score_slot = st.empty()  # model scorecard, filled once the forecast engine has trained (below)
if IS_HISTORICAL:
    _header_body(ticker, raw, cur, profile, False,
                 tz_name if "tz_name" in globals() else "America/New_York")
else:
    st.fragment(run_every=st.session_state.get('live_refresh_s', 2))(_header_body)(
        ticker, raw, cur, profile, market_open_now,
        tz_name if 'tz_name' in globals() else 'America/New_York')

_cls = profile.get("asset_class", "stock")
if profile.get("summary"):
    with st.expander("🏢 About the company" if _cls == "stock" else "ℹ️ About this instrument"):
        st.write(profile["summary"])
        if profile.get("website"):
            st.markdown(f"[{profile['website']}]({profile['website']})")
elif _cls != "stock":
    _kind, _drivers = ra.ASSET_KIND_TEXT.get(_cls, ("instrument", "broad market conditions"))
    with st.expander("ℹ️ About this instrument"):
        st.write(f"**{profile.get('name', ticker)}** ({ticker}) is a {_kind} quoted in "
                 f"{profile.get('currency', 'USD')}. Its price is driven mainly by {_drivers}. "
                 f"Everything below — the forecasts, candlestick reading, technicals and the research "
                 f"report — is computed from its own price history, exactly as it is for a share.")

# ----------------------------- LIVE MARKET (auto-refreshing) ---------------------------------
if IS_HISTORICAL:
    ui.section("01", "Live market", "switched off in historical mode")
    st.info(f"The live intraday panel is disabled while the app is replaying {AS_OF:%d %b %Y}. "
            f"Showing today's tape next to a historical analysis is exactly the confusion this mode "
            f"exists to prevent.")
_refresh_s = st.select_slider("Refresh every", options=[1, 2, 3, 5, 10, 30, 60], value=2,
                              format_func=lambda v: f"{v}s", key="live_refresh_s",
                              help="How often the live panel re-fetches the price. Yahoo's quote updates every "
                                   "~5-15 seconds, so 2s won't show new prices more often -- it just reacts sooner.")
st.caption("Auto-refresh pauses while the rest of the page is still computing (the first run for a new "
           "stock trains the models, about 1-2 minutes) and starts ticking by itself as soon as that finishes.")
# st.fragment is applied at runtime so the interval can be changed from the UI.
if not IS_HISTORICAL:
    ui.section("01", "Live market", "intraday price · live quote · auto-refresh")
    st.fragment(run_every=_refresh_s)(_live_body)(
        ticker, raw, cur, tz_name if "tz_name" in globals() else "America/New_York")

# ----------------------------- HERO CHART: interactive candlestick chart ---------------------------------
ui.section("02", "Price history", "daily candles · volume · overlays")
ind_df = ta.add_overlay_indicators(raw)
hc1, hc2 = st.columns([1.4, 3])
with hc1:
    period = st.radio("Timeframe", ["3M", "6M", "1Y", "2Y", "5Y"], index=1, horizontal=True)
with hc2:
    overlays = st.multiselect(
        "Chart overlays",
        ["SMA 5", "SMA 20", "SMA 50", "SMA 200", "Exp. smoothing (α=0.3)", "Bollinger Bands", "Candlestick patterns",
         "Support / Resistance"],
        default=["SMA 20", "SMA 50", "Candlestick patterns"],
    )
n_bars = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}[period]
recent = ind_df.tail(n_bars)

fig_hero = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
fig_hero.add_trace(go.Candlestick(
    x=recent.index, open=recent["Open"], high=recent["High"],
    low=recent["Low"], close=recent["Close"], name=ticker.upper(),
    increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN,
), row=1, col=1)
for name, col, color in [("SMA 5", "SMA_5", ui.SERIES[3]), ("SMA 20", "SMA_20", ui.SERIES[0]),
                         ("SMA 50", "SMA_50", ui.SERIES[1]), ("SMA 200", "SMA_200", ui.SERIES[2])]:
    if name in overlays:
        fig_hero.add_trace(go.Scatter(x=recent.index, y=recent[col], name=name,
                                      line=dict(color=color, width=1.4)), row=1, col=1)
if "Exp. smoothing (α=0.3)" in overlays:
    from model_smoothing import exponential_smoothing, SMOOTHING_ALPHA
    _ses = exponential_smoothing(ind_df["Close"]).tail(n_bars)  # computed on full history, then trimmed
    fig_hero.add_trace(go.Scatter(x=_ses.index, y=_ses.values, name=f"Exp. smoothing (α={SMOOTHING_ALPHA})",
                                  line=dict(color=ui.SERIES[6], width=2)), row=1, col=1)
if "Bollinger Bands" in overlays:
    fig_hero.add_trace(go.Scatter(x=recent.index, y=recent["BB_Upper"], name="BB upper",
                                  line=dict(color="rgba(100,116,139,0.6)", width=1)), row=1, col=1)
    fig_hero.add_trace(go.Scatter(x=recent.index, y=recent["BB_Lower"], name="BB lower", fill="tonexty",
                                  fillcolor="rgba(100,116,139,0.08)",
                                  line=dict(color="rgba(100,116,139,0.6)", width=1)), row=1, col=1)
if "Candlestick patterns" in overlays:
    ev_hero = cp.pattern_events(raw.tail(n_bars + 30))
    ev_hero = ev_hero[(ev_hero["Date"] >= recent.index[0]) & (ev_hero["Bias_Num"] != 0)]
    for bias, sym, color, ycol, off in [(1, "triangle-up", ui.UP, "Low", 0.985),
                                        (-1, "triangle-down", ui.DOWN, "High", 1.015)]:
        e = ev_hero[ev_hero["Bias_Num"] == bias]
        if not e.empty:
            fig_hero.add_trace(go.Scatter(
                x=e["Date"], y=[raw.at[d, ycol] * off for d in e["Date"]], mode="markers",
                name="Bullish pattern" if bias > 0 else "Bearish pattern",
                marker=dict(symbol=sym, size=[7 + 3 * s for s in e["Strength"]], color=color),
                text=e["Pattern"], hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
            ), row=1, col=1)
if "Support / Resistance" in overlays:
    for lvl in ta.support_resistance_levels(raw):
        fig_hero.add_hline(y=lvl["level"], line_dash="dot", line_width=1,
                           line_color=ui.UP if lvl["kind"] == "support" else ui.DOWN,
                           annotation_text=f"{lvl['kind'].title()} {cur}{lvl['level']:.2f} ({lvl['touches']}×)",
                           annotation_position="top left", annotation_font_size=10, row=1, col=1)
vol_colors = [ui.UP if c >= o else ui.DOWN for o, c in zip(recent["Open"], recent["Close"])]
fig_hero.add_trace(go.Bar(x=recent.index, y=recent["Volume"], name="Volume", marker_color=vol_colors,
                          opacity=0.45, showlegend=False, marker_line_width=0), row=2, col=1)
fig_hero.update_layout(
    height=600, margin=dict(l=10, r=64, t=30, b=10),  # right margin: room for the right-side price axis
    xaxis_rangeslider_visible=False,
    template=PLOTLY_TEMPLATE,
)
fig_hero.update_xaxes(rangebreaks=WEEKEND_BREAK)
fig_hero.update_yaxes(title_text="Volume", row=2, col=1)
fig_hero.update_yaxes(side="right", row=1, col=1)
st.plotly_chart(fig_hero, width="stretch", config={"displaylogo": False})

# ------------- NEWS + MODEL TRAINING (after the price header, so prices show immediately) -------------
news_df, news_info, news_sig = None, {"sources": [], "notes": ["News disabled in the sidebar."]}, None
if use_news:
    _load(18, "Reading the news")
    with st.spinner("📰 Loading news history (GDELT; first time for a stock can take up to ~1.5 min)..."):
        try:
            news_df, news_info = cached_news_features(ticker, history_start, history_end, profile.get("name", ticker),
                                                      bool(av_key), av_key)
        except Exception as e:
            news_df, news_info = None, {"sources": [], "notes": [f"News history unavailable: {e}"]}
    if news_df is not None and news_df.shape[1] > 0:
        news_sig = (tuple(news_df.columns), str(news_df.index.max().date()), len(news_df))
    else:
        news_df = None

# Social platforms keep no free archive, so the history is whatever this
# dashboard has collected day by day; it becomes a model input once there is
# enough of it, and is tested/rejected like every other group.
social_df, social_info, social_sig = None, {"days": 0, "usable": False}, None
if use_social:
    try:
        social_df, social_info = cached_social_features(ticker, str(datetime.date.today()), raw.index)
        if social_df is not None and social_df.shape[1]:
            social_sig = (tuple(social_df.columns), int(social_info.get("days", 0)))
    except Exception as e:
        social_df, social_info = None, {"days": 0, "usable": False, "note": f"social history unavailable: {e}"}

pooled_df = None
if use_universe:
    _load(30, "Asking the universe model")
    with st.spinner("🌐 Asking the universe model (38 symbols) what it expects..."):
        try:
            import pooled_model as pmod
            pooled_df = pmod.pooled_view_for(raw, ticker)
            if pooled_df is not None and pooled_df.empty:
                pooled_df = None
        except Exception as e:
            st.info(f"Universe model unavailable ({e}).")
            pooled_df = None

_load(38, "Loading market & macro data")
with st.spinner("🌐 Loading market, macro, earnings & dividend data..."):
    extra = cached_extra_data(ticker, history_start, history_end)
data_sig = str(last_date.date())  # retrain when a new trading day arrives


def _tape_items(extra, ticker):
    """Latest value + day change for the market/macro series the engine downloaded."""
    items = []
    indian = ticker.upper().endswith((".NS", ".BO"))
    mk = extra.get("market")
    try:
        from data_loader import market_symbols_for
        sector_sym = market_symbols_for(ticker).get("sector_etf", "").lstrip("^")
    except Exception:
        sector_sym = ""
    if mk is not None and len(mk) > 1:
        for col, name in (("SP500_Close", "NIFTY 50" if indian else "S&P 500"),
                          ("VIX_Close", "India VIX" if indian else "VIX"),
                          ("Sector_Close", f"Sector · {sector_sym}" if sector_sym else "Sector")):
            s_ = mk[col].dropna()
            if len(s_) > 1 and not (col == "Sector_Close" and s_.equals(mk["SP500_Close"].dropna())):
                items.append((name, f"{s_.iloc[-1]:,.2f}", s_.iloc[-1] / s_.iloc[-2] - 1))
    mc = extra.get("macro")
    if mc is not None and len(mc) > 1:
        for col, name, fmt in (("US_Mkt", "S&P 500", "{:,.2f}"), ("USDINR", "USD/INR", "{:,.2f}"),
                               ("US10Y", "US 10Y", "{:.2f}%"), ("Dollar", "Dollar idx", "{:,.2f}"),
                               ("Gold", "Gold", "${:,.0f}"), ("Oil", "Crude", "${:,.2f}")):
            if col in mc.columns:
                s_ = mc[col].dropna()
                if len(s_) > 1:
                    items.append((name, fmt.format(s_.iloc[-1]), s_.iloc[-1] / s_.iloc[-2] - 1))
    return items


try:
    tape_slot.markdown(ui.tape_html(_tape_items(extra, ticker)), unsafe_allow_html=True)
except Exception:
    pass
def _render_scorecard(_engine, _last_price):
    """Model quality at a glance -- all held-out numbers the engine already has."""
    try:
        models = getattr(_engine, "models", {}) or {}
        if not models:
            return
        skills, alphas = [], []
        for h, m in models.items():
            vr, nr = getattr(m, "val_rmse", None), getattr(m, "val_naive_rmse", None)
            if vr and nr:
                skills.append((h, (1 - vr / nr) * 100))
            a = getattr(m, "alpha", None)
            if a is not None:
                alphas.append(a)
        if not skills:
            return
        by_h = dict(skills)
        best_h, best_skill = max(skills, key=lambda x: x[1])
        beat = sum(1 for _, v in skills if v > 0)
        m21 = models.get(21) or models.get(max(models))
        alpha21 = float(getattr(m21, "alpha", 0) or 0)
        groups = len(getattr(m21, "groups_used", []) or [])
        total_groups = len(getattr(_engine, "groups", {}) or {}) or 1

        # 80% band width at one month, as a % of price
        spread = None
        try:
            _p = _engine.forecast_path(21)
            row = _p.iloc[min(20, len(_p) - 1)]
            spread = float(row["Upper_80"] - row["Lower_80"]) / _last_price * 100
        except Exception:
            pass

        def tone(v, good=0.0):
            return "up" if v > good else ("down" if v < good else "")

        cards = [
            ui.kpi_html("Model skill vs “no change”", f"{by_h.get(21, best_skill):+.2f}%",
                        f"at 1 month · best {best_skill:+.2f}% at {best_h}d",
                        tone(by_h.get(21, best_skill)), True),
            ui.kpi_html("Horizons with an edge", f"{beat} / {len(skills)}",
                        "beat the naive forecast out-of-sample",
                        "up" if beat > len(skills) / 2 else "down", True),
            ui.kpi_html("Signal strength (α)", f"{alpha21:.2f}",
                        "0 = no view trusted · 1 = fully trusted",
                        "up" if alpha21 > 0.25 else "", True),
            ui.kpi_html("Data groups used", f"{groups} / {total_groups}",
                        "families that survived selection", "", True),
        ]
        if spread is not None:
            cards.append(ui.kpi_html("1-month uncertainty", f"±{spread / 2:.1f}%",
                                     "80% of outcomes land inside this", "", True))
        with score_slot.container():
            ui.render('<div class="section"><span class="kicker">00</span>'
                      '<span class="title">Model scorecard</span>'
                      '<span class="sub">held-out accuracy · measured, not claimed</span></div>')
            ui.kpi_grid(cards)
    except Exception:
        pass


_load(48, f"Training 5 forecast horizons for {ticker}")
with st.spinner(f"🧠 Training the forecast engine for {ticker}: 5 horizons × feature-group selection "
                f"(first run takes 1–2 minutes)..."):
    try:
        engine = cached_engine(ticker, history_start, history_end, data_sig + "|" + raw.attrs["mode"],
                               news_sig, news_df, extra,
                               social_sig=social_sig, _social_df=social_df,
                               pooled_sig=bool(pooled_df is not None), _pooled_df=pooled_df)
    except Exception as e:
        st.error(f"Could not train on '{ticker}': {e}")
        st.stop()
if 1 not in engine.models:
    # Enough rows to reach this point, but not enough LABELLED rows for even the
    # 1-day model (each horizon needs 300 rows whose outcome is already known).
    st.warning(
        f"**{profile.get('name', ticker)} has {len(raw)} trading sessions - still too few to train on.** "
        f"Each horizon's model needs {dl.MIN_ROWS_REQUIRED} sessions that already have a known outcome, so a "
        f"recently listed stock reaches this point a few weeks before the forecasts can start. The live "
        f"price, chart, news and social panels still work; check back once it has about 14 months of trading."
    )
    st.stop()
_load(78, "Scoring the model")
_render_scorecard(engine, last_price)   # fills the slot reserved near the top

m1 = engine.models[1]
weights = m1.weights



# ----------------------------- AI SNAPSHOT ---------------------------------

# ----------------------------- RESEARCH ANALYST ---------------------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_company_data(ticker, as_of=None):  # audit-exempt: fundamentals, not bars
    # as_of is in the key AND passed through, so a historical replay's research report is
    # valued on that date's price rather than today's.
    return ra.gather_company_data(ticker, as_of=as_of)


# Friendly names for the benchmark each asset is measured against.
BENCH_NAMES = {"^GSPC": "the S&P 500", "^NSEI": "the NIFTY 50", "^IXIC": "the Nasdaq",
               "DX-Y.NYB": "the US dollar index", "BTC-USD": "bitcoin", "ETH-USD": "ether",
               "GC=F": "gold", "SI=F": "silver", "^NSEBANK": "the Nifty Bank index"}


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_asset_research(ticker, start, end, _profile, tech_score, _forecast_1m):
    """Research report for a commodity / currency / crypto / ETF / index."""
    hist = cached_load_data(ticker, start, end)
    import data_loader as _dl
    bench_sym = _dl.market_symbols_for(ticker).get("index_symbol", "^GSPC")
    try:
        bench = cached_load_data(bench_sym, start, end)
    except Exception:
        bench = None
    return ra.analyze_asset(hist, ticker, _profile, technical_score=tech_score, forecast_1m=_forecast_1m,
                            bench_hist=bench, bench_name=BENCH_NAMES.get(bench_sym, bench_sym))


is_company = profile.get("asset_class", "stock") == "stock"
_spin = ("🧾 The research analyst is reading the financial statements..." if is_company else
         "🧾 The research analyst is studying the price history, trend and volatility...")
_load(88, "Running the research analyst")
with st.spinner(_spin):
    try:
        _tech_now = float(ta.technical_score_series(raw)["Tech_Score"].iloc[-1])
        _p21 = engine.forecast_path(21).iloc[-1]
        _f1m = {"pct": _p21["Predicted_Close"] / last_price - 1, "lower": _p21["Lower_80"], "upper": _p21["Upper_80"],
                "price": float(_p21["Predicted_Close"]),
                "change_pct": (_p21["Predicted_Close"] / last_price - 1) * 100}
        if is_company:
            research = ra.analyze(cached_company_data(ticker, AS_OF if IS_HISTORICAL else None), technical_score=_tech_now, forecast_1m=_f1m)
        else:
            # No revenue or P/E exists for gold or EURUSD -- score trend, momentum,
            # volatility, drawdowns and strength against a benchmark instead.
            research = cached_asset_research(ticker, history_start, history_end, profile, _tech_now, _f1m)
    except Exception as e:
        research = None
        st.warning(f"Research report unavailable: {e}")

ui.section("03", "AI snapshot", "everything the models and indicators say, at a glance")
if research and research.get("overall") is not None:
    _vc, _ = ui.score_color(research["overall"])
    ui.render(f"""<div class="rcard" style="display:flex;gap:20px;align-items:center;flex-wrap:wrap">
    {ui.ring_html(research["overall"])}<div style="flex:1;min-width:240px">
    <div class="verdict-kicker">🧾 Research analyst verdict</div>
    <div style="font-size:1.4rem;font-weight:800;color:{_vc}">{research["verdict"]} · {research["overall"]:.1f}/10</div>
    <div class="verdict-stance" style="margin:2px 0 0 0">{research["stance"]}</div>
    <div class="verdict-text" style="font-size:0.84rem">Full report in the <b>🧾 Research Report</b> tab below:
    {"fundamentals, financial trends, valuation, price history, ownership & analysts"
     if is_company else "trend, momentum, volatility, drawdowns, seasonality and relative strength"}.
    </div></div></div>""")
snap_cards = []
try:
    _step_range = pd.date_range if TRADES_WEEKENDS else pd.bdate_range
    _next_day = _step_range(engine.table.index[-1] + pd.Timedelta(days=1), periods=1)[0]
    nxt = engine.predict_date(_next_day, buy_threshold, sell_threshold)
    tone = {"BUY": "up", "SELL": "down"}.get(nxt["signal"], "neutral")
    snap_cards.append(ui.snap_html(
        f"Next session · {nxt['target_date'].strftime('%d %b')}", f"{cur}{nxt['predicted_price']:,.2f}",
        f"{nxt['pct_change'] * 100:+.2f}% · model says {nxt['signal']}", tone, SIGNAL_STYLE[nxt["signal"]][2]))
except Exception:
    pass
try:
    _cm, _cw, _adv = cached_train_direction(ticker, history_start, history_end, news_sig, news_df)
    _ind_info = _adv.attrs.get("indicator_info", {})
    p_up, _ = dl.get_direction_signal(_cm, _cw, _adv)
    d_tone, d_icon, d_txt = (("up", "▲", "Likely UP") if p_up >= 0.55 else
                             ("down", "▼", "Likely DOWN") if p_up <= 0.45 else ("neutral", "●", "Uncertain"))
    snap_cards.append(ui.snap_html("Direction model", d_txt, f"{p_up * 100:.0f}% probability of an up day",
                                   d_tone, d_icon))
except Exception:
    pass
try:
    _ev, _rel = cached_candle_analysis(ticker, history_start, history_end, 5)
    _reading = cp.read_candles(raw, lookback=5, reliability=_rel)
    _rating, _score, _votes = ta.technical_scorecard(raw, _reading)
    t_tone = "up" if "Buy" in _rating else "down" if "Sell" in _rating else "neutral"
    snap_cards.append(ui.snap_html("Technical rating", _rating, f"{len(_votes)} indicator votes · score {_score:+.2f}",
                                   t_tone, {"up": "▲", "down": "▼"}.get(t_tone, "●")))
    c_tone = {"Bullish": "up", "Bearish": "down"}.get(_reading["verdict"], "neutral")
    snap_cards.append(ui.snap_html("Candlestick reading", _reading["verdict"],
                                   f"last 5 candles · {len(_reading['events'])} pattern(s)",
                                   c_tone, {"up": "▲", "down": "▼"}.get(c_tone, "●")))
except Exception:
    pass
try:
    _news = cached_news_sentiment(ticker, nh.company_query(profile.get("name", ticker)))
    if _news["article_count"]:
        n_tone = {"Positive": "up", "Negative": "down"}.get(_news["label"], "neutral")
        snap_cards.append(ui.snap_html("News mood", _news["label"],
                                       f"{_news['average_score']:+.2f} avg · {_news['article_count']} headlines",
                                       n_tone, {"up": "▲", "down": "▼"}.get(n_tone, "●")))
except Exception:
    pass
if use_social:
    try:
        _snap_social = cached_social(ticker, profile.get("name", ticker), x_token,
                                     int(time.time() // 60))
        if _snap_social["total_posts"]:
            s_tone = ("up" if (_snap_social["mood"] or 0) >= 0.05 else
                      "down" if (_snap_social["mood"] or 0) <= -0.05 else "neutral")
            snap_cards.append(ui.snap_html(
                "Social mood", _snap_social["mood_label"],
                f"{_snap_social['mood']:+.2f} avg · {_snap_social['total_posts']} posts · "
                f"{len(_snap_social['sources'])} platform(s)",
                s_tone, {"up": "▲", "down": "▼"}.get(s_tone, "●")))
    except Exception:
        pass

_alpha = m1.alpha
snap_cards.append(ui.snap_html("Model signal strength (1-day)", f"{_alpha:.2f}",
                               "no reliable edge — forecast ≈ today" if _alpha < 0.05 else
                               f"uses {len(m1.groups_used)} of {len(engine.groups)} data groups", "info", "◆"))
ui.snap_grid(snap_cards)

# Forecast ladder: dedicated models for each horizon
_path90 = engine.forecast_path(63)
_rungs = []
for label, d in (("Next day", 1), ("1 week", 5), ("2 weeks", 10), ("1 month", 21), ("3 months", 63)):
    p = _path90.iloc[d - 1]
    # The 80% band is +/- 1.2816 sigma, so sigma (and therefore the odds of
    # finishing above today) comes straight back out of the band the model drew.
    _sigma = (p["Upper_80"] - p["Lower_80"]) / (2 * 1.2816 * last_price)
    _move = p["Predicted_Close"] / last_price - 1
    _prob_up = 50.0
    if _sigma > 0:
        from math import erf, sqrt
        _prob_up = 100 * 0.5 * (1 + erf((_move / _sigma) / sqrt(2)))
    _rungs.append((f"{label} · {_path90.index[d - 1].strftime('%d %b')}", p["Predicted_Close"],
                   _move, p["Lower_80"], p["Upper_80"], _prob_up, _sigma * 1.2816 * 100))
ui.ladder(_rungs, cur)
_vol_now = fe.trailing_vol(raw, 21).iloc[-1]
_vol_avg = fe.trailing_vol(raw, 21).tail(756).mean()
_vol_word = ("calmer than usual" if _vol_now < _vol_avg * 0.9 else
             "more turbulent than usual" if _vol_now > _vol_avg * 1.1 else "about as volatile as usual")
st.caption(f"At these horizons the central number is close to today's price on purpose: the models are only "
           f"trusted as far as they beat 'no change' on past data, and for liquid markets that is barely. "
           f"The **chance of finishing higher** and the **80% range** are the parts that carry information — "
           f"a 55% chance with a ±10% swing is a very different bet from 55% with ±2%. "
           f"The range itself now breathes with the market: {ticker} is **{_vol_word}** right now "
           f"({_vol_now * 100:.1f}% vs {_vol_avg * 100:.1f}% typical for a month), so the bands are "
           f"{'narrower' if _vol_now < _vol_avg else 'wider'} than its long-run average. They are also "
           f"**asymmetric**: the edges come from the 10th and 90th percentiles of the model's own past "
           f"misses rather than a symmetric bell curve, so a symbol that historically fell harder than it "
           f"rose gets a longer tail downwards.")
st.caption(f"**Why the dashed centre line looks flat.** The models are kept only as far as "
           f"they beat 'no change' out of sample, and at a month on a liquid symbol that margin "
           f"is tiny, so most of their view is deliberately shrunk away. Two attempts to centre "
           f"the forecast on historical drift instead were measured on held-out data and both "
           f"did worse, so the flat centre stays. The shaded bands are what the model does know: "
           f"the innermost holds half of the simulated outcomes, the outermost 95%. They widen "
           f"with time and they are not symmetric. Read them, not the dashed line — "
           f"**expect movement, not a callable direction.**")

@st.cache_data(show_spinner=False, ttl=3600)
def cached_fan(ticker, start, end, n_days, central, sigma, sig):
    """Simulated paths for the outlook chart, anchored on the model's own centre/spread."""
    return sx.simulate_paths(cached_load_data(ticker, start, end), n_days,
                             target_return=central, target_sigma=sigma)


_f21 = _path90.iloc[20]
_c21 = float(_f21["Predicted_Close"]) / last_price - 1
_s21 = float(_f21["Upper_80"] - _f21["Lower_80"]) / (2 * 1.2816 * last_price)
try:
    _fan = cached_fan(ticker, history_start, history_end, 21, _c21, _s21, data_sig)
except Exception:
    _fan = None

_hist = raw["Close"].tail(90)
fig_out = go.Figure()
fig_out.add_trace(go.Scatter(x=_hist.index, y=_hist.values, name="Actual close",
                             line=dict(color=ui.SERIES[0], width=2), fill="tozeroy",
                             fillcolor="rgba(57,135,229,0.06)"))
ui.forecast_cone(fig_out, _path90.head(21), color=ui.SERIES[1], name="21-day forecast", paths=_fan)
_lo = min(_hist.min(), _path90.head(21)["Lower_80"].min()) * 0.98
_hi = max(_hist.max(), _path90.head(21)["Upper_80"].max()) * 1.02
fig_out.update_layout(height=380, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=64, t=76, b=10),
                      title=dict(text="21-day outlook", y=0.96, yanchor="top"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                  xanchor="left", x=0, font=dict(size=10)),
                      yaxis=dict(range=[_lo, _hi], side="right"))
fig_out.update_xaxes(rangebreaks=WEEKEND_BREAK)
st.plotly_chart(fig_out, width="stretch", config={"displaylogo": False})





# ----------------------------- SCENARIO LAB ---------------------------------
ui.section("04", "Scenario lab", "the range of outcomes behind the forecast · simulated from this symbol's own history")


@st.cache_data(show_spinner=False, ttl=3600)
def cached_paths(ticker, start, end, n_days, central, sigma, sig):
    """Bootstrapped price paths, anchored on the model's own centre and spread."""
    raw_ = cached_load_data(ticker, start, end)
    return sx.simulate_paths(raw_, n_days, target_return=central, target_sigma=sigma)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_analogs(ticker, start, end, n_days, sig):
    return sx.regime_analogs(cached_load_data(ticker, start, end), n_days)


_sc1, _sc2 = st.columns([1, 1])
with _sc1:
    _sc_days = st.select_slider("Look ahead", options=[5, 10, 21, 42, 63], value=21,
                                format_func=lambda d: {5: "1 week", 10: "2 weeks", 21: "1 month",
                                                       42: "2 months", 63: "3 months"}[d],
                                key="scenario_days")
with _sc2:
    _sc_target = st.number_input(f"Target price ({cur})", value=float(round(last_price * 1.05, 2)),
                                 step=float(max(round(last_price * 0.01, 2), 0.01)),
                                 help="Any price you care about — the simulation reports how often the paths "
                                      "reach it, and how long they take.")

_sc_row = engine.forecast_path(_sc_days).iloc[-1]
_sc_central = float(_sc_row["Predicted_Close"]) / last_price - 1
_sc_sigma = (float(_sc_row["Upper_80"]) - float(_sc_row["Lower_80"])) / (2 * 1.2816 * last_price)
_sc_paths = cached_paths(ticker, history_start, history_end, _sc_days, round(_sc_central, 6),
                         round(_sc_sigma, 6), data_sig)

if _sc_paths.size == 0:
    st.info("Not enough history to simulate scenarios for this symbol.")
else:
    _pr = sx.probabilities(_sc_paths, last_price, target_price=_sc_target)
    _pct = sx.path_percentiles(_sc_paths)
    _dates = (pd.date_range(raw.index[-1] + pd.Timedelta(days=1), periods=_sc_days, freq="D")
              if TRADES_WEEKENDS else
              pd.bdate_range(raw.index[-1] + pd.Timedelta(days=1), periods=_sc_days))

    _tgt_up = _sc_target >= last_price
    ui.kpi_grid([
        ui.kpi_html("Chance of finishing higher", f"{_pr['p_up']:.0f}%",
                    f"median outcome {_pr['median_return']:+.1f}%",
                    "up" if _pr["p_up"] >= 50 else "down"),
        ui.kpi_html("Good case (top 10%)", f"{_pr['best_10pct']:+.1f}%",
                    f"{cur}{last_price * (1 + _pr['best_10pct'] / 100):,.2f}", "up"),
        ui.kpi_html("Bad case (bottom 10%)", f"{_pr['worst_10pct']:+.1f}%",
                    f"{cur}{last_price * (1 + _pr['worst_10pct'] / 100):,.2f}", "down"),
        ui.kpi_html("Chance of +5% or more", f"{_pr['p_gain_5']:.0f}%", "by the end of the window", "up"),
        ui.kpi_html("Chance of −5% or worse", f"{_pr['p_fall_5']:.0f}%", "by the end of the window", "down"),
        ui.kpi_html(f"Reaches {cur}{_sc_target:,.2f}", f"{_pr.get('p_touch_target', 0):.0f}%",
                    (f"typically in {_pr['median_days_to_target']:.0f} days" if _pr.get("median_days_to_target")
                     else "at any point in the window"), "up" if _tgt_up else "down"),
        ui.kpi_html("Typical peak on the way", f"{_pr['expected_max_gain']:+.1f}%",
                    "median of the highest point reached", "up"),
        ui.kpi_html("Typical dip on the way", f"{_pr['expected_max_drop']:+.1f}%",
                    "median of the lowest point reached", "down"),
    ])

    _vo = engine.vol_outlook(21)
    _risk = [
        ui.kpi_html("Value at Risk (95%)", f"{_pr['var_95']:+.1f}%",
                    f"{cur}{last_price * (1 + _pr['var_95'] / 100):,.2f} — worse on 1 day in 20", "down"),
        ui.kpi_html("Expected shortfall", f"{_pr['expected_shortfall_95']:+.1f}%",
                    "average outcome in that worst 5%", "down"),
        ui.kpi_html("Chance of a 10% fall", f"{_pr['p_drawdown_10']:.0f}%",
                    "touched at any point in the window", "down"),
        ui.kpi_html("Chance of a 20% fall", f"{_pr['p_drawdown_20']:.0f}%", "same window", "down"),
        ui.kpi_html("Upside / downside", f"{_pr['upside_downside_ratio']:.2f}×"
                    if _pr.get("upside_downside_ratio") else "n/a",
                    "good case ÷ bad case",
                    "up" if (_pr.get("upside_downside_ratio") or 0) >= 1 else "down"),
    ]
    if _vo.get("forecast"):
        _vchg = (_vo["forecast"] / _vo["realised"] - 1) * 100 if _vo.get("realised") else None
        _risk.append(ui.kpi_html(
            "Volatility, next month", f"{_vo['forecast'] * 100:.1f}%",
            (f"{'rising' if _vchg > 3 else 'falling' if _vchg < -3 else 'steady'} vs "
             f"{_vo['realised'] * 100:.1f}% now" if _vchg is not None else "annualised"),
            "down" if (_vchg or 0) > 3 else "up" if (_vchg or 0) < -3 else ""))
    ui.kpi_grid(_risk)
    if _vo.get("forecast"):
        st.caption(f"Volatility is the one thing here that genuinely is forecastable, so it gets its own model: "
                   f"**HAR-RV** (Corsi 2009) on daily, weekly, monthly and quarterly realised volatility plus a "
                   f"leverage term for the way volatility jumps after falls. On held-out data it beat "
                   f"'next month looks like last month' by "
                   f"{_vo.get('skill_vs_trailing', 0):.0f}% for this symbol (12.1% on average across 24 test "
                   f"cases). It is reported here rather than used for the bands, because the bands calibrated "
                   f"better against trailing volatility — measured, not assumed.")

    fig_sc = go.Figure()
    _band = [("p5", "p95", "rgba(90,140,255,0.13)", "5–95% of outcomes"),
             ("p25", "p75", "rgba(90,140,255,0.28)", "25–75% of outcomes")]
    for lo_k, hi_k, fill, name in _band:
        fig_sc.add_trace(go.Scatter(x=list(_dates) + list(_dates[::-1]),
                                    y=list(_pct[hi_k]) + list(_pct[lo_k][::-1]),
                                    fill="toself", fillcolor=fill, line=dict(width=0),
                                    hoverinfo="skip", name=name))
    for i in range(0, min(25, len(_sc_paths))):
        fig_sc.add_trace(go.Scatter(x=_dates, y=_sc_paths[i * 7 % len(_sc_paths)], mode="lines",
                                    line=dict(width=0.7, color="rgba(160,180,220,0.30)"),
                                    hoverinfo="skip", showlegend=False))
    fig_sc.add_trace(go.Scatter(x=_dates, y=_pct["p50"], mode="lines", name="Median path",
                                line=dict(color=ui.ACCENT, width=2.5)))
    fig_sc.add_hline(y=last_price, line_dash="dot", line_width=1, line_color=ui.NEUTRAL,
                     annotation_text=f"today {cur}{last_price:,.2f}", annotation_position="bottom left",
                     annotation_font_size=10)
    fig_sc.add_hline(y=_sc_target, line_dash="dash", line_width=1.2,
                     line_color=ui.UP if _tgt_up else ui.DOWN,
                     annotation_text=f"target {cur}{_sc_target:,.2f}", annotation_position="top left",
                     annotation_font_size=10)
    fig_sc.update_layout(height=430, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=60, t=10, b=10),
                         hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0))
    fig_sc.update_yaxes(side="right")
    st.plotly_chart(fig_sc, width="stretch", config={"displaylogo": False})
    st.caption(f"{len(_sc_paths):,} simulated paths, built by resampling **this symbol's own** daily moves in "
               f"10-day blocks (so its real volatility clustering and fat tails are preserved), then centred on "
               f"the models' forecast ({_sc_central * 100:+.2f}%) and scaled to their validated uncertainty. "
               f"The thin lines are individual paths — the price does move in every one of them; it is the "
               f"*average* of all of them that lands near today, which is exactly why a single predicted number "
               f"looks flat.")

    _an = cached_analogs(ticker, history_start, history_end, _sc_days, data_sig)
    if _an:
        _an_tone = "up" if _an["median"] >= 0 else "down"
        ui.card("📚 When this setup happened before",
                f"<p>Across {_an['n']} past days whose trend, volatility, momentum and drawdown looked most like "
                f"today's, the next {_sc_days} trading days delivered a median of "
                f"<b style='color:{ui.UP if _an['median'] >= 0 else ui.DOWN}'>{_an['median']:+.1f}%</b>, with "
                f"{_an['pct_up']:.0f}% of them positive. The middle half landed between {_an['p25']:+.1f}% and "
                f"{_an['p75']:+.1f}%; the extremes were {_an['worst']:+.1f}% and {_an['best']:+.1f}%. "
                f"This is history for similar conditions, not a forecast — but if the simulation above and "
                f"these analogs disagree, that disagreement is itself worth knowing.</p>")

# ---- the 10-second summary, now that every section has produced its numbers ----
def _write_tldr():
    name = profile.get("name", ticker)
    _when = "today" if not IS_HISTORICAL else f"on {raw.index[-1]:%d %b %Y}"
    day = f"{'up' if day_change_pct >= 0 else 'down'} {abs(day_change_pct):.2f}% {_when}"
    bits = [f"<b>{_html.escape(name)}</b> is {day} at {cur}{last_price:,.2f}."]
    if IS_HISTORICAL:
        bits.append(f"<b>Historical replay as of {AS_OF:%d %b %Y}</b> — nothing after that date is used.")
    tags = []

    horizon_txt = {5: "week", 10: "fortnight", 21: "month", 42: "two months", 63: "quarter"}.get(_sc_days, "period")
    if _sc_paths.size:
        bits.append(
            f"Over the next <b>{horizon_txt}</b> the models see a central outcome of "
            f"<b>{_sc_central * 100:+.2f}%</b> — close to flat, because they only take a view as far as they "
            f"beat 'no change' on past data — but the simulation puts the odds of finishing higher at "
            f"<b>{_pr['p_up']:.0f}%</b>, with a good case of {_pr['best_10pct']:+.1f}% and a bad case of "
            f"{_pr['worst_10pct']:+.1f}%.")
        tags.append((f"{_pr['p_up']:.0f}% chance higher", "up" if _pr["p_up"] >= 50 else "down"))
        tags.append((f"good case {_pr['best_10pct']:+.1f}%", "up"))
        tags.append((f"bad case {_pr['worst_10pct']:+.1f}%", "down"))

    try:
        t_now = float(ta.technical_score_series(raw)["Tech_Score"].iloc[-1])
        t_word = ("strongly bullish" if t_now > 0.5 else "bullish" if t_now > 0.15 else
                  "strongly bearish" if t_now < -0.5 else "bearish" if t_now < -0.15 else "neutral")
        bits.append(f"The indicator scorecard reads <b>{t_word}</b> ({t_now:+.2f} on −1…+1).")
        tags.append((f"technicals {t_word}", "up" if t_now > 0.15 else "down" if t_now < -0.15 else ""))
    except Exception:
        pass

    if research and research.get("overall") is not None:
        kind = "fundamentals" if not research.get("is_asset") else "price behaviour"
        bits.append(f"On {kind} the research analyst scores it <b>{research['verdict']} "
                    f"({research['overall']:.1f}/10)</b> — {research['stance'].lower()}.")
        tags.append((f"research {research['verdict'].lower()}",
                     "up" if research["overall"] >= 6 else "down" if research["overall"] < 4.5 else ""))

    moods = []
    try:
        if _news and _news.get("article_count"):
            moods.append(f"news is {_news['label'].lower()} ({_news['article_count']} headlines)")
    except Exception:
        pass
    try:
        if use_social and _snap_social.get("total_posts"):
            moods.append(f"social chatter is {_snap_social['mood_label'].lower()} "
                         f"({_snap_social['total_posts']} posts)")
    except Exception:
        pass
    if moods:
        bits.append("Right now " + " and ".join(moods) + ".")

    bits.append("<span style='opacity:.65'>Every number here is expanded in the sections below — "
                "and the model only claims an edge where it measured one.</span>")
    tldr_slot.container().markdown("", unsafe_allow_html=True)
    with tldr_slot.container():
        ui.tldr(" ".join(bits), tags)


try:
    _write_tldr()
except Exception:
    pass

# ----------------------------- SOCIAL BUZZ & HEADLINES ---------------------------------
ui.section("05", "Social buzz & headlines",
           "StockTwits · Reddit · YouTube · X · news — fetched live")

_PLATFORM_ICON = {"stocktwits": "💬", "reddit": "👽", "youtube": "▶️", "x": "𝕏"}


def _ago(ts):
    """'3m ago' / '4h ago' / '2d ago' from a UTC timestamp."""
    if ts is None or pd.isna(ts):
        return ""
    mins = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(ts)).total_seconds() / 60
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{int(mins)}m ago"
    if mins < 48 * 60:
        return f"{int(mins // 60)}h ago"
    return f"{int(mins // 1440)}d ago"


def _post_list_html(posts, limit=8):
    if not posts:
        return '<div class="empty">Nothing found on this platform right now.</div>'
    items = []
    for post in posts[:limit]:
        score = post.get("score")
        dot = "🟢" if (score or 0) > 0.05 else "🔴" if (score or 0) < -0.05 else "⚪"
        who = _html.escape(str(post.get("author") or ""))
        when = _ago(post.get("created"))
        meta = " · ".join(x for x in [who, when] if x)
        text = _html.escape((post.get("text") or "")[:180])
        if post.get("url"):
            text = f'<a href="{_html.escape(post["url"])}" target="_blank">{text}</a>'
        tag = post.get("label") or ("Bullish" if (score or 0) > 0.05 else
                                    "Bearish" if (score or 0) < -0.05 else "Neutral")
        items.append(f'<li><span class="tag">{dot} {tag}</span><span>{text}'
                     f'<br><span style="opacity:.6;font-size:0.78rem">{meta}</span></span></li>')
    return "<ul>" + "".join(items) + "</ul>"


def _social_body(ticker, company_name, x_token, news_query):
    """One refresh of the social + headlines panel (wrapped in st.fragment)."""
    c1, c2 = st.columns([3, 1])
    with c2:
        st.caption(f"⟳ live · last checked {datetime.datetime.now():%H:%M:%S}")
    try:
        snap = cached_social(ticker, company_name, x_token, int(time.time() // 60))
    except Exception as e:
        st.info(f"Social feeds unavailable right now ({e}). They will retry on the next refresh.")
        return

    mood, mood_txt = snap["mood"], snap["mood_label"]
    tone = "up" if (mood or 0) >= 0.05 else "down" if (mood or 0) <= -0.05 else ""
    cards = [ui.kpi_html("Social mood", mood_txt if mood is not None else "No data",
                         f"{mood:+.2f} on -1…+1" if mood is not None else "no posts found", tone, small=True),
             ui.kpi_html("Posts tracked", f"{snap['total_posts']}",
                         f"{sum(p['recent_24h'] for p in snap['platforms'].values())} in the last 24h")]
    if snap.get("bull_pct") is not None:
        cards.append(ui.kpi_html("Bullish share", f"{snap['bull_pct']:.0f}%",
                                 "of posts taking a side",
                                 "up" if snap["bull_pct"] >= 50 else "down"))
    for name in sm.PLATFORMS:
        m = snap["platforms"].get(name, {})
        err = snap["errors"].get(name)
        icon = _PLATFORM_ICON.get(name, "")
        if m.get("count"):
            sub = (f"mood {m['mood']:+.2f}" if m.get("mood") is not None else "") + \
                  (f" · {m['bull_pct']:.0f}% bull" if m.get("bull_pct") is not None else "")
            cards.append(ui.kpi_html(f"{icon} {sm.PLATFORM_LABELS[name]}", f"{m['count']} posts", sub,
                                     "up" if (m.get("mood") or 0) >= 0.05 else
                                     "down" if (m.get("mood") or 0) <= -0.05 else ""))
        else:
            reason = ("optional — needs a paid X token" if name == "x" and err and "token" in err.lower()
                      else (err or "nothing found")[:42])
            cards.append(ui.kpi_html(f"{icon} {sm.PLATFORM_LABELS[name]}", "—", reason, "", small=True))
    ui.kpi_grid(cards)

    tabs = st.tabs(["🔥 Most recent posts"] +
                   [f"{_PLATFORM_ICON.get(p, '')} {sm.PLATFORM_LABELS[p]}" for p in sm.PLATFORMS] +
                   ["📰 Headlines"])
    with tabs[0]:
        ui.card("What people are posting right now", _post_list_html(snap["posts"], 10))
    for i, name in enumerate(sm.PLATFORMS, start=1):
        with tabs[i]:
            posts = [post for post in snap["posts"] if post["platform"] == name]
            if name == "x" and x_token:
                st.caption(sm.x_cost_note())
                if st.button("🔄 Fetch X now (billed)", key="x_refetch"):
                    sm.reset_x_cache()
                    cached_social.clear()
                    st.rerun()
            if not posts and snap["errors"].get(name):
                st.info(f"{sm.PLATFORM_LABELS[name]}: {snap['errors'][name]}")
            else:
                ui.card(f"Latest on {sm.PLATFORM_LABELS[name]}", _post_list_html(posts, 12))
    with tabs[-1]:
        try:
            news = cached_news_sentiment(ticker, news_query)
        except Exception as e:
            news = {"article_count": 0, "articles": [], "label": "Unavailable", "average_score": 0.0}
            st.info(f"News feed unavailable: {e}")
        if news["article_count"]:
            rows = []
            for a in news["articles"][:14]:
                dot = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}.get(a["label"], "⚪")
                when = _ago(pd.to_datetime(a.get("publish_time"), errors="coerce", utc=True).tz_localize(None)
                            if a.get("publish_time") else None)
                meta = " · ".join(x for x in [_html.escape(str(a.get("publisher") or "")), when] if x)
                title = _html.escape(a["title"])
                if a.get("link"):
                    title = f'<a href="{_html.escape(a["link"])}" target="_blank">{title}</a>'
                rows.append(f'<li><span class="tag">{dot} {a["score"]:+.2f}</span><span>{title}'
                            f'<br><span style="opacity:.6;font-size:0.78rem">{meta}</span></span></li>')
            ui.card(f"Headlines that can move this stock · overall {news['label'].lower()} "
                    f"({news['average_score']:+.2f})", "<ul>" + "".join(rows) + "</ul>")
        else:
            st.info("No recent headlines found for this symbol.")

    used = "social" in getattr(engine, "groups", {})
    kept = [h for h, mdl in engine.models.items() if "social" in mdl.groups_used]
    if used and kept:
        st.caption(f"✅ Social data is currently used by the {', '.join(f'{h}-day' for h in sorted(kept))} "
                   f"model(s) — it passed the same walk-forward test as every other input.")
    elif used:
        st.caption("Social history is long enough to be tested, but it did not improve accuracy for this "
                   "symbol, so the models left it out.")
    else:
        try:
            days = len(sm.load_history(ticker))      # counted after today's reading was saved
        except Exception:
            days = social_info.get("days", 0)
        st.caption(f"📥 Model input: {days}/{sm.MIN_HISTORY_DAYS} daily readings collected for {ticker}. "
                   f"Social platforms publish no free archive to backfill, so this history only grows from "
                   f"the first day you ran the dashboard; once it is long enough the models test it exactly "
                   f"like every other input and keep it only where it improves accuracy. The live mood above "
                   f"is shown either way.")


if use_social:
    st.fragment(run_every=60)(_social_body)(ticker, profile.get("name", ticker), x_token,
                                            nh.company_query(profile.get("name", ticker)))
else:
    st.caption("Social tracking is switched off in the sidebar.")

ui.section("06", "Deep dive", "research · forecasts · patterns · technicals · model accuracy")

# ----------------------------- TABS ---------------------------------
_load(100, "Done", done=True)

tab_research, tab1, tab_candle, tab2, tab_opt, tab3, tab4 = st.tabs([
    "🧾 Research Report", "🔮 Predict a Date", "🕯️ Candlestick Reading", "🎯 Best Buy/Sell Window",
    "📐 Same-day plan", "📊 Technical Analysis", "🧠 Model Insights"
])

# ===================== TAB: same-day plan (CALL / PUT) =====================
with tab_opt:
    st.subheader("Same-day plan — entry, stop, target, and what it has actually earned")

    @st.cache_data(show_spinner=False, ttl=1800)
    def cached_bracket(ticker, start, end, direction, stop_a, target_a, cost, sig):
        d = cached_load_data(ticker, start, end)
        return opl.build_plan(d, direction, stop_a, target_a), \
            opl.backtest_bracket(d, direction, stop_a, target_a, cost), \
            opl.sweep_brackets(d, direction, cost)

    @st.cache_data(show_spinner=False, ttl=900)
    def cached_chain(ticker, spot, sig):
        return opl.option_chain(ticker, spot)

    oc1, oc2, oc3, oc4 = st.columns(4)
    _dir = oc1.radio("Direction", ["long", "short"], horizontal=True, key="opt_dir",
                     help="long -> you would buy a CALL; short -> you would buy a PUT.")
    _stop_a = oc2.slider("Stop (× ATR)", 0.2, 1.5, 0.8, 0.1, key="opt_stop")
    _tgt_a = oc3.slider("Target (× ATR)", 0.4, 3.0, 1.5, 0.1, key="opt_tgt")
    _ocost = oc4.number_input("Round-trip cost (%)", 0.0, 2.0, 0.30, 0.05, key="opt_cost")

    try:
        _plan, _rec, _sweep = cached_bracket(ticker, history_start, history_end,
                                             _dir, _stop_a, _tgt_a, _ocost, data_sig)
    except Exception as _e:
        _plan, _rec, _sweep = {}, {}, None
        st.error(f"Could not build the plan: {_e}")

    if _rec:
        _exp = _rec["expectancy_pct"]

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Expectancy / trade", f"{_exp:+.3f}%", f"{_rec['sessions']:,} sessions")
        r2.metric("Win rate", f"{_rec['win_rate']:.0f}%",
                  f"target hit {_rec['target_hit_pct']:.0f}% · stopped {_rec['stop_hit_pct']:.0f}%")
        r3.metric("Average win", f"{_rec['avg_win']:+.2f}%")
        r4.metric("Average loss", f"{_rec['avg_loss']:+.2f}%")
        r5.metric("Worst losing run", f"{_rec['worst_losing_streak']}", "sessions in a row")

        st.caption(
            f"Replayed over every session: enter at the open, exit at the target, the stop, or the "
            f"close — whichever the session delivered — minus a {_ocost:.2f}% round trip. "
            f"A daily candle records the session's high and low but **not their order**, so when a "
            f"session touched both the target and the stop it is impossible to know which came "
            f"first; every one of those is counted as a **loss**. That pessimistic convention "
            f"affects {_rec['ambiguous_pct']:.1f}% of sessions here, so the record above is a floor "
            f"rather than a flattering estimate.")

    if _plan:
        st.markdown("---")
        st.markdown(f"#### {'📈 CALL' if _dir == 'long' else '📉 PUT'} — levels for the next session")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Entry", f"{cur}{_plan['entry']:,.2f}", "last close")
        p2.metric("Stop loss", f"{cur}{_plan['stop']:,.2f}", f"-{_plan['risk_pct']:.2f}% risk")
        p3.metric("Target", f"{cur}{_plan['target']:,.2f}", f"+{_plan['reward_pct']:.2f}% reward")
        p4.metric("Reward : risk", f"{_plan['rr']:.2f} : 1",
                  f"ATR {_plan['atr_pct']:.2f}% of price")
        st.caption(
            f"Distances are set in ATR — {_plan['atr_pct']:.2f}% of price for {ticker} right now — "
            f"rather than in round numbers, so they scale with how much this symbol actually moves. "
            f"A 1% stop means something very different on a quiet large cap and on a volatile "
            f"small cap; {_stop_a:.1f}×ATR means the same thing on both.")

        # ---- the option leg ----
        st.markdown("##### Which contract")
        _chain = {}
        try:
            _chain = cached_chain(ticker, float(_plan["entry"]), data_sig)
        except Exception as _e:
            _chain = {"available": False, "reason": str(_e)}

        if _chain.get("available"):
            _side = "calls" if _dir == "long" else "puts"
            _df = _chain[_side]
            st.caption(f"Live chain, expiry **{_chain['expiry']}** · strikes nearest the spot. "
                       f"Break-even = strike ± the mid price, so it already includes what you pay.")
            st.dataframe(_df, hide_index=True, width="stretch", column_config={
                "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "lastPrice": st.column_config.NumberColumn("Last", format="%.2f"),
                "bid": st.column_config.NumberColumn("Bid", format="%.2f"),
                "ask": st.column_config.NumberColumn("Ask", format="%.2f"),
                "mid": st.column_config.NumberColumn("Mid", format="%.2f"),
                "break_even": st.column_config.NumberColumn("Break-even", format="%.2f"),
                "impliedVolatility": st.column_config.NumberColumn("IV", format="%.1%%"),
            })
            _be = _df["break_even"].dropna()
            if len(_be):
                _need = (float(_be.iloc[len(_be) // 2]) / _plan["entry"] - 1) * 100
                st.info(f"The middle strike needs the underlying to move **{abs(_need):.2f}%** just "
                        f"to break even, and your target is {_plan['reward_pct']:.2f}%. "
                        + ("The target clears break-even." if abs(_need) < _plan['reward_pct'] else
                           "**The target does not clear break-even** — the option loses even if the "
                           "underlying reaches your target."))
        else:
            _rule = opl.strike_rule(float(_plan["entry"]), _plan)
            st.warning(f"**No live option chain for {ticker}.** {_chain.get('reason', '')}")
            s1, s2, s3 = st.columns(3)
            s1.metric("Contract", _rule["kind"])
            s2.metric("At-the-money strike", f"{_rule['atm']:,.0f}", f"step {_rule['step']:g}")
            s3.metric("One step OTM", f"{_rule['one_otm']:,.0f}",
                      "reachable" if _rule["otm_reachable"] else "too far for this target")
            st.caption(
                f"{_rule['note'].capitalize()}. No premium, break-even or greeks are shown because "
                f"there is no chain to read them from, and inventing them would be worse than "
                f"leaving them out. Price the contract in your broker before acting on this.")

    if _sweep is not None and not _sweep.empty:
        st.markdown("---")
        st.markdown("**Every stop/target combination, by measured expectancy**")
        st.dataframe(_sweep, hide_index=True, width="stretch", column_config={
            "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg win": st.column_config.NumberColumn(format="%+.2f%%"),
            "Avg loss": st.column_config.NumberColumn(format="%+.2f%%"),
            "Expectancy": st.column_config.NumberColumn(format="%+.3f%%"),
            "Ambiguous": st.column_config.NumberColumn(format="%.2f%%"),
        })
        _best = _sweep.iloc[0]
        st.caption(
            f"Best of the {len(_sweep)} combinations is stop {_best['Stop (ATR)']:.1f}×ATR / target "
            f"{_best['Target (ATR)']:.1f}×ATR at {_best['Expectancy']:+.3f}% per trade. "
            + ("Every combination is negative — there is no setting of this rule that made money on "
               "this symbol after costs. Set the cost slider to 0 and most turn to roughly zero, "
               "which is the real finding: the entry has no edge, and the round trip is the loss. "
               "Measured across six NSE symbols, the bracket is also no better than simply holding "
               "to the close (−0.426% to −0.353% per trade against −0.355% for hold-to-close): a "
               "tight target caps the winners while the stop keeps the full loss, and a wide target "
               "is never reached. The stop is worth having for risk control; the target is not "
               "earning its place."
               if (_sweep['Expectancy'] <= 0).all() else
               "Treat the top row as the least-bad setting rather than a recommendation — picking "
               "the best of many tried settings on the same data overstates what it will do next."))

# Shared by the candlestick tab and the technical scorecard.
FWD_DAYS = 5
try:
    candle_events, candle_reliability = cached_candle_analysis(ticker, history_start, history_end, FWD_DAYS)
except Exception as _candle_err:      # odd/sparse data -> no patterns, rest of the page still works
    candle_events, candle_reliability = pd.DataFrame(), pd.DataFrame()
    st.info(f"Candlestick reading unavailable for this symbol ({_candle_err}).")


# ===================== TAB: research analyst report =====================
def _render_asset_report(R, cur):
    """Gold, EURUSD, bitcoin, an ETF... judged on price behaviour instead of fundamentals."""
    M, PH, P = R["metrics"], R["history_stats"], R["profile"]
    ui.verdict_banner(R["overall"], R["verdict"], R["stance"], R["summary"])
    ui.rings(R["pillars"], _pillar_why(R))
    st.caption("💡 Click any ring above to see the metrics behind that score.")
    ui.bull_bear(R["strengths"], R["risks"])
    ui.card("🔭 What to watch", ui.watch_html(R["watch"]))

    a_over, a_hist, a_seas = st.tabs(["🧭 Overview", "📜 Price history", "📅 Seasonality & scorecard"])

    def _p(v, digits=1):
        return "n/a" if v is None else f"{v * 100:+.{digits}f}%"

    with a_over:
        rows = [("Type", (R.get("kind") or "instrument").title()),
                ("Traded on", P.get("exchange") or "n/a"),
                ("Quoted in", P.get("currency") or "n/a"),
                ("Price now", f"{cur}{R['price']:,.2f}"),
                ("History from", P["listed_since"].strftime("%d %b %Y") if P.get("listed_since") else "n/a"),
                ("Benchmark", (M.get("benchmark") or "n/a").title()),
                ("vs 200-day average", _p(M.get("vs_sma200"))),
                ("Volatility (1y)", _p(M.get("volatility"), 0).lstrip("+"))]
        if P.get("category"):
            rows.insert(3, ("Tracks", P["category"]))
        if P.get("total_assets"):
            rows.insert(4, ("Fund size", f"{cur}{fmt_big(P['total_assets'])}"))
        if P.get("expense_ratio"):
            rows.insert(5, ("Expense ratio", f"{P['expense_ratio']:.2f}%"))
        ui.facts(rows)
        ui.card("What moves this price",
                f"<p>{_html.escape(R['name'])} is a {R.get('kind', 'instrument')}. Its price is driven mainly by "
                f"{_html.escape(R.get('drivers', 'broad market conditions'))}. There are no revenues, margins or "
                f"P/E to read here, so the scorecard below judges what the market actually does with it: the "
                f"direction of the trend, how strong the momentum is, how violently it moves, how deep its falls "
                f"get, and how it compares with {_html.escape(M.get('benchmark') or 'its benchmark')}.</p>")
        if P.get("summary"):
            ui.card("Description", f"<p>{_html.escape(P['summary'])}</p>")

    with a_hist:
        ui.facts([("Since start of history (per year)", _p(PH.get("cagr_all"))),
                  ("10 years (per year)", _p(PH.get("cagr_10y"))), ("5 years (per year)", _p(PH.get("cagr_5y"))),
                  ("1 year", _p(PH.get("ret_1y"))),
                  ("All-time high", f"{PH['ath_date']:%b %Y} · {_p(PH.get('drawdown_from_ath'))} since"
                   if PH.get("ath_date") is not None else "n/a"),
                  ("Worst fall ever", f"{PH['max_dd'] * 100:.0f}% ({PH['max_dd_date']:%b %Y})"
                   if PH.get("max_dd_date") is not None else "n/a"),
                  ("Best / worst year", f"{PH['best_year'][0]}: {PH['best_year'][1] * 100:+.0f}% · "
                                        f"{PH['worst_year'][0]}: {PH['worst_year'][1] * 100:+.0f}%"
                   if PH.get("best_year") else "n/a"),
                  ("Up years", f"{PH['pct_up_years'] * 100:.0f}% of full years"
                   if PH.get("pct_up_years") is not None else "n/a")])
        yr = PH.get("yearly_returns")
        if yr is not None and len(yr):
            fig_y = go.Figure(go.Bar(x=[str(y) for y in yr.index], y=yr.values * 100, marker_line_width=0,
                                     marker_color=[ui.UP if v >= 0 else ui.DOWN for v in yr.values],
                                     hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
            fig_y.update_layout(height=300, template=PLOTLY_TEMPLATE, title="Calendar-year returns",
                                margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(ticksuffix="%"), hovermode="closest")
            st.plotly_chart(fig_y, width="stretch")
        dd = PH.get("drawdown_series")
        if dd is not None and len(dd):
            fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy",
                                          line=dict(color=ui.DOWN, width=1.2),
                                          fillcolor="rgba(208,59,59,0.18)", name="Below previous peak"))
            fig_dd.update_layout(height=260, template=PLOTLY_TEMPLATE,
                                 title="How far below its previous peak (drawdowns)",
                                 margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_dd, width="stretch")

    with a_seas:
        seas = R.get("seasonality")
        if seas is not None and len(seas):
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig_s = go.Figure(go.Bar(x=[months[int(m) - 1] for m in seas.index], y=seas.values * 100,
                                     marker_line_width=0,
                                     marker_color=[ui.UP if v >= 0 else ui.DOWN for v in seas.values],
                                     hovertemplate="%{x}: %{y:+.2f}% average<extra></extra>"))
            fig_s.update_layout(height=300, template=PLOTLY_TEMPLATE,
                                title="Average return by calendar month (whole history)",
                                margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(ticksuffix="%"), hovermode="closest")
            st.plotly_chart(fig_s, width="stretch")
            st.caption("Seasonality is a tendency measured over the years in the history above, not a rule — a strong "
                       "month can still fall. It is shown for context, and is not used by the forecasting models.")
        st.markdown("**Scorecard: how this instrument behaves**")
        st.dataframe(_metric_table(ra.ASSET_PILLARS), width="stretch", hide_index=True)
        st.caption("Each metric is scored 0-10 against general thresholds (e.g. 12% annualised volatility scores 10, "
                   "60% scores 0), then averaged into the six pillars above.")

    lines = [f"# Research report: {R['name']} ({R['ticker']})",
             f"_Generated {datetime.date.today():%d %b %Y} by {ui.APP_NAME}_", "",
             f"## Verdict: {R['verdict']} ({R['overall']:.1f}/10)" if R["overall"] is not None else "## Verdict: n/a",
             f"**{R['stance']}**", "", R["summary"], "", "## Pillar scores"]
    lines += [f"- {k}: {'n/a' if v is None else f'{v:.1f}/10'}" for k, v in R["pillars"].items()]
    lines += ["", "## Strengths"] + ([f"- {x['text']}" for x in R["strengths"]] or ["- none"])
    lines += ["", "## Risks"] + ([f"- {x['text']}" for x in R["risks"]] or ["- none"])
    lines += ["", "## What to watch"] + [f"- {w}" for w in R["watch"]]
    st.download_button("⬇️ Download research report (Markdown)", "\n".join(lines),
                       file_name=f"{R['ticker']}_research_report.md", mime="text/markdown")
    st.caption("Built by transparent rules from Yahoo Finance price data — every line maps to a metric in the "
               "scorecard. An analytical summary, not investment advice.")


_METRIC_LABELS = {
    "revenue_cagr": "Revenue growth (CAGR)", "net_income_cagr": "Net profit growth (CAGR)",
    "revenue_growth_yoy": "Revenue growth (latest, YoY)", "earnings_growth_yoy": "Earnings growth (latest, YoY)",
    "net_margin": "Net profit margin", "operating_margin": "Operating margin", "roe": "Return on equity",
    "margin_trend": "Net margin change over period", "debt_to_equity": "Debt / equity", "current_ratio": "Current ratio",
    "interest_coverage": "Interest coverage (EBIT / interest)", "net_cash_ratio": "Net cash / market cap",
    "fcf_positive_share": "Years with positive free cash flow", "pe": "P/E (trailing)", "forward_pe": "P/E (forward)",
    "peg": "PEG ratio", "ev_ebitda": "EV / EBITDA", "pb": "Price / book", "fcf_yield": "Free-cash-flow yield",
    "pe_vs_history": "P/E vs own history", "dividend_yield": "Dividend yield", "payout_ratio": "Payout ratio",
    "dividend_cagr_5y": "Dividend growth (5y CAGR)", "buyback_yield": "Buyback yield", "vs_sma200": "Price vs 200-day avg",
    "return_1y": "1-year return", "drawdown_from_ath": "From all-time high", "technical_score": "Technical score (-1..+1)",
    "volatility": "Volatility (1y, annualized)", "beta": "Beta", "max_dd_5y": "Worst fall (5y)",
    # --- commodities / currencies / crypto / ETFs ---
    "vs_sma50": "Price vs 50-day avg", "sma50_vs_sma200": "50-day avg vs 200-day avg",
    "return_1m": "1-month return", "return_3m": "3-month return", "return_6m": "6-month return",
    "cagr_all": "Growth per year (whole history)", "rsi14": "RSI (14-day)",
    "vol_vs_history": "Volatility vs its own history", "pct_up_years": "Positive calendar years",
    "pct_up_months": "Positive months", "relative_1y": "1-year return vs benchmark",
    "benchmark_corr": "Correlation with benchmark", "price": "Price",
}


def _metric_table(pillar_names):
    rows = []
    for p in pillar_names:
        for k, sc in research["sub_scores"].get(p, {}).items():
            v = research["metrics"].get(k)
            if v is None and sc is None:
                continue
            color, tag = ui.score_color(sc)
            rows.append({"Area": p, "Metric": _METRIC_LABELS.get(k, k), "Value": ra._fmt_metric(k, v),
                         "Score /10": None if sc is None else round(sc, 1), "Reading": tag})
    return pd.DataFrame(rows)


def _pillar_why(R):
    """{pillar: ([(metric, value, score)], note)} so every ring can explain itself."""
    out = {}
    for pillar, items in (R.get("sub_scores") or {}).items():
        rows = []
        for k, sc in items.items():
            v = R["metrics"].get(k)
            if v is None and sc is None:
                continue
            rows.append((_METRIC_LABELS.get(k, k.replace("_", " ").capitalize()),
                         ra._fmt_metric(k, v), sc))
        if rows:
            scored = [r for r in rows if r[2] is not None]
            weakest = min(scored, key=lambda r: r[2]) if scored else None
            note = ("Each metric is scored 0-10 against fixed thresholds and the ring is their average; "
                    "metrics with no data are skipped rather than counted as zero.")
            if weakest and weakest[2] < 4.5:
                note += f" The score here is held back most by <b>{weakest[0]}</b> ({weakest[1]})."
            out[pillar] = (rows, note)
    return out


def _fy(idx):
    return [f"FY{d.year}" if hasattr(d, "year") else str(d) for d in idx]


def _bar_chart(series_list, title, height=300, pct=False):
    fig = go.Figure()
    for i, (name, s_) in enumerate(series_list):
        if s_ is None or len(s_) == 0:
            continue
        fig.add_trace(go.Bar(x=_fy(s_.index), y=s_.values * (100 if pct else 1), name=name,
                             marker_color=ui.SERIES[i], marker_line_width=0))
    fig.update_layout(height=height, template=PLOTLY_TEMPLATE, title=title, barmode="group", bargap=0.25,
                      margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(ticksuffix="%" if pct else ""))
    return fig


with tab_research:
    if research is None:
        st.info("The research report couldn't be built for this symbol right now.")
    elif research.get("is_asset"):
        _render_asset_report(research, cur)
    else:
        R, M, T, PH = research, research["metrics"], research["trends"], research["history_stats"]
        ui.verdict_banner(R["overall"], R["verdict"], R["stance"], R["summary"])
        ui.rings(R["pillars"], _pillar_why(R))
        st.caption("💡 Click any ring above to see the metrics behind that score.")
        ui.bull_bear(R["strengths"], R["risks"])
        ui.card("🔭 What to watch", ui.watch_html(R["watch"]))

        r_company, r_fin, r_val, r_hist, r_own = st.tabs(
            ["🏢 Company", "📊 Financials", "💵 Valuation", "📜 Price history", "👥 Ownership & analysts"])

        with r_company:
            P = R["profile"]
            hq = ", ".join(x for x in [P.get("city"), P.get("country")] if x) or "n/a"
            site = f'<a href="{P["website"]}" target="_blank">{P["website"].replace("https://", "").replace("http://", "")}</a>' if P.get("website") else "n/a"
            ui.facts([("Sector", P.get("sector") or "n/a"), ("Industry", P.get("industry") or "n/a"), ("Headquarters", hq),
                      ("Employees", f"{P['employees']:,}" if P.get("employees") else "n/a"),
                      ("Listed since", P["listed_since"].strftime("%d %b %Y") if P.get("listed_since") else "n/a"),
                      ("Exchange", P.get("exchange") or "n/a"), ("Market cap", ra.fmt_money(R["info"].get("marketCap"), cur)),
                      ("Website", site)])
            if P.get("summary"):
                ui.card("What the company does", f"<p>{_html.escape(P['summary'])}</p>")
            if P.get("officers"):
                st.markdown("**Leadership**")
                off = pd.DataFrame(P["officers"])
                off["Total pay"] = off["Total pay"].map(lambda v: ra.fmt_money(v, cur) if pd.notna(v) else "")
                off["Age"] = off["Age"].map(lambda v: f"{int(v)}" if pd.notna(v) else "")
                st.dataframe(off, width="stretch", hide_index=True)

        with r_fin:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(_bar_chart([("Revenue", T["revenue"]), ("Net profit", T["net_income"])],
                                           f"Revenue & net profit ({cur})"), width="stretch")
            with c2:
                fig_m = go.Figure()
                for i, (nm, key) in enumerate((("Gross margin", "gross_margin"), ("Operating margin", "operating_margin"),
                                               ("Net margin", "net_margin"))):
                    if key in T and len(T[key]):
                        fig_m.add_trace(go.Scatter(x=_fy(T[key].index), y=T[key].values * 100, name=nm, mode="lines+markers",
                                                   line=dict(color=ui.SERIES[i], width=2.5), marker=dict(size=8)))
                fig_m.update_layout(height=300, template=PLOTLY_TEMPLATE, title="Profit margins", yaxis=dict(ticksuffix="%"),
                                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_m, width="stretch")
            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(_bar_chart([("Operating cash flow", T["ocf"]), ("Free cash flow", T["fcf"])],
                                           f"Cash flow ({cur})"), width="stretch")
            with c4:
                st.plotly_chart(_bar_chart([("Shareholders' equity", T["equity"]), ("Total debt", T["total_debt"])],
                                           f"Equity vs debt ({cur})"), width="stretch")
            q = cached_company_data(ticker, AS_OF if IS_HISTORICAL else None).get("quarterly")
            q_rev, q_ni = ra._row(q, "Total Revenue", "Operating Revenue"), ra._row(q, "Net Income Common Stockholders", "Net Income")
            if q_rev is not None:
                fig_q = go.Figure()
                fig_q.add_trace(go.Bar(x=[d.strftime("%b %Y") for d in q_rev.index], y=q_rev.values, name="Revenue",
                                       marker_color=ui.SERIES[0], marker_line_width=0))
                if q_ni is not None:
                    fig_q.add_trace(go.Bar(x=[d.strftime("%b %Y") for d in q_ni.index], y=q_ni.values, name="Net profit",
                                           marker_color=ui.SERIES[1], marker_line_width=0))
                fig_q.update_layout(height=280, template=PLOTLY_TEMPLATE, title=f"Latest quarters ({cur})", barmode="group",
                                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_q, width="stretch")
            st.markdown("**Scorecard: growth, profitability & financial health**")
            st.dataframe(_metric_table(["Growth", "Profitability", "Financial health"]), width="stretch", hide_index=True)
            st.caption("Scores use general rules of thumb (e.g. net margin 0% → 0/10, 25%+ → 10/10), not sector-specific "
                       "benchmarks. Financial companies skip debt/liquidity scoring.")

        with r_val:
            st.dataframe(_metric_table(["Valuation"]), width="stretch", hide_index=True)
            hpe = R.get("historical_pe")
            if hpe is not None and len(hpe) >= 2:
                fig_pe = go.Figure()
                fig_pe.add_trace(go.Bar(x=_fy(hpe.index), y=hpe.values, name="P/E at fiscal year end",
                                        marker_color=ui.SERIES[0], marker_line_width=0))
                if M.get("pe"):
                    fig_pe.add_trace(go.Bar(x=["Now"], y=[M["pe"]], name="P/E now", marker_color=ui.SERIES[1], marker_line_width=0))
                fig_pe.add_hline(y=hpe.mean(), line_dash="dot", line_color=ui.NEUTRAL,
                                 annotation_text=f"{len(hpe)}-year average {hpe.mean():.1f}x", annotation_font_size=10)
                fig_pe.update_layout(height=300, template=PLOTLY_TEMPLATE, title="P/E ratio vs its own history",
                                     margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(ticksuffix="x"))
                st.plotly_chart(fig_pe, width="stretch")
            st.caption("Cheaper than its own history, a low PEG (price vs growth) and a high free-cash-flow yield all point "
                       "to better value. A low P/E alone can also mean the market expects profits to fall.")

        with r_hist:
            def _p(key):
                v = PH.get(key)
                return "n/a" if v is None else f"{v * 100:+.1f}%"
            ui.facts([("Since listing (per year)", _p("cagr_all")), ("10 years (per year)", _p("cagr_10y")),
                      ("5 years (per year)", _p("cagr_5y")), ("1 year", _p("ret_1y")),
                      ("All-time high", f"{PH['ath_date']:%b %Y} · {_p('drawdown_from_ath')} since" if PH.get("ath_date") is not None else "n/a"),
                      ("Worst fall ever", f"{PH['max_dd'] * 100:.0f}% ({PH['max_dd_date']:%b %Y})" if PH.get("max_dd_date") is not None else "n/a"),
                      ("Best / worst year", f"{PH['best_year'][0]}: {PH['best_year'][1] * 100:+.0f}% · {PH['worst_year'][0]}: {PH['worst_year'][1] * 100:+.0f}%" if PH.get("best_year") else "n/a"),
                      ("Up years", f"{PH['pct_up_years'] * 100:.0f}% of full years" if PH.get("pct_up_years") is not None else "n/a")])
            yr = PH.get("yearly_returns")
            if yr is not None and len(yr):
                fig_y = go.Figure(go.Bar(x=[str(y) for y in yr.index], y=yr.values * 100, marker_line_width=0,
                                         marker_color=[ui.UP if v >= 0 else ui.DOWN for v in yr.values],
                                         hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
                fig_y.update_layout(height=300, template=PLOTLY_TEMPLATE, title="Calendar-year returns (dividends reinvested)",
                                    margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(ticksuffix="%"), hovermode="closest")
                st.plotly_chart(fig_y, width="stretch")
            dd = PH.get("drawdown_series")
            if dd is not None and len(dd):
                fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy", line=dict(color=ui.DOWN, width=1.2),
                                              fillcolor="rgba(208,59,59,0.18)", name="Below previous peak"))
                fig_dd.update_layout(height=260, template=PLOTLY_TEMPLATE, title="How far below its previous peak (drawdowns)",
                                     margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_dd, width="stretch")
            ad = R.get("annual_dividends")
            if ad is not None and len(ad):
                fig_d = go.Figure(go.Bar(x=[str(y) for y in ad.index], y=ad.values, marker_color=ui.SERIES[2], marker_line_width=0))
                fig_d.update_layout(height=260, template=PLOTLY_TEMPLATE, title=f"Dividends paid per share, by year ({cur})",
                                    margin=dict(l=10, r=10, t=40, b=10), hovermode="closest")
                st.plotly_chart(fig_d, width="stretch")

        with r_own:
            P, info_ = R["profile"], R["info"]
            rec_mean = info_.get("recommendationMean")
            ui.facts([("Held by insiders", f"{P['insiders'] * 100:.1f}%" if P.get("insiders") is not None else "n/a"),
                      ("Held by institutions", f"{P['institutions'] * 100:.1f}%" if P.get("institutions") is not None else "n/a"),
                      ("Analysts covering", str(info_.get("numberOfAnalystOpinions") or "n/a")),
                      ("Consensus", f"{(info_.get('recommendationKey') or 'n/a').replace('_', ' ').title()}"
                                    + (f" ({rec_mean:.2f} on 1–5)" if rec_mean else ""))])
            recs = R.get("recs")
            if recs:
                labels = ["Strong buy", "Buy", "Hold", "Sell", "Strong sell"]
                colors = [ui.UP, "#5fd35f", ui.NEUTRAL, "#f07474", ui.DOWN]
                fig_r = go.Figure(go.Bar(x=list(recs.values()), y=labels, orientation="h", marker_color=colors,
                                         marker_line_width=0, text=list(recs.values()), textposition="outside"))
                fig_r.update_layout(height=260, template=PLOTLY_TEMPLATE, title="Analyst ratings (this month)",
                                    margin=dict(l=10, r=30, t=40, b=10), yaxis=dict(autorange="reversed"), hovermode="closest")
                st.plotly_chart(fig_r, width="stretch")
            tl, th = info_.get("targetLowPrice"), info_.get("targetHighPrice")
            if tl and th and R.get("price"):
                ui.kpi_grid([ui.range_bar_html("Analyst price targets · marker = today's price", tl, th, R["price"], cur),
                             ui.kpi_html("Mean target", f"{cur}{info_.get('targetMeanPrice', 0):,.2f}",
                                         f"{M['analyst_upside'] * 100:+.1f}% vs today" if M.get("analyst_upside") is not None else None,
                                         "up" if (M.get("analyst_upside") or 0) >= 0 else "down")])
            st.caption("Ownership and analyst data come from Yahoo Finance and can lag. Insider % includes promoter/"
                       "government holdings for many Indian companies.")

        # ---- downloadable report ----
        lines = [f"# Research report: {R['name']} ({ticker})", f"_Generated {datetime.date.today():%d %b %Y} by {ui.APP_NAME}_", "",
                 f"## Verdict: {R['verdict']} ({R['overall']:.1f}/10)" if R["overall"] is not None else "## Verdict: n/a",
                 f"**{R['stance']}**", "", R["summary"], "", "## Pillar scores"]
        lines += [f"- {p}: {'n/a' if v is None else f'{v:.1f}/10'}" for p, v in R["pillars"].items()]
        lines += ["", "## Strengths"] + [f"- {x['text']}" for x in R["strengths"]] or ["- none"]
        lines += ["", "## Risks"] + [f"- {x['text']}" for x in R["risks"]] or ["- none"]
        lines += ["", "## What to watch"] + [f"- {w}" for w in R["watch"]]
        st.download_button("⬇️ Download research report (Markdown)", "\n".join(lines),
                           file_name=f"{ticker}_research_report.md", mime="text/markdown")
        st.caption("This report is generated by transparent rules from Yahoo Finance data -- every statement maps to a "
                   "metric in the tables above. It's an analytical summary, not investment advice.")

# ===================== TAB 1: single-date prediction =====================
with tab1:
    st.subheader("Predict price & signal for a specific date")

    with st.expander("🧭 Direction Signal & Market Regime (additional, cross-check signal)", expanded=True):
        with st.spinner("Training direction classifiers (Logistic Regression, Random Forest, XGBoost)..."):
            try:
                clf_models, clf_weights, advanced_feat_df = cached_train_direction(ticker, history_start, history_end,
                                                                                   news_sig, news_df)
                prob_up, direction_label = dl.get_direction_signal(clf_models, clf_weights, advanced_feat_df)
                regime_label = dl.get_current_regime(advanced_feat_df)

                dcol1, dcol2, dcol3 = st.columns(3)
                dcol1.metric("Direction signal (next trading day)", direction_label)
                dcol2.metric("Current market regime", regime_label)

                with st.spinner("Fetching latest news..."):
                    news_summary = cached_news_sentiment(ticker, nh.company_query(profile.get("name", ticker)))
                if news_summary["article_count"] > 0:
                    news_emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}[news_summary["label"]]
                    dcol3.metric("Live news sentiment", f"{news_emoji} {news_summary['label']}",
                                 f"{news_summary['average_score']:+.2f} avg, {news_summary['article_count']} articles")
                else:
                    dcol3.metric("Live news sentiment", "N/A", "no recent articles found")

                st.caption(
                    "Direction blends Logistic Regression, Random Forest, and XGBoost classifiers "
                    "trained to predict UP/DOWN (not exact price), using technical indicators, a "
                    "candlestick-pattern feature, market context matched to the stock's market "
                    "(VIX + S&P 500 for US stocks, India VIX + NIFTY 50 for NSE/BSE stocks, plus a "
                    "sector index) and a volatility regime feature. "
                    "The LIVE news box scores today's headlines (Yahoo Finance + Google News, VADER "
                    "sentiment) and saves them daily. Separately, news HISTORY (GDELT tone/attention, "
                    "plus Alpha Vantage if you add a key) is offered to the models as an optional extra "
                    "input -- kept only when it improves validation accuracy; see Model Insights → "
                    "'How news is used'. Use all of this as context alongside the price prediction "
                    "below, not a replacement for it. 50% direction accuracy is a coin flip -- treat "
                    "anything far above ~55% with healthy suspicion rather than excitement."
                )

                if news_summary["article_count"] > 0:
                    show_headlines = st.checkbox("📰 Show recent headlines used for sentiment")
                    if show_headlines:
                        for a in news_summary["articles"]:
                            emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}[a["label"]]
                            link_text = f"[{a['title']}]({a['link']})" if a["link"] else a["title"]
                            st.markdown(f"{emoji} **{a['score']:+.2f}** — {link_text} _({a['publisher']})_")
            except Exception as e:
                st.info(f"Direction/regime/news signal unavailable: {e}")

    target_date = st.date_input(
        "Pick a date",
        value=last_date.date() + datetime.timedelta(days=7),
        min_value=engine.table.index[1].date(),
        key="single_date_picker",
    )

    if st.button("🔍 Predict this date", type="primary"):
        try:
            result = engine.predict_date(pd.Timestamp(target_date), buy_threshold, sell_threshold)
        except ValueError as e:
            st.warning(str(e))
        else:
            colO, colA, colB, colC = st.columns(4)
            if result["predicted_open"] is not None:
                if result["actual_open"] is not None:
                    open_error = result["predicted_open"] - result["actual_open"]
                    colO.metric("Predicted open price", f"{cur}{result['predicted_open']:.2f}", f"{open_error:+.2f} error")
                else:
                    colO.metric("Predicted open price", f"{cur}{result['predicted_open']:.2f}")
            colA.metric("Predicted close price", f"{cur}{result['predicted_price']:.2f}",
                        f"80%: {cur}{result['lower_80']:,.2f} – {cur}{result['upper_80']:,.2f}", delta_color="off", delta_arrow="off")
            if result["actual_price"] is not None:
                error = result["predicted_price"] - result["actual_price"]
                colB.metric("Actual close price", f"{cur}{result['actual_price']:.2f}", f"{error:+.2f} error")
            else:
                colB.metric("Actual close price", "N/A (future date)")
            with colC:
                st.markdown("**Signal**")
                st.markdown(signal_badge_html(result["signal"], result["pct_change"]), unsafe_allow_html=True)
            st.caption(
                f"The signal compares the predicted close with the last close using your sidebar thresholds "
                f"(BUY above +{buy_threshold*100:.1f}%, SELL below −{sell_threshold*100:.1f}%). The 80% range comes "
                f"from the model's own validation errors. Next-day open comes from a separate model of the "
                f"overnight gap; later days' opens assume the previous predicted close."
            )

            if result["mode"] == "historical":
                st.info(
                    f"📊 This date falls within your historical dataset — predicted using only data "
                    f"available up to **{result['reference_date'].date()}**, shown next to the real "
                    f"actual price as a fair backtest."
                )
                window = raw.loc[:pd.Timestamp(target_date)].tail(60)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=window.index, y=window["Close"], name="Actual Close",
                                          line=dict(color=ui.SERIES[0], width=2)))
                fig.add_trace(go.Scatter(x=[pd.Timestamp(target_date)], y=[result["predicted_price"]],
                                          mode="markers", name="Predicted (80% range)",
                                          error_y=dict(type="data", symmetric=False, color=ui.SERIES[1],
                                                       array=[result["upper_80"] - result["predicted_price"]],
                                                       arrayminus=[result["predicted_price"] - result["lower_80"]]),
                                          marker=dict(color=ui.SERIES[1], size=14, symbol="star")))
                fig.update_layout(height=350, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=30, b=10),
                                   title="Last 60 Days + Prediction Point")
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning(
                    f"🔮 This date is beyond your historical data. It's forecast from "
                    f"**{result['reference_date'].date()}** by dedicated 1/5/10/21/63-day models (interpolated "
                    f"between them), not by stacking daily guesses. The shaded band is the 80% likely range: "
                    f"it widens the further ahead you look."
                )
                path = result["forecast_path"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=raw.index[-30:], y=raw["Close"].iloc[-30:], name="Recent Actual Close",
                                          line=dict(color=ui.SERIES[0], width=2)))
                ui.forecast_cone(fig, path, color=ui.SERIES[1], name="Forecast (close)")
                if "Predicted_Open" in path.columns:
                    fig.add_trace(go.Scatter(x=path.index, y=path["Predicted_Open"], name="Forecast Path (Open)",
                                              line=dict(color=ui.SERIES[2], width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=[pd.Timestamp(target_date)], y=[result["predicted_price"]],
                                          mode="markers", name="Target Prediction (Close)",
                                          marker=dict(color=ui.DOWN, size=14, symbol="star")))
                fig.update_layout(height=380, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=30, b=10),
                                   title="Forecast path with 80% likely range")
                st.plotly_chart(fig, width="stretch")

                csv_data = path.reset_index().rename(columns={"index": "Date"}).to_csv(index=False)
                st.download_button("⬇️ Download forecast path (CSV)", csv_data,
                                    file_name=f"{ticker.upper()}_forecast_path.csv", mime="text/csv")

# ===================== TAB: candlestick chart reading =====================
with tab_candle:
    st.subheader("🕯️ Candlestick chart reading")
    lookback = st.slider("Read patterns from the last N candles", 1, 10, 5, key="candle_lookback")
    reading = cp.read_candles(raw, lookback=lookback, reliability=candle_reliability)

    verdict_style = {"Bullish": SIGNAL_STYLE["BUY"], "Bearish": SIGNAL_STYLE["SELL"]}.get(
        reading["verdict"], SIGNAL_STYLE["HOLD"])
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.markdown("**Current candle reading**")
        st.markdown(
            f'<div class="signal-badge" style="background-color:{verdict_style[1]};color:{verdict_style[0]};">'
            f'{reading["emoji"]} {reading["verdict"]}</div>', unsafe_allow_html=True)
        st.caption(f"Pattern score: {reading['score']:+.2f} (> +1.5 bullish, < −1.5 bearish)")
    with rc2:
        st.markdown("**What the chart is saying**")
        st.markdown(reading["today"])
        if reading["notes"]:
            for note in reading["notes"]:
                st.markdown(f"- {note}")
        else:
            st.markdown(f"- No named candlestick patterns in the last {lookback} candles.")

    # Zoomed chart with every pattern labelled.
    zoom = raw.tail(60)
    ev_zoom = cp.pattern_events(raw.tail(90))
    ev_zoom = ev_zoom[ev_zoom["Date"] >= zoom.index[0]]
    if not st.checkbox("Show neutral patterns on chart (Doji, Spinning Top)", value=False):
        ev_zoom = ev_zoom[ev_zoom["Bias_Num"] != 0]
    fig_c = go.Figure(go.Candlestick(
        x=zoom.index, open=zoom["Open"], high=zoom["High"], low=zoom["Low"], close=zoom["Close"],
        name=ticker.upper(), increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN))
    # One label per date and side, so same-day patterns stack instead of overlapping.
    for (d, bullish_side), g in ev_zoom.groupby([ev_zoom["Date"], ev_zoom["Bias_Num"] >= 0]):
        net = g["Bias_Num"].sum()
        y = raw.at[d, "Low"] if bullish_side else raw.at[d, "High"]
        fig_c.add_annotation(
            x=d, y=y, text="<br>".join(g["Pattern"]), showarrow=True, arrowhead=2, arrowsize=0.8,
            ay=30 + 12 * len(g) if bullish_side else -30 - 12 * len(g), ax=0, font=dict(size=10),
            arrowcolor=ui.UP if net > 0 else (ui.DOWN if net < 0 else ui.NEUTRAL),
        )
    fig_c.update_layout(height=480, template=PLOTLY_TEMPLATE, xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=40, b=10), title="Last 60 candles with detected patterns")
    fig_c.update_xaxes(rangebreaks=WEEKEND_BREAK)
    st.plotly_chart(fig_c, width="stretch")

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.markdown("**Recent pattern history**")
        hist = candle_events.tail(25).iloc[::-1].copy()
        if hist.empty:
            st.info("No patterns detected.")
        else:
            hist["Date"] = hist["Date"].dt.date
            hist["Strength"] = hist["Strength"].map({1: "★", 2: "★★", 3: "★★★"})
            hist["Volume"] = hist["Volume_Confirmed"].map({True: "✔ high", False: ""})
            fwd_col = f"Fwd_{FWD_DAYS}d_Return"
            hist[f"Next {FWD_DAYS}d"] = hist[fwd_col].map(lambda v: "pending" if pd.isna(v) else f"{v*100:+.2f}%")
            st.dataframe(hist[["Date", "Pattern", "Bias", "Strength", "Volume", f"Next {FWD_DAYS}d"]],
                         width="stretch", hide_index=True, height=420)
    with cc2:
        st.markdown(f"**How reliable is each pattern on {ticker.upper()}?**")
        if candle_reliability.empty:
            st.info("Not enough history to evaluate patterns.")
        else:
            rel = candle_reliability.copy()
            for col in ["Hit rate", "Base rate", "Edge vs base", f"Avg {FWD_DAYS}d return"]:
                rel[col] = rel[col] * 100
            st.dataframe(
                rel, width="stretch", hide_index=True, height=420,
                column_config={
                    "Hit rate": st.column_config.NumberColumn(format="%.1f%%"),
                    "Base rate": st.column_config.NumberColumn(format="%.1f%%"),
                    "Edge vs base": st.column_config.NumberColumn(format="%+.1f pts"),
                    f"Avg {FWD_DAYS}d return": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )
        st.caption(
            f"**Hit rate** = how often price moved the pattern's expected way over the next {FWD_DAYS} "
            f"trading days, across all of {ticker.upper()}'s history in your date range. **Base rate** = how "
            f"often it moves that way in *any* {FWD_DAYS}-day window. A pattern only has an edge on this stock "
            "if its hit rate beats the base rate — and small occurrence counts are mostly noise."
        )

    with st.expander("📖 Pattern glossary"):
        for name, (bias, strength, desc) in cp.PATTERN_INFO.items():
            st.markdown(f"**{name}** — {cp.BIAS_LABEL[bias]}, {'★' * strength}  \n{desc}")

    st.caption(
        "Candlestick patterns are classical chart-reading heuristics. Their standalone predictive power is "
        "modest and varies by stock, which is why reliability is measured on this stock's own history. "
        "The recent pattern score is also fed into the direction classifiers as the `Candle_Bias_3d` feature."
    )

# ===================== TAB 2: best buy/sell window =====================
with tab2:
    # ---------------- how fast can I get X%? ----------------
    st.markdown("#### 🎯 How fast can I get a move of this size?")
    st.caption("A 1–2% move is usually an intraday event, so asking a daily forecast for it is "
               "asking the wrong model. This answers it from what this symbol has actually done: "
               "every session in its history, measured from that session's own opening price.")
    tcol1, tcol2 = st.columns([2, 1])
    _target = tcol1.slider("Target move (%)", 0.25, 5.0, 1.5, 0.25, key="tgt_move")
    _tcost = tcol2.number_input("Round-trip cost (%)", 0.0, 2.0, 0.30, 0.05, key="tgt_cost")

    @st.cache_data(show_spinner=False, ttl=3600)
    def cached_target(ticker, target, cost, sig):
        return itg.find_target_window(ticker, target, daily=cached_load_data(
            ticker, history_start, history_end), cost_pct=cost)

    @st.cache_data(show_spinner=False, ttl=3600)
    def cached_ladder(ticker, cost, sig):
        return itg.ladder(ticker, cached_load_data(ticker, history_start, history_end),
                          cost_pct=cost)

    try:
        _tw = cached_target(ticker, _target, _tcost, data_sig)
    except Exception as _e:
        _tw = {"error": str(_e)}

    if _tw.get("error"):
        st.info(f"Not enough history to measure this ({_tw['error']}).")
    else:
        _st8, _tm, _md = _tw["stats"], _tw["timing"], _tw["multi_day"]
        _net = _tw["net_pct"]

        def _mins(m):
            if m is None:
                return "n/a"
            if m < 1:
                return "in the first candle"
            if m < 60:
                return f"{m:.0f} min"
            return f"{m / 60:.1f} h"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Same-day odds", f"{_tw['p_up_used']:.0f}%",
                  f"{_st8['sessions']:,} sessions measured")
        k2.metric("Typical time to hit", _mins(_tm.get("median_minutes")),
                  f"from the open · {_tm.get('sessions', 0)} sessions")
        k3.metric("Net after costs", f"{_net:+.2f}%", f"{_target:.2f}% gross")
        k4.metric("Watch candle", _tw["watch_candle"],
                  f"typical move lands {_mins(_tm.get('median_minutes'))} in")

        if _tw["verdict"] == "costs_exceed_target":
            st.error(f"**A {_target:.2f}% target does not survive a {_tcost:.2f}% round trip.** "
                     f"You would net {_net:+.2f}%. Aim larger or trade cheaper.")
        elif _tw["verdict"] == "same_day":
            st.success(
                f"**Same-day is realistic.** {ticker} reaches +{_target:.2f}% above its opening "
                f"price on **{_tw['p_up_used']:.0f}% of sessions**, and when it does the move "
                f"typically lands **{_mins(_tm.get('median_minutes'))} after the open** "
                f"(quickest quarter: {_mins(_tm.get('p25_minutes'))}). Watch the "
                f"**{_tw['watch_candle']}** candle — and set the exit at the target rather than "
                f"holding for the close, because {100 - _tw['p_up_used']:.0f}% of sessions never "
                f"get there at all.")
        elif _tw["verdict"] == "few_days":
            st.info(
                f"**Not usually same-day, but days — not months.** {ticker} reaches "
                f"+{_target:.2f}% within the session on only **{_tw['p_up_used']:.0f}%** of days, "
                f"but **{_md.get('within_3_days', 0):.0f}% of the time it gets there within 3 "
                f"sessions** and the typical wait is **{_md.get('median_days', 0):.0f} "
                f"{'session' if _md.get('median_days', 0) == 1 else 'sessions'}**. That is the "
                f"honest answer for this size of move — not one month.")
        else:
            st.warning(
                f"**{_target:.2f}% is a big ask for {ticker}.** It happens within a session on "
                f"{_tw['p_up_used']:.0f}% of days and within 3 sessions {_md.get('within_3_days', 0):.0f}% "
                f"of the time; the typical wait is {_md.get('median_days', 0):.0f} sessions. "
                f"A smaller target gets you paid sooner — see the table below.")

        _vn = _st8.get("vol_now_vs_typical")
        st.caption(
            f"Measured from **{_st8['sessions']:,} sessions** of daily highs and lows — a daily "
            f"candle already contains the session's extremes, so the hit rate uses the full "
            f"history rather than the ~60 days a provider serves intraday. The timing comes from "
            f"**{_tm.get('sessions', 0)} sessions** of {_tw.get('timing_interval') or 'intraday'} "
            f"candles, which is a much smaller sample — treat it as a guide to the shape of a "
            f"session, not a precise clock. "
            + (f"Right now this symbol is **{_vn:.2f}× its typical volatility**, and the odds "
               f"above are taken from past sessions in similar conditions "
               f"({_st8.get('similar_sessions', 0):,} of them) rather than a flat average. "
               if _vn else "")
            + f"**{_st8['p_either']:.0f}% of sessions touch ±{_target:.2f}% in *either* direction** "
              f"— but you cannot buy the low and sell the high, so the one-directional number is "
              f"the one to plan with.")

        try:
            _lad = cached_ladder(ticker, _tcost, data_sig)
        except Exception:
            _lad = None
        if _lad is not None and not _lad.empty:
            _l = _lad.copy()
            _l["Typical time to hit"] = _l["Typical time to hit"].map(_mins)
            st.markdown("**Bigger move, longer wait — the trade-off for this symbol**")
            st.dataframe(_l, hide_index=True, width="stretch", column_config={
                "Target": st.column_config.NumberColumn(format="%.2f%%"),
                "Net after costs": st.column_config.NumberColumn(format="%+.2f%%"),
                "Same-day odds (long)": st.column_config.NumberColumn(format="%.0f%%"),
                "Touches either way": st.column_config.NumberColumn(format="%.0f%%"),
                "Within 1 day": st.column_config.NumberColumn(format="%.0f%%"),
                "Within 3 days": st.column_config.NumberColumn(format="%.0f%%"),
                "Typical days": st.column_config.NumberColumn(format="%.0f"),
            })
            st.caption("Smaller targets arrive far sooner but hand more of the gain to costs; "
                       "larger ones keep the gain and make you wait. This table is the trade-off "
                       "for this symbol, measured rather than assumed.")

    st.markdown("---")
    st.markdown("#### 📅 Or scan the daily forecast for a buy/sell window")
    horizon = st.slider(f"Look ahead how many {STEP_WORD}?", 5, 90, 30, key="horizon_slider")

    _cost = st.slider("Round-trip cost (%)", 0.0, 1.0, 0.30, 0.05, key="rt_cost",
                      help="Brokerage + STT + exchange fees + slippage, in and out. ~0.30% is "
                           "realistic for Indian equities on a discount broker. A window that "
                           "does not clear this is a loss, however good the gross number looks.")
    _minp = st.slider("Minimum chance of profit (%)", 50, 75, 55, 1, key="min_prob",
                      help="A window is only recommended if the model puts at least this "
                           "probability on finishing above the buy price after costs.")

    if st.button("🎯 Scan for best buy/sell points", type="primary"):
        with st.spinner("Scoring every buy/sell pair on return, time and uncertainty..."):
            scan = engine.scan_best_window(horizon, cost_pct=_cost, min_prob=_minp / 100)

        path = scan["forecast_path"]
        bh = scan.get("buy_hold")

        if scan["verdict"] == "no_trade":
            st.error(
                f"**No window in the next {horizon} {STEP_WORD} is worth trading.** "
                f"Out of {scan['candidates']:,} buy/sell pairs, none clears a {_cost:.2f}% round trip "
                f"with at least a {_minp}% chance of profit. "
                + (f"Holding the whole window would net {bh['net_pct']:+.2f}% over {bh['hold_days']} days "
                   f"({bh['annualised_pct']:+.1f}% a year) with only a {bh['prob_profit']:.0f}% chance of "
                   f"being up — that is a coin flip, not a plan." if bh else ""))
            st.caption("This is the honest answer more often than not for a liquid large cap. "
                       "A scan that always finds a trade is not a scan.")
        else:
            b = scan["best_short"]
            st.success(
                f"**Shortest window worth taking: {b['hold_days']} days** — "
                f"buy {b['buy_date']:%d %b}, sell {b['sell_date']:%d %b}.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📥 Buy", f"{b['buy_date']:%d %b}", f"{cur}{b['buy_price']:,.2f}")
            c2.metric("📤 Sell", f"{b['sell_date']:%d %b}", f"{cur}{b['sell_price']:,.2f}")
            c3.metric("Net after costs", f"{b['net_pct']:+.2f}%", f"{b['gross_pct']:+.2f}% gross")
            c4.metric("Per year", f"{b['annualised_pct']:+.1f}%", f"{b['prob_profit']:.0f}% chance of profit")

            ra, an = scan["best_risk_adj"], scan["best_annual"]
            st.markdown(
                f"- **Shortest** · {b['hold_days']}d · net {b['net_pct']:+.2f}% · "
                f"{b['annualised_pct']:+.1f}%/yr · {b['prob_profit']:.0f}% odds\n"
                f"- **Best odds** · {ra['hold_days']}d · net {ra['net_pct']:+.2f}% · "
                f"{ra['annualised_pct']:+.1f}%/yr · {ra['prob_profit']:.0f}% odds\n"
                f"- **Best per year** · {an['hold_days']}d · net {an['net_pct']:+.2f}% · "
                f"{an['annualised_pct']:+.1f}%/yr · {an['prob_profit']:.0f}% odds"
                + (f"\n- **Buy and hold the whole window** · {bh['hold_days']}d · net {bh['net_pct']:+.2f}% · "
                   f"{bh['annualised_pct']:+.1f}%/yr · {bh['prob_profit']:.0f}% odds" if bh else ""))
            if bh and not scan["beats_buy_hold"]:
                st.warning(
                    f"Timing does not beat patience here: simply holding the whole window earns "
                    f"more per year ({bh['annualised_pct']:+.1f}%) than the shortest tradeable "
                    f"window ({b['annualised_pct']:+.1f}%). Take the short one only if you want "
                    f"your money back sooner, not because it earns more.")

        # ---- what each holding length can actually buy ----
        fr = pd.DataFrame(scan["frontier"])
        if not fr.empty:
            fr = fr[["hold_days", "net_pct", "annualised_pct", "prob_profit"]].rename(columns={
                "hold_days": "Hold (days)", "net_pct": "Net after costs",
                "annualised_pct": "Per year", "prob_profit": "Chance of profit"})
            st.markdown("**What each holding length can actually buy**")
            st.dataframe(fr, hide_index=True, width="stretch",
                         height=min(340, 40 + 28 * len(fr)),
                         column_config={
                             "Net after costs": st.column_config.NumberColumn(format="%+.2f%%"),
                             "Per year": st.column_config.NumberColumn(format="%+.1f%%"),
                             "Chance of profit": st.column_config.NumberColumn(format="%.0f%%")})

        fig = go.Figure()
        ui.forecast_cone(fig, path, color=ui.ACCENT, name="Forecast path")
        if scan.get("buy_date") is not None:
            fig.add_trace(go.Scatter(x=[scan["buy_date"]], y=[scan["buy_price"]], mode="markers+text",
                                     name="Buy", text=["BUY"], textposition="bottom center",
                                     marker=dict(color=ui.UP, size=16, symbol="triangle-up")))
        if scan.get("sell_date") is not None:
            fig.add_trace(go.Scatter(x=[scan["sell_date"]], y=[scan["sell_price"]], mode="markers+text",
                                     name="Sell", text=["SELL"], textposition="top center",
                                     marker=dict(color=ui.DOWN, size=16, symbol="triangle-down")))
        fig.update_layout(height=400, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=30, b=10),
                          title=f"Forecast Path — Next {horizon} Business Days")
        st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Every one of the {scan['candidates']:,} buy/sell pairs is scored on what actually "
            f"decides a trade: the gain **after a {_cost:.2f}% round trip**, the uncertainty of that "
            f"particular hold (not of the price level — the two differ, and only the first matters "
            f"once you are in), and how long your money is tied up. "
            f"**The old version just bought on day 1 and sold on the last day of whatever window you "
            f"picked**, so the 'recommended' holding period was really just this slider. It is not "
            f"any more: move the slider and the recommendation should stay put."
        )

        csv_data = path.reset_index().rename(columns={"index": "Date"}).to_csv(index=False)
        st.download_button("⬇️ Download forecast path (CSV)", csv_data,
                            file_name=f"{ticker.upper()}_buy_sell_scan.csv", mime="text/csv")

# ===================== TAB 3: technical indicators =====================
with tab3:
    st.subheader("Technical summary")
    tech_reading = cp.read_candles(raw, lookback=5, reliability=candle_reliability)
    rating, tech_score, votes_df = ta.technical_scorecard(raw, tech_reading)
    rating_style = {"Strong Buy": SIGNAL_STYLE["BUY"], "Buy": SIGNAL_STYLE["BUY"],
                    "Strong Sell": SIGNAL_STYLE["SELL"], "Sell": SIGNAL_STYLE["SELL"]}.get(rating, SIGNAL_STYLE["HOLD"])

    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.markdown("**Overall technical rating**")
        st.markdown(
            f'<div class="signal-badge" style="background-color:{rating_style[1]};color:{rating_style[0]};">'
            f'{rating_style[2]} {rating}</div>', unsafe_allow_html=True)
        n_buy = (votes_df["Vote"] == "Buy").sum() if not votes_df.empty else 0
        n_sell = (votes_df["Vote"] == "Sell").sum() if not votes_df.empty else 0
        st.caption(f"{n_buy} Buy · {len(votes_df) - n_buy - n_sell} Neutral · {n_sell} Sell  (score {tech_score:+.2f})")

        st.markdown("**Support & resistance levels**")
        levels = ta.support_resistance_levels(raw)
        if levels:
            lv = pd.DataFrame(levels)[::-1]
            lv["Distance"] = (lv["level"] / last_price - 1) * 100
            lv = lv.rename(columns={"level": f"Level ({cur})", "touches": "Touches", "kind": "Type"})
            st.dataframe(lv, hide_index=True, width="stretch",
                         column_config={f"Level ({cur})": st.column_config.NumberColumn(format="%.2f"),
                                        "Distance": st.column_config.NumberColumn(format="%+.1f%%")})
            st.caption("Price levels where the stock repeatedly turned over the last ~year "
                       "(clustered swing highs/lows). More touches = stronger level.")
    with sc2:
        st.markdown("**Indicator votes**")
        st.dataframe(votes_df, hide_index=True, width="stretch")
        st.caption("Each indicator votes Buy / Sell / Neutral using standard textbook thresholds; the rating is "
                   "the average vote. A quick-read summary of classical technical analysis — not a prediction.")

    st.markdown("---")
    st.subheader("Technical indicators (last 180 days)")
    window = ind_df.tail(180)

    fig_ind = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("RSI (14)", "MACD", "Stochastic (14, 3)"),
        row_heights=[0.3, 0.4, 0.3],
    )
    fig_ind.add_trace(go.Scatter(x=window.index, y=window["Stoch_K"], name="%K",
                                  line=dict(color=ui.SERIES[0], width=1.6)), row=3, col=1)
    fig_ind.add_trace(go.Scatter(x=window.index, y=window["Stoch_D"], name="%D",
                                  line=dict(color=ui.SERIES[1], width=1.6)), row=3, col=1)
    fig_ind.add_hline(y=80, line_dash="dash", line_color=ui.DOWN, row=3, col=1)
    fig_ind.add_hline(y=20, line_dash="dash", line_color=ui.UP, row=3, col=1)

    fig_ind.add_trace(go.Scatter(x=window.index, y=window["RSI_14"], name="RSI",
                                  line=dict(color=ui.SERIES[2], width=2)), row=1, col=1)
    fig_ind.add_hline(y=70, line_dash="dash", line_color=ui.DOWN, row=1, col=1)
    fig_ind.add_hline(y=30, line_dash="dash", line_color=ui.UP, row=1, col=1)

    fig_ind.add_trace(go.Scatter(x=window.index, y=window["MACD"], name="MACD",
                                  line=dict(color=ui.SERIES[0], width=2)), row=2, col=1)
    fig_ind.add_trace(go.Scatter(x=window.index, y=window["MACD_Signal"], name="Signal",
                                  line=dict(color=ui.SERIES[1], width=2)), row=2, col=1)
    macd_hist = window["MACD"] - window["MACD_Signal"]
    hist_colors = [ui.UP if v >= 0 else ui.DOWN for v in macd_hist]
    fig_ind.add_trace(go.Bar(x=window.index, y=macd_hist, name="Histogram",
                              marker_color=hist_colors, opacity=0.5), row=2, col=1)

    fig_ind.update_layout(height=700, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_ind, width="stretch")

    st.caption(
        "**RSI** above 70 is often read as overbought, below 30 as oversold. "
        "**MACD** crossing above its signal line is often read as bullish momentum, "
        "crossing below as bearish. **Stochastic** above 80 / below 20 is read as overbought / oversold. "
        "These are classical technical-analysis heuristics, "
        "shown here for context alongside the model predictions — not signals on their own."
    )

# ===================== TAB 4: model insights =====================
with tab4:
    st.subheader("What the forecast uses")
    st.dataframe(engine.summary(), width="stretch", hide_index=True)
    st.caption(
        "Each horizon has its own model. Starting from price & volume, every other data group is tested across "
        "3 consecutive past periods (walk-forward cross-validation) and kept only if it cuts the error by ≥0.3% on "
        "average AND in at least 2 of the 3 periods (✓ kept · ✕ tested, didn't help consistently · – not "
        "available for this stock). If a model cannot beat 'no change' in at least 2 periods, its alpha "
        "is set to 0.\n\n"
        "**Signal strength alpha** scales the prediction toward 'no change'. A small alpha is the normal, "
        "honest outcome for a liquid market: across three separate past periods the models could not beat "
        "simply assuming the price stays put, so the forecast barely moves. Carrying the symbol's own "
        "average drift forward instead was tested and measured **worse** on held-out data (mean -0.3 skill "
        "points across 24 symbol/horizon cases), so it is deliberately not done.\n\n"
        "**80% range** is where the information actually is at these horizons. It is no longer one fixed "
        "width: the engine measures how big its misses were *relative to the volatility of the day*, then "
        "rebuilds the band from today's volatility. Measured across 36 held-out symbol/horizon cases, that cut "
        "the gap between the promised 80% coverage and the actual coverage from 9.5 points to 4.0, and was "
        "closer in 29 of 36 cases."
    )
    _ii = _adv.attrs.get("indicator_info", {}) if "_adv" in dir() else {}
    if _ii.get("offered"):
        st.caption(
            f"**Technical indicator bank ({_ii.get('n', 0)} columns — ADX, Aroon, Ichimoku, Keltner, Donchian, "
            f"MFI, CMF, Ulcer, Hurst, entropy, seasonality and more):** offered to the direction classifier, "
            f"{'**kept**' if _ii.get('used') else 'tested and **rejected**'} for this symbol "
            f"(validation log-loss {_ii.get('val_logloss_without', float('nan')):.4f} without vs "
            f"{_ii.get('val_logloss_with', float('nan')):.4f} with). They are deliberately NOT given to the "
            f"price model: on held-out data they improved direction accuracy (50.6% → 52.6%) but made squared "
            f"error worse, so they are used only where they measurably help."
        )

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("**Model weights (1-day ensemble)**")
        weight_df = pd.DataFrame({"Model": list(weights.keys()), "Weight": list(weights.values())})
        weight_df = weight_df.sort_values("Weight", ascending=True)
        fig_w = go.Figure(go.Bar(
            x=weight_df["Weight"], y=weight_df["Model"], orientation="h",
            marker_color=ui.ACCENT, text=[f"{w:.1%}" for w in weight_df["Weight"]], textposition="outside",
        ))
        fig_w.update_layout(height=260, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=50, t=10, b=10),
                             xaxis_title="Weight", xaxis=dict(tickformat=".0%"), hovermode="closest")
        st.plotly_chart(fig_w, width="stretch")
    with colB:
        st.metric("Signal strength, 1-day (α)", f"{m1.alpha:.2f}",
                  help="How much the ensemble trusts its own predictions, fitted on validation data. "
                       "0 = no reliable pattern found, so the forecast falls back to 'tomorrow ≈ today'.")
        if m1.alpha < 0.05:
            st.info("No reliable next-day pattern was found for this stock on validation data, so the 1-day "
                    "forecast is essentially today's price. Longer horizons may still carry signal -- see the "
                    "table above.")

    st.markdown("---")
    st.markdown("**Held-out accuracy (last 15% of days, never used for training or selection)**")
    st.caption("Compares the full engine (all data groups, chosen on validation) with price/volume only and "
               "with the naive 'no change' forecast, at 1, 5 and 21 trading days ahead. Takes about a minute.")
    if st.button("📐 Compute held-out accuracy"):
        with st.spinner("Selecting features, training and testing for 3 horizons..."):
            ev = cached_engine_eval(ticker, history_start, history_end, data_sig, news_sig, news_df, extra)
        if ev.empty:
            st.info("Not enough history for a held-out test.")
        else:
            cols = st.columns(len(ev))
            for c, (_, r) in zip(cols, ev.iterrows()):
                c.metric(f"{int(r['Horizon (days)'])}-day skill vs naive",
                         f"{r['All groups (selected): skill vs naive (%)']:+.2f}%",
                         f"price/volume only: {r['Price/volume only: skill vs naive (%)']:+.2f}%", delta_color="off", delta_arrow="off")
            st.dataframe(ev, width="stretch", hide_index=True)
            st.caption("**Skill vs naive** = % lower error than predicting 'no change' (positive = better). "
                       "Daily prices barely move, so R² is ~0.97+ even for the naive guess -- that's why it isn't shown.")

    st.markdown("**Direction classifier accuracy (held-out test set)**")
    if st.button("🧭 Compute direction accuracy"):
        with st.spinner("Training direction classifiers on the first 85% and testing on the last 15%..."):
            _, _, adv_df = cached_train_direction(ticker, history_start, history_end, news_sig, news_df)
            dmetrics = cached_direction_accuracy(ticker, history_start, history_end, adv_df, news_sig)
        d1, d2, d3 = st.columns(3)
        d1.metric("Direction accuracy", f"{dmetrics['Direction accuracy (%)']:.1f}%")
        d2.metric("Always-UP baseline", f"{dmetrics['Always-UP baseline (%)']:.1f}%")
        conf_acc = dmetrics["Accuracy when confident (%)"]
        d3.metric("When confident", f"{conf_acc:.1f}%" if conf_acc is not None else "n/a",
                  f"on {dmetrics['Confident-day coverage (%)']:.0f}% of days", delta_color="off", delta_arrow="off")
        if "Accuracy WITH news (%)" in dmetrics:
            st.caption(f"Same test days without → with news: {dmetrics['Accuracy WITHOUT news (%)']:.1f}% → "
                       f"{dmetrics['Accuracy WITH news (%)']:.1f}%")
        st.caption("The model only adds value if it beats the always-UP baseline, not just 50%.")

    st.markdown("---")
    st.markdown("**📰 How news is used**")
    if news_df is None:
        st.info("News history isn't being used right now. " + " ".join(news_info.get("notes", [])))
    else:
        _news_h = [h for h, m in engine.models.items() if "news" in m.groups_used]
        price_news = {"used": bool(_news_h)}
        _, _, adv_now = cached_train_direction(ticker, history_start, history_end, news_sig, news_df)
        dir_news = adv_now.attrs.get("news_info", {})
        n1, n2, n3 = st.columns(3)
        n1.metric("News sources", ", ".join(news_info["sources"]) or "none")
        n2.metric("Price models using news", "Yes" if price_news.get("used") else "No",
                  ("horizons: " + ", ".join(f"{h}d" for h in _news_h)) if _news_h else "didn't help on validation",
                  delta_color="off", delta_arrow="off")
        n3.metric("Direction model uses news?", "Yes" if dir_news.get("used") else "No",
                  f"val log-loss {dir_news.get('val_logloss_without', 0):.4f} → {dir_news.get('val_logloss_with', 0):.4f}",
                  delta_color="off", delta_arrow="off")
        st.caption(
            f"Coverage: articles found on {news_info.get('coverage_pct', 0):.0f}% of days. News is an OPTIONAL extra "
            "input: each model is trained with and without it, and news is kept only when it lowers validation "
            "error (the numbers above: without → with). It uses only news published up to the previous day, so "
            "there's no look-ahead. " + " ".join(news_info.get("notes", [])))

