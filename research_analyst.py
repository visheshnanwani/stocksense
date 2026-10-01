"""
research_analyst.py
---------------------
A rule-based equity research analyst. It gathers everything Yahoo Finance
publishes about a company -- profile, leadership, 4-5 years of income
statements / balance sheets / cash flows, valuation ratios, ownership,
analyst ratings, upcoming events, and the full price history -- and turns it
into a research report:

  * 7 pillar scores (0-10): Growth, Profitability, Financial health,
    Valuation, Shareholder returns, Momentum, Risk
  * an overall score, verdict and one-line stance
  * a written executive summary, strengths (bull case), risks (bear case)
    and what to watch next

Everything is computed from the numbers with transparent rules (thresholds
below), so every statement can be traced back to a metric. The thresholds
are general rules of thumb, not sector-specific -- e.g. banks naturally carry
more debt, so debt metrics are skipped for financial companies.

Usage:
    from research_analyst import gather_company_data, analyze
    report = analyze(gather_company_data("MSFT"))
"""

import datetime

import numpy as np
import pandas as pd

PILLARS = ["Growth", "Profitability", "Financial health", "Valuation", "Shareholder returns", "Momentum", "Risk"]
PILLAR_WEIGHTS = {"Growth": 0.20, "Profitability": 0.20, "Financial health": 0.15, "Valuation": 0.20,
                  "Shareholder returns": 0.10, "Momentum": 0.10, "Risk": 0.05}


# ------------------------------------------------------------------------
# Data gathering
# ------------------------------------------------------------------------
def gather_company_data(ticker: str, as_of=None) -> dict:
    """
    Download everything available. Each piece is optional.

    as_of : when given, every price-derived series is truncated at that moment, so a
            historical replay cannot value the company using today's price. Statements
            (financials, balance sheet, cash flow) are the latest filed versions --
            Yahoo publishes no vintages, so they cannot be rewound; the report says so
            rather than pretending otherwise.
    """
    import yfinance as yf
    tk = yf.Ticker(ticker)
    out = {"ticker": ticker, "as_of": pd.Timestamp(as_of) if as_of is not None else None}

    def safe(fn, default=None):
        try:
            v = fn()
            return default if v is None else v
        except Exception:
            return default

    out["info"] = safe(lambda: tk.info, {}) or {}
    out["financials"] = safe(lambda: tk.financials, pd.DataFrame())
    out["quarterly"] = safe(lambda: tk.quarterly_financials, pd.DataFrame())
    out["balance"] = safe(lambda: tk.balance_sheet, pd.DataFrame())
    out["cashflow"] = safe(lambda: tk.cashflow, pd.DataFrame())
    out["recs"] = safe(lambda: tk.recommendations_summary, pd.DataFrame())
    out["calendar"] = safe(lambda: tk.calendar, {}) or {}
    out["dividends"] = safe(lambda: tk.dividends, pd.Series(dtype=float))
    out["splits"] = safe(lambda: tk.splits, pd.Series(dtype=float))
    # One download gives both: "Close" = actual traded price (split-adjusted only,
    # right for P/E) and "Adj Close" = also dividend-adjusted (right for returns).
    hist = safe(lambda: tk.history(period="max", auto_adjust=False), pd.DataFrame())
    if hist is not None and not hist.empty:
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        if out["as_of"] is not None:
            cutoff = out["as_of"].normalize() + pd.Timedelta(hours=23, minutes=59)
            hist = hist.loc[hist.index <= cutoff]
        out["history_traded"] = hist[["Close"]].copy()
        hist = hist.assign(Close=hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"])
    else:
        out["history_traded"] = pd.DataFrame()
    out["history"] = hist
    for key in ("dividends", "splits"):
        s = out[key]
        if s is not None and len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None)
            if out["as_of"] is not None:
                out[key] = s.loc[s.index <= out["as_of"].normalize()
                                 + pd.Timedelta(hours=23, minutes=59)]
    return out


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
def _row(df, *names):
    """First matching statement row as a chronological Series (oldest -> newest), or None."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            if len(s):
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return None


def _cagr(s):
    if s is None or len(s) < 2:
        return None
    first, last = s.iloc[0], s.iloc[-1]
    years = (s.index[-1] - s.index[0]).days / 365.25
    if first <= 0 or last <= 0 or years < 0.9:
        return None
    return (last / first) ** (1 / years) - 1


def _score(x, bad, good):
    """Linear 0..10 score: `bad` -> 0, `good` -> 10 (works for both directions)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(np.clip((x - bad) / (good - bad) * 10, 0, 10))


def _pct(x, digits=1):
    return "n/a" if x is None else f"{x * 100:+.{digits}f}%" if x < 0 else f"{x * 100:.{digits}f}%"


def fmt_money(x, cur=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{sign}{cur}{x / div:.2f}{suf}"
    return f"{sign}{cur}{x:,.2f}"


def _price_on(hist, date):
    if hist is None or hist.empty:
        return None
    s = hist["Close"].loc[:pd.Timestamp(date)]
    return float(s.iloc[-1]) if len(s) else None


# ------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------
def price_history_stats(hist: pd.DataFrame) -> dict:
    """Long-run behaviour of the (dividend- and split-adjusted) price."""
    if hist is None or hist.empty or len(hist) < 30:
        return {}
    c = hist["Close"].dropna()
    last = c.iloc[-1]
    st = {"first_date": c.index[0], "years": (c.index[-1] - c.index[0]).days / 365.25,
          "ath": float(c.max()), "ath_date": c.idxmax(), "atl": float(c.min()), "atl_date": c.idxmin()}
    st["drawdown_from_ath"] = last / st["ath"] - 1
    st["total_return"] = last / c.iloc[0] - 1
    st["cagr_all"] = (last / c.iloc[0]) ** (1 / st["years"]) - 1 if st["years"] > 1 and c.iloc[0] > 0 else None
    for label, yrs in (("1y", 1), ("3y", 3), ("5y", 5), ("10y", 10)):
        start = c.index[-1] - pd.DateOffset(years=yrs)
        past = c.loc[:start]
        if len(past) and c.index[0] <= start:
            st[f"ret_{label}"] = last / past.iloc[-1] - 1
            st[f"cagr_{label}"] = (last / past.iloc[-1]) ** (1 / yrs) - 1
    daily = c.pct_change().dropna()
    st["vol_1y"] = float(daily.tail(252).std() * np.sqrt(252))
    dd = c / c.cummax() - 1
    st["drawdown_series"] = dd
    st["max_dd"] = float(dd.min())
    st["max_dd_date"] = dd.idxmin()
    recent = c.tail(252 * 5)
    st["max_dd_5y"] = float((recent / recent.cummax() - 1).min())
    yearly = c.resample("YE").last()
    first_year_start = c.iloc[0]
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / first_year_start - 1
    yr.index = yr.index.year
    st["yearly_returns"] = yr
    full = yr.iloc[1:-1] if len(yr) > 2 else yr
    if len(full):
        st["best_year"] = (int(full.idxmax()), float(full.max()))
        st["worst_year"] = (int(full.idxmin()), float(full.min()))
        st["pct_up_years"] = float((full > 0).mean())
    sma200 = c.rolling(200).mean().iloc[-1]
    st["vs_sma200"] = last / sma200 - 1 if pd.notna(sma200) else None
    return st


def financial_trends(d: dict) -> dict:
    fin, bal, cf = d.get("financials"), d.get("balance"), d.get("cashflow")
    t = {
        "revenue": _row(fin, "Total Revenue", "Operating Revenue"),
        "gross_profit": _row(fin, "Gross Profit"),
        "operating_income": _row(fin, "Operating Income", "Total Operating Income As Reported"),
        "net_income": _row(fin, "Net Income Common Stockholders", "Net Income"),
        "eps": _row(fin, "Diluted EPS", "Basic EPS"),
        "ebit": _row(fin, "EBIT"),
        "interest": _row(fin, "Interest Expense", "Interest Expense Non Operating"),
        "equity": _row(bal, "Stockholders Equity", "Common Stock Equity"),
        "total_debt": _row(bal, "Total Debt"),
        "cash": _row(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"),
        "total_assets": _row(bal, "Total Assets"),
        "current_assets": _row(bal, "Current Assets"),
        "current_liabilities": _row(bal, "Current Liabilities"),
        "fcf": _row(cf, "Free Cash Flow"),
        "ocf": _row(cf, "Operating Cash Flow"),
        "dividends_paid": _row(cf, "Cash Dividends Paid"),
        "buybacks": _row(cf, "Repurchase Of Capital Stock"),
    }
    rev, ni = t["revenue"], t["net_income"]
    if rev is not None and ni is not None:
        t["net_margin"] = (ni / rev).dropna()
    if rev is not None and t["operating_income"] is not None:
        t["operating_margin"] = (t["operating_income"] / rev).dropna()
    if rev is not None and t["gross_profit"] is not None:
        t["gross_margin"] = (t["gross_profit"] / rev).dropna()
    if ni is not None and t["equity"] is not None:
        t["roe"] = (ni / t["equity"]).replace([np.inf, -np.inf], np.nan).dropna()
    if t["total_debt"] is not None and t["equity"] is not None:
        t["debt_to_equity"] = (t["total_debt"] / t["equity"]).replace([np.inf, -np.inf], np.nan).dropna()
    t["revenue_cagr"] = _cagr(rev)
    t["net_income_cagr"] = _cagr(ni)
    t["eps_cagr"] = _cagr(t["eps"])
    t["fcf_cagr"] = _cagr(t["fcf"])
    return t


def _historical_pe(hist, eps):
    """P/E at each fiscal year end = adjusted price then / diluted EPS for that year."""
    if eps is None or hist is None or hist.empty:
        return None
    vals = {}
    for dt, e in eps.items():
        p = _price_on(hist, dt)
        if p and e and e > 0:
            vals[dt] = p / e
    return pd.Series(vals).sort_index() if vals else None


def analyze(d: dict, technical_score: float = None, forecast_1m: dict = None) -> dict:
    """
    Build the full research report. Optional extras from the rest of the
    dashboard: `technical_score` (-1..1) and `forecast_1m`
    ({pct, lower, upper} from the forecast engine) for the "what to watch" notes.
    """
    info = d.get("info", {}) or {}
    hist = d.get("history")
    ph = price_history_stats(hist)
    tr = financial_trends(d)
    name = info.get("longName") or info.get("shortName") or d["ticker"]
    sector = info.get("sector") or ""
    is_financial = sector == "Financial Services"
    price = info.get("currentPrice") or info.get("regularMarketPrice") or (hist["Close"].iloc[-1] if hist is not None and len(hist) else None)
    mcap = info.get("marketCap")

    # ---- derived metrics ----
    net_margin = info.get("profitMargins") if info.get("profitMargins") is not None else (tr["net_margin"].iloc[-1] if "net_margin" in tr and len(tr["net_margin"]) else None)
    op_margin = info.get("operatingMargins") if info.get("operatingMargins") is not None else (tr["operating_margin"].iloc[-1] if "operating_margin" in tr and len(tr["operating_margin"]) else None)
    roe = info.get("returnOnEquity") if info.get("returnOnEquity") is not None else (tr["roe"].iloc[-1] if "roe" in tr and len(tr["roe"]) else None)
    margin_trend = None
    if "net_margin" in tr and len(tr["net_margin"]) >= 2:
        margin_trend = tr["net_margin"].iloc[-1] - tr["net_margin"].iloc[0]
    de = info.get("debtToEquity") / 100 if info.get("debtToEquity") is not None else (tr["debt_to_equity"].iloc[-1] if "debt_to_equity" in tr and len(tr["debt_to_equity"]) else None)
    current_ratio = info.get("currentRatio")
    if current_ratio is None and tr["current_assets"] is not None and tr["current_liabilities"] is not None:
        current_ratio = float(tr["current_assets"].iloc[-1] / tr["current_liabilities"].iloc[-1])
    coverage = None
    if tr["ebit"] is not None and tr["interest"] is not None and tr["interest"].iloc[-1] > 0:
        coverage = float(tr["ebit"].iloc[-1] / tr["interest"].iloc[-1])
    cash, debt = info.get("totalCash"), info.get("totalDebt")
    net_cash = (cash - debt) if cash is not None and debt is not None else None
    net_cash_ratio = net_cash / mcap if net_cash is not None and mcap else None
    fcf_latest = tr["fcf"].iloc[-1] if tr["fcf"] is not None else info.get("freeCashflow")
    fcf_pos_share = float((tr["fcf"] > 0).mean()) if tr["fcf"] is not None else None
    fcf_yield = fcf_latest / mcap if fcf_latest is not None and mcap else None
    pe, fpe = info.get("trailingPE"), info.get("forwardPE")
    peg = info.get("pegRatio") or info.get("trailingPegRatio")
    ev_ebitda, pb = info.get("enterpriseToEbitda"), info.get("priceToBook")
    ps = mcap / tr["revenue"].iloc[-1] if mcap and tr["revenue"] is not None else info.get("priceToSalesTrailing12Months")
    hist_pe = _historical_pe(d.get("history_traded"), tr["eps"])
    pe_vs_hist = pe / hist_pe.mean() if pe and hist_pe is not None and len(hist_pe) >= 2 else None
    div_yield = info.get("dividendYield") / 100 if info.get("dividendYield") is not None else None
    payout = info.get("payoutRatio")
    divs = d.get("dividends")
    annual_div = None
    div_cagr = None
    if divs is not None and len(divs):
        annual_div = divs.groupby(divs.index.year).sum()
        full_years = annual_div.loc[annual_div.index < datetime.date.today().year]
        if len(full_years) >= 6:
            a, b = full_years.iloc[-6], full_years.iloc[-1]
            div_cagr = (b / a) ** (1 / 5) - 1 if a > 0 and b > 0 else None
    buyback_yield = (-tr["buybacks"].iloc[-1] / mcap) if tr["buybacks"] is not None and mcap else None
    beta = info.get("beta")
    target_mean = info.get("targetMeanPrice")
    upside = target_mean / price - 1 if target_mean and price else None

    m = {  # metric -> value (used in scores, text and the UI tables)
        "revenue_cagr": tr["revenue_cagr"], "net_income_cagr": tr["net_income_cagr"],
        "revenue_growth_yoy": info.get("revenueGrowth"), "earnings_growth_yoy": info.get("earningsGrowth"),
        "net_margin": net_margin, "operating_margin": op_margin, "roe": roe, "margin_trend": margin_trend,
        "debt_to_equity": de, "current_ratio": current_ratio, "interest_coverage": coverage,
        "net_cash_ratio": net_cash_ratio, "fcf_positive_share": fcf_pos_share,
        "pe": pe, "forward_pe": fpe, "peg": peg, "ev_ebitda": ev_ebitda, "pb": pb, "ps": ps,
        "fcf_yield": fcf_yield, "pe_vs_history": pe_vs_hist,
        "dividend_yield": div_yield, "payout_ratio": payout, "dividend_cagr_5y": div_cagr, "buyback_yield": buyback_yield,
        "vs_sma200": ph.get("vs_sma200"), "return_1y": ph.get("ret_1y"), "drawdown_from_ath": ph.get("drawdown_from_ath"),
        "technical_score": technical_score,
        "volatility": ph.get("vol_1y"), "beta": beta, "max_dd_5y": ph.get("max_dd_5y"),
        "analyst_upside": upside,
    }

    # ---- pillar scores ----
    def pe_score(x):
        return None if x is None else (2.0 if x <= 0 else _score(x, 40, 10))

    def payout_score(x):
        if x is None or div_yield in (None, 0):
            return None
        return 10.0 if 0.2 <= x <= 0.7 else 6.0 if x < 0.2 else _score(x, 1.2, 0.7)

    sub = {
        "Growth": {"revenue_cagr": _score(m["revenue_cagr"], 0, 0.15), "net_income_cagr": _score(m["net_income_cagr"], -0.05, 0.20),
                   "revenue_growth_yoy": _score(m["revenue_growth_yoy"], -0.05, 0.20),
                   "earnings_growth_yoy": _score(m["earnings_growth_yoy"], -0.10, 0.25)},
        "Profitability": {"net_margin": _score(m["net_margin"], 0, 0.25), "operating_margin": _score(m["operating_margin"], 0.05, 0.30),
                          "roe": _score(m["roe"], 0.05, 0.25), "margin_trend": _score(m["margin_trend"], -0.03, 0.03)},
        "Financial health": {} if is_financial else {
            "debt_to_equity": _score(m["debt_to_equity"], 2.0, 0.2), "current_ratio": _score(m["current_ratio"], 0.8, 2.0),
            "interest_coverage": _score(m["interest_coverage"], 2, 15), "net_cash_ratio": _score(m["net_cash_ratio"], -0.5, 0.1),
            "fcf_positive_share": _score(m["fcf_positive_share"], 0.5, 1.0)},
        "Valuation": {"pe": pe_score(m["pe"]), "forward_pe": pe_score(m["forward_pe"]), "peg": _score(m["peg"], 3, 1),
                      "ev_ebitda": _score(m["ev_ebitda"], 25, 8), "pb": _score(m["pb"], 10, 1.5),
                      "fcf_yield": _score(m["fcf_yield"], 0, 0.07), "pe_vs_history": _score(m["pe_vs_history"], 1.3, 0.8)},
        "Shareholder returns": {"dividend_yield": _score(m["dividend_yield"] or 0.0, 0, 0.04), "payout_ratio": payout_score(m["payout_ratio"]),
                                "dividend_cagr_5y": _score(m["dividend_cagr_5y"], -0.05, 0.10),
                                "buyback_yield": _score(m["buyback_yield"], 0, 0.03)},
        "Momentum": {"vs_sma200": _score(m["vs_sma200"], -0.15, 0.15), "return_1y": _score(m["return_1y"], -0.20, 0.30),
                     "drawdown_from_ath": _score(m["drawdown_from_ath"], -0.40, 0.0),
                     "technical_score": _score(m["technical_score"], -0.5, 0.5)},
        "Risk": {"volatility": _score(m["volatility"], 0.50, 0.15), "beta": _score(m["beta"], 2.0, 0.7),
                 "max_dd_5y": _score(m["max_dd_5y"], -0.60, -0.15)},
    }
    pillars = {}
    for p, items in sub.items():
        vals = [v for v in items.values() if v is not None]
        pillars[p] = float(np.mean(vals)) if vals else None
    avail = {p: s for p, s in pillars.items() if s is not None}
    total_w = sum(PILLAR_WEIGHTS[p] for p in avail)
    overall = sum(PILLAR_WEIGHTS[p] * s for p, s in avail.items()) / total_w if total_w else None

    quality_parts = [pillars[p] for p in ("Growth", "Profitability", "Financial health") if pillars.get(p) is not None]
    quality = float(np.mean(quality_parts)) if quality_parts else None
    value = pillars.get("Valuation")
    verdict = ("Strong" if overall >= 7.5 else "Good" if overall >= 6 else "Mixed" if overall >= 4.5 else "Weak") if overall is not None else "Not enough data"
    stance = _stance(quality, value)

    report = {"name": name, "ticker": d["ticker"], "info": info, "metrics": m, "sub_scores": sub, "pillars": pillars,
              "overall": overall, "verdict": verdict, "stance": stance, "quality": quality, "trends": tr,
              "history_stats": ph, "historical_pe": hist_pe, "annual_dividends": annual_div, "is_financial": is_financial,
              "price": price}
    report["strengths"], report["risks"] = _bull_bear(m, sub, is_financial)
    report["summary"] = _executive_summary(report)
    report["watch"] = _what_to_watch(report, d.get("calendar", {}), forecast_1m)
    report["profile"] = _profile(info, ph)
    report["recs"] = _recs(d.get("recs"))
    return report


def _stance(quality, value):
    if quality is None or value is None:
        return "Not enough data for a full view"
    if quality >= 6.5:
        return ("High-quality business at an attractive price" if value >= 6 else
                "High-quality business, but richly valued" if value < 4 else "High-quality business at a fair price")
    if quality < 4.5:
        return "Looks cheap, but the business is weak -- possible value trap" if value >= 6 else "Weak fundamentals"
    return ("Average business at an attractive price" if value >= 6 else
            "Average business, and not cheap" if value < 4 else "Average business at a fair price")


# metric -> (strength text, risk text); {v} = formatted value
_TEXT = {
    "revenue_cagr": ("Revenue has compounded at {v} a year", "Revenue has barely grown ({v} a year)"),
    "net_income_cagr": ("Profits have grown {v} a year", "Profits have been shrinking or stagnant ({v} a year)"),
    "revenue_growth_yoy": ("Sales are growing fast right now ({v} year-on-year)", "Sales are falling ({v} year-on-year)"),
    "earnings_growth_yoy": ("Latest earnings are up {v} year-on-year", "Latest earnings are down {v} year-on-year"),
    "net_margin": ("Very profitable: keeps {v} of revenue as net profit", "Thin or negative profit margin ({v})"),
    "operating_margin": ("Strong operating margin of {v}", "Weak operating margin ({v})"),
    "roe": ("High return on equity ({v}): uses shareholders' money efficiently", "Low return on equity ({v})"),
    "margin_trend": ("Profit margins are expanding ({v} over the period)", "Profit margins are shrinking ({v} over the period)"),
    "debt_to_equity": ("Low debt: debt is only {v} of equity", "Heavy debt: debt is {v} of equity"),
    "current_ratio": ("Comfortable short-term liquidity (current ratio {v})", "Tight short-term liquidity (current ratio {v})"),
    "interest_coverage": ("Earnings cover interest costs {v} times over", "Earnings cover interest only {v} times"),
    "net_cash_ratio": ("Net cash position worth {v} of its market value", "Net debt equal to {v} of its market value"),
    "fcf_positive_share": ("Generated positive free cash flow in {v} of recent years", "Free cash flow was negative in several recent years ({v} positive)"),
    "pe": ("Inexpensive at {v} times earnings", "Expensive at {v} times earnings"),
    "forward_pe": ("Cheap on next year's expected earnings ({v}x)", "Expensive even on next year's expected earnings ({v}x)"),
    "peg": ("Attractive price for its growth (PEG {v})", "Price looks high relative to growth (PEG {v})"),
    "ev_ebitda": ("Low EV/EBITDA of {v}", "High EV/EBITDA of {v}"),
    "fcf_yield": ("High free-cash-flow yield of {v}", "Low free-cash-flow yield ({v})"),
    "pe_vs_history": ("Trading below its own historical P/E ({v} of its average)", "Trading above its own historical P/E ({v} of its average)"),
    "dividend_yield": ("Generous dividend yield of {v}", None),
    "payout_ratio": ("Sustainable dividend payout ({v} of profits)", "Pays out {v} of profits as dividends -- may not be sustainable"),
    "dividend_cagr_5y": ("Dividends have grown {v} a year over 5 years", "Dividends have been cut over the last 5 years ({v} a year)"),
    "buyback_yield": ("Returning cash through buybacks ({v} of market value last year)", None),
    "vs_sma200": ("In a long-term uptrend: {v} above its 200-day average", "In a long-term downtrend: {v} vs its 200-day average"),
    "return_1y": ("Strong 1-year price performance ({v})", "Weak 1-year price performance ({v})"),
    "drawdown_from_ath": ("Trading near its all-time high ({v} from the peak)", "Far below its all-time high ({v} from the peak)"),
    "technical_score": ("Technical indicators lean bullish", "Technical indicators lean bearish"),
    "volatility": ("Relatively calm stock ({v} annual volatility)", "Very volatile stock ({v} annual volatility)"),
    "beta": ("Moves less than the market (beta {v})", "Swings more than the market (beta {v})"),
    "max_dd_5y": ("Held up well in sell-offs (worst 5-year fall {v})", "Suffered deep sell-offs (worst 5-year fall {v})"),
}
_RATIO_METRICS = {"pe", "forward_pe", "peg", "ev_ebitda", "current_ratio", "interest_coverage", "beta", "pb"}


def _fmt_metric(k, v):
    if v is None:
        return "n/a"
    if k in _RATIO_METRICS:
        return f"{v:.1f}"
    if k == "pe_vs_history":
        return f"{v * 100:.0f}%"
    if k == "margin_trend":
        return f"{v * 100:+.1f} pts"
    if k == "technical_score":
        return f"{v:+.2f}"
    if k in ("rsi14",):
        return f"{v:.0f}"
    if k in ("price",):
        return f"{v:,.2f}"
    if k in ("benchmark_corr",):
        return f"{v:+.2f}"
    if k == "benchmark":
        return str(v)
    if k in ("fcf_positive_share", "payout_ratio", "debt_to_equity"):
        return f"{v * 100:.0f}%"
    return f"{v * 100:.1f}%"


def _bull_bear(m, sub, is_financial):
    items = []
    for pillar, scores in sub.items():
        for k, s in scores.items():
            if s is None or k not in _TEXT:
                continue
            items.append((k, s, pillar))
    strengths, risks = [], []
    for k, s, pillar in sorted(items, key=lambda t: -t[1]):
        if s >= 7.5 and _TEXT[k][0]:
            strengths.append({"pillar": pillar, "text": _TEXT[k][0].format(v=_fmt_metric(k, m[k])), "score": s})
    for k, s, pillar in sorted(items, key=lambda t: t[1]):
        if s <= 2.5 and _TEXT[k][1]:
            risks.append({"pillar": pillar, "text": _TEXT[k][1].format(v=_fmt_metric(k, m[k])), "score": s})
    if is_financial:
        risks.append({"pillar": "Financial health", "text": "Financial company: debt/liquidity ratios aren't comparable "
                      "to other sectors, so financial health wasn't scored", "score": 5})
    return strengths[:6], risks[:6]


def _executive_summary(r):
    m, info, ph, tr = r["metrics"], r["info"], r["history_stats"], r["trends"]
    parts = []
    what = info.get("industry") or info.get("sector") or "listed"
    where = f" based in {info.get('country')}" if info.get("country") else ""
    listed = f", with {ph['years']:.0f} years of trading history" if ph.get("years") else ""
    parts.append(f"{r['name']} is a {what.lower()} company{where}{listed}.")
    rev = tr.get("revenue")
    if m["revenue_cagr"] is not None and rev is not None:
        years = len(rev) - 1
        ni_txt = f" and net profit {_pct(m['net_income_cagr'])} a year" if m["net_income_cagr"] is not None else ""
        parts.append(f"Over the last {years} fiscal years revenue grew {_pct(m['revenue_cagr'])} a year{ni_txt}.")
    prof = []
    if m["net_margin"] is not None:
        prof.append(f"a {_pct(m['net_margin'])} net margin")
    if m["roe"] is not None:
        prof.append(f"a {_pct(m['roe'])} return on equity")
    if prof:
        parts.append("It currently earns " + " and ".join(prof) + ".")
    if not r["is_financial"] and m["net_cash_ratio"] is not None:
        parts.append("The balance sheet holds more cash than debt." if m["net_cash_ratio"] > 0 else
                     "The company carries more debt than cash.")
    if m["pe"] is not None:
        val = "cheap" if (r["pillars"].get("Valuation") or 5) >= 6 else "expensive" if (r["pillars"].get("Valuation") or 5) < 4 else "fairly valued"
        hist_txt = ""
        hpe = r.get("historical_pe")
        if hpe is not None and len(hpe) >= 2:
            hist_txt = f" (its own {len(hpe)}-year average is {hpe.mean():.1f}x)"
        parts.append(f"At {m['pe']:.1f}x earnings{hist_txt} it looks {val}.")
    if m["analyst_upside"] is not None and info.get("numberOfAnalystOpinions"):
        parts.append(f"{info['numberOfAnalystOpinions']} analysts' average target implies {_pct(m['analyst_upside'])} from here.")
    if ph.get("cagr_all") is not None:
        parts.append(f"Long-term, the stock has returned {_pct(ph['cagr_all'])} a year (dividends reinvested), "
                     f"with a worst-ever fall of {_pct(ph['max_dd'], 0)}.")
    if r["overall"] is not None:
        parts.append(f"Overall verdict: **{r['verdict']} ({r['overall']:.1f}/10)** — {r['stance'].lower()}.")
    return " ".join(parts)


def _what_to_watch(r, cal, forecast_1m):
    w = []
    def first_date(v):
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        try:
            ts = pd.Timestamp(v)
            return None if pd.isna(ts) else ts.date()   # NaT survives pd.Timestamp()
        except Exception:
            return None
    ed = first_date(cal.get("Earnings Date")) if isinstance(cal, dict) else None
    if ed:
        est = cal.get("Earnings Average")
        est_txt = f" (EPS estimate {est:.2f})" if isinstance(est, (int, float)) else ""
        w.append(f"📅 Next earnings: **{ed:%d %b %Y}**{est_txt} -- results often move the price sharply.")
    xd = first_date(cal.get("Ex-Dividend Date")) if isinstance(cal, dict) else None
    if xd and xd >= datetime.date.today():
        w.append(f"💰 Ex-dividend date: **{xd:%d %b %Y}** -- the price usually drops by about the dividend that day.")
    weakest = min(((p, s) for p, s in r["pillars"].items() if s is not None), key=lambda t: t[1], default=None)
    if weakest:
        w.append(f"🔍 Weakest area: **{weakest[0]}** ({weakest[1]:.1f}/10) -- the main thing that could hurt the investment case.")
    mt = r["metrics"].get("margin_trend")
    if mt is not None and mt < -0.02:
        w.append("📉 Profit margins have been shrinking -- check whether costs or competition are rising.")
    if forecast_1m:
        w.append(f"🔮 The price model's 1-month outlook: **{forecast_1m['pct'] * 100:+.1f}%** "
                 f"(80% range {forecast_1m['lower']:,.2f} – {forecast_1m['upper']:,.2f}).")
    return w


def _profile(info, ph):
    officers = []
    for o in (info.get("companyOfficers") or [])[:6]:
        officers.append({"Name": o.get("name"), "Title": o.get("title"),
                         "Age": o.get("age"), "Total pay": o.get("totalPay")})
    first = info.get("firstTradeDateMilliseconds")
    listed = datetime.datetime.utcfromtimestamp(first / 1000).date() if first else (ph.get("first_date").date() if ph.get("first_date") is not None else None)
    return {"sector": info.get("sector"), "industry": info.get("industry"), "country": info.get("country"),
            "city": info.get("city"), "employees": info.get("fullTimeEmployees"), "website": info.get("website"),
            "summary": info.get("longBusinessSummary"), "listed_since": listed, "officers": officers,
            "insiders": info.get("heldPercentInsiders"), "institutions": info.get("heldPercentInstitutions"),
            "exchange": info.get("fullExchangeName")}


def _recs(recs):
    if recs is None or not isinstance(recs, pd.DataFrame) or recs.empty:
        return None
    row = recs.iloc[0]
    cols = ["strongBuy", "buy", "hold", "sell", "strongSell"]
    if not all(c in recs.columns for c in cols):
        return None
    return {c: int(row[c]) for c in cols}


# ========================================================================
# Non-equity instruments (commodities, currencies, crypto, ETFs, indices)
# ------------------------------------------------------------------------
# Gold, EURUSD or NIFTYBEES have no revenue, margins or P/E, so the equity
# scorecard above cannot be applied. They are judged on what they do have:
# trend, momentum, volatility, drawdowns, strength against a benchmark and
# how consistent their yearly record is. Same output shape as analyze(),
# so the dashboard renders it with the same rings / verdict widgets.
# ========================================================================

ASSET_PILLARS = ["Trend", "Momentum", "Volatility", "Drawdown", "Relative strength", "Consistency"]
ASSET_PILLAR_WEIGHTS = {"Trend": 0.25, "Momentum": 0.25, "Volatility": 0.15,
                        "Drawdown": 0.10, "Relative strength": 0.15, "Consistency": 0.10}

ASSET_KIND_TEXT = {
    "commodity": ("commodity", "supply/demand news, the US dollar and real interest rates"),
    "forex": ("currency pair", "interest-rate differences, inflation data and central-bank decisions"),
    "crypto": ("crypto asset", "risk appetite, liquidity and flows into the big crypto funds"),
    "etf": ("fund", "the index or metal it tracks, plus its tracking error and costs"),
    "index": ("index", "the earnings and flows of its member companies"),
}


def _rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _ret_over(c, days):
    """Simple return over the last `days` trading rows (None if history is too short)."""
    return float(c.iloc[-1] / c.iloc[-days - 1] - 1) if len(c) > days else None


def asset_seasonality(hist):
    """Average calendar-month return (Jan..Dec) over the whole history."""
    c = hist["Close"].dropna()
    monthly = c.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return pd.Series(dtype=float)
    return monthly.groupby(monthly.index.month).mean()


def analyze_asset(hist, ticker, profile=None, technical_score=None, forecast_1m=None,
                  bench_hist=None, bench_name=None):
    """Research report for anything that is not a company share."""
    profile = profile or {}
    if hist is None or hist.empty or len(hist) < 60:
        return None
    c = hist["Close"].dropna()
    price = float(c.iloc[-1])
    ph = price_history_stats(hist)
    kind, drivers = ASSET_KIND_TEXT.get(profile.get("asset_class"), ("instrument", "broad market conditions"))

    sma50 = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
    daily = c.pct_change().dropna()
    vol_1y = float(daily.tail(252).std() * np.sqrt(252)) if len(daily) > 60 else None
    vol_hist = float(daily.std() * np.sqrt(252)) if len(daily) > 250 else None

    m = {
        "price": price,
        "vs_sma50": price / sma50 - 1 if pd.notna(sma50) else None,
        "vs_sma200": price / sma200 - 1 if pd.notna(sma200) else None,
        "sma50_vs_sma200": sma50 / sma200 - 1 if pd.notna(sma50) and pd.notna(sma200) else None,
        "return_1m": _ret_over(c, 21), "return_3m": _ret_over(c, 63), "return_6m": _ret_over(c, 126),
        "return_1y": ph.get("ret_1y"), "cagr_all": ph.get("cagr_all"),
        "rsi14": _rsi(c) if len(c) > 30 else None,
        "technical_score": technical_score,
        "volatility": vol_1y,
        "vol_vs_history": (vol_1y / vol_hist - 1) if (vol_1y and vol_hist) else None,
        "drawdown_from_ath": ph.get("drawdown_from_ath"), "max_dd_5y": ph.get("max_dd_5y"),
        "pct_up_years": ph.get("pct_up_years"),
        "pct_up_months": float((c.resample("ME").last().pct_change().dropna() > 0).mean()) if len(c) > 260 else None,
    }

    # strength against a benchmark (S&P 500, NIFTY 50, the dollar index, bitcoin, ...)
    rel_1y = corr = None
    if bench_hist is not None and not bench_hist.empty:
        b = bench_hist["Close"].dropna()
        joined = pd.concat([c, b], axis=1, join="inner").dropna()
        if len(joined) > 60:
            a_ret, b_ret = joined.iloc[:, 0], joined.iloc[:, 1]
            n = min(252, len(joined) - 1)
            rel_1y = float(a_ret.iloc[-1] / a_ret.iloc[-n - 1] - b_ret.iloc[-1] / b_ret.iloc[-n - 1])
            corr = float(a_ret.pct_change().tail(252).corr(b_ret.pct_change().tail(252)))
    m["relative_1y"] = rel_1y
    m["benchmark_corr"] = corr
    m["benchmark"] = bench_name

    sub = {
        "Trend": {"vs_sma50": _score(m["vs_sma50"], -0.10, 0.10), "vs_sma200": _score(m["vs_sma200"], -0.15, 0.20),
                  "sma50_vs_sma200": _score(m["sma50_vs_sma200"], -0.06, 0.06),
                  "technical_score": _score(m["technical_score"], -0.5, 0.5)},
        "Momentum": {"return_3m": _score(m["return_3m"], -0.12, 0.15), "return_6m": _score(m["return_6m"], -0.18, 0.25),
                     "return_1y": _score(m["return_1y"], -0.20, 0.30),
                     "rsi14": _score(m["rsi14"], 30, 60) if m["rsi14"] is not None else None},
        "Volatility": {"volatility": _score(m["volatility"], 0.60, 0.12),
                       "vol_vs_history": _score(m["vol_vs_history"], 0.50, -0.25)},
        "Drawdown": {"drawdown_from_ath": _score(m["drawdown_from_ath"], -0.45, 0.0),
                     "max_dd_5y": _score(m["max_dd_5y"], -0.70, -0.15)},
        "Relative strength": {"relative_1y": _score(m["relative_1y"], -0.20, 0.20)},
        "Consistency": {"pct_up_years": _score(m["pct_up_years"], 0.35, 0.75),
                        "pct_up_months": _score(m["pct_up_months"], 0.40, 0.65),
                        "cagr_all": _score(m["cagr_all"], 0.0, 0.15)},
    }
    pillars = {}
    for pl, items in sub.items():
        vals = [v for v in items.values() if v is not None]
        pillars[pl] = float(np.mean(vals)) if vals else None
    avail = {pl: sc for pl, sc in pillars.items() if sc is not None}
    total_w = sum(ASSET_PILLAR_WEIGHTS[pl] for pl in avail)
    overall = sum(ASSET_PILLAR_WEIGHTS[pl] * sc for pl, sc in avail.items()) / total_w if total_w else None
    if overall is None:
        verdict = "Not enough data"
    else:
        verdict = "Strong" if overall >= 7.5 else "Good" if overall >= 6 else "Mixed" if overall >= 4.5 else "Weak"

    trend_up = (m["vs_sma200"] or 0) > 0 and (m["sma50_vs_sma200"] or 0) > 0
    if trend_up and (pillars.get("Momentum") or 0) >= 6:
        stance = "Uptrend intact"
    elif trend_up:
        stance = "Trending up, but momentum is fading"
    elif (m["vs_sma200"] or 0) < 0:
        stance = "Below its long-term average - the trend is against it"
    else:
        stance = "Sideways - no clear trend"

    report = {
        "name": profile.get("name", ticker), "ticker": ticker, "asset_class": profile.get("asset_class"),
        "kind": kind, "drivers": drivers, "is_asset": True, "info": profile, "metrics": m, "sub_scores": sub,
        "pillars": pillars, "overall": overall, "verdict": verdict, "stance": stance,
        "history_stats": ph, "price": price, "seasonality": asset_seasonality(hist),
        "profile": {"exchange": profile.get("exchange"), "currency": profile.get("currency"),
                    "listed_since": ph.get("first_date"), "category": profile.get("category"),
                    "fund_family": profile.get("fund_family"), "total_assets": profile.get("total_assets"),
                    "expense_ratio": profile.get("expense_ratio"), "summary": profile.get("summary")},
    }
    report["strengths"], report["risks"] = _asset_bull_bear(m, ph, kind)
    report["summary"] = _asset_summary(report)
    report["watch"] = _asset_watch(report, forecast_1m)
    return report


def _asset_bull_bear(m, ph, kind):
    """Same {pillar, text, score} shape as the equity bull/bear lists."""
    good, bad = [], []

    def add(lst, pillar, text, score):
        lst.append({"pillar": pillar, "text": text, "score": score})

    if m.get("vs_sma200") is not None:
        if m["vs_sma200"] > 0.02:
            add(good, "Trend", f"Trading {_pct(m['vs_sma200'])} above its 200-day average, so the long-term "
                               f"trend is up", 8)
        elif m["vs_sma200"] < -0.02:
            add(bad, "Trend", f"Trading {_pct(abs(m['vs_sma200']))} below its 200-day average, so the long-term "
                              f"trend is down", 2)
    if m.get("sma50_vs_sma200") is not None:
        if m["sma50_vs_sma200"] > 0:
            add(good, "Trend", "The 50-day average sits above the 200-day one, the classic uptrend signal", 8)
        else:
            add(bad, "Trend", "The 50-day average sits below the 200-day one, the classic downtrend signal", 2)
    if m.get("return_1y") is not None:
        if m["return_1y"] > 0.10:
            add(good, "Momentum", f"Up {_pct(m['return_1y'])} over the past year", 8)
        elif m["return_1y"] < -0.05:
            add(bad, "Momentum", f"Down {_pct(abs(m['return_1y']))} over the past year", 2)
    if m.get("relative_1y") is not None:
        bench = m.get("benchmark") or "its benchmark"
        if m["relative_1y"] > 0.03:
            add(good, "Relative strength", f"Beating {bench} by {_pct(m['relative_1y'])} over the last year", 8)
        elif m["relative_1y"] < -0.03:
            add(bad, "Relative strength", f"Lagging {bench} by {_pct(abs(m['relative_1y']))} over the last year", 2)
    if m.get("volatility") is not None:
        if m["volatility"] > 0.35:
            add(bad, "Volatility", f"Swings hard: about {_pct(m['volatility'])} annualised volatility, so position "
                                   f"sizes need to be smaller", 2)
        elif m["volatility"] < 0.15:
            add(good, "Volatility", f"Calm by historical standards: about {_pct(m['volatility'])} annualised "
                                    f"volatility", 8)
    if m.get("drawdown_from_ath") is not None:
        if m["drawdown_from_ath"] < -0.15:
            add(bad, "Drawdown", f"Still {_pct(abs(m['drawdown_from_ath']))} below its all-time high", 2)
        elif m["drawdown_from_ath"] > -0.03:
            add(good, "Drawdown", "Sitting at or very near its all-time high", 8)
    if ph.get("max_dd_5y") is not None and ph["max_dd_5y"] < -0.35:
        add(bad, "Drawdown", f"Has fallen {_pct(abs(ph['max_dd_5y']))} peak-to-trough within the last five years", 2)
    if m.get("pct_up_years") is not None and m["pct_up_years"] >= 0.6:
        add(good, "Consistency", f"Positive in {m['pct_up_years']:.0%} of the calendar years on record", 8)
    if m.get("cagr_all") is not None and m["cagr_all"] > 0.08:
        add(good, "Consistency", f"Has compounded at {_pct(m['cagr_all'])} a year over its whole history", 8)
    if m.get("rsi14") is not None:
        if m["rsi14"] > 70:
            add(bad, "Momentum", f"RSI is {m['rsi14']:.0f}: overbought in the short term", 2)
        elif m["rsi14"] < 30:
            add(good, "Momentum", f"RSI is {m['rsi14']:.0f}: oversold, which often precedes a bounce", 7)
    return good[:6], bad[:6]


def _asset_summary(r):
    m, ph = r["metrics"], r["history_stats"]
    bits = [f"**{r['name']}** is a {r['kind']}; its price is driven mainly by {r['drivers']}."]
    if m.get("return_1y") is not None:
        tail = f" and {_pct(ph['cagr_all'])} a year since {ph['first_date']:%b %Y}." if ph.get("cagr_all") else "."
        bits.append(f"It has returned {_pct(m['return_1y'])} over the past year{tail}")
    if m.get("vs_sma200") is not None:
        side = "above" if m["vs_sma200"] >= 0 else "below"
        dd = m.get("drawdown_from_ath")
        dd_txt = f", and sits {_pct(abs(dd))} below its all-time high" if dd is not None and dd < 0 else ""
        bits.append(f"It trades {_pct(abs(m['vs_sma200']))} {side} its 200-day average{dd_txt}.")
    if m.get("volatility") is not None:
        vv = m.get("vol_vs_history")
        tail = f", {'higher' if vv > 0 else 'lower'} than its own long-run average." if vv is not None else "."
        bits.append(f"Annualised volatility is about {_pct(m['volatility'])}{tail}")
    if m.get("relative_1y") is not None and m.get("benchmark"):
        verb = "ahead of" if m["relative_1y"] >= 0 else "behind"
        corr = f" (correlation {m['benchmark_corr']:.2f})" if m.get("benchmark_corr") is not None else ""
        bits.append(f"Over the last year it is {_pct(abs(m['relative_1y']))} {verb} {m['benchmark']}{corr}.")
    if r.get("overall") is not None:
        bits.append(f"Overall verdict: **{r['verdict']} ({r['overall']:.1f}/10)** - {r['stance'].lower()}.")
    return " ".join(b for b in bits if b)


def _asset_watch(r, forecast_1m):
    """Plain strings (markdown bold allowed) -- same as the equity report."""
    m, out = r["metrics"], []
    if m.get("vs_sma200"):
        lvl = m["price"] / (1 + m["vs_sma200"])
        out.append(f"**200-day average at {lvl:,.2f}** - the line that separates a long-term uptrend from a downtrend "
                   f"(price is {_pct(abs(m['vs_sma200']))} {'above' if m['vs_sma200'] > 0 else 'below'} it).")
    ph = r["history_stats"]
    if ph.get("ath"):
        out.append(f"**All-time high {ph['ath']:,.2f}** set on {ph['ath_date']:%d %b %Y}"
                   + (f", {_pct(abs(ph['drawdown_from_ath']))} above today's price."
                      if ph.get("drawdown_from_ath") else "."))
    seas = r.get("seasonality")
    if seas is not None and len(seas) == 12:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        best, worst = int(seas.idxmax()), int(seas.idxmin())
        out.append(f"**Seasonality:** historically strongest in {months[best - 1]} ({seas.max():+.1%} average month) "
                   f"and weakest in {months[worst - 1]} ({seas.min():+.1%}). A tendency, not a rule.")
    if m.get("volatility"):
        daily = m["volatility"] / (252 ** 0.5)
        out.append(f"**Typical daily move** is about {daily:.1%}, so anything much larger is news-driven.")
    if forecast_1m and forecast_1m.get("price"):
        out.append(f"**The model's 1-month view:** {forecast_1m['price']:,.2f} "
                   f"({forecast_1m.get('change_pct', 0):+.2f}%) - see the forecast tabs for the likely range.")
    if m.get("benchmark") and m.get("benchmark_corr") is not None:
        link = ("moves closely with" if m["benchmark_corr"] > 0.5 else
                "moves loosely with" if m["benchmark_corr"] > 0.15 else
                "moves against" if m["benchmark_corr"] < -0.15 else "barely tracks")
        out.append(f"**Link to {m['benchmark']}:** it {link} it (correlation {m['benchmark_corr']:.2f} over the "
                   f"last year), which matters for diversification.")
    return out
