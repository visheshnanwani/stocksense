"""
ipo_analyzer.py
-----------------
IPO research for Indian IPOs (NSE mainboard + SME):

  Data sources (all public web pages / feeds, fetched lightly and cached):
    * NSE (nseindia.com)  -- official list of open / upcoming / past issues,
      price band, lot size, issue details and LIVE subscription by investor
      category (QIB, NII, Retail, employees).
    * IPOWatch (ipowatch.in) -- grey market premium (GMP), and per-IPO pages
      with company financials, KPIs (ROE, ROCE, margins, debt), valuation,
      listed peers, promoter holding, fresh issue vs offer-for-sale, anchor
      investors and objects of the issue; plus a history of past IPOs'
      GMP vs. their actual listing price.

  Analysis:
    * GMP reliability -- how well GMP predicted the actual listing price
      over hundreds of past IPOs (direction hit rate, typical error).
    * A rule-based "should I apply?" scorecard for each IPO, split into a
      LISTING-GAIN view (GMP, subscription -- especially QIB demand) and a
      LONG-TERM view (growth, profitability, valuation vs peers, issue
      structure, promoter holding, debt), ending in a recommendation:
      Apply / Apply for listing gains only / Long-term only / Neutral / Avoid.

  IMPORTANT: GMP is an UNOFFICIAL, unregulated grey-market quote and can
  change sharply or be manipulated; subscription numbers change through the
  bidding window. This is an analytical aid, not advice.
  Data belongs to the respective sites -- keep request volumes low (the
  dashboard caches everything) and respect their terms of use.
"""

import datetime
import difflib
import io
import re

import numpy as np
import pandas as pd
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")
IPOWATCH_GMP_URL = "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/"
# IPOWatch moved the listing-performance table off the GMP page onto its own
# page some time before Oct 2026. The GMP page now carries only the two live
# boards, which is why anything keyed off "history" silently went blank.
IPOWATCH_PERF_URL = "https://ipowatch.in/ipo-performance-tracker/"
NSE = "https://www.nseindia.com"


# ------------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------------
def to_num(x):
    """'₹1,091.68 Crores' -> 1091.68 ; '15.37 %' -> 15.37 ; '–' / '[.]' -> None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("₹", "").replace("Rs.", "").strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def norm_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"\b(limited|ltd|ipo|india|private|pvt|the|co|company)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _header_table(t: pd.DataFrame) -> pd.DataFrame:
    """IPOWatch tables put the header in the first row."""
    t = t.copy()
    t.columns = [str(c).strip() for c in t.iloc[0]]
    return t.iloc[1:].reset_index(drop=True)


def _kv_table(t: pd.DataFrame) -> dict:
    """Two-column 'label | value' tables -> dict."""
    out = {}
    for _, r in t.iterrows():
        k = str(r.iloc[0]).strip().rstrip(":")
        v = r.iloc[1] if len(r) > 1 else None
        if k and k.lower() != "nan":
            out[k] = None if (isinstance(v, float) and np.isnan(v)) else str(v).strip()
    return out


# ------------------------------------------------------------------------
# IPOWatch: GMP lists + GMP history
# ------------------------------------------------------------------------
def fetch_gmp_tables() -> dict:
    """
    Returns {"mainboard": df, "sme": df, "history": df}.
    Live tables: name, gmp, price, est_listing, est_gain_pct, dates, status, updated, link, type.
    History: name, ipo_price, gmp, listing_price, link, predicted_gain_pct, actual_gain_pct.
    """
    from bs4 import BeautifulSoup
    html = requests.get(IPOWATCH_GMP_URL, headers={"User-Agent": UA}, timeout=30).text
    soup = BeautifulSoup(html, "lxml")
    out = {"mainboard": pd.DataFrame(), "sme": pd.DataFrame(), "history": pd.DataFrame()}
    for table in soup.find_all("table"):
        heading = table.find_previous(["h2", "h3", "h4"])
        heading = heading.get_text(strip=True).lower() if heading else ""
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        head = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])]
        recs = []
        for r in rows[1:]:
            cells = r.find_all(["td", "th"])
            a = r.find("a", href=True)
            vals = [c.get_text(" ", strip=True) for c in cells]
            if len(vals) != len(head):
                continue
            rec = dict(zip(head, vals))
            rec["link"] = a["href"] if a else None
            recs.append(rec)
        df = pd.DataFrame(recs)
        if df.empty:
            continue
        if "performance" in heading:
            df = df.rename(columns={"IPO Name": "name", "IPO Price": "ipo_price", "IPO GMP": "gmp", "Listing Price": "listing_price"})
            for c in ("ipo_price", "gmp", "listing_price"):
                df[c] = df[c].map(to_num)
            df = df.dropna(subset=["ipo_price", "listing_price"])
            df = df[df["ipo_price"] > 0]
            df["predicted_gain_pct"] = df["gmp"] / df["ipo_price"] * 100
            df["actual_gain_pct"] = (df["listing_price"] / df["ipo_price"] - 1) * 100
            out["history"] = df
        elif "gmp" in heading:
            df = df.rename(columns={"IPO Name": "name", "IPO GMP*": "gmp", "IPO GMP": "gmp", "Price Band": "price",
                                    "Est. Listing": "est_listing", "Date": "dates", "Status": "status",
                                    "Last Updated": "updated", "Trend": "trend"})
            df["gmp"] = df["gmp"].map(to_num)
            df["price"] = df["price"].map(to_num)
            df["est_gain_pct"] = df["est_listing"].map(
                lambda s: to_num(re.search(r"\(([^)]*)\)", s).group(1)) if isinstance(s, str) and "(" in s else None)
            df["est_listing"] = df["est_listing"].map(to_num)
            df["type"] = "SME" if "sme" in heading else "Mainboard"
            out["sme" if "sme" in heading else "mainboard"] = df
    return out


def fetch_listing_history() -> pd.DataFrame:
    """
    Every IPO IPOWatch has tracked to listing: issue price, listing price and
    the gain, back to 2022. Columns match what fetch_gmp_tables used to return
    under "history" so callers do not change -- except `gmp`, which this source
    does not publish and which is therefore NaN. Historical GMP has to be
    accumulated from live snapshots instead (see gmp_log.py).
    """
    from bs4 import BeautifulSoup
    html = requests.get(IPOWATCH_PERF_URL, headers={"User-Agent": UA}, timeout=30).text
    soup = BeautifulSoup(html, "lxml")
    frames = []
    for table in soup.find_all("table"):
        heading = table.find_previous(["h2", "h3", "h4"])
        heading = heading.get_text(strip=True) if heading else ""
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        head = [c.get_text(strip=True).lower() for c in rows[0].find_all(["td", "th"])]
        if not any("ipo name" in h for h in head) or not any("listing price" in h for h in head):
            continue
        recs = []
        for r in rows[1:]:
            vals = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
            if len(vals) != len(head):
                continue
            a = r.find("a", href=True)
            rec = dict(zip(head, vals))
            gain_key = next((k for k in rec if "listing gain" in k or k == "gain/loss"), None)
            recs.append({"name": rec.get("ipo name"),
                         "ipo_price": to_num(rec.get("ipo price")),
                         "listing_price": to_num(rec.get("listing price")),
                         "printed_gain": to_num(rec.get(gain_key)) if gain_key else None,
                         "link": a["href"] if a else None,
                         "table": heading})
        if recs:
            frames.append(pd.DataFrame(recs))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["name", "ipo_price", "listing_price"])
    df = df[df["ipo_price"] > 0]
    df["actual_gain_pct"] = (df["listing_price"] / df["ipo_price"] - 1) * 100

    # CROSS-CHECK, because the source has real typos in it. Waaree Energies is
    # printed as "Rs 1,503 -> Rs 200, 66%": the gain is right and the listing
    # price has lost a digit, which recomputing from the prices turns into a
    # phantom -87% listing. The older table also prints its gain unsigned, so
    # only the MAGNITUDES can be compared. A row is kept when the two agree;
    # when they contradict each other the row is not trustworthy either way and
    # is dropped rather than guessed at.
    pr = pd.to_numeric(df["printed_gain"], errors="coerce").abs()
    gap = (df["actual_gain_pct"].abs() - pr).abs()
    tol = np.maximum(3.0, pr * 0.10)
    df["consistent"] = pr.isna() | (gap <= tol)
    df = df[df["consistent"]].drop(columns=["printed_gain", "consistent"])

    df["gmp"] = np.nan
    df["predicted_gain_pct"] = np.nan
    df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    return df


def listing_base_rates(hist: pd.DataFrame) -> dict:
    """What the listing record itself says, with no GMP involved."""
    h = hist.dropna(subset=["actual_gain_pct"])
    if len(h) < 10:
        return {}
    g = h["actual_gain_pct"]
    return {"n": len(h), "pct_listed_up": float((g > 0).mean()),
            "avg_actual_gain": float(g.mean()), "median_gain": float(g.median()),
            "pct_up_10": float((g >= 10).mean()), "pct_down_10": float((g <= -10).mean()),
            "best": float(g.max()), "worst": float(g.min()),
            "best_name": str(h.loc[g.idxmax(), "name"]), "worst_name": str(h.loc[g.idxmin(), "name"])}


def gmp_reliability(history: pd.DataFrame) -> dict:
    """How well did GMP predict the actual listing price?"""
    h = history.dropna(subset=["predicted_gain_pct", "actual_gain_pct"])
    if len(h) < 10:
        return {}
    pos = h[h["predicted_gain_pct"] > 0]
    zero = h[h["predicted_gain_pct"] <= 0]
    err = h["actual_gain_pct"] - h["predicted_gain_pct"]
    return {
        "n": len(h),
        "direction_hit_rate": float(((h["predicted_gain_pct"] > 0) == (h["actual_gain_pct"] > 0)).mean()),
        "listed_up_when_gmp_positive": float((pos["actual_gain_pct"] > 0).mean()) if len(pos) else None,
        "listed_up_when_gmp_zero": float((zero["actual_gain_pct"] > 0).mean()) if len(zero) else None,
        "median_abs_error_pts": float(err.abs().median()),
        "mean_error_pts": float(err.mean()),
        "correlation": float(h["predicted_gain_pct"].corr(h["actual_gain_pct"])),
        "avg_actual_gain": float(h["actual_gain_pct"].mean()),
        "pct_listed_up": float((h["actual_gain_pct"] > 0).mean()),
    }


# ------------------------------------------------------------------------
# IPOWatch: one IPO's detail page
# ------------------------------------------------------------------------
def fetch_ipo_detail(url: str) -> dict:
    """Parse the main-content tables of an IPOWatch IPO page by their headings."""
    from bs4 import BeautifulSoup
    html = requests.get(url, headers={"User-Agent": UA}, timeout=30).text
    soup = BeautifulSoup(html, "lxml")
    content = soup.find(class_=lambda c: c and ("entry-content" in c or "post-content" in c)) or soup
    d = {"url": url, "source": "IPOWatch",
         "title": soup.title.get_text(strip=True) if soup.title else None}
    for table in content.find_all("table"):
        heading = table.find_previous(["h2", "h3", "h4"])
        heading = heading.get_text(strip=True).lower() if heading else ""
        try:
            t = pd.read_html(io.StringIO(str(table)), flavor="lxml")[0]
        except Exception:
            continue
        if "financial" in heading and "financials" not in d:
            ft = _header_table(t)
            if {"Revenue", "PAT"}.issubset(ft.columns):
                for c in ft.columns[1:]:
                    ft[c] = ft[c].map(to_num)
                d["financials"] = ft.rename(columns={ft.columns[0]: "Period"})
        elif "valuation" in heading or ("kpi" in str(t.iloc[0, 0]).lower()):
            kv = _kv_table(t.iloc[1:] if str(t.iloc[0, 0]).strip().upper() == "KPI" else t)
            d["kpi"] = {k: to_num(v) for k, v in kv.items()}
        elif "peer" in heading:
            d["peers"] = _header_table(t)
        elif "promoter" in heading or "holding" in heading:
            ht = _header_table(t)
            d["holding"] = ht
            prom = ht[ht.iloc[:, 0].astype(str).str.contains("Promoter", case=False)]
            if len(prom):
                row = prom.iloc[0]
                d["promoter_pre_pct"] = to_num(row.get("Pre IPO % Shares"))
                d["promoter_post_pct"] = to_num(row.get("Post IPO % Shares"))
        elif "objects" in heading:
            d["objects"] = _header_table(t) if t.shape[1] >= 2 else t
        elif "market lot" in heading or "lot" in heading:
            d["lots"] = _header_table(t)
        elif "reservation" in heading:
            d["reservation"] = _header_table(t)
        elif "anchor" in heading:
            d["anchor"] = _kv_table(t)
        elif "dates" in heading:
            d["dates"] = _kv_table(t)
        elif "details" in heading and "issue" not in d:
            d["issue"] = _kv_table(t)
    iss = d.get("issue", {})
    d["issue_size_cr"] = to_num(iss.get("Issue Size"))
    d["fresh_issue_cr"] = to_num(iss.get("Fresh Issue"))
    ofs_txt = next((v for k, v in iss.items() if "offer for sale" in k.lower()), None)
    d["ofs_text"] = ofs_txt
    d["price_band"] = iss.get("IPO Price Band")
    d["listing_at"] = next((v for k, v in iss.items() if "listing at" in k.lower()), None)
    return d


# ------------------------------------------------------------------------
# NSE: official issue lists, details and subscription
# ------------------------------------------------------------------------
def _nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                      "Accept-Language": "en-US,en;q=0.9", "Referer": f"{NSE}/market-data/all-upcoming-issues-ipo"})
    s.get(NSE, timeout=25)  # sets the cookies NSE's API requires
    return s


def fetch_nse_lists() -> dict:
    """{"current": df, "upcoming": df, "past": df} from NSE's public APIs."""
    s = _nse_session()
    out = {}
    for key, path in (("current", "/api/ipo-current-issue"), ("upcoming", "/api/all-upcoming-issues?category=ipo"),
                      ("past", "/api/public-past-issues")):
        try:
            out[key] = pd.DataFrame(s.get(NSE + path, timeout=25).json())
        except Exception:
            out[key] = pd.DataFrame()
    cur = out.get("current")
    if cur is not None and not cur.empty and "noOfTime" in cur.columns:
        cur["subscription_x"] = pd.to_numeric(cur["noOfTime"], errors="coerce")
    return out


def fetch_nse_detail(symbol: str, series: str = "EQ") -> dict:
    """Issue info (lot, face value, lead managers, registrar...) and category-wise subscription."""
    s = _nse_session()
    try:
        j = s.get(f"{NSE}/api/ipo-detail?symbol={symbol}&series={series}", timeout=25).json()
    except Exception:
        return {}
    info = {}
    for x in (j.get("issueInfo") or {}).get("dataList", []):
        t, v = x.get("title"), x.get("value")
        if t and v and not str(v).startswith("<a"):
            info[t] = re.sub(r"\s+", " ", str(v).strip('"')).strip()
    subs = {}
    for b in j.get("bidDetails") or []:
        cat, times = b.get("category") or "", to_num(b.get("noOfTime"))
        if times is None:
            continue
        c = cat.lower().strip()
        key = ("QIB" if c.startswith("qualified institutional") else "NII" if c.startswith("non institutional")
               else "Retail" if c.startswith("retail") else "Employees" if c.startswith("employee")
               else "Shareholders" if "shareholder" in c else "Total" if c.startswith("total") else None)
        if key and key not in subs:
            subs[key] = times
    return {"info": info, "subscription": subs}


# ------------------------------------------------------------------------
# Matching NSE <-> IPOWatch names
# ------------------------------------------------------------------------
def match_name(name: str, candidates) -> str:
    """
    Match an IPO name across sources ("Moneyview" vs "Moneyview Limited").
    The distinctive FIRST word must agree -- shared generic words like
    "engineering" or "industries" alone are never enough.
    """
    target = norm_name(name)
    if not target:
        return None
    options = {norm_name(c): c for c in candidates if isinstance(c, str) and norm_name(c)}
    if target in options:
        return options[target]
    first = target.split()[0]
    same_first = {k: v for k, v in options.items() if k.split()[0] == first}
    for k, v in same_first.items():  # one name is a prefix of the other
        if k.startswith(target) or target.startswith(k):
            return v
    best = difflib.get_close_matches(target, list(same_first.keys()), n=1, cutoff=0.75)
    return same_first[best[0]] if best else None


def nse_names(df: pd.DataFrame) -> pd.Series:
    """NSE lists use 'companyName' for some rows and 'company' for others."""
    a = df["companyName"] if "companyName" in df.columns else pd.Series(index=df.index, dtype=object)
    b = df["company"] if "company" in df.columns else pd.Series(index=df.index, dtype=object)
    return a.fillna(b)


# ------------------------------------------------------------------------
# Scorecard + recommendation
# ------------------------------------------------------------------------
def _score(x, bad, good):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(np.clip((x - bad) / (good - bad) * 10, 0, 10))


def _cagr(values, years):
    if values is None or len(values) < 2 or years <= 0:
        return None
    a, b = values[0], values[-1]
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return (b / a) ** (1 / years) - 1


def analyze_ipo(row: dict, detail: dict = None, nse: dict = None, reliability: dict = None,
                subscription_final: bool = True) -> dict:
    """
    row: the IPO's GMP-table row (gmp, price, est_gain_pct, type, status...)
    detail: fetch_ipo_detail() output; nse: fetch_nse_detail() output.
    subscription_final: False while bidding is still in progress -- early-day
    subscription is naturally low, so it's shown but not scored.
    """
    detail, nse = detail or {}, nse or {}
    m = {}
    price = row.get("price") or to_num(detail.get("price_band", "").split("to")[-1] if detail.get("price_band") else None)
    m["gmp_gain_pct"] = row.get("est_gain_pct") if row.get("est_gain_pct") is not None else (
        row["gmp"] / price * 100 if row.get("gmp") is not None and price else None)
    # A Rs.0 GMP before the issue opens usually means grey-market trading hasn't started yet.
    m["gmp_not_started"] = (row.get("gmp") == 0 and str(row.get("status", "")).lower().startswith("upcoming"))
    m["subscription_final"] = subscription_final
    subs = nse.get("subscription", {})
    m["subscription_total"] = subs.get("Total")
    m["subscription_qib"] = subs.get("QIB")
    m["subscription_retail"] = subs.get("Retail")
    m["subscription_nii"] = subs.get("NII")

    fin = detail.get("financials")
    annual = None
    if fin is not None and len(fin):
        annual = fin[fin["Period"].astype(str).str.fullmatch(r"\d{4}")]  # full fiscal years only
        if len(annual) >= 2:
            yrs = int(annual["Period"].iloc[-1]) - int(annual["Period"].iloc[0])
            m["revenue_cagr"] = _cagr(list(annual["Revenue"]), yrs)
            m["pat_cagr"] = _cagr(list(annual["PAT"]), yrs)
        last = (annual if annual is not None and len(annual) else fin).iloc[-1]
        if last.get("Revenue"):
            m["pat_margin"] = last["PAT"] / last["Revenue"] if last.get("PAT") is not None else None
        m["latest_pat"] = last.get("PAT")
        m["loss_making"] = last.get("PAT") is not None and last["PAT"] < 0
    kpi = detail.get("kpi", {}) or {}
    m["roe"] = kpi.get("ROE") / 100 if kpi.get("ROE") is not None else None
    m["roce"] = kpi.get("ROCE") / 100 if kpi.get("ROCE") is not None else None
    m["debt_to_equity"] = kpi.get("Debt to equity ratio")
    m["pe"] = kpi.get("Price/Earning P/E Ratio")
    if m["pe"] is None and kpi.get("Earning Per Share (EPS)") and price:
        eps = kpi["Earning Per Share (EPS)"]
        m["pe"] = price / eps if eps and eps > 0 else None
    peers = detail.get("peers")
    peer_pe = None
    if peers is not None and len(peers):
        pe_col = next((c for c in peers.columns if "p/e" in c.lower() or "pe ratio" in c.lower()), None)
        if pe_col:
            vals = pd.to_numeric(peers[pe_col].map(to_num), errors="coerce")
            vals = vals[(vals > 0) & (vals < 500)]
            if len(vals):
                peer_pe = float(vals.median())
    m["peer_median_pe"] = peer_pe
    m["pe_vs_peers"] = m["pe"] / peer_pe if m.get("pe") and peer_pe else None
    size, fresh = detail.get("issue_size_cr"), detail.get("fresh_issue_cr")
    m["issue_size_cr"] = size
    m["fresh_share"] = min(fresh / size, 1.0) if size and fresh is not None else (1.0 if size and not detail.get("ofs_text") else None)
    m["promoter_post_pct"] = detail.get("promoter_post_pct")
    m["is_sme"] = row.get("type") == "SME"

    sub = {
        "Listing demand": {
            "gmp_gain_pct": None if m["gmp_not_started"] else _score(m["gmp_gain_pct"], -5, 30),
            "subscription_total": _score(np.log10(m["subscription_total"]) if subscription_final and m.get("subscription_total")
                                         and m["subscription_total"] > 0 else None, 0, np.log10(50)),
            "subscription_qib": _score(np.log10(m["subscription_qib"]) if subscription_final and m.get("subscription_qib")
                                       and m["subscription_qib"] > 0 else None, 0, np.log10(50)),
        },
        "Growth": {"revenue_cagr": _score(m.get("revenue_cagr"), 0, 0.30), "pat_cagr": _score(m.get("pat_cagr"), 0, 0.30)},
        "Profitability": {"pat_margin": _score(m.get("pat_margin"), 0, 0.20), "roe": _score(m.get("roe"), 0.05, 0.25),
                          "roce": _score(m.get("roce"), 0.08, 0.30)},
        "Valuation": {"pe_vs_peers": _score(m.get("pe_vs_peers"), 1.5, 0.7),
                      "pe": _score(m.get("pe"), 60, 15) if m.get("pe") else None},
        "Issue quality": {"fresh_share": _score(m.get("fresh_share"), 0, 1.0),
                          "promoter_post_pct": _score(m.get("promoter_post_pct"), 20, 60),
                          "debt_to_equity": _score(m.get("debt_to_equity"), 2.0, 0.2)},
    }
    if m.get("loss_making"):
        sub["Profitability"]["pat_margin"] = 0.0
    pillars = {p: (float(np.mean([v for v in s.values() if v is not None])) if any(v is not None for v in s.values()) else None)
               for p, s in sub.items()}
    listing = pillars["Listing demand"]
    lt_parts = [pillars[p] for p in ("Growth", "Profitability", "Valuation", "Issue quality") if pillars[p] is not None]
    long_term = float(np.mean(lt_parts)) if lt_parts else None

    rec, tone, why = _recommend(listing, long_term, m)
    strengths, risks = _ipo_points(m, sub, reliability)
    return {"metrics": m, "sub_scores": sub, "pillars": pillars, "listing_score": listing, "long_term_score": long_term,
            "recommendation": rec, "tone": tone, "why": why, "strengths": strengths, "risks": risks, "price": price}


def _recommend(listing, long_term, m):
    if listing is None and long_term is None:
        return "Not enough data yet", "neutral", "Financials and demand data aren't available for this IPO yet."
    L = listing if listing is not None else 5.0
    T = long_term if long_term is not None else 5.0
    if T >= 6.5 and L >= 5:
        return "Apply", "up", "Solid fundamentals and healthy demand -- suitable for listing gains and holding."
    if L >= 6.5 and T < 5:
        return "Apply for listing gains only", "warn", ("Strong demand/GMP but weak fundamentals or rich valuation -- "
                                                        "consider selling on listing day rather than holding.")
    if T >= 6.5 and L < 5:
        return "Long-term only", "info", "Good business, but demand/GMP is soft -- apply only if you can hold through a flat listing."
    if L < 4.5 and T < 4.5:
        return "Avoid", "down", "Both demand and fundamentals look weak."
    return "Neutral -- apply only with risk appetite", "neutral", "Mixed signals: no strong case either way."


def _ipo_points(m, sub, reliability):
    s, r = [], []
    g = None if m.get("gmp_not_started") else m.get("gmp_gain_pct")
    if m.get("gmp_not_started"):
        r.append("No grey-market premium yet (₹0) -- grey-market trading usually starts a few days before the issue opens")
    if not m.get("subscription_final") and m.get("subscription_total") is not None:
        r.append(f"Bidding still in progress ({m['subscription_total']:.2f}x so far) -- subscription usually jumps on the "
                 "last day, so it isn't scored yet")
    if g is not None and g >= 15:
        s.append(f"Grey market premium implies a {g:+.1f}% listing gain")
    elif g is not None and g <= 2:
        r.append(f"Grey market premium implies only {g:+.1f}% on listing")
    q = m.get("subscription_qib") if m.get("subscription_final") else None
    if q is not None:
        if q >= 20:
            s.append(f"Institutions (QIB) subscribed {q:.1f}x -- strong smart-money demand")
        elif q < 1:
            r.append(f"QIB portion only {q:.2f}x subscribed so far -- weak institutional demand")
    t = m.get("subscription_total") if m.get("subscription_final") else None
    if t is not None and t >= 30:
        s.append(f"Overall subscription {t:.1f}x")
    if m.get("revenue_cagr") is not None:
        (s if m["revenue_cagr"] >= 0.2 else r if m["revenue_cagr"] < 0.05 else [])\
            .append(f"Revenue grew {m['revenue_cagr'] * 100:.1f}% a year")
    if m.get("pat_cagr") is not None:
        (s if m["pat_cagr"] >= 0.2 else r if m["pat_cagr"] < 0 else [])\
            .append(f"Profit (PAT) grew {m['pat_cagr'] * 100:+.1f}% a year")
    if m.get("loss_making"):
        r.append("Company is loss-making in the latest period")
    elif m.get("pat_margin") is not None:
        (s if m["pat_margin"] >= 0.15 else r if m["pat_margin"] < 0.05 else []).append(f"Net (PAT) margin {m['pat_margin'] * 100:.1f}%")
    if m.get("roe") is not None:
        (s if m["roe"] >= 0.2 else r if m["roe"] < 0.08 else []).append(f"Return on equity {m['roe'] * 100:.1f}%")
    if m.get("pe_vs_peers") is not None:
        (s if m["pe_vs_peers"] <= 0.85 else r if m["pe_vs_peers"] >= 1.3 else [])\
            .append(f"Priced at {m['pe']:.1f}x earnings vs listed peers' median {m['peer_median_pe']:.1f}x")
    if m.get("fresh_share") is not None:
        if m["fresh_share"] >= 0.8:
            s.append(f"{m['fresh_share'] * 100:.0f}% of the issue is fresh capital going into the company")
        elif m["fresh_share"] <= 0.3:
            r.append(f"Mostly an offer-for-sale: only {m['fresh_share'] * 100:.0f}% is fresh money -- existing holders are cashing out")
    if m.get("promoter_post_pct") is not None and m["promoter_post_pct"] < 30:
        r.append(f"Promoters will hold only {m['promoter_post_pct']:.1f}% after the IPO")
    if m.get("debt_to_equity") is not None and m["debt_to_equity"] > 1.5:
        r.append(f"High debt (debt/equity {m['debt_to_equity']:.2f})")
    if m.get("is_sme"):
        r.append("SME IPO: smaller company, lower liquidity, large lot sizes and higher risk than mainboard IPOs")
    if reliability and g is not None:
        r.append(f"GMP is unofficial: it predicted the listing direction correctly in "
                 f"{reliability['direction_hit_rate'] * 100:.0f}% of {reliability['n']} past IPOs, typical miss "
                 f"±{reliability['median_abs_error_pts']:.0f} pts")
    return s[:7], r[:7]


def is_subscription_final(end_date_str) -> bool:
    """True once bidding has closed or it's the last bidding day (NSE end dates like '24-Sep-2026')."""
    try:
        end = pd.to_datetime(end_date_str, dayfirst=True).date()
    except Exception:
        return True
    return datetime.date.today() >= end


# ------------------------------------------------------------------------
# Fallback detail sources: NSE's own issue info, and Yahoo once listed
# ------------------------------------------------------------------------
_AMOUNT_UNITS = {"million": 0.1, "millions": 0.1, "mn": 0.1,        # 1 crore = 10 million
                 "crore": 1.0, "crores": 1.0, "cr": 1.0,
                 "billion": 100.0, "billions": 100.0, "bn": 100.0,
                 "lakh": 0.01, "lakhs": 0.01}


def _amount_to_cr(text):
    """'Rs.3,600 million' -> 360.0 (crore). None when nothing parseable."""
    if not text:
        return None
    m = re.search(r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(million|millions|mn|crores?|cr|billion|billions|bn|lakhs?)",
                  str(text), re.I)
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return value * _AMOUNT_UNITS.get(m.group(2).lower(), 1.0)


def parse_nse_issue_info(info: dict, listing_date: str = None) -> dict:
    """Turn NSE's issue-information block into the same shape fetch_ipo_detail()
    returns, so the analysis works for IPOs IPOWatch never listed."""
    info = info or {}
    d = {"source": "NSE"}
    if info.get("Price Range"):
        d["price_band"] = info["Price Range"]

    lot_text = info.get("Bid Lot") or info.get("Minimum Order Quantity") or ""
    m = re.search(r"([\d,]+)", str(lot_text))
    if m:
        d["lot_size"] = to_num(m.group(1))

    size_text = str(info.get("Issue Size") or "")
    if size_text:
        # "... Fresh Issue aggregating up to Rs.3,600 million and Offer for Sale
        # aggregating up to Rs.1,400 million ..." -- split on the OFS phrase,
        # because the amounts themselves contain full stops ("Rs.3,600").
        parts = re.split(r"offer for sale", size_text, flags=re.I)
        fresh = _amount_to_cr(parts[0]) if re.search(r"fresh issue", parts[0], re.I) else None
        ofs = _amount_to_cr(parts[1]) if len(parts) > 1 else None
        ofs_part = re.search(r"offer for sale.*", size_text, re.I) if ofs else None
        total = None
        if fresh is not None or ofs is not None:
            total = (fresh or 0) + (ofs or 0)
        else:
            total = _amount_to_cr(size_text)
        d["issue_size_cr"] = total
        d["fresh_issue_cr"] = fresh
        if ofs:
            d["ofs_text"] = ofs_part.group(0)
        d["issue_size_text"] = size_text

    period = str(info.get("Issue Period") or "")
    if " to " in period:
        opened, closed = [p.strip() for p in period.split(" to ", 1)]
        d["dates"] = {"IPO Open Date": opened, "IPO Close Date": closed}
    if listing_date:
        d.setdefault("dates", {})["IPO Listing Date"] = str(listing_date)
    return d


def detail_from_yahoo(symbol: str, issue_price: float = None) -> dict:
    """Financials + ratios for an IPO that has already listed. Yahoo publishes
    3-4 years of statements for new listings within days, which is exactly what
    the growth / profitability / valuation pillars need."""
    import yfinance as yf

    out = {"source": "Yahoo Finance"}
    tk = None
    for suffix in (".NS", ".BO"):
        try:
            t = yf.Ticker(symbol + suffix)
            if (t.info or {}).get("marketCap"):
                tk, out["symbol"] = t, symbol + suffix
                break
        except Exception:
            continue
    if tk is None:
        return {}

    info = tk.info or {}
    try:
        fin = tk.financials
    except Exception:
        fin = None
    if fin is not None and not fin.empty:
        rows = {}
        for label, names in (("Revenue", ("Total Revenue", "Operating Revenue")),
                             ("PAT", ("Net Income Common Stockholders", "Net Income",
                                      "Net Income From Continuing Operation Net Minority Interest"))):
            for n in names:
                if n in fin.index:
                    rows[label] = fin.loc[n]
                    break
        if "Revenue" in rows and "PAT" in rows:
            periods = [c.year for c in fin.columns][::-1]        # oldest first
            out["financials"] = pd.DataFrame({
                "Period": [str(p) for p in periods],
                "Revenue": [rows["Revenue"].get(c) for c in fin.columns][::-1],
                "PAT": [rows["PAT"].get(c) for c in fin.columns][::-1],
            })

    kpi = {}
    if info.get("returnOnEquity") is not None:
        kpi["ROE"] = info["returnOnEquity"] * 100
    if info.get("debtToEquity") is not None:
        kpi["Debt to equity ratio"] = info["debtToEquity"] / 100
    eps = info.get("trailingEps")
    if not eps and out.get("financials") is not None and info.get("sharesOutstanding"):
        pat = out["financials"]["PAT"].iloc[-1]                   # newest full year
        if pat and pat > 0:
            eps = float(pat) / float(info["sharesOutstanding"])
    if eps:
        kpi["Earning Per Share (EPS)"] = eps
        if issue_price and eps > 0:
            kpi["Price/Earning P/E Ratio"] = issue_price / eps    # P/E at the ISSUE price
    if kpi:
        out["kpi"] = kpi
    out["market_cap"] = info.get("marketCap")
    return out


def merge_details(*details) -> dict:
    """Combine detail dicts, earlier ones winning (IPOWatch first when present)."""
    merged, sources = {}, []
    for d in details:
        if not d:
            continue
        src = d.get("source")
        if src and src not in sources:
            sources.append(src)
        for k, v in d.items():
            if k == "source":
                continue
            have = merged.get(k)
            empty = have is None or (hasattr(have, "empty") and have.empty) or have == {} or have == []
            if empty:
                merged[k] = v
            elif isinstance(have, dict) and isinstance(v, dict):
                merged[k] = {**v, **have}
    merged["sources"] = sources
    return merged
