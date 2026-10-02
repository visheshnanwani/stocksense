"""
ipo_page.py
-------------
The dashboard's "IPO Center" page (Indian mainboard + SME IPOs): live GMP,
NSE subscription, a searchable IPO analyser with an apply / avoid verdict,
recent listing performance, and how reliable GMP has been historically.

All data logic is in ipo_analyzer.py; this file is only the Streamlit UI.
"""

import html as _html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import gmp_log
import ipo_analyzer as ia
import ui_theme as ui

TEMPLATE = ui.PLOTLY_TEMPLATE


# ----------------------------- cached data -----------------------------
@st.cache_data(ttl=30 * 60, show_spinner=False)
def _gmp_tables():
    return ia.fetch_gmp_tables()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _listing_history():
    """Listing performance. Its own page since IPOWatch split it off the GMP page."""
    return ia.fetch_listing_history()


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _nse_lists():
    return ia.fetch_nse_lists()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _ipo_detail(url):
    return ia.fetch_ipo_detail(url)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _nse_detail(symbol, series):
    return ia.fetch_nse_detail(symbol, series)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _current_prices(symbols):
    """Latest close for NSE symbols (tries SYMBOL.NS, then SYMBOL-SM.NS for SME)."""
    import yfinance as yf
    out = {}
    for sym in symbols:
        for cand in (f"{sym}.NS", f"{sym}-SM.NS"):
            try:
                h = yf.Ticker(cand).history(period="5d", auto_adjust=False)
                if h is not None and len(h):
                    out[sym] = float(h["Close"].iloc[-1])
                    break
            except Exception:
                continue
    return out


def _fmt_x(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.2f}x"


@st.cache_data(show_spinner=False, ttl=120)
def _listed_quote(symbol):  # audit-exempt: timeframe-free (a quote is one instant, not a bar)
    # The IPO page is current-mode only by design: it answers "what is happening with
    # this listing right now". It is never driven by a historical as-of date, so these
    # fetches are live snapshots rather than analysable history.
    """Live price + listing-day open for a freshly listed NSE symbol."""
    import yfinance as yf
    for suffix in (".NS", ".BO"):
        try:
            df = yf.download(symbol + suffix, period="1mo", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df):
                return {"price": float(df["Close"].iloc[-1]), "open": float(df["Open"].iloc[0]),
                        "sessions": int(len(df)), "symbol": symbol + suffix}
        except Exception:
            continue
    return None


IPO_METRIC_LABELS = {
    "gmp_gain_pct": "Grey-market premium vs issue price",
    "subscription_total": "Overall subscription",
    "subscription_qib": "Institutional (QIB) subscription",
    "subscription_retail": "Retail subscription",
    "subscription_nii": "HNI / NII subscription",
    "revenue_cagr": "Revenue growth (CAGR)", "pat_cagr": "Profit growth (CAGR)",
    "pat_margin": "Net profit margin", "roe": "Return on equity (ROE)",
    "roce": "Return on capital (ROCE)", "pe": "P/E at the issue price",
    "pe_vs_peers": "P/E vs listed peers", "peer_median_pe": "Peer median P/E",
    "fresh_share": "Fresh capital share of the issue", "promoter_post_pct": "Promoter holding after IPO",
    "debt_to_equity": "Debt / equity", "issue_size_cr": "Issue size",
}

IPO_PILLAR_NOTES = {
    "Listing demand": "What decides the listing pop: grey-market premium and, once bidding closes, how many "
                      "times the issue was subscribed - institutional (QIB) money counts most.",
    "Growth": "Revenue and profit growth from the last 3-4 financial years in the prospectus (or, for an "
              "already-listed company, its published statements).",
    "Profitability": "How much of each rupee of sales becomes profit, and what the business earns on the "
                     "capital it uses.",
    "Valuation": "What you pay for those earnings at the issue price, against the company's own profits and "
                 "its listed peers. A high P/E scores low.",
    "Issue quality": "How much of the money goes INTO the company rather than to selling shareholders, how "
                     "much promoters keep, and how much debt the company carries.",
}


def _fmt_ipo_metric(key, value):
    if value is None:
        return "n/a"
    if key.startswith("subscription"):
        return f"{value:,.1f}x"
    if key in ("pe", "peer_median_pe", "pe_vs_peers", "debt_to_equity"):
        return f"{value:,.2f}x"
    if key == "issue_size_cr":
        return f"₹{value:,.0f} Cr"
    if key in ("promoter_post_pct",):
        return f"{value:,.1f}%"
    if key in ("gmp_gain_pct",):
        return f"{value:+.1f}%"
    return f"{value * 100:,.1f}%"       # ratios: margins, CAGR, ROE, fresh share


def _ipo_pillar_why(a):
    """{pillar: ([(metric, value, score)], note)} for the clickable rings."""
    out = {}
    for pillar, items in (a.get("sub_scores") or {}).items():
        rows = []
        for k, sc in items.items():
            v = a["metrics"].get(k)
            if v is None and sc is None:
                continue
            rows.append((IPO_METRIC_LABELS.get(k, k.replace("_", " ").capitalize()),
                         _fmt_ipo_metric(k, v), sc))
        if not rows:
            continue
        note = IPO_PILLAR_NOTES.get(pillar, "")
        if all(r[2] is None for r in rows):
            note += " Nothing here could be scored - see the sources note under the key facts."
        out[pillar] = (rows, note)
    return out


def _nse_row(name, nse):
    """Find this IPO in NSE's current / upcoming / past lists."""
    for key in ("current", "upcoming", "past"):
        df = nse.get(key)
        if df is None or df.empty:
            continue
        names_s = ia.nse_names(df)
        match = ia.match_name(name, names_s.dropna().tolist())
        if match:
            r = df[names_s == match].iloc[0].to_dict()
            r["_list"] = key
            return r
    return None


# ----------------------------- page -----------------------------
def _cols(df, wanted, label=""):
    """
    Select `wanted` from `df`, keeping only what is really there.

    These tables are scraped from a third party that changes its schema without
    notice. Losing the "GMP updated" column should cost the user that column, not
    the whole page, so the missing names are returned for a caption rather than
    raised. Returns (frame, missing_names).
    """
    have = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    return df[have].copy(), missing


def render(on_progress=None):
    """
    on_progress(pct, label, done=False) -- optional. The host page owns the top
    loading bar; without this the bar froze at 6% here, because this page runs
    to completion and then calls st.stop(), so nothing downstream ever advanced it.
    """
    def _p(pct, label, done=False):
        if on_progress:
            try:
                on_progress(pct, label, done)
            except Exception:
                pass

    _p(12, "Loading the IPO board")
    ui.render("""
    <div class="hero"><div class="hero-left">
    <div class="hero-eyebrow"><span class="tick">IPO</span><span class="chip">NSE · BSE</span><span class="chip">Mainboard & SME</span>
    <span class="chip">GMP</span><span class="chip">Live subscription</span></div>
    <h1 class="hero-title">IPO Center</h1>
    <div class="hero-sub">Live grey market premium · subscription by investor type · company financials · apply-or-avoid analysis</div>
    </div></div>
    """)

    with st.spinner("Loading live IPO data from NSE and IPOWatch..."):
        try:
            gmp = _gmp_tables()
        except Exception as e:
            gmp = {"mainboard": pd.DataFrame(), "sme": pd.DataFrame(), "history": pd.DataFrame()}
            st.warning(f"GMP data unavailable right now ({e}).")
        try:
            nse = _nse_lists()
        except Exception as e:
            nse = {"current": pd.DataFrame(), "upcoming": pd.DataFrame(), "past": pd.DataFrame()}
            st.warning(f"NSE data unavailable right now ({e}).")

    _p(55, "Scoring GMP reliability")
    live = pd.concat([gmp["mainboard"], gmp["sme"]], ignore_index=True)
    try:
        hist = _listing_history()
    except Exception as e:
        hist = pd.DataFrame()
        st.warning(f"Listing history unavailable right now ({e}).")
    base = ia.listing_base_rates(hist) if not hist.empty else {}

    # Record what the grey market is quoting today. GMP is published only while
    # an IPO is open and is gone the moment it lists, so the only way to ever
    # answer "was GMP right?" is to keep the board as it goes past.
    try:
        gmp_log.record(live)
    except Exception:
        pass
    rel = gmp_log.reliability(hist, match=ia.match_name) if not hist.empty else {"matched": 0, "logged": 0}
    rel_ready = rel.get("matched", 0) >= 10
    _p(72, "Building the IPO board")

    open_now = live[live["status"].astype(str).str.lower().str.contains("open")] if not live.empty else live
    upcoming = live[live["status"].astype(str).str.lower().str.contains("upcoming")] if not live.empty else live
    ui.kpi_grid([
        ui.kpi_html("Open now", f"{len(open_now)}", f"{len(nse.get('current', []))} on NSE right now"),
        ui.kpi_html("Upcoming", f"{len(upcoming)}", "mainboard + SME"),
        ui.kpi_html("GMP direction accuracy", f"{rel['direction_hit_rate'] * 100:.0f}%" if rel_ready else "building",
                    f"over {rel['n']} past IPOs" if rel_ready else f"{rel.get('logged', 0)} logged, needs 10 listed"),
        ui.kpi_html("Listed above issue price", f"{base['pct_listed_up'] * 100:.0f}%" if base else "n/a",
                    f"of {base['n']} IPOs since 2022" if base else None),
        ui.kpi_html("Avg listing gain", f"{base['avg_actual_gain']:+.1f}%" if base else "n/a",
                    f"median {base['median_gain']:+.1f}% — a few big pops pull the average up" if base else None,
                    "up" if base and base["avg_actual_gain"] >= 0 else "down"),
        ui.kpi_html("Data sources", "NSE · IPOWatch", "GMP is unofficial"),
    ])

    # ---- search ----
    ui.section("01", "Search & analyse an IPO", "live, upcoming or recently listed")
    names = list(dict.fromkeys(list(live["name"]) + (list(hist["name"]) if not hist.empty else [])))
    for key in ("current", "upcoming"):
        df = nse.get(key)
        if df is not None and not df.empty:
            for n in ia.nse_names(df).dropna():
                if not ia.match_name(n, names):
                    names.append(n)
    # Just-listed IPOs (e.g. one that listed this morning) are already off the
    # GMP boards, so pull the recent ones out of NSE's past-issues list too.
    past = nse.get("past")
    if past is not None and not past.empty:
        recent = past.copy()
        recent["_listed"] = pd.to_datetime(recent.get("listingDate"), errors="coerce", dayfirst=True)
        recent = recent[recent["_listed"] >= pd.Timestamp.today() - pd.Timedelta(days=120)]
        for n in ia.nse_names(recent).dropna():
            if not ia.match_name(n, names):
                names.append(n)
    choice = st.selectbox("Search IPO", names, index=None, placeholder="🔎 Type an IPO name… e.g. Moneyview, Varmora",
                          label_visibility="collapsed", key="ipo_search")
    if choice:
        _render_analysis(choice, live, hist, nse, rel)
    else:
        st.caption("Pick an IPO above to get its full analysis: GMP, subscription, financials, valuation vs peers, "
                   "promoter holding and an apply / avoid verdict.")

    _p(100, "Done", done=True)

    # ---- lists ----
    ui.section("02", "IPO market", "live GMP board · recent listings · how reliable GMP is")
    t_live, t_recent, t_rel = st.tabs(["🟢 Live & upcoming", "📊 Recent listings", "🧪 How reliable is GMP?"])

    with t_live:
        if live.empty:
            st.info("No live GMP data available right now.")
        else:
            board, _missing = _cols(live, ["name", "type", "status", "dates", "price", "gmp",
                                           "est_listing", "est_gain_pct", "updated"], "live GMP board")
            cur = nse.get("current")
            if cur is not None and not cur.empty:
                subs = {}
                for _, r in cur.iterrows():
                    mt = ia.match_name(r["companyName"], board["name"].tolist())
                    if mt:
                        subs[mt] = r.get("subscription_x")
                board["subscription"] = board["name"].map(subs)
            board = board.rename(columns={"name": "IPO", "type": "Type", "status": "Status", "dates": "Dates",
                                          "price": "Price (₹)", "gmp": "GMP (₹)", "est_listing": "Est. listing (₹)",
                                          "est_gain_pct": "Est. gain", "updated": "GMP updated", "subscription": "Subscribed"})
            _cfg = {"Price (₹)": st.column_config.NumberColumn(format="₹%.0f"),
                    "GMP (₹)": st.column_config.NumberColumn(format="₹%.0f"),
                    "Est. listing (₹)": st.column_config.NumberColumn(format="₹%.0f"),
                    "Subscribed": st.column_config.NumberColumn(format="%.2fx")}
            if "Est. gain" in board.columns:
                _mx = pd.to_numeric(board["Est. gain"], errors="coerce").max()
                _cfg["Est. gain"] = st.column_config.ProgressColumn(
                    "Est. listing gain", format="%.1f%%", min_value=-10,
                    max_value=float(max(60.0, _mx if pd.notna(_mx) else 0.0)))
            st.dataframe(board, width="stretch", hide_index=True,
                         height=min(640, 40 + 35 * len(board)),
                         column_config={k: v for k, v in _cfg.items() if k in board.columns})
            if _missing:
                st.caption(f"Not shown: {', '.join(_missing)} — the source stopped publishing "
                           f"{'it' if len(_missing) == 1 else 'them'}. Everything else is live.")
            st.caption("GMP = grey market premium: what unofficial traders pay above the issue price before listing. "
                       "It's unregulated and can swing daily. Subscription is live from NSE for open issues.")

    with t_recent:
        if hist.empty:
            st.info("Listing history could not be loaded from IPOWatch just now. "
                    "Everything else on this page is unaffected.")
        else:
            st.caption(f"{len(hist)} IPOs tracked from issue price to listing day. "
                       "Rows where the source's own prices and its printed gain contradict "
                       "each other are dropped rather than guessed at.")
            n_show = st.slider("How many recent listings to show", 10, 60, 25, 5, key="ipo_recent_n")
            recent = hist.head(n_show).copy()

            past = nse.get("past")
            syms = {}
            if past is not None and not past.empty:
                past_names = ia.nse_names(past)
                name_list = past_names.dropna().tolist()
                for n in recent["name"]:
                    mt = ia.match_name(n, name_list)
                    if mt:
                        syms[n] = past.loc[past_names == mt, "symbol"].iloc[0]
            with st.spinner("Fetching current prices of recent listings..."):
                prices = _current_prices(tuple(sorted(set(syms.values()))))
            recent["current_price"] = recent["name"].map(lambda n: prices.get(syms.get(n)))
            recent["return_since_ipo"] = (recent["current_price"] / recent["ipo_price"] - 1) * 100

            fig = go.Figure()
            fig.add_trace(go.Bar(y=recent["name"], x=recent["actual_gain_pct"], name="Listing-day gain",
                                 orientation="h", marker_color=ui.SERIES[0], marker_line_width=0))
            if recent["return_since_ipo"].notna().any():
                fig.add_trace(go.Bar(y=recent["name"], x=recent["return_since_ipo"],
                                     name="Return since IPO (today)", orientation="h",
                                     marker_color=ui.SERIES[1], marker_line_width=0))
            fig.update_layout(height=max(420, 26 * len(recent)), template=TEMPLATE, barmode="group",
                              hovermode="closest", title="Recent IPOs: listing gain vs return since IPO",
                              margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(ticksuffix="%"),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")

            show, _missing_r = _cols(recent, ["name", "ipo_price", "listing_price", "actual_gain_pct",
                                              "current_price", "return_since_ipo"], "recent listings")
            show = show.rename(columns={
                "name": "IPO", "ipo_price": "Issue price", "listing_price": "Listing price",
                "actual_gain_pct": "Actual listing gain", "current_price": "Price now",
                "return_since_ipo": "Return since IPO"})
            st.dataframe(show, width="stretch", hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%+.1f%%")
                for c in ("Actual listing gain", "Return since IPO") if c in show.columns})
            st.caption("'Price now' is blank where the IPO could not be matched to an NSE symbol "
                       "(most SME issues); the listing gain beside it is still real.")

    with t_rel:
        if base:
            st.markdown("##### What the listing record itself says")
            ui.kpi_grid([
                ui.kpi_html("Listed above issue price", f"{base['pct_listed_up'] * 100:.0f}%",
                            f"{base['n']} IPOs since 2022"),
                ui.kpi_html("Average listing gain", f"{base['avg_actual_gain']:+.1f}%",
                            "pulled up by a few big pops",
                            "up" if base["avg_actual_gain"] >= 0 else "down"),
                ui.kpi_html("Median listing gain", f"{base['median_gain']:+.1f}%",
                            "the typical IPO, not the average one"),
                ui.kpi_html("Gained 10%+", f"{base['pct_up_10'] * 100:.0f}%", "on listing day"),
                ui.kpi_html("Fell 10%+", f"{base['pct_down_10'] * 100:.0f}%", "on listing day", "down"),
                ui.kpi_html("Range", f"{base['worst']:+.0f}% to {base['best']:+.0f}%",
                            f"{base['worst_name']} / {base['best_name']}"),
            ])
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=hist["actual_gain_pct"], nbinsx=50,
                                       marker_color=ui.SERIES[0], marker_line_width=0,
                                       name="Listing gain"))
            fig.add_vline(x=0, line=dict(color=ui.NEUTRAL, dash="dot", width=1))
            fig.add_vline(x=base["median_gain"], line=dict(color=ui.SERIES[1], width=2),
                          annotation_text=f"median {base['median_gain']:+.1f}%")
            fig.update_layout(height=380, template=TEMPLATE, margin=dict(l=10, r=10, t=40, b=10),
                              title=f"Listing-day gain, all {base['n']} tracked IPOs",
                              xaxis=dict(title="Listing-day gain", ticksuffix="%"),
                              yaxis=dict(title="IPOs"), showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("Most IPOs list up, and the average is flattered by a long right tail: the median "
                       f"({base['median_gain']:+.1f}%) is the number to plan around, not the mean "
                       f"({base['avg_actual_gain']:+.1f}%).")

        st.markdown("##### Was the grey market right?")
        if rel_ready:
            ui.kpi_grid([
                ui.kpi_html("Direction correct", f"{rel['direction_hit_rate'] * 100:.0f}%", f"{rel['n']} IPOs"),
                ui.kpi_html("Listed up when GMP > 0",
                            f"{rel['listed_up_when_gmp_positive'] * 100:.0f}%"
                            if rel.get("listed_up_when_gmp_positive") is not None else "n/a"),
                ui.kpi_html("Listed up when GMP = 0",
                            f"{rel['listed_up_when_gmp_zero'] * 100:.0f}%"
                            if rel.get("listed_up_when_gmp_zero") is not None else "n/a"),
                ui.kpi_html("Typical miss", f"+/-{rel['median_abs_error_pts']:.1f} pts",
                            "median absolute error"),
                ui.kpi_html("Average bias", f"{rel['mean_error_pts']:+.1f} pts", "actual minus GMP-implied",
                            "down" if rel["mean_error_pts"] < 0 else "up"),
                ui.kpi_html("Correlation", f"{rel['correlation']:.2f}", "1.0 = perfect"),
            ])
            m = rel["frame"]
            lim = [min(m["predicted_gain_pct"].min(), m["actual_gain_pct"].min()) - 5,
                   max(m["predicted_gain_pct"].max(), m["actual_gain_pct"].max()) + 5]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                                     line=dict(color=ui.NEUTRAL, dash="dot", width=1)))
            fig.add_trace(go.Scatter(x=m["predicted_gain_pct"], y=m["actual_gain_pct"], mode="markers",
                                     name="Past IPOs", text=m["name"],
                                     marker=dict(size=9, color=ui.SERIES[0], opacity=0.75,
                                                 line=dict(width=1, color="#0b0e14")),
                                     hovertemplate="%{text}<br>GMP implied %{x:.1f}%"
                                                   "<br>Actual %{y:.1f}%<extra></extra>"))
            fig.update_layout(height=460, template=TEMPLATE, hovermode="closest",
                              margin=dict(l=10, r=10, t=40, b=10),
                              title="GMP-implied vs actual listing gain",
                              xaxis=dict(title="GMP-implied gain", ticksuffix="%"),
                              yaxis=dict(title="Actual listing gain", ticksuffix="%"))
            st.plotly_chart(fig, width="stretch")
            st.caption("Points above the dotted line listed better than GMP suggested, points below did "
                       "worse. GMP says something about listing day and nothing about long-term returns.")
        else:
            logged, matched = rel.get("logged", 0), rel.get("matched", 0)
            plural = "s" if logged != 1 else ""
            st.markdown(
                "This one is **being measured from scratch, and is honest about it.**\n\n"
                "Answering it needs two numbers per IPO: the premium quoted *while it was still "
                "open*, and the price it *actually listed at*. The second is published permanently "
                f"-- that is the {len(hist)} listings on the previous tab. The first is not: the "
                "grey-market board drops an IPO the moment it lists, and no source keeps the history.\n\n"
                "IPOWatch used to publish both together in one table. They stopped, which is exactly "
                "why this panel went blank. Reconstructing an old premium from the listing gain would "
                "only measure the reconstruction, so instead the live board is now saved every time "
                "this page loads.\n\n"
                f"**Logged so far: {logged} IPO{plural} - {matched} of them already listed - needs 10.** "
                "Open this page while IPOs are live and it fills in on its own.")
            if logged:
                fin = gmp_log.final_gmp()
                fin["last_seen"] = pd.to_datetime(fin["last_seen"]).dt.strftime("%d %b %Y")
                st.dataframe(fin.rename(columns={"name": "IPO", "gmp": "Last GMP", "price": "Issue price",
                                                 "last_seen": "Last seen", "days_logged": "Days logged"}),
                             width="stretch", hide_index=True)


def _render_analysis(choice, live, hist, nse, rel):
    live_row = live[live["name"] == choice]
    hist_row = hist[hist["name"] == choice] if not hist.empty else hist
    if len(live_row):
        row = live_row.iloc[0].to_dict()
    elif len(hist_row):
        h = hist_row.iloc[0]
        row = {"name": choice, "gmp": h["gmp"], "price": h["ipo_price"], "est_gain_pct": h["predicted_gain_pct"],
               "status": "Listed", "type": "Mainboard", "link": h.get("link"), "listing_price": h["listing_price"]}
    else:
        row = {"name": choice, "gmp": None, "price": None, "est_gain_pct": None, "status": "", "type": "", "link": None}
    nse_r = _nse_row(choice, nse)
    nse_d, final = {}, True
    if nse_r and nse_r.get("symbol"):
        series = "SME" if str(nse_r.get("series") or nse_r.get("securityType") or "").upper() == "SME" else "EQ"
        if not row.get("type"):
            row["type"] = "SME" if series == "SME" else "Mainboard"
        with st.spinner("Fetching subscription data from NSE..."):
            try:
                nse_d = _nse_detail(nse_r["symbol"], series)
            except Exception:
                nse_d = {}
        final = ia.is_subscription_final(nse_r.get("issueEndDate") or nse_r.get("ipoEndDate"))
    det = {}
    if row.get("link"):
        with st.spinner("Reading the company's financials, valuation and issue details..."):
            try:
                det = _ipo_detail(row["link"])
            except Exception as e:
                st.info(f"Detailed page unavailable ({e}).")
    # IPOWatch is the only GMP source, but it is also where issue size, lot
    # size, financials and promoter holding came from -- so for an IPO it never
    # covered, fill what we can from NSE's own issue info and (once listed)
    # Yahoo's financial statements.
    issue_price = ia.to_num((nse_r or {}).get("issuePrice")) or row.get("price")
    det_nse = ia.parse_nse_issue_info(nse_d.get("info", {}), (nse_r or {}).get("listingDate")) \
        if nse_d.get("info") else {}
    det_yahoo = {}
    if nse_r and nse_r.get("symbol") and nse_r.get("_list") == "past":
        with st.spinner("Reading the listed company's financials from Yahoo Finance..."):
            try:
                det_yahoo = ia.detail_from_yahoo(nse_r["symbol"], issue_price)
            except Exception:
                det_yahoo = {}
    det = ia.merge_details(det, det_nse, det_yahoo)   # IPOWatch wins where it has data

    if row.get("price") is None:
        row["price"] = issue_price or (ia.to_num(str(det["price_band"]).split("to")[-1])
                                       if det.get("price_band") else None)
    a = ia.analyze_ipo(row, det, nse_d, rel, subscription_final=final)
    M = a["metrics"]

    # Already listed? Then the real numbers beat any estimate -- show them first.
    if nse_r and nse_r.get("_list") == "past" and nse_r.get("symbol"):
        issue = ia.to_num(nse_r.get("issuePrice"))
        quote = _listed_quote(nse_r["symbol"])
        listed_on = str(nse_r.get("listingDate") or "")
        cards = [ui.kpi_html("Issue price", f"₹{issue:,.2f}" if issue else "n/a", "what you paid if allotted"),
                 ui.kpi_html("Listed on", listed_on or "n/a", "first trading day")]
        if quote:
            gain_open = (quote["open"] / issue - 1) * 100 if issue and quote.get("open") else None
            gain_now = (quote["price"] / issue - 1) * 100 if issue else None
            cards += [
                ui.kpi_html("Listing-day open", f"₹{quote['open']:,.2f}" if quote.get("open") else "n/a",
                            f"{gain_open:+.1f}% vs issue" if gain_open is not None else None,
                            "up" if (gain_open or 0) >= 0 else "down"),
                ui.kpi_html("Price now", f"₹{quote['price']:,.2f}",
                            f"{gain_now:+.1f}% vs issue" if gain_now is not None else None,
                            "up" if (gain_now or 0) >= 0 else "down"),
                ui.kpi_html("Sessions traded", f"{quote['sessions']}",
                            "needs ~300 for the forecasting models", "", True),
            ]
        else:
            cards.append(ui.kpi_html("Price now", "n/a", "Yahoo has no data for this symbol yet", "", True))
        ui.section("", f"{choice} has already listed", "actual performance, not an estimate")
        ui.kpi_grid(cards)
        if quote and quote["sessions"] < 300:
            st.caption("The 📈 Stock analysis page needs about 300 trading sessions (~14 months) before it can "
                       "train forecasts on a stock, so this one is covered here for now.")

    # ---- verdict ----
    tone_color = {"up": ui.UP, "down": ui.DOWN, "warn": ui.WARN, "info": ui.ACCENT}.get(a["tone"], ui.NEUTRAL)
    ui.render(f"""
    <div class="verdict">
    <div style="display:flex;gap:18px;flex-wrap:wrap">
      <details class="ring-card why" style="text-align:center;background:transparent;border:0">
        <summary>{ui.ring_html(a['listing_score'], big=True)}
        <div class="ring-label" style="margin-top:6px">Listing-gain view</div>
        <div class="why-hint"></div></summary>
        {ui.why_html([("Listing demand pillar", "see ring below", a["pillars"].get("Listing demand"))],
                     "Will it pop on listing day? Built from grey-market premium and final subscription "
                     "(institutional demand weighted highest). It says nothing about the business itself.")}
      </details>
      <details class="ring-card why" style="text-align:center;background:transparent;border:0">
        <summary>{ui.ring_html(a['long_term_score'], big=True)}
        <div class="ring-label" style="margin-top:6px">Long-term view</div>
        <div class="why-hint"></div></summary>
        {ui.why_html([(p, "see ring below", a["pillars"].get(p))
                      for p in ("Growth", "Profitability", "Valuation", "Issue quality")],
                     "Is it a business worth owning after listing? The average of these four pillars, so a "
                     "cheap price with weak growth and an expensive price with strong growth can score alike.")}
      </details>
    </div>
    <div style="flex:1;min-width:260px"><div class="verdict-kicker">🚀 IPO verdict · {_html.escape(choice)} · {row.get('type') or ''}</div>
    <div class="verdict-title" style="color:{tone_color}">{a['recommendation']}</div>
    <div class="verdict-stance">{a['why']}</div>
    <div class="verdict-text">Listing-gain view scores grey-market premium and (once bidding closes) subscription, especially
    institutional (QIB) demand. Long-term view scores revenue/profit growth, margins, ROE/ROCE, valuation vs listed peers,
    fresh-issue share, promoter holding and debt.</div></div></div>
    """)
    ui.rings({k: v for k, v in a["pillars"].items()}, _ipo_pillar_why(a))
    st.caption("💡 Click any ring to see exactly which numbers produced that score.")
    ui.bull_bear([{"pillar": "IPO", "text": t} for t in a["strengths"]], [{"pillar": "IPO", "text": t} for t in a["risks"]])

    # ---- key facts ----
    info = nse_d.get("info", {})
    lot = ia.to_num(info.get("Bid Lot") or info.get("Minimum Order Quantity"))
    if lot is None and det.get("lots") is not None and len(det["lots"]):
        lot = ia.to_num(det["lots"].iloc[0].get("Shares"))
    if lot is None:
        lot = det.get("lot_size")          # parsed out of NSE's "35 Equity Shares and in multiples thereof"
    price = a.get("price")
    dates = det.get("dates", {}) or {}
    gmp_txt = "n/a" if row.get("gmp") is None else (f"₹{row['gmp']:,.0f} → est. listing ₹{price + row['gmp']:,.0f} ({M['gmp_gain_pct']:+.1f}%)"
                                                     if price else f"₹{row['gmp']:,.0f}")
    ui.facts([
        ("Price band", det.get("price_band") or info.get("Price Range") or (f"₹{price:,.0f}" if price else "n/a")),
        ("Lot size / min. investment", f"{int(lot)} shares · ₹{lot * price:,.0f}" if lot and price else "n/a"),
        ("Issue size", f"₹{M['issue_size_cr']:,.2f} Cr" if M.get("issue_size_cr") else "n/a"),
        ("Fresh issue share", f"{M['fresh_share'] * 100:.0f}% fresh · {100 - M['fresh_share'] * 100:.0f}% offer-for-sale" if M.get("fresh_share") is not None else "n/a"),
        ("GMP", gmp_txt),
        ("Bidding", " → ".join(x for x in [dates.get("IPO Open Date"), dates.get("IPO Close Date")] if x) or info.get("Issue Period", "n/a")),
        ("Allotment / listing", " · ".join(x for x in [dates.get("Basis of Allotment"), dates.get("IPO Listing Date")] if x) or "n/a"),
        ("Promoter holding", f"{det.get('promoter_pre_pct', 0) or 0:.1f}% → {M['promoter_post_pct']:.1f}% after IPO" if M.get("promoter_post_pct") is not None else "n/a"),
    ])
    if info:
        ui.facts([("Lead managers", info.get("Book Running Lead Managers", "n/a")), ("Registrar", info.get("Name of the Registrar", "n/a")),
                  ("Face value", info.get("Face Value", "n/a")), ("Issue type", info.get("Issue Type", "n/a"))])
    missing = []
    if row.get("gmp") is None:
        missing.append("**GMP** — this issue is not on IPOWatch's grey-market boards (the only free GMP "
                       "source), so there is no premium to report")
    if M.get("promoter_post_pct") is None:
        missing.append("**promoter holding** — published in the RHP/prospectus, which neither NSE's API nor "
                       "Yahoo expose")
    if det.get("peers") is None:
        missing.append("**peer valuation** — the peer table comes from the same IPOWatch page")
    srcs = ", ".join(det.get("sources") or []) or "NSE"
    st.caption("Sources for this IPO: " + srcs + " (subscription from NSE; financials from Yahoo once listed)."
               + (" Not available: " + "; ".join(missing) + "." if missing else ""))

    if row.get("listing_price"):
        st.success(f"Listed at ₹{row['listing_price']:,.2f} ({(row['listing_price'] / price - 1) * 100:+.1f}% vs issue price; "
                   f"GMP had implied {M['gmp_gain_pct']:+.1f}%).")

    c1, c2 = st.columns(2)
    with c1:
        subs = nse_d.get("subscription", {})
        if subs:
            cats = [c for c in ("QIB", "NII", "Retail", "Employees", "Shareholders", "Total") if c in subs]
            fig = go.Figure(go.Bar(x=[subs[c] for c in cats], y=cats, orientation="h", marker_line_width=0,
                                   marker_color=[ui.SERIES[i] for i in range(len(cats))],
                                   text=[f"{subs[c]:.2f}x" for c in cats], textposition="outside"))
            fig.update_layout(height=300, template=TEMPLATE, hovermode="closest", margin=dict(l=10, r=40, t=40, b=10),
                              title="Subscription by investor category" + ("" if final else " (bidding in progress)"),
                              yaxis=dict(autorange="reversed"), xaxis=dict(ticksuffix="x"))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Subscription data isn't available for this IPO (not open yet, or no longer on NSE's live feed).")
    with c2:
        fin = det.get("financials")
        if fin is not None and len(fin):
            fig = go.Figure()
            for i, col in enumerate(["Revenue", "PAT", "Assets"]):
                if col in fin.columns:
                    fig.add_trace(go.Bar(x=fin["Period"].astype(str), y=fin[col], name=col, marker_color=ui.SERIES[i], marker_line_width=0))
            fig.update_layout(height=300, template=TEMPLATE, barmode="group", margin=dict(l=10, r=10, t=40, b=10),
                              title="Company financials (₹ Crore)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Company financials aren't available for this IPO yet.")

    k1, k2 = st.columns(2)
    with k1:
        kpi = det.get("kpi")
        if kpi:
            st.markdown("**Key ratios (from the offer document)**")
            st.dataframe(pd.DataFrame([{"KPI": k, "Value": v} for k, v in kpi.items()]), width="stretch", hide_index=True)
        if M.get("pe") and M.get("peer_median_pe"):
            st.caption(f"IPO P/E {M['pe']:.1f}x vs listed peers' median {M['peer_median_pe']:.1f}x.")
    with k2:
        peers = det.get("peers")
        if peers is not None and len(peers):
            st.markdown("**Listed peer comparison**")
            st.dataframe(peers, width="stretch", hide_index=True)
        obj = det.get("objects")
        if obj is not None and len(obj):
            st.markdown("**What the money will be used for**")
            st.dataframe(obj, width="stretch", hide_index=True)
    src = [f"[IPOWatch page]({row['link']})" if row.get("link") else None,
           "[NSE issue page](https://www.nseindia.com/market-data/all-upcoming-issues-ipo)" if nse_r else None]
    st.caption("Sources: " + " · ".join(s for s in src if s) + ". GMP is unofficial and changes daily; always read the "
               "offer document (RHP). This analysis is rule-based and not investment advice.")
