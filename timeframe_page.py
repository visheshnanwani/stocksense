"""
timeframe_page.py
-------------------
The analysis page for any candle size other than daily.

When the user picks 5-minute, 15-minute, hourly, 4-hour or weekly candles, the
WHOLE pipeline runs on those candles: the OHLCV comes from the provider at that
interval (2h/4h resampled from hourly, which is stated), the indicators, support
and resistance, candlestick patterns and volume profile are computed on those
bars, the models are trained on those bars with horizons measured in those
candles, and the chart, forecast and confidence band are drawn on them.

Nothing is relabelled. A 15-minute RSI here is an RSI of 15-minute closes.

What is deliberately NOT carried over to an intraday timeframe: daily news tone,
macro series, earnings dates, the daily social reading, the universe model and
the fundamental research report. Each of those describes a whole day, and
spreading one value across 25 intraday candles would repeat a single
observation 25 times as if it were new evidence. The page says so rather than
quietly doing it.
"""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import candlestick_patterns as cp
import data_loader as dl
import forecast_engine as fe
import scenarios as sx
import technical_analysis as ta
import timeframes as tfm
import ui_theme as ui

Z80 = 1.2816


# ------------------------------------------------------------------ caching
@st.cache_data(show_spinner=False, ttl=120)
def load_bars(ticker: str, tf_code: str, stamp: str):
    """OHLCV at the chosen candle size. `stamp` busts the cache every 2 minutes
    for intraday so the latest candle is current."""
    return dl.load_ohlcv(ticker, tf_code)


@st.cache_resource(show_spinner=False)
def train_engine(ticker: str, tf_code: str, bars: int, last_stamp: str, horizons: tuple,
                 ann_factor: float, _raw):
    """One engine per (symbol, timeframe) -- never share a 5-minute model with a
    daily one. The cache key carries the timeframe and the last candle."""
    return fe.ForecastEngine(_raw, ticker, horizons=list(horizons), timeframe=tf_code,
                             bars_per_year=ann_factor).fit()


@st.cache_data(show_spinner=False)
def candle_scan(ticker: str, tf_code: str, bars: int, forward: int, _raw):
    return cp.pattern_events(_raw, forward), cp.pattern_reliability(_raw, forward)


# ------------------------------------------------------------------ helpers
def _session_breaks(raw: pd.DataFrame, intraday: bool):
    """Hide the overnight gap so intraday candles sit side by side."""
    if not intraday or len(raw) < 10:
        return [dict(bounds=["sat", "mon"])]
    times = sorted({t for t in raw.index.time})
    if not times:
        return [dict(bounds=["sat", "mon"])]
    open_h = times[0].hour + times[0].minute / 60
    close_h = times[-1].hour + times[-1].minute / 60 + 0.25
    breaks = [dict(bounds=["sat", "mon"])]
    if close_h < 24:
        breaks.append(dict(bounds=[close_h, open_h], pattern="hour"))
    return breaks


def _fmt_stamp(ts, intraday: bool) -> str:
    return ts.strftime("%d %b %H:%M") if intraday else ts.strftime("%d %b %Y")


# ------------------------------------------------------------------ the page
def render(ticker: str, profile: dict, tf_code: str, horizon_label: str, horizon_days: float,
           cur: str, buy_threshold: float, sell_threshold: float):
    spec = tfm.get(tf_code)
    intraday = spec["intraday"]
    tz = profile.get("timezone")
    asset_class = profile.get("asset_class", "stock")

    # ---------------- data ----------------
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")[:-1] if intraday \
        else datetime.date.today().isoformat()
    try:
        with st.spinner(f"Fetching {spec['label']} candles for {ticker}…"):
            raw = load_bars(ticker, tf_code, stamp)
    except Exception as e:
        st.error(f"**{spec['label']} candles are not available for {ticker}.** {e}")
        st.info("Pick a different candle size in the sidebar — the menu only lists intervals this data "
                "source can serve, but individual symbols (especially newly listed or illiquid ones) can "
                "still be missing intraday history.")
        return

    bars = len(raw)
    per_session = tfm.candles_per_session(tf_code, tz, asset_class)
    horizon_candles = tfm.horizon_to_candles(tf_code, horizon_days, tz, asset_class)
    max_candles = max(int(bars * tfm.MAX_HORIZON_SHARE), 1)
    capped = horizon_candles > max_candles
    used_candles = min(horizon_candles, max_candles)

    last_close = float(raw["Close"].iloc[-1])
    prev_close = float(raw["Close"].iloc[-2]) if bars > 1 else last_close
    change_pct = (last_close / prev_close - 1) * 100

    # ---------------- header ----------------
    chips = [f"⏱️ {spec['label']} candles", profile.get("exchange"),
             f"{bars:,} candles", profile.get("currency")]
    if spec["resample"]:
        chips.append("resampled from 1h")
    ui.hero(name=profile.get("name", ticker), ticker=ticker.upper(), chips=[c for c in chips if c],
            price=last_close, change_abs=last_close - prev_close, change_pct=change_pct,
            market_open=False, as_of=_fmt_stamp(raw.index[-1], intraday), cur=cur)

    ui.kpi_grid([
        ui.kpi_html("Selected candle", spec["label"],
                    f"resampled from 1-hour bars" if spec["resample"] else "native from the provider",
                    "", True),
        ui.kpi_html("Candles loaded", f"{bars:,}",
                    f"{_fmt_stamp(raw.index[0], intraday)} → {_fmt_stamp(raw.index[-1], intraday)}",
                    "", True),
        ui.kpi_html("Candles per session", f"{per_session:g}",
                    f"{tfm.session_minutes(tz, asset_class)} min session" if intraday else "one per day"),
        ui.kpi_html("Forecast horizon", horizon_label,
                    f"= {horizon_candles:,} candles at this size", "", True),
        ui.kpi_html("Forecast candles used", f"{used_candles:,}",
                    "capped by available history" if capped else "as requested",
                    "down" if capped else ""),
        ui.kpi_html("Last candle change", f"{'▲' if change_pct >= 0 else '▼'} {change_pct:+.2f}%",
                    _fmt_stamp(raw.index[-1], intraday), "up" if change_pct >= 0 else "down"),
    ])

    if capped:
        st.warning(
            f"**{horizon_label} is {horizon_candles:,} candles at {spec['label']}, and only "
            f"{bars:,} candles of history exist.** A forecast that far out would rest on roughly "
            f"{bars / max(horizon_candles, 1):.0f} independent examples, which cannot be validated, so the "
            f"forecast is capped at {used_candles:,} candles "
            f"(≈ {tfm.candles_to_trading_days(tf_code, used_candles, tz, asset_class):.2f} trading days). "
            f"For a longer view, switch to a larger candle: the provider serves "
            f"{tfm.max_history_days('1h')} days of hourly and full history of daily candles.")

    st.caption(
        f"**Selected timeframe: {tf_code}** — every number on this page is computed from these "
        f"{spec['label'].lower()} candles: the indicators, support and resistance, candlestick reading, "
        f"the models and the forecast band."
        + (f" Two- and four-hour candles are built by resampling the provider's hourly bars within each "
           f"trading session, because its native 4-hour feed only reaches back 60 days."
           if spec["resample"] else ""))

    if intraday:
        st.info(
            "**Not included at this timeframe:** daily news tone, macro series, earnings dates, the daily "
            "social reading, the universe model and the fundamental research report. Each of those is one "
            "observation per day — repeating it across every intraday candle would look like new evidence "
            "when it is not. Switch to **1 day** candles to get them.")

    # ---------------- chart ----------------
    ui.section("01", f"{spec['label']} chart", "candles · volume · moving averages · support and resistance")
    show_n = st.slider("Candles to show", min_value=60, max_value=min(bars, 1500),
                       value=min(300, bars), step=20, key=f"tf_show_{tf_code}")
    view = raw.tail(show_n)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"],
                                 close=view["Close"], name=ticker,
                                 increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN), row=1, col=1)
    for window, colour in ((20, ui.SERIES[0]), (50, ui.SERIES[1])):
        if bars > window:
            fig.add_trace(go.Scatter(x=view.index, y=raw["Close"].rolling(window).mean().reindex(view.index),
                                     name=f"SMA {window}", line=dict(color=colour, width=1.4)), row=1, col=1)
    try:
        for lvl in ta.support_resistance_levels(raw)[:6]:
            fig.add_hline(y=lvl["level"], line_dash="dot", line_width=1,
                          line_color=ui.UP if lvl["kind"] == "support" else ui.DOWN,
                          annotation_text=f"{lvl['kind'][:3].title()} {cur}{lvl['level']:.2f}",
                          annotation_position="top left", annotation_font_size=9, row=1, col=1)
    except Exception:
        pass
    vol_colours = [ui.UP if c >= o else ui.DOWN for o, c in zip(view["Open"], view["Close"])]
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume", marker_color=vol_colours,
                         opacity=0.45, marker_line_width=0, showlegend=False), row=2, col=1)
    fig.update_xaxes(rangebreaks=_session_breaks(raw, intraday))
    fig.update_layout(height=560, template=ui.PLOTLY_TEMPLATE, margin=dict(l=10, r=60, t=40, b=10),
                      xaxis_rangeslider_visible=False,
                      title=f"{ticker} — {spec['label']} candles + {horizon_label} forecast")
    fig.update_yaxes(side="right", row=1, col=1)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # ---------------- technicals at this timeframe ----------------
    ui.section("02", f"{spec['label']} technicals", "every indicator computed on these candles")
    try:
        import features_indicators as fi
        ind = fi.indicator_features(raw)
        now_ind = ind.iloc[-1]
        tscore = ta.technical_score_series(raw)
        score = float(tscore["Tech_Score"].iloc[-1])
        verdict = ("Strong buy" if score > 0.5 else "Buy" if score > 0.15 else
                   "Strong sell" if score < -0.5 else "Sell" if score < -0.15 else "Neutral")
        tone = "up" if score > 0.15 else "down" if score < -0.15 else ""

        def _val(col, fmt, transform=lambda v: v, default="n/a"):
            v = now_ind.get(col)
            return default if v is None or pd.isna(v) else fmt.format(transform(float(v)))

        rsi_txt = _val("Mom_RSI14", "{:.1f}", lambda v: (v + 0.5) * 100)
        per_candle = f"{spec['minutes']} min" if intraday else spec["label"].lower()
        ui.kpi_grid([
            ui.kpi_html(f"{spec['label']} rating", verdict, f"score {score:+.2f} on −1…+1", tone),
            ui.kpi_html("RSI (14 candles)", rsi_txt, f"14 × {per_candle} of trading"),
            ui.kpi_html("MACD histogram", _val("Mom_MACD_Hist", "{:+.3f}%", lambda v: v * 100),
                        "12/26/9 on these candles"),
            ui.kpi_html("ADX (14)", _val("Trend_ADX14", "{:.1f}", lambda v: v * 100),
                        "above 25 = trending"),
            ui.kpi_html("ATR", _val("Vol_ATR_Pct", "{:.2f}%", lambda v: v * 100),
                        f"typical range per {per_candle} candle"),
            ui.kpi_html("Bollinger %B", _val("Vol_BB_PctB", "{:+.2f}", lambda v: v),
                        "+0.5 upper band · −0.5 lower"),
            ui.kpi_html("Price vs SMA 20", _val("Trend_Px_vs_SMA20", "{:+.2f}%", lambda v: v * 100),
                        f"{spec['label']} average"),
            ui.kpi_html("Stochastic %K", _val("Mom_Stoch_K", "{:.0f}", lambda v: (v + 0.5) * 100),
                        "over 80 overbought · under 20 oversold"),
        ])
        st.caption(
            f"All {len(ind.columns)} indicators are recomputed on {spec['label'].lower()} candles — "
            + (f"a 14-period RSI here covers {14 * (spec['minutes'] or 0)} minutes of trading, not 14 days, "
               f"and the ATR is the typical range of a single {per_candle} candle."
               if intraday else
               f"the same bank used on the daily page, applied to {spec['label'].lower()} bars."))
    except Exception as e:
        st.info(f"Technical scorecard unavailable at this timeframe ({e}).")

    # ---------------- candlestick reading ----------------
    try:
        fwd = max(int(round(per_session / 4)), 3)
        events, reliability = candle_scan(ticker, tf_code, bars, fwd, raw)
        reading = cp.read_candles(raw, lookback=5, reliability=reliability)
        c_tone = {"Bullish": "up", "Bearish": "down"}.get(reading["verdict"].split("/")[0].strip(), "")
        ui.card(f"🕯️ Candlestick reading on {spec['label']} candles",
                f"<p><b>{reading['verdict']}</b> — {len(reading['events'])} pattern(s) in the last 5 candles. "
                f"Reliability is measured on this symbol's own {spec['label'].lower()} history "
                f"({len(events):,} past occurrences, scored {fwd} candles forward).</p>")
    except Exception as e:
        st.info(f"Candlestick reading unavailable at this timeframe ({e}).")

    # ---------------- forecast ----------------
    ui.section("03", f"{horizon_label} forecast", f"models trained on {spec['label'].lower()} candles")
    horizons = tfm.model_horizons(tf_code, bars, tz, asset_class)
    if bars < tfm.MIN_BARS_FOR_MODEL:
        st.warning(f"Only {bars:,} candles available — at least {tfm.MIN_BARS_FOR_MODEL} are needed to train "
                   f"and validate a model. The chart and indicators above are still computed on real "
                   f"{spec['label'].lower()} data.")
        return

    try:
        with st.spinner(f"Training {len(horizons)} models on {bars:,} {spec['label'].lower()} candles…"):
            engine = train_engine(ticker, tf_code, bars, str(raw.index[-1]), tuple(horizons),
                                  tfm.annualisation_factor(tf_code, tz, asset_class), raw)
    except Exception as e:
        st.error(f"Could not train on {spec['label'].lower()} candles: {e}")
        return
    if not engine.models:
        st.warning("No horizon had enough labelled candles to train on.")
        return

    path = engine.forecast_path(used_candles)
    target_row = path.iloc[-1]
    pred = float(target_row["Predicted_Close"])
    move = pred / last_close - 1
    signal = "BUY" if move > buy_threshold else "SELL" if move < -sell_threshold else "HOLD"

    ui.kpi_grid([
        ui.kpi_html("Last close", f"{cur}{last_close:,.2f}", _fmt_stamp(raw.index[-1], intraday)),
        ui.kpi_html(f"Forecast ({horizon_label})", f"{cur}{pred:,.2f}",
                    f"{move * 100:+.2f}% · {used_candles:,} candles ahead",
                    "up" if move >= 0 else "down"),
        ui.kpi_html("80% range", f"{cur}{float(target_row['Lower_80']):,.2f} – "
                                 f"{cur}{float(target_row['Upper_80']):,.2f}", "validated on this timeframe",
                    "", True),
        ui.kpi_html("Signal", signal, f"thresholds ±{buy_threshold * 100:.1f}%",
                    "up" if signal == "BUY" else "down" if signal == "SELL" else ""),
        ui.kpi_html("Forecast lands", _fmt_stamp(path.index[-1], intraday), "end of the horizon", "", True),
    ])

    hist = raw.tail(max(int(used_candles * 4), 80))
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Actual close",
                               line=dict(color=ui.SERIES[0], width=1.8)))
    fig_f.add_trace(go.Scatter(x=list(path.index) + list(path.index[::-1]),
                               y=list(path["Upper_80"]) + list(path["Lower_80"][::-1]),
                               fill="toself", fillcolor="rgba(90,140,255,0.16)", line=dict(width=0),
                               name="80% likely range", hoverinfo="skip"))
    fig_f.add_trace(go.Scatter(x=path.index, y=path["Predicted_Close"], name="Forecast",
                               line=dict(color=ui.ACCENT, width=2.2, dash="dot")))
    fig_f.update_xaxes(rangebreaks=_session_breaks(raw, intraday))
    fig_f.update_layout(height=430, template=ui.PLOTLY_TEMPLATE, margin=dict(l=10, r=60, t=40, b=10),
                        hovermode="x unified",
                        title=f"{ticker} — {spec['label']} history + {horizon_label} forecast "
                              f"({used_candles:,} candles)")
    fig_f.update_yaxes(side="right")
    st.plotly_chart(fig_f, width="stretch", config={"displaylogo": False})

    summary = engine.summary()
    summary = summary.rename(columns={"Horizon (trading days)": "Horizon (candles)"})
    summary["Horizon (time)"] = [
        f"{tfm.candles_to_trading_days(tf_code, h, tz, asset_class):.2f} sessions" if intraday
        else f"{h} {'week' if tf_code == '1wk' else 'day'}{'s' if h > 1 else ''}"
        for h in summary["Horizon (candles)"]]
    st.dataframe(summary, width="stretch", hide_index=True)
    st.caption(f"Each horizon is a separate model trained on {spec['label'].lower()} candles, validated by "
               f"walk-forward cross-validation on this timeframe's own history. The models, their feature "
               f"selection and their caches are keyed by timeframe — a {spec['label'].lower()} model is "
               f"never reused for another candle size.")

    # ---------------- scenarios at this timeframe ----------------
    ui.section("04", "Scenario range", f"simulated from this symbol's own {spec['label'].lower()} moves")
    sigma = (float(target_row["Upper_80"]) - float(target_row["Lower_80"])) / (2 * Z80 * last_close)
    paths = sx.simulate_paths(raw, used_candles, target_return=move, target_sigma=sigma)
    if paths.size:
        pr = sx.probabilities(paths, last_close)
        ui.kpi_grid([
            ui.kpi_html("Chance of finishing higher", f"{pr['p_up']:.0f}%",
                        f"median {pr['median_return']:+.2f}%", "up" if pr["p_up"] >= 50 else "down"),
            ui.kpi_html("Good case (top 10%)", f"{pr['best_10pct']:+.2f}%",
                        f"{cur}{last_close * (1 + pr['best_10pct'] / 100):,.2f}", "up"),
            ui.kpi_html("Bad case (bottom 10%)", f"{pr['worst_10pct']:+.2f}%",
                        f"{cur}{last_close * (1 + pr['worst_10pct'] / 100):,.2f}", "down"),
            ui.kpi_html("Typical peak on the way", f"{pr['expected_max_gain']:+.2f}%", "median high", "up"),
            ui.kpi_html("Typical dip on the way", f"{pr['expected_max_drop']:+.2f}%", "median low", "down"),
        ])
        st.caption(f"3,000 paths of {used_candles:,} {spec['label'].lower()} candles each, resampled in "
                   f"blocks from this symbol's own {spec['label'].lower()} returns, then centred on the "
                   f"forecast above and scaled to its validated uncertainty.")
