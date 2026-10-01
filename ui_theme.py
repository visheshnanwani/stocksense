"""
ui_theme.py
-------------
The dashboard's visual design system: colors, CSS, the Plotly chart theme,
and small HTML building blocks (hero header, KPI cards, range bars,
snapshot cards, section headings). dashboard.py only calls these helpers,
so the look can be changed in one place.

Color choices:
  - Chart series use the first slots of a colorblind-validated categorical
    palette (blue, orange, aqua, ...) in fixed order.
  - Up/down (gain/loss) use reserved status colors, and are always paired
    with an arrow/label, never color alone.
"""

import numpy as _np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

APP_NAME = "StockSense"
APP_TAGLINE = "AI market analytics"

# ---- palette (dark) ----
BG = "#0b0e14"
SURFACE = "#151a23"
BORDER = "rgba(255,255,255,0.07)"
TEXT = "#e8ecf3"
TEXT_2 = "#9aa4b2"
TEXT_3 = "#6b7483"
ACCENT = "#3987e5"
ACCENT_2 = "#9085e9"

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
UP = "#0ca30c"        # status: good
DOWN = "#d03b3b"      # status: critical
WARN = "#fab219"      # status: warning
NEUTRAL = "#8b95a5"

# BUY/SELL/HOLD badge: (text color, background, icon)
SIGNAL_STYLE = {
    "BUY": ("#5fd35f", "rgba(12,163,12,0.16)", "▲"),
    "SELL": ("#f07474", "rgba(208,59,59,0.16)", "▼"),
    "HOLD": ("#c3c9d4", "rgba(139,149,165,0.16)", "●"),
}

PLOTLY_TEMPLATE = "stocksense"


def _register_plotly_template():
    axis = dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False, linecolor="rgba(255,255,255,0.08)",
                tickfont=dict(color=TEXT_3, size=11), title=dict(font=dict(color=TEXT_2, size=12)),
                showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                spikedash="dot", spikecolor="rgba(255,255,255,0.28)")
    pio.templates[PLOTLY_TEMPLATE] = go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_2, size=12),
        title=dict(font=dict(color=TEXT, size=15), x=0.01, xanchor="left"),
        colorway=SERIES,
        xaxis=axis, yaxis=axis,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1b2130", bordercolor="rgba(255,255,255,0.12)",
                        font=dict(family="Inter, sans-serif", color=TEXT, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c3c9d4", size=11),
                    orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=10, r=10, t=40, b=10),
    ))


_register_plotly_template()


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
  --bg: #0b0e14; --surface: rgba(21,26,35,0.72); --surface-solid: #151a23;
  --border: rgba(255,255,255,0.07); --border-strong: rgba(255,255,255,0.12);
  --text: #e8ecf3; --text-2: #9aa4b2; --text-3: #6b7483;
  --accent: #3987e5; --accent-2: #9085e9;
  --up: #0ca30c; --up-text: #5fd35f; --down: #d03b3b; --down-text: #f07474; --warn: #fab219;
  --radius: 16px;
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, rgba(57,135,229,0.16), transparent 60%),
    radial-gradient(900px 420px at 95% -5%, rgba(144,133,233,0.13), transparent 60%),
    var(--bg) !important;
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] [data-testid="stAppDeployButton"], .stDeployButton, footer { display: none !important; }
.block-container { padding-top: 1.6rem !important; max-width: 1400px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f131b 0%, #0b0e14 100%) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1 { font-size: 0.78rem !important; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text-3) !important; font-weight: 700 !important; margin: 0.6rem 0 0.2rem 0 !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 0.9rem 0 !important; }
.brand { display: flex; align-items: center; gap: 12px; padding: 4px 0 14px 0; }
.brand-logo { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)); box-shadow: 0 8px 24px rgba(57,135,229,0.35);
  color: white; font-weight: 800; font-size: 18px; }
.brand-name { font-size: 1.15rem; font-weight: 800; color: var(--text); letter-spacing: -0.01em; }
.brand-sub { font-size: 0.72rem; color: var(--text-3); letter-spacing: 0.04em; }
.side-ticker { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
  background: rgba(57,135,229,0.12); border: 1px solid rgba(57,135,229,0.35); color: #9cc4f5;
  font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 600; }

/* ---------- hero ---------- */
.hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap;
  padding: 26px 30px; border-radius: 22px; border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(57,135,229,0.12), rgba(144,133,233,0.06) 55%, rgba(21,26,35,0.6));
  box-shadow: 0 20px 50px rgba(0,0,0,0.35); margin-bottom: 18px; position: relative; overflow: hidden; }
.hero::after { content: ""; position: absolute; right: -80px; top: -80px; width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle, rgba(144,133,233,0.22), transparent 70%); pointer-events: none; }
.hero-eyebrow { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
.tick { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.8rem; color: white;
  padding: 4px 10px; border-radius: 8px; background: linear-gradient(135deg, var(--accent), var(--accent-2)); }
.chip { font-size: 0.74rem; color: var(--text-2); padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--border-strong); background: rgba(255,255,255,0.03); }
.hero-title { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.025em; line-height: 1.1; margin: 0;
  background: linear-gradient(90deg, #ffffff, #b8c7ff); -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; }
.hero-sub { color: var(--text-3); font-size: 0.85rem; margin-top: 8px; }
.hero-right { text-align: right; z-index: 1; }
.price { font-family: 'JetBrains Mono', monospace; font-size: 2.9rem; font-weight: 600; color: var(--text);
  letter-spacing: -0.03em; line-height: 1; }
.chg { display: inline-flex; gap: 6px; align-items: center; margin-top: 10px; padding: 5px 12px; border-radius: 999px;
  font-weight: 700; font-size: 0.95rem; font-family: 'JetBrains Mono', monospace; }
.chg.up { color: var(--up-text); background: rgba(12,163,12,0.14); border: 1px solid rgba(12,163,12,0.35); }
.chg.down { color: var(--down-text); background: rgba(208,59,59,0.14); border: 1px solid rgba(208,59,59,0.35); }
.status { margin-top: 10px; font-size: 0.78rem; color: var(--text-3); display: flex; gap: 8px; align-items: center;
  justify-content: flex-end; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-3); display: inline-block; }
.dot.live { background: var(--up); box-shadow: 0 0 0 0 rgba(12,163,12,0.7); animation: pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(12,163,12,0.6); } 70% { box-shadow: 0 0 0 9px rgba(12,163,12,0); }
  100% { box-shadow: 0 0 0 0 rgba(12,163,12,0); } }

/* ---------- KPI cards ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 700px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .hero-right { text-align: left; }
  .status { justify-content: flex-start; } .hero-title { font-size: 1.7rem; } .price { font-size: 2.2rem; } }
.kpi { padding: 16px 18px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border);
  backdrop-filter: blur(10px); transition: transform .15s ease, border-color .15s ease; }
.kpi:hover { transform: translateY(-2px); border-color: rgba(57,135,229,0.35); }
.kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3); font-weight: 600; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: clamp(1.0rem, 1.6vw, 1.35rem);
  font-weight: 600; color: var(--text); margin-top: 6px;
  /* wrap rather than clip -- an ellipsis on a price ("$578. ...") is worse than a second line */
  white-space: normal; overflow-wrap: anywhere; line-height: 1.25; }
/* the 10-second summary card */
.tldr { background: linear-gradient(135deg, rgba(90,140,255,0.10), rgba(90,140,255,0.02));
  border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 14px;
  padding: 16px 18px; margin: 14px 0 6px 0; }
.tldr-k { font-size: 0.68rem; letter-spacing: .1em; text-transform: uppercase; color: var(--accent);
  font-weight: 700; margin-bottom: 8px; }
.tldr-body { font-size: 0.95rem; line-height: 1.6; color: var(--text); }
.tldr-body b { color: var(--text); }
.tldr-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.tldr-tag { font-size: 0.74rem; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line);
  background: var(--surface-2); color: var(--text-2); }
.tldr-tag.up { color: var(--up-text); border-color: rgba(12,163,12,0.45); }
.tldr-tag.down { color: var(--down-text); border-color: rgba(208,59,59,0.45); }

/* click-to-explain score rings */
.ring-card.why { cursor: pointer; position: relative; transition: border-color .15s ease, transform .15s ease; }
.ring-card.why > summary { list-style: none; cursor: pointer; }
.ring-card.why > summary::-webkit-details-marker { display: none; }
.ring-card.why:hover { border-color: var(--accent); }
.ring-card.why .why-hint { font-size: 0.68rem; letter-spacing: .06em; text-transform: uppercase;
  color: var(--text-3); margin-top: 8px; }
.ring-card.why[open] .why-hint::after { content: " \2014 tap to close"; }
.ring-card.why:not([open]) .why-hint::after { content: " \2014 why?"; }
/* an opened explanation takes the full row so the text is readable */
.ring-card.why[open] { grid-column: 1 / -1; text-align: left; }
.ring-card.why[open] > summary { display: flex; align-items: center; gap: 14px; }
.ring-card.why[open] > summary .ring-label,
.ring-card.why[open] > summary .ring-tag { text-align: left; }
.why-body { text-align: left; margin-top: 12px; border-top: 1px solid var(--line); padding-top: 10px; }
.why-body ul { list-style: none; margin: 0; padding: 0; }
.why-body li { display: flex; justify-content: space-between; gap: 10px; align-items: baseline;
  padding: 5px 0; border-bottom: 1px dashed var(--line); font-size: 0.82rem; }
.why-body li:last-child { border-bottom: 0; }
.why-body .why-k { color: var(--text-2); }
.why-body .why-v { font-family: 'JetBrains Mono', monospace; color: var(--text); white-space: nowrap; }
.why-body .why-s { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; min-width: 46px; text-align: right; }
.why-note { font-size: 0.76rem; color: var(--text-3); margin-top: 8px; line-height: 1.45; }
.rung-prob { font-size: 0.78rem; margin-top: 6px; font-weight: 600; }
.rung-prob.up { color: var(--up-text); } .rung-prob.down { color: var(--down-text); }
.rung-prob.flat { color: var(--text-2); }
.rung-swing { font-size: 0.72rem; color: var(--text-3); margin-top: 2px; }
.kpi-value.sm { font-size: 0.95rem; line-height: 1.3; white-space: normal; overflow: visible; text-overflow: clip; }
.kpi-sub { font-size: 0.76rem; margin-top: 4px; color: var(--text-2); }
.kpi-sub.up { color: var(--up-text); } .kpi-sub.down { color: var(--down-text); }
.kpi.wide { grid-column: span 2; }
.range-track { position: relative; height: 6px; border-radius: 999px; margin: 14px 0 8px 0;
  background: linear-gradient(90deg, rgba(208,59,59,0.55), rgba(250,178,25,0.45), rgba(12,163,12,0.55)); }
.range-marker { position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; transform: translate(-50%, -50%);
  background: white; border: 3px solid var(--accent); box-shadow: 0 0 0 4px rgba(57,135,229,0.25); }
.range-labels { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem; color: var(--text-2); }

/* ---------- snapshot cards ---------- */
.snap-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 4px 0 20px 0; }
@media (max-width: 700px) { .snap-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.snap { padding: 16px 18px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border);
  position: relative; overflow: hidden; }
.snap::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--text-3); }
.snap.up::before { background: var(--up); } .snap.down::before { background: var(--down); }
.snap.warn::before { background: var(--warn); } .snap.info::before { background: var(--accent); }
.snap-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3); font-weight: 600;
  display: flex; gap: 6px; align-items: center; }
.snap-value { font-size: 1.4rem; font-weight: 700; color: var(--text); margin-top: 8px; letter-spacing: -0.01em; }
.snap-value .ic-up { color: var(--up-text); } .snap-value .ic-down { color: var(--down-text); }
.snap-value .ic-flat { color: var(--text-2); }
.snap-sub { font-size: 0.78rem; color: var(--text-2); margin-top: 4px; }

/* ---------- section headings ---------- */
.section { display: flex; align-items: baseline; gap: 12px; margin: 26px 0 10px 0; }
.section-kicker { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--accent); font-weight: 600;
  padding: 2px 8px; border-radius: 6px; background: rgba(57,135,229,0.12); }
.section-title { font-size: 1.3rem; font-weight: 700; color: var(--text); letter-spacing: -0.01em; }
.section-sub { font-size: 0.85rem; color: var(--text-3); }

/* ---------- native widgets ---------- */
div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 14px 18px !important; backdrop-filter: blur(10px); }
div[data-testid="stMetric"] label, div[data-testid="stMetricLabel"] { color: var(--text-3) !important;
  text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.72rem !important; font-weight: 600; }
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-weight: 600 !important;
  font-size: 1.2rem !important; }
div[data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; line-height: 1.3; }
/* tabs: role-based selectors work on both older (baseweb) and newer Streamlit tab markup */
.stTabs [role="tablist"] { gap: 6px; background: rgba(255,255,255,0.03); padding: 6px; border-radius: 14px;
  border: 1px solid var(--border); height: auto !important; }
.stTabs [role="tab"] { height: 44px; border-radius: 10px; padding: 0 20px !important; flex: 1 1 auto; justify-content: center; color: var(--text-2);
  font-weight: 600; background: transparent; display: flex; align-items: center; border: none !important; }
.stTabs [role="tab"] p { font-weight: 600 !important; font-size: 0.92rem !important; }
.stTabs [role="tab"]:hover { color: var(--text); background: rgba(255,255,255,0.04); }
.stTabs [role="tab"][aria-selected="true"] { color: white !important;
  background: linear-gradient(135deg, rgba(57,135,229,0.9), rgba(144,133,233,0.85)) !important;
  box-shadow: 0 6px 18px rgba(57,135,229,0.3); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stButton > button, .stDownloadButton > button { border-radius: 12px !important; font-weight: 600 !important;
  border: 1px solid var(--border-strong) !important; transition: all .15s ease; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  border: none !important; box-shadow: 0 8px 22px rgba(57,135,229,0.35); color: white !important; }
.stButton > button:hover { transform: translateY(-1px); }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
  background: var(--surface); }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stPlotlyChart"] { background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
  padding: 8px 6px 2px 6px; }
[data-testid="stAlert"] { border-radius: 14px !important; }
h2, h3 { letter-spacing: -0.015em; font-weight: 700 !important; }

.signal-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 18px; border-radius: 999px;
  font-size: 1.25rem; font-weight: 800; letter-spacing: 0.02em; font-family: 'JetBrains Mono', monospace; }


/* ---------- forecast ladder & feature-group chips ---------- */
.ladder { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 4px 0 16px 0; }
@media (max-width: 900px) { .ladder { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.rung { padding: 14px 16px; border-radius: var(--radius); border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(57,135,229,0.10), rgba(21,26,35,0.7)); position: relative; }
.rung-h { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3); font-weight: 700; }
.rung-p { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: var(--text); margin-top: 6px; }
.rung-c { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 600; margin-top: 2px; }
.rung-c.up { color: var(--up-text); } .rung-c.down { color: var(--down-text); } .rung-c.flat { color: var(--text-2); }
.rung-r { font-size: 0.72rem; color: var(--text-3); margin-top: 6px; font-family: 'JetBrains Mono', monospace; }
.gchips { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 14px 0; align-items: center; }
.gchips-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3); font-weight: 700; margin-right: 4px; }
.gchip { font-size: 0.78rem; padding: 5px 12px; border-radius: 999px; font-weight: 600; border: 1px solid var(--border-strong); }
.gchip.on { color: #9cc4f5; background: rgba(57,135,229,0.14); border-color: rgba(57,135,229,0.45); }
.gchip.off { color: var(--text-3); background: rgba(255,255,255,0.02); text-decoration: line-through; }
.gchip.na { color: var(--text-3); background: transparent; border-style: dashed; }

/* ---------- market ticker tape ---------- */
.tape { display: flex; gap: 10px; overflow-x: auto; padding: 2px 0 14px 0; scrollbar-width: none; }
.tape::-webkit-scrollbar { display: none; }
.tape-item { flex: 0 0 auto; display: flex; gap: 10px; align-items: baseline; padding: 8px 14px; border-radius: 12px;
  background: var(--surface); border: 1px solid var(--border); white-space: nowrap; }
.tape-name { font-size: 0.72rem; color: var(--text-3); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.tape-val { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text); font-weight: 600; }
.tape-chg { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600; }
.tape-chg.up { color: var(--up-text); } .tape-chg.down { color: var(--down-text); } .tape-chg.flat { color: var(--text-2); }

/* ---------- data-usage matrix ---------- */
.umx { width: 100%; border-collapse: separate; border-spacing: 0 6px; margin: 2px 0 12px 0; }
.umx th { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3); font-weight: 700;
  text-align: center; padding: 0 6px 2px 6px; }
.umx th:first-child { text-align: left; }
.umx td { background: var(--surface); padding: 8px 10px; text-align: center; border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border); font-size: 0.82rem; }
.umx td:first-child { text-align: left; border-left: 1px solid var(--border); border-radius: 10px 0 0 10px;
  color: var(--text); font-weight: 600; white-space: nowrap; }
.umx td:last-child { border-right: 1px solid var(--border); border-radius: 0 10px 10px 0; }
.umx .on { display: inline-block; min-width: 28px; padding: 2px 8px; border-radius: 999px; font-weight: 700;
  color: #9cc4f5; background: rgba(57,135,229,0.18); border: 1px solid rgba(57,135,229,0.5); }
.umx .off { color: var(--text-3); }
.umx .na { color: var(--text-3); opacity: 0.6; font-style: italic; font-size: 0.74rem; }
.umx .alpha td { background: transparent; border: none; color: var(--text-2); font-family: 'JetBrains Mono', monospace;
  font-size: 0.76rem; padding-top: 2px; }
.umx .alpha td:first-child { color: var(--text-3); font-family: 'Inter', sans-serif; font-weight: 600; }

/* ================= animations ================= */
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes heroShift { 0% { background-position: 0% 50%; } 100% { background-position: 100% 50%; } }
@keyframes shimmer { to { background-position: 200% center; } }
@keyframes marquee { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes ping { 0% { transform: scale(1); opacity: .7; } 80%, 100% { transform: scale(2.4); opacity: 0; } }
@keyframes glowPulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(57,135,229,0); } 50% { box-shadow: 0 0 28px 0 rgba(57,135,229,0.25); } }
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
@keyframes ringFill { from { --p: 0; } }

.hero, .kpi, .snap, .rung, .ring-card, .verdict, .rcard, .live-head, .umx tbody tr {
  animation: fadeUp .6s cubic-bezier(.2,.7,.2,1) both; }
.kpi-grid > :nth-child(1), .snap-grid > :nth-child(1), .ladder > :nth-child(1), .rings > :nth-child(1) { animation-delay: 0.04s; }
.kpi-grid > :nth-child(2), .snap-grid > :nth-child(2), .ladder > :nth-child(2), .rings > :nth-child(2) { animation-delay: 0.08s; }
.kpi-grid > :nth-child(3), .snap-grid > :nth-child(3), .ladder > :nth-child(3), .rings > :nth-child(3) { animation-delay: 0.12s; }
.kpi-grid > :nth-child(4), .snap-grid > :nth-child(4), .ladder > :nth-child(4), .rings > :nth-child(4) { animation-delay: 0.16s; }
.kpi-grid > :nth-child(5), .snap-grid > :nth-child(5), .ladder > :nth-child(5), .rings > :nth-child(5) { animation-delay: 0.20s; }
.kpi-grid > :nth-child(6), .snap-grid > :nth-child(6), .ladder > :nth-child(6), .rings > :nth-child(6) { animation-delay: 0.24s; }
.kpi-grid > :nth-child(7), .snap-grid > :nth-child(7), .ladder > :nth-child(7), .rings > :nth-child(7) { animation-delay: 0.28s; }
.kpi-grid > :nth-child(8), .snap-grid > :nth-child(8), .ladder > :nth-child(8), .rings > :nth-child(8) { animation-delay: 0.32s; }
.kpi-grid > :nth-child(9), .snap-grid > :nth-child(9), .ladder > :nth-child(9), .rings > :nth-child(9) { animation-delay: 0.36s; }
.kpi-grid > :nth-child(10), .snap-grid > :nth-child(10), .ladder > :nth-child(10), .rings > :nth-child(10) { animation-delay: 0.40s; }
.kpi-grid > :nth-child(11), .snap-grid > :nth-child(11), .ladder > :nth-child(11), .rings > :nth-child(11) { animation-delay: 0.44s; }
.kpi-grid > :nth-child(12), .snap-grid > :nth-child(12), .ladder > :nth-child(12), .rings > :nth-child(12) { animation-delay: 0.48s; }

.hero { background-size: 220% 220% !important; animation: fadeUp .6s cubic-bezier(.2,.7,.2,1) both, heroShift 14s ease-in-out infinite alternate; }
.hero-title { background: linear-gradient(90deg, #ffffff 0%, #b8c7ff 25%, #ffffff 50%, #b8c7ff 75%, #ffffff 100%) !important;
  background-size: 200% auto !important; -webkit-background-clip: text !important; background-clip: text !important;
  animation: shimmer 8s linear infinite; }
.kpi:hover, .snap:hover, .rung:hover, .ring-card:hover, .rcard:hover {
  box-shadow: 0 0 0 1px rgba(57,135,229,0.35), 0 12px 32px rgba(57,135,229,0.14); transform: translateY(-2px); }
.snap, .rung, .ring-card, .rcard { transition: transform .15s ease, box-shadow .2s ease; }

/* scrolling ticker tape */
.tape { overflow: hidden !important; -webkit-mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent);
  mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent); }
.tape-track { display: flex; gap: 10px; width: max-content; animation: marquee 50s linear infinite; }
.tape:hover .tape-track { animation-play-state: paused; }

/* score rings */
.rings { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 12px; margin: 6px 0 18px 0; }
@media (max-width: 1100px) { .rings { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
@media (max-width: 700px) { .rings { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.ring-card { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 16px 8px 14px 8px;
  border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border); }
.ring { --p: var(--val); width: 88px; height: 88px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,0.07) 0);
  animation: ringFill 1.4s cubic-bezier(.2,.7,.2,1) both; }
.ring.big { width: 150px; height: 150px; }
.ring-inner { width: calc(100% - 16px); height: calc(100% - 16px); border-radius: 50%; background: #10141c;
  display: grid; place-items: center; text-align: center; }
.ring-num { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.25rem; color: var(--text); line-height: 1; }
.ring.big .ring-num { font-size: 2.4rem; }
.ring-den { font-size: 0.7rem; color: var(--text-3); }
.ring-label { font-size: 0.78rem; font-weight: 700; color: var(--text); text-align: center; }
.ring-tag { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }

/* verdict banner */
.verdict { display: flex; gap: 28px; align-items: center; padding: 24px 28px; border-radius: 22px; margin-bottom: 16px;
  border: 1px solid var(--border); background: linear-gradient(135deg, rgba(57,135,229,0.14), rgba(144,133,233,0.07) 60%, rgba(21,26,35,0.6));
  animation: fadeUp .6s cubic-bezier(.2,.7,.2,1) both, glowPulse 5s ease-in-out 1s infinite; flex-wrap: wrap; }
.verdict-kicker { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3); font-weight: 700; }
.verdict-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 4px 0 2px 0; }
.verdict-stance { font-size: 1.02rem; color: #b8c7ff; font-weight: 600; margin-bottom: 10px; }
.verdict-text { color: var(--text-2); font-size: 0.92rem; line-height: 1.6; max-width: 900px; }
.verdict-text b { color: var(--text); }

/* research cards */
.rgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
@media (max-width: 900px) { .rgrid { grid-template-columns: 1fr; } }
.rcard { padding: 18px 20px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border); margin-bottom: 14px; }
.rcard h4 { margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 700; color: var(--text); }
.rcard ul { margin: 0; padding-left: 0; list-style: none; }
.rcard li { padding: 7px 0; border-bottom: 1px dashed var(--border); color: var(--text-2); font-size: 0.88rem; display: flex; gap: 10px; }
.rcard li:last-child { border-bottom: none; }
.rcard li b { color: var(--text); }
.rcard li .tag { flex: 0 0 auto; font-size: 0.66rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  padding: 2px 8px; border-radius: 999px; height: fit-content; margin-top: 2px; }
.rcard.bull { border-top: 3px solid var(--up); } .rcard.bear { border-top: 3px solid var(--down); }
.rcard.bull .tag { color: var(--up-text); background: rgba(12,163,12,0.14); }
.rcard.bear .tag { color: var(--down-text); background: rgba(208,59,59,0.14); }
.rcard .empty { color: var(--text-3); font-size: 0.85rem; font-style: italic; }
.rcard p { color: var(--text-2); font-size: 0.9rem; line-height: 1.6; margin: 0; }
.facts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }
@media (max-width: 900px) { .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.fact { padding: 12px 14px; border-radius: 12px; background: var(--surface); border: 1px solid var(--border); }
.fact-k { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3); font-weight: 700; }
.fact-v { font-size: 0.95rem; color: var(--text); font-weight: 600; margin-top: 4px; word-break: break-word; }

/* live panel header */
.live-head { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
  padding: 14px 18px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border); margin-bottom: 10px; }
.live-badge { display: inline-flex; align-items: center; gap: 8px; font-size: 0.74rem; font-weight: 800; letter-spacing: 0.12em;
  padding: 5px 12px; border-radius: 999px; }
.live-badge.on { color: #ffffff; background: rgba(208,59,59,0.9); }
.live-badge.off { color: var(--text-2); background: rgba(255,255,255,0.06); border: 1px solid var(--border-strong); }
.live-dot { position: relative; width: 8px; height: 8px; border-radius: 50%; background: #fff; }
.live-dot::after { content: ""; position: absolute; inset: 0; border-radius: 50%; background: #fff; animation: ping 1.4s cubic-bezier(0,0,.2,1) infinite; }
.live-price { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 600; color: var(--text); }
.live-stats { display: flex; gap: 22px; flex-wrap: wrap; }
.live-stat .k { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3); font-weight: 700; }
.live-stat .v { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: var(--text); font-weight: 600; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
.footer { margin-top: 40px; padding: 18px 0 8px 0; border-top: 1px solid var(--border); color: var(--text-3);
  font-size: 0.8rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }

/* ================= FX LAYER v2 ================= */

@property --ang { syntax: '<angle>'; initial-value: 0deg; inherits: false; }

@keyframes auroraDrift {
  0%   { transform: translate3d(-8%, -5%, 0) scale(1.1) rotate(0deg); }
  33%  { transform: translate3d(7%, 4%, 0) scale(1.28) rotate(6deg); }
  66%  { transform: translate3d(-4%, 8%, 0) scale(1.16) rotate(-5deg); }
  100% { transform: translate3d(-8%, -5%, 0) scale(1.1) rotate(0deg); }
}
@keyframes hueRoll   { to { filter: blur(52px) saturate(190%) hue-rotate(360deg); } }
@keyframes spinBorder{ to { --ang: 360deg; } }
@keyframes scanLine  { 0% { top: -20%; opacity: 0; } 8% { opacity: 1; }
                       92% { opacity: 1; } 100% { top: 120%; opacity: 0; } }
@keyframes neonBreathe {
  0%, 100% { box-shadow: 0 0 0 1px rgba(57,135,229,.30), 0 0 18px rgba(57,135,229,.16),
                         0 10px 30px rgba(0,0,0,.45); }
  50%      { box-shadow: 0 0 0 1px rgba(120,110,255,.55), 0 0 34px rgba(120,110,255,.34),
                         0 10px 30px rgba(0,0,0,.45); }
}
@keyframes slideTilt { from { opacity: 0; transform: translateY(26px) rotateX(10deg) scale(.95); }
                       to   { opacity: 1; transform: none; } }
@keyframes gridPan   { to { background-position: 64px 64px; } }
@keyframes floatY    { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }
@keyframes underline { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes textGlow  { 0%,100% { filter: drop-shadow(0 0 14px rgba(120,180,255,.45)); }
                       50%     { filter: drop-shadow(0 0 30px rgba(150,120,255,.75)); } }
@keyframes barFlow   { to { background-position: 300% 0; } }
@keyframes popNum    { from { opacity: 0; transform: translateY(10px) scale(.9); } to { opacity:1; transform:none; } }

/* ---------- AURORA: loud ---------- */
.stApp::before {
  content: ""; position: fixed; inset: -30vmax; z-index: 0; pointer-events: none;
  background:
    radial-gradient(40vmax 32vmax at 16% 10%, rgba(64,150,255,0.62), transparent 60%),
    radial-gradient(36vmax 30vmax at 84% 20%, rgba(160,110,255,0.58), transparent 60%),
    radial-gradient(34vmax 28vmax at 66% 88%, rgba(0,224,200,0.42), transparent 60%),
    radial-gradient(32vmax 26vmax at 10% 82%, rgba(255,80,180,0.38), transparent 60%),
    radial-gradient(30vmax 24vmax at 48% 48%, rgba(255,170,60,0.22), transparent 62%);
  filter: blur(52px) saturate(190%);
  opacity: .55;
  animation: auroraDrift 22s ease-in-out infinite, hueRoll 40s linear infinite;
  will-change: transform, filter;
}
/* visible grid */
.stApp::after {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: .55;
  background-image:
    linear-gradient(rgba(130,180,255,0.085) 1px, transparent 1px),
    linear-gradient(90deg, rgba(130,180,255,0.085) 1px, transparent 1px);
  background-size: 64px 64px;
  mask-image: radial-gradient(ellipse 95% 70% at 50% 0%, #000 30%, transparent 82%);
  -webkit-mask-image: radial-gradient(ellipse 95% 70% at 50% 0%, #000 30%, transparent 82%);
  animation: gridPan 18s linear infinite;
}
.stApp > * { position: relative; z-index: 1; }

/* ---------- scan line sweeping the page ---------- */
[data-testid="stAppViewContainer"] { position: relative; }
[data-testid="stAppViewContainer"]::after {
  content: ""; position: fixed; left: 0; right: 0; height: 22vh; z-index: 2; pointer-events: none;
  background: linear-gradient(180deg, transparent, rgba(120,190,255,0.12) 45%,
              rgba(170,130,255,0.16) 55%, transparent);
  animation: scanLine 9s ease-in-out infinite; mix-blend-mode: screen;
}

/* ---------- glass + neon on every card ---------- */
.kpi, .snap, .hero, .tldr, .ladder, .rings, .live-head, [data-testid="stMetric"] {
  backdrop-filter: blur(14px) saturate(150%);
  -webkit-backdrop-filter: blur(14px) saturate(150%);
  background-color: rgba(14,19,28,0.55) !important;
  border: 1px solid rgba(120,160,255,0.22) !important;
}
.kpi, [data-testid="stMetric"] { animation: slideTilt .7s cubic-bezier(.2,.8,.2,1) both,
                                            neonBreathe 4.5s ease-in-out infinite 1s; }
.kpi-grid { perspective: 1200px; }
.kpi-grid > .kpi:nth-child(1){animation-delay:.00s,1.0s}
.kpi-grid > .kpi:nth-child(2){animation-delay:.07s,1.1s}
.kpi-grid > .kpi:nth-child(3){animation-delay:.14s,1.2s}
.kpi-grid > .kpi:nth-child(4){animation-delay:.21s,1.3s}
.kpi-grid > .kpi:nth-child(5){animation-delay:.28s,1.4s}
.kpi-grid > .kpi:nth-child(6){animation-delay:.35s,1.5s}
.kpi-grid > .kpi:nth-child(7){animation-delay:.42s,1.6s}
.kpi-grid > .kpi:nth-child(8){animation-delay:.49s,1.7s}

.kpi, .snap {
  position: relative;
  background-image: linear-gradient(100deg, transparent 38%, rgba(255,255,255,0.14) 50%, transparent 62%);
  background-size: 240% 100%; background-position: 180% 0; background-repeat: no-repeat;
  transition: transform .34s cubic-bezier(.2,.8,.2,1), box-shadow .34s ease,
              border-color .34s ease, background-position .85s ease;
}
.kpi:hover, .snap:hover, [data-testid="stMetric"]:hover {
  transform: translateY(-10px) scale(1.04);
  border-color: rgba(150,200,255,.85) !important;
  box-shadow: 0 0 0 1px rgba(150,200,255,.7), 0 0 46px rgba(90,150,255,.55),
              0 26px 60px rgba(0,0,0,.6) !important;
  background-position: -80% 0;
  z-index: 5;
}
.kpi:hover .kpi-value { text-shadow: 0 0 26px rgba(150,200,255,.9); }

/* ---------- hero: thick spinning rim + glowing text ---------- */
.hero { position: relative; isolation: isolate; }
.hero::before {
  content: ""; position: absolute; inset: -2px; border-radius: inherit; z-index: -1; padding: 2px;
  background: conic-gradient(from var(--ang), #4096ff, #a06eff, #00e0c8, #ff50b4, #ffaa3c, #4096ff);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude;
  animation: spinBorder 5s linear infinite; opacity: 1;
  filter: drop-shadow(0 0 12px rgba(120,150,255,.7));
}
.hero-title {
  background: linear-gradient(92deg, #fff, #7fd0ff 30%, #c0a0ff 55%, #ff9ad5 75%, #fff);
  background-size: 300% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 4s linear infinite, textGlow 3s ease-in-out infinite;
}
.hero-price {
  background: linear-gradient(92deg, #fff, #9fe0ff 40%, #fff);
  background-size: 260% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 5s linear infinite, textGlow 2.6s ease-in-out infinite;
}

/* ---------- section headers ---------- */
.section { position: relative; }
.section .kicker {
  animation: floatY 3s ease-in-out infinite;
  background: linear-gradient(135deg, #4096ff, #a06eff); color: #fff !important;
  box-shadow: 0 0 22px rgba(120,140,255,.6);
}
.section h2, .section .title { position: relative; display: inline-block; }
.section h2::after, .section .title::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: -7px; height: 3px; border-radius: 3px;
  background: linear-gradient(90deg, #4096ff, #a06eff, #00e0c8, transparent);
  background-size: 300% 100%;
  transform-origin: left;
  animation: underline .9s cubic-bezier(.2,.8,.2,1) both .1s, barFlow 4s linear infinite;
  box-shadow: 0 0 16px rgba(120,150,255,.7);
}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
  position: relative; border-radius: 12px !important;
  border: 1px solid rgba(120,170,255,.5) !important;
  transition: transform .2s cubic-bezier(.2,.8,.2,1), box-shadow .25s ease, filter .25s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-4px) scale(1.05);
  box-shadow: 0 0 34px rgba(90,150,255,.7), 0 14px 36px rgba(0,0,0,.5) !important;
  filter: brightness(1.2);
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab"] { transition: color .25s ease, transform .25s ease; }
.stTabs [data-baseweb="tab"]:hover { transform: translateY(-2px); color: #dce9ff !important; }
.stTabs [data-baseweb="tab-highlight"] {
  background: linear-gradient(90deg, #4096ff, #a06eff, #00e0c8) !important;
  background-size: 300% 100%; animation: barFlow 3s linear infinite;
  box-shadow: 0 0 20px rgba(120,150,255,.9); height: 4px !important; border-radius: 4px;
}

/* ---------- charts / tables ---------- */
.js-plotly-plot, [data-testid="stDataFrame"] {
  animation: slideTilt .8s cubic-bezier(.2,.8,.2,1) both; border-radius: 16px;
  transition: box-shadow .35s ease, transform .35s ease;
}
.js-plotly-plot:hover, [data-testid="stDataFrame"]:hover {
  box-shadow: 0 0 0 1px rgba(120,170,255,.45), 0 0 40px rgba(90,150,255,.3), 0 20px 50px rgba(0,0,0,.5);
}

/* ---------- sidebar ---------- */
.brand-logo {
  position: relative; animation: floatY 3.4s ease-in-out infinite;
  box-shadow: 0 0 40px rgba(120,150,255,.8) !important;
}
.brand-logo::after {
  content: ""; position: absolute; inset: -6px; border-radius: 18px; z-index: -1;
  background: conic-gradient(from var(--ang), #4096ff, #a06eff, #00e0c8, #4096ff);
  filter: blur(12px); opacity: .9; animation: spinBorder 4s linear infinite;
}
.brand-name {
  background: linear-gradient(92deg, #fff, #7fd0ff 45%, #c0a0ff 70%, #fff); background-size: 280% auto;
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  animation: shimmer 4.5s linear infinite;
}
[data-testid="stSidebar"] { border-right: 1px solid rgba(120,160,255,.28) !important; }

/* ---------- count-up numbers (driven by JS, see boot_fx) ---------- */
.kpi-value.counted, [data-testid="stMetricValue"].counted { animation: popNum .5s cubic-bezier(.2,.8,.2,1) both; }

/* ---------- misc ---------- */
::-webkit-scrollbar { width: 12px; height: 12px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,.04); }
::-webkit-scrollbar-thumb { border-radius: 999px; border: 2px solid transparent; background-clip: padding-box;
  background-color: rgba(120,160,255,.55); }
::-webkit-scrollbar-thumb:hover { background-color: rgba(150,190,255,.9); }
.chip:hover, .tick:hover { transform: translateY(-3px) scale(1.06);
  box-shadow: 0 0 22px rgba(120,160,255,.6); }
.ring:hover { transform: scale(1.1) rotate(3deg); filter: drop-shadow(0 0 26px rgba(120,160,255,.8)); }
.signal-badge { animation: slideTilt .6s cubic-bezier(.2,.8,.2,1) both; }
.signal-badge:hover { transform: scale(1.08); box-shadow: 0 0 34px rgba(120,160,255,.6); }

@media (prefers-reduced-motion: reduce) {
  .stApp::before, .stApp::after, [data-testid="stAppViewContainer"]::after { animation: none !important; }
}

/* ---------- top loading bar ---------- */
@keyframes barShine { to { background-position: 300% 0; } }
@keyframes barFade  { to { opacity: 0; transform: translateY(-6px); } }
@keyframes stagePop { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
/* `.stApp > * { z-index: 1 }` (needed so content sits above the aurora) puts every
   Streamlit container in its own stacking context, so a fixed child's z-index only
   ranks it INSIDE that container -- a later sibling still paints over it. Verified
   with elementFromPoint(400,4), which returned DIV.hero rather than this bar.
   Lifting the container that actually holds the bar fixes it. */
.loadwrap {
  position: fixed; top: 0; left: 0; right: 0; z-index: 10000; pointer-events: none;
  display: flex; flex-direction: column; gap: 0;
}
.loadwrap.done { animation: barFade .9s ease .5s both; }
/* Deliberately chunky: at 3px this was invisible to anyone who did not already
   know it was there. 8px with a track, a leading spark and a bold pill reads as
   "the page is loading" at a glance. */
.loadbar {
  height: 8px; width: 100%;
  background: rgba(10,14,22,.92);
  border-bottom: 1px solid rgba(120,160,255,.28);
  box-shadow: 0 2px 22px rgba(0,0,0,.65);
}
.loadbar > i {
  display: block; height: 100%; position: relative;
  background: linear-gradient(90deg, #4096ff, #a06eff, #00e0c8, #ff50b4, #4096ff);
  background-size: 300% 100%;
  animation: barShine 1.8s linear infinite;
  box-shadow: 0 0 20px rgba(120,150,255,1), 0 0 44px rgba(160,110,255,.75),
              0 0 70px rgba(64,150,255,.4);
  transition: width .55s cubic-bezier(.2,.8,.2,1);
  border-radius: 0 6px 6px 0;
}
/* bright spark riding the leading edge */
.loadbar > i::after {
  content: ""; position: absolute; right: -3px; top: -3px; bottom: -3px; width: 14px;
  border-radius: 50%; background: #eaf4ff;
  box-shadow: 0 0 16px 5px rgba(180,220,255,.95), 0 0 34px 10px rgba(120,150,255,.65);
  animation: breathe 1s ease-in-out infinite;
}
.loadstage {
  align-self: flex-start; margin: 10px 0 0 18px; padding: 8px 18px; border-radius: 999px;
  font-size: 0.92rem; font-weight: 700; letter-spacing: .01em;
  color: #eaf4ff; background: rgba(12,17,26,.94);
  border: 1px solid rgba(140,180,255,.6);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 10px 34px rgba(0,0,0,.7), 0 0 30px rgba(90,150,255,.45);
  animation: stagePop .35s ease both;
  display: flex; align-items: center; gap: 12px;
}
.loadstage::before {
  content: ""; width: 12px; height: 12px; border-radius: 50%; flex: 0 0 auto;
  background: #4096ff; box-shadow: 0 0 0 0 rgba(64,150,255,.8);
  animation: pulse 1.4s infinite;
}
.loadstage .pct {
  color: #7fd0ff; font-family: 'JetBrains Mono', monospace; font-size: 1.0rem; font-weight: 800;
  text-shadow: 0 0 14px rgba(127,208,255,.8);
}


/* ---------- sidebar must scroll ----------
   The sidebar had no height constraint, so it grew to its content (2,294px on a
   768px viewport) and the view container grew with it. `.stApp` clips at the
   viewport, so everything past the fold was unreachable -- and because nothing
   in the sidebar was a scroller (scrollHeight == clientHeight), no wheel event
   had anywhere to go. Pinning both to the viewport gives the inner div
   something to scroll. */
[data-testid="stAppViewContainer"] {
  height: 100vh !important; max-height: 100vh !important; overflow: hidden !important;
}
[data-testid="stSidebar"] {
  height: 100vh !important; max-height: 100vh !important; align-self: flex-start !important;
}
[data-testid="stSidebar"] > div {
  height: 100% !important; max-height: 100vh !important; overflow-y: auto !important;
}
[data-testid="stSidebar"] > div::-webkit-scrollbar { width: 10px; }
[data-testid="stSidebar"] > div::-webkit-scrollbar-thumb {
  border-radius: 999px; border: 2px solid transparent; background-clip: padding-box;
  background-color: rgba(120,160,255,.45);
}

</style>
"""


def inject_css():
    # Flattened: a blank line inside <style> would end Streamlit's HTML block early.
    st.markdown(_h(CSS), unsafe_allow_html=True)


def _h(html: str) -> str:
    """Strip indentation -- Streamlit's markdown renders indented HTML as a code block."""
    # Join with a space so prose split across lines keeps its word breaks.
    return " ".join(line.strip() for line in html.splitlines() if line.strip())


def render(html: str):
    st.markdown(_h(html), unsafe_allow_html=True)



def boot_fx():
    """
    The JS half of the effects: count-up numbers and a cursor glow.

    Runs inside a zero-height component iframe and reaches the app through
    window.parent.document, because st.markdown strips <script>. Safe to call
    once per run; it re-arms itself when Streamlit swaps DOM nodes.
    """
    import streamlit.components.v1 as components
    components.html(
        r"""
<script>
(function () {
  const doc = window.parent.document;
  if (!doc) return;

  /* ---------- cursor glow ---------- */
  if (!doc.getElementById('fx-cursor')) {
    const g = doc.createElement('div');
    g.id = 'fx-cursor';
    g.style.cssText = [
      'position:fixed', 'width:460px', 'height:460px', 'border-radius:50%',
      'pointer-events:none', 'z-index:0', 'opacity:0',
      'transition:opacity .45s ease',
      'background:radial-gradient(circle,rgba(110,170,255,.16),rgba(160,110,255,.07) 42%,transparent 68%)',
      'transform:translate(-50%,-50%)', 'will-change:left,top'
    ].join(';');
    doc.body.appendChild(g);
    doc.addEventListener('mousemove', function (e) {
      g.style.left = e.clientX + 'px';
      g.style.top = e.clientY + 'px';
      g.style.opacity = '1';
    }, { passive: true });
    doc.addEventListener('mouseleave', function () { g.style.opacity = '0'; });
  }

  /* ---------- count-up ---------- */
  const NUM = /^([^0-9\-+]*)([+\-]?[0-9][0-9,]*\.?[0-9]*)(.*)$/;
  const ease = t => 1 - Math.pow(1 - t, 3);

  function animate(el) {
    if (el.dataset.fxCounted) return;
    const raw = (el.textContent || '').trim();
    const m = raw.match(NUM);
    if (!m) { el.dataset.fxCounted = '1'; return; }
    const pre = m[1], body = m[2], post = m[3];
    const target = parseFloat(body.replace(/,/g, ''));
    if (!isFinite(target) || Math.abs(target) > 1e15) { el.dataset.fxCounted = '1'; return; }
    el.dataset.fxCounted = '1';
    const dot = body.indexOf('.');
    const dp = dot < 0 ? 0 : body.length - dot - 1;
    const grouped = body.indexOf(',') >= 0;
    const fmt = v => {
      let t = Math.abs(v).toFixed(dp);
      if (grouped) {
        const parts = t.split('.');
        parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        t = parts.join('.');
      }
      return (v < 0 ? '-' : (body[0] === '+' ? '+' : '')) + t;
    };
    const DUR = 900, t0 = performance.now();
    el.classList.add('counted');
    function step(now) {
      const k = Math.min((now - t0) / DUR, 1);
      el.textContent = pre + fmt(target * ease(k)) + post;
      if (k < 1) requestAnimationFrame(step);
      else el.textContent = raw;
    }
    requestAnimationFrame(step);
  }

  /* ---------- lift the loading bar out of Streamlit's stacking contexts ----------
     `.stApp > * { z-index: 1 }` gives every Streamlit container its own stacking
     context, so a fixed child's z-index only ranks it inside that container and a
     later sibling paints over it (elementFromPoint at the top returned DIV.hero).
     Re-parenting the bar to <body> puts it above everything, for good. */
  function liftBar() {
    doc.querySelectorAll('.loadwrap').forEach(function (el) {
      if (el.parentElement !== doc.body) doc.body.appendChild(el);
    });
  }
  liftBar();

  const SEL = '.kpi-value, [data-testid="stMetricValue"] div, .hero-price, .snap .v';
  function scan() {
    doc.querySelectorAll(SEL).forEach(el => {
      if (el.dataset.fxSeen) return;
      el.dataset.fxSeen = '1';
      io.observe(el);
    });
  }
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { animate(e.target); io.unobserve(e.target); } });
  }, { threshold: 0.25 });

  scan();
  new MutationObserver(function () { scan(); liftBar(); })
    .observe(doc.body, { childList: true, subtree: true });
})();
</script>
        """,
        height=0,
    )


def progress_html(pct: int, label: str, done: bool = False) -> str:
    """The top loading bar. `pct` 0-100, `label` names the stage in flight."""
    pct = max(0, min(100, int(pct)))
    cls = "loadwrap done" if done else "loadwrap"
    tail = "" if done else f'<div class="loadstage">{label}<span class="pct">{pct}%</span></div>'
    return (f'<div class="{cls}"><div class="loadbar"><i style="width:{pct}%"></i></div>{tail}</div>')

def brand():
    render(f"""
    <div class="brand"><div class="brand-logo">S</div>
    <div><div class="brand-name">{APP_NAME}</div><div class="brand-sub">{APP_TAGLINE}</div></div></div>
    """)


def hero(name, ticker, chips, price, change_abs, change_pct, market_open, as_of, cur):
    direction = "up" if change_pct >= 0 else "down"
    arrow = "▲" if change_pct >= 0 else "▼"
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips if c)
    status = ('<span class="dot live"></span>Market open · live quote' if market_open
              else f'<span class="dot"></span>Market closed · last close {as_of}')
    render(f"""
    <div class="hero">
      <div class="hero-left">
        <div class="hero-eyebrow"><span class="tick">{ticker}</span>{chips_html}</div>
        <h1 class="hero-title">{name}</h1>
        <div class="hero-sub">AI ensemble forecasts · candlestick reading · technicals · news sentiment</div>
      </div>
      <div class="hero-right">
        <div class="price">{cur}{price:,.2f}</div>
        <div class="chg {direction}">{arrow} {change_abs:+,.2f} ({change_pct:+.2f}%)</div>
        <div class="status">{status}</div>
      </div>
    </div>
    """)


def range_bar_html(label, low, high, value, cur):
    pct = 50 if high == low else max(0, min(100, (value - low) / (high - low) * 100))
    return f"""
    <div class="kpi wide"><div class="kpi-label">{label}</div>
    <div class="range-track"><div class="range-marker" style="left:{pct:.1f}%"></div></div>
    <div class="range-labels"><span>{cur}{low:,.2f}</span><span>{pct:.0f}% of range</span><span>{cur}{high:,.2f}</span></div></div>
    """


def kpi_html(label, value, sub=None, tone="", small=False):
    """small=True for wordy values (asset class, fund category) that would
    otherwise be cut off by the monospace KPI font."""
    sub_html = f'<div class="kpi-sub {tone}">{sub}</div>' if sub else ""
    vcls = "kpi-value sm" if small else "kpi-value"
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="{vcls}">{value}</div>{sub_html}</div>'


def kpi_grid(items_html):
    render(f'<div class="kpi-grid">{"".join(items_html)}</div>')


def snap_html(label, value, sub="", tone="neutral", icon=""):
    """tone: up / down / warn / info / neutral. `icon` should be an arrow/symbol so meaning isn't color-only."""
    ic_cls = {"up": "ic-up", "down": "ic-down"}.get(tone, "ic-flat")
    icon_html = f'<span class="{ic_cls}">{icon}</span> ' if icon else ""
    return (f'<div class="snap {tone}"><div class="snap-label">{label}</div>'
            f'<div class="snap-value">{icon_html}{value}</div><div class="snap-sub">{sub}</div></div>')


def snap_grid(items_html):
    render(f'<div class="snap-grid">{"".join(items_html)}</div>')


def section(kicker, title, sub=""):
    sub_html = f'<span class="section-sub">{sub}</span>' if sub else ""
    render(f'<div class="section"><span class="section-kicker">{kicker}</span>'
           f'<span class="section-title">{title}</span>{sub_html}</div>')


def signal_badge(signal, pct_change=None):
    color, bg, icon = SIGNAL_STYLE[signal]
    pct = f"&nbsp;({pct_change * 100:+.2f}%)" if pct_change is not None else ""
    return (f'<div class="signal-badge" style="background:{bg};color:{color};border:1px solid {color}55;">'
            f'{icon} {signal}{pct}</div>')


def footer():
    render(f"""
    <div class="footer"><span>⚡ {APP_NAME} · research & backtesting platform — not financial or investment advice.</span>
    <span>Data: Yahoo Finance · GDELT · Google News</span></div>
    """)


def ladder(rungs, cur):
    """rungs: (label, price, pct_change, lower, upper) or the same plus
    (prob_up, swing_pct) -- the chance of ending above today's price and how
    wide the 80% range is, which is what carries the information at longer
    horizons where the central forecast is close to flat."""
    cells = []
    for rung in rungs:
        label, price, pct, lo, hi = rung[:5]
        prob_up, swing = (rung[5], rung[6]) if len(rung) >= 7 else (None, None)
        tone = "up" if pct > 0.0005 else "down" if pct < -0.0005 else "flat"
        arrow = {"up": "▲", "down": "▼", "flat": "●"}[tone]
        extra = ""
        if prob_up is not None:
            p_tone = "up" if prob_up >= 53 else "down" if prob_up <= 47 else "flat"
            extra = (f'<div class="rung-prob {p_tone}">{prob_up:.0f}% chance higher than today</div>'
                     f'<div class="rung-swing">typical swing ±{swing:.1f}%</div>')
        cells.append(f'<div class="rung"><div class="rung-h">{label}</div>'
                     f'<div class="rung-p">{cur}{price:,.2f}</div>'
                     f'<div class="rung-c {tone}">{arrow} {pct * 100:+.2f}%</div>'
                     f'{extra}'
                     f'<div class="rung-r">80%: {cur}{lo:,.0f} – {cur}{hi:,.0f}</div></div>')
    render(f'<div class="ladder">{"".join(cells)}</div>')


def group_chips(label, states):
    """states: list of (name, state) with state in {'on', 'off', 'na'}."""
    icon = {"on": "✓", "off": "✕", "na": "–"}
    chips = "".join(f'<span class="gchip {st_}">{icon[st_]} {name}</span>' for name, st_ in states)
    render(f'<div class="gchips"><span class="gchips-label">{label}</span>{chips}</div>')


def forecast_cone(fig, path, color=ACCENT, name="Forecast", paths=None, n_show=60):
    """
    Add a forecast line + shaded 80% range to a Plotly figure.

    paths : optional (n_sims x n_days) array of simulated prices. When given, a
            sample of them is drawn behind the cone. This matters because the
            centre line is usually near-flat -- the models are only trusted as far
            as they beat "no change", and for a liquid symbol that is barely -- and
            a flat line on its own reads as "nothing happens". The paths show what
            the model is really saying: plenty of movement, no callable direction.
    """
    if paths is not None and len(paths):
        # A graded fan, not spaghetti: three nested ribbons carry the spread, its
        # growth and its asymmetry in three shapes rather than sixty lines.
        n = len(path)
        sims = _np.asarray(paths)[:, :n]
        for lo, hi, shade, label in ((2.5, 97.5, 0.07, "95% of outcomes"),
                                     (10.0, 90.0, 0.11, "80% of outcomes"),
                                     (25.0, 75.0, 0.16, "50% of outcomes")):
            y_lo = _np.percentile(sims, lo, axis=0)
            y_hi = _np.percentile(sims, hi, axis=0)
            fig.add_trace(go.Scatter(
                x=list(path.index) + list(path.index[::-1]),
                y=list(y_hi) + list(y_lo[::-1]),
                fill="toself", fillcolor=f"rgba(90,150,225,{shade})",
                line=dict(width=0), hoverinfo="skip", name=label))
    if paths is None or not len(paths):
        # Only when there is no simulated fan -- otherwise the 80% ribbon above
        # already shows this and two overlapping bands just muddy the picture.
        x = list(path.index) + list(path.index[::-1])
        y = list(path["Upper_80"]) + list(path["Lower_80"][::-1])
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor="rgba(57,135,229,0.16)",
                                 line=dict(width=0), hoverinfo="skip", name="80% likely range"))
    fig.add_trace(go.Scatter(x=path.index, y=path["Predicted_Close"], name=name,
                             line=dict(color=color, width=2.5, dash="dash")))
    return fig


def tape_html(items):
    """items: list of (name, value_str, pct_change or None)."""
    cells = []
    for name, val, pct in items:
        if pct is None:
            chg = ""
        else:
            tone = "up" if pct > 0 else "down" if pct < 0 else "flat"
            arrow = {"up": "▲", "down": "▼", "flat": "●"}[tone]
            chg = f'<span class="tape-chg {tone}">{arrow} {pct * 100:+.2f}%</span>'
        cells.append(f'<div class="tape-item"><span class="tape-name">{name}</span>'
                     f'<span class="tape-val">{val}</span>{chg}</div>')
    track = "".join(cells) * 2  # duplicated so the scrolling loop is seamless
    return _h(f'<div class="tape"><div class="tape-track">{track}</div></div>')


def usage_matrix(rows, horizons, alphas):
    """
    rows: list of (group label, [state per horizon]) with state in {'on','off','na'}
    horizons: column labels; alphas: signal strength per horizon (shown as a footer row).
    """
    head = "".join(f"<th>{h}</th>" for h in horizons)
    body = []
    for label, states in rows:
        cells = "".join(
            '<td><span class="on">✓</span></td>' if st_ == "on" else
            '<td><span class="off">✕</span></td>' if st_ == "off" else
            '<td><span class="na">n/a</span></td>' for st_ in states)
        body.append(f"<tr><td>{label}</td>{cells}</tr>")
    alpha_cells = "".join(f"<td>α {a:.2f}</td>" for a in alphas)
    body.append(f'<tr class="alpha"><td>Signal strength</td>{alpha_cells}</tr>')
    render(f'<table class="umx"><thead><tr><th>Data group</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>')


import re as _re


def _md_bold(text):
    return _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text or "")


def score_color(score):
    """(color, tag) -- the tag text means the rating isn't carried by color alone."""
    if score is None:
        return TEXT_3, "n/a"
    return (UP, "Strong") if score >= 7 else (WARN, "Fair") if score >= 4.5 else (DOWN, "Weak")


def ring_html(score, label=None, big=False):
    color, tag = score_color(score)
    val = 0 if score is None else score * 10
    num = "–" if score is None else f"{score:.1f}"
    lab = f'<div class="ring-label">{label}</div><div class="ring-tag" style="color:{color}">{tag}</div>' if label else ""
    return (f'<div class="ring{" big" if big else ""}" style="--val:{val:.0f};--c:{color}">'
            f'<div class="ring-inner"><div><div class="ring-num">{num}</div><div class="ring-den">/ 10</div></div></div></div>{lab}')


def why_html(items, note=None):
    """The breakdown panel: [(metric, value, score)] -> rows with their scores."""
    rows = []
    for label, value, score in items or []:
        color, tag = score_color(score)
        sc_txt = "n/a" if score is None else f"{score:.1f}/10"
        rows.append(f'<li><span class="why-k">{label}</span><span class="why-v">{value}</span>'
                    f'<span class="why-s" style="color:{color}">{sc_txt}</span></li>')
    body = "<ul>" + "".join(rows) + "</ul>" if rows else '<div class="empty">No inputs available.</div>'
    if note:
        body += f'<div class="why-note">{note}</div>'
    return f'<div class="why-body">{body}</div>'


def rings(pillars, why=None):
    """Score rings. `why` maps a pillar -> ([(metric, value, score), ...], note);
    those cards become click-to-expand explanations of how the score was reached."""
    cards = []
    for p, sc in pillars.items():
        detail = (why or {}).get(p)
        if not detail:
            cards.append(f'<div class="ring-card">{ring_html(sc, p)}</div>')
            continue
        items, note = detail if isinstance(detail, tuple) else (detail, None)
        cards.append(f'<details class="ring-card why"><summary>{ring_html(sc, p)}'
                     f'<div class="why-hint"></div></summary>{why_html(items, note)}</details>')
    render(f'<div class="rings">{"".join(cards)}</div>')


def tldr(body_html, tags=None):
    """The summary card: a sentence per thing that matters, plus chips."""
    chips = "".join(f'<span class="tldr-tag {t[1]}">{t[0]}</span>' for t in (tags or []))
    render(f'<div class="tldr"><div class="tldr-k">⚡ The 10-second version</div>'
           f'<div class="tldr-body">{body_html}</div>'
           f'{f"<div class=\'tldr-tags\'>{chips}</div>" if chips else ""}</div>')


def verdict_banner(score, verdict, stance, summary):
    color, _ = score_color(score)
    render(f"""
    <div class="verdict">{ring_html(score, big=True)}
    <div style="flex:1;min-width:260px"><div class="verdict-kicker">🧾 Research analyst verdict</div>
    <div class="verdict-title" style="color:{color}">{verdict}</div>
    <div class="verdict-stance">{stance}</div>
    <div class="verdict-text">{_md_bold(summary)}</div></div></div>
    """)


def bull_bear(strengths, risks):
    def items(lst, empty):
        if not lst:
            return f'<div class="empty">{empty}</div>'
        return "<ul>" + "".join(f'<li><span class="tag">{x["pillar"]}</span><span>{x["text"]}</span></li>' for x in lst) + "</ul>"
    render(f"""
    <div class="rgrid">
    <div class="rcard bull"><h4>▲ Strengths — the bull case</h4>{items(strengths, "No standout strengths by these measures.")}</div>
    <div class="rcard bear"><h4>▼ Risks — the bear case</h4>{items(risks, "No major red flags by these measures.")}</div>
    </div>
    """)


def card(title, body_html):
    render(f'<div class="rcard"><h4>{title}</h4>{body_html}</div>')


def watch_html(items):
    if not items:
        return '<div class="empty">Nothing specific flagged.</div>'
    return "<ul>" + "".join(f"<li><span>{_md_bold(i)}</span></li>" for i in items) + "</ul>"


def facts(pairs):
    cells = "".join(f'<div class="fact"><div class="fact-k">{k}</div><div class="fact-v">{v}</div></div>' for k, v in pairs)
    render(f'<div class="facts">{cells}</div>')


def live_header(is_live, price, change, change_pct, stats, cur, updated):
    badge = ('<span class="live-badge on"><span class="live-dot"></span>LIVE</span>' if is_live
             else '<span class="live-badge off">● MARKET CLOSED · LAST SESSION</span>')
    tone = "up" if change_pct >= 0 else "down"
    arrow = "▲" if change_pct >= 0 else "▼"
    stats_html = "".join(f'<div class="live-stat"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in stats)
    render(f"""
    <div class="live-head"><div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">{badge}
    <span class="live-price">{cur}{price:,.2f}</span><span class="chg {tone}">{arrow} {change:+,.2f} ({change_pct * 100:+.2f}%)</span></div>
    <div class="live-stats">{stats_html}<div class="live-stat"><div class="k">Updated</div><div class="v">{updated}</div></div></div></div>
    """)
