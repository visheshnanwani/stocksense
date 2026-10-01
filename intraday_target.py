"""
intraday_target.py
------------------
"I want 1-2%. When do I get it?"

The buy/sell window scan works on daily candles and daily forecasts, so the
shortest thing it can ever suggest is a multi-day hold. But a 1-2% move is
usually an INTRADAY event -- HDFCBANK.NS touches +/-1% of its opening price on
63% of sessions -- so asking a daily model for it is asking the wrong model.

WHAT THIS MEASURES, AND WHY IT IS MEASURED THAT WAY
---------------------------------------------------
A daily candle already contains the session's High and Low. So "did the price
reach +1% above the open at some point that day" is answerable from DAILY bars,
over the full history -- 3,148 sessions for HDFCBANK.NS rather than the 58 that
Yahoo's 15-minute feed allows. Hit rates therefore come from daily bars.

Intraday bars are used only for the one thing daily bars cannot answer: WHEN in
the session the move usually lands. That sample is small (60 days at 15m) and is
labelled as such rather than presented as equally solid.

Cross-checked on the same 58 sessions: daily-derived 51% / 36% at +0.5% / +1.0%
against 15-minute-derived 52% / 28%. The method agrees with itself.

THE HONEST PART
---------------
"A stock moves 1-2% in a day" is true of its RANGE and not of what you can
capture. Over full history HDFCBANK.NS touches:

    +/-1% from the open on 63% of sessions      <- the range claim, true
      +1% above the open on 33% of sessions     <- what a long position gets
      +2% above the open on  8% of sessions

You cannot buy the low and sell the high, so the range is the wrong number to
plan with. Everything here reports the directional figure, and every return is
shown after costs, because a 0.30% round trip takes a third of a 1% target.
"""

import numpy as np
import pandas as pd

import timeframes as tfm

# Candle to watch, by how long the move typically takes. A move that lands in
# twenty minutes is invisible on an hourly chart; one that takes all session is
# just noise on a 1-minute chart.
_CANDLE_BY_MINUTES = [(25, "5m"), (75, "15m"), (180, "30m"), (10 ** 9, "1h")]


def recommend_candle(median_minutes):
    if median_minutes is None or not np.isfinite(median_minutes):
        return "15m"
    for cap, tf in _CANDLE_BY_MINUTES:
        if median_minutes <= cap:
            return tf
    return "1h"


def session_touch_stats(daily: pd.DataFrame, target_pct: float,
                        vol_window: int = 21, vol_tol: float = 0.25) -> dict:
    """
    How often a session reaches +/-target% away from its own open.

    Computed from daily High/Low/Open, so it uses the whole history rather than
    the 60 days a provider will serve intraday.

    Also reported conditionally: only those sessions whose prior-day volatility
    was within +/-vol_tol of today's. An average over all regimes overstates a
    quiet market and understates a wild one, and the user is trading today's.
    """
    if daily is None or len(daily) < 60:
        return {}
    op = daily["Open"].astype(float)
    up = (daily["High"].astype(float) / op - 1.0) * 100
    dn = (1.0 - daily["Low"].astype(float) / op) * 100
    t = float(target_pct)

    hit_up, hit_dn = up >= t, dn >= t
    out = {
        "target_pct": t, "sessions": int(len(daily)),
        "p_up": float(hit_up.mean() * 100),
        "p_down": float(hit_dn.mean() * 100),
        "p_either": float((hit_up | hit_dn).mean() * 100),
        "median_up_move": float(up.median()),
        "median_range": float(((daily["High"] / daily["Low"] - 1) * 100).median()),
    }

    # --- same question, restricted to days that looked like today -------------
    ret = daily["Close"].astype(float).pct_change()
    vol = ret.rolling(vol_window).std().shift(1)        # shift: prior day only
    vol_now = vol.iloc[-1]
    if pd.notna(vol_now) and vol_now > 0:
        similar = (vol >= vol_now * (1 - vol_tol)) & (vol <= vol_now * (1 + vol_tol))
        n = int(similar.sum())
        if n >= 60:
            out.update({
                "similar_sessions": n,
                "p_up_similar": float(hit_up[similar].mean() * 100),
                "p_down_similar": float(hit_dn[similar].mean() * 100),
                "p_either_similar": float((hit_up | hit_dn)[similar].mean() * 100),
            })
        vol_avg = vol.tail(756).mean()
        out["vol_now_vs_typical"] = float(vol_now / vol_avg) if vol_avg and vol_avg > 0 else None
    return out


def time_to_touch(intraday: pd.DataFrame, target_pct: float, side: str = "up",
                  tz_minutes: int = None) -> dict:
    """
    WHEN in the session the move usually lands, from intraday candles.

    Small sample by construction -- a provider serves ~60 days of 15-minute bars
    -- so the session count is returned and must be shown alongside the answer.
    """
    if intraday is None or len(intraday) == 0:
        return {}
    df = intraday.copy()
    df["_day"] = df.index.normalize()
    t = float(target_pct)
    first_min, hit_days, n_days = [], 0, 0

    for _, g in df.groupby("_day"):
        if len(g) < 4:
            continue
        n_days += 1
        op = float(g["Open"].iloc[0])
        if side == "down":
            moved = (1.0 - g["Low"].astype(float) / op) * 100
        elif side == "either":
            moved = np.maximum((g["High"].astype(float) / op - 1.0) * 100,
                               (1.0 - g["Low"].astype(float) / op) * 100)
        else:
            moved = (g["High"].astype(float) / op - 1.0) * 100
        idx = np.flatnonzero(np.asarray(moved) >= t)
        if len(idx):
            hit_days += 1
            first_min.append((g.index[idx[0]] - g.index[0]).total_seconds() / 60.0)

    if not n_days:
        return {}
    res = {"sessions": n_days, "hit_rate": hit_days / n_days * 100,
           "side": side, "target_pct": t}
    if first_min:
        a = np.asarray(first_min, dtype=float)
        res.update({
            "median_minutes": float(np.median(a)),
            "p25_minutes": float(np.percentile(a, 25)),
            "p75_minutes": float(np.percentile(a, 75)),
            "fastest_minutes": float(a.min()),
        })
    return res


def days_to_touch(daily: pd.DataFrame, target_pct: float, max_days: int = 20) -> dict:
    """
    When one session is not enough: how many sessions until the price is
    target% above the entry at some point. Entry is the next session's open, so
    this is a tradeable sequence rather than a hindsight scan.
    """
    if daily is None or len(daily) < 100:
        return {}
    op = daily["Open"].astype(float).to_numpy()
    hi = daily["High"].astype(float).to_numpy()
    t = float(target_pct)
    n = len(op)
    waits, never = [], 0
    for i in range(n - 1):
        entry = op[i]
        limit = min(i + max_days, n)
        run = hi[i:limit] / entry - 1.0
        idx = np.flatnonzero(run >= t / 100.0)
        if len(idx):
            waits.append(int(idx[0]) + 1)
        else:
            never += 1
    if not waits:
        return {}
    a = np.asarray(waits)
    return {"median_days": float(np.median(a)), "p25_days": float(np.percentile(a, 25)),
            "p75_days": float(np.percentile(a, 75)),
            "within_1_day": float((a <= 1).mean() * 100),
            "within_3_days": float((a <= 3).mean() * 100),
            "within_5_days": float((a <= 5).mean() * 100),
            "never_pct": float(never / (n - 1) * 100), "max_days": max_days}


def find_target_window(ticker: str, target_pct: float, daily: pd.DataFrame = None,
                       cost_pct: float = 0.30, interval: str = None,
                       min_hit: float = 40.0) -> dict:
    """
    The whole answer for one target: how often, how long, on which candle, and
    whether it survives costs.

    min_hit : below this hit rate the same-day plan is not recommended, and the
              multi-day figures are what the user should read instead.
    """
    import data_loader as dl

    if daily is None:
        daily = dl.load_stock_data(ticker, "2014-01-01", None)
    stats = session_touch_stats(daily, target_pct)
    if not stats:
        return {"error": "not enough daily history to measure this"}

    # intraday timing: prefer 15m, fall back to whatever the provider serves
    timing, used_interval = {}, None
    for tf in ([interval] if interval else ["15m", "30m", "1h"]):
        if not tf:
            continue
        try:
            bars = dl.load_ohlcv(ticker, tf)
            timing = time_to_touch(bars, target_pct, side="up")
            if timing.get("sessions", 0) >= 15:
                used_interval = tf
                break
        except Exception:
            continue

    multi = days_to_touch(daily, target_pct)
    net = target_pct - cost_pct
    med_min = timing.get("median_minutes")
    p_up = stats.get("p_up_similar", stats.get("p_up", 0.0))

    res = {
        "ticker": ticker, "target_pct": target_pct, "cost_pct": cost_pct,
        "net_pct": net, "stats": stats, "timing": timing, "multi_day": multi,
        "timing_interval": used_interval,
        "watch_candle": recommend_candle(med_min),
        "same_day_realistic": bool(p_up >= min_hit),
        "p_up_used": p_up,
        "session_minutes": tfm.session_minutes(
            str(getattr(daily.index, "tz", None) or "")),
    }
    if net <= 0:
        res["verdict"] = "costs_exceed_target"
    elif p_up >= min_hit:
        res["verdict"] = "same_day"
    elif multi.get("within_3_days", 0) >= 50:
        res["verdict"] = "few_days"
    else:
        res["verdict"] = "patience"
    return res


def ladder(ticker: str, daily: pd.DataFrame, targets=(0.5, 1.0, 1.5, 2.0),
           cost_pct: float = 0.30, interval: str = "15m") -> pd.DataFrame:
    """One row per target: the trade-off between size of move and how long it takes."""
    import data_loader as dl

    bars = None
    try:
        bars = dl.load_ohlcv(ticker, interval)
    except Exception:
        pass
    rows = []
    for t in targets:
        st = session_touch_stats(daily, t)
        if not st:
            continue
        tm = time_to_touch(bars, t, "up") if bars is not None else {}
        md = days_to_touch(daily, t)
        rows.append({
            "Target": t,
            "Net after costs": t - cost_pct,
            "Same-day odds (long)": st.get("p_up_similar", st.get("p_up")),
            "Touches either way": st.get("p_either_similar", st.get("p_either")),
            "Typical time to hit": tm.get("median_minutes"),
            "Within 1 day": md.get("within_1_day"),
            "Within 3 days": md.get("within_3_days"),
            "Typical days": md.get("median_days"),
            "Watch candle": recommend_candle(tm.get("median_minutes")),
        })
    return pd.DataFrame(rows)
