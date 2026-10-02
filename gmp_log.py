"""
gmp_log.py
----------
A record of what the grey market was quoting, kept day by day.

WHY THIS FILE EXISTS
    "How reliable is GMP?" needs two numbers per IPO: the premium quoted while
    it was still open, and the price it actually listed at. The second is
    published forever (ipowatch.in's performance tracker, 400+ IPOs). The first
    is not published at all once an IPO lists -- the live board simply drops it,
    and no source on the site keeps a history of it.

    That column used to arrive for free: IPOWatch's old GMP page carried a
    combined table with both. They removed it, which is what left the dashboard
    showing "Not enough history to measure GMP reliability" -- it was reading a
    table that no longer exists.

    Backfilling it is not possible honestly. Guessing a past GMP from the
    listing gain would measure nothing but the guess. So this logs the live
    board every time the IPO page loads, and reliability is computed from the
    LAST premium seen before each IPO listed, joined to its real listing gain.

    That means the statistic starts empty and fills in as IPOs list. It is slow,
    and it is the only version of the number that is actually true.

The log is append-only, one row per IPO per day, stored as CSV next to the code.
"""

import datetime
import os

import numpy as np
import pandas as pd

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gmp_history.csv")
COLUMNS = ["date", "name", "type", "gmp", "price", "est_gain_pct", "status"]


def load() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(LOG_PATH, parse_dates=["date"])
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    return df.reindex(columns=COLUMNS)


def record(live: pd.DataFrame, today=None) -> int:
    """
    Append today's board. Returns the number of rows added.

    One row per IPO per day: re-running the page does not inflate the log, and
    the last row for a name before it lists is the premium the market was
    quoting at the end -- which is the one the reliability question is about.
    """
    if live is None or live.empty or "name" not in live.columns:
        return 0
    today = pd.Timestamp(today or datetime.date.today()).normalize()
    snap = pd.DataFrame({
        "date": today,
        "name": live["name"].astype(str).str.strip(),
        "type": live.get("type", pd.Series(index=live.index, dtype=object)),
        "gmp": pd.to_numeric(live.get("gmp"), errors="coerce"),
        "price": pd.to_numeric(live.get("price"), errors="coerce"),
        "est_gain_pct": pd.to_numeric(live.get("est_gain_pct"), errors="coerce"),
        "status": live.get("status", pd.Series(index=live.index, dtype=object)),
    }).dropna(subset=["name"])
    snap = snap[snap["name"] != ""]
    if snap.empty:
        return 0

    old = load()
    both = pd.concat([old, snap], ignore_index=True)
    both["date"] = pd.to_datetime(both["date"], errors="coerce")
    both = both.dropna(subset=["date"])
    both = both.drop_duplicates(subset=["date", "name"], keep="last")
    both = both.sort_values(["name", "date"])
    added = len(both) - len(old)
    if added > 0:
        try:
            both.to_csv(LOG_PATH, index=False)
        except OSError:
            return 0
    return max(added, 0)


def final_gmp() -> pd.DataFrame:
    """The last premium logged for each IPO -- its closing grey-market quote."""
    log = load()
    if log.empty:
        return pd.DataFrame(columns=["name", "gmp", "price", "last_seen", "days_logged"])
    log = log.dropna(subset=["name"])
    g = log.sort_values("date").groupby("name")
    out = g.tail(1)[["name", "gmp", "price", "date"]].rename(columns={"date": "last_seen"})
    out["days_logged"] = out["name"].map(g.size())
    return out.reset_index(drop=True)


def reliability(hist: pd.DataFrame, match=None) -> dict:
    """
    Join logged premiums to real listing gains.

    `hist` is the listing-performance table (name, ipo_price, actual_gain_pct).
    `match` is an optional name matcher -- the two sources spell names slightly
    differently ("Moneyview" vs "Moneyview Limited").
    """
    fin = final_gmp()
    if fin.empty or hist is None or hist.empty:
        return {"matched": 0, "logged": len(fin)}
    h = hist.dropna(subset=["actual_gain_pct", "ipo_price"]).copy()
    names = list(h["name"].astype(str))

    rows = []
    for _, r in fin.iterrows():
        if pd.isna(r["gmp"]):
            continue
        hit = match(r["name"], names) if match else (r["name"] if r["name"] in names else None)
        if not hit:
            continue
        hr = h[h["name"] == hit].iloc[0]
        price = r["price"] if pd.notna(r["price"]) and r["price"] > 0 else hr["ipo_price"]
        if not price or price <= 0:
            continue
        rows.append({"name": hit, "gmp": float(r["gmp"]),
                     "predicted_gain_pct": float(r["gmp"]) / float(price) * 100,
                     "actual_gain_pct": float(hr["actual_gain_pct"]),
                     "last_seen": r["last_seen"]})

    if not rows:
        return {"matched": 0, "logged": len(fin)}
    m = pd.DataFrame(rows)
    err = m["actual_gain_pct"] - m["predicted_gain_pct"]
    pos, zero = m[m.predicted_gain_pct > 0], m[m.predicted_gain_pct <= 0]
    return {
        "matched": len(m), "logged": len(fin), "frame": m,
        "n": len(m),
        "direction_hit_rate": float(((m.predicted_gain_pct > 0) == (m.actual_gain_pct > 0)).mean()),
        "listed_up_when_gmp_positive": float((pos.actual_gain_pct > 0).mean()) if len(pos) else None,
        "listed_up_when_gmp_zero": float((zero.actual_gain_pct > 0).mean()) if len(zero) else None,
        "median_abs_error_pts": float(err.abs().median()),
        "mean_error_pts": float(err.mean()),
        "correlation": float(m.predicted_gain_pct.corr(m.actual_gain_pct)) if len(m) > 2 else np.nan,
        "avg_actual_gain": float(m.actual_gain_pct.mean()),
        "pct_listed_up": float((m.actual_gain_pct > 0).mean()),
    }
