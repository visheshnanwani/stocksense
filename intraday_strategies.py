"""
intraday_strategies.py
----------------------
Intraday rules with published support, tested properly.

Everything daily in this project failed, and the diagnosis was that next-day
direction on a liquid large cap has no edge. Intraday is the one area left with
real literature behind it:

  opening range breakout  the first N minutes set a range; a break of it tends
                          to continue (widely traded, weakly documented)
  intraday momentum       the FIRST half-hour return predicts the LAST half-hour
                          return (Gao, Han, Li & Zhou 2018, JFE) -- the best
                          evidenced of the group
  VWAP reversion          price stretched from VWAP tends to snap back
  gap fade                an opening gap tends to fill

THE DISCIPLINE THAT MAKES THE NUMBER MEAN ANYTHING
--------------------------------------------------
"Maximum profit on last week" is trivial to fake: try enough rules on last week
and one will look brilliant. So the week is never used to choose anything.

    TRAIN   the 53 sessions before last week -- every rule and every parameter
            is scored here, and the winner is picked here
    TEST    last week, traded once, blind, with the winner already fixed

The test number is therefore a real out-of-sample result. It is also five days
long, which is far too short to prove anything -- so the spread of all training
weeks is printed beside it.

Costs: 0.15% round trip (0.082% charges + slippage), the corrected intraday rate.
"""

import numpy as np
import pandas as pd

COST = 0.15
BARS_PER_SESSION = 73          # NSE 5-minute session
SESSION_MIN = 375


def sessions(df):
    """Split an intraday frame into one DataFrame per trading day."""
    for day, g in df.groupby(df.index.normalize()):
        if len(g) >= 20:
            yield day, g


def _exit_with_stop(g, entry_i, side, entry_px, stop_px, exit_i=None):
    """Walk forward from entry; stop if breached, else exit at exit_i (or close)."""
    end = exit_i if exit_i is not None else len(g) - 1
    seg = g.iloc[entry_i + 1:end + 1]
    if len(seg) == 0:
        return entry_px, "flat"
    if side == "long":
        hit = seg.index[seg["Low"] <= stop_px]
    else:
        hit = seg.index[seg["High"] >= stop_px]
    if len(hit):
        return stop_px, "stopped"
    return float(seg["Close"].iloc[-1]), "closed"


# ------------------------------------------------------------------ strategies
def orb(g, open_bars=6, stop_frac=0.5, exit_bar=None):
    """Opening range breakout: trade the first break of the first `open_bars`."""
    if len(g) < open_bars + 5:
        return None
    opening = g.iloc[:open_bars]
    hi, lo = float(opening["High"].max()), float(opening["Low"].min())
    rng = hi - lo
    if rng <= 0:
        return None
    rest = g.iloc[open_bars:]
    up = rest.index[rest["High"] >= hi]
    dn = rest.index[rest["Low"] <= lo]
    first_up = up[0] if len(up) else None
    first_dn = dn[0] if len(dn) else None
    if first_up is None and first_dn is None:
        return None
    if first_dn is None or (first_up is not None and first_up <= first_dn):
        side, entry_px = "long", hi
        entry_i = g.index.get_loc(first_up)
        stop = hi - stop_frac * rng
    else:
        side, entry_px = "short", lo
        entry_i = g.index.get_loc(first_dn)
        stop = lo + stop_frac * rng
    ex, how = _exit_with_stop(g, entry_i, side, entry_px, stop, exit_bar)
    pct = ((ex / entry_px - 1) if side == "long" else (1 - ex / entry_px)) * 100
    return {"side": side, "entry": entry_px, "exit": ex, "how": how, "gross": pct}


def intraday_momentum(g, first_bars=6, last_bars=6, stop_frac=None):
    """Gao et al.: the first half-hour's return predicts the last half-hour's."""
    if len(g) < first_bars + last_bars + 2:
        return None
    o = float(g["Open"].iloc[0])
    r1 = float(g["Close"].iloc[first_bars - 1]) / o - 1
    if r1 == 0:
        return None
    side = "long" if r1 > 0 else "short"
    entry_i = len(g) - last_bars - 1
    entry_px = float(g["Close"].iloc[entry_i])
    exit_px = float(g["Close"].iloc[-1])
    pct = ((exit_px / entry_px - 1) if side == "long" else (1 - exit_px / entry_px)) * 100
    return {"side": side, "entry": entry_px, "exit": exit_px, "how": "closed", "gross": pct}


def vwap_reversion(g, dev=1.5, entry_after=12, stop_frac=1.5):
    """Fade a stretch away from VWAP, measured in that day's own bar volatility."""
    if len(g) < entry_after + 6:
        return None
    tp = (g["High"] + g["Low"] + g["Close"]) / 3
    vol = g["Volume"].replace(0, np.nan)
    vwap = (tp * vol).cumsum() / vol.cumsum()
    bar_sd = g["Close"].pct_change().rolling(12).std()
    z = (g["Close"] / vwap - 1) / bar_sd.replace(0, np.nan)
    z = z.iloc[entry_after:]
    hit = z.index[z.abs() >= dev]
    if not len(hit):
        return None
    t0 = hit[0]
    i = g.index.get_loc(t0)
    side = "short" if z.loc[t0] > 0 else "long"
    entry_px = float(g["Close"].iloc[i])
    stop = entry_px * (1 + stop_frac / 100) if side == "short" else entry_px * (1 - stop_frac / 100)
    ex, how = _exit_with_stop(g, i, side, entry_px, stop, None)
    pct = ((ex / entry_px - 1) if side == "long" else (1 - ex / entry_px)) * 100
    return {"side": side, "entry": entry_px, "exit": ex, "how": how, "gross": pct}


def gap_fade(g, prev_close, min_gap=0.3, stop_frac=1.0):
    """Fade an opening gap back toward the prior close."""
    if prev_close is None or len(g) < 10:
        return None
    o = float(g["Open"].iloc[0])
    gap = (o / prev_close - 1) * 100
    if abs(gap) < min_gap:
        return None
    side = "short" if gap > 0 else "long"
    entry_px = o
    stop = entry_px * (1 + stop_frac / 100) if side == "short" else entry_px * (1 - stop_frac / 100)
    ex, how = _exit_with_stop(g, 0, side, entry_px, stop, None)
    pct = ((ex / entry_px - 1) if side == "long" else (1 - ex / entry_px)) * 100
    return {"side": side, "entry": entry_px, "exit": ex, "how": how, "gross": pct}


def gap_continuation(g, prev_close, min_gap=0.3, stop_frac=1.0):
    """The opposite bet: a gap keeps going."""
    r = gap_fade(g, prev_close, min_gap, stop_frac)
    if r is None:
        return None
    o = float(g["Open"].iloc[0])
    gap = (o / prev_close - 1) * 100
    side = "long" if gap > 0 else "short"
    entry_px = o
    stop = entry_px * (1 - stop_frac / 100) if side == "long" else entry_px * (1 + stop_frac / 100)
    ex, how = _exit_with_stop(g, 0, side, entry_px, stop, None)
    pct = ((ex / entry_px - 1) if side == "long" else (1 - ex / entry_px)) * 100
    return {"side": side, "entry": entry_px, "exit": ex, "how": how, "gross": pct}


# ------------------------------------------------------------------ grid
def strategy_grid():
    """Every rule and parameter to be scored on the TRAINING window."""
    g = []
    for ob in (3, 6, 12):
        for sf in (0.5, 1.0):
            g.append((f"ORB {ob * 5}min stop{sf:g}",
                      lambda s, pc, ob=ob, sf=sf: orb(s, ob, sf)))
    for fb in (3, 6, 12):
        for lb in (3, 6, 12):
            g.append((f"IntradayMom first{fb * 5}/last{lb * 5}",
                      lambda s, pc, fb=fb, lb=lb: intraday_momentum(s, fb, lb)))
    for dv in (1.0, 1.5, 2.0):
        g.append((f"VWAP revert z{dv:g}",
                  lambda s, pc, dv=dv: vwap_reversion(s, dv)))
    for mg in (0.2, 0.4):
        g.append((f"Gap fade >{mg:g}%", lambda s, pc, mg=mg: gap_fade(s, pc, mg)))
        g.append((f"Gap cont. >{mg:g}%", lambda s, pc, mg=mg: gap_continuation(s, pc, mg)))
    return g
