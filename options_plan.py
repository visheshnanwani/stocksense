"""
options_plan.py
---------------
Same-day trade plan: direction, entry, stop-loss, target -- and the measured
track record of that exact plan, rather than a promise about it.

WHAT IS REAL HERE AND WHAT IS NOT
---------------------------------
The plan is built on the UNDERLYING, because that is what can be measured. Entry,
stop and target come from this symbol's own intraday behaviour (ATR, and how often
it has historically travelled that far from the open), and the whole bracket is
then replayed over every session in the history to see what it would actually have
done.

Option chains are a different matter. Yahoo serves them for US symbols and NOT for
Indian ones -- HDFCBANK.NS, RELIANCE.NS and NIFTY all return zero expiries, and
NSE's own option-chain endpoint is closed to public scraping. So:

    US symbols    real strikes, real bid/ask, real break-even
    Indian        the underlying plan plus a strike-SELECTION rule, and no
                  premium numbers at all, because inventing them would be lying

NO GUARANTEES
-------------
Nothing here guarantees a return, and the module deliberately has no function that
claims one. What it reports instead is the measured record of the rule: how often
the target was reached before the stop, the average win, the average loss, the
expectancy per trade after costs, and the worst losing run. Those are facts about
the past with a sample size attached.

THE TIE-BREAK, STATED PLAINLY
-----------------------------
A daily candle records the session's High and Low but not their ORDER, so when a
session touched both the target and the stop it is impossible to know from daily
data which came first. Every such session is counted as a LOSS. That is the
pessimistic convention, it makes the reported record a floor rather than a
flattering estimate, and the share of sessions it affects is reported as
`ambiguous_pct` so the size of the assumption is visible.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- levels
def atr(daily: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = daily["High"].astype(float), daily["Low"].astype(float), daily["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_plan(daily: pd.DataFrame, direction: str = "long", stop_atr: float = 0.6,
               target_atr: float = 1.0, entry: float = None) -> dict:
    """
    Entry / stop / target for the next session, sized in ATR rather than in round
    numbers, so the distances scale with how much this symbol actually moves.
    """
    if daily is None or len(daily) < 30:
        return {}
    a = float(atr(daily).iloc[-1])
    px = float(entry if entry is not None else daily["Close"].iloc[-1])
    if not np.isfinite(a) or a <= 0:
        return {}
    if direction == "long":
        stop, target = px - stop_atr * a, px + target_atr * a
    else:
        stop, target = px + stop_atr * a, px - target_atr * a
    risk = abs(px - stop)
    reward = abs(target - px)
    return {
        "direction": direction, "entry": px, "stop": stop, "target": target,
        "atr": a, "atr_pct": a / px * 100,
        "risk_pct": risk / px * 100, "reward_pct": reward / px * 100,
        "rr": reward / risk if risk > 0 else None,
        "stop_atr": stop_atr, "target_atr": target_atr,
    }


# ----------------------------------------------------------------- record
def backtest_bracket(daily: pd.DataFrame, direction: str = "long", stop_atr: float = 0.6,
                     target_atr: float = 1.0, cost_pct: float = 0.30,
                     lookback: int = None) -> dict:
    """
    Replay the same bracket over every session: enter at the open, exit at the
    target, the stop, or the close -- whichever the session delivers.

    Sessions that touched BOTH target and stop are counted as losses (see the
    module docstring). `ambiguous_pct` says how many sessions that was.
    """
    if daily is None or len(daily) < 120:
        return {}
    d = daily.tail(lookback) if lookback else daily
    a = atr(d).shift(1)                      # yesterday's ATR: known at the open
    o = d["Open"].astype(float)
    h = d["High"].astype(float)
    l = d["Low"].astype(float)
    c = d["Close"].astype(float)
    ok = a.notna() & (a > 0)
    o, h, l, c, a = o[ok], h[ok], l[ok], c[ok], a[ok]
    if len(o) < 100:
        return {}

    long = direction == "long"
    stop = o - stop_atr * a if long else o + stop_atr * a
    tgt = o + target_atr * a if long else o - target_atr * a

    hit_t = (h >= tgt) if long else (l <= tgt)
    hit_s = (l <= stop) if long else (h >= stop)
    both = hit_t & hit_s

    # Gross return of each session, in percent of the entry.
    # Long:  profit = (exit - entry)/entry.   Short: profit = (entry - exit)/entry.
    exit_px = pd.Series(np.nan, index=o.index)
    win = hit_t & ~hit_s
    lose = hit_s                                   # includes `both`: pessimistic
    flat = ~hit_t & ~hit_s
    exit_px[win] = tgt[win]
    exit_px[lose] = stop[lose]
    exit_px[flat] = c[flat]
    ret = ((exit_px / o - 1) if long else (1 - exit_px / o)) * 100
    net = ret - cost_pct

    n = int(len(net))
    wins, losses = net[net > 0], net[net <= 0]
    # worst losing streak
    worst_streak, cur = 0, 0
    for v in net:
        cur = cur + 1 if v <= 0 else 0
        worst_streak = max(worst_streak, cur)
    equity = net.cumsum()
    dd = float((equity - equity.cummax()).min())

    return {
        "sessions": n, "direction": direction,
        "target_hit_pct": float(win.mean() * 100),
        "stop_hit_pct": float(lose.mean() * 100),
        "neither_pct": float(flat.mean() * 100),
        "ambiguous_pct": float(both.mean() * 100),
        "win_rate": float((net > 0).mean() * 100),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_pct": float(net.mean()),
        "median_pct": float(net.median()),
        "best_pct": float(net.max()), "worst_pct": float(net.min()),
        "worst_losing_streak": int(worst_streak),
        "total_pct": float(net.sum()), "max_drawdown_pct": dd,
        "cost_pct": cost_pct,
        "stop_atr": stop_atr, "target_atr": target_atr,
    }


def sweep_brackets(daily: pd.DataFrame, direction: str = "long", cost_pct: float = 0.30,
                   stops=(0.4, 0.6, 0.8, 1.0), targets=(0.6, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Every stop/target pair, scored on expectancy. Shows the trade-off instead of
    asserting one setting is right."""
    rows = []
    for s in stops:
        for t in targets:
            r = backtest_bracket(daily, direction, s, t, cost_pct)
            if r:
                rows.append({"Stop (ATR)": s, "Target (ATR)": t,
                             "Win rate": r["win_rate"], "Avg win": r["avg_win"],
                             "Avg loss": r["avg_loss"], "Expectancy": r["expectancy_pct"],
                             "Worst streak": r["worst_losing_streak"],
                             "Ambiguous": r["ambiguous_pct"]})
    return pd.DataFrame(rows).sort_values("Expectancy", ascending=False) if rows else pd.DataFrame()


# ----------------------------------------------------------------- options
def option_chain(ticker: str, spot: float, expiry: str = None) -> dict:
    """
    Live chain if the provider has one. Returns {"available": False, "reason": ...}
    rather than a guess when it does not -- which is every Indian symbol.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        exps = list(tk.options or [])
    except Exception as e:
        return {"available": False, "reason": f"chain lookup failed ({type(e).__name__})"}
    if not exps:
        return {"available": False,
                "reason": "the data provider publishes no option chain for this symbol "
                          "(this is the case for every NSE/BSE stock, and NSE's own "
                          "option-chain endpoint is closed to public access)"}
    exp = expiry or exps[0]
    try:
        ch = tk.option_chain(exp)
    except Exception as e:
        return {"available": False, "reason": f"chain fetch failed ({type(e).__name__})"}

    def near(df, kind):
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        d["distance"] = (d["strike"] - spot).abs()
        d = d.nsmallest(5, "distance").sort_values("strike")
        keep = [c for c in ("strike", "lastPrice", "bid", "ask", "volume",
                            "openInterest", "impliedVolatility") if c in d.columns]
        d = d[keep]
        mid = ((df.set_index("strike").get("bid", pd.Series(dtype=float)) +
                df.set_index("strike").get("ask", pd.Series(dtype=float))) / 2)
        d["mid"] = d["strike"].map(mid)
        d["break_even"] = d["strike"] + d["mid"] if kind == "call" else d["strike"] - d["mid"]
        d["moneyness"] = np.where(
            (d["strike"] < spot) if kind == "call" else (d["strike"] > spot), "ITM",
            np.where(np.isclose(d["strike"], spot, rtol=0.005), "ATM", "OTM"))
        return d

    return {"available": True, "expiry": exp, "expiries": exps[:6],
            "calls": near(ch.calls, "call"), "puts": near(ch.puts, "put")}


def strike_rule(spot: float, plan: dict, step: float = None) -> dict:
    """
    Which strike to use when there is no chain to read.

    The rule is the standard one and it is stated rather than hidden: for a
    same-day move, the strike nearest the spot (ATM) tracks the underlying most
    closely, and one step out-of-the-money gives more leverage at the cost of
    needing the move to actually arrive. The target move decides whether OTM is
    even reachable.
    """
    if step is None:                       # typical NSE strike spacing
        step = 20.0 if spot >= 1000 else 10.0 if spot >= 500 else 5.0 if spot >= 200 else 2.5
    atm = round(spot / step) * step
    long = plan.get("direction") == "long"
    otm = atm + step if long else atm - step
    reach = abs(plan.get("target", spot) - spot)
    return {
        "kind": "CALL" if long else "PUT",
        "step": step, "atm": atm, "one_otm": otm,
        "otm_reachable": bool(reach >= step),
        "note": ("the target move is larger than one strike step, so one-step OTM is "
                 "reachable within the session" if reach >= step else
                 "the target move is smaller than one strike step — an OTM strike would "
                 "need a bigger move than this plan expects, so ATM is the sane choice"),
    }
