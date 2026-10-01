"""
nse_live_options.py
-------------------
LIVE NSE option prices, during market hours.

The bhavcopy that everything else in this project uses is end-of-day only --
checked at 12:03 on 30 Sep 2026 with the market open, and that session's file
did not exist. So the backtest can be run on history but not traded from.

This endpoint is different. It serves live contracts while the market is open:

    https://www.nseindia.com/api/liveEquity-derivatives?index=nse50_opt

1,577 NIFTY contracts, stamped to the minute, with lastPrice, open, high, low,
volume, open interest and the live underlying.

WHAT IT GIVES YOU
    live last-traded price per contract
    live underlying, so the true ATM strike right now
    volume and OI, so you can see what is actually liquid today

WHAT IT DOES NOT GIVE YOU, AND WHY IT MATTERS
    **No bid and no ask.** Only the last trade. You therefore cannot know the
    spread, and the spread decides your actual fill. The backtest assumed a sell
    at the session's open price; a real order is filled at the bid, which is
    worse. The strategy's measured edge is about +0.12% of margin per session, so
    a spread of a couple of points on a straddle is a material fraction of it.
    Treat the P&L here as an upper bound on what a live fill would achieve.

    Only nse50_opt (NIFTY) responds; the BANKNIFTY and all-F&O variants return
    HTTP 500.

So: live monitoring and live strike selection, yes. Live fill prices, no.
"""

import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
LIVE = "https://www.nseindia.com/api/liveEquity-derivatives?index={idx}"
STATUS = "https://www.nseindia.com/api/marketStatus"
LOT = {"NIFTY": 75}
MARGIN_FRAC = 0.12
COST_RS = 60.0

_s = None


def session():
    global _s
    if _s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
        try:
            s.get("https://www.nseindia.com/", timeout=25)
        except Exception:
            pass
        s.headers.update({
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch"})
        _s = s
    return _s


def market_open():
    try:
        j = session().get(STATUS, timeout=20).json()
        for m in j.get("marketState", []):
            if m.get("market") == "Capital Market":
                return m.get("marketStatus") == "Open", m.get("tradeDate"), m.get("last")
    except Exception:
        pass
    return None, None, None


def live_chain(index="nse50_opt") -> pd.DataFrame:
    r = session().get(LIVE.format(idx=index), timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    j = r.json()
    df = pd.DataFrame(j.get("data", []))
    if df.empty:
        return df
    df["fetched"] = j.get("timestamp")
    for c in ("strikePrice", "lastPrice", "openPrice", "highPrice", "lowPrice",
              "change", "pChange", "volume", "openInterest", "underlyingValue"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["expiry"] = pd.to_datetime(df["expiryDate"], format="%d-%b-%Y", errors="coerce")
    df["cp"] = df["optionType"].map({"Call": "CE", "Put": "PE"})
    return df


def atm_straddle(df, symbol="NIFTY", ref_spot=None, expiry=None):
    """
    The ATM call and put for the nearest expiry.

    ref_spot : the strike is chosen from this price. Pass YESTERDAY's close to
               match the backtest exactly; leave it None to use the live spot,
               which is what you would actually see on screen.
    """
    d = df[(df.underlying == symbol) & df.cp.notna()].copy()
    if d.empty:
        return pd.DataFrame()
    exp = expiry or d.expiry.min()
    d = d[d.expiry == exp]
    spot = ref_spot if ref_spot is not None else float(d.underlyingValue.iloc[0])
    strikes = sorted(d.strikePrice.unique())
    k = min(strikes, key=lambda x: abs(x - spot))
    out = d[d.strikePrice == k].copy()
    out["ref_spot"] = spot
    out["lot"] = out.underlying.map(LOT).fillna(75).astype(int)
    return out.sort_values("cp")


def straddle_view(symbol="NIFTY", ref_spot=None):
    """One printable snapshot of the ATM straddle right now."""
    is_open, trade_date, spot_live = market_open()
    df = live_chain()
    if df.empty:
        return None
    leg = atm_straddle(df, symbol, ref_spot)
    if leg.empty:
        return None
    lot = int(leg.lot.iloc[0])
    strike = float(leg.strikePrice.iloc[0])
    prem_now = float(leg.lastPrice.sum())
    prem_open = float(leg.openPrice.sum())
    margin = float(leg.underlyingValue.iloc[0]) * lot * MARGIN_FRAC * 2
    return {
        "fetched": leg.fetched.iloc[0], "market_open": is_open, "trade_date": trade_date,
        "symbol": symbol, "expiry": leg.expiry.iloc[0], "strike": strike,
        "spot": float(leg.underlyingValue.iloc[0]), "ref_spot": float(leg.ref_spot.iloc[0]),
        "lot": lot, "legs": leg[["cp", "strikePrice", "openPrice", "lastPrice",
                                 "highPrice", "lowPrice", "volume", "openInterest"]],
        "premium_at_open": prem_open, "premium_now": prem_now,
        "pnl_if_sold_at_open": (prem_open - prem_now) * lot - 2 * COST_RS,
        "margin": margin,
    }


if __name__ == "__main__":
    v = straddle_view()
    if not v:
        print("no live data (market closed, or the endpoint refused)")
        raise SystemExit(0)
    print("=" * 74)
    print(f"LIVE ATM STRADDLE  {v['symbol']}   fetched {v['fetched']}")
    print(f"market open: {v['market_open']}   NSE says: {v['trade_date']}")
    print("=" * 74)
    print(f"  spot {v['spot']:,.2f}   strike {v['strike']:,.0f}   "
          f"expiry {v['expiry']:%d %b %Y}   lot {v['lot']}")
    print()
    print(v["legs"].to_string(index=False))
    print()
    print(f"  straddle premium at the open  {v['premium_at_open']:,.2f} pts")
    print(f"  straddle premium now          {v['premium_now']:,.2f} pts")
    print(f"  P&L if sold at the open       Rs {v['pnl_if_sold_at_open']:+,.0f}  "
          f"(on ~Rs {v['margin']:,.0f} margin)")
    print()
    print("  NOTE: lastPrice only -- this feed carries no bid/ask, so a real fill")
    print("  would be worse. Treat this as an upper bound, not an achievable price.")
