"""
market_data.py
----------------
Phases 2, 3 and 9: one gate that all price data passes through, and a hard
separation between three kinds of data that were previously allowed to mix.

    CURRENT_DATA     what the market is doing right now. Mutable, refreshed
                     every few seconds, safe to show in a header.
    HISTORICAL_DATA  what the market did up to a chosen moment. IMMUTABLE.
                     Nothing later than the as-of moment may ever enter it.
    FORECAST_DATA    what a model says comes next. Never mixed into either.

THE BUG THIS EXISTS TO KILL
---------------------------
The dashboard overlaid the live quote onto the price frame so the header would
tick in real time. That overlay appended a row for *today* whenever the frame
ended earlier -- which is exactly what happens when a historical end date is
chosen. Reproduced before the fix:

    frame ending 2026-08-14 -> close 727.00
    after live overlay      -> frame ends 2026-09-28, close 722.40

so a historical analysis silently reported today's price, and every number
built on it (KPIs, chart, the forecast's reference price, the scenario range)
inherited today's market instead of the selected date's.

The overlay is now impossible on a historical frame: `apply_live_quote_safe`
refuses unless the frame is explicitly marked as current, and the refusal is
reported rather than silent.
"""

import datetime

import numpy as np
import pandas as pd

MODE_CURRENT = "current"
MODE_HISTORICAL = "historical"


class MarketDataError(Exception):
    """Raised when data fails validation. Never swallowed into a substitution."""


# Above this share of malformed candles the feed itself is suspect rather than
# noisy, and repairing would be inventing data instead of tidying it. FX is the
# reason it is not tighter: Yahoo has no consolidated OHLC feed for currency
# pairs or thin futures, so USDJPY=X (6%), CT=F (12%), JPYINR=X (14%) and KC=F
# (16%) all arrive needing a tidy-up while their Close series are clean -- no
# NaNs, no absurd returns. Since the forecast is built on Close, refusing those
# markets would deny a working analysis over a defect in a few indicators.
MAX_REPAIR_SHARE = 0.25

# Above this rate the repair is still safe, but the candle bodies were unreliable
# often enough that High/Low-derived measures should be treated with suspicion.
NOISY_OHLC_SHARE = 0.05

# Instruments where a price at or below zero is a real market event rather than
# corrupt data. WTI settled at -$37.63 on 20 April 2020.
NEGATIVE_OK_SUFFIX = ("=F",)


# ------------------------------------------------------------------------
# Is the intraday RANGE of each bar trustworthy?
# ------------------------------------------------------------------------
# Found on 01 Oct 2026, and it was producing a spectacular false result.
#
# A pooled model scored 80%+ directional accuracy on every freely-floating
# currency pair -- which is not a thing that happens on daily FX. The cause is
# that Yahoo's daily FX bars are malformed. Spot FX trades 24 hours, and the
# High/Low it reports span a window that does not end where the Close is
# stamped, so:
#
#     EURUSD=X   Open == Close on 55.8% of bars   (AAPL: 0.1%)
#                Close outside its own High/Low on 2.8%   (AAPL: 0.0%)
#                TOMORROW's close inside TODAY's range on 75.3%   (AAPL: 45.2%)
#
# That last line is the leak stated plainly: the bar's range already contains
# the next day's price. Every feature derived from High/Low therefore carries a
# piece of the answer, and the strongest of them, S_Lower_Wick_Pct, correlated
# 0.490 with the NEXT day's return on FX against 0.047 for the best equity
# feature. Nothing was wrong with the model; it was reading the future out of
# the data.
#
# A second, milder defect shows up on thin futures (PL=F, PA=F, ALI=F): Open
# equals Close on 69-90% of bars and the range is stale rather than forward
# looking. That one does not leak, but the range features are noise, so they
# are dropped as well.
#
# Close-only features (returns, momentum, moving averages, volatility of
# returns) are unaffected either way and keep working.

RANGE_LEAK_NEXT_IN_RANGE = 0.62   # vs ~0.45 on a well-formed daily bar
RANGE_LEAK_CLOSE_OUTSIDE = 0.01   # a close outside its own high/low is impossible
# Either symptom alone is enough once it is this extreme. The first version of
# this check required BOTH, and a frame that spliced Yahoo's daily close onto a
# high/low rebuilt from hourly bars slipped through it: the splice has no close
# outside its own range by construction, so the AND was never satisfied even
# though the next day's close landed inside the range 94% of the time. Two
# clocks in one bar leak worse than either clock alone.
RANGE_LEAK_NEXT_IN_RANGE_HARD = 0.80
RANGE_DEGENERATE_FLAT = 0.25      # open == close this often means there is no real range


def ohlc_range_quality(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Decide whether the High/Low of each bar can be used as a feature.

    Returns {"verdict": "ok" | "leaking" | "degenerate", "reason": str, ...}.
    "leaking" means the range extends past the close and must never be used.
    """
    out = {"verdict": "ok", "reason": "", "flat_share": 0.0,
           "close_outside_share": 0.0, "next_in_range_share": 0.0, "n": int(len(df))}
    need = {"Open", "High", "Low", "Close"}
    if df is None or len(df) < 120 or not need.issubset(df.columns):
        return out

    o, h, l, c = (pd.to_numeric(df[k], errors="coerce") for k in ("Open", "High", "Low", "Close"))
    ok = o.notna() & h.notna() & l.notna() & c.notna()
    if ok.sum() < 120:
        return out
    o, h, l, c = o[ok], h[ok], l[ok], c[ok]

    eps = (h - l).abs().median() * 1e-6 + 1e-12
    out["flat_share"] = float(np.isclose(o, c, rtol=0, atol=eps).mean())
    out["close_outside_share"] = float(((c > h + eps) | (c < l - eps)).mean())
    nxt = c.shift(-1)
    inside = ((nxt <= h + eps) & (nxt >= l - eps))
    out["next_in_range_share"] = float(inside[:-1].mean())

    if (out["next_in_range_share"] >= RANGE_LEAK_NEXT_IN_RANGE_HARD
            or (out["next_in_range_share"] >= RANGE_LEAK_NEXT_IN_RANGE
                and out["close_outside_share"] >= RANGE_LEAK_CLOSE_OUTSIDE)):
        out["verdict"] = "leaking"
        out["reason"] = (f"the next day's close falls inside this bar's high/low "
                         f"{out['next_in_range_share'] * 100:.0f}% of the time and the close "
                         f"sits outside its own high/low on {out['close_outside_share'] * 100:.1f}% "
                         f"of bars -- the range extends past the close, so anything built "
                         f"from it can see ahead")
    elif out["flat_share"] >= RANGE_DEGENERATE_FLAT:
        out["verdict"] = "degenerate"
        out["reason"] = (f"open equals close on {out['flat_share'] * 100:.0f}% of bars -- "
                         f"this feed is not reporting a real intraday range")
    return out


def repair_market_data(df: pd.DataFrame, ticker: str = ""):
    """
    Fix the malformed candles a provider occasionally emits, and report what was
    touched. Returns (frame, notes). Nothing is invented here: a candle's High is
    only ever raised to a price the candle already contains, and its Low lowered
    the same way.
    """
    notes = []
    if df is None or len(df) == 0 or not {"Open", "High", "Low", "Close"} <= set(df.columns):
        return df, notes
    out = df.copy()
    out.attrs = dict(df.attrs)
    hi_should = out[["Open", "High", "Low", "Close"]].max(axis=1)
    lo_should = out[["Open", "High", "Low", "Close"]].min(axis=1)
    n_hi = int((out["High"] < hi_should - 1e-9).sum())
    n_lo = int((out["Low"] > lo_should + 1e-9).sum())
    share = (n_hi + n_lo) / max(len(out), 1)
    if (n_hi or n_lo) and share > MAX_REPAIR_SHARE:
        # Deliberately NOT repaired: at this rate the feed is wrong rather than
        # noisy, and tidying it would hide that. Validation refuses it next.
        notes.append(f"REFUSED to repair {n_hi + n_lo} of {len(out)} candles "
                     f"({share:.1%}) -- that is a broken feed, not provider noise")
    elif n_hi or n_lo:
        out["High"], out["Low"] = hi_should, lo_should
        notes.append(f"repaired {n_hi + n_lo} malformed candle(s) "
                     f"({n_hi} high below open/close, {n_lo} low above it)")
        if share > NOISY_OHLC_SHARE:
            notes.append(f"{share:.0%} of candles needed repair — this feed's intraday "
                         f"high/low is unreliable, so ATR, candlestick patterns and "
                         f"range-based volatility should be treated with caution. The "
                         f"closing prices are clean, and the forecast is built on those")
    if "Volume" in out.columns:
        n_v = int((out["Volume"] < 0).sum())
        if n_v:
            out["Volume"] = out["Volume"].clip(lower=0)
            notes.append(f"clamped {n_v} negative volume value(s) to zero")
    return out, notes


# ------------------------------------------------------------------ validation
def validate_market_data(df: pd.DataFrame, ticker: str, interval: str = "1d",
                         as_of=None, mode: str = MODE_CURRENT, min_rows: int = 2) -> dict:
    """
    Phase 9. Every check that has to pass before data may reach a model.

    Returns a report dict; raises MarketDataError on anything fatal. A warning
    is recorded but not fatal when it degrades quality without invalidating the
    data (a few duplicate stamps, thin volume).
    """
    errors, warnings_ = [], []

    if df is None or len(df) == 0:
        raise MarketDataError(f"{ticker}: no data returned")
    if len(df) < min_rows:
        errors.append(f"only {len(df)} rows")

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            errors.append(f"missing column {col}")
    if errors:
        raise MarketDataError(f"{ticker}: " + "; ".join(errors))

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        errors.append("index is not timestamps")
    else:
        if not idx.is_monotonic_increasing:
            errors.append("timestamps are not sorted")
        dupes = int(idx.duplicated().sum())
        if dupes:
            warnings_.append(f"{dupes} duplicate timestamp(s)")
        if idx.tz is not None:
            warnings_.append("timestamps are timezone-aware; the project works in exchange-local naive time")

        # nothing may be dated after the moment this data claims to describe
        cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
        if interval in ("1d", "1wk") and as_of is not None:
            cutoff = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
        future = idx[idx > cutoff]
        if len(future):
            errors.append(f"{len(future)} candle(s) dated after the as-of moment "
                          f"(latest {future[-1]}, cutoff {cutoff})")

    ohlc = df[["Open", "High", "Low", "Close"]]
    if not np.isfinite(ohlc.to_numpy(dtype=float)).all():
        warnings_.append("non-finite OHLC values present")
    bad_high = int((df["High"] < df[["Open", "Close"]].max(axis=1) - 1e-9).sum())
    bad_low = int((df["Low"] > df[["Open", "Close"]].min(axis=1) + 1e-9).sum())
    bad_share = (bad_high + bad_low) / max(len(df), 1)
    if bad_share > MAX_REPAIR_SHARE:
        # Too many to be provider noise -- the feed itself is wrong, so refuse.
        errors.append(f"{bad_high + bad_low} of {len(df)} candles are malformed "
                      f"({bad_share:.1%}); the feed is unreliable, not merely noisy")
    elif bad_high or bad_low:
        warnings_.append(f"{bad_high + bad_low} malformed candle(s) present "
                         f"({bad_share:.2%}) -- repairable")
    n_nonpos = int((df[["Open", "High", "Low", "Close"]] <= 0).to_numpy().sum())
    if n_nonpos:
        # Real for futures (WTI settled negative on 20 Apr 2020), impossible for
        # anything with a share price.
        if str(ticker).upper().endswith(NEGATIVE_OK_SUFFIX):
            warnings_.append(f"{n_nonpos} non-positive price(s) -- expected for a futures contract")
        else:
            errors.append(f"{n_nonpos} non-positive price(s), which is impossible for {ticker}")
    if "Volume" in df.columns:
        if (df["Volume"] < 0).any():
            warnings_.append("negative volume present -- repairable")
        elif float((df["Volume"] == 0).mean()) > 0.5 and interval in ("1d", "1wk"):
            warnings_.append("more than half the sessions report zero volume")

    if errors:
        raise MarketDataError(f"{ticker} [{interval}, {mode}]: " + "; ".join(errors))

    return {"ticker": ticker, "interval": interval, "mode": mode, "rows": len(df),
            "first": idx[0], "last": idx[-1], "warnings": warnings_,
            "as_of": pd.Timestamp(as_of) if as_of is not None else None}


# ------------------------------------------------------------------ the engine
def get_market_data(ticker: str, start=None, end=None, interval: str = "1d",
                    as_of=None, validate: bool = True):
    """
    The single entry point for price history.

    as_of : the moment the caller is pretending it is. When it is earlier than
            today the frame is HISTORICAL: it is truncated at that moment, it is
            marked immutable, and the live quote can never be applied to it.

    Returns (df, report). The frame carries its mode in df.attrs so that any
    later code can refuse to contaminate it.
    """
    import data_loader as dl
    import timeframes as tfm

    today = pd.Timestamp(datetime.date.today())
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else None
    historical = as_of_ts is not None and as_of_ts.normalize() < today
    mode = MODE_HISTORICAL if historical else MODE_CURRENT

    if tfm.get(interval)["intraday"]:
        df = dl.load_ohlcv(ticker, interval)
        if as_of_ts is not None:
            df = df.loc[df.index <= as_of_ts]
    else:
        end_eff = (as_of_ts or pd.Timestamp(end) if end else today)
        end_eff = pd.Timestamp(end_eff)
        if as_of_ts is not None:
            end_eff = min(end_eff, as_of_ts)
        df = dl.load_stock_data(ticker, start or "2014-01-01", end_eff.strftime("%Y-%m-%d"))
        df = df.loc[df.index <= end_eff]

    df, _repairs = repair_market_data(df, ticker)
    df = df.copy()
    df.attrs.update({
        "ticker": ticker, "interval": interval, "mode": mode, "repairs": _repairs,
        "as_of": as_of_ts, "immutable": historical,
        "source": "yahoo", "loaded_at": pd.Timestamp.now(),
    })
    report = validate_market_data(df, ticker, interval, as_of=as_of_ts, mode=mode) if validate else {}
    return df, report


def cache_key(ticker: str, start, end, interval: str, mode: str, feature_version: str = "v2",
              model_version: str = "v1", horizon: int = 0, adjustment: str = "raw",
              timezone: str = "exchange") -> str:
    """Phase 8: a key that cannot collide across any dimension that changes the answer."""
    return "|".join(str(x) for x in (ticker, interval, start, end, mode, adjustment, timezone,
                                     feature_version, model_version, horizon))


# ------------------------------------------------------------------ live quote
def apply_live_quote_safe(raw: pd.DataFrame, quote: dict, is_historical: bool = None):
    """
    Overlay the live quote onto the newest candle -- but ONLY on a current frame.

    Returns (frame, quote, applied). On a historical frame it returns the frame
    untouched with applied=False, so the caller can say so in the UI instead of
    silently showing today's price for a past date.
    """
    if raw is None or len(raw) == 0:
        return raw, quote, False
    if is_historical is None:
        is_historical = bool(raw.attrs.get("immutable")) or raw.attrs.get("mode") == MODE_HISTORICAL
    if is_historical:
        return raw, quote, False                      # Phase 2: never mutate history
    if not quote or not quote.get("price"):
        return raw, quote, False

    q_time = pd.Timestamp(quote["time"], unit="s", tz="UTC").tz_convert(
        quote.get("tz") or "UTC").tz_localize(None)
    q_day = pd.Timestamp(q_time.date())
    if q_day < raw.index[-1].normalize():
        return raw, quote, False

    out = raw.copy()
    out.attrs = dict(raw.attrs)
    if q_day > raw.index[-1].normalize():
        out.loc[q_day] = out.iloc[-1]
        out.loc[q_day, "Open"] = quote.get("open") or quote["price"]
        out.loc[q_day, "Volume"] = quote.get("volume") or 0
    row = out.index[-1]
    out.loc[row, "Close"] = quote["price"]
    if quote.get("open"):
        out.loc[row, "Open"] = quote["open"]
    out.loc[row, "High"] = max(quote.get("day_high") or quote["price"], quote["price"])
    out.loc[row, "Low"] = min(quote.get("day_low") or quote["price"], quote["price"])
    if quote.get("volume"):
        out.loc[row, "Volume"] = quote["volume"]
    if "Adj Close" in out.columns:
        out.loc[row, "Adj Close"] = quote["price"]
    out.attrs["live_quote_applied"] = True
    return out, quote, True


def describe_mode(df: pd.DataFrame) -> str:
    """One line the UI can show so the user always knows which world they are in."""
    mode = df.attrs.get("mode", MODE_CURRENT)
    last = df.index[-1]
    if mode == MODE_HISTORICAL:
        return (f"HISTORICAL MODE · as of {df.attrs.get('as_of', last):%d %b %Y} · "
                f"last candle {last:%d %b %Y} · live data is switched off")
    return f"CURRENT MODE · last candle {last:%d %b %Y} · live quote active"
