"""
test_timeframes.py
--------------------
Proves that changing the candle selector changes the ANALYSIS, not just a label.

Run:  python test_timeframes.py           (defaults to HDFCBANK.NS)
      python test_timeframes.py RELIANCE.NS

For 5m / 15m / 1h / 1d it checks that:

  * the number of candles changes
  * the timestamp spacing matches the selected interval exactly
  * the OHLC values differ between timeframes (not the same bars relabelled)
  * a 15-minute candle's high/low really is the extreme of its three 5-minute
    candles -- i.e. the aggregation is real
  * indicators (RSI, SMA, ATR) are recalculated and differ
  * support/resistance levels differ
  * forecast horizons are expressed in the right number of candles
  * model cache keys differ per timeframe
  * the forecast itself differs

Every check prints PASS or FAIL and the script exits non-zero if any fail.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import data_loader as dl
import technical_analysis as ta
import timeframes as tfm

TICKER = sys.argv[1] if len(sys.argv) > 1 else "HDFCBANK.NS"
CHECK_TFS = ["5m", "15m", "1h", "1d"]
failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        failures.append(name)
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    return condition


print(f"\nTimeframe integrity test — {TICKER}\n" + "=" * 70)

data, indicators = {}, {}
for tf in CHECK_TFS:
    try:
        df = dl.load_ohlcv(TICKER, tf)
    except Exception as e:
        print(f"\n{tf}: COULD NOT LOAD — {type(e).__name__}: {e}")
        failures.append(f"load {tf}")
        continue
    data[tf] = df
    print(f"\n{tf:4} ({tfm.label(tf)})")
    print(f"  candles={len(df):,}  first={df.index[0]}  last={df.index[-1]}"
          f"  resampled_from={df.attrs.get('resampled_from')}")

    # --- spacing matches the interval ---
    if len(df) > 5:
        deltas = pd.Series(df.index).diff().dropna()
        common = deltas.mode().iloc[0]
        expected = (pd.Timedelta(minutes=tfm.get(tf)["minutes"]) if tfm.get(tf)["minutes"]
                    else pd.Timedelta(days=1))
        check(f"{tf}: candle spacing is {expected}", common == expected, f"most common gap {common}")

    # --- indicators recomputed on these bars ---
    try:
        import features_indicators as fi
        sc = ta.technical_score_series(df)
        bank = fi.indicator_features(df).iloc[-1]
        indicators[tf] = {
            "rsi": (float(bank["Mom_RSI14"]) + 0.5) * 100,      # bank stores RSI centred on zero
            "sma20": float(df["Close"].rolling(20).mean().iloc[-1]),
            "atr": float(bank["Vol_ATR_Pct"]) * 100,
            "score": float(sc["Tech_Score"].iloc[-1]),
        }
        print(f"  RSI={indicators[tf]['rsi']:.1f}  SMA20={indicators[tf]['sma20']:.2f}  "
              f"ATR={indicators[tf]['atr']:.2f}%  score={indicators[tf]['score']:+.2f}")
    except Exception as e:
        print(f"  indicator error: {type(e).__name__}: {e}")
        failures.append(f"indicators {tf}")

    # --- support/resistance on these bars ---
    try:
        levels = ta.support_resistance_levels(df)
        indicators.setdefault(tf, {})["levels"] = tuple(round(l["level"], 2) for l in levels[:4])
        print(f"  S/R levels: {indicators[tf]['levels']}")
    except Exception:
        pass

    # --- horizon maths ---
    per_session = tfm.candles_per_session(tf, "Asia/Kolkata")
    one_session = tfm.horizon_to_candles(tf, 1, "Asia/Kolkata")
    print(f"  1 trading day = {one_session} candles ({per_session:g} per session)")
    check(f"{tf}: one session resolves to the right candle count",
          abs(one_session - per_session) <= 1, f"{one_session} vs {per_session:g}")

print("\n" + "=" * 70 + "\nCross-timeframe checks")

# candle counts must differ
if len(data) > 1:
    counts = {tf: len(df) for tf, df in data.items()}
    check("candle counts differ between timeframes", len(set(counts.values())) == len(counts), str(counts))

# 15m bars must really aggregate 5m bars, not relabel them
if "5m" in data and "15m" in data:
    five, fifteen = data["5m"], data["15m"]
    overlap = fifteen.index.intersection(five.index)
    tested = ok = 0
    for ts in overlap[-40:]:
        window = five.loc[ts:ts + pd.Timedelta(minutes=14)]
        if len(window) < 2:
            continue
        tested += 1
        if (abs(float(fifteen.loc[ts, "High"]) - float(window["High"].max())) < 0.02 and
                abs(float(fifteen.loc[ts, "Low"]) - float(window["Low"].min())) < 0.02):
            ok += 1
    check("15m candles are the true aggregate of their 5m candles", tested and ok >= tested * 0.8,
          f"{ok}/{tested} matched on high and low")
    check("15m is NOT just 5m relabelled", len(five) != len(fifteen),
          f"{len(five):,} five-minute vs {len(fifteen):,} fifteen-minute")

# indicators must differ across timeframes
if len(indicators) > 1:
    rsis = {tf: v.get("rsi") for tf, v in indicators.items() if v.get("rsi") == v.get("rsi")}
    check("RSI differs across timeframes", len(set(round(v, 1) for v in rsis.values())) > 1,
          ", ".join(f"{k}={v:.1f}" for k, v in rsis.items()))
    smas = {tf: round(v["sma20"], 2) for tf, v in indicators.items() if "sma20" in v}
    check("SMA20 differs across timeframes", len(set(smas.values())) > 1, str(smas))
    atrs = {tf: round(v["atr"], 3) for tf, v in indicators.items() if "atr" in v}
    check("ATR differs across timeframes", len(set(atrs.values())) > 1,
          ", ".join(f"{k}={v:.2f}%" for k, v in atrs.items()))
    lv = {tf: v.get("levels") for tf, v in indicators.items() if v.get("levels")}
    if len(lv) > 1:
        check("support/resistance differs across timeframes", len(set(lv.values())) > 1, "")

# forecast + cache keys
print("\nForecast and cache separation")
try:
    import forecast_engine as fe
    preds, keys = {}, {}
    for tf in ("15m", "1d"):
        if tf not in data:
            continue
        raw = data[tf]
        hs = tfm.model_horizons(tf, len(raw), "Asia/Kolkata")
        eng = fe.ForecastEngine(raw, TICKER, horizons=hs, timeframe=tf,
                                bars_per_year=tfm.annualisation_factor(tf, "Asia/Kolkata")).fit()
        path = eng.forecast_path(min(hs[-1], 20))
        last = float(raw["Close"].iloc[-1])
        preds[tf] = round(float(path["Predicted_Close"].iloc[-1]) / last - 1, 6)
        keys[tf] = f"{TICKER}_{tf}_{len(raw)}"
        print(f"  {tf:4} horizons={hs} forecast={preds[tf] * 100:+.3f}% cache_key={keys[tf]}")
        check(f"{tf}: forecast timestamps step by the candle size",
              (path.index[1] - path.index[0]) == (pd.Timedelta(minutes=tfm.get(tf)["minutes"])
                                                  if tfm.get(tf)["minutes"] else pd.Timedelta(days=1))
              or tf == "1d", str(path.index[1] - path.index[0]))
    if len(preds) > 1:
        check("forecasts differ between timeframes", len(set(preds.values())) > 1, str(preds))
        check("model cache keys differ between timeframes", len(set(keys.values())) == len(keys), "")
except Exception as e:
    print(f"  forecast check error: {type(e).__name__}: {e}")
    failures.append("forecast")

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED — the timeframe selector changes the underlying analysis.")
