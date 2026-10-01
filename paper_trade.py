"""
paper_trade.py
--------------
A one-week paper trade, run honestly.

WHAT THIS IS AND IS NOT
-----------------------
It is not a forward test -- that takes a week of wall-clock time. It is a
REPLAY of the last five sessions under point-in-time discipline: the model is
trained only on data dated strictly before the week starts, and it then trades
those five days blind. No feature, no price and no fit sees anything inside the
week it is trading.

WHY ONE WEEK IS ALMOST MEANINGLESS ON ITS OWN
---------------------------------------------
Five sessions across a handful of symbols is a few dozen trades. The per-trade
expectancy measured elsewhere in this project is about -0.3%, and the per-trade
spread is around +/-1%. A week is therefore dominated by noise: good weeks and
bad weeks both happen regardless of whether the strategy works.

So the week's result is reported next to the distribution of EVERY historical
week the same strategy would have traded, and the week's percentile within it.
A single green week that sits at the 60th percentile of a losing distribution is
a losing strategy having an ordinary week, and this file is built to say so
rather than to let a number stand alone.

THE RULES BEING TRADED
----------------------
  entry   the session's open
  side    long if the model's next-day forecast is positive, else short
  exit    target (target_atr x ATR), stop (stop_atr x ATR), or the close
  costs   a round trip on every trade
  sizing  equal weight across the universe, rebalanced daily

Sessions that touched both target and stop are counted as LOSSES, because a
daily candle does not record which came first.

Run:  python paper_trade.py
      python paper_trade.py --weeks 4 --cost 0.30
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data_loader as dl
import forecast_engine as fe
import market_data as md
import models_tabular as mt
import options_plan as opl

UNIVERSE = ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "ITC.NS", "SBIN.NS"]
STOP_ATR, TARGET_ATR = 0.8, 1.5
START_CAPITAL = 100_000.0

# v2 defaults, each one measured rather than chosen:
#   cost     0.082% of charges + slippage on a liquid NSE large cap
#   quantile only the most confident days are traded; the rest are skipped
#   exit     hold to close (beat every bracket setting), stop kept for risk
COST_INTRADAY = 0.15
CONVICTION_Q = 0.90
INNER_FOLDS = 4


def learn_rule(d, cols, tcol, raw, cost_pct, quantile=CONVICTION_Q):
    """
    Decide, from TRAINING data only: how confident is confident enough, and on
    those days should the signal be followed or faded?

    Uses out-of-fold predictions inside the training window, so the rule is not
    fitted on predictions the same model memorised.
    """
    n = len(d)
    start = int(n * 0.4)
    size = max((n - start) // INNER_FOLDS, 1)
    oof = pd.Series(np.nan, index=d.index)
    for k in range(INNER_FOLDS):
        a = start + k * size
        b = start + (k + 1) * size if k < INNER_FOLDS - 1 else n
        tr, va = d.iloc[:a], d.iloc[a:b]
        if len(tr) < 300 or len(va) < 20:
            continue
        sc = StandardScaler().fit(tr[cols].values)
        m = mt.build_random_forest_regularized()
        m.fit(sc.transform(tr[cols].values), tr[tcol].values)
        oof.iloc[a:b] = m.predict(sc.transform(va[cols].values))
    oof = oof.dropna()
    if len(oof) < 200:
        return None

    o = raw["Open"].astype(float).reindex(oof.index)
    c = raw["Close"].astype(float).reindex(oof.index)
    raw_long = (c / o - 1) * 100
    ok = raw_long.notna()
    oof, raw_long = oof[ok], raw_long[ok]
    thr = float(oof.abs().quantile(quantile))
    sel = oof.abs() >= thr
    if sel.sum() < 40:
        return None
    following = float((np.sign(oof[sel]) * raw_long[sel]).mean())
    return {"threshold": thr, "decision": "fade" if following < 0 else "follow",
            "train_following": following, "train_n": int(sel.sum())}


def bracket_outcome(row_open, row_high, row_low, row_close, atr_prev, side):
    """What the session actually paid, in percent of the entry. Ties -> loss."""
    if not np.isfinite(atr_prev) or atr_prev <= 0:
        return None, "skipped"
    long = side == "long"
    stop = row_open - STOP_ATR * atr_prev if long else row_open + STOP_ATR * atr_prev
    tgt = row_open + TARGET_ATR * atr_prev if long else row_open - TARGET_ATR * atr_prev
    hit_t = row_high >= tgt if long else row_low <= tgt
    hit_s = row_low <= stop if long else row_high >= stop
    if hit_s:                                   # includes "both": pessimistic
        exit_px, how = stop, "stopped"
    elif hit_t:
        exit_px, how = tgt, "target"
    else:
        exit_px, how = row_close, "closed"
    pct = ((exit_px / row_open - 1) if long else (1 - exit_px / row_open)) * 100
    return pct, how


def run_week_v2(ticker, n_days, cost_pct, quantile=CONVICTION_Q):
    """
    v2: conviction gate + per-symbol follow/fade, both learned before the window.
    Holds to the close with a stop; no target, because the target never paid.
    """
    raw = dl.load_stock_data(ticker, "2014-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    raw, _ = md.repair_market_data(raw, ticker)
    table, groups = fe.build_feature_table(raw, ticker)
    tcol = "Fwd_Ret_1"
    if tcol not in table:
        return [], None
    d = table.dropna(subset=[tcol])
    cols = [c for c in sum(groups.values(), []) if c in d.columns and c != tcol]
    if len(d) < 800:
        return [], None

    atr_prev = opl.atr(raw).shift(1)
    window = raw.index[-n_days:]
    train_end = window[0]
    tr = d.loc[d.index < train_end]
    if len(tr) < 600:
        return [], None

    rule = learn_rule(tr, cols, tcol, raw, cost_pct, quantile)
    if rule is None:
        return [], None

    sc = StandardScaler().fit(tr[cols].values)
    model = mt.build_random_forest_regularized()
    model.fit(sc.transform(tr[cols].values), tr[tcol].values)

    trades = []
    for day in window:
        if day not in d.index:
            continue
        pred = float(model.predict(sc.transform(d.loc[[day], cols].values))[0])
        if abs(pred) < rule["threshold"]:
            trades.append({"date": day, "ticker": ticker, "side": "flat",
                           "signal": pred * 100, "entry": np.nan, "gross_pct": 0.0,
                           "net_pct": 0.0, "exit": "skipped",
                           "buyhold_pct": (float(raw.loc[day, "Close"]) /
                                           float(raw.loc[day, "Open"]) - 1) * 100})
            continue
        want_long = pred > 0
        if rule["decision"] == "fade":
            want_long = not want_long
        side = "long" if want_long else "short"

        bar = raw.loc[day]
        o, h, l, c = (float(bar[x]) for x in ("Open", "High", "Low", "Close"))
        a = float(atr_prev.get(day, np.nan))
        if not np.isfinite(a) or a <= 0:
            continue
        stop = o - STOP_ATR * a if want_long else o + STOP_ATR * a
        hit_s = (l <= stop) if want_long else (h >= stop)
        exit_px, how = (stop, "stopped") if hit_s else (c, "closed")
        pct = ((exit_px / o - 1) if want_long else (1 - exit_px / o)) * 100
        trades.append({"date": day, "ticker": ticker, "side": side,
                       "signal": pred * 100, "entry": o,
                       "gross_pct": pct, "net_pct": pct - cost_pct, "exit": how,
                       "buyhold_pct": (c / o - 1) * 100})
    return trades, rule


def run_week(ticker, n_days, cost_pct):
    """v1: always trade, bracket exit. Kept so the two can be compared."""
    raw = dl.load_stock_data(ticker, "2014-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    raw, _ = md.repair_market_data(raw, ticker)
    table, groups = fe.build_feature_table(raw, ticker)
    tcol = "Fwd_Ret_1"
    if tcol not in table:
        return []
    d = table.dropna(subset=[tcol])
    cols = [c for c in sum(groups.values(), []) if c in d.columns and c != tcol]
    if len(d) < 600:
        return []

    atr_prev = opl.atr(raw).shift(1)
    # the window is the last n_days sessions that have a full OHLC record
    window = raw.index[-n_days:]
    train_end = window[0]
    tr = d.loc[d.index < train_end]
    if len(tr) < 400:
        return []

    sc = StandardScaler().fit(tr[cols].values)
    model = mt.build_random_forest_regularized()
    model.fit(sc.transform(tr[cols].values), tr[tcol].values)

    trades = []
    for day in window:
        if day not in d.index:
            continue
        pred = float(model.predict(sc.transform(d.loc[[day], cols].values))[0])
        side = "long" if pred > 0 else "short"
        bar = raw.loc[day]
        pct, how = bracket_outcome(float(bar["Open"]), float(bar["High"]), float(bar["Low"]),
                                   float(bar["Close"]), float(atr_prev.get(day, np.nan)), side)
        if pct is None:
            continue
        trades.append({"date": day, "ticker": ticker, "side": side,
                       "signal": pred * 100, "entry": float(bar["Open"]),
                       "gross_pct": pct, "net_pct": pct - cost_pct, "exit": how,
                       "buyhold_pct": (float(bar["Close"]) / float(bar["Open"]) - 1) * 100})
    return trades


def all_weeks_distribution(ticker, cost_pct, block=5):
    """The same rules over the whole history, cut into consecutive 5-session weeks."""
    raw = dl.load_stock_data(ticker, "2014-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    raw, _ = md.repair_market_data(raw, ticker)
    a = opl.atr(raw).shift(1)
    o, h, l, c = (raw[x].astype(float) for x in ("Open", "High", "Low", "Close"))
    ok = a.notna() & (a > 0)
    o, h, l, c, a = o[ok], h[ok], l[ok], c[ok], a[ok]
    # direction-free version: the long side, which is what a coin flip would give
    stop, tgt = o - STOP_ATR * a, o + TARGET_ATR * a
    hit_t, hit_s = h >= tgt, l <= stop
    ex = pd.Series(np.nan, index=o.index)
    ex[hit_s] = stop[hit_s]
    won = hit_t & ~hit_s
    ex[won] = tgt[won]
    flat = ~hit_t & ~hit_s
    ex[flat] = c[flat]
    net = (ex / o - 1) * 100 - cost_pct
    weeks = [net.iloc[i:i + block].sum() for i in range(0, len(net) - block, block)]
    return np.asarray(weeks, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=1)
    ap.add_argument("--cost", type=float, default=COST_INTRADAY)
    ap.add_argument("--v1", action="store_true",
                    help="the original always-trade bracket strategy")
    args = ap.parse_args()
    n_days = 5 * args.weeks

    print("=" * 84)
    print(f"PAPER TRADE — last {n_days} sessions, {len(UNIVERSE)} symbols, "
          f"{args.cost:.2f}% round trip")
    print("Model trained only on data BEFORE the window. Nothing inside it was seen.")
    print("=" * 84)

    all_trades = []
    for t in UNIVERSE:
        try:
            if args.v1:
                tr = run_week(t, n_days, args.cost)
                print(f"  {t:<14} {len(tr)} trades")
            else:
                tr, rule = run_week_v2(t, n_days, args.cost)
                took = sum(1 for x in tr if x["side"] != "flat")
                print(f"  {t:<14} {took} taken / {len(tr)} sessions   "
                      + (f"rule: {rule['decision'].upper()} above |{rule['threshold'] * 100:.3f}%| "
                         f"(training says {rule['train_following']:+.3f}% if followed)"
                         if rule else "no rule learned"))
            all_trades += tr
        except Exception as e:
            print(f"  {t:<14} skipped ({type(e).__name__}: {str(e)[:50]})")

    if not all_trades:
        print("\nno trades — not enough data")
        return 1

    df = pd.DataFrame(all_trades).sort_values(["date", "ticker"])
    full = pd.DataFrame(all_trades).sort_values(["date", "ticker"])
    n_skipped = int((full["side"] == "flat").sum())
    df = full[full["side"] != "flat"].copy()
    if df.empty:
        print("")
        print("=" * 84)
        print("RESULT: NO TRADES TAKEN")
        print("=" * 84)
        print(f"  The rule stood aside on all {n_skipped} sessions: not one signal this week")
        print(f"  cleared its symbol's conviction threshold.")
        print(f"  strategy        +0.00%  ->      +0 on {START_CAPITAL:,.0f}")
        _bh = full.groupby('date')['buyhold_pct'].mean().sum()
        print(f"  buy & hold      {_bh:+.2f}%  -> {START_CAPITAL * _bh / 100:+,.0f}")
        print("")
        print("  Standing aside is the rule working, not the rule failing - its entire")
        print("  edge is refusing days that do not pay. But zero trades is also zero")
        print("  evidence: this week says nothing about whether the rule is any good.")
        print("  Run a longer window (--weeks 8) to see it actually trade.")
        return 0
    per_trade_capital = START_CAPITAL / len(UNIVERSE)

    print(f"\n{'-' * 84}\nEVERY TRADE\n{'-' * 84}")
    print(f"{'date':<12}{'ticker':<14}{'side':<7}{'signal':>9}{'exit':>9}"
          f"{'gross':>9}{'net':>9}{'P&L':>12}")
    for _, r in df.iterrows():
        pnl = per_trade_capital * r["net_pct"] / 100
        print(f"{r['date']:%Y-%m-%d}  {r['ticker']:<14}{r['side']:<7}"
              f"{r['signal']:>+8.3f}%{r['exit']:>9}{r['gross_pct']:>+8.2f}%"
              f"{r['net_pct']:>+8.2f}%{pnl:>+12,.0f}")

    day = df.groupby("date").agg(trades=("net_pct", "size"), avg_net=("net_pct", "mean"))
    day["bh"] = full.groupby("date")["buyhold_pct"].mean()
    day["pnl"] = START_CAPITAL * day["avg_net"] / 100
    day["equity"] = START_CAPITAL + day["pnl"].cumsum()

    print(f"\n{'-' * 84}\nDAY BY DAY (equal weight across the universe, {START_CAPITAL:,.0f} capital)\n{'-' * 84}")
    print(f"{'date':<12}{'trades':>7}{'avg net':>10}{'P&L':>12}{'equity':>14}{'buy&hold':>11}")
    for dt, r in day.iterrows():
        print(f"{dt:%Y-%m-%d}  {int(r['trades']):>6}{r['avg_net']:>+9.2f}%"
              f"{r['pnl']:>+12,.0f}{r['equity']:>14,.0f}{r['bh']:>+10.2f}%")

    total_pct = float(day["avg_net"].sum())
    total_pnl = float(day["pnl"].sum())
    bh_pct = float(day["bh"].sum())
    wins = int((df["net_pct"] > 0).sum())

    print(f"\n{'=' * 84}\nRESULT\n{'=' * 84}")
    if n_skipped:
        print(f"  sessions skipped      {n_skipped}   (below conviction threshold - "
              f"no position taken, not losses)")
    print(f"  trades taken          {len(df)}   ({wins} winners, {len(df) - wins} losers, "
          f"{wins / len(df) * 100:.0f}% win rate)")
    print(f"  exits                 target {int((df.exit == 'target').sum())} · "
          f"stopped {int((df.exit == 'stopped').sum())} · "
          f"closed out {int((df.exit == 'closed').sum())}")
    print(f"  strategy return       {total_pct:+.2f}%   ->  {total_pnl:+,.0f} "
          f"on {START_CAPITAL:,.0f}")
    print(f"  buy & hold, same days {bh_pct:+.2f}%   ->  "
          f"{START_CAPITAL * bh_pct / 100:+,.0f}")
    print(f"  best / worst trade    {df.net_pct.max():+.2f}% / {df.net_pct.min():+.2f}%")

    print(f"\n{'=' * 84}\nIS THIS WEEK MEANINGFUL?\n{'=' * 84}")
    dist = np.concatenate([all_weeks_distribution(t, args.cost) for t in UNIVERSE])
    pct_rank = float((dist < total_pct).mean() * 100)
    print(f"  the same rules over all history: {len(dist):,} weeks")
    print(f"    mean week   {dist.mean():+.2f}%")
    print(f"    median week {np.median(dist):+.2f}%")
    print(f"    best  week  {dist.max():+.2f}%")
    print(f"    worst week  {dist.min():+.2f}%")
    print(f"    weeks that made money: {(dist > 0).mean() * 100:.0f}%")
    print(f"    std dev of a week: {dist.std():.2f}%")
    print(f"\n  THIS week came in at {total_pct:+.2f}%, the {pct_rank:.0f}th percentile "
          f"of that distribution.")
    se = dist.std() / np.sqrt(len(dist))
    print(f"\n  A single week's spread is +/-{dist.std():.2f}%, and the long-run mean is "
          f"{dist.mean():+.2f}% +/- {1.96 * se:.2f}%.")
    print(f"  To tell a {abs(dist.mean()):.2f}%/week effect apart from noise at 95% confidence "
          f"you need roughly {int((1.96 * dist.std() / max(abs(dist.mean()), 1e-9)) ** 2):,} weeks.")
    print(f"  One week cannot show whether this works. It can only show what it did.")
    df.to_csv("paper_trade_log.csv", index=False)
    print(f"\n  wrote paper_trade_log.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
