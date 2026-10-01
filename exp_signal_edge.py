"""Does the model's own signal turn a coin-flip bracket into an edge?

A blind same-day bracket on HDFCBANK.NS has expectancy -0.30%/trade at 0.30%
costs and -0.00% at zero cost: no edge in the entry, and the round trip is the
whole loss. The question that justifies the entire system is whether taking the
trade ONLY when the model points that way changes it.

Walk-forward: train on everything before a cut, act on the next block, never the
reverse. Compared against the same bracket taken blindly on the same sessions.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import data_loader as dl, market_data as md, forecast_engine as fe, models_tabular as mt, options_plan as op

TICKERS = ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "MSFT", "AAPL", "JPM"]
COST = 0.30
N_FOLDS = 4


def session_returns(d, direction, stop_atr=0.8, target_atr=1.5):
    """Per-session net % of the bracket, aligned to d.index."""
    a = op.atr(d).shift(1)
    o, h, l, c = (d[x].astype(float) for x in ("Open", "High", "Low", "Close"))
    long = direction == "long"
    stop = o - stop_atr * a if long else o + stop_atr * a
    tgt = o + target_atr * a if long else o - target_atr * a
    hit_t = (h >= tgt) if long else (l <= tgt)
    hit_s = (l <= stop) if long else (h >= stop)
    ex = pd.Series(np.nan, index=o.index)
    win = hit_t & ~hit_s
    ex[win] = tgt[win]; ex[hit_s] = stop[hit_s]
    flat = ~hit_t & ~hit_s
    ex[flat] = c[flat]
    r = ((ex / o - 1) if long else (1 - ex / o)) * 100
    return r - COST


rows = []
for t in TICKERS:
    raw = dl.load_stock_data(t, "2014-01-01", "2026-09-29"); raw, _ = md.repair_market_data(raw, t)
    tbl, g = fe.build_feature_table(raw, t)
    tgt_col = "Fwd_Ret_1"
    if tgt_col not in tbl: continue
    d = tbl.dropna(subset=[tgt_col])
    cols = [c for c in sum(g.values(), []) if c in d.columns and c != tgt_col]
    long_r = session_returns(raw, "long").reindex(d.index)
    short_r = session_returns(raw, "short").reindex(d.index)

    n = len(d); start = int(n * 0.5); size = (n - start) // N_FOLDS
    sig_ret, blind_ret, taken = [], [], 0
    for k in range(N_FOLDS):
        a, b = start + k * size, (start + (k + 1) * size if k < N_FOLDS - 1 else n)
        tr, te = d.iloc[:a], d.iloc[a:b]
        if len(tr) < 300 or len(te) < 20: continue
        sc = StandardScaler().fit(tr[cols].values)
        m = mt.build_random_forest_regularized()
        m.fit(sc.transform(tr[cols].values), tr[tgt_col].values)
        pred = m.predict(sc.transform(te[cols].values))
        for i, ix in enumerate(te.index):
            lr, sr = long_r.get(ix, np.nan), short_r.get(ix, np.nan)
            if not np.isfinite(lr) or not np.isfinite(sr): continue
            blind_ret.append(lr)                      # blind = always long
            sig_ret.append(lr if pred[i] > 0 else sr) # signal picks the side
            taken += 1
    if taken < 100: continue
    s, bl = np.array(sig_ret), np.array(blind_ret)
    rows.append({"ticker": t, "n": taken,
                 "signal_exp": s.mean(), "blind_exp": bl.mean(),
                 "signal_win": (s > 0).mean() * 100, "blind_win": (bl > 0).mean() * 100,
                 "edge": s.mean() - bl.mean()})
    print(f"  {t:<13} n={taken:<5} signal {s.mean():+.3f}%/trade ({(s>0).mean()*100:.0f}% win)   "
          f"blind {bl.mean():+.3f}% ({(bl>0).mean()*100:.0f}% win)   edge {s.mean()-bl.mean():+.3f}pp")

df = pd.DataFrame(rows)
print("\n" + "=" * 78)
print(f"signal expectancy {df.signal_exp.mean():+.3f}%/trade   "
      f"blind {df.blind_exp.mean():+.3f}%/trade   edge {df.edge.mean():+.3f}pp")
print(f"signal beats blind in {(df.edge > 0).sum()}/{len(df)} symbols")
print(f"signal is PROFITABLE after costs in {(df.signal_exp > 0).sum()}/{len(df)} symbols")
df.to_csv("exp_signal_edge.csv", index=False); print("wrote exp_signal_edge.csv")
