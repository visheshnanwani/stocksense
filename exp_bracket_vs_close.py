"""The paper trade hit its target 0 times in 24 trades. Is the bracket broken?

Target = 1.5 x ATR. If a session rarely travels that far from its open, the
bracket can only ever be stopped or drift to the close -- it has a loss branch
and no win branch. This measures, per target size, how often the target is
actually reachable, and compares the whole bracket against simply holding to
the close.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import data_loader as dl, market_data as md, options_plan as opl

UNI = ["HDFCBANK.NS","RELIANCE.NS","TCS.NS","INFY.NS","ITC.NS","SBIN.NS"]
COST = 0.30
rows = []
for t in UNI:
    raw = dl.load_stock_data(t, "2014-01-01", "2026-09-29"); raw,_ = md.repair_market_data(raw,t)
    a = opl.atr(raw).shift(1)
    o,h,l,c = (raw[x].astype(float) for x in ("Open","High","Low","Close"))
    ok = a.notna() & (a>0); o,h,l,c,a = o[ok],h[ok],l[ok],c[ok],a[ok]
    # hold to close, long
    close_only = (c/o - 1)*100 - COST
    for ta in (0.3,0.5,0.8,1.0,1.5,2.0):
        stop, tgt = o - 0.8*a, o + ta*a
        hit_t, hit_s = h>=tgt, l<=stop
        ex = pd.Series(np.nan, index=o.index)
        ex[hit_s] = stop[hit_s]
        won = hit_t & ~hit_s; ex[won] = tgt[won]
        flat = ~hit_t & ~hit_s; ex[flat] = c[flat]
        net = (ex/o - 1)*100 - COST
        rows.append({"ticker":t,"target_atr":ta,
                     "target_pct_of_price": float((ta*a/o*100).median()),
                     "target_reached": float(won.mean()*100),
                     "stopped": float(hit_s.mean()*100),
                     "drifted_to_close": float(flat.mean()*100),
                     "bracket_exp": float(net.mean()),
                     "close_only_exp": float(close_only.mean())})
df = pd.DataFrame(rows)
g = df.groupby("target_atr").mean(numeric_only=True)
print("Averaged over 6 NSE symbols, stop fixed at 0.8 x ATR, 0.30% costs\n")
print(f"{'target':>7}{'= % of price':>14}{'target hit':>12}{'stopped':>10}{'drifted':>10}"
      f"{'bracket':>10}{'hold to close':>15}")
for ta, r in g.iterrows():
    print(f"{ta:>7.1f}{r.target_pct_of_price:>13.2f}%{r.target_reached:>11.1f}%"
          f"{r.stopped:>9.1f}%{r.drifted_to_close:>9.1f}%"
          f"{r.bracket_exp:>+9.3f}%{r.close_only_exp:>+14.3f}%")
print(f"\nhold-to-close expectancy is the same in every row by construction: "
      f"{g.close_only_exp.iloc[0]:+.3f}%")
best = g.bracket_exp.idxmax()
print(f"best bracket: target {best:.1f} x ATR at {g.bracket_exp.max():+.3f}%  "
      f"vs hold-to-close {g.close_only_exp.iloc[0]:+.3f}%")
df.to_csv("exp_bracket_vs_close.csv", index=False)
