"""Every approach tried, on the same out-of-sample period, against doing nothing."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import exp_anomalies as ea

px = ea.build_panel()
split = px.index[int(len(px)*0.6)]
oos = px.loc[split:]
rets = oos.pct_change()
yrs = len(oos)/252

def stats(series_pct):
    """series_pct: per-period % returns"""
    s = pd.Series(series_pct).dropna()
    if len(s) < 5: return None
    tot = (1+s/100).prod()
    per_yr = len(s)/yrs
    return {"ann": (tot**(1/yrs)-1)*100,
            "sharpe": s.mean()/s.std()*np.sqrt(per_yr) if s.std()>0 else 0,
            "maxdd": float((( (1+s/100).cumprod() / (1+s/100).cumprod().cummax() )-1).min()*100),
            "n": len(s)}

rows=[]
ew = rets.mean(axis=1)*100
r = stats(ew); rows.append(("Equal-weight buy & hold (no trading)", r))
try:
    import data_loader as dl
    nif = dl.load_stock_data('^NSEI','2014-01-01','2026-09-29').loc[split:]
    r = stats(nif['Close'].pct_change()*100); rows.append(("NIFTY 50 index", r))
except Exception: pass

sig = {"Momentum 12-1 (long-only, 21d)": (-(px.shift(21)/px.shift(252)-1), 21, False),
       "Momentum 12-1 (long/short)":      (-(px.shift(21)/px.shift(252)-1), 21, True),
       "Reversal 10d (long-only)":        (px.pct_change().rolling(10).sum(), 10, False),
       "Reversal 10d (long/short)":       (px.pct_change().rolling(10).sum(), 10, True),
       "Low volatility (long-only)":      (px.pct_change().rolling(60).std(), 21, False)}
for name,(s,h,ls) in sig.items():
    b = ea.backtest_rank(oos, s.loc[split:], h, long_short=ls)
    if b: rows.append((name, stats(b["series"]["ret"])))

print("="*92)
print(f"OUT-OF-SAMPLE {split:%b %Y} to {oos.index[-1]:%b %Y}  ({yrs:.1f} years, {px.shape[1]} NSE large caps)")
print("="*92)
print(f"{'strategy':<40}{'annualised':>13}{'Sharpe':>9}{'max drawdown':>15}{'rebalances':>12}")
for name, r in rows:
    if r: print(f"{name:<40}{r['ann']:>+12.1f}%{r['sharpe']:>9.2f}{r['maxdd']:>14.1f}%{r['n']:>12}")
print()
print("  The ML day-trading strategies measured elsewhere, for comparison:")
print(f"  {'Same-day bracket (v1, always trade)':<40}{-0.281*252:>+12.1f}%{'':>9}{'':>15}{'daily':>12}")
print(f"  {'Selective fade (v2, 12-week test)':<40}{-3.28*4:>+12.1f}%{'':>9}{'':>15}{'46 trades':>12}")
