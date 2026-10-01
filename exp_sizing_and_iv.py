"""The +3% OTM call has a POSITIVE average day (+2.0%) and still goes to zero.

Two things decide whether that positive average is real money or an illusion:
  1. POSITION SIZE -- a multiplicative bet with huge variance destroys capital
     even with a positive mean. Betting small may capture it.
  2. WHAT YOU PAY -- the +2.0% was computed pricing options at REALISED vol.
     The market charges IMPLIED vol, which runs ~3.4 points higher (measured).
     Paying more for the same payoff moves the mean down.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from math import log, sqrt, exp, erf
import data_loader as dl, market_data as md, stock_search as ss

R=0.065
def N(x): return 0.5*(1+erf(x/sqrt(2)))
def call(S,K,T,r,s):
    if T<=1e-9: return max(S-K,0.0)
    d1=(log(S/K)+(r+0.5*s*s)*T)/(s*sqrt(T)); return S*N(d1)-K*exp(-r*T)*N(d1-s*sqrt(T))

syms=[r[0] for r in ss.POPULAR_STOCKS if r[0].endswith('.NS')][:12]

def series(vrp_points, otm=3.0, spread=1.0):
    """Daily % return of buying the OTM call, paying IV = realised + vrp."""
    out=[]
    for t in syms:
        try:
            d=dl.load_stock_data(t,'2014-01-01','2026-09-29'); d,_=md.repair_market_data(d,t)
            rv=d['Close'].pct_change().rolling(21).std()*np.sqrt(252)
            o=d['Open'].astype(float); c=d['Close'].astype(float)
            for i in range(30,len(d)):
                s=float(rv.iloc[i-1])
                if not np.isfinite(s) or s<=0: continue
                iv=s+vrp_points/100.0
                S0=float(o.iloc[i]); S1=float(c.iloc[i]); K=S0*(1+otm/100)
                T0=5/252.0; T1=max(T0-1/252.0,1e-6)
                p0=call(S0,K,T0,R,iv)*(1+spread/200)     # pay the ask
                p1=call(S1,K,T1,R,iv)*(1-spread/200)     # sell the bid
                if p0<=1e-6: continue
                out.append((p1/p0-1)*100)
        except Exception: continue
    return np.array(out)

print("="*88)
print("1. DOES POSITION SIZING RESCUE IT?   (+3% OTM, priced at realised vol, no spread)")
print("="*88)
a=series(0.0, spread=0.0)
print(f"   mean day {a.mean():+.2f}%   median {np.median(a):+.2f}%   n={len(a):,}")
print()
print(f"   {'fraction of capital per trade':<34}{'ending value':>18}{'result':>12}")
rng=np.random.default_rng(7); path=rng.permutation(a)[:3000]
for f in (1.0,0.25,0.10,0.05,0.02,0.01):
    eq=100000.0
    for x in path:
        eq*= (1+f*x/100)
        if eq<1: break
    print(f"   {f*100:>5.0f}%{'':<28}{('Rs %s'%f'{eq:,.0f}') if eq>=1 else 'Rs 0':>18}"
          f"{'wiped out' if eq<1000 else ('grew' if eq>100000 else 'lost'):>12}")

print()
print("="*88)
print("2. WHAT HAPPENS WHEN YOU PAY THE REAL PRICE?  (IV above realised, plus spread)")
print("="*88)
print(f"   {'you pay':<30}{'mean day':>12}{'median day':>13}{'best sizing outcome':>24}")
for vrp,lbl in ((0.0,"realised vol, no spread"),(0.0,"realised vol + 1% spread"),
                (2.0,"IV = realised +2 pts"),(3.4,"IV = realised +3.4 pts (measured)"),
                (5.0,"IV = realised +5 pts")):
    sp = 1.0 if "spread" in lbl or vrp>0 else 0.0
    b=series(vrp, spread=sp)
    best=None
    p=rng.permutation(b)[:3000]
    for f in (0.25,0.10,0.05,0.02,0.01):
        eq=100000.0
        for x in p:
            eq*=(1+f*x/100)
            if eq<1: break
        if best is None or eq>best[1]: best=(f,eq)
    print(f"   {lbl:<30}{b.mean():>+11.2f}%{np.median(b):>+12.2f}%"
          f"{('Rs %s at %.0f%%'%(f'{best[1]:,.0f}',best[0]*100)):>24}")
