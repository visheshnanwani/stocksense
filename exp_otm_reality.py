"""
Do options really move 100%+ in a day?  YES. Does that make money?  Separate question.

Real NSE underlying moves, 12 years. Each session: buy an out-of-the-money weekly
option at the open, sell at the close. Priced with Black-Scholes on each symbol's
own realised vol. OTM is used deliberately because that is where the 100% moves are.
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
out={}
for otm in (0.0, 1.0, 2.0, 3.0):
    allr=[]
    for t in syms:
        try:
            d=dl.load_stock_data(t,'2014-01-01','2026-09-29'); d,_=md.repair_market_data(d,t)
            rv=d['Close'].pct_change().rolling(21).std()*np.sqrt(252)
            o=d['Open'].astype(float); c=d['Close'].astype(float)
            for i in range(30,len(d)):
                s=float(rv.iloc[i-1])
                if not np.isfinite(s) or s<=0: continue
                S0=float(o.iloc[i]); S1=float(c.iloc[i])
                K=S0*(1+otm/100)
                T0=5/252.0; T1=max(T0-1/252.0,1e-6)
                p0=call(S0,K,T0,R,s); p1=call(S1,K,T1,R,s)
                if p0<=0.01*S0*0.001: continue
                allr.append((p1/p0-1)*100)
        except Exception: continue
    a=np.array(allr)
    out[otm]=a
print("BUY AN OTM CALL AT THE OPEN, SELL AT THE CLOSE — %d NSE symbols, ~12 years"%len(syms))
print()
print("%-10s%9s%11s%11s%11s%11s%11s%11s"%("strike","n","gain>100%","gain>50%","loss>50%","loss>80%","median","MEAN"))
for otm,a in out.items():
    lbl = "ATM" if otm==0 else "+%.0f%% OTM"%otm
    print("%-10s%9d%10.1f%%%10.1f%%%10.1f%%%10.1f%%%10.1f%%%10.1f%%"%(
        lbl,len(a),(a>100).mean()*100,(a>50).mean()*100,(a<-50).mean()*100,(a<-80).mean()*100,
        np.median(a),a.mean()))
print()
a=out[3.0]
print("  For the +3%% OTM call — the lottery ticket:")
print("    biggest single day gain : %+.0f%%"%a.max())
print("    days gaining over 100%%  : %.1f%% of sessions"%((a>100).mean()*100))
print("    days losing over 80%%    : %.1f%% of sessions"%((a<-80).mean()*100))
print("    median day              : %+.1f%%"%np.median(a))
print("    average day             : %+.1f%%"%a.mean())
print()
print("  Rs 100,000 buying that option every session, full size:")
for otm,a in out.items():
    lbl = "ATM" if otm==0 else "+%.0f%% OTM"%otm
    eq=100000.0
    for x in a[:2000]:
        eq*= (1+x/100)
        if eq<1: break
    print("    %-10s after %4d sessions -> Rs %s"%(lbl,min(2000,len(a)),
          f"{eq:,.0f}" if eq>=1 else "0 (wiped out)"))
