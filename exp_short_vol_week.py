"""
What a short-straddle (variance-risk-premium) trade would have made last week.

DATA LIMITATION, STATED FIRST
  There is no historical option chain available. Yahoo serves no NSE options at
  all; NSE's own current endpoint (/api/option-chain-v3 -- they moved from the
  old path) returns an empty body even from a real browser with valid cookies;
  and no source here provides LAST WEEK's premiums at any price.

  So this is a SIMULATION, not a backtest. The realised move is real -- it comes
  from actual prices. The premium collected is modelled with Black-Scholes at an
  assumed implied vol. Everything therefore hinges on that assumption, so the
  result is shown across a range of it rather than at one flattering value.

THE TRADE
  Monday open: sell the at-the-money straddle (one call + one put) on each of 8
  NSE large caps, equal weight. Friday close: buy it back. Costs both ways.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from math import log, sqrt, exp, erf
import data_loader as dl, market_data as md

U=['HDFCBANK.NS','RELIANCE.NS','TCS.NS','INFY.NS','ITC.NS','SBIN.NS','ICICIBANK.NS','AXISBANK.NS']
R=0.065; CAP=100_000.0; COST_LEG=0.02   # % of spot, per option leg, round trip

def N(x): return 0.5*(1+erf(x/sqrt(2)))
def call(S,K,T,r,s):
    if T<=0: return max(S-K,0.0)
    d1=(log(S/K)+(r+0.5*s*s)*T)/(s*sqrt(T)); return S*N(d1)-K*exp(-r*T)*N(d1-s*sqrt(T))
def put(S,K,T,r,s):
    if T<=0: return max(K-S,0.0)
    d1=(log(S/K)+(r+0.5*s*s)*T)/(s*sqrt(T)); return K*exp(-r*T)*N(-(d1-s*sqrt(T)))-S*N(-d1)

data={}
for t in U:
    d=dl.load_stock_data(t,'2014-01-01','2026-09-29'); d,_=md.repair_market_data(d,t)
    data[t]=d

print("="*94)
print("SHORT ATM STRADDLE, last week (23-29 Sep), 8 NSE large caps, Rs %s" % f"{CAP:,.0f}")
print("simulated premiums - see the limitation note at the top of this file")
print("="*94)

for vrp in (0.0, 2.0, 3.4, 5.0, 8.0):
    rows=[]
    for t,d in data.items():
        wk=d.loc['2026-09-23':]
        if len(wk)<3: continue
        S0=float(wk['Open'].iloc[0]); S1=float(wk['Close'].iloc[-1]); K=S0
        rv=float(d['Close'].pct_change().tail(21).std()*np.sqrt(252))
        iv=rv+vrp/100.0                       # assumed implied = realised + premium
        T0=len(wk)/252.0
        prem = call(S0,K,T0,R,iv)+put(S0,K,T0,R,iv)      # collected
        buyback = max(S1-K,0.0)+max(K-S1,0.0)            # intrinsic at exit
        costs = 2*COST_LEG/100.0*S0
        pnl_pts = prem - buyback - costs
        rows.append({'t':t,'spot':S0,'prem_pct':prem/S0*100,'move_pct':abs(S1/S0-1)*100,
                     'pnl_pct':pnl_pts/S0*100})
    r=pd.DataFrame(rows)
    # capital: margin-based sizing is broker specific, so express P&L per rupee of spot exposure
    per_sym=CAP/len(r)
    pnl=(r.pnl_pct/100*per_sym).sum()
    tag = "  <- measured VRP" if abs(vrp-3.4)<0.01 else ""
    print(f"  IV = realised {vrp:+.1f} pts   premium collected {r.prem_pct.mean():5.2f}% of spot   "
          f"realised move {r.move_pct.mean():5.2f}%   P&L {pnl:+9,.0f}  ({r.pnl_pct.mean():+.2f}%){tag}")

print()
print("  per-symbol detail at the measured +3.4 point premium:")
rows=[]
for t,d in data.items():
    wk=d.loc['2026-09-23':]
    S0=float(wk['Open'].iloc[0]); S1=float(wk['Close'].iloc[-1]); K=S0
    rv=float(d['Close'].pct_change().tail(21).std()*np.sqrt(252)); iv=rv+0.034
    T0=len(wk)/252.0
    prem=call(S0,K,T0,R,iv)+put(S0,K,T0,R,iv); bb=abs(S1-K); costs=2*COST_LEG/100*S0
    rows.append((t,prem/S0*100,abs(S1/S0-1)*100,(prem-bb-costs)/S0*100))
print("  %-14s%12s%12s%11s" % ('symbol','premium','move','P&L'))
for t,p,m,pl in rows:
    print("  %-14s%11.2f%%%11.2f%%%10.2f%%" % (t,p,m,pl))
w=sum(1 for _,_,_,pl in rows if pl>0)
print(f"\n  {w} of {len(rows)} symbols profitable")
