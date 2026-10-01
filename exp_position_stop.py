"""A stop on the POSITION, not on each leg.

30 March: both puts rose past 2x but stopped at 2.32x and 2.13x, under the 2.5x
per-leg trigger, so nothing fired and the session cost Rs 48,097. A per-leg stop
only sees one leg at a time; the thing that actually hurts is the straddle's
COMBINED value running away while the winning leg decays to nothing.

So: exit both legs when (call + put) reaches N x what was collected. Bounded with
the same pessimistic convention -- the combined high is approximated by each
leg's own high, which overstates how far the pair ever traded together, so this
test makes the stop fire EARLIER and more often than reality would. If it still
helps under that handicap, the effect is real.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import exp_straddle_improvements as b

def run(d, pos_mult=None, slip=0.0, leg_mult=None):
    rows=[]
    for (dt,sym),g in d.groupby(["date","symbol"]):
        lot=int(g.lot.iloc[0]); ps=float(g.prev_spot.iloc[0])
        ce=b.pick_leg(g,"CE",0.0); pe=b.pick_leg(g,"PE",0.0)
        if ce is None or pe is None: continue
        collected=float(ce.open)+float(pe.open)
        # upper bound on the pair's intraday value
        pair_high=float(ce.high)+float(pe.high)
        stopped=False
        if pos_mult is not None and pair_high >= collected*pos_mult:
            exit_val = collected*pos_mult*(1+slip/100); stopped=True
        elif leg_mult is not None and (float(ce.high)>=float(ce.open)*leg_mult or
                                       float(pe.high)>=float(pe.open)*leg_mult):
            ce_x = float(ce.open)*leg_mult*(1+slip/100) if float(ce.high)>=float(ce.open)*leg_mult else float(ce.close)
            pe_x = float(pe.open)*leg_mult*(1+slip/100) if float(pe.high)>=float(pe.open)*leg_mult else float(pe.close)
            exit_val = ce_x+pe_x; stopped=True
        else:
            exit_val = float(ce.close)+float(pe.close)
        pts = collected-exit_val
        rows.append({"date":dt,"pnl":pts*lot-b.COST_RS*2,
                     "margin":ps*lot*b.MARGIN_FRAC*2,"stopped":stopped})
    t=pd.DataFrame(rows)
    return t.groupby("date").agg(n=("pnl","size"),pnl=("pnl","sum"),
                                 margin=("margin","sum"),stopped=("stopped","sum")).reset_index()

def st(day):
    r=day.pnl/day.margin*100; eq=day.pnl.cumsum()
    return dict(total=day.pnl.sum(), sharpe=r.mean()/r.std()*np.sqrt(252),
                worst=day.pnl.min(), dd=float((eq-eq.cummax()).min()),
                fired=int(day.stopped.sum()),
                mar=day[(day.date>='2026-03-01')&(day.date<'2026-04-01')].pnl.sum(),
                sep=day[day.date>='2026-09-01'].pnl.sum())

d=b.load()
print("POSITION-LEVEL STOP vs PER-LEG STOP  (158 sessions, real prices)")
print()
print("{:<26}{:>11}{:>8}{:>12}{:>12}{:>11}{:>11}{:>7}".format(
  "variant","total","Sharpe","worst day","max DD","March","Sept","fired"))
rows=[("no stop",dict()),("per-leg 2.5x",dict(leg_mult=2.5)),("per-leg 2.0x",dict(leg_mult=2.0))]
for m in (1.4,1.6,1.8,2.0,2.5):
    rows.append(("position {:g}x".format(m),dict(pos_mult=m)))
for name,kw in rows:
    s=st(run(d,**kw))
    print("{:<26}{:>+11,.0f}{:>8.2f}{:>+12,.0f}{:>+12,.0f}{:>+11,.0f}{:>+11,.0f}{:>7}".format(
      name,s['total'],s['sharpe'],s['worst'],s['dd'],s['mar'],s['sep'],s['fired']))
print()
print("with a 5% worse fill on the stop:")
for m in (1.4,1.6,1.8,2.0):
    s=st(run(d,pos_mult=m,slip=5.0))
    print("  position {:g}x -> total {:+,.0f}  Sharpe {:.2f}  March {:+,.0f}  Sept {:+,.0f}".format(
      m,s['total'],s['sharpe'],s['mar'],s['sep']))
