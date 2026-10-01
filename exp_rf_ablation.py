import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import data_loader as dl, market_data as md, forecast_engine as fe, models_tabular as mt

TICKERS=["HDFCBANK.NS","RELIANCE.NS","TCS.NS","ITC.NS","MSFT","AAPL","JPM","KO"]
HZ=[5,21]; TEST=0.25
rows=[]
for t in TICKERS:
    raw=dl.load_stock_data(t,"2014-01-01","2026-09-29"); raw,_=md.repair_market_data(raw,t)
    tbl,g=fe.build_feature_table(raw,t)
    for h in HZ:
        tgt=f"Fwd_Ret_{h}"
        d=tbl.dropna(subset=[tgt]); cols=[c for c in sum(g.values(),[]) if c in d.columns and c!=tgt]
        if len(d)<600: continue
        cut=int(len(d)*(1-TEST)); tr,te=d.iloc[:cut-h],d.iloc[cut:]
        sc=StandardScaler().fit(tr[cols].values)
        Xtr,Xte=sc.transform(tr[cols].values),sc.transform(te[cols].values)
        ytr,yte=tr[tgt].values,te[tgt].values
        fitted={}
        for name,b in (("Ridge",mt.build_ridge),("RF",mt.build_random_forest_regularized),
                       ("XGB",mt.build_xgboost_regularized)):
            m=b(); m.fit(Xtr,ytr); fitted[name]=m.predict(Xte)
        naive=np.sqrt(np.mean(yte**2))
        def rmse(p): return float(np.sqrt(np.mean((yte-p)**2)))
        combos={"Ridge+RF+XGB":["Ridge","RF","XGB"],"Ridge+XGB (no RF)":["Ridge","XGB"],
                "RF alone":["RF"],"XGB alone":["XGB"],"Ridge alone":["Ridge"]}
        r={"ticker":t,"h":h,"naive":naive}
        for label,mem in combos.items():
            p=np.mean([fitted[m] for m in mem],axis=0)
            r[label]=(1-rmse(p)/naive)*100
        rows.append(r)
        print(f"  {t:<13} h={h:<3} " + "  ".join(f"{k} {r[k]:+.2f}" for k in combos))
df=pd.DataFrame(rows)
print("\n"+"="*78)
print("SKILL vs naive (percentage points, higher = better), averaged over 8 symbols")
print("="*78)
for c in ["Ridge+RF+XGB","Ridge+XGB (no RF)","RF alone","XGB alone","Ridge alone"]:
    print(f"  {c:<22} mean {df[c].mean():+6.2f}   median {df[c].median():+6.2f}   worst {df[c].min():+7.2f}")
d1,d2=df["Ridge+RF+XGB"],df["Ridge+XGB (no RF)"]
print(f"\n  removing RF changes skill by {(d2-d1).mean():+.2f} pts on average; "
      f"the full blend wins in {(d1>d2).sum()}/{len(df)} cases")
df.to_csv("rf_ablation.csv",index=False); print("wrote rf_ablation.csv")
