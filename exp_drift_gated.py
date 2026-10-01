"""
exp_drift_gated.py
------------------
Round 2 on the shrinkage target.

Round 1 (exp_drift_target.py) came back genuinely mixed:

    MEDIAN skill   h=5  zero +0.00  drift +1.04
                   h=21 zero +0.00  drift +1.04
                   h=63 zero +0.06  drift +2.36
    MEAN   skill        zero +0.69  drift +0.03
    win rate            drift beat zero in 25/48 (52%)

So drift helps the typical symbol but occasionally fails hard -- ITC.NS at h=63
lost 25 points -- and those failures eat the average. A coin-flip win rate is
not something to ship.

The failures share a cause: the training-period mean return did not persist.
Extrapolating a drift you measured is only safe when that drift is actually
distinguishable from noise. So gate it.

    t = mu / (sd / sqrt(n_eff)),  n_eff = n / h

n_eff, not n, because Fwd_Ret_h on daily rows overlaps h-fold -- using n would
overstate significance by sqrt(h) and wave through exactly the drifts that
later fail. Variants:

    zero        current behaviour, shrink toward no-change
    drift       always shrink toward the training mean
    gated_t2    shrink toward the mean only if |t| > 2, else toward zero
    gated_t1    same, looser bar at |t| > 1
    gated_stab  |t| > 1 AND the mean has the same sign in every training fold
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data_loader as dl
import forecast_engine as fe

from exp_drift_target import TICKERS, HORIZONS, TEST_FRAC, N_FOLDS


def run(tr, te, cols, target, h):
    folds = fe._folds(tr, N_FOLDS)
    fold_preds, fold_y = [], []
    for a, b in folds:
        _, preds = fe._fit_predict(a, b, cols, target, fe._full_members)
        fold_preds.append(np.vstack(preds))
        fold_y.append(b[target].values)
    P, y = np.hstack(fold_preds), np.concatenate(fold_y)
    rm = np.sqrt(((P - y) ** 2).mean(axis=1))
    w = 1 / np.maximum(rm, 1e-12)
    w = w / w.sum()
    blended = w @ P

    a_zero = float(np.clip((blended @ y) / (blended @ blended + 1e-12), 0.0, 1.0))
    if sum(np.sqrt(np.mean((fy - a_zero * (w @ fp)) ** 2)) < np.sqrt(np.mean(fy ** 2))
           for fp, fy in zip(fold_preds, fold_y)) < 2:
        a_zero = 0.0

    mu = float(y.mean())
    fbar = float(blended.mean())
    bd, yd = blended - fbar, y - mu
    a_drift = float(np.clip((bd @ yd) / (bd @ bd + 1e-12), 0.0, 1.0))
    if sum(np.sqrt(np.mean((fy - (mu + a_drift * (w @ fp - fbar))) ** 2))
           < np.sqrt(np.mean((fy - mu) ** 2)) for fp, fy in zip(fold_preds, fold_y)) < 2:
        a_drift = 0.0

    # --- is this drift real? overlapping returns -> effective sample size n/h
    n_eff = max(len(y) / h, 2.0)
    sd = float(y.std(ddof=1))
    t = mu / (sd / np.sqrt(n_eff)) if sd > 0 else 0.0
    same_sign = len({np.sign(float(fy.mean())) for fy in fold_y}) == 1

    scaler = StandardScaler().fit(tr[cols].values)
    Xtr, Xte = scaler.transform(tr[cols].values), scaler.transform(te[cols].values)
    members = fe._full_members()
    items = members.items() if isinstance(members, dict) else [(type(m).__name__, m) for m in members]
    pr = []
    for n_, m in items:
        m.fit(Xtr, tr[target].values)
        pr.append(m.predict(Xte))
    f_te = np.array([w[i] * pr[i] for i in range(len(pr))]).sum(axis=0)

    zero = a_zero * f_te
    drift = mu + a_drift * (f_te - fbar)
    out = {
        "zero": zero,
        "drift": drift,
        "gated_t2": drift if abs(t) > 2 else zero,
        "gated_t1": drift if abs(t) > 1 else zero,
        "gated_stab": drift if (abs(t) > 1 and same_sign) else zero,
    }
    return out, {"t": t, "mu": mu, "same_sign": same_sign,
                 "a_zero": a_zero, "a_drift": a_drift, "n_eff": n_eff}


def main():
    rows = []
    for t_ in TICKERS:
        try:
            raw = dl.load_stock_data(t_, "2014-01-01", "2026-09-28")
            tbl, groups = fe.build_feature_table(raw, t_)
        except Exception as e:
            print(f"  skip {t_}: {type(e).__name__}")
            continue
        for h in HORIZONS:
            target = f"Fwd_Ret_{h}"
            if target not in tbl:
                continue
            d = tbl.dropna(subset=[target])
            cols = [c for c in sum(groups.values(), []) if c in d.columns and c != target]
            if len(d) < 500:
                continue
            cut = int(len(d) * (1 - TEST_FRAC))
            tr, te = d.iloc[:cut - h], d.iloc[cut:]
            try:
                preds, diag = run(tr, te, cols, target, h)
            except Exception as e:
                print(f"  {t_} h={h} failed: {type(e).__name__}: {e}")
                continue
            actual = te[target].values
            naive = float(np.sqrt(np.mean(actual ** 2)))
            rows.append({"ticker": t_, "h": h, "naive": naive, **diag,
                         **{k: float(np.sqrt(np.mean((actual - v) ** 2))) for k, v in preds.items()}})
            print(f"  {t_:<12} h={h:<3} t={diag['t']:+6.2f} mu={diag['mu'] * 100:+6.2f}%"
                  f" stable={str(diag['same_sign']):<5}")

    df = pd.DataFrame(rows)
    VAR = ["zero", "drift", "gated_t2", "gated_t1", "gated_stab"]
    print(f"\n{'=' * 84}\nSKILL vs naive (1 - RMSE/naive), in percentage points\n{'=' * 84}")
    print(f"{'':>12}" + "".join(f"{v:>13}" for v in VAR))
    for h in HORIZONS:
        s = df[df.h == h]
        if s.empty:
            continue
        print(f"{'h=' + str(h) + ' mean':>12}" + "".join(f"{(1 - s[v] / s.naive).mean() * 100:>13.3f}" for v in VAR))
        print(f"{'median':>12}" + "".join(f"{(1 - s[v] / s.naive).median() * 100:>13.3f}" for v in VAR))
    print(f"\n{'ALL mean':>12}" + "".join(f"{(1 - df[v] / df.naive).mean() * 100:>13.3f}" for v in VAR))
    print(f"{'median':>12}" + "".join(f"{(1 - df[v] / df.naive).median() * 100:>13.3f}" for v in VAR))
    print(f"{'worst':>12}" + "".join(f"{(1 - df[v] / df.naive).min() * 100:>13.3f}" for v in VAR))
    print(f"{'beats zero':>12}" + "".join(
        f"{((1 - df[v] / df.naive) > (1 - df.zero / df.naive)).sum():>10}/{len(df):<3}" for v in VAR))
    print(f"\ngate fired: t>2 in {(df.t.abs() > 2).sum()}/{len(df)},  "
          f"t>1 in {(df.t.abs() > 1).sum()}/{len(df)},  "
          f"t>1 & stable in {((df.t.abs() > 1) & df.same_sign).sum()}/{len(df)}")
    df.to_csv("exp_drift_gated.csv", index=False)
    print("wrote exp_drift_gated.csv")


if __name__ == "__main__":
    main()
