"""
exp_drift_target.py
-------------------
The flat-forecast question, asked properly.

WHY THE LINE IS FLAT
    predict() returns  alpha * blended_signal .
    alpha is fitted so the shrunk signal beats a NO-CHANGE forecast, and it is
    forced to 0 unless it wins in at least 2 of 3 folds. For MSFT at h=21 the
    model's own validation numbers are

        val_rmse       0.075963
        val_naive_rmse 0.076106      -> it beats "unchanged" by 0.19%

    so alpha comes out at 0.097 and 90% of the model's view is thrown away.
    The line is flat because the model genuinely has almost no edge on the MEAN
    of a 21-day large-cap return. That part is honest and should stay.

THE ACTUAL FLAW
    Shrinking toward ZERO assumes the no-skill forecast is "the price does not
    move". It isn't. A stock's unconditional 21-day return is POSITIVE. When the
    conditional signal is weak, the right fallback is the unconditional drift,
    not zero -- shrink toward the grand mean, not toward the origin.

    So:   current   y = a * f(x)
          proposed  y = mu + a * (f(x) - mean_train(f))

    mu is estimated on TRAINING data only, inside each fold, and never sees the
    evaluation block.

VARIANTS MEASURED
    zero      current behaviour
    drift     shrink toward the symbol's own training-period mean return
    drift_js  shrink toward a James-Stein blend of the symbol's mean and the
              cross-sectional mean (a single symbol's mean is a noisy estimate)
    driftonly mu alone, no conditional signal at all -- the control that says
              whether the signal is contributing anything

Scored on a held-out tail that no fitting touched, with a purge gap of h days.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data_loader as dl
import forecast_engine as fe

TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "JPM", "XOM", "KO", "PG",
           "HDFCBANK.NS", "TCS.NS", "ITC.NS", "RELIANCE.NS", "INFY.NS",
           "GC=F", "CL=F", "BTC-USD"]
HORIZONS = [5, 21, 63]
TEST_FRAC = 0.25
N_FOLDS = 3


def fit_variant(tr, te, cols, target, h, cross_mu):
    """Return {variant: predictions on te} plus diagnostics."""
    # --- inner walk-forward on the TRAINING block only: weights + alpha -------
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

    # alpha against a ZERO target (what the engine does today)
    a_zero = float(np.clip((blended @ y) / (blended @ blended + 1e-12), 0.0, 1.0))
    wins = sum(np.sqrt(np.mean((fy - a_zero * (w @ fp)) ** 2)) < np.sqrt(np.mean(fy ** 2))
               for fp, fy in zip(fold_preds, fold_y))
    if wins < 2:
        a_zero = 0.0

    # alpha against a DRIFT target: demean both sides, then shrink
    mu = float(y.mean())
    fbar = float(blended.mean())
    bd, yd = blended - fbar, y - mu
    a_drift = float(np.clip((bd @ yd) / (bd @ bd + 1e-12), 0.0, 1.0))
    wins_d = sum(np.sqrt(np.mean((fy - (mu + a_drift * (w @ fp - fbar))) ** 2))
                 < np.sqrt(np.mean((fy - mu) ** 2))
                 for fp, fy in zip(fold_preds, fold_y))
    if wins_d < 2:
        a_drift = 0.0

    # James-Stein: pull the symbol's own mean toward the cross-sectional mean.
    # The shrink weight follows the noise in mu: var/n over the spread of means.
    se2 = float(y.var(ddof=1)) / max(len(y), 1)
    spread = max((mu - cross_mu) ** 2, 1e-12)
    k = float(np.clip(se2 / (se2 + spread), 0.0, 1.0))
    mu_js = k * cross_mu + (1 - k) * mu

    # --- refit members on the whole training block, predict the held-out tail -
    scaler = StandardScaler().fit(tr[cols].values)
    Xtr, Xte = scaler.transform(tr[cols].values), scaler.transform(te[cols].values)
    members = fe._full_members()
    items = members.items() if isinstance(members, dict) else [(type(m).__name__, m) for m in members]
    preds_te, names = [], []
    for n, m in items:
        m.fit(Xtr, tr[target].values)
        preds_te.append(m.predict(Xte))
        names.append(n)
    f_te = np.array([w[i] * preds_te[i] for i in range(len(preds_te))]).sum(axis=0)

    return {
        "zero":      a_zero * f_te,
        "drift":     mu + a_drift * (f_te - fbar),
        "drift_js":  mu_js + a_drift * (f_te - fbar),
        "driftonly": np.full(len(te), mu),
    }, {"a_zero": a_zero, "a_drift": a_drift, "mu": mu, "mu_js": mu_js, "k": k}


def main():
    # cross-sectional mean h-day return, from an early slice only (no look-ahead
    # into anyone's evaluation window)
    rows = []
    tables = {}
    for t in TICKERS:
        try:
            raw = dl.load_stock_data(t, "2014-01-01", "2026-09-28")
            tbl, groups = fe.build_feature_table(raw, t)
            tables[t] = (tbl, groups)
        except Exception as e:
            print(f"  skip {t}: {type(e).__name__}: {e}")

    for h in HORIZONS:
        target = f"Fwd_Ret_{h}"
        # cross-sectional drift from the TRAINING portion of every symbol
        mus = []
        for t, (tbl, _) in tables.items():
            d = tbl.dropna(subset=[target])
            if len(d) < 400:
                continue
            cut = int(len(d) * (1 - TEST_FRAC))
            mus.append(float(d[target].iloc[:cut].mean()))
        cross_mu = float(np.mean(mus)) if mus else 0.0
        print(f"\n{'=' * 78}\nHORIZON {h}   cross-sectional training drift = {cross_mu * 100:+.3f}%\n{'=' * 78}")
        print(f"{'ticker':<13}{'a_zero':>8}{'a_drift':>9}{'mu%':>8}"
              f"{'RMSE zero':>11}{'RMSE drift':>12}{'RMSE d_js':>11}{'RMSE muonly':>13}{'naive':>9}")

        for t, (tbl, groups) in tables.items():
            d = tbl.dropna(subset=[target])
            cols = sum(groups.values(), [])
            cols = [c for c in cols if c in d.columns and c != target]
            if len(d) < 500:
                continue
            cut = int(len(d) * (1 - TEST_FRAC))
            tr, te = d.iloc[:cut - h], d.iloc[cut:]      # purge h rows
            try:
                preds, diag = fit_variant(tr, te, cols, target, h, cross_mu)
            except Exception as e:
                print(f"{t:<13} failed: {type(e).__name__}: {e}")
                continue
            actual = te[target].values
            naive = float(np.sqrt(np.mean(actual ** 2)))
            r = {k: float(np.sqrt(np.mean((actual - v) ** 2))) for k, v in preds.items()}
            print(f"{t:<13}{diag['a_zero']:>8.3f}{diag['a_drift']:>9.3f}{diag['mu'] * 100:>8.2f}"
                  f"{r['zero']:>11.5f}{r['drift']:>12.5f}{r['drift_js']:>11.5f}"
                  f"{r['driftonly']:>13.5f}{naive:>9.5f}")
            rows.append({"ticker": t, "h": h, "naive": naive, **r,
                         "dir_zero": float(np.mean(np.sign(preds["zero"]) == np.sign(actual))),
                         "dir_drift": float(np.mean(np.sign(preds["drift"]) == np.sign(actual))),
                         **diag})

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 78}\nSUMMARY -- skill = 1 - RMSE/naive, averaged over symbols\n{'=' * 78}")
    print(f"{'h':>4}{'zero':>10}{'drift':>10}{'drift_js':>11}{'muonly':>10}"
          f"{'dir zero':>10}{'dir drift':>11}{'n':>4}")
    for h in HORIZONS:
        s = df[df.h == h]
        if s.empty:
            continue
        sk = lambda c: (1 - s[c] / s["naive"]).mean() * 100
        print(f"{h:>4}{sk('zero'):>10.3f}{sk('drift'):>10.3f}{sk('drift_js'):>11.3f}"
              f"{sk('driftonly'):>10.3f}{s['dir_zero'].mean() * 100:>10.1f}"
              f"{s['dir_drift'].mean() * 100:>11.1f}{len(s):>4}")
    print(f"\n{'ALL':>4}", end="")
    sk = lambda c: (1 - df[c] / df["naive"]).mean() * 100
    print(f"{sk('zero'):>10.3f}{sk('drift'):>10.3f}{sk('drift_js'):>11.3f}{sk('driftonly'):>10.3f}"
          f"{df['dir_zero'].mean() * 100:>10.1f}{df['dir_drift'].mean() * 100:>11.1f}{len(df):>4}")
    df.to_csv("exp_drift_target.csv", index=False)
    print("\nwrote exp_drift_target.csv")


if __name__ == "__main__":
    main()
