"""
exp_speed.py
------------
The dashboard is not broken for commodities, it is slow enough that people give
up -- "graphs are not loading". Copper took 454s on a loaded machine. This
measures where the time goes and whether the obvious fixes are real.

The five horizons are independent: each does its own greedy group selection
(~12 CV runs x 3 folds) and then one final fit. That is embarrassingly parallel,
but every learner inside is already asking for all 16 cores, so naive nesting
makes it worse rather than better. Variants:

    baseline    as shipped: sequential horizons, inner n_jobs=-1
    threads     horizons on a thread pool, inner n_jobs=-1  (oversubscribed)
    threads_1   horizons on a thread pool, inner n_jobs=1
    threads_2   horizons on a thread pool, inner n_jobs=2

Identical forecasts are required, not just a faster wall clock: a speedup that
changes the answer is not a speedup.
"""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np

import data_loader as dl
import forecast_engine as fe
import market_data as md
import models_tabular as mt

TICKERS = ["MSFT", "HG=F"]


def set_jobs(n):
    """Rebuild the ensemble builders with a fixed thread budget per learner."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    mt.ENSEMBLE_MODEL_BUILDERS["Random Forest"] = lambda random_state=42: RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=50, random_state=random_state, n_jobs=n)
    mt.ENSEMBLE_MODEL_BUILDERS["XGBoost"] = lambda random_state=42: XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=5.0, objective="reg:squarederror",
        random_state=random_state, n_jobs=n, verbosity=0)
    fe._fast_members = lambda: [Ridge(alpha=10.0),
                                XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                                             reg_lambda=5.0, n_jobs=n, random_state=42, verbosity=0)]


def fit_sequential(table, groups, raw, horizons):
    out = {}
    for h in horizons:
        target = f"Fwd_Ret_{h}"
        lab = table[table[target].notna()]
        if len(lab) < 300:
            continue
        out[h] = fe._train_horizon(lab, dict(groups), target,
                                   vol=fe.trailing_vol(raw, h)).fit_final(lab, target)
    return out


def fit_threaded(table, groups, raw, horizons, workers=5):
    def one(h):
        target = f"Fwd_Ret_{h}"
        lab = table[table[target].notna()]
        if len(lab) < 300:
            return h, None
        return h, fe._train_horizon(lab, dict(groups), target,
                                    vol=fe.trailing_vol(raw, h)).fit_final(lab, target)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return {h: m for h, m in ex.map(one, horizons) if m is not None}


def preds(models, table):
    row = table.iloc[[-1]]
    return {h: float(m.predict(row)[0]) for h, m in sorted(models.items())}


def main():
    for t in TICKERS:
        raw = dl.load_stock_data(t, "2014-01-01", "2026-09-28")
        raw, _ = md.repair_market_data(raw, t)
        table, groups = fe.build_feature_table(raw, t)
        horizons = [1, 5, 10, 21, 63]
        print(f"\n{'=' * 72}\n{t}   rows={len(table)}  groups={len(groups)}\n{'=' * 72}")
        results = {}
        for label, jobs, mode in (("baseline", -1, "seq"), ("threads", -1, "thr"),
                                  ("threads_1", 1, "thr"), ("threads_2", 2, "thr")):
            set_jobs(jobs)
            s = time.time()
            m = fit_sequential(table, groups, raw, horizons) if mode == "seq" \
                else fit_threaded(table, groups, raw, horizons)
            el = time.time() - s
            p = preds(m, table)
            results[label] = (el, p)
            print(f"  {label:<11} {el:7.1f}s   " +
                  "  ".join(f"h{h}={v * 100:+.3f}%" for h, v in p.items()))
        base_t, base_p = results["baseline"]
        print(f"\n  {'variant':<11}{'speedup':>9}{'max |pred diff|':>18}")
        for label, (el, p) in results.items():
            d = max((abs(p.get(h, 0) - base_p.get(h, 0)) for h in base_p), default=0.0)
            print(f"  {label:<11}{base_t / el:>8.2f}x{d * 100:>17.6f}pp")


if __name__ == "__main__":
    main()
