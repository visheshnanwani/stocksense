"""
lab_configs.py
----------------
Candidate forecasting recipes, each isolating ONE change from the control so
model_lab can attribute any gain to the thing that caused it.

control      price/volume features, Ridge+RF+XGB, raw return target
+IND         adds the 71-column technical indicator bank
+VOL         predicts return / trailing volatility, then rescales
+MODELS      widens the ensemble (HistGradientBoosting, ExtraTrees, ElasticNet,
             Huber, kNN) -- everything scikit-learn/XGBoost offer that suits
             small, noisy, tabular financial data
+WEIGHTS     exponentially decayed sample weights, so 2015 matters less than 2026
+CONFORMAL   replaces the Gaussian 80% band with empirical residual quantiles
+DIRECTION   maps a classifier's P(up) into an expected return using the
             historical size of up and down moves

Shared machinery: an inner time-ordered validation slice (the last 20% of the
training window) decides member weights, the shrinkage alpha and the band. The
test set is never involved.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import (ElasticNet, HuberRegressor, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

EPS = 1e-12
Z80 = 1.2816


# ------------------------------------------------------------------ members
def base_members():
    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "rf": lambda: RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=20,
                                            n_jobs=-1, random_state=0),
        "xgb": lambda: XGBRegressor(n_estimators=160, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                    n_jobs=-1, random_state=0, verbosity=0),
    }


def wide_members():
    m = base_members()
    m.update({
        # sklearn's histogram booster is the LightGBM-style learner, no extra install
        "hgb": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03,
                                                     max_iter=200, min_samples_leaf=25,
                                                     l2_regularization=1.0, random_state=0),
        "et": lambda: ExtraTreesRegressor(n_estimators=150, max_depth=8, min_samples_leaf=20,
                                          n_jobs=-1, random_state=0),
        "enet": lambda: ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "huber": lambda: HuberRegressor(alpha=0.001, max_iter=500),
        "knn": lambda: KNeighborsRegressor(n_neighbors=50, weights="distance", n_jobs=-1),
    })
    return m


def fit_stack(pred_matrix, y, extra=None):
    """
    Non-negative least squares over member predictions (and any extra signals).

    Non-negativity matters: a member that is consistently wrong gets weight 0
    rather than a negative weight that would bet against it, which is the kind
    of thing that looks brilliant in-sample and explodes out of sample.
    Coefficients are in return units, so no separate shrinkage is needed -- a
    stack that finds nothing simply returns near-zero coefficients.
    """
    X = pred_matrix if extra is None else np.column_stack([pred_matrix, extra])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if ok.sum() < 200:
        return None
    model = LinearRegression(positive=True, fit_intercept=False)
    model.fit(X[ok], y[ok])
    fitted = model.predict(X[ok])
    # the same safety gate as everywhere else: beat "no change" or stand down
    if np.sqrt(np.mean((y[ok] - fitted) ** 2)) >= np.sqrt(np.mean(y[ok] ** 2)):
        return None
    return model


CONFIGS = {
    "control":     dict(indicators=False, vol_scale=False, members=base_members, weights=False,
                        conformal=False, direction=False),
    "+IND":        dict(indicators=True, vol_scale=False, members=base_members, weights=False,
                        conformal=False, direction=False),
    "+VOL":        dict(indicators=False, vol_scale=True, members=base_members, weights=False,
                        conformal=False, direction=False),
    "+MODELS":     dict(indicators=False, vol_scale=False, members=wide_members, weights=False,
                        conformal=False, direction=False),
    "+WEIGHTS":    dict(indicators=False, vol_scale=False, members=base_members, weights=True,
                        conformal=False, direction=False),
    "+CONFORMAL":  dict(indicators=False, vol_scale=False, members=base_members, weights=False,
                        conformal=True, direction=False),
    "+DIRECTION":  dict(indicators=False, vol_scale=False, members=base_members, weights=False,
                        conformal=False, direction=True),
    # stacking: fit the combination rather than averaging it
    "STACK":       dict(indicators=False, vol_scale=False, members=base_members, weights=False,
                        conformal=False, direction=False, stack=True),
    "STACK+WIDE":  dict(indicators=False, vol_scale=False, members=wide_members, weights=False,
                        conformal=False, direction=False, stack=True),
    "STACK+IND":   dict(indicators=True, vol_scale=False, members=wide_members, weights=False,
                        conformal=False, direction=False, stack=True),
    "STACK+SIG":   dict(indicators=False, vol_scale=False, members=wide_members, weights=False,
                        conformal=False, direction=False, stack=True, extra_signals=True),
    # --- combinations: the indicator bank was good for DIRECTION, volatility
    # scaling was good for CALIBRATION, and it should also stop the one blow-up
    # (a low-volatility ETF where unscaled targets produced huge predictions).
    "VOL+IND":     dict(indicators=True, vol_scale=True, members=base_members, weights=False,
                        conformal=False, direction=False),
    "VOL+IND+W":   dict(indicators=True, vol_scale=True, members=base_members, weights=True,
                        conformal=False, direction=False),
    "VOL+IND+W+M": dict(indicators=True, vol_scale=True, members=wide_members, weights=True,
                        conformal=False, direction=False),
    "VOL+IND+W+C": dict(indicators=True, vol_scale=True, members=base_members, weights=True,
                        conformal=True, direction=False),
}


def trailing_sigma(raw: pd.DataFrame, h: int, window: int = 21) -> pd.Series:
    """Volatility of an h-day move, estimated from the last `window` days only."""
    log_ret = np.log(raw["Close"].clip(lower=EPS)).diff()
    return (log_ret.rolling(window).std() * np.sqrt(h)).replace(0, np.nan)


def _recency_weights(n, half_life_years=2.0, per_year=252):
    """Exponential decay: a day two years ago counts half as much as today."""
    age = np.arange(n)[::-1] / per_year
    return 0.5 ** (age / half_life_years)


def _fit_members(X, y, members, sample_weight=None):
    out = {}
    for name, build in members().items():
        model = build()
        try:
            if sample_weight is not None and name not in ("knn", "enet"):
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
            out[name] = model
        except Exception:
            continue
    return out


def run_config(table, raw, target, h, train, test, cfg, indicator_cols=None, allowed_cols=None):
    """
    Fit one recipe on `train`, predict `test`. Returns (pred_ret, lo, hi, info).

    `allowed_cols` MUST be passed: the raw feature table also carries legacy
    helper columns (Target_Return, raw price levels) that would leak the answer
    -- the engine never sees them because it works off its registered feature
    groups, and neither may anything measured here.
    """
    pool = list(allowed_cols) if allowed_cols is not None else [
        c for c in table.columns if not c.startswith(("Fwd_Ret", "Fwd_Open", "Target_"))]
    if cfg["indicators"] and indicator_cols:
        pool = list(dict.fromkeys(list(pool) + list(indicator_cols)))
    elif indicator_cols:
        pool = [c for c in pool if c not in set(indicator_cols)]
    banned = ("Fwd_Ret", "Fwd_Open", "Target_")
    cols = [c for c in pool if c in train.columns and not c.startswith(banned)]
    cols = [c for c in cols if train[c].notna().mean() > 0.9 and train[c].std() > EPS]

    # inner validation slice: the most recent 20% of the training window,
    # with the h-day overlap purged so nothing leaks backwards
    cut = int(len(train) * 0.8)
    inner_tr, inner_va = train.iloc[:max(cut - h, 50)], train.iloc[cut:]
    if len(inner_va) < 40:
        inner_tr, inner_va = train.iloc[:-60], train.iloc[-60:]

    sigma = trailing_sigma(raw, h).reindex(table.index)
    scale_tr = sigma.reindex(inner_tr.index).fillna(sigma.median())
    scale_va = sigma.reindex(inner_va.index).fillna(sigma.median())
    scale_full = sigma.reindex(train.index).fillna(sigma.median())
    scale_te = sigma.reindex(test.index).fillna(sigma.median())

    def _y(frame, scale):
        y = frame[target].values
        return y / np.maximum(scale.values, EPS) if cfg["vol_scale"] else y

    scaler = StandardScaler().fit(inner_tr[cols].values)
    w = _recency_weights(len(inner_tr)) if cfg["weights"] else None
    members = _fit_members(scaler.transform(inner_tr[cols].values), _y(inner_tr, scale_tr),
                           cfg["members"], w)
    if not members:
        return None

    # member weights and shrinkage from the inner validation slice
    Xva = scaler.transform(inner_va[cols].values)
    yva = _y(inner_va, scale_va)
    preds = {n: m.predict(Xva) for n, m in members.items()}
    rmses = {n: np.sqrt(np.mean((p - yva) ** 2)) for n, p in preds.items()}
    inv = {n: 1 / max(r, EPS) for n, r in rmses.items()}
    total = sum(inv.values())
    weights = {n: v / total for n, v in inv.items()}
    blend_va = sum(weights[n] * preds[n] for n in members)
    alpha = float(np.clip((blend_va @ yva) / (blend_va @ blend_va + EPS), 0.0, 1.0))
    if np.sqrt(np.mean((yva - alpha * blend_va) ** 2)) >= np.sqrt(np.mean(yva ** 2)):
        alpha = 0.0                                   # no edge on validation -> defer

    stack_model, stack_names, stack_extra_cols = None, list(members), []
    if cfg.get("stack"):
        P_va = np.column_stack([preds[n] for n in stack_names])
        extra_va = None
        if cfg.get("extra_signals"):
            # signals of a different KIND: a trend score, a candle bias, a
            # volatility state. The members cannot express these on their own.
            stack_extra_cols = [c for c in ("Tech_Score_Feat", "S_Dist_SMA_20", "S_Vol_21",
                                            "Candle_Bias_3d")
                                if c in inner_va.columns]
            if stack_extra_cols:
                extra_va = inner_va[stack_extra_cols].values
        stack_model = fit_stack(P_va, yva, extra_va)

    resid_va = yva - alpha * blend_va
    if cfg["conformal"]:
        lo_q, hi_q = np.quantile(resid_va, [0.10, 0.90])   # empirical, can be asymmetric
    else:
        lo_q, hi_q = -Z80 * np.std(resid_va), Z80 * np.std(resid_va)

    # refit on the full training window, then predict the test set
    scaler_full = StandardScaler().fit(train[cols].values)
    w_full = _recency_weights(len(train)) if cfg["weights"] else None
    members_full = _fit_members(scaler_full.transform(train[cols].values), _y(train, scale_full),
                                cfg["members"], w_full)
    Xte = scaler_full.transform(test[cols].values)
    if stack_model is not None:
        P_te = np.column_stack([members_full[n].predict(Xte) for n in stack_names
                                if n in members_full])
        extra_te = test[stack_extra_cols].values if stack_extra_cols else None
        X_te = P_te if extra_te is None else np.column_stack([P_te, extra_te])
        pred = stack_model.predict(np.nan_to_num(X_te))
    else:
        blend_te = sum(weights[n] * members_full[n].predict(Xte) for n in members_full if n in weights)
        pred = alpha * blend_te
    lo, hi = pred + lo_q, pred + hi_q

    if cfg["direction"]:
        # P(up) from a classifier, turned into an expected return with the
        # historical average size of up moves and down moves
        yc = (train[target].values > 0).astype(int)
        if 0 < yc.mean() < 1:
            clf = LogisticRegression(max_iter=2000, C=0.1)
            clf.fit(scaler_full.transform(train[cols].values), yc)
            p_up = clf.predict_proba(Xte)[:, 1]
            up_mean = train[target][train[target] > 0].mean()
            dn_mean = train[target][train[target] <= 0].mean()
            dir_pred = p_up * up_mean + (1 - p_up) * dn_mean
            # blend 50/50 with the regression view, both already shrunk
            pred = 0.5 * pred + 0.5 * dir_pred
            lo, hi = pred + lo_q, pred + hi_q

    if cfg["vol_scale"]:
        s = np.maximum(scale_te.values, EPS)
        pred, lo, hi = pred * s, lo * s, hi * s

    return pred, lo, hi, {"alpha": round(alpha, 3), "n_features": len(cols),
                          "members": ",".join(members_full)}
