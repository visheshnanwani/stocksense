"""
volatility_model.py
---------------------
Forecasting volatility, which -- unlike returns -- genuinely is predictable.

Returns are close to unforecastable in liquid markets; volatility is not. It
clusters (a violent week is usually followed by a violent one), it mean-reverts
over months, and it jumps after falls far more than after rallies. That makes
the *width* of a forecast the part of the problem worth attacking hardest, and
the width is what every probability on the dashboard depends on.

Model: HAR-RV (Corsi, 2009) with a leverage term, the standard workhorse for
realised volatility, fitted here on log volatility:

    log RV_forward(h)  ~  a + b1*log RV(1d) + b2*log RV(5d) + b3*log RV(22d)
                             + b4*log RV(66d) + b5*negative_return_share
                             + b6*|return|_recent

Three horizon windows capture the way short-, medium- and long-term traders each
leave their own footprint on volatility; the leverage terms capture the
asymmetry between falls and rallies. It is a linear model on logs, so it cannot
blow up, and log-space keeps it positive by construction.

Everything is fitted on past data only and predicted one step beyond, so the
comparison against the trailing-window estimate the engine used before is honest.

Usage:
    from volatility_model import HARVolatilityModel
    vm = HARVolatilityModel(h=21).fit(raw)
    vm.predict_latest(raw)        # forecast volatility of the next 21 days
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

EPS = 1e-12
TRADING_DAYS = 252


def realised_vol(raw: pd.DataFrame, window: int) -> pd.Series:
    """Annualised realised volatility over a trailing window."""
    log_ret = np.log(raw["Close"].clip(lower=EPS)).diff()
    return (log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)).replace(0, np.nan)


def har_features(raw: pd.DataFrame) -> pd.DataFrame:
    """The HAR cascade plus leverage, all known on the day they are dated."""
    log_ret = np.log(raw["Close"].clip(lower=EPS)).diff()
    f = pd.DataFrame(index=raw.index)
    for w, name in ((1, "d"), (5, "w"), (22, "m"), (66, "q")):
        rv = (log_ret.rolling(max(w, 2)).std() * np.sqrt(TRADING_DAYS)).replace(0, np.nan)
        f[f"log_rv_{name}"] = np.log(rv.clip(lower=1e-4))
    # leverage: volatility responds more to falls than to rallies
    f["neg_share_22"] = (log_ret < 0).rolling(22).mean()
    f["neg_mag_22"] = (-log_ret.clip(upper=0)).rolling(22).mean() * np.sqrt(TRADING_DAYS)
    f["abs_ret_5"] = log_ret.abs().rolling(5).mean() * np.sqrt(TRADING_DAYS)
    # range-based estimators are less noisy than close-to-close
    hl = np.log(raw["High"].clip(lower=EPS) / raw["Low"].clip(lower=EPS))
    f["log_parkinson_22"] = np.log(((hl ** 2).rolling(22).mean() / (4 * np.log(2))).clip(lower=1e-8) ** 0.5
                                   * np.sqrt(TRADING_DAYS))
    return f.replace([np.inf, -np.inf], np.nan)


class HARVolatilityModel:
    """Predicts the volatility of the NEXT h days. Falls back to the trailing
    window if it cannot be fitted or fails to beat that baseline."""

    def __init__(self, h: int = 21, ridge_alpha: float = 1.0):
        self.h = h
        self.ridge_alpha = ridge_alpha
        self.model = None
        self.cols = None
        self.val_rmse = None
        self.baseline_rmse = None
        self.skill = None          # % improvement over the trailing-window estimate

    # -- target: the volatility actually realised over the next h days --------
    def _target(self, raw: pd.DataFrame) -> pd.Series:
        log_ret = np.log(raw["Close"].clip(lower=EPS)).diff()
        fwd = log_ret.shift(-1).rolling(self.h).std() * np.sqrt(TRADING_DAYS)
        fwd = fwd.shift(-(self.h - 1))
        return np.log(fwd.clip(lower=1e-4))

    def fit(self, raw: pd.DataFrame, val_frac: float = 0.25):
        f = har_features(raw)
        y = self._target(raw)
        data = f.join(y.rename("y")).dropna()
        if len(data) < 300:
            return self
        self.cols = list(f.columns)
        cut = int(len(data) * (1 - val_frac))
        tr, va = data.iloc[:cut], data.iloc[cut:]
        if len(va) < 60:
            return self

        model = Ridge(alpha=self.ridge_alpha).fit(tr[self.cols].values, tr["y"].values)
        pred_va = model.predict(va[self.cols].values)
        self.val_rmse = float(np.sqrt(np.mean((pred_va - va["y"].values) ** 2)))
        # baseline the engine used before: "the next h days will be as volatile
        # as the last 21 were"
        self.baseline_rmse = float(np.sqrt(np.mean((va["log_rv_m"].values - va["y"].values) ** 2)))
        self.skill = (1 - self.val_rmse / self.baseline_rmse) * 100 if self.baseline_rmse else None

        # only keep the model if it beat that baseline out of sample
        if self.skill is not None and self.skill > 0:
            self.model = Ridge(alpha=self.ridge_alpha).fit(data[self.cols].values, data["y"].values)
        return self

    def predict_latest(self, raw: pd.DataFrame) -> float:
        """Forecast annualised volatility for the next h days (None if unfitted)."""
        f = har_features(raw).dropna()
        if self.model is None or self.cols is None or f.empty:
            return None
        return float(np.exp(self.model.predict(f[self.cols].values[-1:])[0]))

    def predict_series(self, raw: pd.DataFrame) -> pd.Series:
        f = har_features(raw)
        ok = f.dropna()
        if self.model is None or ok.empty:
            return pd.Series(index=raw.index, dtype=float)
        return pd.Series(np.exp(self.model.predict(ok[self.cols].values)), index=ok.index)

    def horizon_sigma(self, raw: pd.DataFrame) -> float:
        """The same forecast expressed as the std of an h-day RETURN, which is
        what a prediction band needs."""
        annual = self.predict_latest(raw)
        return None if annual is None else float(annual * np.sqrt(self.h / TRADING_DAYS))


def fit_horizon_vols(raw: pd.DataFrame, horizons=(1, 5, 10, 21, 63)) -> dict:
    """One HAR model per forecast horizon; the engine asks these for band widths."""
    out = {}
    for h in horizons:
        try:
            m = HARVolatilityModel(h=h).fit(raw)
            if m.model is not None:
                out[h] = m
        except Exception:
            continue
    return out
