"""
scenarios.py
--------------
What a single "predicted price" hides.

At a one-month horizon the honest central forecast for a liquid market sits very
close to today's price -- but that does NOT mean the model expects nothing to
happen. It means the up-cases and down-cases roughly cancel. This module shows
the cases instead of the average:

  * `simulate_paths`  - thousands of possible price paths, built by block
                        bootstrapping the symbol's OWN daily returns, so real
                        volatility clustering, fat tails and gaps are preserved
                        (a plain normal random walk badly understates crashes).
  * `path_percentiles`- the fan: median, 25/75 and 5/95 envelopes per day.
  * `probabilities`   - chance of finishing higher, of a >=5% gain, of a >=5%
                        fall, of touching a target at any point on the way.

The simulation is anchored on the engine's own numbers: its central forecast
sets the mean of the paths, and its validated uncertainty sets the spread, so
the fan agrees with the 80% band the models produced rather than inventing a
second opinion.

Nothing here is a prediction of WHICH path happens. It is the distribution the
model implies, which is the part that carries information at these horizons.
"""

import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_SIMS = 3000
BLOCK = 10          # days per bootstrap block: long enough to keep vol clustering


def _historical_log_returns(raw: pd.DataFrame, lookback: int = 1260) -> np.ndarray:
    c = raw["Close"].dropna()
    r = np.log(c.clip(lower=EPS)).diff().dropna().values
    return r[-lookback:] if len(r) > lookback else r


def simulate_paths(raw: pd.DataFrame, n_days: int, target_return: float = 0.0,
                   target_sigma: float = None, n_sims: int = DEFAULT_SIMS,
                   seed: int = 7) -> np.ndarray:
    """
    (n_sims x n_days) array of simulated PRICES.

    target_return : where the model expects the price to be after n_days
                    (as a return); the paths are shifted so their mean lands there.
    target_sigma  : the model's own uncertainty for that horizon (std of the
                    n-day return); the paths are scaled to match it.
    """
    rng = np.random.default_rng(seed)
    hist = _historical_log_returns(raw)
    if len(hist) < 60:
        return np.empty((0, n_days))
    last = float(raw["Close"].iloc[-1])

    n_blocks = int(np.ceil(n_days / BLOCK))
    starts = rng.integers(0, max(len(hist) - BLOCK, 1), size=(n_sims, n_blocks))
    idx = starts[:, :, None] + np.arange(BLOCK)[None, None, :]
    paths = hist[np.clip(idx, 0, len(hist) - 1)].reshape(n_sims, -1)[:, :n_days]

    # re-centre and re-scale so the simulation agrees with the model
    total = paths.sum(axis=1)
    want_mu = np.log1p(target_return)
    paths = paths + (want_mu - total.mean()) / n_days
    if target_sigma and total.std() > EPS:
        scale = (np.log1p(target_sigma) / total.std())
        centre = paths.mean(axis=0, keepdims=True)
        paths = centre + (paths - centre) * np.clip(scale, 0.3, 3.0)
    return last * np.exp(np.cumsum(paths, axis=1))


def path_percentiles(paths: np.ndarray, levels=(5, 25, 50, 75, 95)) -> pd.DataFrame:
    """Per-day percentile envelopes of the simulated paths."""
    if paths.size == 0:
        return pd.DataFrame()
    return pd.DataFrame({f"p{l}": np.percentile(paths, l, axis=0) for l in levels})


def probabilities(paths: np.ndarray, last_price: float, thresholds=(0.05, 0.10),
                  target_price: float = None) -> dict:
    """Outcome odds implied by the simulation."""
    if paths.size == 0:
        return {}
    final = paths[:, -1]
    out = {
        "p_up": float((final > last_price).mean()) * 100,
        "median_return": float(np.median(final) / last_price - 1) * 100,
        "best_10pct": float(np.percentile(final, 90) / last_price - 1) * 100,
        "worst_10pct": float(np.percentile(final, 10) / last_price - 1) * 100,
        "expected_max_gain": float(np.median(paths.max(axis=1)) / last_price - 1) * 100,
        "expected_max_drop": float(np.median(paths.min(axis=1)) / last_price - 1) * 100,
    }
    for t in thresholds:
        out[f"p_gain_{int(t * 100)}"] = float((final >= last_price * (1 + t)).mean()) * 100
        out[f"p_fall_{int(t * 100)}"] = float((final <= last_price * (1 - t)).mean()) * 100
    # Standard risk numbers, read straight off the simulated distribution
    rets = final / last_price - 1
    out["var_95"] = float(np.percentile(rets, 5)) * 100          # 1-in-20 bad outcome
    out["expected_shortfall_95"] = float(rets[rets <= np.percentile(rets, 5)].mean()) * 100
    out["var_99"] = float(np.percentile(rets, 1)) * 100
    dd = (paths.min(axis=1) / last_price - 1)
    out["p_drawdown_10"] = float((dd <= -0.10).mean()) * 100     # touches -10% at any point
    out["p_drawdown_20"] = float((dd <= -0.20).mean()) * 100
    out["upside_downside_ratio"] = (abs(out["best_10pct"] / out["worst_10pct"])
                                    if out.get("worst_10pct") else None)

    if target_price:
        touched = (paths.max(axis=1) >= target_price) if target_price > last_price \
            else (paths.min(axis=1) <= target_price)
        out["p_touch_target"] = float(touched.mean()) * 100
        out["p_close_beyond_target"] = float(((final >= target_price) if target_price > last_price
                                              else (final <= target_price)).mean()) * 100
        # median number of days until the target is first touched, among paths that touch it
        if touched.any():
            sub = paths[touched]
            hit = (sub >= target_price) if target_price > last_price else (sub <= target_price)
            first = np.argmax(hit, axis=1) + 1
            out["median_days_to_target"] = float(np.median(first))
    return out


def regime_analogs(raw: pd.DataFrame, n_days: int, k: int = 40) -> dict:
    """
    What happened historically from days that looked like today.

    'Looked like' = closest match on four standardised state variables:
    distance from the 200-day average, 21-day realised volatility, 21-day
    momentum and the drawdown from the running peak. Purely descriptive -- it
    is history, not a forecast -- but it answers "has this setup happened
    before, and what followed?".
    """
    c = raw["Close"].dropna()
    if len(c) < 400:
        return {}
    log_ret = np.log(c.clip(lower=EPS)).diff()
    state = pd.DataFrame({
        "vs_sma200": c / c.rolling(200).mean() - 1,
        "vol21": log_ret.rolling(21).std() * np.sqrt(252),
        "mom21": c.pct_change(21),
        "drawdown": c / c.cummax() - 1,
    }).dropna()
    fwd = (c.shift(-n_days) / c - 1).reindex(state.index)
    usable = state.iloc[:-n_days] if n_days < len(state) else state
    fwd = fwd.loc[usable.index].dropna()
    usable = usable.loc[fwd.index]
    if len(usable) < 100:
        return {}
    z = (usable - usable.mean()) / (usable.std() + EPS)
    today = (state.iloc[-1] - usable.mean()) / (usable.std() + EPS)
    dist = np.sqrt(((z - today) ** 2).sum(axis=1))
    nearest = dist.nsmallest(min(k, len(dist))).index
    outcomes = fwd.loc[nearest]
    return {
        "n": int(len(outcomes)),
        "median": float(outcomes.median()) * 100,
        "mean": float(outcomes.mean()) * 100,
        "pct_up": float((outcomes > 0).mean()) * 100,
        "best": float(outcomes.max()) * 100,
        "worst": float(outcomes.min()) * 100,
        "p25": float(outcomes.quantile(0.25)) * 100,
        "p75": float(outcomes.quantile(0.75)) * 100,
        "dates": [d.strftime("%b %Y") for d in outcomes.sort_values().index[:3]],
    }
