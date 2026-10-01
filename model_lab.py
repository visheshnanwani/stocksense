"""
model_lab.py
--------------
The measurement bench used to decide what actually goes into the forecast
engine. Every idea -- a new indicator, a new learner, a new target transform --
is judged here, the same way, on data the fitting never saw.

Why this file exists: it is very easy to make a forecast *look* better (fit the
same folds you tuned on, or add an upward drift) and very hard to make it
*be* better. Everything in the engine had to win here first.

Protocol for one (symbol, horizon, configuration):

    |<---------------- fit / select ---------------->|  purge  |<-- test -->|
                                                      h days      last 15%

  * The last 15% of history is the test set and is never touched while fitting,
    selecting features or tuning anything.
  * Rows whose h-day outcome would overlap the test window are PURGED from
    training, so no label leaks across the boundary.
  * Inside the training part, group selection and coefficients use the engine's
    own walk-forward folds.

Reported for each run:

  skill      % reduction in RMSE vs the naive "price stays put" forecast.
             Positive = better than assuming no change. This is the headline.
  dir_acc    direction accuracy on the rows where the model actually took a
             view (|prediction| > 0).
  cover80    what fraction of outcomes landed inside the 80% band. 80 is
             perfect; 95 means the bands are too wide to be useful, 60 means
             they are dangerously narrow.
  move       average |predicted move|, so we can see whether a configuration
             is actually saying anything or just echoing today's price.

Usage:
    python model_lab.py                      # baseline over the default panel
    python model_lab.py MSFT RELIANCE.NS     # specific symbols
"""

import datetime
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import data_loader as dl
import forecast_engine as fe

# A panel deliberately mixed across asset classes, countries and volatility
# levels -- an idea that only helps US mega-caps is not an improvement.
PANEL = ["MSFT", "AAPL", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS",
         "^NSEI", "GC=F", "CL=F", "BTC-USD", "GOLDBEES.NS", "EURUSD=X"]
HORIZONS = (1, 5, 21)
TEST_FRACTION = 0.15
START = "2014-01-01"


def load(ticker, start=START, end=None):
    return dl.load_stock_data(ticker, start, end or datetime.date.today().strftime("%Y-%m-%d"))


def split(table, target, h, test_fraction=TEST_FRACTION):
    """(train, test) with the h-day overlap purged from the end of train."""
    labeled = table[table[target].notna()]
    if len(labeled) < 400:
        return None, None
    pos = pd.Series(np.arange(len(table)), index=table.index)
    test_start_pos = int(len(labeled) * (1 - test_fraction))
    test_start = labeled.index[test_start_pos]
    lab_pos = pos.reindex(labeled.index)
    train = labeled[(lab_pos + h) < pos[test_start]]
    test = labeled[labeled.index >= test_start]
    return train, test


def score_predictions(pred_ret, actual_ret, lower=None, upper=None):
    """skill vs naive, direction accuracy, band coverage, average move."""
    pred_ret = np.asarray(pred_ret, dtype=float)
    actual_ret = np.asarray(actual_ret, dtype=float)
    naive_rmse = float(np.sqrt(np.mean(actual_ret ** 2)))
    rmse = float(np.sqrt(np.mean((actual_ret - pred_ret) ** 2)))
    out = {
        "skill": (1 - rmse / naive_rmse) * 100 if naive_rmse else np.nan,
        "move": float(np.mean(np.abs(pred_ret))) * 100,
        "rmse": rmse,
    }
    moving = np.abs(pred_ret) > 1e-9
    out["dir_acc"] = (float(np.mean(np.sign(pred_ret[moving]) == np.sign(actual_ret[moving]))) * 100
                      if moving.any() else np.nan)
    if lower is not None and upper is not None:
        inside = (actual_ret >= np.asarray(lower)) & (actual_ret <= np.asarray(upper))
        out["cover80"] = float(np.mean(inside)) * 100
    return out


def run_baseline(ticker, horizons=HORIZONS, extra=None, news_df=None):
    """The engine exactly as it ships today -- the number every idea must beat."""
    raw = load(ticker)
    table, groups = fe.build_feature_table(raw, ticker, extra, news_df)
    rows = []
    for h in horizons:
        target = f"Fwd_Ret_{h}"
        train, test = split(table, target, h)
        if train is None or len(test) < 30:
            continue
        chosen, cols, _ = fe.select_groups(train, groups, target)
        s = fe._cv_score(train, cols, target, fe._full_members)
        model = fe.HorizonModel(cols, s["weights"], s["alpha"], s["sigma"], chosen, [],
                                s["rmse"], s["naive_rmse"]).fit_final(train, target)
        pred = model.predict(test)
        band = fe.Z80 * s["sigma"]
        m = score_predictions(pred, test[target].values, pred - band, pred + band)
        m.update(symbol=ticker, h=h, config="baseline", alpha=round(s["alpha"], 3),
                 groups=",".join(chosen), n_test=len(test))
        rows.append(m)
    return rows


def report(rows, key="config"):
    """Per-configuration summary across the whole panel."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (df.groupby([key, "h"])
             .agg(skill=("skill", "mean"), dir_acc=("dir_acc", "mean"),
                  cover80=("cover80", "mean"), move=("move", "mean"), n=("skill", "size"))
             .round(2).reset_index())
    return agg


if __name__ == "__main__":
    symbols = sys.argv[1:] or PANEL
    all_rows = []
    for t in symbols:
        try:
            r = run_baseline(t)
        except Exception as e:
            print(f"{t:14} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        all_rows += r
        for m in r:
            print(f"{t:14} h={m['h']:>3} skill={m['skill']:>6.2f} dir={m.get('dir_acc', float('nan')):>5.1f} "
                  f"cover={m.get('cover80', float('nan')):>5.1f} move={m['move']:>5.2f} "
                  f"alpha={m['alpha']:.2f}", flush=True)
    print("\n=== panel summary (baseline) ===")
    print(report(all_rows).to_string(index=False))
