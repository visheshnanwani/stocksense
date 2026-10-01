"""
ensemble_accuracy.py
----------------------
Prints the accuracy of ALL models combined into one ensemble, evaluated
fairly on a held-out test set they never saw during training. This is
the number to show your professor when asked "what's the accuracy of
all the models combined."

Run:
    python ensemble_accuracy.py
"""

import pandas as pd
from data_loader import load_stock_data
import dashboard_logic as dl
from stock_search import resolve_ticker

TICKER = None                  # None = ask for a stock name when you run the script (e.g. set "MSFT" to skip the prompt)
START_DATE = "2014-01-01"
END_DATE = "2025-01-01"


def main():
    global TICKER
    TICKER = resolve_ticker(TICKER)
    print(f"Loading {TICKER} data...")
    raw = load_stock_data(TICKER, START_DATE, END_DATE)

    print("Training all 5 models and evaluating the combined ensemble on held-out test data...")
    metrics, weights = dl.evaluate_ensemble_accuracy(raw)

    print("\n===== COMBINED ENSEMBLE ACCURACY (held-out test set) =====")
    print(pd.DataFrame([metrics]).to_string(index=False))

    print("\nModel weights in the ensemble:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.3f}")


if __name__ == "__main__":
    main()
