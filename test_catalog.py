"""
test_catalog.py
---------------
Every symbol the UI offers, actually loaded.

A catalog entry is a promise: the symbol appears in search or in the quick picks,
so clicking it must produce an analysis. This walks the whole catalog and checks
that promise end to end -- load, repair, validate, profile -- and prints exactly
which symbols cannot keep it, so dead tickers get removed rather than shipped.

Run:  python test_catalog.py           (whole catalog, slow)
      python test_catalog.py --quick   (quick picks only)
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import datetime

import data_loader as dl
import market_data as md
import stock_search as ss

MIN_ROWS = 200


def universe(quick_only=False):
    """Every symbol the UI can offer: the quick-pick buttons plus the search catalog."""
    out, seen = [], []
    qp = [t for group in getattr(ss, "QUICK_PICKS", {}).values() for t in group]
    labels = getattr(ss, "QUICK_LABELS", {})
    for t in qp:
        if t not in seen:
            seen.append(t)
            out.append((t, labels.get(t, t)))
    if quick_only:
        return out
    for name in ("POPULAR_ASSETS", "POPULAR_STOCKS", "POPULAR_INDIAN", "POPULAR_US"):
        for row in getattr(ss, name, []) or []:
            t = row[0] if isinstance(row, (tuple, list)) else row
            if t not in seen:
                seen.append(t)
                out.append((t, row[1] if isinstance(row, (tuple, list)) and len(row) > 1 else t))
    return out


def main():
    quick = "--quick" in sys.argv
    syms = universe(quick)
    print(f"checking {len(syms)} symbols\n")
    ok, dead, thin, warned = [], [], [], []
    for t, label in syms:
        try:
            d = dl.load_stock_data(t, "2014-01-01", datetime.date.today().strftime("%Y-%m-%d"))
        except Exception as e:
            dead.append((t, label, f"{type(e).__name__}: {str(e)[:60]}"))
            print(f"  DEAD  {t:<16} {str(label)[:34]:<34} {str(e)[:50]}")
            continue
        if d is None or len(d) == 0:
            dead.append((t, label, "no rows"))
            print(f"  DEAD  {t:<16} {str(label)[:34]:<34} no rows")
            continue
        d, reps = md.repair_market_data(d, t)
        try:
            rep = md.validate_market_data(d, t, "1d")
        except md.MarketDataError as e:
            dead.append((t, label, f"validation: {str(e)[:60]}"))
            print(f"  BLOCK {t:<16} {str(label)[:34]:<34} {str(e)[:50]}")
            continue
        if len(d) < MIN_ROWS:
            thin.append((t, label, len(d)))
            print(f"  THIN  {t:<16} {str(label)[:34]:<34} {len(d)} rows")
            continue
        note = ""
        if reps or rep.get("warnings"):
            warned.append((t, label, reps + rep.get("warnings", [])))
            note = "  [" + "; ".join(reps + rep.get("warnings", []))[:58] + "]"
        ok.append(t)
        print(f"  ok    {t:<16} {str(label)[:34]:<34} {len(d):>5} rows{note}")

    print(f"\n{'=' * 78}")
    print(f"ok {len(ok)}   repaired/warned {len(warned)}   thin {len(thin)}   "
          f"dead-or-blocked {len(dead)}   of {len(syms)}")
    if dead:
        print("\nSYMBOLS THAT CANNOT BE ANALYSED -- remove or replace these:")
        for t, label, why in dead:
            print(f"  {t:<16} {str(label)[:40]:<40} {why}")
    if thin:
        print("\nTOO LITTLE HISTORY TO MODEL (live view only):")
        for t, label, n in thin:
            print(f"  {t:<16} {str(label)[:40]:<40} {n} rows")
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
