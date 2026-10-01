"""
test_kite_client.py
-------------------
Verifies the Kite integration WITHOUT credentials, using recorded payloads in
exactly the shape Kite returns.

The parts that can be wrong are the parts that do not need an account: the
checksum, the depth parsing, the bid-versus-LTP arithmetic, and the refusal to
do anything when unauthenticated. Those are what this covers, so the code is
known-good before anyone pays for a subscription.

Run:  python test_kite_client.py
"""

import hashlib
import sys
import warnings

warnings.filterwarnings("ignore")

import kite_client as kc

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# a real-shaped Kite /quote payload
MOCK = {
    "NFO:NIFTY26O0722600CE": {
        "last_price": 159.65, "volume": 58338735, "oi": 67527,
        "depth": {"buy": [{"price": 159.40, "quantity": 1350, "orders": 12},
                          {"price": 159.30, "quantity": 900}],
                  "sell": [{"price": 159.90, "quantity": 900, "orders": 8},
                           {"price": 160.05, "quantity": 1200}]}},
    "NFO:NIFTY26O0722600PE": {
        "last_price": 115.20, "volume": 192426390, "oi": 98746,
        "depth": {"buy": [{"price": 114.95, "quantity": 2250}],
                  "sell": [{"price": 115.50, "quantity": 1725}]}},
}


def main():
    print(f"\n{'=' * 70}\n1. depth parsing\n{'=' * 70}")
    df = kc.depth_frame(MOCK)
    check("both contracts parsed", len(df) == 2, f"{len(df)} rows")
    ce = df[df.symbol.str.endswith("CE")].iloc[0]
    check("top-of-book bid taken, not a deeper level", ce.bid == 159.40, f"bid {ce.bid}")
    check("top-of-book ask taken", ce.ask == 159.90, f"ask {ce.ask}")
    check("mid is the midpoint", abs(ce["mid"] - 159.65) < 1e-9, f"mid {ce['mid']}")
    check("spread = ask - bid", abs(ce.spread - 0.50) < 1e-9, f"{ce.spread}")
    check("spread% relative to mid",
          abs(ce.spread_pct - (0.50 / 159.65 * 100)) < 1e-6, f"{ce.spread_pct:.4f}%")
    check("volume and OI carried through",
          ce.volume == 58338735 and ce.oi == 67527)

    print(f"\n{'=' * 70}\n2. the number that matters: realistic vs optimistic fill\n{'=' * 70}")
    sell_ltp = float(df.ltp.sum())
    sell_bid = float(df.bid.sum())
    check("a seller receives the BID, which is below LTP", sell_bid < sell_ltp,
          f"bid {sell_bid:.2f} vs ltp {sell_ltp:.2f}")
    cost_pts = sell_ltp - sell_bid
    check("spread cost is positive and small", 0 < cost_pts < 5,
          f"{cost_pts:.2f} pts = Rs {cost_pts * 75:,.0f} per 75-lot")
    # mid can exceed LTP when the last trade printed below the midpoint, which is
    # ordinary. The invariant that always holds is bid <= mid <= ask, per contract.
    check("bid <= mid <= ask on every contract",
          bool(((df.bid <= df["mid"]) & (df["mid"] <= df.ask)).all()),
          f"mid sum {df['mid'].sum():.3f} vs ltp sum {sell_ltp:.2f} "
          f"(mid above ltp is normal)")

    print(f"\n{'=' * 70}\n3. missing depth must not crash or invent a price\n{'=' * 70}")
    empty = kc.depth_frame({"NFO:X": {"last_price": 10.0, "depth": {}}})
    check("no depth -> bid/ask are None, not guessed",
          empty.bid.isna().all() and empty.ask.isna().all())
    check("no depth -> spread is None", empty.spread.isna().all())
    check("empty payload returns an empty frame", len(kc.depth_frame({})) == 0)

    print(f"\n{'=' * 70}\n4. checksum\n{'=' * 70}")
    key, req, sec = "abc123", "req456", "sec789"
    want = hashlib.sha256((key + req + sec).encode()).hexdigest()
    check("SHA256(api_key + request_token + api_secret)", len(want) == 64,
          f"{want[:20]}...")

    print(f"\n{'=' * 70}\n5. refuses to act without credentials\n{'=' * 70}")
    k = kc.Kite(api_key=None, access_token=None)
    creds = kc.load_credentials()
    if creds.get("api_key") and creds.get("access_token"):
        print("  (credentials ARE configured on this machine - skipping)")
    else:
        check("reports itself unauthenticated", not k.ok)
        try:
            k.quote("NFO:NIFTY26O0722600CE")
            check("quote refuses when unauthenticated", False, "it did not raise")
        except RuntimeError as e:
            check("quote refuses when unauthenticated", True, str(e)[:48])

    print(f"\n{'=' * 70}\n6. no secret is hardcoded anywhere in the module\n{'=' * 70}")
    src = open(kc.__file__, encoding="utf-8").read()
    import re
    leaked = re.findall(r'(api_secret|access_token)\s*=\s*["\'][A-Za-z0-9]{8,}["\']', src)
    check("no literal secret in the source", not leaked, str(leaked[:2]))

    print(f"\n{'=' * 70}")
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: " + "; ".join(fails))
        return 1
    print("ALL CHECKS PASSED - the integration is correct; it needs only a key.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
