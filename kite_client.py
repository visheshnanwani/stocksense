"""
kite_client.py
--------------
Zerodha Kite Connect, for the one thing every other source here is missing:
LIVE BID AND ASK.

WHY THIS EXISTS
    NSE's free live feed (nse_live_options.py) carries lastPrice and nothing
    else. The spread is what decides the price you are actually filled at, and
    the short-straddle edge measured on real exchange data is about +0.12% of
    margin per session -- thin enough that a couple of points of spread is a
    material share of it. Without bid/ask every live P&L in this project is an
    upper bound. Kite's quote endpoint returns a five-level depth book, which
    closes that gap.

CREDENTIALS -- YOURS, NEVER TYPED INTO A CHAT
    This module never prompts for a secret and none is written in the code.
    Put them in ONE of:

      1. environment variables
             KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN
      2. a local file  kite_credentials.json  beside this module
             {"api_key": "...", "api_secret": "...", "access_token": "..."}
         Add it to .gitignore. It is a key to a real brokerage account.

WHAT YOU HAVE TO DO YOURSELF (it cannot be automated, by design)
    1. Subscribe to Kite Connect at https://developers.kite.trade
       (about Rs 2,000/month; a Zerodha trading account alone is not enough).
    2. Create an app there and note the api_key and api_secret.
    3. Once per day, mint an access token:
           python kite_client.py login          # prints the login URL
       Open it, log in with your own credentials in Zerodha's own browser
       window, and you are redirected to your redirect URL carrying
       ?request_token=XXXX. Then:
           python kite_client.py token XXXX     # exchanges it, saves the token
       The access token expires around 6 a.m. the next day, so this is a daily
       step. That is Zerodha's design and there is no way around it.

    You log in. This code never sees your password, and never places an order --
    it only reads quotes.
"""

import csv
import hashlib
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import requests

ROOT = "https://api.kite.trade"
LOGIN = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
HERE = os.path.dirname(os.path.abspath(__file__))
CRED_FILE = os.path.join(HERE, "kite_credentials.json")


# ----------------------------------------------------------------- credentials
def load_credentials() -> dict:
    """Environment first, then the local file. Returns {} when nothing is set."""
    c = {}
    for k in ("api_key", "api_secret", "access_token"):
        v = os.environ.get("KITE_" + k.upper())
        if v:
            c[k] = v.strip()
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE) as f:
                disk = json.load(f)
            for k, v in disk.items():
                c.setdefault(k, str(v).strip())
        except Exception as e:
            print(f"  could not read {os.path.basename(CRED_FILE)}: {e}")
    return c


def save_access_token(token: str):
    c = {}
    if os.path.exists(CRED_FILE):
        try:
            c = json.load(open(CRED_FILE))
        except Exception:
            c = {}
    c["access_token"] = token
    with open(CRED_FILE, "w") as f:
        json.dump(c, f, indent=2)
    os.chmod(CRED_FILE, 0o600) if os.name != "nt" else None
    print(f"  access token saved to {os.path.basename(CRED_FILE)}")


def login_url(api_key=None) -> str:
    c = load_credentials()
    key = api_key or c.get("api_key")
    if not key:
        raise RuntimeError("no api_key -- set KITE_API_KEY or put it in kite_credentials.json")
    return LOGIN.format(api_key=key)


def exchange_request_token(request_token: str) -> str:
    """Swap the one-time request_token for a day-long access_token."""
    c = load_credentials()
    key, secret = c.get("api_key"), c.get("api_secret")
    if not (key and secret):
        raise RuntimeError("need api_key AND api_secret to exchange a request token")
    checksum = hashlib.sha256((key + request_token + secret).encode()).hexdigest()
    r = requests.post(f"{ROOT}/session/token",
                      data={"api_key": key, "request_token": request_token,
                            "checksum": checksum},
                      headers={"X-Kite-Version": "3"}, timeout=30)
    j = r.json()
    if r.status_code != 200 or j.get("status") != "success":
        raise RuntimeError(f"token exchange failed: {j.get('message', r.text[:200])}")
    tok = j["data"]["access_token"]
    save_access_token(tok)
    return tok


# ----------------------------------------------------------------- client
class Kite:
    """Read-only Kite client. Quotes and instruments; it never places an order."""

    def __init__(self, api_key=None, access_token=None):
        c = load_credentials()
        self.api_key = api_key or c.get("api_key")
        self.access_token = access_token or c.get("access_token")
        self.ok = bool(self.api_key and self.access_token)
        self._s = requests.Session()
        if self.ok:
            self._s.headers.update({
                "X-Kite-Version": "3",
                "Authorization": f"token {self.api_key}:{self.access_token}"})

    def _get(self, path, **params):
        if not self.ok:
            raise RuntimeError("not authenticated -- run: python kite_client.py login")
        r = self._s.get(ROOT + path, params=params, timeout=30)
        if r.status_code == 403:
            raise RuntimeError("403 -- access token expired (they last until ~6am). "
                               "Run: python kite_client.py login")
        j = r.json()
        if j.get("status") != "success":
            raise RuntimeError(j.get("message", r.text[:200]))
        return j["data"]

    def profile(self):
        return self._get("/user/profile")

    def instruments(self, exchange="NFO") -> pd.DataFrame:
        """The full contract master. ~100k rows for NFO; cache it, it changes daily."""
        if not self.ok:
            raise RuntimeError("not authenticated")
        r = self._s.get(f"{ROOT}/instruments/{exchange}", timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(list(csv.DictReader(io.StringIO(r.text))))
        for c in ("strike", "last_price", "lot_size", "tick_size"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "expiry" in df:
            df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        return df

    def quote(self, symbols) -> dict:
        """Full quote WITH five-level depth. Max 500 instruments per call."""
        if isinstance(symbols, str):
            symbols = [symbols]
        out = {}
        for i in range(0, len(symbols), 200):
            out.update(self._get("/quote", i=symbols[i:i + 200]))
        return out


# ----------------------------------------------------------------- bid/ask
def depth_frame(quotes: dict) -> pd.DataFrame:
    """Flatten Kite's quote payload into the numbers that decide a fill."""
    rows = []
    for sym, q in (quotes or {}).items():
        d = (q.get("depth") or {})
        buy = (d.get("buy") or [{}])[0]
        sell = (d.get("sell") or [{}])[0]
        bid, ask = buy.get("price"), sell.get("price")
        ltp = q.get("last_price")
        mid = ((bid + ask) / 2) if (bid and ask) else None
        rows.append({
            "symbol": sym, "ltp": ltp, "bid": bid, "ask": ask, "mid": mid,
            "bid_qty": buy.get("quantity"), "ask_qty": sell.get("quantity"),
            "spread": (ask - bid) if (bid and ask) else None,
            "spread_pct": ((ask - bid) / mid * 100) if (bid and ask and mid) else None,
            "volume": q.get("volume"), "oi": q.get("oi"),
        })
    return pd.DataFrame(rows)


def realistic_straddle_fill(kite: Kite, tradingsymbols) -> dict:
    """
    What SELLING this straddle would really get you, and what the optimistic
    lastPrice version claims. The difference is the spread you would have paid.
    """
    q = kite.quote([f"NFO:{s}" for s in tradingsymbols])
    df = depth_frame(q)
    if df.empty or df.bid.isna().any():
        return {"ok": False, "depth": df}
    return {
        "ok": True, "depth": df,
        "sell_at_bid": float(df.bid.sum()),        # what a seller actually receives
        "sell_at_ltp": float(df.ltp.sum()),        # the optimistic number
        "sell_at_mid": float(df["mid"].sum()),
        "spread_cost": float(df.ltp.sum() - df.bid.sum()),
        "total_spread": float(df.spread.sum()),
    }


# ----------------------------------------------------------------- cli
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    c = load_credentials()

    if cmd == "status":
        print("Kite Connect credentials")
        for k in ("api_key", "api_secret", "access_token"):
            v = c.get(k)
            print(f"  {k:<14} {'set (' + v[:4] + '...)' if v else 'NOT SET'}")
        if not c.get("api_key"):
            print("\n  Set KITE_API_KEY / KITE_API_SECRET, or create kite_credentials.json")
            print("  See the docstring at the top of this file.")
            return 0
        k = Kite()
        if k.ok:
            try:
                p = k.profile()
                print(f"\n  authenticated as {p.get('user_name')} ({p.get('user_id')})")
            except Exception as e:
                print(f"\n  token not usable: {e}")
        return 0

    if cmd == "login":
        if not c.get("api_key"):
            print("No api_key configured yet.")
            print("")
            print("  1. Subscribe at https://developers.kite.trade (about Rs 2,000/month)")
            print("  2. Create an app there; note its api_key and api_secret")
            print("  3. Put them in kite_credentials.json beside this file:")
            print('       {"api_key": "your_key", "api_secret": "your_secret"}')
            print("     or export KITE_API_KEY / KITE_API_SECRET")
            print("  4. Run this command again")
            return 1
        print("1. Open this URL and log in with your own Zerodha credentials:\n")
        print("   " + login_url())
        print("\n2. You are redirected to your redirect URL with ?request_token=XXXX")
        print("3. Then run:  python kite_client.py token XXXX")
        return 0

    if cmd == "token":
        if len(sys.argv) < 3:
            print("usage: python kite_client.py token <request_token>")
            return 1
        tok = exchange_request_token(sys.argv[2])
        print(f"  access token obtained ({tok[:6]}...), valid until ~6am tomorrow")
        return 0

    if cmd == "quote":
        k = Kite()
        syms = sys.argv[2:] or ["NFO:NIFTY26O0722600CE"]
        df = depth_frame(k.quote(syms))
        print(df.to_string(index=False))
        return 0

    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
