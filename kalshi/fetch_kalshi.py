"""Pull open NFL markets from Kalshi's public (read-only, no-auth) market-data API.
Only GET requests against /series, /events and /markets. Never touches trading or account endpoints."""
import json, sys, time, urllib.request, urllib.error

B = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLTEAMTOTAL", "KXNFLTD", "KXNFLRECYDS",
          "KXNFLRSHYDS", "KXNFLPASSYDS", "KXNFLREC", "KXNFLPASSTDS"]

def get(path):
    for i in range(6):
        try:
            return json.load(urllib.request.urlopen(B + path, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500: raise
        except urllib.error.URLError:
            pass
        time.sleep(1.5 * 2 ** i)
    raise RuntimeError("Kalshi API unreachable: " + path)

def series_markets(series):
    out, cur = [], ""
    while True:
        d = get(f"/markets?series_ticker={series}&status=open&limit=1000" + (f"&cursor={cur}" if cur else ""))
        out += d.get("markets", [])
        cur = d.get("cursor")
        time.sleep(0.4)
        if not cur: return out

if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "kalshi_markets.json"
    data = {s: series_markets(s) for s in SERIES}
    data["_fetched_at"] = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    json.dump(data, open(dest, "w"))
    print({k: len(v) for k, v in data.items() if k[0] != "_"})
