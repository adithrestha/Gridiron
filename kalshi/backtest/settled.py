"""GET-only: settled Kalshi NFL prop markets (read-only market data)."""
import json, sys, time, urllib.request, urllib.error
sys.path.insert(0, "/mnt/project-files/kalshi")
from fetch_kalshi import get
KEEP = ["ticker","event_ticker","open_time","close_time","occurrence_datetime","floor_strike","yes_sub_title","no_sub_title",
        "result","last_price_dollars","volume_fp","rules_primary","created_time","custom_strike"]
out = {}
for s in ["KXNFLRECYDS","KXNFLRSHYDS","KXNFLPASSYDS","KXNFLREC","KXNFLTD","KXNFLPASSTDS"]:
    rows, cur = [], ""
    while True:
        d = get(f"/markets?series_ticker={s}&status=settled&limit=1000" + (f"&cursor={cur}" if cur else ""))
        rows += [{k: m.get(k) for k in KEEP} for m in d.get("markets", [])]
        cur = d.get("cursor"); time.sleep(0.4)
        if not cur: break
    out[s] = rows; print(s, len(rows), flush=True)
json.dump(out, open("settled.json", "w"))
