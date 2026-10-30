"""GET-only: Kalshi pre-kickoff prices for settled 2026 NFL prop markets (batch candlesticks, market data only)."""
import json, sys, time, pandas as pd
sys.path.insert(0, "/mnt/project-files/kalshi")
from fetch_kalshi import get
K2N = {"JAC": "JAX", "LAR": "LA"}
g = pd.read_csv("fdata/games.csv"); g = g[(g.season == 2026) & (g.game_type == "REG")]
g["kick"] = pd.to_datetime(g.gameday + " " + g.gametime).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
g["dkey"] = pd.to_datetime(g.gameday).dt.strftime("%y%b%d").str.upper()
games = {(r.dkey, r.away_team, r.home_team): r for r in g.itertuples()}
def match(ev):
    code = ev.split("-")[1]; dk, teams = code[:7], code[7:]
    for (d, a, h), r in games.items():
        if d != dk: continue
        inv = {v: k for k, v in K2N.items()}
        if teams == inv.get(a, a) + inv.get(h, h): return r
    return None
d = json.load(open("settled.json"))
rows = []
for s, ms in d.items():
    for m in ms:
        if m["result"] not in ("yes", "no"): continue
        r = match(m["event_ticker"])
        if r is None: continue
        if pd.Timestamp(m["open_time"]) >= r.kick: continue
        rows.append(dict(series=s, ticker=m["ticker"], game_id=r.game_id, week=r.week, kick=int(r.kick.timestamp()),
                         sub=m["yes_sub_title"], strike=m["floor_strike"], result=int(m["result"] == "yes"),
                         volume=float(m["volume_fp"] or 0)))
df = pd.DataFrame(rows); df = df[df.week >= 3]; print(len(df), df.week.value_counts().sort_index().to_dict(), flush=True)
import os
px = json.load(open("px_cache.json")) if os.path.exists("px_cache.json") else {}
for kick, grp in df.groupby("kick"):
    t = list(grp.ticker)
    t = [x for x in t if x not in px]
    for i in range(0, len(t), 50):
        chunk = t[i:i + 50]
        try: r = get(f"/markets/candlesticks?market_tickers={','.join(chunk)}&start_ts={kick - 72*3600}&end_ts={kick}&period_interval=60")
        except Exception as e:
            print("skip chunk", e.__class__.__name__, flush=True); time.sleep(20); continue
        for mk in r.get("markets", []):
            c = [x for x in mk["candlesticks"] if x["end_period_ts"] <= kick]
            if not c: continue
            last = c[-1]; f = lambda o, k: float(o[k]["close_dollars"]) if o.get(k) and o[k].get("close_dollars") is not None else None
            bid, ask = f(last, "yes_bid"), f(last, "yes_ask")
            px[mk["market_ticker"]] = dict(bid=bid, ask=ask, close=f(last, "price"), hrs=(kick - last["end_period_ts"]) / 3600,
                                           pre_vol=sum(float(x.get("volume_fp") or 0) for x in c))
        time.sleep(0.3)
    json.dump(px, open("px_cache.json", "w")); print(kick, len(px), flush=True)
df = df.join(pd.DataFrame.from_dict(px, orient="index"), on="ticker")
df.to_csv("kalshi_settled_props.csv", index=False); print(df.notna().sum().to_dict())
