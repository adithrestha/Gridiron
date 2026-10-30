"""Builds output/fantasy.html: a Predictions tab (the game engine's picks, read from its outputs) and a Fantasy
tab (fantasy.py's projections). Each signed-in viewer's leagues and imported rosters live in the page's
database (`accounts/<viewer id>`, `rosters/<viewer id>`), so republishing never touches them."""
import json, os, numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
ENGINE_OUT = "/mnt/project-files/nfl_engine/v2/output"
p = pd.read_csv(f"{OUT}/fantasy_projections.csv", dtype={"id": str})
run = json.load(open(f"{OUT}/fantasy_run.json"))
week = run["week"]

# accuracy summary from the walk-forward backtest
acc = {}
if os.path.exists(f"{OUT}/fantasy_backtest.csv"):
    b = pd.read_csv(f"{OUT}/fantasy_backtest.csv")
    acc = {"n": int(len(b)), "mae": round(float((b.half - b.actual).abs().mean()), 2),
           "naive": round(float((b.naive_season - b.actual).abs().mean()), 2),
           "r": round(float(b.half.corr(b.actual)), 3),
           "pos": {pos: round(float((x.half - x.actual).abs().mean()), 2) for pos, x in b.groupby("position")}}

r1 = lambda v: None if pd.isna(v) else round(float(v), 1)
players = []
for pid, x in p.groupby("id", sort=False):
    x = x.sort_values("week")
    cur = x[x.week == week]
    first = x.iloc[0]
    rec = {"id": pid, "n": first["name"], "p": first.position, "t": first.team,
           "w": {int(r.week): [r.opp, r1(r.implied), r1(r.std), r1(r.half), r1(r.ppr), round(float(r.p_out), 2)]
                 for r in x.itertuples()}}
    if len(cur):
        c = cur.iloc[0]
        rec["s"] = "" if pd.isna(c.status) else c.status
        if first.position in ("QB", "RB", "WR", "TE"):
            rec["l"] = [r1(c[k]) for k in ("pyd", "ptd", "int", "car", "ryd", "rtd", "tgt", "rec", "wyd", "wtd")]
    players.append(rec)

# the game engine's outputs (read-only)
eng = {}
if os.path.exists(f"{ENGINE_OUT}/predictions.csv"):
    g = pd.read_csv(f"{ENGINE_OUT}/predictions.csv")
    eng["games"] = g.round(3).astype(object).where(g.notna(), None).to_dict("records")
    s = pd.read_csv(f"{ENGINE_OUT}/projected_standings.csv")
    eng["standings"] = s.round(2).to_dict("records")
    bt = pd.read_csv(f"{ENGINE_OUT}/backtest.csv")
    eng["backtest"] = bt.round(4).astype(object).where(bt.notna(), None).to_dict("records")
    eng["run"] = json.load(open(f"{ENGINE_OUT}/run.json")) if os.path.exists(f"{ENGINE_OUT}/run.json") else {}

meta = json.load(open(f"{OUT}/fantasy_meta.json")) if os.path.exists(f"{OUT}/fantasy_meta.json") else {}
hist = json.load(open(f"{OUT}/fantasy_history.json")) if os.path.exists(f"{OUT}/fantasy_history.json") else {}
now = json.load(open(f"{OUT}/fantasy_actual.json")) if os.path.exists(f"{OUT}/fantasy_actual.json") else {}
mu = json.load(open(f"{OUT}/fantasy_matchups.json")) if os.path.exists(f"{OUT}/fantasy_matchups.json") else {}
# Betting tab: Kalshi prices vs the engine, built by /mnt/project-files/kalshi (read-only here)
BET = "/mnt/project-files/kalshi/output/kalshi_tab_data.json"
bet = json.load(open(BET)) if os.path.exists(BET) else None
data = json.dumps({"run": run, "acc": acc, "players": players, "eng": eng, "meta": meta, "mu": mu, "hist": hist, "now": now, "bet": bet}, separators=(",", ":"))

html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fantasy_template.html")).read()
with open(f"{OUT}/fantasy.html", "w") as f:
    f.write(html.replace("__DATA__", data))
print(f"wrote {OUT}/fantasy.html ({len(players)} players, {os.path.getsize(f'{OUT}/fantasy.html') // 1024} KB)")
