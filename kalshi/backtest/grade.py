import sys, math, json, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/project-files/kalshi")
import kalshi_edges as KE
SER = {"KXNFLPASSYDS": "pyd", "KXNFLRSHYDS": "ryd", "KXNFLRECYDS": "wyd", "KXNFLREC": "rec", "KXNFLTD": "td", "KXNFLPASSTDS": "ptd"}
def load():
    k = pd.read_csv("kalshi_settled_props.csv")
    p = pd.read_pickle("proj_bt.pkl").reset_index().rename(columns={"index": "player_id"})
    if "player_id" not in p: p = p.rename(columns={p.columns[0]: "player_id"})
    p = p[p.season == 2026]
    p["td"] = p.rtd + p.wtd
    g = pd.read_csv("fdata/games.csv"); g = g[g.season == 2026]
    k = k.merge(g[["game_id", "away_team", "home_team"]], on="game_id")
    k["name"] = k["sub"].str.rsplit(":", n=1).str[0]
    k["nk"] = k.name.map(KE.norm_name)
    p["nk"] = p.name.map(KE.norm_name)
    p = p.drop_duplicates(["nk", "team", "week"])
    k = k.merge(p, on=["nk", "week"], suffixes=("", "_p"))
    k = k[(k.team == k.away_team) | (k.team == k.home_team)]
    k["stat"] = k.series.map(SER)
    k["mu"] = [getattr(r, s) for r, s in zip(k.itertuples(), k.stat)]
    k["mid"] = (k.bid + k.ask) / 2
    return k
def model_p(k, tables, scale=None):
    out = []
    for r in k.itertuples():
        if r.stat in ("td", "ptd"): out.append(KE.poisson_over(r.mu, r.strike))
        else: out.append(KE.p_over(tables, r.series, r.mu, r.strike))
    return np.clip(np.array(out), 0.005, 0.995)
