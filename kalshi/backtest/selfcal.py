import numpy as np, pandas as pd, sys
sys.path.insert(0, "/mnt/project-files/kalshi"); import kalshi_edges as KE
p = pd.read_pickle("proj_bt.pkl"); p.index.name = "player_id"; p = p.reset_index()
a = pd.concat([pd.read_csv(f"fdata/player_{y}.csv", low_memory=False) for y in (2025, 2026)])
a = a[a.season_type == "REG"][["player_id", "season", "week", "passing_yards", "rushing_yards", "receiving_yards", "receptions", "rushing_tds", "receiving_tds", "passing_tds"]].fillna(0)
a["td"] = a.rushing_tds + a.receiving_tds
x = p.merge(a, on=["player_id", "season", "week"])
x["tdp"] = x.rtd + x.wtd
tables = KE.ratio_tables("/tmp/kalshi_work/data")
M = [("KXNFLRECYDS", "wyd", "receiving_yards", 20), ("KXNFLRSHYDS", "ryd", "rushing_yards", 20), ("KXNFLPASSYDS", "pyd", "passing_yards", 150), ("KXNFLREC", "rec", "receptions", 2)]
for ser, c, ac, mn in M:
    y = x[x[c] >= mn]
    print(f"{ac}: n={len(y)} mean actual/proj={y[ac].sum()/y[c].sum():.3f}  MAE={abs(y[ac]-y[c]).mean():.1f}  corr={y[ac].corr(y[c]):.3f}")
    # calibration at lines around each player's projection (like Kalshi ladders)
    rows = []
    for f in (0.6, 0.8, 1.0, 1.2, 1.5):
        line = np.floor(y[c] * f) + 0.5
        pm = np.array([KE.p_over(tables, ser, m, l) for m, l in zip(y[c], line)])
        rows.append((f, round(pm.mean(), 3), round((y[ac] > line).mean(), 3)))
    print("   line=f*proj: (f, model P(over), actual)", rows)
for c, ac in (("tdp", "td"), ("ptd", "passing_tds")):
    y = x[x[c] >= 0.15]
    print(f"{ac}: proj mean {y[c].mean():.3f} actual {y[ac].mean():.3f}; P(1+) model {np.mean(1-np.exp(-y[c])):.3f} actual {(y[ac]>=1).mean():.3f}; P(2+) model {np.mean([KE.poisson_over(m,1.5) for m in y[c]]):.3f} actual {(y[ac]>=2).mean():.3f}")
    y = y.assign(b=pd.cut(y[c], [0, .2, .3, .4, .5, .7, 1, 5]))
    print(y.groupby("b", observed=True).agg(n=(ac, "size"), proj=(c, "mean"), act=(ac, "mean"), p1=(ac, lambda v: (v >= 1).mean())).round(3).to_string())
x.to_pickle("selfcal.pkl")
