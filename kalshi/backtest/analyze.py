import numpy as np, pandas as pd, sys
from grade import load, model_p, KE
k = load()
k = k[k.bid.notna() & k.ask.notna() & (k.week >= 3)]
k = k[(k.ask - k.bid <= 0.10) & (k.mid >= 0.03) & (k.mid <= 0.97) & (k.mu > 0)]
tables = KE.ratio_tables("/tmp/kalshi_work/data")
k["pm"] = model_p(k, tables)
k["padj"] = k.mid + 0.25 * (k.pm - k.mid)
ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
br = lambda p, y: np.mean((p - y) ** 2)
print("n", len(k), "by week", k.week.value_counts().sort_index().to_dict())
rows = []
for s, x in [("ALL", k)] + list(k.groupby("stat")):
    rows.append(dict(stat=s, n=len(x), yes_rate=x.result.mean(), kalshi_avg=x.mid.mean(), model_avg=x.pm.mean(),
                     ll_kalshi=ll(x.mid, x.result), ll_model=ll(x.pm, x.result), ll_shrunk=ll(x.padj, x.result),
                     br_kalshi=br(x.mid, x.result), br_model=br(x.pm, x.result)))
print(pd.DataFrame(rows).round(4).to_string())
# calibration of model
k["b"] = pd.cut(k.pm, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
print(k.groupby("b", observed=True).agg(n=("result", "size"), model=("pm", "mean"), kalshi=("mid", "mean"), actual=("result", "mean")).round(3))
# betting sim with live rules
def sim(k, padj, min_edge=0.03):
    out = []
    for r, pa in zip(k.itertuples(), padj):
        for side, p, cost in (("YES", pa, r.ask), ("NO", 1 - pa, 1 - r.bid)):
            if not (0.10 <= cost <= 0.90): continue
            e = p - cost - KE.fee(cost)
            if e >= min_edge:
                win = r.result if side == "YES" else 1 - r.result
                out.append(dict(stat=r.stat, side=side, cost=cost, edge=e, win=win, pnl=win - cost - KE.fee(cost)))
    return pd.DataFrame(out)
b = sim(k, k.padj)
print("\nlive rules: bets", len(b), "win%", round(b.win.mean(), 3), "avg cost", round(b.cost.mean(), 3), "ROI", round(b.pnl.sum() / b.cost.sum(), 3))
print(b.groupby(["stat", "side"]).agg(n=("pnl", "size"), win=("win", "mean"), cost=("cost", "mean"), pnl=("pnl", "sum")).round(3))
k.to_pickle("graded.pkl")
