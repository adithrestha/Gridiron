import numpy as np, pandas as pd, sys
sys.path.insert(0, "/mnt/project-files/kalshi"); import kalshi_edges as KE
k = pd.read_pickle("graded2.pkl")
ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
print("trust grid (log loss; lower better)")
T = [0, .1, .2, .25, .3, .4, .5]
print(pd.DataFrame({s: [ll(x.mid + t * (x.pm - x.mid), x.result) for t in T] for s, x in k.groupby("stat")}, index=T).round(4).to_string())
def sim(k, trust, min_edge=0.03, skip=(), dedupe=True, vol=0):
    out = []
    for r in k.itertuples():
        if r.stat in skip or r.pre_vol < vol: continue
        pa = r.mid + trust.get(r.stat, .25) * (r.pm - r.mid)
        for side, p, cost in (("YES", pa, r.ask), ("NO", 1 - pa, 1 - r.bid)):
            if not (0.10 <= cost <= 0.90): continue
            e = p - cost - KE.fee(cost)
            if e >= min_edge:
                win = r.result if side == "YES" else 1 - r.result
                out.append(dict(game=r.game_id, key=f"{r.game_id}|{r.nk}|{r.stat}", week=r.week, stat=r.stat, side=side, cost=cost, edge=e, win=win, pnl=win - cost - KE.fee(cost)))
    b = pd.DataFrame(out)
    if dedupe and len(b): b = b.sort_values("edge").groupby("key").tail(1)
    return b
def report(name, b, boot=True):
    roi = b.pnl.sum() / b.cost.sum()
    lo = hi = np.nan
    if boot:
        g = b.groupby("game").agg(p=("pnl", "sum"), c=("cost", "sum")); rng = np.random.default_rng(0)
        bs = [ (lambda s: s.p.sum() / s.c.sum())(g.sample(len(g), replace=True, random_state=rng.integers(1e9))) for _ in range(1000)]
        lo, hi = np.percentile(bs, [5, 95])
    print(f"{name:42s} bets {len(b):4d}  win {b.win.mean():.3f}  avg cost {b.cost.mean():.2f}  ROI {roi:+.3f}  90% CI [{lo:+.3f}, {hi:+.3f}]")
base = {s: .25 for s in k.stat.unique()}
report("current rules, all ladder rungs", sim(k, base, dedupe=False))
report("current rules, one bet per player-stat", sim(k, base))
report("  ... skip pass yards", sim(k, base, skip=("pyd",)))
for me in (0.0, 0.02, 0.05):
    report(f"  ... skip pass yards, min edge {me:.2f}", sim(k, base, min_edge=me, skip=("pyd",)))
report("  ... skip pyd, pre-game volume>=1000", sim(k, base, skip=("pyd",), vol=1000))
b = sim(k, base, skip=("pyd",))
print(b.groupby("week").agg(n=("pnl", "size"), pnl=("pnl", "sum"), cost=("cost", "sum")).assign(roi=lambda x: x.pnl / x.cost).round(3).to_string())
print(b.groupby(["stat", "side"]).agg(n=("pnl", "size"), win=("win", "mean"), roi=("pnl", "sum")).round(3).to_string())
# closing-line check: is the kickoff price itself well calibrated?
print("Kalshi calibration:", k.assign(b=pd.cut(k.mid, [0, .2, .4, .6, .8, 1])).groupby("b", observed=True).agg(n=("result", "size"), price=("mid", "mean"), actual=("result", "mean")).round(3).to_dict("index"))
