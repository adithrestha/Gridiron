import numpy as np, pandas as pd, sys
sys.path.insert(0, "/mnt/project-files/kalshi"); import kalshi_edges as KE
from sklearn.linear_model import LogisticRegression
k = pd.read_pickle("graded.pkl")
L = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
k["lm"], k["lp"] = L(k.mid), L(k.pm)
k["pyd_flag"] = (k.stat == "pyd").astype(int)
# leave-one-week-out stacking, per stat group
def stack(feats):
    out = pd.Series(index=k.index, dtype=float)
    for w in k.week.unique():
        tr, te = k[k.week != w], k[k.week == w]
        for s in k.stat.unique():
            a, b = tr[tr.stat == s], te[te.stat == s]
            if len(b) == 0: continue
            m = LogisticRegression(C=1.0).fit(a[feats], a.result)
            out[b.index] = m.predict_proba(b[feats])[:, 1]
    return out
k["p_stack"] = stack(["lm", "lp"])
for s in ["ALL"] + sorted(k.stat.unique()):
    x = k if s == "ALL" else k[k.stat == s]
    print(f"{s:4s} n={len(x):5d}  kalshi {ll(x.mid, x.result):.4f}  shrink25 {ll(x.padj, x.result):.4f}  stacked {ll(x.p_stack, x.result):.4f}")
# coefficients on all data
for s in sorted(k.stat.unique()):
    x = k[k.stat == s]; m = LogisticRegression(C=1.0).fit(x[["lm", "lp"]], x.result)
    print(s, "kalshi w", round(m.coef_[0][0], 2), "model w", round(m.coef_[0][1], 2), "int", round(m.intercept_[0], 2))
k.to_pickle("graded2.pkl")
