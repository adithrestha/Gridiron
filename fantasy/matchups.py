"""Matchup data for the fantasy projections.

  * defense_components   what each defense allows to each position, stat by stat (pass attempts, carries,
                         targets, and yards/TDs/catches per opportunity) vs league average. This separates
                         a defense that is soft against RB receiving from one that is soft against RB runs.
  * coverage             how often each defense plays man vs zone (nflverse/NGS participation data), and how
                         each pass catcher scores against man vs zone per team pass play.
  * vs_top5              how a player (or an offense) scores against defenses ranked top 5 against his position
                         at the time, relative to how much top-5 defenses hold everyone down.

Every split is shrunk toward "no effect" so a couple of big games don't swing a projection.
"""
import os
import numpy as np, pandas as pd

SKILL = ["QB", "RB", "WR", "TE"]
CATCHERS = ["RB", "WR", "TE"]
COLS = ["att", "pyd", "ptd", "car", "ryd", "rtd", "tgt", "rec", "wyd", "wtd"]
# per-opportunity rates and the opportunities of league-average evidence each defense starts with
RATE_PRIOR = {"pyd": ("att", 200), "ptd": ("att", 400), "ryd": ("car", 100), "rtd": ("car", 200),
              "rec": ("tgt", 60), "wyd": ("tgt", 80), "wtd": ("tgt", 200)}
VOL_PRIOR = 4          # games of league-average evidence for volume allowed
MAN_PRIOR = 150        # pass plays of league-average man rate each defense starts with
COV_PRIOR = 150        # team pass plays of a player's overall rate behind each of his man/zone rates
T5_PRIOR = 6           # games of "no split" behind each player's top-5-defense split


# ----------------------------------------------------------------------------- stat-by-stat defense
def defense_components(s, tkey, weights):
    """DataFrame indexed by (defense, position): allowed / league average for each stat in COLS.
    Volume stats (att, car, tgt) are per game; the rest are per opportunity."""
    sk = s[s.position.isin(SKILL) & (s.key < tkey)]
    per = (sk.groupby(["opponent_team", "position", "game_id", "season", "key"])[COLS].sum().reset_index()
             .rename(columns={"opponent_team": "defense"}))
    out = []
    for pos, x in per.groupby("position"):
        h = weights(x, tkey, "defense")
        tot = h[COLS].multiply(h.w, axis=0).assign(w=h.w, defense=h.defense).groupby("defense").sum()
        lg = x[COLS].sum()
        f = pd.DataFrame(index=tot.index)
        for v in ("att", "car", "tgt"):
            lgv = lg[v] / len(x)
            f[v] = (tot[v] + VOL_PRIOR * lgv) / (tot.w + VOL_PRIOR) / lgv if lgv > 0 else 1.0
        for num, (den, k) in RATE_PRIOR.items():
            lr = lg[num] / lg[den] if lg[den] > 0 else 0
            f[num] = (tot[num] + k * lr) / (tot[den] + k) / lr if lr > 0 else 1.0
        f["position"] = pos
        out.append(f.reset_index())
    return pd.concat(out, ignore_index=True).set_index(["defense", "position"])


def component_mult(r, comp, w):
    """Per-player multipliers on each projected stat from the opponent's stat-by-stat profile."""
    m = pd.DataFrame(1.0, index=r.index, columns=COLS)
    if w <= 0:
        return m
    keys = list(zip(r.opp, r.position))
    f = comp.reindex(keys).fillna(1.0)
    f.index = r.index
    f = f[COLS].clip(0.5, 2.0) ** w
    m["att"], m["car"], m["tgt"] = f.att, f.car, f.tgt
    m["pyd"], m["ptd"] = f.att * f.pyd, f.att * f.ptd
    m["ryd"], m["rtd"] = f.car * f.ryd, f.car * f.rtd
    m["rec"], m["wyd"], m["wtd"] = f.tgt * f.rec, f.tgt * f.wyd, f.tgt * f.wtd
    return m


# ----------------------------------------------------------------------------- man vs zone
def load_coverage(data_dir, years):
    """One row per pass attempt (no sacks) with the targeted player, his half-PPR receiving points on the
    play and whether the defense was in man coverage (NaN where no participation data)."""
    rows = []
    for y in years:
        pf = os.path.join(data_dir, f"pbp_{y}.csv.gz")
        if not os.path.exists(pf):
            continue
        p = pd.read_csv(pf, low_memory=False, usecols=["game_id", "play_id", "season", "week", "season_type",
                        "posteam", "defteam", "pass_attempt", "sack", "receiver_player_id", "yards_gained",
                        "complete_pass", "pass_touchdown"])
        p = p[(p.season_type == "REG") & (p.pass_attempt == 1) & (p.sack != 1)].copy()
        p["man"] = np.nan
        qf = os.path.join(data_dir, f"part_{y}.csv")
        if os.path.exists(qf):
            q = pd.read_csv(qf, usecols=["nflverse_game_id", "play_id", "defense_man_zone_type"])
            q = q[q.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
            mz = dict(zip(zip(q.nflverse_game_id, q.play_id), (q.defense_man_zone_type == "MAN_COVERAGE").astype(float)))
            p["man"] = [mz.get(k, np.nan) for k in zip(p.game_id, p.play_id)]
        cp = p.complete_pass.fillna(0)
        p["pts"] = cp * (0.5 + 0.1 * p.yards_gained.fillna(0)) + 6 * p.pass_touchdown.fillna(0)
        rows.append(p[["game_id", "season", "week", "posteam", "defteam", "receiver_player_id", "man", "pts"]])
    if not rows:
        return pd.DataFrame(columns=["game_id", "season", "week", "key", "posteam", "defteam", "receiver_player_id",
                                     "man", "pts"])
    c = pd.concat(rows, ignore_index=True)
    c["key"] = c.season * 100 + c.week
    return c


def coverage_factors(cov, s, tkey, season_keep):
    """Returns (defense man rate Series, league man rate, player DataFrame with f_man/f_zone).
    Plays before tkey, older seasons down-weighted by season_keep."""
    c = cov[(cov.key < tkey) & cov.man.notna()].copy()
    if not len(c):
        return pd.Series(dtype=float), 0.3, pd.DataFrame(columns=["f_man", "f_zone", "plays"])
    c["w"] = season_keep ** (tkey // 100 - c.season)
    lg = (c.w * c.man).sum() / c.w.sum()
    d = c.groupby("defteam").apply(lambda x: ((x.w * x.man).sum() + MAN_PRIOR * lg) / (x.w.sum() + MAN_PRIOR),
                                   include_groups=False)
    # team pass plays by coverage in each game, credited to every pass catcher who played that game
    tp = c.groupby(["game_id", "posteam", "man"]).w.sum().unstack("man", fill_value=0)
    tp.columns = ["zone" if v == 0 else "man" for v in tp.columns]
    games = s[(s.key < tkey) & s.position.isin(CATCHERS)][["player_id", "game_id", "team"]]
    games = games.join(tp, on=["game_id", "team"], how="inner")
    den = games.groupby("player_id")[["man", "zone"]].sum()
    num = c.dropna(subset=["receiver_player_id"]).assign(wp=lambda x: x.w * x.pts)
    num = num.groupby(["receiver_player_id", "man"]).wp.sum().unstack("man", fill_value=0)
    num.columns = ["zone" if v == 0 else "man" for v in num.columns]
    num = num.reindex(den.index).fillna(0)
    for col in ("man", "zone"):
        if col not in num:
            num[col] = 0.0
    plays = den.man + den.zone
    rate = (num.man + num.zone) / plays.replace(0, np.nan)
    f = pd.DataFrame({"plays": plays})
    f["f_man"] = ((num.man + COV_PRIOR * rate) / (den.man + COV_PRIOR) / rate).fillna(1.0)
    f["f_zone"] = ((num.zone + COV_PRIOR * rate) / (den.zone + COV_PRIOR) / rate).fillna(1.0)
    return d, lg, f


def coverage_mult(r, cov_f, w):
    """Multiplier on a pass catcher's receiving line from this opponent's man rate and his man/zone split."""
    out = pd.Series(1.0, index=r.index)
    if w <= 0 or cov_f is None:
        return out
    dman, lg, f = cov_f
    m = r.opp.map(dman).fillna(lg)
    fm = f.f_man.reindex(r.index).fillna(1.0)
    fz = f.f_zone.reindex(r.index).fillna(1.0)
    ratio = (m * fm + (1 - m) * fz) / (lg * fm + (1 - lg) * fz)
    out[:] = np.where(r.position.isin(CATCHERS), (ratio ** w).clip(0.85, 1.15), 1.0)
    return out


# ----------------------------------------------------------------------------- vs top-5 defenses
def top5_flags(s, keys, opp_factors):
    """For each game key, the defenses ranked top 5 (fewest fantasy points allowed) against each position,
    using only games before that week. Returns a set of (key, defense, position)."""
    tough = set()
    for k in keys:
        f = opp_factors(s, k)
        if not len(f):
            continue
        for pos, x in f.groupby("position"):
            for dfn in x.nsmallest(5, "opp_ratio").defense:
                tough.add((k, dfn, pos))
    return tough


def top5_factors(s, tkey, tough, weights, by="player_id"):
    """How each player (or offense, by='team') scores vs top-5 defenses relative to what top-5 defenses
    usually do to his position. 1.0 = no split."""
    sk = s[s.position.isin(SKILL) & (s.key < tkey) & (s.key >= tkey - 300)].copy()
    sk["tough"] = [(k, o, p) in tough for k, o, p in zip(sk.key, sk.opponent_team, sk.position)]
    if by == "team":
        sk = sk.groupby(["team", "position", "game_id", "season", "key", "tough"]).half.sum().reset_index()
        grp = ["team", "position"]
    else:
        grp = ["player_id"]
    # league suppression: average points in tough games / average overall, by position
    L = sk.groupby("position").apply(lambda x: x[x.tough].half.mean() / x.half.mean(), include_groups=False)
    h = weights(sk, tkey, grp)
    h["pos_L"] = h.position.map(L)
    g = h.assign(wp=h.w * h.half, wt=h.w * h.tough, wpt=h.w * h.half * h.tough).groupby(grp)[["w", "wp", "wt", "wpt"]].sum()
    avg = g.wp / g.w
    Lg = h.groupby(grp).pos_L.first()
    exp_t = g.wt * Lg * avg
    f = (g.wpt + T5_PRIOR * Lg * avg) / (exp_t + T5_PRIOR * Lg * avg)
    return f.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.6, 1.6), L
