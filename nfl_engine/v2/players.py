"""
Player-level features for the engine
====================================
  * QB ratings: EPA/play and CPOE blended into one number per QB, exponentially weighted across
    games and seasons and shrunk toward replacement level when the sample is small.
  * Starters: the QB who took most dropbacks in played games; for upcoming games the current
    depth chart QB1 (checked against the weekly roster), swapped for the backup in proportion
    to his injury designation.
  * Injury load: each injured non-QB is weighted by position x snap share with his CURRENT team,
    so a starting left tackle counts far more than a backup and new signings count little.
Ratings are keyed by player id, so a traded or newly signed QB brings his rating with him.
"""
import numpy as np, pandas as pd

TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
QB_DECAY = 0.94           # weight kept on older evidence each time the QB plays (~11 game half-life)
OFFSEASON_KEEP = 0.7      # weight kept on prior seasons when a new season starts
PRIOR_PLAYS = 250         # shrinkage: plays of replacement-level evidence every QB starts with
CPOE_W = 0.01             # rating = EPA/play + 0.01 * CPOE (pct points), roughly equal spread
TEAM_QB_HALFLIFE = 8      # games, for the team's "normal" QB level (the Elo-style adjustment)

POS_W = {"T": 1.5, "G": 1.0, "C": 1.0, "OL": 1.0, "WR": 1.2, "TE": 0.8, "RB": 0.8, "FB": 0.3,
         "CB": 1.2, "S": 0.8, "FS": 0.8, "SS": 0.8, "DB": 0.8, "DE": 1.2, "DT": 0.9, "NT": 0.8,
         "DL": 1.0, "LB": 0.8, "OLB": 1.0, "ILB": 0.8, "MLB": 0.8, "K": 0.3, "P": 0.1, "LS": 0.1}
STATUS_W = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}
INACTIVE_ROSTER = {"RES", "CUT", "RET", "TRD", "TRT", "EXE", "DEV", "SUS", "NWT"}
NEW_PLAYER_SHARE = 0.1    # snap share for players with no snaps for their current team yet


def _week_key(season, week):
    return season * 100 + week


# ----------------------------------------------------------------------------- QB ratings
def qb_games(ps):
    """One row per QB per game with plays, EPA and CPOE inputs; flags the team's starter."""
    q = ps[(ps.position == "QB") & ps.season_type.isin(["REG", "POST"])].copy()
    q["team"] = q.team.replace(TEAM_FIX)
    f = lambda c: q[c].fillna(0)
    q["dropbacks"] = f("attempts") + f("sacks_suffered")
    q["plays"] = q.dropbacks + f("carries")
    q["epa"] = f("passing_epa") + f("rushing_epa")
    q["cpoe_att"] = np.where(q.passing_cpoe.notna(), f("attempts"), 0)
    q["cpoe_sum"] = q.passing_cpoe.fillna(0) * q.cpoe_att
    q = q[q.plays > 0]
    q["starter"] = q.dropbacks == q.groupby(["game_id", "team"]).dropbacks.transform("max")
    q = q.sort_values(["game_id", "team", "dropbacks"], ascending=[True, True, False])
    q.loc[q.duplicated(["game_id", "team"]), "starter"] = False
    return q.sort_values(["season", "week"]).reset_index(drop=True)


def replacement_level(q):
    """EPA/play and CPOE of fringe QBs: seasons with 50-250 dropbacks (backups, spot starters)."""
    s = q.groupby(["player_id", "season"])[["dropbacks", "plays", "epa", "cpoe_att", "cpoe_sum"]].sum()
    s = s[(s.dropbacks >= 50) & (s.dropbacks <= 250)]
    return s.epa.sum() / s.plays.sum(), s.cpoe_sum.sum() / max(s.cpoe_att.sum(), 1)


def qb_ratings(q):
    """Returns (q with pre-game `qb_rating` per row, dict of current rating per QB, replacement)."""
    rep_epa, rep_cpoe = replacement_level(q)
    rep = rep_epa + CPOE_W * rep_cpoe
    state, pre = {}, np.empty(len(q))
    for i, r in enumerate(q.itertuples()):
        s = state.get(r.player_id)
        if s is None:
            s = state[r.player_id] = [0.0, 0.0, 0.0, 0.0, r.season]   # epa, plays, cpoe_sum, cpoe_att, season
        if s[4] != r.season:
            s[:4] = [v * OFFSEASON_KEEP ** (r.season - s[4]) for v in s[:4]]
            s[4] = r.season
        pre[i] = _rating(s, rep_epa, rep_cpoe)
        s[:4] = [s[0] * QB_DECAY + r.epa, s[1] * QB_DECAY + r.plays,
                 s[2] * QB_DECAY + r.cpoe_sum, s[3] * QB_DECAY + r.cpoe_att]
    q = q.copy()
    q["qb_rating"] = pre
    current = {pid: _rating(s, rep_epa, rep_cpoe) for pid, s in state.items()}
    return q, current, rep


def _rating(s, rep_epa, rep_cpoe):
    epa = (s[0] + PRIOR_PLAYS * rep_epa) / (s[1] + PRIOR_PLAYS)
    cpoe = (s[2] + PRIOR_PLAYS * rep_cpoe) / (s[3] + PRIOR_PLAYS)
    return epa + CPOE_W * cpoe


def team_qb_norm(starts):
    """Team's usual QB level entering each game: EWM of its previous starters' ratings."""
    starts = starts.sort_values(["team", "season", "week"])
    starts["team_qb_norm"] = starts.groupby("team").qb_rating.transform(
        lambda x: x.ewm(halflife=TEAM_QB_HALFLIFE).mean().shift(1))
    latest = starts.groupby("team").qb_rating.apply(lambda x: x.ewm(halflife=TEAM_QB_HALFLIFE).mean().iloc[-1])
    return starts, latest.to_dict()


# ----------------------------------------------------------------------------- starters
def out_probability(inj):
    """Probability each listed player misses the game, from designation or (pre-Friday) practice."""
    inj = inj.copy()
    has_final = inj.groupby(["season", "week", "team"]).report_status.transform("count") > 0
    prac = inj.practice_status.fillna("")
    est = np.where(prac.str.startswith("Did Not"), 0.6, np.where(prac.str.startswith("Limited"), 0.2, 0.0))
    inj["p_out"] = np.where(has_final, inj.report_status.map(STATUS_W).fillna(0), est)
    return inj


def depth_chart_qbs(depth, roster):
    """Ordered QB list per team from the latest depth chart, minus anyone not on the active roster."""
    d = depth[depth.dt == depth.dt.max()]
    d = d[d.pos_abb == "QB"].sort_values("pos_rank").drop_duplicates(["team", "gsis_id"])
    d["team"] = d.team.replace(TEAM_FIX)
    r = roster[roster.week == roster.week.max()]
    status = dict(zip(zip(r.gsis_id, r.team.replace(TEAM_FIX)), r.status))
    ok = [status.get((g, t), "ACT") not in INACTIVE_ROSTER for g, t in zip(d.gsis_id, d.team)]
    d = d[ok]
    return {t: list(zip(x.gsis_id, x.player_name)) for t, x in d.groupby("team")}


def projected_qb(team, season, week, depth_qbs, last_starter, ratings, rep, inj_p, names):
    """Expected QB rating for an upcoming game: QB1, blended with the next healthy QB by P(out)."""
    qbs = depth_qbs.get(team) or [last_starter.get(team, (None, None))]
    rating = lambda pid: ratings.get(pid, rep)
    exp, remaining, label = 0.0, 1.0, None
    qb1_out = inj_p.get((season, week, team, qbs[0][0]), 0.0) >= 0.5
    for pid, name in qbs:
        p = inj_p.get((season, week, team, pid), 0.0)
        if label is None and p < 0.5:
            label = name or names.get(pid)       # most likely starter
        exp += remaining * (1 - p) * rating(pid)
        remaining *= p
        if remaining < 1e-3:
            break
    exp += remaining * rep
    return exp, label or qbs[0][1], qb1_out


# ----------------------------------------------------------------------------- snap-share injuries
def snap_shares(snaps, players):
    """Exponentially weighted snap share per player per team after each game he played."""
    ids = players.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    s = snaps[snaps.game_type == "REG"].copy() if "game_type" in snaps else snaps.copy()
    s["gsis_id"] = s.pfr_player_id.map(ids)
    s = s.dropna(subset=["gsis_id"])
    s["team"] = s.team.replace(TEAM_FIX)
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1).fillna(0)
    s["key"] = _week_key(s.season.astype(int), s.week.astype(int))
    s = s.sort_values("key")
    s["share_ew"] = s.groupby(["gsis_id", "team"]).share.transform(lambda x: x.ewm(halflife=3).mean())
    return s[["gsis_id", "team", "key", "season", "share_ew"]]


def injury_load(inj, shares, first_snap_season):
    """Sum over injured non-QBs of position weight x snap share x P(out), per team-week."""
    inj = inj[inj.game_type.isin(["REG", "WC", "DIV", "CON", "SB"]) | inj.game_type.isna()].copy()
    inj["team"] = inj.team.replace(TEAM_FIX)
    inj = out_probability(inj)
    inj = inj[(inj.position != "QB") & (inj.p_out > 0) & inj.gsis_id.notna()].copy()
    inj["key"] = _week_key(inj.season, inj.week)
    inj = inj.sort_values("key")
    m = pd.merge_asof(inj, shares.rename(columns={"season": "share_season"}), on="key",
                      by=["gsis_id", "team"], allow_exact_matches=False)
    stale = (m.season - m.share_season).fillna(0).clip(lower=0)
    m["share"] = m.share_ew * 0.8 ** stale
    m["share"] = m.share.fillna(pd.Series(np.where(m.season < first_snap_season, 0.5, NEW_PLAYER_SHARE), index=m.index))
    m["w"] = m.position.map(POS_W).fillna(0.6) * m.share * m.p_out
    return m.groupby(["season", "week", "team"]).w.sum().rename("inj_load").reset_index()
