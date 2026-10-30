"""
Imports each signed-in user's fantasy teams from Sleeper and ESPN.

    python import_leagues.py accounts.json rosters.json [--pending existing_rosters_dir]

--pending      only import viewers whose leagues changed since their last import: their account's
               import_requested_at (or updated_at) is newer than imported_at in existing_rosters_dir/<id>.json,
               or they have no roster document yet. Exits with nothing to do when nobody is pending.

accounts.json  the page's `accounts` collection: a list of {"id": <viewer id>, "data": {...}} (or a dict keyed by
               viewer id), where data = {"sleeper": ["username", ...], "espn": [{"league": "123456"}, ...]}
rosters.json   written here: {<viewer id>: {"imported_at", "errors": [...], "leagues": [...]}}, one document per
               viewer for the page's `rosters/<viewer id>` path.

Each league: {platform, league_id, name, scoring (points per reception), slots, my_team (team id or null),
teams: [{id, name, players: [{id, n, p}]}], standings, results}. standings: [{id, w, l, t, pf, pa}] from the
league's own matchups; results: my team's weeks so far, [{w, final, pts, opp, opp_pts, res, lineup: [[id, slot,
pts, n, p]]}] with the league's own scoring, bench players carrying slot "BN" (ESPN IR "IR"). Player ids are nflverse gsis ids (the projection page's ids);
defenses are DST_<TEAM>. Every rostered player in the league is listed, so the page can tell who is a free agent.
ESPN only answers for leagues set to public ("viewable to public"); private leagues report an error.
"""
import json, os, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEASON = 2026
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
ESPN_TEAMS = {1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
              11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
              21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR",
              30: "JAX", 33: "BAL", 34: "HOU"}
ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
ESPN_SLOTS = {0: "QB", 2: "RB", 3: "RB/WR", 4: "WR", 5: "WR/TE", 6: "TE", 7: "SUPERFLEX", 16: "DST", 17: "K", 23: "FLEX"}
SLEEPER_SLOTS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DST", "FLEX": "FLEX",
                 "SUPER_FLEX": "SUPERFLEX", "REC_FLEX": "WR/TE", "WRRB_FLEX": "RB/WR"}
ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league}"
            "?view=mTeam&view=mRoster&view=mSettings")
SLEEPER = "https://api.sleeper.app/v1"


def get_json(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gridiron-fantasy/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            err = e
        except Exception as e:     # network hiccup
            err = e
        time.sleep(2 ** i)
    raise err


class IdMap:
    """Maps Sleeper / ESPN player ids (and names, as a fallback) to nflverse gsis ids."""
    def __init__(self):
        r = pd.read_csv(os.path.join(DATA, f"roster_{SEASON}.csv"), low_memory=False)
        r = r.sort_values("week").drop_duplicates("gsis_id", keep="last").dropna(subset=["gsis_id"])
        num = lambda v: str(int(v)) if pd.notna(v) else None
        self.sleeper = {num(s): g for s, g in zip(r.sleeper_id, r.gsis_id) if num(s)}
        self.espn = {num(e): g for e, g in zip(r.espn_id, r.gsis_id) if num(e)}
        self.info = {g: (n, p) for g, n, p in zip(r.gsis_id, r.full_name, r.position)}
        self.by_name = {}
        for n, g, pos in zip(r.full_name, r.gsis_id, r.position):
            self.by_name.setdefault((norm(n), pos), g)

    def get(self, table, pid, name, pos):
        if pos == "DST":
            return None
        return table.get(str(pid)) or self.by_name.get((norm(name or ""), pos))


def norm(s):
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))


def player(gsis, name, pos, team=None):
    if pos == "DST":
        return {"id": f"DST_{team}", "n": f"{team} D/ST", "p": "DST"}
    return {"id": gsis or f"x:{name}", "n": name, "p": pos}


def current_week():
    """(latest week with every game final, latest week with any game played) from nflverse's schedule."""
    g = pd.read_csv(os.path.join(DATA, "games.csv"), low_memory=False)
    g = g[(g.season == SEASON) & (g.game_type == "REG")]
    done = g.groupby("week").home_score.apply(lambda x: x.notna().all())
    started = g.groupby("week").home_score.apply(lambda x: x.notna().any())
    last = int(done[done].index.max()) if done.any() else 0
    return last, int(started[started].index.max()) if started.any() else 0


def standings(games, team_ids):
    """games: [(week, final, team_a, pts_a, team_b, pts_b)] -> [{id, w, l, t, pf, pa}], best record first."""
    st = {t: {"id": t, "w": 0, "l": 0, "t": 0, "pf": 0.0, "pa": 0.0} for t in team_ids}
    for _, final, a, pa, b, pb in games:
        if not final or a not in st or b not in st:
            continue
        st[a]["pf"] += pa; st[a]["pa"] += pb; st[b]["pf"] += pb; st[b]["pa"] += pa
        if pa > pb: st[a]["w"] += 1; st[b]["l"] += 1
        elif pb > pa: st[b]["w"] += 1; st[a]["l"] += 1
        else: st[a]["t"] += 1; st[b]["t"] += 1
    for v in st.values():
        v["pf"], v["pa"] = round(v["pf"], 2), round(v["pa"], 2)
    return sorted(st.values(), key=lambda v: (-(v["w"] + v["t"] / 2), -v["pf"]))


def result_row(w, final, pts, opp, opp_pts, lineup):
    res = None
    if final and opp is not None:
        res = "W" if pts > opp_pts else "L" if pts < opp_pts else "T"
    return {"w": w, "final": final, "pts": round(pts, 2), "opp": opp, "opp_pts": None if opp_pts is None else round(opp_pts, 2),
            "res": res, "lineup": lineup}

# ----------------------------------------------------------------------------- Sleeper
def sleeper_leagues(username, ids, sleeper_players):
    user = get_json(f"{SLEEPER}/user/{username}")
    if not user:
        raise ValueError(f"Sleeper user '{username}' not found")
    uid = user["user_id"]
    out = []
    for lg in get_json(f"{SLEEPER}/user/{uid}/leagues/nfl/{SEASON}") or []:
        lid = lg["league_id"]
        rosters = get_json(f"{SLEEPER}/league/{lid}/rosters") or []
        users = {u["user_id"]: u for u in (get_json(f"{SLEEPER}/league/{lid}/users") or [])}
        teams, mine = [], None
        for ro in rosters:
            owner = users.get(ro.get("owner_id") or "", {})
            name = (owner.get("metadata") or {}).get("team_name") or owner.get("display_name") or f"Team {ro['roster_id']}"
            plist = []
            for sid in ro.get("players") or []:
                info = sleeper_players.get(sid, {})
                if sid.isalpha():                                   # team defense, e.g. "PIT"
                    plist.append(player(None, None, "DST", TEAM_FIX.get(sid, sid)))
                    continue
                pos = info.get("position") or "?"
                name_ = info.get("full_name") or " ".join(filter(None, [info.get("first_name"), info.get("last_name")])) or sid
                gsis = ids.get(ids.sleeper, sid, name_, pos)
                if gsis in ids.info:                                # nflverse knows him: use its name/position
                    name_, pos = ids.info[gsis]
                plist.append(player(gsis, name_, pos))
            teams.append({"id": str(ro["roster_id"]), "name": name, "players": plist})
            if uid == ro.get("owner_id") or uid in (ro.get("co_owners") or []):
                mine = str(ro["roster_id"])
        slots = [SLEEPER_SLOTS[s] for s in lg.get("roster_positions", []) if s in SLEEPER_SLOTS]
        league = {"platform": "sleeper", "league_id": lid, "name": lg.get("name") or "Sleeper league",
                  "scoring": float((lg.get("scoring_settings") or {}).get("rec", 0) or 0),
                  "slots": slots, "my_team": mine, "teams": teams}
        try:
            sleeper_results(league, lg, ids, sleeper_players)
        except Exception as e:
            print(f"  Sleeper results for {lid}: {e}")
        out.append(league)
    return out


def sleeper_pid(sid, ids, sleeper_players):
    """(page id, name, position) for a Sleeper player id."""
    if sid.isalpha():
        t = TEAM_FIX.get(sid, sid)
        return f"DST_{t}", f"{t} D/ST", "DST"
    info = sleeper_players.get(sid, {})
    pos = info.get("position") or "?"
    name = info.get("full_name") or " ".join(filter(None, [info.get("first_name"), info.get("last_name")])) or sid
    gsis = ids.get(ids.sleeper, sid, name, pos)
    if gsis in ids.info:
        name, pos = ids.info[gsis]
    return gsis or f"x:{name}", name, pos


def sleeper_results(league, lg, ids, sleeper_players):
    """Weekly scores from Sleeper's matchups: league standings and my team's week-by-week lineup and points."""
    last, live = current_week()
    start = int((lg.get("settings") or {}).get("start_week") or 1)
    slots = league["slots"]
    games, mine = [], []
    for w in range(start, live + 1):
        ms = get_json(f"{SLEEPER}/league/{league['league_id']}/matchups/{w}") or []
        final = w <= last
        by = {}
        for m in ms:
            if m.get("matchup_id") is not None:
                by.setdefault(m["matchup_id"], []).append(m)
        for pair in by.values():
            if len(pair) == 2:
                a, b = pair
                games.append((w, final, str(a["roster_id"]), a.get("points") or 0, str(b["roster_id"]), b.get("points") or 0))
        me = next((m for m in ms if str(m.get("roster_id")) == league["my_team"]), None)
        if not me:
            continue
        opp = next((m for m in by.get(me.get("matchup_id"), []) if m is not me), None) if me.get("matchup_id") is not None else None
        pp = me.get("players_points") or {}
        starters = [x for x in (me.get("starters") or [])]
        lineup = []
        for i, sid in enumerate(starters):
            if not sid or sid == "0":
                continue
            pid, n, p = sleeper_pid(sid, ids, sleeper_players)
            lineup.append([pid, slots[i] if i < len(slots) else "FLEX", round(pp.get(sid, 0) or 0, 2), n, p])
        for sid in me.get("players") or []:
            if sid not in starters:
                pid, n, p = sleeper_pid(sid, ids, sleeper_players)
                lineup.append([pid, "BN", round(pp.get(sid, 0) or 0, 2), n, p])
        mine.append(result_row(w, final, me.get("points") or 0, str(opp["roster_id"]) if opp else None,
                               (opp.get("points") or 0) if opp else None, lineup))
    league["standings"] = standings(games, [t["id"] for t in league["teams"]])
    league["results"] = mine


def sleeper_player_table():
    """Sleeper's player directory (~5 MB), only for names/positions of ids nflverse can't map. Cached per day."""
    path = os.path.join(DATA, "sleeper_players.json")
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > 86400:
        try:
            with open(path, "w") as f:
                json.dump(get_json(f"{SLEEPER}/players/nfl"), f)
        except Exception as e:
            print(f"  could not fetch Sleeper player directory: {e}")
            return {}
    return json.load(open(path))

# ----------------------------------------------------------------------------- ESPN
def espn_league(league_id, ids):
    try:
        j = get_json(ESPN_URL.format(season=SEASON, league=league_id))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ValueError(f"ESPN league {league_id} is private. In ESPN league settings, turn on "
                             "'Make league viewable to public', or paste your roster instead.")
        if e.code == 404:
            raise ValueError(f"ESPN league {league_id} was not found for {SEASON}")
        raise
    st = j.get("settings", {})
    counts = (st.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
    slots = []
    for sid, n in sorted(counts.items(), key=lambda kv: int(kv[0])):
        if int(sid) in ESPN_SLOTS:
            slots += [ESPN_SLOTS[int(sid)]] * int(n)
    order = ["QB", "RB", "RB/WR", "WR", "WR/TE", "TE", "FLEX", "SUPERFLEX", "K", "DST"]
    slots.sort(key=order.index)
    rec = next((i.get("points", 0) for i in (st.get("scoringSettings") or {}).get("scoringItems", [])
                if i.get("statId") == 53), 0)
    teams = []
    for t in j.get("teams", []):
        name = t.get("name") or " ".join(filter(None, [t.get("location"), t.get("nickname")])) or t.get("abbrev") or f"Team {t['id']}"
        plist = []
        for e in (t.get("roster") or {}).get("entries", []):
            p = (e.get("playerPoolEntry") or {}).get("player") or {}
            pos = ESPN_POS.get(p.get("defaultPositionId"), "?")
            if pos == "DST":
                plist.append(player(None, None, "DST", ESPN_TEAMS.get(p.get("proTeamId"), "?")))
            else:
                plist.append(player(ids.get(ids.espn, e.get("playerId"), p.get("fullName"), pos), p.get("fullName") or "?", pos))
        teams.append({"id": str(t["id"]), "name": name, "players": plist})
    return {"platform": "espn", "league_id": str(league_id), "name": st.get("name") or f"ESPN league {league_id}",
            "scoring": float(rec), "slots": slots, "my_team": None, "teams": teams}


ESPN_SCORES = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league}"
               "?view=mMatchupScore&view=mBoxscore&scoringPeriodId={week}")
ESPN_BENCH = {20: "BN", 21: "IR"}


def espn_results(league, ids):
    """Weekly scores from ESPN's schedule and box scores: league standings and my team's weeks."""
    last, live = current_week()
    games, mine = [], []
    for w in range(1, live + 1):
        j = get_json(ESPN_SCORES.format(season=SEASON, league=league["league_id"], week=w))
        final = w <= last
        for m in j.get("schedule", []):
            if m.get("matchupPeriodId") != w or "away" not in m:
                continue
            h, a = m["home"], m["away"]
            games.append((w, final and m.get("winner") != "UNDECIDED", str(h["teamId"]), h.get("totalPoints") or 0,
                          str(a["teamId"]), a.get("totalPoints") or 0))
            for me, opp in ((h, a), (a, h)):
                if str(me["teamId"]) != league["my_team"]:
                    continue
                lineup = []
                for e in (me.get("rosterForCurrentScoringPeriod") or {}).get("entries", []):
                    ppe = e.get("playerPoolEntry") or {}
                    p = ppe.get("player") or {}
                    pos = ESPN_POS.get(p.get("defaultPositionId"), "?")
                    if pos == "DST":
                        t = ESPN_TEAMS.get(p.get("proTeamId"), "?")
                        pid, n = f"DST_{t}", f"{t} D/ST"
                    else:
                        n = p.get("fullName") or "?"
                        pid = ids.get(ids.espn, e.get("playerId"), n, pos) or f"x:{n}"
                        if pid in ids.info:
                            n, pos = ids.info[pid]
                    slot = ESPN_BENCH.get(e.get("lineupSlotId"), ESPN_SLOTS.get(e.get("lineupSlotId"), "BN"))
                    lineup.append([pid, slot, round(ppe.get("appliedStatTotal") or 0, 2), n, pos])
                mine.append(result_row(w, final, me.get("totalPoints") or 0, str(opp["teamId"]), opp.get("totalPoints") or 0, lineup))
    league["standings"] = standings(games, [t["id"] for t in league["teams"]])
    league["results"] = mine

# ----------------------------------------------------------------------------- main
def friendly(source, host, e):
    """Readable error for the page; network-policy blocks name the host to allow."""
    msg = str(e)
    if "Tunnel connection failed" in msg or "Forbidden" in msg and "urlopen" in msg:
        return (f"{source}: can't reach {host} yet. The page owner needs to allow {host} in the project's network "
                "settings; your leagues will import automatically once that's done.")
    return f"{source}: {msg}"


def import_all(accounts):
    ids = IdMap()
    sp = None
    out = {}
    for uid, acct in accounts.items():
        leagues, errors = [], []
        for name in acct.get("sleeper") or []:
            name = str(name).strip()
            if not name:
                continue
            try:
                sp = sp if sp is not None else sleeper_player_table()
                leagues += sleeper_leagues(name, ids, sp)
            except Exception as e:
                errors.append(friendly(f"Sleeper {name}", "api.sleeper.app", e))
        for e in acct.get("espn") or []:
            lid = str(e.get("league") if isinstance(e, dict) else e).strip()
            if not lid:
                continue
            try:
                lg = espn_league(lid, ids)
                if isinstance(e, dict) and e.get("team"):
                    lg["my_team"] = str(e["team"])
                try:
                    espn_results(lg, ids)
                except Exception as ex2:
                    print(f"  ESPN results for {lid}: {ex2}")
                leagues.append(lg)
            except Exception as ex:
                errors.append(friendly(f"ESPN league {lid}", "lm-api-reads.fantasy.espn.com", ex))
        out[uid] = {"imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "errors": errors, "leagues": leagues, "retry": bool(errors)}
        print(f"{uid}: {len(leagues)} leagues, {len(errors)} errors")
    return out


def when(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def pending(accounts, rosters_dir):
    """Viewers whose leagues changed after their last import (or who were never imported)."""
    out = {}
    for uid, acct in accounts.items():
        if not (acct.get("sleeper") or acct.get("espn")):
            continue
        path = os.path.join(rosters_dir, f"{uid}.json")
        done = None
        if os.path.exists(path):
            doc = json.load(open(path))
            done = when((doc.get("data") or doc).get("imported_at"))
        asked = when(acct.get("import_requested_at")) or when(acct.get("updated_at"))
        retry = os.path.exists(path) and bool((doc.get("data") or doc).get("retry"))
        if done is None or retry or (asked is not None and asked > done):
            out[uid] = acct
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src, dst = args[0], args[1]
    raw = json.load(open(src))
    if isinstance(raw, dict) and "documents" in raw:
        raw = raw["documents"]
    if isinstance(raw, list):
        accounts = {d.get("id") or d.get("doc_id"): d.get("data", d) for d in raw}
    else:
        accounts = raw
    if "--pending" in sys.argv:
        accounts = pending(accounts, sys.argv[sys.argv.index("--pending") + 1])
        if not accounts:
            print("nothing pending")
    with open(dst, "w") as f:
        json.dump(import_all(accounts), f)


if __name__ == "__main__":
    main()
