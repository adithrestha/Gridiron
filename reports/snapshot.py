"""Keeps the last pre-kickoff copy of every published prediction, so the weekly report grades what was actually shown.

Usage: python3 snapshot.py engine|fantasy|kalshi|all
Called at the end of each refresh script (nfl_engine/v2/refresh.sh, fantasy/refresh_fantasy.sh, kalshi/refresh_kalshi.sh).
For every game that hasn't kicked off yet, it replaces that game's rows in
/mnt/project-files/reports/snapshots/<season>_wk<NN>/{games,fantasy,kalshi}.csv with the current outputs.
Rows for games that already started are never touched, so after kickoff the file holds the last version published before it.
Read-only on the engine, fantasy and Kalshi outputs. Never fails the calling script (errors print and exit 0)."""
import io, json, os, sys, time, urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

PF = "/mnt/project-files"
SNAP = f"{PF}/reports/snapshots"
ET = ZoneInfo("America/New_York")
GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
KEEP_DAYS = 9  # only snapshot games kicking off within this many days
ONLY_WEEK = None  # set by one-off backfills


def schedule():
    """Kickoff time (UTC) for every game: nflverse schedule, falling back to the engine's predictions.csv."""
    try:
        g = pd.read_csv(io.BytesIO(urllib.request.urlopen(GAMES_URL, timeout=60).read()),
                        usecols=["game_id", "season", "week", "game_type", "gameday", "gametime", "away_team", "home_team"])
    except Exception as e:
        print("snapshot: schedule download failed, using predictions.csv:", e)
        g = pd.read_csv(f"{PF}/nfl_engine/v2/output/predictions.csv",
                        usecols=["season", "week", "gameday", "gametime", "away_team", "home_team"])
    g = g.dropna(subset=["gameday", "gametime"]).copy()
    g["kick"] = [datetime.fromisoformat(f"{d}T{t}").replace(tzinfo=ET).astimezone(timezone.utc)
                 for d, t in zip(g.gameday, g.gametime)]
    return g


def upsert(path, new, key, replace_keys):
    """Drop rows of `path` whose key is in replace_keys, append `new`, write atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"id": str})
        old = old[~old[key].isin(replace_keys)]
        new = pd.concat([old, new], ignore_index=True)
    tmp = path + ".tmp"
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def folder(season, week):
    return f"{SNAP}/{int(season)}_wk{int(week):02d}"


def snap_engine(g, now, stamp):
    p = pd.read_csv(f"{PF}/nfl_engine/v2/output/predictions.csv")
    run = json.load(open(f"{PF}/nfl_engine/v2/output/run.json"))
    p = p.merge(g[["season", "week", "away_team", "home_team", "kick"]], on=["season", "week", "away_team", "home_team"])
    p = p[(p.kick > now) & (p.kick < now + timedelta(days=KEEP_DAYS)) & ((ONLY_WEEK is None) | (p.week == ONLY_WEEK))]
    for (s, w), x in p.groupby(["season", "week"]):
        x = x.assign(game=x.away_team + "@" + x.home_team, snap_at=stamp, source_run=run.get("scored_at", "")).drop(columns="kick")
        n = upsert(f"{folder(s, w)}/games.csv", x, "game", set(x.game))
        print(f"snapshot engine: {len(x)} games -> {folder(s, w)} ({n} total)")


FCOLS = ["id", "name", "position", "team", "opp", "week", "status", "p_out", "std", "half", "ppr",
         "pyd", "ptd", "int", "car", "ryd", "rtd", "tgt", "rec", "wyd", "wtd"]


def snap_fantasy(g, now, stamp, src=None, built=None):
    src = src or f"{PF}/fantasy/output/fantasy_projections.csv"
    run = json.load(open(f"{PF}/fantasy/output/fantasy_run.json"))
    built = built or run.get("built_at", "")
    p = pd.read_csv(src, dtype={"id": str})
    season = run["season"]
    gw = g[g.season == season]
    team_kick = pd.concat([gw[["week", "away_team", "kick"]].rename(columns={"away_team": "team"}),
                           gw[["week", "home_team", "kick"]].rename(columns={"home_team": "team"})])
    p = p.merge(team_kick, on=["week", "team"])
    p = p[(p.kick > now) & (p.kick < now + timedelta(days=KEEP_DAYS)) & ((ONLY_WEEK is None) | (p.week == ONLY_WEEK))]
    for w, x in p.groupby("week"):
        x = x[[c for c in FCOLS if c in x.columns]].assign(snap_at=stamp, source_run=built)
        x["key"] = x.id + "|" + x.team
        n = upsert(f"{folder(season, w)}/fantasy.csv", x, "key", set(x.key))
        print(f"snapshot fantasy: {len(x)} players -> {folder(season, w)} ({n} total)")


KCOLS = ["ticker", "kind", "week", "game", "label", "player", "stat", "line", "proj", "side", "p_model", "p_side", "p_adj",
         "yes_bid", "yes_ask", "no_ask", "mid", "last", "book_spread", "volume", "buy_at", "fee", "edge", "kelly", "value",
         "top_pick", "why_not"]


def top_picks(rows, S):
    """Same rule as Gridiron's Betting tab 'Model's top picks' (fantasy_template.html renderBet)."""
    def liquid(r):
        return (r.get("volume", 0) >= S.get("min_volume", 0) and r.get("last") and r.get("book_spread", 1) <= S.get("max_spread", 1)
                and 0.10 <= r["buy_at"] <= 0.90 and "model far" not in (r.get("why_not") or ""))
    top = {}
    for r in rows:
        if not liquid(r) or r["edge"] <= 0:
            continue
        k = f"{r['game']}|{r.get('player')}|{r.get('stat')}" if r["kind"] == "Player prop" else f"{r['game']}|{r['kind']}"
        b = top.get(k)
        if not b or r["p_side"] > b["p_side"] or (r["p_side"] == b["p_side"] and r["side"] == "YES"):
            top[k] = r
    return {r["ticker"] for r in sorted(top.values(), key=lambda r: -r["p_side"])[:12]}


def snap_kalshi(g, now, stamp, src=None):
    src = src or f"{PF}/kalshi/output/kalshi_edges.json"
    d = json.load(open(src))
    rows = d["rows"]
    tops = top_picks(rows, d.get("settings", {}))
    k = pd.DataFrame(rows)
    if k.empty:
        return
    k["top_pick"] = k.ticker.isin(tops)
    k["kick"] = [datetime.fromisoformat(f"{a}T{b}").replace(tzinfo=ET).astimezone(timezone.utc) for a, b in zip(k.gameday, k.gametime)]
    k = k[(k.kick > now) & (k.kick < now + timedelta(days=KEEP_DAYS)) & ((ONLY_WEEK is None) | (k.week == ONLY_WEEK))]
    season = int(g.season.max())
    for w, x in k.groupby("week"):
        x = x[[c for c in KCOLS if c in x.columns]].assign(snap_at=stamp, fetched_at=d.get("fetched_at", ""))
        n = upsert(f"{folder(season, w)}/kalshi.csv", x, "game", set(x.game))
        print(f"snapshot kalshi: {len(x)} markets ({int(x.value.sum())} flagged, {int(x.top_pick.sum())} top picks) -> {folder(season, w)} ({n} total)")


def main(what):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    g = schedule()
    jobs = {"engine": snap_engine, "fantasy": snap_fantasy, "kalshi": snap_kalshi}
    for name in (jobs if what == "all" else [what]):
        try:
            jobs[name](g, now, stamp)
        except Exception as e:
            print(f"snapshot {name} failed: {e!r}")


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else "all")
    except Exception as e:
        print("snapshot failed:", repr(e))
