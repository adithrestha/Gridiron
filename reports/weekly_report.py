"""Weekly accuracy report: grades game picks, fantasy projections and Kalshi betting against what really happened,
using the pre-kickoff copies saved by snapshot.py. Run after Monday Night Football (scheduled Tuesday morning).

Usage: python3 weekly_report.py [--week N] [--season YYYY]
  default week = latest week of the season whose games are all final.
Writes reports/graded/<season>_wk<NN>/{games,fantasy,kalshi}.csv (row-level grades),
reports/output/report.json (all graded weeks + season to date; embedded in Gridiron's Report tab)
and reports/output/<season>_wk<NN>.md (short summary).
Exit 0 = done; exit 3 = fantasy stats for that week not posted by nflverse yet (game and betting grades are still written).
Read-only on everything outside reports/. Kalshi: public GET of market results only."""
import argparse, glob, io, json, math, os, sys, time, urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

PF = "/mnt/project-files"
R = f"{PF}/reports"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
KAPI = "https://api.elections.kalshi.com/trade-api/v2/markets"
ET = ZoneInfo("America/New_York")
FMIN = 5.0  # fantasy players graded: projected at least this many half-PPR points


def get(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(url, timeout=90).read()
        except Exception:
            if i == 4:
                raise
            time.sleep(2 ** i)


def r(x, d=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


# ------------------------------------------------------------------ games
def grade_games(snap, sched):
    s = sched.copy()
    s["game"] = s.away_team + "@" + s.home_team
    g = snap.merge(s[["game", "away_score", "home_score", "spread_line", "total_line", "home_moneyline", "away_moneyline", "gameday"]],
                   on="game", suffixes=("", "_s"))
    g = g.dropna(subset=["home_score"]).copy()
    g["margin"] = g.home_score - g.away_score
    g["total"] = g.home_score + g.away_score
    g["home_won"] = np.where(g.margin > 0, 1.0, np.where(g.margin < 0, 0.0, 0.5))
    g["correct"] = np.where(g.margin == 0, np.nan, ((g.home_win_prob > 0.5) == (g.margin > 0)).astype(float))
    g["brier"] = (g.home_win_prob - g.home_won) ** 2
    p = g.home_win_prob.clip(1e-6, 1 - 1e-6)
    g["logloss"] = -(g.home_won * np.log(p) + (1 - g.home_won) * np.log(1 - p))
    g["margin_err"] = g.pred_margin - g.margin
    g["total_err"] = g.pred_total - g.total
    # Vegas: favorite from the closing line, vig-free win chance from the moneylines
    g["vegas_correct"] = np.where((g.spread_line == 0) | (g.margin == 0), np.nan, ((g.spread_line > 0) == (g.margin > 0)).astype(float))
    def ml(m):
        return np.where(m < 0, -m / (-m + 100), 100 / (m + 100))
    ph, pa = ml(g.home_moneyline.astype(float)), ml(g.away_moneyline.astype(float))
    g["vegas_home_prob"] = ph / (ph + pa)
    g["vegas_brier"] = (g.vegas_home_prob - g.home_won) ** 2
    g["vegas_margin_err"] = g.spread_line - g.margin
    g["vegas_total_err"] = g.total_line - g.total
    line = g.vegas_home_spread.fillna(g.spread_line)
    g["ats_line"] = line
    g["ats_pick"] = np.where(g.pred_margin > line, g.home_team, g.away_team)
    g["ats_right"] = np.where(g.margin == line, np.nan, ((g.pred_margin > line) == (g.margin > line)).astype(float))
    g["ou_pick"] = np.where(g.pred_total > g.total_line, "Over", "Under")
    g["ou_right"] = np.where(g.total == g.total_line, np.nan, ((g.pred_total > g.total_line) == (g.total > g.total_line)).astype(float))
    return g


def sum_games(g):
    if g is None or not len(g):
        return None
    n = lambda c: int(g[c].notna().sum())
    return dict(games=len(g), correct=int(g.correct.sum()), accuracy=r(g.correct.mean()),
                brier=r(g.brier.mean()), logloss=r(g.logloss.mean()),
                margin_mae=r(g.margin_err.abs().mean(), 1), total_mae=r(g.total_err.abs().mean(), 1),
                vegas_correct=int(g.vegas_correct.sum()), vegas_games=n("vegas_correct"), vegas_accuracy=r(g.vegas_correct.mean()),
                vegas_brier=r(g.vegas_brier.mean()), vegas_margin_mae=r(g.vegas_margin_err.abs().mean(), 1),
                vegas_total_mae=r(g.vegas_total_err.abs().mean(), 1),
                ats_w=int(g.ats_right.sum()), ats_n=n("ats_right"), ou_w=int(g.ou_right.sum()), ou_n=n("ou_right"))


def game_rows(g):
    out = []
    for x in g.sort_values(["gameday", "gametime"]).itertuples():
        out.append(dict(game=f"{x.away_team} @ {x.home_team}", pick=x.pick, prob=r(max(x.home_win_prob, 1 - x.home_win_prob)),
                        pred=f"{x.away_team} {x.pred_away_pts:.0f}, {x.home_team} {x.pred_home_pts:.0f}",
                        final=f"{x.away_team} {x.away_score:.0f}, {x.home_team} {x.home_score:.0f}",
                        correct=None if np.isnan(x.correct) else bool(x.correct),
                        vegas_correct=None if np.isnan(x.vegas_correct) else bool(x.vegas_correct),
                        ats=f"{x.ats_pick} {'+' if (x.ats_pick == x.away_team) == (x.ats_line > 0) else '-'}{abs(x.ats_line):g}",
                        ats_right=None if np.isnan(x.ats_right) else bool(x.ats_right),
                        ou=f"{x.ou_pick} {x.total_line:g}", ou_right=None if np.isnan(x.ou_right) else bool(x.ou_right),
                        margin_err=r(x.margin_err, 1), total_err=r(x.total_err, 1)))
    return out


# ------------------------------------------------------------------ fantasy
def load_stats(season):
    s = pd.read_csv(io.BytesIO(get(f"{REL}/stats_player/stats_player_week_{season}.csv")), low_memory=False)
    return s[s.season_type == "REG"] if "season_type" in s else s


def grade_fantasy(snap, stats, week, teams_played):
    wk = stats[stats.week == week].copy()
    missing = sorted(set(teams_played) - set(wk.team.unique()))
    if missing:
        return None, missing
    f = snap[~snap.position.isin(["K", "DST"]) & snap.team.isin(teams_played)].copy()
    a = wk.assign(act_half=wk.fantasy_points + 0.5 * wk.receptions, act_ppr=wk.fantasy_points_ppr)[
        ["player_id", "act_half", "act_ppr", "fantasy_points"]].rename(columns={"fantasy_points": "act_std"})
    f = f.merge(a, left_on="id", right_on="player_id", how="left")
    f["played"] = f.act_half.notna()
    for c in ("act_half", "act_ppr", "act_std"):
        f[c] = f[c].fillna(0.0)
    prior = stats[stats.week < week]
    prior = prior.assign(h=prior.fantasy_points + 0.5 * prior.receptions).groupby("player_id").h.mean()
    f["naive_half"] = f.id.map(prior)
    f["err_half"] = f.half - f.act_half
    f["err_ppr"] = f.ppr - f.act_ppr
    f["week"] = week
    return f.drop(columns=["player_id"]), []


def sum_fantasy(f):
    if f is None or not len(f):
        return None
    t = f[f.half >= FMIN]
    nv = t.dropna(subset=["naive_half"])
    return dict(players=len(t), mae_half=r(t.err_half.abs().mean(), 2), mae_ppr=r(t.err_ppr.abs().mean(), 2),
                bias_half=r(t.err_half.mean(), 2), within5=r((t.err_half.abs() <= 5).mean()),
                corr=r(t.half.corr(t.act_half)), dnp=int((~t.played).sum()),
                naive_n=len(nv), naive_mae=r((nv.naive_half - nv.act_half).abs().mean(), 2),
                model_mae_same=r(nv.err_half.abs().mean(), 2),
                by_pos={p: dict(n=len(x), mae=r(x.err_half.abs().mean(), 2), bias=r(x.err_half.mean(), 2))
                        for p, x in t.groupby("position")})


def fantasy_rows(f):
    t = f[f.half >= FMIN]
    pick = lambda x: [dict(name=y.name, pos=y.position, team=y.team, proj=r(y.half, 1), act=r(y.act_half, 1),
                           err=r(y.err_half, 1), played=bool(y.played)) for y in x.itertuples()]
    return dict(top=pick(t.sort_values("half", ascending=False).head(25)),
                over=pick(t.sort_values("err_half").head(6)),   # scored far more than projected
                under=pick(t.sort_values("err_half", ascending=False).head(6)))  # scored far less


# ------------------------------------------------------------------ kalshi
def settle(tickers):
    res = {}
    t = list(tickers)
    for i in range(0, len(t), 100):
        d = json.loads(get(f"{KAPI}?limit=1000&tickers=" + ",".join(t[i:i + 100])))
        for m in d.get("markets", []):
            res[m["ticker"]] = m.get("result", "")
        time.sleep(0.2)
    return res


def grade_kalshi(snap):
    k = snap.copy()
    res = settle(k.ticker.unique())
    k["result"] = k.ticker.map(res).fillna("")
    k = k[k.result.isin(["yes", "no"])].copy()
    y = (k.result == "yes").astype(float)
    k["yes"] = y
    k["p_yes_shown"] = np.where(k.side == "YES", k.p_adj, 1 - k.p_adj)
    k["brier_model"] = (k.p_model - y) ** 2
    k["brier_shown"] = (k.p_yes_shown - y) ** 2
    k["brier_kalshi"] = (k.mid - y) ** 2
    k["won"] = ((k.side == "YES") == (y == 1)).astype(float)
    k["cost"] = k.buy_at + k.fee
    k["profit"] = k.won - k.cost
    for c in ("value", "top_pick"):
        k[c] = k[c].astype(str).str.lower().eq("true")
    return k


def sum_bets(x):
    if not len(x):
        return dict(n=0, won=0, cost=0.0, profit=0.0, roi=None)
    return dict(n=len(x), won=int(x.won.sum()), cost=r(x.cost.sum(), 2), profit=r(x.profit.sum(), 2), roi=r(x.profit.sum() / x.cost.sum()))


def sum_kalshi(k):
    if k is None or not len(k):
        return None
    return dict(markets=len(k), brier_model=r(k.brier_model.mean(), 4), brier_shown=r(k.brier_shown.mean(), 4),
                brier_kalshi=r(k.brier_kalshi.mean(), 4), lean_right=r(k.won.mean()),
                value_bets=sum_bets(k[k.value]), top_picks=sum_bets(k[k.top_pick]),
                by_kind={kd: dict(n=len(x), brier_model=r(x.brier_model.mean(), 4), brier_shown=r(x.brier_shown.mean(), 4),
                                  brier_kalshi=r(x.brier_kalshi.mean(), 4), lean_right=r(x.won.mean()))
                         for kd, x in k.groupby("kind")})


def kalshi_rows(k):
    x = k[k.value | k.top_pick].sort_values(["game", "label"])
    return [dict(game=y.game, label=y.label, side=y.side, price=r(y.buy_at, 2), p=r(y.p_side), edge=r(y.edge),
                 value=bool(y.value), top=bool(y.top_pick), won=bool(y.won), profit=r(y.profit, 2)) for y in x.itertuples()]


# ------------------------------------------------------------------ coverage (how fresh the graded copy was)
def coverage(snap, sched, team_col=None):
    s = sched.copy()
    s["kick"] = [datetime.fromisoformat(f"{d}T{t}").replace(tzinfo=ET).astimezone(timezone.utc) for d, t in zip(s.gameday, s.gametime)]
    if team_col:
        m = pd.concat([s[["away_team", "kick"]].rename(columns={"away_team": "t"}), s[["home_team", "kick"]].rename(columns={"home_team": "t"})])
        x = snap.merge(m, left_on=team_col, right_on="t")
    else:
        s["game"] = s.away_team + " @ " + s.home_team if " @ " in str(snap.game.iloc[0]) else s.away_team + "@" + s.home_team
        x = snap.merge(s[["game", "kick"]], on="game")
    snap_t = pd.to_datetime(x.snap_at, utc=True)
    hrs = (x.kick - snap_t).dt.total_seconds() / 3600
    games = x.groupby(x.kick.astype(str)).ngroups
    return dict(median_hours_before=r(hrs.median(), 1), max_hours_before=r(hrs.max(), 1), groups=games,
                copies=sorted(x.snap_at.unique().tolist()))


# ------------------------------------------------------------------ main
def latest_final_week(sched, season):
    s = sched[(sched.season == season) & (sched.game_type == "REG")]
    done = s.groupby("week").home_score.apply(lambda v: v.notna().all())
    done = done[done]
    return int(done.index.max()) if len(done) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int)
    ap.add_argument("--season", type=int)
    ap.add_argument("--partial", action="store_true", help="grade the finished games of an unfinished week (testing)")
    a = ap.parse_args()
    sched_all = pd.read_csv(io.BytesIO(get(f"{REL}/schedules/games.csv")))
    season = a.season or int(sched_all[sched_all.home_score.notna()].season.max())
    week = a.week or latest_final_week(sched_all, season)
    sched = sched_all[(sched_all.season == season) & (sched_all.week == week)]
    if not sched.home_score.notna().all():
        if not a.partial:
            sys.exit(f"week {week} is not final yet")
        sched = sched[sched.home_score.notna()]
    sd = f"{R}/snapshots/{season}_wk{week:02d}"
    gd = f"{R}/graded/{season}_wk{week:02d}"
    os.makedirs(gd, exist_ok=True)
    print(f"grading {season} week {week} from {sd}")
    status, cov = {}, {}

    if os.path.exists(f"{sd}/games.csv"):
        gs = pd.read_csv(f"{sd}/games.csv")
        g = grade_games(gs, sched)
        g.to_csv(f"{gd}/games.csv", index=False)
        status["games"] = f"graded {len(g)} of {len(sched)} games"
        cov["games"] = coverage(gs, sched)
    else:
        status["games"] = "no pre-kickoff copy saved"

    pending = []
    if os.path.exists(f"{sd}/fantasy.csv"):
        fs = pd.read_csv(f"{sd}/fantasy.csv", dtype={"id": str})
        teams = pd.concat([sched.away_team, sched.home_team]).unique()
        try:
            f, pending = grade_fantasy(fs, load_stats(season), week, teams)
        except Exception as e:
            f, pending = None, [f"stats download failed: {e!r}"]
        if f is not None:
            f.to_csv(f"{gd}/fantasy.csv", index=False)
            status["fantasy"] = f"graded {int((f.half >= FMIN).sum())} players projected {FMIN:g}+ half-PPR"
        else:
            status["fantasy"] = "waiting for nflverse box scores: " + ", ".join(pending)
        cov["fantasy"] = coverage(fs, sched, team_col="team")
    else:
        status["fantasy"] = "no pre-kickoff copy saved"

    if os.path.exists(f"{sd}/kalshi.csv"):
        ks = pd.read_csv(f"{sd}/kalshi.csv")
        k = grade_kalshi(ks)
        k.to_csv(f"{gd}/kalshi.csv", index=False)
        status["kalshi"] = f"graded {len(k)} settled markets of {len(ks)} saved"
        cov["kalshi"] = coverage(ks, sched)
    else:
        status["kalshi"] = "no pre-kickoff copy saved"
    json.dump(dict(status=status, coverage=cov, graded_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")),
              open(f"{gd}/status.json", "w"), indent=1, default=str)

    build_report(season)
    print(json.dumps(status, indent=1))
    sys.exit(3 if pending else 0)


def load_graded(season, name, dtype=None):
    parts = []
    for d in sorted(glob.glob(f"{R}/graded/{season}_wk*/{name}.csv")):
        x = pd.read_csv(d, dtype=dtype)
        x["wk"] = int(d.split("_wk")[1][:2])
        parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else None


def backtest_refs(season, first_live):
    out = {}
    try:
        b = pd.read_csv(f"{PF}/nfl_engine/v2/output/backtest.csv")
        row = b[b.period.str.contains(str(season))].iloc[-1]
        out["games"] = dict(period=row.period, games=int(row.games), accuracy=r(row.accuracy), logloss=r(row.log_loss),
                            brier=r(row.brier), vegas_accuracy=r(row.vegas_fav_accuracy), margin_mae=r(row.margin_mae, 1),
                            ats=r(row.ats_accuracy))
    except Exception as e:
        print("engine backtest ref:", e)
    try:
        b = pd.read_csv(f"{PF}/fantasy/output/fantasy_backtest.csv")
        b = b[(b.season == season) & (b.week < (first_live or 99)) & (b.half >= FMIN)]
        out["fantasy"] = dict(period=f"{season} wks {int(b.week.min())}-{int(b.week.max())}", players=len(b),
                              mae_half=r((b.half - b.actual).abs().mean(), 2), naive_mae=r((b.naive_season - b.actual).abs().mean(), 2))
    except Exception as e:
        print("fantasy backtest ref:", e)
    out["kalshi"] = dict(period=f"{season} wks 3-8 player props", markets=20140, logloss_kalshi=0.4710, logloss_model=0.4884,
                         logloss_blend=0.4699, bets=257, roi=0.118, roi_lo=-0.036, roi_hi=0.28)
    return out


def build_report(season):
    G = load_graded(season, "games")
    F = load_graded(season, "fantasy", {"id": str})
    K = load_graded(season, "kalshi")
    weeks = sorted({int(d.split("_wk")[1][:2]) for d in glob.glob(f"{R}/graded/{season}_wk*")})
    rep = dict(season=season, generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), weeks={}, fmin=FMIN)
    for w in weeks:
        st = json.load(open(f"{R}/graded/{season}_wk{w:02d}/status.json"))
        g = G[G.wk == w] if G is not None else None
        f = F[F.wk == w] if F is not None else None
        k = K[K.wk == w] if K is not None else None
        rep["weeks"][w] = dict(status=st["status"], coverage=st.get("coverage", {}), graded_at=st.get("graded_at"),
                               games=sum_games(g), game_rows=game_rows(g) if g is not None and len(g) else [],
                               fantasy=sum_fantasy(f), fantasy_rows=fantasy_rows(f) if f is not None and len(f) else None,
                               kalshi=sum_kalshi(k), kalshi_rows=kalshi_rows(k) if k is not None and len(k) else [])
    rep["season_to_date"] = dict(weeks=weeks, games=sum_games(G), fantasy=sum_fantasy(F), kalshi=sum_kalshi(K))
    rep["trend"] = [dict(week=w, acc=(rep["weeks"][w]["games"] or {}).get("accuracy"),
                         vegas=(rep["weeks"][w]["games"] or {}).get("vegas_accuracy"),
                         fmae=(rep["weeks"][w]["fantasy"] or {}).get("mae_half"),
                         fnaive=(rep["weeks"][w]["fantasy"] or {}).get("naive_mae"),
                         kb=(rep["weeks"][w]["kalshi"] or {}).get("brier_shown"),
                         kk=(rep["weeks"][w]["kalshi"] or {}).get("brier_kalshi")) for w in weeks]
    rep["backtest"] = backtest_refs(season, weeks[0] if weeks else None)
    rep["notes"] = {}
    if os.path.exists(f"{R}/notes.json"):
        rep["notes"] = json.load(open(f"{R}/notes.json"))
    os.makedirs(f"{R}/output", exist_ok=True)
    json.dump(rep, open(f"{R}/output/report.json", "w"), separators=(",", ":"), default=str)
    if weeks:
        write_md(rep, weeks[-1])
    print(f"wrote {R}/output/report.json ({len(weeks)} weeks)")


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def write_md(rep, w):
    W, S = rep["weeks"][w], rep["season_to_date"]
    L = [f"# Week {w} accuracy ({rep['season']})", ""]
    g = W["games"]
    if g:
        L.append(f"- Games: {g['correct']}-{g['games'] - g['correct']} picking winners ({pct(g['accuracy'])}), Vegas favorites {g['vegas_correct']}-{g['vegas_games'] - g['vegas_correct']}. "
                 f"Against the spread {g['ats_w']}-{g['ats_n'] - g['ats_w']}, totals {g['ou_w']}-{g['ou_n'] - g['ou_w']}. Score miss: margin {g['margin_mae']} pts, total {g['total_mae']} pts.")
    f = W["fantasy"]
    if f:
        L.append(f"- Fantasy: {f['players']} players projected {rep['fmin']:g}+ half-PPR, average miss {f['mae_half']} pts "
                 f"(season-average guess {f['naive_mae']}), bias {f['bias_half']:+} pts.")
    else:
        L.append(f"- Fantasy: {W['status'].get('fantasy')}")
    k = W["kalshi"]
    if k:
        v, t = k["value_bets"], k["top_picks"]
        L.append(f"- Betting: {k['markets']} settled markets. Brier model {k['brier_shown']} vs Kalshi {k['brier_kalshi']} (lower is better). "
                 f"Value bets {v['won']}-{v['n'] - v['won']} ({'%+.2f' % v['profit']} $ on {v['cost']} staked); top picks {t['won']}-{t['n'] - t['won']} ({'%+.2f' % t['profit']} $).")
    sg = S["games"]
    if sg:
        L += ["", f"Season to date (live weeks {', '.join(map(str, S['weeks']))}): winners {pct(sg['accuracy'])} vs Vegas {pct(sg['vegas_accuracy'])}."]
    open(f"{R}/output/{rep['season']}_wk{w:02d}.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
