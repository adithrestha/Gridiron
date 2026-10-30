"""Grade the CAR @ GB snapshot against what happened. Run: python3 grade.py
Reads the snapshot files in this folder; pulls Kalshi settlements and nflverse results."""
import glob, io, json, os, sys, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "https://github.com/nflverse/nflverse-data/releases/download"
K = "https://api.elections.kalshi.com/trade-api/v2/markets/"

def get(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            if i == 4: raise
            time.sleep(2 ** i)

snap = lambda p: sorted(glob.glob(os.path.join(HERE, p)))[0]
game = pd.read_csv(snap("game_*.csv")).iloc[0]
fan = pd.read_csv(snap("fantasy_*.csv"))
kal = json.load(open(snap("kalshi_2*.json")))
out = {}

# Game result
g = pd.read_csv(io.BytesIO(get(f"{REL}/schedules/games.csv")))
r = g[(g.season == 2026) & (g.week == 8) & (g.home_team == "GB") & (g.away_team == "CAR")].iloc[0]
if pd.isna(r.home_score):
    sys.exit("No final score in nflverse games.csv yet")
hs, as_ = int(r.home_score), int(r.away_score)
margin, total = hs - as_, hs + as_
out["game"] = dict(final=f"CAR {as_}, GB {hs}", pick=game.pick,
    pick_right=(game.pick == "GB") == (margin > 0), win_prob_gb=round(game.home_win_prob, 3),
    pred_score=f"CAR {game.pred_away_pts:.1f}, GB {game.pred_home_pts:.1f}",
    margin_err=round(game.pred_margin - margin, 1), total_err=round(game.pred_total - total, 1),
    gb_pts_err=round(game.pred_home_pts - hs, 1), car_pts_err=round(game.pred_away_pts - as_, 1),
    vegas_line_gb=-game.vegas_home_spread,
    ats_pick=("GB" if game.pred_margin > game.vegas_home_spread else "CAR"),
    ats_right=(game.pred_margin > game.vegas_home_spread) == (margin > game.vegas_home_spread)
              if margin != game.vegas_home_spread else None,
    brier=round((game.home_win_prob - (margin > 0)) ** 2, 3))

# Kalshi settlements
rows = []
for m in kal["rows"]:
    res = json.loads(get(K + m["ticker"]))["market"].get("result", "")
    time.sleep(0.15)
    if res not in ("yes", "no"):
        continue
    y = res == "yes"
    p_yes = m["p_model"]  # p_model is always P(YES)
    rows.append(dict(kind=m["kind"], label=m["label"], side=m["side"], buy_at=m["buy_at"],
        p_model=p_yes, mid=m["mid"], flagged=m["value"], result=res,
        lean_right=(m["side"] == "YES") == y,
        pnl_per_contract=round((1 - m["buy_at"] - m["fee"]) if (m["side"] == "YES") == y else (-m["buy_at"] - m["fee"]), 3),
        brier_model=(p_yes - y) ** 2, brier_kalshi=(m["mid"] - y) ** 2))
kr = pd.DataFrame(rows)
kr.to_csv(os.path.join(HERE, "graded_kalshi.csv"), index=False)
if len(kr):
    out["kalshi"] = dict(settled=len(kr), unsettled=len(kal["rows"]) - len(kr),
        flagged=int(kr.flagged.sum()),
        by_kind={k: dict(n=len(d), lean_right=int(d.lean_right.sum()),
                         brier_model=round(d.brier_model.mean(), 4), brier_kalshi=round(d.brier_kalshi.mean(), 4),
                         pnl_if_1_each=round(d.pnl_per_contract.sum(), 2))
                 for k, d in kr.groupby("kind")})

# Fantasy (half-PPR; nflverse fantasy_points is standard scoring, no kickers/DST)
try:
    s = pd.read_csv(io.BytesIO(get(f"{REL}/stats_player/stats_player_week_2026.csv")), low_memory=False)
    s = s[(s.week == 8) & (s.team.isin(["CAR", "GB"]))]
except Exception as e:
    s = pd.DataFrame()
if len(s):
    s = s.assign(act_half=s.fantasy_points + 0.5 * s.receptions)
    keep = ["player_id", "act_half", "fantasy_points_ppr", "passing_yards", "passing_tds", "rushing_yards",
            "rushing_tds", "receptions", "receiving_yards", "receiving_tds"]
    f = fan[~fan.position.isin(["K", "DST"])].merge(s[keep], left_on="id", right_on="player_id", how="left")
    f["act_half"] = f.act_half.fillna(0)
    f["err_half"] = (f.half - f.act_half).round(2)
    cols = ["name", "position", "team", "half", "act_half", "err_half", "ppr", "fantasy_points_ppr",
            "pyd", "passing_yards", "ryd", "rushing_yards", "rec", "receptions", "wyd", "receiving_yards"]
    f[cols].sort_values("half", ascending=False).to_csv(os.path.join(HERE, "graded_fantasy.csv"), index=False)
    top = f[f.half >= 5]
    out["fantasy"] = dict(players=len(top), mae_half=round(top.err_half.abs().mean(), 2),
        within_3=int((top.err_half.abs() <= 3).sum()),
        rows=top.sort_values("half", ascending=False)[["name", "half", "act_half", "err_half"]].to_dict("records"))
else:
    out["fantasy"] = "nflverse week 8 player stats not posted yet"

json.dump(out, open(os.path.join(HERE, "grades.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
