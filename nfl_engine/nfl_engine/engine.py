"""
NFL Prediction Engine — starter version
=======================================
Free data (nflverse, refreshed daily on GitHub releases):
  * schedules/games.csv     results 1999-now, rest days, home/neutral site, div games,
                            roof, surface, temp, wind, Vegas spread/total/moneylines
  * stats_team_week_YYYY    weekly team box stats incl. passing/rushing EPA, sacks, turnovers
  * injuries_YYYY           official weekly injury reports (Out / Doubtful / Questionable)

Features (all computed only from info available BEFORE kickoff):
  Elo ratings (MOV-adjusted, season regression), exponentially weighted scoring form,
  offensive & defensive EPA/play, turnover/sack margins, rest days, bye / short week,
  home field / neutral site, division game, dome / weather, position-weighted injuries,
  QB-out flag.

Models (stacked ensemble, each makes predictions, a meta-model blends them):
  Win probability: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting,
                   Histogram GBM (LightGBM-style), Neural Net (MLP), SVM, k-NN
                   (+ XGBoost / LightGBM automatically if installed)
  Margin & total:  Ridge, Random Forest, Extra Trees, Gradient Boosting, Histogram GBM, MLP

The model does NOT use the Vegas line as an input, so it can be compared against it.

Usage:  python engine.py            (download data, backtest, predict rest of season)
        python engine.py --no-download
"""
import argparse, os, sys, warnings, urllib.request
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, mean_absolute_error

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
SEASON, START_WEEK = 2026, 8
FIRST_TRAIN = 2012
REL = "https://github.com/nflverse/nflverse-data/releases/download"
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# ----------------------------------------------------------------------------- data
def download(years):
    os.makedirs(DATA, exist_ok=True)
    files = {"games.csv": f"{REL}/schedules/games.csv"}
    for y in years:
        files[f"team_{y}.csv"] = f"{REL}/stats_team/stats_team_week_{y}.csv"
        files[f"injuries_{y}.csv"] = f"{REL}/injuries/injuries_{y}.csv"
    for name, url in files.items():
        path = os.path.join(DATA, name)
        current = str(SEASON) in name or name == "games.csv"
        if os.path.exists(path) and not current:
            continue
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"  could not download {name}: {e}")

def load(years):
    g = pd.read_csv(os.path.join(DATA, "games.csv"), low_memory=False)
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIX)
    g = g[g.season >= 1999].copy()
    teams, inj = [], []
    for y in years:
        p = os.path.join(DATA, f"team_{y}.csv")
        if os.path.exists(p): teams.append(pd.read_csv(p, low_memory=False))
        p = os.path.join(DATA, f"injuries_{y}.csv")
        if os.path.exists(p): inj.append(pd.read_csv(p, low_memory=False))
    return g, pd.concat(teams, ignore_index=True), pd.concat(inj, ignore_index=True)

# ----------------------------------------------------------------------------- Elo
def elo_ratings(g, k=20, hfa=48, revert=1 / 3):
    """538-style Elo with margin-of-victory multiplier. Returns pre-game Elo per game_id."""
    g = g.sort_values(["season", "week", "gameday"])
    elo, last_season, pre = {}, None, {}
    for r in g.itertuples():
        if r.season != last_season:                       # regress 1/3 to the mean each offseason
            elo = {t: 1505 + (e - 1505) * (1 - revert) for t, e in elo.items()}
            last_season = r.season
        h, a = elo.get(r.home_team, 1505), elo.get(r.away_team, 1505)
        pre[r.game_id] = (h, a)
        if pd.isna(r.home_score):
            continue
        adv = 0 if r.location == "Neutral" else hfa
        exp_h = 1 / (1 + 10 ** (-(h + adv - a) / 400))
        res = r.home_score - r.away_score
        s = 1.0 if res > 0 else 0.0 if res < 0 else 0.5
        mult = np.log(abs(res) + 1) * 2.2 / ((h + adv - a if res > 0 else a - h - adv) * 0.001 + 2.2)
        delta = k * mult * (s - exp_h)
        elo[r.home_team], elo[r.away_team] = h + delta, a - delta
    return pd.DataFrame([(gid, h, a) for gid, (h, a) in pre.items()],
                        columns=["game_id", "home_elo", "away_elo"]), elo

# ----------------------------------------------------------------------------- team form
def team_form(g, ts, halflife=6):
    """Exponentially weighted form per team entering each game (carries across seasons)."""
    played = g[g.home_score.notna()]
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = played[["game_id", "season", "week", "gameday", f"{side}_team", f"{side}_score", f"{opp}_score"]]
        d.columns = ["game_id", "season", "week", "gameday", "team", "pf", "pa"]
        rows.append(d)
    tg = pd.concat(rows)
    ts = ts.copy()
    ts["plays"] = ts.attempts + ts.carries + ts.sacks_suffered
    ts["off_epa"] = (ts.passing_epa.fillna(0) + ts.rushing_epa.fillna(0)) / ts.plays.clip(lower=1)
    ts["giveaways"] = ts.passing_interceptions + ts.rushing_fumbles_lost + ts.sack_fumbles_lost
    ts["pass_rate"] = ts.attempts / ts.plays.clip(lower=1)
    opp = ts[["game_id", "team", "off_epa", "giveaways", "sacks_suffered"]].rename(
        columns={"team": "opponent_team", "off_epa": "def_epa", "giveaways": "takeaways",
                 "sacks_suffered": "def_sacks_made"})
    ts = ts.merge(opp, on=["game_id", "opponent_team"], how="left")
    tg = tg.merge(ts[["game_id", "team", "off_epa", "def_epa", "giveaways", "takeaways",
                      "sacks_suffered", "def_sacks_made", "pass_rate"]], on=["game_id", "team"], how="left")
    tg = tg.sort_values(["team", "gameday"])
    stats = ["pf", "pa", "off_epa", "def_epa", "giveaways", "takeaways", "sacks_suffered", "def_sacks_made", "pass_rate"]
    out = []
    for team, d in tg.groupby("team"):
        ew = d[stats].ewm(halflife=halflife, min_periods=1).mean()
        d = d[["game_id", "team", "gameday"]].copy()
        d[[f"f_{s}" for s in stats]] = ew.shift(1).values      # shift: only games BEFORE this one
        last = ew.iloc[-1].values                               # form entering next unplayed game
        out.append(d)
        out.append(pd.DataFrame([[None, team, "9999"] + list(last)], columns=d.columns))
    form = pd.concat(out, ignore_index=True)
    form["f_ptdiff"] = form.f_pf - form.f_pa
    form["f_epadiff"] = form.f_off_epa - form.f_def_epa
    form["f_tomargin"] = form.f_takeaways - form.f_giveaways
    return form

# ----------------------------------------------------------------------------- injuries
POS_W = {"QB": 6.0, "T": 1.5, "G": 1.0, "C": 1.0, "WR": 1.2, "TE": 0.8, "RB": 0.8, "CB": 1.2,
         "S": 0.8, "DE": 1.2, "DT": 0.9, "LB": 0.8, "OLB": 1.0, "ILB": 0.8, "MLB": 0.8, "NT": 0.8,
         "FS": 0.8, "SS": 0.8, "DB": 0.8, "K": 0.3, "P": 0.1, "LS": 0.1, "FB": 0.3, "OL": 1.0, "DL": 1.0}
STATUS_W = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}

def injury_load(inj):
    inj = inj[inj.game_type.isin(["REG", "WC", "DIV", "CON", "SB"]) | inj.game_type.isna()].copy()
    inj["team"] = inj.team.replace(TEAM_FIX)
    inj["w"] = inj.position.map(POS_W).fillna(0.6) * inj.report_status.map(STATUS_W).fillna(0)
    # Current week before the final (Friday) report: no game designations yet, so estimate
    # from practice participation (DNP ~ likely out, Limited ~ questionable).
    has_final = inj.groupby(["season", "week", "team"]).report_status.transform("count") > 0
    prac = inj.practice_status.fillna("")
    est = np.where(prac.str.startswith("Did Not"), 0.6, np.where(prac.str.startswith("Limited"), 0.2, 0.0))
    inj.loc[~has_final, "w"] = inj.loc[~has_final, "position"].map(POS_W).fillna(0.6) * est[~has_final.values]
    inj["qb_out"] = ((inj.position == "QB") & (inj.report_status.isin(["Out", "Doubtful"]) |
                     (~has_final & prac.str.startswith("Did Not")))).astype(int)
    agg = inj.groupby(["season", "week", "team"]).agg(inj_load=("w", "sum"), qb_out=("qb_out", "max")).reset_index()
    return agg

# ----------------------------------------------------------------------------- features
FEATURES = ["elo_diff", "home_elo", "away_elo", "neutral", "rest_diff", "home_bye", "away_bye",
            "home_short", "away_short", "div_game", "dome", "temp", "wind", "grass", "primetime",
            "ptdiff_diff", "epadiff_diff", "off_vs_def_home", "off_vs_def_away", "tomargin_diff",
            "sack_diff", "home_pf", "away_pf", "home_pa", "away_pa", "inj_diff", "home_qb_out",
            "away_qb_out", "week"]

def build(g, ts, inj):
    g = g.copy()
    elo_df, final_elo = elo_ratings(g)
    form = team_form(g, ts)
    df = g.merge(elo_df, on="game_id", how="left")
    # team form: for played games use game-specific row; for future games use latest form
    played_form = form[form.game_id.notna()]
    latest = form[form.game_id.isna()].drop(columns=["game_id", "gameday"])
    for side in ("home", "away"):
        pf = played_form.drop(columns="gameday").add_prefix(f"{side}_").rename(
            columns={f"{side}_game_id": "game_id", f"{side}_team": f"{side}_team"})
        df = df.merge(pf, on=["game_id", f"{side}_team"], how="left")
        lt = latest.add_prefix(f"{side}_").rename(columns={f"{side}_team": f"{side}_team"})
        fut = df[f"{side}_f_pf"].isna()
        filled = df.loc[fut, [f"{side}_team"]].merge(lt, on=f"{side}_team", how="left")
        for c in lt.columns:
            if c != f"{side}_team":
                df.loc[fut, c] = filled[c].values
    # future games: Elo = current rating (pre-game Elo frozen at latest value)
    unplayed = df.home_score.isna()
    df.loc[unplayed, "home_elo"] = df.loc[unplayed, "home_team"].map(final_elo)
    df.loc[unplayed, "away_elo"] = df.loc[unplayed, "away_team"].map(final_elo)
    # injuries: weekly report; future weeks without a report -> team's season-average load
    il = injury_load(inj)
    for side in ("home", "away"):
        x = il.rename(columns={"team": f"{side}_team", "inj_load": f"{side}_inj", "qb_out": f"{side}_qb_out"})
        df = df.merge(x, on=["season", "week", f"{side}_team"], how="left")
        avg = il.groupby(["season", "team"]).inj_load.mean()
        miss = df[f"{side}_inj"].isna()
        key = list(zip(df.loc[miss, "season"], df.loc[miss, f"{side}_team"]))
        df.loc[miss, f"{side}_inj"] = [avg.get(k, il.inj_load.mean()) for k in key]
        df[f"{side}_qb_out"] = df[f"{side}_qb_out"].fillna(0)
    # situational
    df["neutral"] = (df.location == "Neutral").astype(int)
    df["elo_diff"] = df.home_elo - df.away_elo + np.where(df.neutral == 1, 0, 48)
    df["rest_diff"] = df.home_rest - df.away_rest
    for s in ("home", "away"):
        df[f"{s}_bye"] = (df[f"{s}_rest"] >= 13).astype(int)
        df[f"{s}_short"] = (df[f"{s}_rest"] <= 5).astype(int)
    df["dome"] = df.roof.isin(["dome", "closed"]).astype(int)
    df["temp"] = np.where(df.dome == 1, 70, df.temp.fillna(55))
    df["wind"] = np.where(df.dome == 1, 0, df.wind.fillna(8))
    df["grass"] = df.surface.fillna("").str.contains("grass").astype(int)
    hr = pd.to_numeric(df.gametime.fillna("13:00").str[:2], errors="coerce").fillna(13)
    df["primetime"] = ((hr >= 19) | (df.weekday.isin(["Monday", "Thursday"]))).astype(int)
    df["ptdiff_diff"] = df.home_f_ptdiff - df.away_f_ptdiff
    df["epadiff_diff"] = df.home_f_epadiff - df.away_f_epadiff
    df["off_vs_def_home"] = df.home_f_off_epa + df.away_f_def_epa
    df["off_vs_def_away"] = df.away_f_off_epa + df.home_f_def_epa
    df["tomargin_diff"] = df.home_f_tomargin - df.away_f_tomargin
    df["sack_diff"] = (df.home_f_def_sacks_made - df.home_f_sacks_suffered) - (df.away_f_def_sacks_made - df.away_f_sacks_suffered)
    df["home_pf"], df["away_pf"], df["home_pa"], df["away_pa"] = df.home_f_pf, df.away_f_pf, df.home_f_pa, df.away_f_pa
    df["inj_diff"] = df.home_inj - df.away_inj
    df["margin"] = df.home_score - df.away_score
    df["points"] = df.home_score + df.away_score
    return df

# ----------------------------------------------------------------------------- models
def classifiers():
    sc = lambda m: make_pipeline(StandardScaler(), m)
    base = [
        ("logreg", sc(LogisticRegression(C=0.3, max_iter=2000))),
        ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=20, n_jobs=-1, random_state=1)),
        ("et", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=20, n_jobs=-1, random_state=1)),
        ("gbm", GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=1)),
        ("histgbm", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=4, l2_regularization=1.0, random_state=1)),
        ("mlp", sc(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=1500, early_stopping=True, random_state=1))),
        ("svm", sc(SVC(C=0.5, probability=True, random_state=1))),
        ("knn", sc(KNeighborsClassifier(n_neighbors=75))),
    ]
    try:
        from xgboost import XGBClassifier
        base.append(("xgb", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)))
    except ImportError: pass
    try:
        from lightgbm import LGBMClassifier
        base.append(("lgbm", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1)))
    except ImportError: pass
    return StackingClassifier(base, final_estimator=LogisticRegression(C=1.0, max_iter=2000), cv=5, n_jobs=1)

def regressors():
    sc = lambda m: make_pipeline(StandardScaler(), m)
    base = [
        ("ridge", sc(Ridge(alpha=10))),
        ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=1)),
        ("et", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=1)),
        ("gbm", GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=1)),
        ("histgbm", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=4, l2_regularization=1.0, random_state=1)),
        ("mlp", sc(MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=1500, early_stopping=True, random_state=1))),
    ]
    return StackingRegressor(base, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1)

def fit(train):
    X, clf = train[FEATURES], classifiers()
    win = train[train.margin != 0]
    clf.fit(win[FEATURES], (win.margin > 0).astype(int))
    reg_m, reg_t = regressors(), regressors()
    reg_m.fit(X, train.margin); reg_t.fit(X, train.points)
    return clf, reg_m, reg_t

def evaluate(name, clf, reg_m, test):
    t = test[test.margin != 0]
    p = clf.predict_proba(t[FEATURES])[:, 1]
    y = (t.margin > 0).astype(int)
    vegas = t.spread_line.notna()
    m = reg_m.predict(t[FEATURES])
    ats = t[vegas & (t.margin != t.spread_line)]
    ats_pick = reg_m.predict(ats[FEATURES]) > ats.spread_line
    ats_hit = (ats_pick == (ats.margin > ats.spread_line)).mean()
    res = dict(period=name, games=len(t), accuracy=accuracy_score(y, p > 0.5), log_loss=log_loss(y, p),
               brier=brier_score_loss(y, p), vegas_fav_accuracy=((t[vegas].spread_line > 0) == (t[vegas].margin > 0)).mean(),
               margin_mae=mean_absolute_error(t.margin, m), ats_accuracy=ats_hit)
    # per-model accuracy (base learners)
    for nm, est in clf.named_estimators_.items():
        res[f"acc_{nm}"] = accuracy_score(y, est.predict_proba(t[FEATURES])[:, 1] > 0.5)
    return res

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-download", action="store_true"); a = ap.parse_args()
    years = list(range(FIRST_TRAIN, SEASON + 1))
    if not a.no_download:
        print("Downloading data..."); download(years)
    g, ts, inj = load(years)
    df = build(g, ts, inj)
    df = df[df.season >= FIRST_TRAIN]
    played = df[df.margin.notna() & df[FEATURES].notna().all(axis=1)]
    os.makedirs(OUT, exist_ok=True)

    print("Backtesting (train on prior seasons, test on later ones)...")
    results = []
    for test_name, tr, te in [("2024 season", played.season <= 2023, played.season == 2024),
                              ("2025 season", played.season <= 2024, played.season == 2025),
                              (f"{SEASON} wks 1-{START_WEEK-1}", played.season <= SEASON - 1,
                               (played.season == SEASON) & (played.week < START_WEEK))]:
        clf, rm, _ = fit(played[tr])
        results.append(evaluate(test_name, clf, rm, played[te]))
        print(f"  {test_name}: acc {results[-1]['accuracy']:.3f} vs Vegas fav {results[-1]['vegas_fav_accuracy']:.3f}")
    pd.DataFrame(results).to_csv(os.path.join(OUT, "backtest.csv"), index=False)

    print("Training final model on all games through last week...")
    clf, rm, rt = fit(played)
    fut = df[(df.season == SEASON) & (df.week >= START_WEEK) & df.margin.isna() & (df.game_type == "REG")].copy()
    fut[FEATURES] = fut[FEATURES].fillna(played[FEATURES].median())
    fut["home_win_prob"] = clf.predict_proba(fut[FEATURES])[:, 1]
    fut["pred_margin"] = rm.predict(fut[FEATURES])
    fut["pred_total"] = rt.predict(fut[FEATURES])
    fut["pred_home_pts"] = (fut.pred_total + fut.pred_margin) / 2
    fut["pred_away_pts"] = (fut.pred_total - fut.pred_margin) / 2
    fut["pick"] = np.where(fut.home_win_prob >= 0.5, fut.home_team, fut.away_team)
    fut["confidence"] = np.maximum(fut.home_win_prob, 1 - fut.home_win_prob)
    fut["vegas_home_spread"] = fut.spread_line
    fut["edge_vs_vegas"] = fut.pred_margin - fut.spread_line
    cols = ["season", "week", "gameday", "gametime", "away_team", "home_team", "pick", "confidence",
            "home_win_prob", "pred_away_pts", "pred_home_pts", "pred_margin", "pred_total",
            "vegas_home_spread", "edge_vs_vegas", "home_rest", "away_rest", "home_inj", "away_inj",
            "home_qb_out", "away_qb_out", "home_elo", "away_elo"]
    fut[cols].sort_values(["week", "gameday", "gametime"]).to_csv(os.path.join(OUT, "predictions.csv"), index=False)

    # projected final regular-season records = current wins + sum of win probabilities
    cur = df[(df.season == SEASON) & df.margin.notna() & (df.game_type == "REG")]
    recs = {}
    for r in cur.itertuples():
        for t, w in ((r.home_team, r.margin > 0), (r.away_team, r.margin < 0)):
            recs.setdefault(t, [0, 0, 0]); recs[t][0 if w else (2 if r.margin == 0 else 1)] += 1
    proj = []
    for t in sorted(set(fut.home_team) | set(fut.away_team) | set(recs)):
        w, l, ti = recs.get(t, [0, 0, 0])
        exp = fut.loc[fut.home_team == t, "home_win_prob"].sum() + (1 - fut.loc[fut.away_team == t, "home_win_prob"]).sum()
        n = ((fut.home_team == t) | (fut.away_team == t)).sum()
        proj.append(dict(team=t, wins=w, losses=l, ties=ti, remaining=n, proj_wins=w + exp + ti * 0.5,
                         proj_losses=l + n - exp + ti * 0.5))
    pd.DataFrame(proj).sort_values("proj_wins", ascending=False).to_csv(os.path.join(OUT, "projected_standings.csv"), index=False)
    print(f"Done. Wrote {len(fut)} game predictions to {OUT}/")

if __name__ == "__main__":
    main()
