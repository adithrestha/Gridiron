"""Build the Kalshi Edge Board page from kalshi_edges.py output.
Usage: python build_kalshi_page.py EDGES_JSON OUT_HTML"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def implied_line(ladder):
    """Line where the market's P(over) crosses 50%, interpolated from a strike ladder of (strike, mid)."""
    pts = sorted(ladder)
    for (s1, p1), (s2, p2) in zip(pts, pts[1:]):
        if p1 >= 0.5 >= p2 and p1 != p2:
            return s1 + (p1 - 0.5) / (p1 - p2) * (s2 - s1)
    return None

def main(edges_path, out_path):
    e = json.load(open(edges_path))
    d = pd.DataFrame(e["rows"])
    pred = pd.read_csv("/mnt/project-files/nfl_engine/v2/output/predictions.csv")
    games = []
    for game, g in d.groupby("game", sort=False):
        away, home = game.split(" @ ")
        p = pred[(pred.away_team == away) & (pred.home_team == home)].iloc[0]
        win = g[g.kind == "Winner"]
        # Kalshi home win chance = mid of the home "wins" market; fall back to 1 - away mid
        k_home = None
        for _, r in win.iterrows():
            full = r.ticker.split("-")[-1]
            if full in (home, {"JAX": "JAC", "LA": "LAR"}.get(home)): k_home = r.mid
        if k_home is None and len(win): k_home = 1 - win.iloc[0].mid
        tot = g[g.kind == "Total"]
        k_total = implied_line([(float(l.split()[-1]), m) for l, m in zip(tot.label, tot.mid)]) if len(tot) else None
        sp = g[g.kind == "Spread"]
        # home margin ladder: "HOME wins by over X" -> P(margin > X); "AWAY wins by over X" -> P(margin < -X)
        lad = []
        for _, r in sp.iterrows():
            x = float(r.label.split("over ")[1].split()[0])
            team = r.ticker.split("-")[-1].rstrip("0123456789")
            is_home = team in (home, {"JAX": "JAC", "LA": "LAR"}.get(home))
            lad.append((x, r.mid) if is_home else (-x, 1 - r.mid))
        k_margin = implied_line(lad) if lad else None
        # Kalshi's main lines as its app lists them: the rung trading closest to 50/50
        k_spread_txt = k_total_txt = None
        if len(sp):
            r = sp.iloc[(sp.mid - 0.5).abs().argmin()]
            team = r.ticker.split("-")[-1].rstrip("0123456789")
            k_spread_txt = f"{team} by {float(r.label.split('over ')[1].split()[0]) + 0.5:g}+ · {round(100 * r.mid)}%"
        if len(tot):
            r = tot.iloc[(tot.mid - 0.5).abs().argmin()]
            k_total_txt = f"Over {r.label.split()[-1]} · {round(100 * r.mid)}%"
        games.append(dict(game=game, week=int(p.week), gameday=p.gameday, gametime=p.gametime, away=away, home=home,
                          m_home=round(float(p.home_win_prob), 3), k_home=None if k_home is None else round(float(k_home), 3),
                          m_margin=round(float(p.pred_margin), 1), k_margin=None if k_margin is None else round(k_margin, 1),
                          m_total=round(float(p.pred_total), 1), k_total=None if k_total is None else round(k_total, 1),
                          k_spread_txt=k_spread_txt, k_total_txt=k_total_txt,
                          vegas_spread=None if pd.isna(p.vegas_home_spread) else float(p.vegas_home_spread),
                          home_qb=p.home_qb_name, away_qb=p.away_qb_name,
                          n_markets=int(len(g)), n_value=int(g.value.sum())))
    games.sort(key=lambda g: (g["gameday"], g["gametime"]))
    # keep page light: drop near-certain ladder rungs nobody would trade
    keep = d[(d.buy_at >= 0.03) & (d.buy_at <= 0.97)]
    cols = ["ticker", "kind", "week", "gameday", "gametime", "game", "label", "p_model", "yes_bid", "yes_ask", "no_ask",
            "mid", "book_spread", "volume", "side", "buy_at", "fee", "p_side", "p_adj", "raw_edge", "edge", "kelly", "value",
            "player", "position", "stat", "proj", "p_out", "status", "injury", "news", "news_url", "last", "why_not", "game_gap"]
    keep = keep[[c for c in cols if c in keep.columns]]
    rows = json.loads(keep.to_json(orient="records"))
    data = dict(fetched_at=e["fetched_at"], engine_run=e["engine_run"], fantasy_run=e["fantasy_run"],
                settings=e["settings"], unmatched=e["unmatched_player_markets"],
                news_checked_at=e.get("news_checked_at"), excluded=e.get("excluded_players", []), games=games, rows=rows)
    # slimmer copy for the Gridiron page's Betting tab (fantasy/build_fantasy_page.py picks it up)
    slim_cols = ["kind", "week", "game", "label", "mid", "book_spread", "volume", "side", "buy_at", "p_side", "p_adj",
                 "edge", "kelly", "value", "player", "stat", "proj", "status", "injury", "news", "news_url",
                 "yes_bid", "yes_ask", "no_ask", "last", "why_not", "ticker"]
    slim = keep[(keep.buy_at >= 0.05) & (keep.buy_at <= 0.95)][[c for c in slim_cols if c in keep.columns]].copy()
    for c in ("mid", "p_side", "p_adj", "edge", "kelly"): slim[c] = slim[c].round(3)
    tab = dict(data, rows=json.loads(slim.to_json(orient="records")))
    json.dump(tab, open(os.path.join(os.path.dirname(out_path), "kalshi_tab_data.json"), "w"), separators=(",", ":"))
    html = open(os.path.join(HERE, "kalshi_template.html")).read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    open(out_path, "w").write(html)
    print(out_path, len(rows), "rows,", len(games), "games,", f"{len(html)/1e6:.2f} MB")

if __name__ == "__main__":
    main(*sys.argv[1:3])
