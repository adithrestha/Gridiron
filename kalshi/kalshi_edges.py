"""Compare Kalshi NFL prices with the engine's game probabilities and the fantasy model's player projections.
Usage: python kalshi_edges.py MARKETS_JSON PLAYER_WEEKLY_DIR OUT_JSON
Reads (never writes) nfl_engine/v2/output/predictions.csv and fantasy/output/fantasy_projections.csv."""
import json, math, os, re, sys, unicodedata
from statistics import NormalDist
import numpy as np, pandas as pd

ROOT = "/mnt/project-files"
PRED = f"{ROOT}/nfl_engine/v2/output/predictions.csv"
PROJ = f"{ROOT}/fantasy/output/fantasy_projections.csv"
# Outcome spread around a good pre-game mean, from 2018-2025 results vs closing lines (nflverse games.csv).
# The engine's own in-sample margin sigma (12.3) is narrower than the market's out-of-sample 12.7, so use the wider ones.
SD_MARGIN, SD_TOTAL, SD_TEAM = 12.7, 13.6, 9.4
KALSHI_TO_NFLVERSE = {"JAC": "JAX", "LAR": "LA"}
# The engine is roughly at Vegas level (ATS 51-53%), so a raw model-vs-price gap is mostly model error.
# Shrink the model toward the market midpoint before calling anything value. Judgment calls, not backtested:
# game markets keep 35% of the model's disagreement, player props (projections never tested against prop lines) 25%.
TRUST = {"Winner": 0.35, "Spread": 0.35, "Total": 0.35, "Team total": 0.35, "Player prop": 0.25}
# Per-stat override for props, from the 2026 wk3-8 backtest vs Kalshi kickoff prices (kalshi/backtest/):
# blending in the model made pass-yard prices worse at every weight, so pass yards follow Kalshi and are never flagged.
PROP_TRUST = {"pass yds": 0.0}
MIN_EDGE = 0.03          # flag value only when the shrunk probability beats the buy price by 3c after fees
KELLY_FRACTION = 0.25
MAX_SPREAD = 0.04        # ...and the book is tight enough that the price means something
# Accuracy rules (Adit, 2026-10-30): only flag prices people are actually trading at, and skip games the model reads very
# differently from Kalshi (that gap is usually information the model lacks, as with props).
MIN_VOLUME = 1000        # contracts traded on that market (Kalshi's volume), and at least one trade
GAME_GAP = 0.10          # |model home win chance - Kalshi home win chance| at or above this: no game-market flags
MAX_P_OUT = 0.25         # skip players with real injury risk
# Role filter (Adit, 2026-10-29): props for players whose role is new or shifting are left out entirely, since the
# projections lag role changes and depth-chart news. Opportunities = carries + targets (+ pass attempts for QBs).
MIN_GAMES = 3            # games with an offensive touch this season
MIN_OPPS = 3.0           # season average opportunities per game
SNAP_SHIFT = 0.15        # |recent snap share - long-run snap share|
LAST_GAME_RATIO = 2.0    # last game's opportunities vs the season average before it, either direction
N = NormalDist()

def fee(p):  # Kalshi taker fee per $1 contract: ceil(0.07 * P * (1-P)) cents
    return math.ceil(7 * p * (1 - p) - 1e-9) / 100 if 0 < p < 1 else 0.0

def norm_name(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)

def price(x, side):
    k = f"{side}_ask_dollars"
    return float(x[k]) if x.get(k) not in (None, "") else None

# ---------- player stat distributions ----------
# Shape of game-to-game outcomes around a player's expected value, from nflverse weekly stats 2023-2025:
# ratio of actual to trailing-8-game mean, rescaled so its mean is 1 (assumes our projection is unbiased),
# kept separately for low / mid / high expectation tiers. Trailing means are noisier than real projections,
# so these spreads run a little wide, which pulls long-shot probabilities up. That is why long shots are not flagged.
STATS = {"KXNFLPASSYDS": ("passing_yards", "pyd", "QB"), "KXNFLRSHYDS": ("rushing_yards", "ryd", None),
         "KXNFLRECYDS": ("receiving_yards", "wyd", None), "KXNFLREC": ("receptions", "rec", None)}

def ratio_tables(weekly_dir):
    d = pd.concat([pd.read_csv(f"{weekly_dir}/pw_{y}.csv", low_memory=False) for y in (2023, 2024, 2025)])
    d = d[d.season_type == "REG"].sort_values(["player_id", "season", "week"])
    out = {}
    for series, (col, _, pos) in STATS.items():
        x = d if pos is None else d[d.position == pos]
        x = x[["player_id", col]].fillna(0).copy()
        x["m"] = x.groupby("player_id")[col].transform(lambda v: v.shift().rolling(8, min_periods=4).mean())
        x = x[x.m > x.m.quantile(0.4)]
        edges = x.m.quantile([0, 1 / 3, 2 / 3]).values
        tiers = []
        for i, lo in enumerate(edges):
            hi = edges[i + 1] if i + 1 < len(edges) else np.inf
            r = (x[col] / x.m)[(x.m >= lo) & (x.m < hi)].values
            tiers.append((lo, np.sort(r / r.mean())))
        out[series] = tiers
    return out

def p_over(tables, series, mean, line):
    tiers = tables[series]
    r = next(t for lo, t in reversed(tiers) if mean >= lo) if mean >= tiers[0][0] else tiers[0][1]
    return float(np.mean(r * mean > line))

def role_flags(weekly_dir, season):
    """player_id -> reason the player's role is too unsettled to price, from this season's weekly stats."""
    path = f"{weekly_dir}/pw_{season}.csv"
    try:
        w = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        return None
    w = w[w.season_type == "REG"].fillna({"carries": 0, "targets": 0, "attempts": 0})
    w["opp"] = w.carries + w.targets + np.where(w.position == "QB", w.attempts, 0)
    w = w[w.opp > 0].sort_values("week")
    out = {}
    for pid, g in w.groupby("player_id"):
        if len(g) < MIN_GAMES:
            out[pid] = f"only {len(g)} game{'s' if len(g) != 1 else ''} this season"; continue
        if g.opp.mean() < MIN_OPPS:
            out[pid] = f"{g.opp.mean():.1f} touches+targets a game"; continue
        prev, last = g.opp.iloc[:-1].mean(), g.opp.iloc[-1]
        if last >= LAST_GAME_RATIO * prev or last * LAST_GAME_RATIO <= prev:
            out[pid] = f"last game {int(last)} touches+targets vs {prev:.0f} before"
    return out

def poisson_over(mu, line):  # P(X > line) for integer X
    k = int(math.floor(line))
    return 1 - sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(k + 1))

# ---------- main ----------
def main(markets_path, weekly_dir, out_path):
    mk = json.load(open(markets_path))
    pred = pd.read_csv(PRED)
    proj = pd.read_csv(PROJ)
    # News notes written by Claude's news scan (refresh button / scheduled refresh): {"checked_at", "players": {"Name (TEAM)": {action, note, source}}}
    news_path = os.path.join(os.path.dirname(os.path.abspath(markets_path)), "kalshi_news.json")
    news_doc = json.load(open(news_path)) if os.path.exists(news_path) else {}
    news = news_doc.get("players", {})
    tables = ratio_tables(weekly_dir)
    season = int(pred.season.max())
    shaky = role_flags(weekly_dir, season)
    if shaky is None: raise SystemExit(f"missing {weekly_dir}/pw_{season}.csv (needed for the role filter)")
    games = {}
    for _, g in pred.iterrows():
        games[(g.away_team, g.home_team, str(g.gameday))] = g

    def game_of(event_ticker):
        code = event_ticker.split("-")[1]  # e.g. 26NOV01BALBUF
        date = pd.to_datetime(code[:7], format="%y%b%d").strftime("%Y-%m-%d")
        teams = code[7:]
        for (a, h, day), g in games.items():
            if day != date: continue
            ka = {v: k for k, v in KALSHI_TO_NFLVERSE.items()}.get(a, a)
            kh = {v: k for k, v in KALSHI_TO_NFLVERSE.items()}.get(h, h)
            if teams == ka + kh: return g, ka, kh
        return None, None, None

    # Kalshi's own home-win chance per game (winner-market midpoint), for the game-gap rule
    k_home = {}
    for x in mk.get("KXNFLGAME", []):
        g, ka, kh = game_of(x["event_ticker"])
        if g is None or x["ticker"].split("-")[-1] != kh: continue
        k_home[(g.away_team, g.home_team)] = (float(x["yes_bid_dollars"]) + float(x["yes_ask_dollars"])) / 2

    rows = []
    def add(x, kind, g, label, p_model, extra=None):
        ya, na = price(x, "yes"), price(x, "no")
        yb = float(x["yes_bid_dollars"]) if x.get("yes_bid_dollars") else None
        if ya is None or na is None or yb is None: return
        spread = round(ya - yb, 2)
        mid = (ya + yb) / 2
        trust = PROP_TRUST.get((extra or {}).get("stat"), TRUST[kind]) if kind == "Player prop" else TRUST[kind]
        p_adj = mid + trust * (p_model - mid)
        best = None
        for side, p, pa, cost in (("YES", p_model, p_adj, ya), ("NO", 1 - p_model, 1 - p_adj, na)):
            if not 0 < cost < 1: continue
            e = pa - cost - fee(cost)
            if best is None or e > best[1]: best = (side, e, cost, p, pa)
        side, edge, cost, p_side, pa_side = best
        raw_edge = p_side - cost - fee(cost)
        # fractional Kelly on the shrunk probability, for a binary contract costing `cost` (fee included)
        c = cost + fee(cost)
        kelly = max(0.0, (pa_side - c) / (1 - c)) * KELLY_FRACTION if c < 1 else 0.0
        vol = float(x.get("volume_fp") or 0)
        last = float(x.get("last_price_dollars") or 0)
        kg = k_home.get((g.away_team, g.home_team))
        game_gap = None if kg is None else abs(g.home_win_prob - kg)
        why_not = []
        if vol < MIN_VOLUME or last <= 0: why_not.append("thin trading")
        if kind != "Player prop" and (game_gap is None or game_gap >= GAME_GAP): why_not.append("model far from Kalshi on this game")
        rows.append(dict(ticker=x["ticker"], last=last, game_gap=None if game_gap is None else round(game_gap, 3), kind=kind, week=int(g.week), gameday=str(g.gameday), gametime=str(g.gametime),
                         game=f"{g.away_team} @ {g.home_team}", label=label, p_model=round(p_model, 4),
                         yes_bid=yb, yes_ask=ya, no_ask=na, mid=round(mid, 3), book_spread=spread,
                         volume=float(x.get("volume_fp") or 0), oi=float(x.get("open_interest_fp") or 0),
                         side=side, buy_at=cost, fee=fee(cost), p_side=round(p_side, 4), p_adj=round(pa_side, 4),
                         raw_edge=round(raw_edge, 4), edge=round(edge, 4), kelly=round(kelly, 4),
                         value=bool(edge >= MIN_EDGE and spread <= MAX_SPREAD and 0.10 <= cost <= 0.90
                                    and not (extra or {}).get("injury") and not why_not),
                         why_not=", ".join(why_not), **(extra or {})))

    for x in mk.get("KXNFLGAME", []):
        g, ka, kh = game_of(x["event_ticker"])
        if g is None: continue
        team = x["ticker"].split("-")[-1]
        p = g.home_win_prob if team == kh else 1 - g.home_win_prob
        add(x, "Winner", g, f"{x['yes_sub_title']} wins", p)
    for x in mk.get("KXNFLSPREAD", []):
        g, ka, kh = game_of(x["event_ticker"])
        if g is None or x.get("floor_strike") is None: continue
        team = re.match(r"[A-Z]+", x["ticker"].split("-")[-1]).group()
        m = g.pred_margin if team == kh else -g.pred_margin
        add(x, "Spread", g, x["yes_sub_title"], 1 - N.cdf((x["floor_strike"] - m) / SD_MARGIN))
    for x in mk.get("KXNFLTOTAL", []):
        g, ka, kh = game_of(x["event_ticker"])
        if g is None or x.get("floor_strike") is None: continue
        add(x, "Total", g, f"Total over {x['floor_strike']}", 1 - N.cdf((x["floor_strike"] - g.pred_total) / SD_TOTAL))
    for x in mk.get("KXNFLTEAMTOTAL", []):
        g, ka, kh = game_of(x["event_ticker"])
        if g is None or x.get("floor_strike") is None: continue
        team = re.match(r"[A-Z]+", x["ticker"].split("-")[-1]).group()
        pts = g.pred_home_pts if team == kh else g.pred_away_pts
        add(x, "Team total", g, x["yes_sub_title"], 1 - N.cdf((x["floor_strike"] - pts) / SD_TEAM))

    # players
    proj["key"] = proj.name.map(norm_name) + "|" + proj.team
    w26 = pd.read_csv(f"{weekly_dir}/pw_{season}.csv", low_memory=False, usecols=["player_id", "season_type"])
    played = set(w26[w26.season_type == "REG"].player_id)
    by_key = {(k, int(w)): r for k, w, r in zip(proj.key, proj.week, proj.itertuples())}
    unmatched, excluded = 0, {}
    for series in ["KXNFLTD", "KXNFLPASSTDS"] + list(STATS):
        for x in mk.get(series, []):
            g, ka, kh = game_of(x["event_ticker"])
            if g is None or x.get("floor_strike") is None: continue
            name = x["yes_sub_title"].rsplit(":", 1)[0]
            seg = x["ticker"].split("-")[2]
            kteam = ka if seg.startswith(ka) else kh
            team = KALSHI_TO_NFLVERSE.get(kteam, kteam)
            r = by_key.get((norm_name(name) + "|" + team, int(g.week)))
            if r is None:
                unmatched += 1; continue
            why = shaky.get(r.id)
            if why is None and abs(r.snap_recent - r.snap_ew) >= SNAP_SHIFT:
                why = f"snap share {r.snap_ew:.0%} long-run vs {r.snap_recent:.0%} recently"
            if why is None and r.id not in set(shaky) and r.id not in played:
                why = "no games this season"
            pkey = f"{name} ({team})"
            nw = news.get(pkey, {})
            if why is None and nw.get("action") == "exclude":
                why = "news: " + nw.get("note", "")
            if why:
                excluded.setdefault(pkey, why); continue
            line = x["floor_strike"]
            if series == "KXNFLTD": mu = r.rtd + r.wtd; p = poisson_over(mu, line); stat = "TDs"
            elif series == "KXNFLPASSTDS": mu = r.ptd; p = poisson_over(mu, line); stat = "pass TDs"
            else:
                col = STATS[series][1]; mu = getattr(r, col); p = p_over(tables, series, mu, line)
                stat = {"pyd": "pass yds", "ryd": "rush yds", "wyd": "rec yds", "rec": "receptions"}[col]
            if mu <= 0: continue
            add(x, "Player prop", g, f"{name} {int(line + 0.5)}+ {stat}", p,
                dict(player=name, news=nw.get("note", ""), news_url=nw.get("source", ""), position=r.position, stat=stat, proj=round(float(mu), 2), p_out=float(r.p_out),
                     status=r.status if isinstance(r.status, str) else "", injury=bool(r.p_out >= MAX_P_OUT)))
    out = dict(fetched_at=mk.get("_fetched_at"), engine_run=json.load(open(f"{ROOT}/nfl_engine/v2/output/run.json")),
               fantasy_run=json.load(open(f"{ROOT}/fantasy/output/fantasy_run.json")),
               settings=dict(sd_margin=SD_MARGIN, sd_total=SD_TOTAL, sd_team=SD_TEAM, min_edge=MIN_EDGE, trust=TRUST, prop_trust=PROP_TRUST,
                             kelly_fraction=KELLY_FRACTION,
                             max_spread=MAX_SPREAD, max_p_out=MAX_P_OUT, min_volume=MIN_VOLUME, game_gap=GAME_GAP),
               unmatched_player_markets=unmatched, rows=rows,
               news_checked_at=news_doc.get("checked_at"),
               excluded_players=[dict(player=k, why=v) for k, v in sorted(excluded.items())])
    json.dump(out, open(out_path, "w"))
    df = pd.DataFrame(rows)
    print(len(df), "markets matched;", unmatched, "player markets unmatched;", len(excluded), "players left out by the role filter")
    print(df.groupby("kind").agg(n=("edge", "size"), value=("value", "sum")))

if __name__ == "__main__":
    main(*sys.argv[1:4])
