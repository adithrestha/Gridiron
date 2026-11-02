# Kalshi betting refresh (button comment "Refresh the Kalshi betting board now..." or scheduled run)

Read-only Kalshi market data only. Never touch trading/account endpoints, never retrain the engine, never run fantasy.py.

1. `bash /mnt/project-files/kalshi/refresh_kalshi.sh` (scheduled runs pass `auto`; it prints SKIP on non-game days -> stop quietly).
   If plain pip/python deps are missing: `pip install --proxy "$HTTPS_PROXY" pandas numpy`.
2. News scan. Gridiron's Betting tab now shows "Model's top picks" (Adit 2026-10-30): top 12 liquid markets (>=1000 traded, a last trade, gap <=4c, price 10-90c, not 'model far from Kalshi') where the model's side has positive edge, one per market, ranked by model win chance; 'Value' tag when edge >= 3c. Check news for the players in those picks as well as flagged ones. List flagged markets: `python -c "import json,pandas as pd;r=pd.DataFrame(json.load(open('/mnt/project-files/kalshi/output/kalshi_edges.json'))['rows']);print(r[r.value][['game','label','player']].to_string())"`.
   For each flagged player (and each team in a flagged game market), WebSearch current injury / depth-chart / snap news
   (this week; team injury report, beat writers). Decide per player:
   - `exclude` when news explains the gap against the model: the player or a teammate ahead of/behind them is ruled out or returning,
     a demotion/promotion, a trade, or a role change the projection can't see.
   - `keep` with a one-sentence factual note otherwise.
   Write `/mnt/project-files/kalshi/output/kalshi_news.json` as
   `{"checked_at": "<UTC ISO, minute>", "players": {"Name (TEAM)": {"action": "exclude"|"keep", "note": "...", "source": "<url>"}}}`
   (keys match the `player` name + nflverse team code, e.g. "MarShawn Lloyd (GB)"). Keep earlier entries for players whose news still applies; drop stale ones.
   Re-run so the notes apply and newly surfaced flags get checked too (repeat step 2 for any new flagged player, at most twice):
   `K=/mnt/project-files/kalshi; python $K/kalshi_edges.py $K/output/kalshi_markets.json /tmp/kalshi_work/data $K/output/kalshi_edges.json && python $K/build_kalshi_page.py $K/output/kalshi_edges.json $K/output/kalshi.html && python /mnt/project-files/fantasy/build_fantasy_page.py && python3 /mnt/project-files/reports/snapshot.py kalshi`
   (the last command saves the pre-kickoff board for the weekly accuracy report; keep it)
3. Copy `fantasy/output/fantasy.html` to the scratchpad and republish to https://claude.ai/artifact/GoQ8AvGrxxGbGTdqrYhFdq
   (Gridiron; pass `url`, omit `capabilities` so they carry forward; read first if asked). The standalone Kalshi Edge Board
   (https://claude.ai/artifact/73ioBVCWcqVrKBzm3K6DPQ) is deprecated as of 2026-10-30 (Adit): do NOT republish it. `kalshi.html` is still built but unused.
4. If triggered by a comment: reply in that comment thread with what changed (flags added/removed, news found), 1-3 sentences.
