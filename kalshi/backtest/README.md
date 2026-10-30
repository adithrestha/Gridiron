# Player-prop backtest vs Kalshi (2026-10-30)
Scripts ran from a scratch folder (relative paths). Order: settled.py (settled prop markets, GET-only) -> proj.py (fantasy.py's own
backtest() imported read-only, data to ./fdata, writes proj_bt.pkl) -> prices.py (hourly candle at kickoff, batch candlesticks; slow, ~30 min)
-> analyze.py -> improve.py -> bets.py; selfcal.py checks the model against real stat lines (no Kalshi).
kalshi_settled_props.csv = 22,608 settled pre-game markets (2026 wk3-8) with kickoff bid/ask and result.
Results: 20,140 graded markets. Log loss Kalshi .4710, model alone .4884, model blended 25% toward Kalshi .4699.
Pass yards: blending made it worse at every weight -> PROP_TRUST pass yds = 0. One bet per player-stat with live rules
(edge>=3c, price 10-90c, buy at ask, fee, no pass yds): 257 bets, ROI +11.8%, 90% CI -3.6%..+28% (game-clustered bootstrap).
Caveat: backtest projections know who actually played and the actual starting QB.
