# CAR @ GB, 2026 week 8 (Thu Oct 29, 8:15 PM ET): pre-game snapshot

Taken 2026-10-29 19:36 UTC, before kickoff. Copies only; the engines were not touched.
Sources: engine run scored 2026-10-29 15:40 UTC (trained through wk 7), fantasy run built 17:55 UTC,
Kalshi prices fetched 18:43 UTC, news check 18:55 UTC.

Files (suffix 20261029T1936Z):
- game_*.csv: engine row for this game
- fantasy_*.csv: all 36 projected players on CAR and GB (std / half / PPR plus stat lines)
- kalshi_*.json: all 265 Kalshi markets for the game with model prob, price, edge, flag
- kalshi_news_*.json: the news overrides in force

## Engine
GB to win, 70.1%. Score GB 26.8, CAR 20.3. Margin GB by 6.5, total 47.2. Vegas line GB -3.5.

## Kalshi
Market had GB 63.5% to win. Zero bets flagged (none cleared the 3c edge after fees).
Model leaned GB to cover big numbers and against CAR's run game / passing props
(e.g. NO on Hubbard 50+ rush yds at 29c, NO on Bryce Young 10+ rush yds at 49c),
all under the flag threshold.

## Top fantasy projections (half-PPR)
Love 19.0, Young 14.9, Jacobs 13.7, Watson 12.7, Hubbard 12.2, Golden 12.0 (Q),
McMillan 11.0, Kraft 9.8, Coker 8.7.
Note: Lloyd was projected 7.8 but flagged by news as a backup now that Jacobs is back.

Grading script: grade.py (run after the game).
