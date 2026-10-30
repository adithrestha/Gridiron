# CAR @ GB grades (final: GB 27, CAR 19)

Graded 2026-10-30 04:35 UTC against the pre-game snapshot (2026-10-29 19:36 UTC).
Game score from nflverse; Kalshi from each market's official settlement (264 of 265 settled);
player stats from nflverse (finalized 2026-10-30 13:45 UTC).

## Game: right on every call
| | Predicted | Actual | Off by |
|---|---|---|---|
| Winner | GB (70%) | GB | correct |
| GB points | 26.8 | 27 | 0.2 |
| CAR points | 20.3 | 19 | 1.3 |
| Margin | GB by 6.5 | GB by 8 | 1.5 |
| Total | 47.2 | 46 | 1.2 |
| vs Vegas GB -3.5 | GB covers | GB covered | correct |
| Total vs 47.5 market | slight under | under | correct |

## Kalshi: no bets were flagged, so nothing was staked
The model's leaning side on each market (not bets) graded like this:
| Market type | Markets | Lean right | Model Brier | Kalshi Brier | $1 on each lean |
|---|---|---|---|---|---|
| Winner | 2 | 2 | .090 | .133 | +$0.68 |
| Spread | 25 | 18 | .090 | .108 | +$3.74 |
| Team total | 27 | 17 | .065 | .070 | +$1.58 |
| Game total | 19 | 6 | .102 | .098 | -$2.67 |
| Player props | 191 | 64 | .133 | .123 | -$18.73 |
(Lower Brier = better probabilities. The model beat Kalshi on winner, spread and team totals, lost on props.)

Props that went wrong: the model leaned under on Hubbard (81 rush yds, TD, 7+ catches), Watson (113 yds),
Kraft (TD) and Golden (TD). Props that went right: Bryce Young unders (209 yds, 1 TD, under 10 rush yds) and most McMillan lines.

## Fantasy (half-PPR, final from nflverse week 8 stats)
| Player | Projected | Actual | Projected minus actual |
|---|---|---|---|
| Jordan Love | 19.0 | 21.5 | -2.5 |
| Bryce Young | 14.9 | 10.4 | +4.6 |
| Josh Jacobs | 13.7 | 20.2 | -6.5 |
| Christian Watson | 12.7 | 13.3 | -0.6 |
| Chuba Hubbard | 12.2 | 21.4 | -9.2 |
| Matthew Golden | 12.0 | 17.4 | -5.4 |
| Tetairoa McMillan | 11.0 | 8.8 | +2.2 |
| Tucker Kraft | 9.8 | 14.3 | -4.5 |
| Jalen Coker | 8.7 | 10.6 | -1.9 |
| MarShawn Lloyd | 7.8 | 2.1 | +5.7 |
| Pierre Strong | 7.7 | 0.2 | +7.5 |
| Darren Waller | 5.5 | 1.6 | +3.9 |
Average miss on the 12 players projected 5+: 4.5 pts (backtest MAE ~5.3).
The top 10 projected missed by 4.3. Biggest misses: Hubbard and Jacobs both outscored projections,
and GB backups Lloyd and Strong barely played once Jacobs returned (the news check had already pulled Lloyd's props).
Watson's fumble was lost, so he finished at 13.3. Full list in graded_fantasy.csv.

Files: grades.json, graded_kalshi.csv (every market with result).
