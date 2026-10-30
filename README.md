# Gridiron

NFL game predictions, fantasy football projections and Kalshi prop edges, rebuilt from free
nflverse play-by-play data. This repo is a storage copy of the project folder; the scheduled
refreshes run from the project folder, not from here.

## Layout

| Folder | What it is |
| --- | --- |
| `nfl_engine/v2/` | Current game engine (v3 features): win probability, margin and total per game, projected standings. Includes its own commit history. |
| `nfl_engine/nfl_engine/` | The original engine as first uploaded, kept for reference. |
| `fantasy/` | Weekly std / half-PPR / PPR player projections, lineups and pickups, Sleeper and ESPN league import, and the Gridiron page template. |
| `kalshi/` | Kalshi NFL prop prices compared with the model (read-only, no trading), plus a settled-props backtest. |
| `snapshots/` | Pre-game snapshots and post-game grades (e.g. CAR @ GB, 2026 week 8). |

Each folder has an `output/` with the latest generated files. `nfl_engine/v2/models/model.joblib`
is the trained model (~38 MB).

## Running

Needs Python 3 with `pandas`, `scikit-learn` and `joblib`. Data is downloaded from nflverse into a
local `data/` folder (~450 MB, not stored here).

```bash
cd nfl_engine/v2
python engine.py          # download, backtest, train, score (~8 min)
python engine.py train    # train and score, skip the backtest
python engine.py score    # daily re-score with the saved model (~20 s)

cd ../../fantasy
python fantasy.py backtest && python fantasy.py && python build_fantasy_page.py
```

The `refresh*.sh` scripts are the scheduled jobs. They expect the project folder at
`/mnt/project-files/`, so adjust their paths before running them elsewhere.

## Accuracy

The game engine is roughly at Vegas level in backtests, not better: 68.8% / 66.5% / 64.2% straight-up
for 2024 / 2025 / 2026 weeks 1 to 7, versus 70.5% / 65.8% / 64.2% for the Vegas favorite. Fantasy
half-PPR MAE is about 5.2 to 5.3 points on players averaging 5+, versus 5.6 to 5.9 for season average.
