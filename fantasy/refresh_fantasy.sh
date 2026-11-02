#!/usr/bin/env bash
# Rebuilds the fantasy projections and page. Works from a fresh container: restores the code from the
# project folder, downloads data locally (too big for the network mount), then copies outputs back.
# Reads the game engine's predictions.csv but never changes the engine.
set -euo pipefail
SRC=/mnt/project-files/fantasy
WORK=/home/user/fantasy
python3 -c "import pandas, sklearn, joblib" 2>/dev/null || pip install -q pandas scikit-learn joblib
mkdir -p "$WORK/output"
cp "$SRC"/*.py "$SRC"/*.sh "$SRC"/*.html "$WORK/"
cd "$WORK"
python3 -u fantasy.py backtest
python3 -u fantasy.py --no-download
python3 build_fantasy_page.py
mkdir -p "$SRC/output"
cp output/fantasy_projections.csv output/fantasy_run.json output/fantasy_meta.json output/fantasy_matchups.json output/fantasy_history.json output/fantasy_actual.json output/fantasy_backtest.csv output/fantasy.html "$SRC/output/"

# weekly accuracy report: keep the last pre-kickoff copy of these outputs (never fails the refresh)
python3 /mnt/project-files/reports/snapshot.py fantasy || true
