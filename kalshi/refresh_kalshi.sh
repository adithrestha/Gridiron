#!/usr/bin/env bash
# Refresh Kalshi prices, recompute edges against the latest engine + fantasy outputs, rebuild the page.
# Read-only: never runs the engine or fantasy model, never touches Kalshi trading/account endpoints.
set -euo pipefail
# "auto": only run when there is an NFL game today (US Eastern), per the engine's predictions.csv
if [ "${1:-}" = auto ]; then
  today=$(TZ=America/New_York date +%F)
  if ! cut -d, -f3 /mnt/project-files/nfl_engine/v2/output/predictions.csv | grep -qx "$today"; then echo "SKIP: no game $today"; exit 0; fi
fi
K=/mnt/project-files/kalshi
W=${WORK:-/tmp/kalshi_work}; mkdir -p "$W/data" "$K/output"
for y in 2023 2024 2025; do
  [ -s "$W/data/pw_$y.csv" ] || curl -sSL -o "$W/data/pw_$y.csv" \
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_$y.csv"
done
# this season's weekly stats change every week: always re-download (role filter)
SEASON=$(python -c "import pandas as p; print(int(p.read_csv('/mnt/project-files/nfl_engine/v2/output/predictions.csv').season.max()))")
curl -sSL -o "$W/data/pw_$SEASON.csv" "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_$SEASON.csv"
python "$K/fetch_kalshi.py" "$K/output/kalshi_markets.json"
python "$K/kalshi_edges.py" "$K/output/kalshi_markets.json" "$W/data" "$K/output/kalshi_edges.json"
python "$K/build_kalshi_page.py" "$K/output/kalshi_edges.json" "$K/output/kalshi.html"
# Gridiron page's Betting tab: rebuild fantasy.html from the fantasy thread's current outputs (no model run)
python /mnt/project-files/fantasy/build_fantasy_page.py

# weekly accuracy report: keep the last pre-kickoff copy of these outputs (never fails the refresh)
python3 /mnt/project-files/reports/snapshot.py kalshi || true
