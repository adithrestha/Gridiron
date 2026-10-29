#!/usr/bin/env bash
# Scheduled refresh. Usage: refresh.sh train|score
#   train  weekly: download everything, retrain, save model, re-score
#   score  daily:  download current-season files, re-score with the saved model
# Works from a fresh container: restores the engine from the project folder, runs it
# locally (data/ is too big for the network mount), then copies code + outputs back.
set -euo pipefail
MODE=${1:?usage: refresh.sh train|score}
SRC=/mnt/project-files/nfl_engine/v2
WORK=/home/user/nfl_engine

python3 -c "import sklearn, joblib" 2>/dev/null || pip install -q scikit-learn pandas joblib
mkdir -p "$WORK"
if [ ! -f "$WORK/engine.py" ]; then
  cp -r "$SRC/." "$WORK/"
fi
cd "$WORK"
python3 -u engine.py "$MODE"
mkdir -p "$SRC/output" "$SRC/models"
cp output/* "$SRC/output/"
[ "$MODE" = train ] && cp models/model.joblib "$SRC/models/model.joblib"
cat output/run.json; echo
