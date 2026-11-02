"""One-off (2026-11-02): rebuild week 8 pre-kickoff snapshots from the copies that survived, oldest first, so the
latest pre-kickoff copy of each game wins. Engine: predictions.csv scored 2026-10-29 15:40 UTC (the engine did not
rerun before Sunday). Fantasy/Kalshi: Thursday's CAR-GB snapshot, then the GitHub copy from 2026-10-30
(fantasy built 15:02 UTC, Kalshi prices 15:23 UTC) for Sunday games, then the live outputs for Monday night."""
import sys
from datetime import datetime, timezone
sys.path.insert(0, "/mnt/project-files/reports")
import snapshot as S
S.ONLY_WEEK = 8
g = S.schedule()
T = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
C = "/mnt/project-files/snapshots/car_gb_2026wk8"
B = "/mnt/project-files/reports/backfill_wk08"
S.snap_engine(g, T("2026-10-29T15:41"), "2026-10-29T15:41Z")
S.snap_fantasy(g, T("2026-10-29T19:36"), "2026-10-29T19:36Z", f"{C}/fantasy_20261029T1936Z.csv", "2026-10-29T17:55Z")
S.snap_fantasy(g, T("2026-10-30T15:02"), "2026-10-30T15:02Z", f"{B}/fantasy_projections_20261030T1502Z.csv", "2026-10-30T15:02Z")
S.snap_kalshi(g, T("2026-10-29T19:36"), "2026-10-29T19:36Z", f"{C}/kalshi_20261029T1936Z.json")
S.snap_kalshi(g, T("2026-10-30T15:23"), "2026-10-30T15:23Z", f"{B}/kalshi_edges_20261030T1523Z.json")
now = datetime.now(timezone.utc)
for f in (S.snap_engine, S.snap_fantasy, S.snap_kalshi):
    f(g, now, now.strftime("%Y-%m-%dT%H:%MZ"))
