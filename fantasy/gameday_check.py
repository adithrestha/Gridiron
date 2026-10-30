"""Prints GAMEDAY if an NFL game was played on the US Eastern date that just ended (run shortly after midnight
Eastern), else SKIP. Reads a fresh copy of nflverse's schedule."""
import io, urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
day = (datetime.now(ZoneInfo("America/New_York")) - timedelta(hours=6)).date().isoformat()
g = pd.read_csv(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()), usecols=["gameday", "game_type"])
print("GAMEDAY" if (g.gameday == day).any() else "SKIP", day)
