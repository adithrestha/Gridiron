"""
Game-context features
=====================
  * Travel and body clock: miles each team travels, time zones crossed, and what time kickoff
    feels like on each team's home clock (a West Coast team at 1 PM ET plays at 10 AM).
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

NY, CHI, DEN, LA = "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"
STADIUMS = {  # stadium_id: (lat, lon, time zone)
    "ATL00": (33.758, -84.401, NY), "ATL97": (33.755, -84.401, NY), "BAL00": (39.278, -76.623, NY),
    "BOS00": (42.091, -71.264, NY), "BUF00": (42.774, -78.787, NY), "BUF01": (43.641, -79.389, "America/Toronto"),
    "CAR00": (35.226, -80.853, NY), "CHI98": (41.862, -87.617, CHI), "CIN00": (39.095, -84.516, NY),
    "CLE00": (41.506, -81.700, NY), "DAL00": (32.748, -97.093, CHI), "DEN00": (39.744, -105.020, DEN),
    "DET00": (42.340, -83.046, NY), "FRA00": (50.069, 8.645, "Europe/Berlin"), "GER00": (48.219, 11.625, "Europe/Berlin"),
    "GNB00": (44.501, -88.062, CHI), "HOU00": (29.685, -95.411, CHI), "IND00": (39.760, -86.164, NY),
    "JAX00": (30.324, -81.637, NY), "KAN00": (39.049, -94.484, CHI), "LAX01": (33.953, -118.339, LA),
    "LAX97": (33.864, -118.261, LA), "LAX99": (34.014, -118.288, LA), "LON00": (51.556, -0.280, "Europe/London"),
    "LON01": (51.456, -0.342, "Europe/London"), "LON02": (51.604, -0.066, "Europe/London"),
    "MAD01": (40.453, -3.688, "Europe/Madrid"), "MEL00": (-37.820, 144.983, "Australia/Melbourne"),
    "MEX00": (19.303, -99.150, "America/Mexico_City"), "MIA00": (25.958, -80.239, NY), "MIN00": (44.974, -93.258, CHI),
    "MIN01": (44.974, -93.258, CHI), "MIN98": (44.976, -93.225, CHI), "MUN01": (48.219, 11.625, "Europe/Berlin"),
    "NAS00": (36.166, -86.771, CHI), "NOR00": (29.951, -90.081, CHI), "NYC01": (40.814, -74.074, NY),
    "OAK00": (37.752, -122.201, LA), "PAR00": (48.924, 2.360, "Europe/Paris"), "PHI00": (39.901, -75.168, NY),
    "PHO00": (33.528, -112.263, "America/Phoenix"), "PIT00": (40.447, -80.016, NY),
    "RIO00": (-22.912, -43.230, "America/Sao_Paulo"), "SAO00": (-23.545, -46.474, "America/Sao_Paulo"),
    "SDG00": (32.783, -117.120, LA), "SEA00": (47.595, -122.332, LA), "SFO00": (37.714, -122.386, LA),
    "SFO01": (37.403, -121.970, LA), "STL00": (38.633, -90.188, CHI), "TAM00": (27.976, -82.503, NY),
    "VEG00": (36.091, -115.184, LA), "WAS00": (38.908, -76.865, NY),
}


def _miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(a))


def _local_hour(gameday, gametime, tz):
    """Kickoff (given in US Eastern, as nflverse lists it) expressed as a decimal hour in `tz`."""
    gametime = gametime if isinstance(gametime, str) else "13:00"
    t = datetime.fromisoformat(f"{gameday} {gametime}").replace(tzinfo=ZoneInfo(NY)).astimezone(ZoneInfo(tz))
    return t.hour + t.minute / 60, t.utcoffset().total_seconds() / 3600


def travel(g):
    """Adds {home,away}_miles, _tz_shift and _body_hour, using each team's usual home stadium that season."""
    g = g.copy()
    known = g.stadium_id.isin(STADIUMS.keys())
    home_games = g[(g.location != "Neutral") & known]
    base = home_games.groupby(["season", "home_team"]).stadium_id.agg(lambda s: s.value_counts().index[0])
    base = base.reset_index().sort_values("season")
    last_base = base.groupby("home_team").stadium_id.last()
    def team_base(season, team):
        b = base_map.get((season, team))
        return b if b is not None else last_base.get(team)
    base_map = {(s, t): b for s, t, b in base.itertuples(index=False)}
    venue = np.where(known, g.stadium_id, [team_base(s, t) for s, t in zip(g.season, g.home_team)])
    cache = {}
    for side in ("home", "away"):
        miles, shift, body = [], [], []
        for s, t, v, day, time in zip(g.season, g[f"{side}_team"], venue, g.gameday, g.gametime):
            b = team_base(s, t)
            if b is None or v not in STADIUMS:
                miles.append(np.nan); shift.append(np.nan); body.append(np.nan); continue
            (la1, lo1, tz_home), (la2, lo2, tz_venue) = STADIUMS[b], STADIUMS[v]
            key = (day, time, tz_home, tz_venue)
            if key not in cache:
                h_home, off_home = _local_hour(day, time, tz_home)
                _, off_venue = _local_hour(day, time, tz_venue)
                cache[key] = (h_home, abs(off_venue - off_home))
            h_home, dz = cache[key]
            miles.append(_miles(la1, lo1, la2, lo2)); shift.append(dz); body.append(h_home)
        g[f"{side}_miles"], g[f"{side}_tz_shift"], g[f"{side}_body_hour"] = miles, shift, body
    return g
