"""Builds output/picks.html from the engine's CSV outputs."""
import json, os, pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
p = pd.read_csv(f"{OUT}/predictions.csv")
s = pd.read_csv(f"{OUT}/projected_standings.csv")
b = pd.read_csv(f"{OUT}/backtest.csv")
run = json.load(open(f"{OUT}/run.json")) if os.path.exists(f"{OUT}/run.json") else {}
games = p.round(3).where(p.notna(), None).to_dict("records")
stand = s.round(2).to_dict("records")
back = b.round(4).to_dict("records")
data = json.dumps({"games": games, "standings": stand, "backtest": back, "run": run})

html = r"""<title>Gridiron Engine 2026</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one narrow column — header, week strip, game cards, then standings and model report */
:root{
  --bg:#F3F5F1; --surface:#FFFFFF; --fg:#16211B; --muted:#5E6B63; --line:#DCE1DA;
  --accent:#1E6A44; --accent-soft:#E2EFE6; --pos:#1E6A44; --neg:#B4432E; --bar:#C9D3CB;
  --display:"Barlow Condensed","Arial Narrow",sans-serif;
  --body:"IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#111713; --surface:#18201B; --fg:#E7ECE8; --muted:#94A39A; --line:#2A352E;
  --accent:#5FC18D; --accent-soft:#1F3328; --pos:#5FC18D; --neg:#EE8A73; --bar:#33413A; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#111713; --surface:#18201B; --fg:#E7ECE8; --muted:#94A39A; --line:#2A352E;
  --accent:#5FC18D; --accent-soft:#1F3328; --pos:#5FC18D; --neg:#EE8A73; --bar:#33413A; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.5}
.wrap{max-width:760px;margin:0 auto;padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
h1,h2{font-family:var(--display);font-weight:700;letter-spacing:.01em;text-wrap:balance;margin:0;line-height:1.05}
h1{font-size:2.6rem;text-transform:uppercase}
h2{font-size:1.6rem;text-transform:uppercase}
.lede{color:var(--muted);margin:6px 0 0;max-width:62ch}
.label{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.weeks{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}
.weeks button{font:600 .95rem var(--display);letter-spacing:.04em;border:1px solid var(--line);background:var(--surface);color:var(--fg);padding:6px 12px;border-radius:6px;cursor:pointer;white-space:nowrap}
.weeks button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.weeks button:focus-visible,.toggle:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.games{display:grid;gap:10px}
.game{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px;display:grid;gap:10px}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.match{font-family:var(--display);font-size:1.45rem;font-weight:700;letter-spacing:.02em}
.match .at{color:var(--muted);font-weight:600;padding:0 6px}
.match .win{color:var(--accent)}
.when{color:var(--muted);font-size:.82rem}
.bar{position:relative;height:8px;border-radius:4px;background:var(--bar);overflow:hidden}
.bar span{position:absolute;top:0;bottom:0;background:var(--accent)}
.row{display:flex;justify-content:space-between;gap:12px;font-size:.86rem;flex-wrap:wrap}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.stats div{display:grid;gap:2px}
.stats .v{font-family:var(--mono);font-size:.95rem}
.edge-pos{color:var(--pos)} .edge-neg{color:var(--neg)}
.flag{font-size:.72rem;font-weight:600;background:var(--accent-soft);color:var(--accent);border-radius:4px;padding:1px 6px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}
td{font-family:var(--mono);font-variant-numeric:tabular-nums}
td:first-child{font-family:var(--body);font-weight:600}
section{display:grid;gap:12px}
.note{color:var(--muted);font-size:.86rem;max-width:65ch;margin:0}
.toggle{font:600 .8rem var(--body);background:none;border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer;justify-self:start}
ul.src{margin:0;padding-left:18px;display:grid;gap:4px;font-size:.88rem}
@media (max-width:440px){h1{font-size:2.1rem}.match{font-size:1.3rem}}
</style>
<div class="wrap">
  <header>
    <div class="label">2026 regular season · weeks __WEEKS__</div>
    <h1>Gridiron Engine</h1>
    <p class="lede">Stacked machine-learning ensemble predictions for every remaining game. The model never sees the Vegas line, so the edge column shows where it disagrees with the market.</p>
    <p class="note" id="updated"></p>
  </header>
  <nav class="weeks" id="weeks" aria-label="Week"></nav>
  <div class="games" id="games"></div>
  <section>
    <h2>Projected final records</h2>
    <p class="note">Current record plus the sum of win probabilities for each remaining game.</p>
    <div class="tablewrap"><table id="stand"></table></div>
    <button class="toggle" id="moreStand" type="button">Show all 32 teams</button>
  </section>
  <section>
    <h2>How it backtests</h2>
    <p class="note">Each period was predicted by a model trained only on earlier seasons. ATS needs 52.4% to break even at standard −110 odds.</p>
    <div class="tablewrap"><table id="back"></table></div>
    <div class="tablewrap"><table id="models"></table></div>
  </section>
  <section>
    <h2>Data and models</h2>
    <ul class="src">
      <li><b>Results, rest days, home/neutral site, division games, roof, surface, weather, Vegas lines:</b> nflverse schedules, 1999–2026</li>
      <li><b>Team scoring, EPA per play, turnovers, sacks:</b> nflverse weekly team stats, 2012–2026; EPA is also adjusted for opponent strength and split into passing and rushing matchups</li>
      <li><b>Travel:</b> miles traveled, time zones crossed and what time kickoff feels like on each team's body clock</li>
      <li><b>Quarterbacks:</b> each QB rated on EPA per play and completion % over expected, weighted toward recent games and pulled toward replacement level on small samples. Upcoming games use the depth-chart starter, swapped for the backup by injury status; ratings follow players when they change teams</li>
      <li><b>Injuries:</b> official NFL injury reports, each player weighted by position and his snap share with his current team; practice reports are used until final designations post</li>
      <li><b>Elo:</b> margin-of-victory adjusted ratings since 1999, regressed ⅓ each offseason</li>
      <li><b>Win model:</b> logistic regression, random forest, extra trees, gradient boosting, histogram GBM, neural net, SVM and k-NN, blended by a stacking meta-model, then averaged with the score model so the pick, win % and projected score always agree</li>
      <li><b>Score model:</b> separate stacked regressors for margin and total points</li>
    </ul>
    <p class="note">Later weeks use each team's current form, Elo, starting QB and season-average injury load, so they will shift as games are played. The model retrains weekly; injuries and depth charts refresh daily.</p>
  </section>
</div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const fmt = (x,d=1) => x==null ? "–" : Number(x).toFixed(d);
const sign = x => (x>0?"+":x<0?"−":"") + Math.abs(x).toFixed(1);
const weeks = [...new Set(D.games.map(g=>g.week))].sort((a,b)=>a-b);
let cur = weeks[0];
try { const w = Number(localStorage.getItem("ge-week")); if (weeks.includes(w)) cur = w; } catch(e) {}
function spreadText(team, m){ return m===0 ? "PK" : `${team} −${Math.abs(m).toFixed(1)}`; }
function dateText(g){
  const d = new Date(g.gameday+"T12:00:00");
  const day = d.toLocaleDateString("en-US",{weekday:"short",month:"short",day:"numeric"});
  if(!g.gametime) return day;
  let [h,m] = g.gametime.split(":").map(Number); h -= 3; // nflverse times are Eastern
  const ap = h>=12?"PM":"AM"; const h12 = ((h+11)%12)+1;
  return `${day} · ${h12}:${String(m).padStart(2,"0")} ${ap} PT`;
}
function renderWeeks(){
  $("weeks").innerHTML = weeks.map(w=>`<button type="button" aria-pressed="${w===cur}" data-w="${w}">WK ${w}</button>`).join("");
  $("weeks").querySelectorAll("button").forEach(b=>b.onclick=()=>{cur=+b.dataset.w;try{localStorage.setItem("ge-week",cur)}catch(e){};renderWeeks();renderGames();});
}
function renderGames(){
  const gs = D.games.filter(g=>g.week===cur).sort((a,b)=>(b.confidence-a.confidence));
  $("games").innerHTML = gs.map(g=>{
    const hp = g.home_win_prob, ap = 1-hp, homeWin = hp>=0.5;
    const m = g.pred_margin;
    const modelLine = spreadText(m>=0?g.home_team:g.away_team, m);
    const vegas = g.vegas_home_spread;
    const vegasLine = vegas==null ? "Not posted" : spreadText(vegas>=0?g.home_team:g.away_team, vegas);
    let edge = "–", edgeCls = "";
    if (vegas!=null){ const e = g.edge_vs_vegas; const side = e>=0?g.home_team:g.away_team;
      edge = Math.abs(e)<0.5 ? "In line" : `${side} ${sign(Math.abs(e))}`; edgeCls = Math.abs(e)>=3 ? "edge-pos" : ""; }
    const flags = [];
    if (g.away_qb_out) flags.push(`${g.away_team} starting QB out`);
    if (g.home_qb_out) flags.push(`${g.home_team} starting QB out`);
    if (g.home_rest>=13) flags.push(`${g.home_team} off bye`);
    if (g.away_rest>=13) flags.push(`${g.away_team} off bye`);
    if (g.home_rest<=5||g.away_rest<=5) flags.push("Short week");
    return `<article class="game">
      <div class="top"><div class="match"><span class="${homeWin?"":"win"}">${g.away_team}</span><span class="at">@</span><span class="${homeWin?"win":""}">${g.home_team}</span></div>
      <div class="when">${dateText(g)}</div></div>
      <div class="row"><span><b>${g.pick}</b> to win · <span class="num">${Math.round(g.confidence*100)}%</span></span>
        <span class="num">${g.away_team} ${fmt(g.pred_away_pts,0)} – ${fmt(g.pred_home_pts,0)} ${g.home_team}</span></div>
      ${g.away_qb_name||g.home_qb_name?`<div class="row when"><span>QBs: ${g.away_qb_name||"?"} vs ${g.home_qb_name||"?"}</span></div>`:""}
      <div class="bar" aria-label="${g.away_team} ${Math.round(ap*100)}%, ${g.home_team} ${Math.round(hp*100)}%"><span style="${homeWin?`right:0;width:${hp*100}%`:`left:0;width:${ap*100}%`}"></span></div>
      <div class="stats">
        <div><span class="label">Model line</span><span class="v">${modelLine}</span></div>
        <div><span class="label">Vegas</span><span class="v">${vegasLine}</span></div>
        <div><span class="label">Edge</span><span class="v ${edgeCls}">${edge}</span></div>
      </div>
      ${flags.length?`<div class="row" style="justify-content:flex-start;gap:6px">${flags.map(f=>`<span class="flag">${f}</span>`).join("")}</div>`:""}
    </article>`;}).join("");
}
let showAll = false;
function renderStand(){
  const rows = showAll ? D.standings : D.standings.slice(0,12);
  $("stand").innerHTML = `<thead><tr><th>Team</th><th>Now</th><th>Left</th><th>Proj W</th><th>Proj L</th></tr></thead><tbody>`+
    rows.map(r=>`<tr><td>${r.team}</td><td>${r.wins}-${r.losses}${r.ties?"-"+r.ties:""}</td><td>${r.remaining}</td><td>${fmt(r.proj_wins)}</td><td>${fmt(r.proj_losses)}</td></tr>`).join("")+"</tbody>";
  $("moreStand").textContent = showAll ? "Show top 12" : "Show all 32 teams";
}
$("moreStand").onclick = ()=>{showAll=!showAll;renderStand();};
function pct(x){return (x*100).toFixed(1)+"%";}
$("back").innerHTML = `<thead><tr><th>Test period</th><th>Games</th><th>Model</th><th>Vegas fav</th><th>ATS</th><th>Brier</th><th>Margin MAE</th></tr></thead><tbody>`+
  D.backtest.map(b=>`<tr><td>${b.period}</td><td>${b.games}</td><td>${pct(b.accuracy)}</td><td>${pct(b.vegas_fav_accuracy)}</td><td>${pct(b.ats_accuracy)}</td><td>${b.brier.toFixed(3)}</td><td>${b.margin_mae.toFixed(1)}</td></tr>`).join("")+"</tbody>";
const names = {logreg:"Logistic regression",rf:"Random forest",et:"Extra trees",gbm:"Gradient boosting",histgbm:"Histogram GBM",mlp:"Neural net",svm:"SVM",knn:"k-NN",xgb:"XGBoost",lgbm:"LightGBM"};
const keys = Object.keys(D.backtest[0]).filter(k=>k.startsWith("acc_"));
$("models").innerHTML = `<thead><tr><th>Win accuracy by model</th>${D.backtest.map(b=>`<th>${b.period}</th>`).join("")}</tr></thead><tbody>`+
  `<tr><td>Stacked ensemble</td>${D.backtest.map(b=>`<td>${pct(b.accuracy)}</td>`).join("")}</tr>`+
  keys.map(k=>`<tr><td>${names[k.slice(4)]||k}</td>${D.backtest.map(b=>`<td>${pct(b[k])}</td>`).join("")}</tr>`).join("")+"</tbody>";
if (D.run && D.run.scored_at) $("updated").textContent = `Updated ${new Date(D.run.scored_at).toLocaleString("en-US",{weekday:"short",month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})} · model trained through ${D.run.trained_through}`;
renderWeeks(); renderGames(); renderStand();
</script>
"""
weeks = f"{p.week.min()}–{p.week.max()}" if p.week.nunique() > 1 else str(p.week.min())
open(f"{OUT}/picks.html", "w").write(html.replace("__DATA__", data).replace("__WEEKS__", weeks))
print("wrote picks.html")
