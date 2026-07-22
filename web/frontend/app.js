// Live training map: renders StreamWrapper coordinate batches relayed by
// the telemetry server onto the full Kanto map with per-agent trails.

// map_data.json coordinates are tile positions on the stitched Kanto map
// image, which is exactly 436 x 444 tiles at 16 px/tile (6976 x 7104 px).
// Note: the engine's global_map.py adds a 20-tile PAD on top of these for its
// internal exploration arrays — that pad does NOT apply to the image.
const GRID_COLS = 436;
const GRID_ROWS = 444;

const TRAIL_TTL_MS = 60_000; // points fade out over a minute
const MAX_POINTS_PER_AGENT = 2000;

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const connEl = document.getElementById("conn");
const statsEl = document.getElementById("stats");

let mapImage = null;
let regions = null; // map_n -> {coordinates: [x, y]}
const agents = new Map(); // agent key -> {color, points: [{gx, gy, t}]}
let batches = 0;
let lastBatchAt = null; // ms timestamp of the last received batch
// Active-run info mirrored from /api/train/status polling, so the stats line
// can distinguish "envs still booting" from "pipeline gone quiet".
let trainInfo = { running: false, startedAt: null };

// px per tile, computed once the image loads
let scaleX = 1;
let scaleY = 1;

// viewport transform
let view = { x: 0, y: 0, zoom: 0.12 };

const PALETTE = [
  "#e3350d", "#3fb950", "#58a6ff", "#d29922", "#bc8cff",
  "#f778ba", "#39d2c0", "#ffa657", "#7ee787", "#79c0ff",
];

function agentColor(key, metaColor) {
  if (metaColor) return metaColor;
  let h = 0;
  for (const c of key) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

function localToGlobal(x, y, mapN) {
  const region = regions[mapN];
  if (!region) return null;
  const [mapX, mapY] = region.coordinates;
  const gx = x + mapX;
  const gy = y + mapY;
  if (gx < 0 || gx >= GRID_COLS || gy < 0 || gy >= GRID_ROWS) return null;
  return [gx, gy];
}

function handleBatch(msg) {
  const meta = msg.metadata || {};
  const key = `${meta.user || "anon"}/${meta.env_id ?? "0"}`;
  if (!agents.has(key)) {
    agents.set(key, { color: agentColor(key, meta.color), points: [] });
  }
  const agent = agents.get(key);
  const t = (msg.ts ? msg.ts * 1000 : Date.now());
  for (const [x, y, mapN] of msg.coords || []) {
    const g = localToGlobal(x, y, mapN);
    if (g) agent.points.push({ gx: g[0], gy: g[1], t });
  }
  if (agent.points.length > MAX_POINTS_PER_AGENT) {
    agent.points.splice(0, agent.points.length - MAX_POINTS_PER_AGENT);
  }
  batches++;
  // Use the batch's own timestamp so replayed history doesn't read as live.
  lastBatchAt = Math.max(lastBatchAt ?? 0, t);
}

function updateStats() {
  let text = `${agents.size} agents · ${batches} batches`;
  const noDataFromRun =
    trainInfo.running && trainInfo.startedAt && (!lastBatchAt || lastBatchAt < trainInfo.startedAt);
  if (noDataFromRun) {
    // Booting all the emulators takes a few minutes on CPU before the first
    // coordinates arrive — say so instead of showing a silently blank map.
    const bootAge = Math.round((Date.now() - trainInfo.startedAt) / 1000);
    text += ` · warming up (${bootAge}s) — envs booting, first dots in a few minutes`;
  } else if (lastBatchAt) {
    const age = Math.round((Date.now() - lastBatchAt) / 1000);
    text += age < 30 ? " · live data" : ` · last data ${age}s ago`;
  }
  statsEl.textContent = text;
}

function draw() {
  requestAnimationFrame(draw);
  if (!mapImage) return;
  const w = canvas.width = window.innerWidth * devicePixelRatio;
  const h = canvas.height = window.innerHeight * devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(view.x * devicePixelRatio, view.y * devicePixelRatio);
  ctx.scale(view.zoom * devicePixelRatio, view.zoom * devicePixelRatio);
  ctx.imageSmoothingEnabled = view.zoom < 1;
  ctx.drawImage(mapImage, 0, 0);

  const now = Date.now();
  const r = Math.max(4, 0.45 * scaleX);
  for (const agent of agents.values()) {
    // Expire old trail points, but always keep the newest one so each agent's
    // last known position stays on the map through quiet spells (startup,
    // torch.compile, long rollouts) instead of the map going blank.
    if (agent.points.length > 1) {
      const keep = agent.points.filter((p) => now - p.t < TRAIL_TTL_MS);
      agent.points = keep.length ? keep : [agent.points[agent.points.length - 1]];
    }
    ctx.fillStyle = agent.color;
    for (const p of agent.points) {
      ctx.globalAlpha = Math.max(0.15, 1 - (now - p.t) / TRAIL_TTL_MS);
      ctx.beginPath();
      ctx.arc((p.gx + 0.5) * scaleX, (p.gy + 0.5) * scaleY, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  ctx.restore();
}

function setupInput() {
  let dragging = false;
  let last = null;
  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    last = [e.clientX, e.clientY];
    canvas.classList.add("dragging");
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    canvas.classList.remove("dragging");
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    view.x += e.clientX - last[0];
    view.y += e.clientY - last[1];
    last = [e.clientX, e.clientY];
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = Math.exp(-e.deltaY * 0.001);
    const newZoom = Math.min(4, Math.max(0.03, view.zoom * factor));
    // zoom around the cursor
    view.x = e.clientX - (e.clientX - view.x) * (newZoom / view.zoom);
    view.y = e.clientY - (e.clientY - view.y) * (newZoom / view.zoom);
    view.zoom = newZoom;
  }, { passive: false });
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/live`);
  ws.onopen = () => {
    connEl.textContent = "live";
    connEl.className = "ok";
  };
  ws.onmessage = (e) => {
    try {
      handleBatch(JSON.parse(e.data));
    } catch { /* ignore malformed frames */ }
  };
  ws.onclose = () => {
    connEl.textContent = "reconnecting…";
    connEl.className = "err";
    setTimeout(connect, 2000);
  };
}

// --- training run control ---------------------------------------------------

function $(id) { return document.getElementById(id); }

const ENVS_PER_WORKER = 12; // mirrors server derive_env_layout

function updateInstancesHint() {
  const n = parseInt($("cfg-instances").value, 10);
  $("cfg-instances-val").textContent = n;
  // RAM is the real limit on a 16GB machine (~350MB/instance + ~2GB trainer):
  // past ~6 instances alongside normal apps, workers hit swap and env
  // stepping collapses (measured: 8 instances → 25x slower per env).
  const note = n <= 6
    ? "fits alongside normal use"
    : "dedicated machine only — swaps and crawls otherwise";
  $("cfg-instances-hint").textContent =
    `→ ${n * ENVS_PER_WORKER} environments (${ENVS_PER_WORKER}/worker) · ${note}`;
}

async function loadConfigDefaults() {
  updateInstancesHint();
  $("cfg-instances").addEventListener("input", updateInstancesHint);
  let d;
  try {
    d = await (await fetch("/api/config/defaults")).json();
  } catch { return; }
  if (d.device) $("cfg-device").value = d.device;
  if (d.total_timesteps != null) $("cfg-total-timesteps").value = d.total_timesteps;
  const wrapSel = $("cfg-wrappers");
  wrapSel.innerHTML = "";
  for (const w of d.wrappers || []) {
    const o = document.createElement("option");
    o.value = o.textContent = w;
    if (w === d.default_wrappers_name) o.selected = true;
    wrapSel.appendChild(o);
  }
  const rewSel = $("cfg-reward");
  rewSel.innerHTML = "";
  for (const r of d.rewards || []) {
    const o = document.createElement("option");
    o.value = o.textContent = r;
    if (r === d.default_reward_name) o.selected = true;
    rewSel.appendChild(o);
  }
}

function collectStartOptions() {
  const num = (id) => { const v = parseInt($(id).value, 10); return Number.isFinite(v) ? v : null; };
  return {
    device: $("cfg-device").value || null,
    instances: num("cfg-instances"),
    total_timesteps: num("cfg-total-timesteps"),
    wrappers_name: $("cfg-wrappers").value || null,
    reward_name: $("cfg-reward").value || null,
    debug: $("cfg-debug").checked,
  };
}

// Mirror the active run's parameters into the launch form so the config
// drawer reflects reality (e.g. slider at 8 when an 8-instance run is live)
// instead of the form defaults.
let syncedRunId = null;

function syncFormToRun(status) {
  if (!status.running || !status.params || status.run_id === syncedRunId) return;
  syncedRunId = status.run_id;
  const p = status.params;
  if (p.instances != null) {
    $("cfg-instances").value = p.instances;
    updateInstancesHint();
  }
  if (p.total_timesteps != null) $("cfg-total-timesteps").value = p.total_timesteps;
  if (p.device) $("cfg-device").value = p.device;
  if (p.wrappers_name) $("cfg-wrappers").value = p.wrappers_name;
  if (p.reward_name) $("cfg-reward").value = p.reward_name;
  if (p.debug != null) $("cfg-debug").checked = !!p.debug;
}

function statusClass(s) {
  return ["running", "stopped", "exited", "failed"].includes(s) ? s : "";
}

async function refreshRuns() {
  const panel = $("runs-panel");
  if (!panel.classList.contains("open")) return;
  let data;
  try { data = await (await fetch("/api/runs")).json(); } catch { return; }
  const list = $("run-list");
  list.innerHTML = "";
  for (const r of data.runs || []) {
    const li = document.createElement("li");
    const params = Object.entries(r.params || {}).map(([k, v]) => `${k}=${v}`).join(" ");
    li.innerHTML =
      `<span class="run-status ${statusClass(r.status)}">${r.status}</span>` +
      `<span class="run-id">${r.id}</span>` +
      `<div class="run-meta">${r.command}${params ? " · " + params : ""}</div>` +
      (r.exp_id ? `<div class="run-meta">ckpt: ${r.exp_id}</div>` : "");
    list.appendChild(li);
  }
}

function drawSparkline(canvasEl, values) {
  const w = (canvasEl.width = canvasEl.clientWidth * devicePixelRatio);
  const h = (canvasEl.height = canvasEl.clientHeight * devicePixelRatio);
  const c = canvasEl.getContext("2d");
  c.clearRect(0, 0, w, h);
  if (values.length < 2) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  c.strokeStyle = "#e3350d";
  c.lineWidth = 1.5 * devicePixelRatio;
  c.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - min) / span) * (h - 4 * devicePixelRatio) - 2 * devicePixelRatio;
    i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
  });
  c.stroke();
}

function fmtMetric(v) {
  return typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3)) : v;
}

// Short hover descriptions for metric names. Losses are the engine's fixed
// PPO loss terms (see cleanrl_puffer.py's Losses dataclass); stats are
// reward/env-state counters and vary by reward set, so unknown keys fall
// back to a generic description instead of going tooltip-less.
const LOSS_DESCRIPTIONS = {
  policy_loss: "PPO clipped surrogate objective — how much the policy is changing per update.",
  value_loss: "Mean-squared error of the value function's return estimate.",
  entropy: "Policy entropy — higher means more exploration/randomness in actions.",
  old_approx_kl: "KL divergence estimate using the pre-update log-probs (cheap, biased).",
  approx_kl: "KL divergence between the old and new policy after this update — large values mean the step was too aggressive.",
  clipfrac: "Fraction of samples where PPO's probability ratio was clipped.",
  explained_variance: "How well the value function predicts actual returns (1.0 = perfect, 0 = no better than the mean).",
};

// Exact-match descriptions for the engine's fixed environment.py stat keys
// (see agent_stats() in pokemonred_puffer/environment.py). Nested dicts
// (menu, safari_zone, exploration, events, reward, required/useful items)
// get flattened to "parent/child" keys by pufferlib's unroll_nested_dict,
// so those are matched by prefix instead — see STAT_PREFIX_DESCRIPTIONS.
const STAT_DESCRIPTIONS = {
  "stats/badge": "Number of gym badges obtained.",
  "stats/max_map_progress": "Furthest map-progression milestone reached so far.",
  "stats/party_count": "Current number of Pokemon in the party.",
  "stats/hp": "HP fraction of the active/lead Pokemon.",
  "stats/coord": "Total distinct tile coordinates visited (overall exploration coverage).",
  "stats/warps": "Distinct warp tiles (doors/stairs) stepped on.",
  "stats/a_press": "Distinct spots where the A button was pressed to interact.",
  "stats/map_id": "Sum of distinct map IDs visited.",
  "stats/npc": "Distinct NPCs seen/interacted with.",
  "stats/hidden_obj": "Distinct hidden objects found.",
  "stats/sign": "Distinct signs read.",
  "stats/deaths": "Number of times the party has fainted (blacked out).",
  "stats/healr": "Total HP healed over the episode.",
  "stats/caught_pokemon": "Distinct Pokemon species caught.",
  "stats/seen_pokemon": "Distinct Pokemon species seen (Pokedex entries).",
  "stats/obtained_move_ids": "Distinct moves the party has learned.",
  "stats/opponent_level": "Highest level opponent Pokemon encountered so far.",
  "stats/taught_cut": "Whether the party knows the HM move Cut.",
  "stats/taught_surf": "Whether the party knows the HM move Surf.",
  "stats/taught_strength": "Whether the party knows the HM move Strength.",
  "stats/cut_tiles": "Tiles where Cut has been used.",
  "stats/valid_cut_coords": "Coordinates where using Cut was valid (hit a cuttable tile).",
  "stats/invalid_cut_coords": "Coordinates where using Cut was wasted/invalid.",
  "stats/valid_pokeflute_coords": "Coordinates where using the Poke Flute was valid.",
  "stats/invalid_pokeflute_coords": "Coordinates where using the Poke Flute was wasted/invalid.",
  "stats/valid_surf_coords": "Coordinates where using Surf was valid (hit water).",
  "stats/invalid_surf_coords": "Coordinates where using Surf was wasted/invalid.",
  "stats/blackout_check": "Map ID used to gate wild encounters near the last blackout.",
  "stats/blackout_count": "Number of times the player has blacked out (all party fainted).",
  "stats/item_count": "Number of items currently in the bag.",
  "stats/reset_count": "Number of episode resets so far this run.",
  "stats/pokecenter": "Distinct Pokemon Centers visited.",
  "stats/pokecenter_heal": "Healing done specifically at Pokemon Centers.",
  "stats/in_battle": "Whether the agent is currently in a battle.",
  "stats/event": "Weighted reward accumulated from story/event-flag progress.",
  "stats/max_steps": "Current max steps allowed per episode (can change with curriculum).",
  "stats/required_count": "Combined count of required story events + required items still tracked.",
  "stats/use_ball_count": "Poke Balls thrown/used.",
  "stats/step": "Cumulative step counter, including steps from prior episode resets.",
  "stats/last_action": "Index of the action taken on the most recent step.",
  "stats/action_hist": "Histogram of action counts taken so far.",
  reward_sum: "Total shaped reward for the step — the sum of every reward/* component below.",
};

const STAT_PREFIX_DESCRIPTIONS = [
  ["stats/menu/", "How many times this in-game menu screen has been opened (a UI-literacy proxy)."],
  ["stats/safari_zone/", "Steps taken in this Safari Zone area."],
  ["stats/exploration/", "Fraction of this map tileset's coordinates that have been visited."],
  ["stats/badge_", "Whether this specific gym badge has been obtained."],
  ["events/", "Story/game-progress event flag (boolean) read from Pokemon Red's RAM."],
  ["required_items/", "Whether this story-required item is currently in the bag."],
  ["useful_items/", "Whether this optional-but-useful item is currently in the bag."],
  ["reward/", "Weighted reward contribution from this component of the shaped reward function."],
];

function describeStat(key) {
  if (STAT_DESCRIPTIONS[key]) return STAT_DESCRIPTIONS[key];
  for (const [prefix, desc] of STAT_PREFIX_DESCRIPTIONS) {
    if (key.startsWith(prefix)) return desc;
  }
  return `Reward/environment-state counter from the engine (reward set-specific): ${key}`;
}

function tipAttr(text) {
  return `data-tip="${text.replace(/"/g, "&quot;")}"`;
}

async function refreshMetrics() {
  const panel = $("metrics-panel");
  if (!panel.classList.contains("open")) return;
  let latest, history;
  try {
    [latest, history] = await Promise.all([
      (await fetch("/api/metrics/latest")).json(),
      (await fetch("/api/metrics/history")).json(),
    ]);
  } catch { return; }

  const empty = $("metrics-empty");
  const body = $("metrics-body");
  if (!latest || latest.global_step == null) {
    empty.style.display = "block";
    body.style.display = "none";
    return;
  }
  empty.style.display = "none";
  body.style.display = "block";

  $("m-steps").textContent = fmtMetric(latest.global_step);
  $("m-epoch").textContent = fmtMetric(latest.epoch);
  $("m-sps").textContent = fmtMetric(Math.round(latest.sps));
  const up = latest.uptime || 0;
  $("m-uptime").textContent = `${Math.floor(up / 60)}m${Math.floor(up % 60)}s`;

  const points = history.points || [];
  drawSparkline($("sps-sparkline"), points.map((p) => p.sps));

  const lossesEl = $("m-losses");
  lossesEl.innerHTML = "";
  for (const [k, v] of Object.entries(latest.losses || {})) {
    const tip = tipAttr(LOSS_DESCRIPTIONS[k] || `PPO loss term: ${k}`);
    lossesEl.innerHTML += `<div class="metric-row"><span ${tip}>${k}</span><span>${fmtMetric(v)}</span></div>`;
  }

  const statsEl2 = $("m-stats");
  statsEl2.innerHTML = "";
  const statEntries = Object.entries(latest.stats || {}).sort(([a], [b]) => a.localeCompare(b));
  for (const [k, v] of statEntries) {
    const tip = tipAttr(describeStat(k));
    statsEl2.innerHTML += `<div class="metric-row"><span ${tip}>${k}</span><span>${fmtMetric(v)}</span></div>`;
  }
}

function setupDrawers() {
  const pairs = [
    ["config-toggle", "config-panel"],
    ["metrics-toggle", "metrics-panel"],
    ["runs-toggle", "runs-panel"],
  ];
  for (const [btnId, panelId] of pairs) {
    $(btnId).addEventListener("click", () => {
      const panel = $(panelId);
      // Close the other drawer so they don't overlap.
      for (const [, otherId] of pairs) if (otherId !== panelId) $(otherId).classList.remove("open");
      panel.classList.toggle("open");
      $(btnId).style.background = panel.classList.contains("open") ? "var(--accent)" : "#30363d";
      if (panelId === "runs-panel") refreshRuns();
      if (panelId === "metrics-panel") refreshMetrics();
    });
  }
}

function setupTrainControls() {
  const btn = document.getElementById("train-btn");
  const stateEl = document.getElementById("train-state");
  const logToggle = document.getElementById("log-toggle");
  const logEl = document.getElementById("log");
  let running = false;
  let busy = false;

  function render(status) {
    running = !!status.running;
    stateEl.classList.remove("degraded");
    btn.textContent = running ? "Stop" : "Train";
    btn.classList.toggle("stop", running);
    btn.disabled = busy;
    if (busy) {
      stateEl.textContent = running ? "stopping…" : "starting…";
    } else if (running) {
      const up = status.uptime_seconds ?? 0;
      let text = `running · ${Math.floor(up / 60)}m${Math.floor(up % 60)}s`;
      // Server-side watchdog: zombie workers or telemetry silence past the
      // grace windows. Without this the run reads "running" while dead.
      if (status.health === "degraded") text += ` · ⚠ ${status.health_reason}`;
      stateEl.textContent = text;
      stateEl.classList.toggle("degraded", status.health === "degraded");
    } else if (status.returncode != null) {
      stateEl.textContent = `exited (${status.returncode})`;
    } else {
      stateEl.textContent = "idle";
    }
    if (logEl.classList.contains("open") && Array.isArray(status.log)) {
      const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
      logEl.textContent = status.log.join("\n");
      if (atBottom) logEl.scrollTop = logEl.scrollHeight;
    }
  }

  let lastRunning = null;

  async function refresh() {
    try {
      const res = await fetch("/api/train/status");
      const status = await res.json();
      trainInfo = {
        running: !!status.running,
        startedAt: status.started_at ? Date.parse(status.started_at) : null,
      };
      syncFormToRun(status);
      render(status);
      // When the run state flips, refresh the history panel.
      if (status.running !== lastRunning) {
        lastRunning = status.running;
        refreshRuns();
      }
    } catch { /* server down; leave last state */ }
  }

  btn.addEventListener("click", async () => {
    busy = true;
    render({ running });
    try {
      const path = running ? "/api/train/stop" : "/api/train/start";
      const opts = running
        ? { method: "POST" }
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(collectStartOptions()),
          };
      const res = await fetch(path, opts);
      const status = await res.json();
      if (status.error) alert(status.error);
      busy = false;
      render(status);
      refreshRuns();
    } catch (e) {
      busy = false;
      alert("request failed: " + e);
      refresh();
    }
  });

  logToggle.addEventListener("click", () => {
    logEl.classList.toggle("open");
    logToggle.style.background = logEl.classList.contains("open") ? "var(--accent)" : "#30363d";
    refresh();
  });

  refresh();
  setInterval(refresh, 2000);
  setInterval(updateStats, 1000);
}

async function main() {
  const [imgBlobRes, mapDataRes] = await Promise.all([
    fetch("/assets/kanto_map.png"),
    fetch("/api/map-data"),
  ]);
  const raw = await mapDataRes.json();
  regions = {};
  for (const r of raw.regions) regions[parseInt(r.id)] = r;

  const img = new Image();
  img.src = URL.createObjectURL(await imgBlobRes.blob());
  await img.decode();
  mapImage = img;
  scaleX = img.width / GRID_COLS;
  scaleY = img.height / GRID_ROWS;

  // center the map initially
  view.zoom = Math.min(window.innerWidth / img.width, window.innerHeight / img.height) * 0.95;
  view.x = (window.innerWidth - img.width * view.zoom) / 2;
  view.y = (window.innerHeight - img.height * view.zoom) / 2;

  setupInput();
  setupDrawers();
  await loadConfigDefaults();
  setupTrainControls();
  connect();
  draw();
  setInterval(refreshMetrics, 3000);
}

main();
