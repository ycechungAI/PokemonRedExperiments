// Live training map: renders StreamWrapper coordinate batches relayed by
// the telemetry server onto the full Kanto map with per-agent trails.

// Global map grid used by the engine's global_map.py: 476 cols x 484 rows
// of game tiles including a 20-tile pad on every side.
const PAD = 20;
const GRID_COLS = 436 + PAD * 2;
const GRID_ROWS = 444 + PAD * 2;

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
  const gx = x + mapX + PAD;
  const gy = y + mapY + PAD;
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
  statsEl.textContent = `${agents.size} agents · ${batches} batches`;
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
    agent.points = agent.points.filter((p) => now - p.t < TRAIL_TTL_MS);
    ctx.fillStyle = agent.color;
    for (const p of agent.points) {
      ctx.globalAlpha = Math.max(0.05, 1 - (now - p.t) / TRAIL_TTL_MS);
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
  connect();
  draw();
}

main();
