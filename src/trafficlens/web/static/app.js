/* TrafficLens control room — no dependencies, talks to the local API. */
"use strict";

const $ = (id) => document.getElementById(id);
const overlay = $("overlay");
const VB = { w: 1000, h: 562 }; // overlay viewBox units

const state = {
  running: false,
  sourceInfo: null,
  gates: [],            // {name, start:[nx,ny], end:[nx,ny], label_positive, label_negative}
  calibration: null,    // CalibrationConfig as sent to the API
  calibClicks: [],      // normalized points while calibrating
  tool: null,           // "gate" | "calib" | null
  draft: null,          // gate being drawn {start, end}
  eventSeq: 0,
  unit: "kmh",
  pollTimer: null,
};

/* ── helpers ──────────────────────────────────────────────── */

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/* Gate and class names are user-typed; escape them wherever they meet
   innerHTML so a gate called "<img onerror=…>" stays a name, not markup. */
function esc(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

function srcAspect() {
  const s = state.sourceInfo;
  return s && s.width && s.height ? s.width / s.height : 16 / 9;
}

/* The <img> uses object-fit:contain, so the video content occupies a
   letterboxed rectangle inside the viewport. All gate math lives in
   normalized video coordinates; these two functions convert. */
function contentRect() {
  const a = srcAspect(), boxA = VB.w / VB.h;
  if (a >= boxA) { const h = VB.w / a; return { x: 0, y: (VB.h - h) / 2, w: VB.w, h }; }
  const w = VB.h * a; return { x: (VB.w - w) / 2, y: 0, w, h: VB.h };
}
function toNorm(evt) {
  const r = overlay.getBoundingClientRect(), c = contentRect();
  const vx = ((evt.clientX - r.left) / r.width) * VB.w;
  const vy = ((evt.clientY - r.top) / r.height) * VB.h;
  return [clamp01((vx - c.x) / c.w), clamp01((vy - c.y) / c.h)];
}
function toView(p) {
  const c = contentRect();
  return [c.x + p[0] * c.w, c.y + p[1] * c.h];
}

function svg(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

const unitLabel = () => (state.unit === "mph" ? "mph" : "km/h");

/* ── boot: metadata → form ────────────────────────────────── */

async function boot() {
  const meta = await api("/api/meta");
  const modelSel = $("f-model");
  for (const m of meta.models) {
    const opt = document.createElement("option");
    opt.value = opt.textContent = m;
    if (m === "yolo11n.pt") opt.selected = true;
    modelSel.appendChild(opt);
  }
  const chips = $("class-chips");
  for (const cls of meta.traffic_classes) addChip(chips, cls, ["car", "truck", "bus", "motorcycle"].includes(cls));
  const datalist = $("all-classes");
  for (const cls of meta.classes) {
    const opt = document.createElement("option");
    opt.value = cls;
    datalist.appendChild(opt);
  }
  const status = await api("/api/session");
  if (status.running) attachToRunningSession(status);
}

function addChip(row, cls, pressed) {
  if ([...row.children].some((c) => c.dataset.cls === cls)) return;
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.dataset.cls = cls;
  b.textContent = cls;
  b.setAttribute("aria-pressed", String(pressed));
  b.addEventListener("click", () => {
    b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
  });
  row.appendChild(b);
}

$("btn-add-class").addEventListener("click", () => {
  const input = $("f-class-extra");
  const cls = input.value.trim();
  if (cls) addChip($("class-chips"), cls, true);
  input.value = "";
});

$("f-conf").addEventListener("input", (e) => { $("conf-out").textContent = e.target.value; });

for (const btn of $("quick-sources").querySelectorAll("button")) {
  btn.addEventListener("click", () => { $("f-source").value = btn.dataset.src; });
}

/* ── session lifecycle ────────────────────────────────────── */

$("start-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errBox = $("form-error");
  errBox.hidden = true;
  const classes = [...$("class-chips").children]
    .filter((c) => c.getAttribute("aria-pressed") === "true")
    .map((c) => c.dataset.cls);
  if (!classes.length) { errBox.textContent = "Pick at least one class to count."; errBox.hidden = false; return; }
  const limit = parseFloat($("f-limit").value);
  $("btn-start").disabled = true;
  $("btn-start").textContent = "Loading model…";
  try {
    const res = await api("/api/session", {
      method: "POST",
      body: JSON.stringify({
        source: $("f-source").value.trim(),
        model: $("f-model").value,
        classes,
        confidence: parseFloat($("f-conf").value),
        speed_limit: Number.isFinite(limit) ? limit : null,
        speed_unit: $("f-unit").value,
      }),
    });
    state.unit = $("f-unit").value;
    state.sourceInfo = res.source;
    enterRunning();
  } catch (err) {
    errBox.textContent = err.message;
    errBox.hidden = false;
  } finally {
    $("btn-start").disabled = false;
    $("btn-start").textContent = "Start analysis";
  }
});

function attachToRunningSession(status) {
  state.sourceInfo = status.source;
  state.unit = status.speed_unit || "kmh";
  enterRunning();
  // adopt the running session's gates (e.g. started from the CLI or a reload)
  if (status.gates && status.gates.length) {
    state.gates = status.gates;
    renderGates();
    renderGateList();
    showHint(null);
  }
}

function enterRunning() {
  state.running = true;
  state.gates = [];
  state.eventSeq = 0;
  state.incidentSeq = 0;
  $("stream").src = "/api/stream?" + Date.now();
  $("stream").style.display = "block";
  $("viewport-empty").hidden = true;
  $("btn-start").hidden = true;
  $("btn-stop").hidden = false;
  $("gate-module").hidden = false;
  $("dl-events").hidden = false;
  $("dl-summary").hidden = false;
  for (const id of ["tool-gate", "tool-calib"]) $(id).disabled = false;
  setRunState("running", "analysing");
  showHint("Draw a counting gate: press “+ Counting gate”, then drag a line across the road.");
  renderGates();
  renderGateList();
  state.pollTimer = setInterval(poll, 650);
}

$("btn-stop").addEventListener("click", async () => {
  try { await api("/api/session", { method: "DELETE" }); } catch { /* already gone */ }
  exitRunning("standing by");
});

function exitRunning(label) {
  state.running = false;
  clearInterval(state.pollTimer);
  $("stream").src = "";
  $("stream").style.display = "none";
  $("viewport-empty").hidden = false;
  $("btn-start").hidden = false;
  $("btn-stop").hidden = true;
  for (const id of ["tool-gate", "tool-calib", "tool-clear-calib"]) $(id).disabled = true;
  disarm();
  setRunState("idle", label);
  showHint(null);
}

function setRunState(kind, label) {
  $("run-state").dataset.state = kind;
  $("run-label").textContent = label;
}

function showHint(text) {
  const strip = $("hint-strip");
  strip.hidden = !text;
  if (text) strip.textContent = text;
}

/* ── polling: stats, events, violations ───────────────────── */

let violationTick = 0;

async function poll() {
  let status;
  try { status = await api("/api/session"); } catch { return; }
  if (!status.running) {
    if (status.error) setRunState("error", "source error");
    else exitRunning("source ended");
    return;
  }
  const stats = status.stats || {};
  $("tile-tracks").querySelector(".tile-value").textContent = stats.live_tracks ?? 0;
  $("tile-fps").querySelector(".tile-value").textContent = stats.fps ? stats.fps.toFixed(0) : "—";
  const summary = stats.summary;
  if (summary) {
    renderCountTiles(summary);
    renderSpeedTable(summary);
    $("tile-violations").querySelector(".tile-value").textContent = summary.violations ?? 0;
  }
  try {
    const ev = await api(`/api/events?after=${state.eventSeq}`);
    if (ev.events.length) {
      state.eventSeq = ev.latest;
      appendEvents(ev.events);
    }
  } catch { /* session raced away */ }
  try {
    const inc = await api(`/api/incidents?after=${state.incidentSeq || 0}`);
    if (inc.incidents.length) {
      state.incidentSeq = inc.latest;
      appendIncidents(inc.incidents);
    }
  } catch { /* session raced away */ }
  if (++violationTick % 4 === 0) refreshViolations();
}

function appendIncidents(incidents) {
  const body = $("incident-rows");
  body.querySelector(".board-empty")?.remove();
  for (const inc of incidents.slice().reverse()) {
    const tr = document.createElement("tr");
    tr.className = "fresh";
    const kindCls = inc.kind === "wrong_way" ? "flag" : "flag-warn";
    tr.innerHTML =
      `<td>${inc.t.toFixed(1)}s</td>` +
      `<td><span class="${kindCls}">${esc(inc.kind.replace("_", " ").toUpperCase())}</span></td>` +
      `<td>${esc(inc.class)} #${inc.track}</td><td>${esc(inc.detail)}</td>`;
    body.prepend(tr);
  }
  while (body.children.length > 24) body.lastChild.remove();
}

function renderCountTiles(summary) {
  const box = $("count-tiles");
  const gateNames = Object.keys(summary.gates || {});
  if (!gateNames.length) return;
  box.innerHTML = "";
  for (const name of gateNames) {
    const g = summary.gates[name];
    const dirs = Object.entries(g.by_direction || {});
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.innerHTML =
      `<span class="tile-value">${g.total}</span>` +
      `<span class="tile-label">${esc(name)}</span>` +
      `<span class="dir-split">${dirs.map(([d, n], i) =>
        `<span class="${i === 0 ? "in" : "out"}">${esc(d)} ${n}</span>`).join("")}</span>`;
    box.appendChild(tile);
  }
}

function renderSpeedTable(summary) {
  $("unit-echo").textContent = `(${unitLabel()})`;
  const rows = $("speed-rows");
  const classes = Object.keys(summary.speed_by_class || {});
  if (!classes.length) return;
  rows.innerHTML = "";
  for (const cls of classes.sort()) {
    const s = summary.speed_by_class[cls];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(cls)}</td><td>${s.n}</td><td>${s.median}</td><td>${s.p85}</td><td>${s.max}</td>`;
    rows.appendChild(tr);
  }
}

function appendEvents(events) {
  const body = $("event-rows");
  body.querySelector(".board-empty")?.remove();
  for (const e of events.reverse()) {
    const tr = document.createElement("tr");
    tr.className = "fresh";
    const dirCls = e.direction === "out" ? "dir-out" : "dir-in";
    tr.innerHTML =
      `<td>${e.t.toFixed(1)}s</td><td>${esc(e.gate)}</td>` +
      `<td>${esc(e.class)} #${e.track}</td><td class="${dirCls}">${esc(e.direction)}</td>` +
      `<td>${e.speed != null ? e.speed.toFixed(0) + " " + unitLabel() : "—"}` +
      `${e.violation ? ' <span class="flag">OVER</span>' : ""}</td>`;
    body.prepend(tr);
  }
  while (body.children.length > 40) body.lastChild.remove();
}

async function refreshViolations() {
  let data;
  try { data = await api("/api/violations"); } catch { return; }
  if (!data.violations.length) return;
  const box = $("mugshots");
  box.innerHTML = "";
  for (const v of data.violations.slice().reverse().slice(0, 8)) {
    const fig = document.createElement("figure");
    fig.className = "mugshot";
    fig.innerHTML =
      `<img src="/api/violations/${v.seq}.jpg" alt="${esc(v.class)} over the limit">` +
      `<figcaption>${esc(v.class)} #${v.track} — <b>${v.speed ?? "?"} ${unitLabel()}</b></figcaption>`;
    box.appendChild(fig);
  }
}

/* ── tools: arming ────────────────────────────────────────── */

$("tool-gate").addEventListener("click", () => arm(state.tool === "gate" ? null : "gate"));
$("tool-calib").addEventListener("click", () => arm(state.tool === "calib" ? null : "calib"));
$("tool-clear-calib").addEventListener("click", async () => {
  await api("/api/calibration", { method: "PUT", body: JSON.stringify({ calibration: null }) });
  state.calibration = null;
  $("tool-clear-calib").disabled = true;
  renderGates();
  showHint("Calibration removed — speeds are off until you calibrate again.");
});

function arm(tool) {
  state.tool = tool;
  state.calibClicks = [];
  state.draft = null;
  $("tool-gate").classList.toggle("armed", tool === "gate");
  $("tool-calib").classList.toggle("armed", tool === "calib");
  overlay.classList.toggle("armed", tool !== null);
  if (tool === "gate") showHint("Drag a line across the road. Traffic is counted when it crosses the line — both directions.");
  else if (tool === "calib") showHint("Click 4 road-plane points in order: near-left → near-right → far-right → far-left of a rectangle you know the size of.");
  else if (state.running) showHint(null);
  renderGates();
}

function disarm() {
  arm(null);
}

/* ── gate drawing ─────────────────────────────────────────── */

let dragHandle = null; // {gateIdx, end:"start"|"end"}

overlay.addEventListener("pointerdown", (e) => {
  if (!state.running) return;
  const target = e.target.closest(".gate-handle");
  if (target && state.tool === null) {
    dragHandle = { gateIdx: +target.dataset.gate, end: target.dataset.end };
    overlay.setPointerCapture(e.pointerId);
    return;
  }
  if (state.tool === "gate") {
    state.draft = { start: toNorm(e), end: toNorm(e) };
    overlay.setPointerCapture(e.pointerId);
    renderGates();
  } else if (state.tool === "calib") {
    state.calibClicks.push(toNorm(e));
    renderGates();
    if (state.calibClicks.length === 4) openCalibPopover(e);
  }
});

overlay.addEventListener("pointermove", (e) => {
  if (dragHandle) {
    state.gates[dragHandle.gateIdx][dragHandle.end] = toNorm(e);
    renderGates();
  } else if (state.draft) {
    state.draft.end = toNorm(e);
    renderGates();
  }
});

overlay.addEventListener("pointerup", async (e) => {
  if (dragHandle) {
    overlay.releasePointerCapture(e.pointerId);
    dragHandle = null;
    await pushGates();
    return;
  }
  if (state.draft) {
    overlay.releasePointerCapture(e.pointerId);
    const d = state.draft;
    const [x1, y1] = toView(d.start), [x2, y2] = toView(d.end);
    if (Math.hypot(x2 - x1, y2 - y1) < 25) { state.draft = null; renderGates(); return; }
    openGatePopover(e);
  }
});

function openGatePopover(evt) {
  const pop = $("gate-popover");
  pop.hidden = false;
  positionPopover(pop, evt);
  const input = $("gate-name");
  input.value = `gate ${state.gates.length + 1}`;
  input.select();
  input.focus();
}

$("gate-save").addEventListener("click", commitGate);
$("gate-name").addEventListener("keydown", (e) => { if (e.key === "Enter") commitGate(); });
$("gate-cancel").addEventListener("click", () => {
  $("gate-popover").hidden = true;
  state.draft = null;
  disarm();
});

async function commitGate() {
  const name = $("gate-name").value.trim() || `gate ${state.gates.length + 1}`;
  if (state.gates.some((g) => g.name === name)) {
    $("gate-name").value = `${name} (2)`;
    return;
  }
  state.gates.push({
    name,
    start: state.draft.start,
    end: state.draft.end,
    label_positive: "in",
    label_negative: "out",
  });
  state.draft = null;
  $("gate-popover").hidden = true;
  disarm();
  await pushGates();
}

async function pushGates() {
  await api("/api/gates", { method: "PUT", body: JSON.stringify({ gates: state.gates }) });
  renderGates();
  renderGateList();
}

async function removeGate(idx) {
  state.gates.splice(idx, 1);
  await pushGates();
}

function renderGateList() {
  const ul = $("gate-list");
  ul.innerHTML = "";
  if (!state.gates.length) {
    const li = document.createElement("li");
    li.textContent = "none yet — draw one on the video";
    ul.appendChild(li);
    return;
  }
  state.gates.forEach((g, i) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = g.name;
    const kill = document.createElement("button");
    kill.className = "gate-kill";
    kill.title = `Delete ${g.name}`;
    kill.setAttribute("aria-label", `Delete ${g.name}`);
    kill.textContent = "✕";
    kill.addEventListener("click", () => removeGate(i));
    li.append(label, kill);
    ul.appendChild(li);
  });
}

/* ── calibration ──────────────────────────────────────────── */

function openCalibPopover(evt) {
  const pop = $("calib-popover");
  pop.hidden = false;
  positionPopover(pop, evt);
  $("calib-w").focus();
}

$("calib-save").addEventListener("click", async () => {
  const w = parseFloat($("calib-w").value);
  const l = parseFloat($("calib-l").value);
  if (!(w > 0) || !(l > 0)) return;
  const calibration = {
    mode: "homography",
    image_points: state.calibClicks,
    world_points: [[0, 0], [w, 0], [w, l], [0, l]],
  };
  try {
    await api("/api/calibration", { method: "PUT", body: JSON.stringify({ calibration }) });
    state.calibration = calibration;
    $("tool-clear-calib").disabled = false;
    showHint(`Calibrated: ${w} m × ${l} m. Speeds now appear on every tracked object.`);
  } catch (err) {
    showHint(`Calibration rejected: ${err.message}`);
  }
  $("calib-popover").hidden = true;
  disarm();
});

$("calib-cancel").addEventListener("click", () => {
  $("calib-popover").hidden = true;
  state.calibClicks = [];
  disarm();
});

function positionPopover(pop, evt) {
  const pad = 12;
  const x = Math.min(evt.clientX + pad, window.innerWidth - 340);
  const y = Math.min(evt.clientY + pad, window.innerHeight - 200);
  pop.style.left = `${Math.max(pad, x)}px`;
  pop.style.top = `${Math.max(pad, y)}px`;
}

/* ── SVG rendering ────────────────────────────────────────── */

function renderGates() {
  overlay.innerHTML = "";
  state.gates.forEach((g, i) => drawGate(g, i));
  if (state.draft) drawGate({ ...state.draft, name: "", label_positive: "in" }, -1, true);
  if (state.calibration) drawCalibQuad(state.calibration.image_points);
  state.calibClicks.forEach((p, i) => drawCalibPoint(p, i + 1));
}

function drawGate(g, idx, isDraft = false) {
  const [x1, y1] = toView(g.start), [x2, y2] = toView(g.end);
  overlay.appendChild(svg("line", { x1, y1, x2, y2, class: "gate-line" }));

  // Direction chevron: points to the gate's positive ("in") side.
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len; // normal toward the positive half-plane
  const cx = mx + nx * 16, cy = my + ny * 16;
  const tx = dx / len, ty = dy / len;
  overlay.appendChild(svg("polygon", {
    class: "gate-chevron",
    points: `${cx + nx * 8},${cy + ny * 8} ${cx - tx * 7},${cy - ty * 7} ${cx + tx * 7},${cy + ty * 7}`,
  }));

  // The annotated stream already prints each gate's name and running
  // count on the video, so the overlay draws no duplicate tag — it only
  // adds what the server can't: drag handles and the direction chevron.
  if (!isDraft) {
    for (const end of ["start", "end"]) {
      const [hx, hy] = toView(g[end]);
      overlay.appendChild(svg("circle", {
        cx: hx, cy: hy, r: 7, class: "gate-handle",
        "data-gate": idx, "data-end": end,
      }));
    }
  }
}

function drawCalibPoint(p, n) {
  const [x, y] = toView(p);
  overlay.appendChild(svg("circle", { cx: x, cy: y, r: 8, class: "calib-pt" }));
  const t = svg("text", { x: x - 3, y: y + 4, class: "calib-num" });
  t.textContent = String(n);
  overlay.appendChild(t);
}

function drawCalibQuad(points) {
  const pts = points.map((p) => toView(p).join(",")).join(" ");
  overlay.appendChild(svg("polygon", { points: pts, class: "calib-quad" }));
}

/* ── go ───────────────────────────────────────────────────── */

boot().catch((err) => {
  setRunState("error", "backend unreachable");
  console.error(err);
});
