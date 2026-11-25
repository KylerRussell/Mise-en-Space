// Scan monitor: start a scan, watch every stage live, see which stretches of the video mapped
// and why the rest didn't. Plus a live coach that checks the phone camera's view as you film.

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const enc = encodeURIComponent;

const STAGES = [
  ["frames", "Pick sharp frames"], ["check", "Check the footage"], ["features", "Match features"],
  ["solve", "Solve cameras"], ["place", "Scale & join"], ["layout", "Trace rooms"],
  ["builtins", "Find built-ins"], ["finishes", "Floors & walls"], ["scene", "Build scene"],
];
const QC = {
  ok: ["#3d9a5b", "Good"], blurry: ["#e0a43a", "Blurry"], low_texture: ["#c9b458", "Plain view"],
  close_up: ["#e07b39", "Too close"], turning: ["#d64545", "Turning on the spot"], lost: ["#8f2d56", "Lost track"],
  dark: ["#5b5f8a", "Too dark"],
};
const MAP = {
  reference: ["#2f8a4c", "Mapped"], joined: ["#3aa39a", "Mapped (joined)"], "could not join": ["#e0a43a", "Solved alone, not joined"],
  bent: ["#d64545", "Solve collapsed"], "too small": ["#b9a27a", "Too few frames"], none: ["#d7b9b9", "Not mapped"],
};
const ISSUE = {
  turning: "Turning on the spot", lost: "Lost track", blurry: "Blurry", low_texture: "Plain view", close_up: "Too close",
  dark: "Too dark", unplaced: "Didn't map", furniture_wall: "Wall traced along furniture",
};

// ---- tabs --------------------------------------------------------------------------------
for (const b of document.querySelectorAll(".tabs button")) {
  b.onclick = () => {
    for (const x of document.querySelectorAll(".tabs button")) x.classList.toggle("on", x === b);
    $("#tab-jobs").hidden = b.dataset.tab !== "jobs";
    $("#tab-coach").hidden = b.dataset.tab !== "coach";
    if (b.dataset.tab === "jobs") draw();
  };
}

// ---- scan jobs ---------------------------------------------------------------------------
let jobs = [];
let cur = null; // the job on screen
let ws = null;

function blankJob(summary) {
  return { ...summary, events: [], frames: [], qc: {}, reg: null, pieces: {}, issues: [], layout: null, builtins: [],
    stages: {}, stageInfo: {}, duration: null, solved: 0, selIssue: null };
}

async function loadJobs() {
  try {
    jobs = await (await fetch("/api/scans")).json();
  } catch { return; }
  $("#job-list").innerHTML = jobs.length ? jobs.map((j) => `
    <li data-id="${esc(j.id)}" class="${cur?.id === j.id ? "sel" : ""}">
      <span class="dot ${esc(j.status)}"></span>
      <span class="spacer">${esc(j.name)}<br /><span class="dim">${esc(j.scene_id)} · ${new Date(j.created * 1000).toLocaleString()}</span></span>
    </li>`).join("") : '<li class="dim">No scans yet.</li>';
  for (const li of $("#job-list").querySelectorAll("li[data-id]")) li.onclick = () => openJob(li.dataset.id);
}

function openJob(id) {
  const s = jobs.find((j) => j.id === id) || { id };
  cur = blankJob(s);
  history.replaceState(null, "", `?scan=${enc(id)}`);
  $("#job-empty").hidden = true;
  $("#job").hidden = false;
  if (ws) ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/scans/${enc(id)}`);
  ws.onmessage = (m) => { if (cur?.id === id) { apply(JSON.parse(m.data)); schedule(); } };
  ws.onerror = () => poll(id);  // no websocket (a proxy?): fall back to polling
  loadJobs();
  schedule();
}

async function poll(id) {
  while (cur?.id === id && !["done", "failed", "cancelled"].includes(cur.status)) {
    const since = cur.events.length ? cur.events[cur.events.length - 1].seq : -1;
    try {
      const j = await (await fetch(`/api/scans/${enc(id)}?since=${since}`)).json();
      for (const e of j.events) apply(e);
      cur.status = j.status;
      schedule();
    } catch {}
    await new Promise((r) => setTimeout(r, 1000));
  }
}

function apply(e) {
  let j = cur;
  if (e.kind === "start" && j.events.length) {  // the scan was run again: start the picture over
    cur = j = blankJob({ id: j.id, name: j.name, scene_id: j.scene_id, status: "running" });
  }
  if (j.events.length && e.seq <= j.events[j.events.length - 1].seq) return;
  j.events.push(e);
  switch (e.kind) {
    case "start": j.status = "running"; j.options = e.options; break;
    case "video": j.duration = e.duration_s; break;
    case "frames": j.frames = e.frames; break;
    case "qc": j.qc[e.name] = e.status; break;
    case "stage": j.stages[e.stage] = e.status; j.stageInfo[e.stage] = e; break;
    case "progress": if (e.registered) j.solved = e.registered; break;
    case "registration": j.reg = e.frames; break;
    case "piece": j.pieces[e.id] = e; break;
    case "issue": j.issues.push(e); break;
    case "layout": j.layout = e; break;
    case "builtin": j.builtins.push(e); break;
    case "done": j.status = "done"; j.result = e.result; loadJobs(); break;
    case "error": j.status = "failed"; j.error = e.message; loadJobs(); break;
    case "cancelled": j.status = "cancelled"; loadJobs(); break;
  }
}

let raf = 0;
function schedule() {
  if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
}

function draw() {
  const j = cur;
  if (!j) return;
  $("#job-title").textContent = j.name || j.scene_id || j.id;
  const sub = [j.scene_id && `scene ${j.scene_id}`, j.frames.length && `${j.frames.length} frames`,
    j.status === "failed" ? `failed: ${j.error || ""}` : j.status].filter(Boolean);
  $("#job-sub").textContent = sub.join(" · ");
  $("#cancel").hidden = j.status !== "running" || j.id.startsWith("cli_");  // a terminal run is stopped there
  const done = j.status === "done";
  $("#open-scene").hidden = $("#open-empty").hidden = !done;
  if (done) {
    $("#open-scene").href = `/?scene=${enc(j.scene_id)}`;
    $("#open-empty").href = `/?scene=${enc(j.scene_id)}&fixtures=0`;
  }
  drawStages(j);
  drawTimeline(j);
  drawPlan(j);
  drawIssues(j);
}

function stageNote(j, id) {
  const e = j.stageInfo[id] || {};
  if (id === "solve" && j.stages.solve === "running" && j.solved) return `${j.solved} of ${j.frames.length} frames`;
  if (e.count !== undefined) return `${e.count}`;
  if (e.registered !== undefined) return `${e.registered} solved · ${e.pieces} piece${e.pieces === 1 ? "" : "s"}`;
  if (e.placed !== undefined) return `${e.placed} of ${e.of} placed`;
  if (e.rooms !== undefined) return `${e.rooms} rooms · ${e.doorways} doorways`;
  if (e.issues !== undefined) return `${e.issues} issue${e.issues === 1 ? "" : "s"}`;
  if (e.spaces !== undefined) return `${e.spaces} spaces`;
  if (e.reason) return "skipped";
  if (j.status === "failed" && !j.stages[id] && j.events.some((x) => x.kind === "error" && x.stage === id)) return "failed";
  return "";
}

function drawStages(j) {
  const err = j.events.find((e) => e.kind === "error");
  $("#stages").innerHTML = STAGES.map(([id, label]) => {
    let st = j.stages[id] || "pending";
    if (err && err.stage === id) st = "failed";
    return `<li class="${st}"><b>${label}</b><span>${esc(stageNote(j, id) || (st === "pending" ? "waiting" : st))}</span></li>`;
  }).join("");
}

// ---- timeline ----------------------------------------------------------------------------
const TL = { gutter: 70, rows: { qc: [14, 22], map: [46, 22], iss: [80, 34] } };

function pieceOf(j, name) {
  if (!j.reg) return null;
  const p = j.reg[name];
  if (p === null || p === undefined) return "none";
  return j.pieces[p]?.status || "joined";
}

function timeRange(j) {
  const ts = j.frames.map((f) => f.t);
  const lo = ts.length ? Math.min(...ts) : 0;
  const hi = ts.length ? Math.max(...ts) : j.duration || 1;
  return [lo, Math.max(hi, lo + 1)];
}

function drawTimeline(j) {
  const cv = $("#timeline");
  const dpr = devicePixelRatio || 1;
  const W = cv.clientWidth, H = 132;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  const muted = css.getPropertyValue("--muted").trim() || "#777";
  g.font = "11px system-ui, sans-serif";
  g.fillStyle = muted;
  g.textBaseline = "middle";
  g.fillText("Footage", 0, TL.rows.qc[0] + 11);
  g.fillText("Mapped", 0, TL.rows.map[0] + 11);
  g.fillText("Issues", 0, TL.rows.iss[0] + 9);
  const [lo, hi] = timeRange(j);
  const x = (t) => TL.gutter + (t - lo) / (hi - lo) * (W - TL.gutter - 6);
  const n = j.frames.length;
  for (let i = 0; i < n; i++) {
    const f = j.frames[i];
    const x0 = x(i ? (j.frames[i - 1].t + f.t) / 2 : f.t), x1 = x(i < n - 1 ? (f.t + j.frames[i + 1].t) / 2 : f.t) + 1;
    const q = j.qc[f.name];
    g.fillStyle = q ? QC[q]?.[0] || "#999" : "#e6e1d7";
    g.fillRect(x0, TL.rows.qc[0], Math.max(1, x1 - x0), TL.rows.qc[1]);
    const p = pieceOf(j, f.name);
    g.fillStyle = p ? MAP[p]?.[0] || "#3aa39a" : "#e6e1d7";
    g.fillRect(x0, TL.rows.map[0], Math.max(1, x1 - x0), TL.rows.map[1]);
  }
  // issues: stacked bars in two lanes
  const lanes = [0, 0];
  j._bars = [];
  for (const [k, iss] of j.issues.entries()) {
    if (iss.t0 === null || iss.t0 === undefined) continue;
    const a = x(iss.t0), b = Math.max(x(iss.t1) , a + 3);
    const lane = lanes[0] <= a ? 0 : lanes[1] <= a ? 1 : 0;
    lanes[lane] = b + 2;
    const y = TL.rows.iss[0] + lane * 17;
    g.fillStyle = iss.severity === "fail" ? "#d64545" : "#e0a43a";
    g.globalAlpha = j.selIssue === k ? 1 : 0.8;
    g.fillRect(a, y, b - a, 14);
    g.globalAlpha = 1;
    if (j.selIssue === k) { g.strokeStyle = css.getPropertyValue("--ink"); g.lineWidth = 1.5; g.strokeRect(a, y, b - a, 14); }
    const label = ISSUE[iss.issue] || iss.issue;
    if (b - a > g.measureText(label).width + 8) { g.fillStyle = "#fff"; g.fillText(label, a + 4, y + 7.5); }
    j._bars.push([a, b, y, k]);
  }
  // seconds axis
  g.fillStyle = muted;
  g.textBaseline = "top";
  const step = niceStep((hi - lo) / Math.max(2, (W - TL.gutter) / 70));
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
    g.fillRect(x(t), TL.rows.map[0] + TL.rows.map[1] + 2, 1, 4);
    g.fillText(`${Math.round(t)}s`, x(t) - 8, TL.rows.map[0] + TL.rows.map[1] + 6);
  }
  if (!n) { g.textBaseline = "middle"; g.fillText("Frames appear here once they are picked.", TL.gutter, 25); }
  $("#legend").innerHTML = [...Object.values(QC).slice(0, 7).map(([c, l]) => [c, l]), ...Object.values(MAP).map(([c, l]) => [c, l])]
    .filter((v, i, a) => a.findIndex((w) => w[1] === v[1]) === i)
    .map(([c, l]) => `<span><i style="background:${c}"></i>${esc(l)}</span>`).join("");
}

function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
}

$("#timeline").onmousemove = (ev) => {
  const j = cur;
  if (!j || !j.frames.length) return;
  const r = ev.target.getBoundingClientRect();
  const px = ev.clientX - r.left;
  if (px < TL.gutter) { $("#peek").hidden = true; return; }
  const [lo, hi] = timeRange(j);
  const t = lo + (px - TL.gutter) / (r.width - TL.gutter - 6) * (hi - lo);
  const f = j.frames.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
  const q = j.qc[f.name], p = pieceOf(j, f.name);
  $("#peek-img").src = `/api/scans/${enc(j.id)}/thumbs/${enc(f.name)}`;
  $("#peek-text").textContent = `${f.t.toFixed(1)} s · ${q ? QC[q]?.[1] : "not checked yet"}${p ? ` · ${MAP[p]?.[1] || p}` : ""}`;
  $("#peek").style.left = `${px}px`;
  $("#peek").hidden = false;
};
$("#timeline").onmouseleave = () => { $("#peek").hidden = true; };
$("#timeline").onclick = (ev) => {
  const j = cur;
  if (!j?._bars) return;
  const r = ev.target.getBoundingClientRect();
  const px = ev.clientX - r.left, py = ev.clientY - r.top;
  const hit = j._bars.find(([a, b, y]) => px >= a && px <= b && py >= y && py <= y + 14);
  if (hit) selectIssue(hit[3]);
};

// ---- plan --------------------------------------------------------------------------------
function drawPlan(j) {
  const cv = $("#plan");
  const dpr = devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, W, H);
  const L = j.layout;
  $("#plan-note").textContent = L ? `${L.rooms.length} rooms · ${L.doorways.length} doorways` : "appears once the rooms are traced";
  if (!L) return;
  const all = [...L.cams, ...L.rooms.flatMap((r) => r.polygon)];
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs) - 0.5, Math.max(...xs) + 0.5, Math.min(...ys) - 0.5, Math.max(...ys) + 0.5];
  const s = Math.min(W / (x1 - x0), H / (y1 - y0));
  const ox = (W - (x1 - x0) * s) / 2, oy = (H - (y1 - y0) * s) / 2;
  const P = ([x, y]) => [ox + (x - x0) * s, H - oy - (y - y0) * s];
  // points
  g.fillStyle = "rgba(90, 84, 74, 0.28)";
  for (const p of L.points) { const [a, b] = P(p); g.fillRect(a, b, 1.5, 1.5); }
  // rooms
  const cols = ["#4e79a7", "#59a14f", "#b07aa1", "#f28e2b", "#76b7b2", "#edc948", "#e15759"];
  L.rooms.forEach((r, i) => {
    g.beginPath();
    r.polygon.forEach((p, k) => { const [a, b] = P(p); k ? g.lineTo(a, b) : g.moveTo(a, b); });
    g.closePath();
    g.fillStyle = cols[i % cols.length] + "22";
    g.fill();
    g.strokeStyle = "#3b3833";
    g.lineWidth = 2;
    g.stroke();
    const c = r.polygon.reduce((m, p) => [m[0] + p[0] / r.polygon.length, m[1] + p[1] / r.polygon.length], [0, 0]);
    const [a, b] = P(c);
    g.fillStyle = "#3b3833";
    g.font = "600 12px system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText(`Room ${i + 1}`, a, b - 2);
    g.font = "11px system-ui, sans-serif";
    g.fillText(`${r.area_m2.toFixed(1)} m²`, a, b + 12);
  });
  g.textAlign = "start";
  // built-ins
  for (const bi of j.builtins) {
    const [a, b] = P([bi.box[0], bi.box[3]]), [c, d] = P([bi.box[2], bi.box[1]]);
    g.fillStyle = "rgba(120, 130, 150, 0.35)";
    g.fillRect(a, b, c - a, d - b);
    g.strokeStyle = "#6c7891";
    g.lineWidth = 1;
    g.strokeRect(a, b, c - a, d - b);
  }
  // camera path
  g.strokeStyle = "rgba(31, 111, 235, 0.65)";
  g.lineWidth = 1.2;
  g.beginPath();
  L.cams.forEach((p, k) => { const [a, b] = P(p); k ? g.lineTo(a, b) : g.moveTo(a, b); });
  g.stroke();
  // suspect walls (furniture fronts that were pushed back)
  for (const f of L.suspect || []) {
    g.strokeStyle = "#e07b39";
    g.lineWidth = 3;
    g.setLineDash([6, 4]);
    g.beginPath();
    const [a, b] = P(f.a), [c, d] = P(f.b);
    g.moveTo(a, b); g.lineTo(c, d); g.stroke();
    g.setLineDash([]);
  }
  for (const d of L.doorways) {
    const [a, b] = P(d.center);
    g.fillStyle = "#fffdf9"; g.strokeStyle = "#3b3833"; g.lineWidth = 1.5;
    g.beginPath(); g.arc(a, b, 4, 0, 7); g.fill(); g.stroke();
  }
  const sel = j.selIssue !== null ? j.issues[j.selIssue] : null;
  if (sel?.where) {
    g.strokeStyle = "#d62828"; g.lineWidth = 5;
    const [a, b] = P(sel.where[0]), [c, d] = P(sel.where[1]);
    g.beginPath(); g.moveTo(a, b); g.lineTo(c, d); g.stroke();
  }
  // scale bar: 1 m
  g.fillStyle = "#3b3833";
  g.fillRect(12, H - 16, s, 3);
  g.font = "11px system-ui, sans-serif";
  g.fillText("1 m", 12, H - 22);
}

// ---- issues ------------------------------------------------------------------------------
function drawIssues(j) {
  const n = j.issues.length;
  $("#issue-count").textContent = n;
  $("#issue-count").className = `badge ${j.issues.some((i) => i.severity === "fail") ? "bad" : ""}`;
  const order = j.issues.map((iss, k) => [iss, k]).sort((a, b) => (a[0].severity === "fail" ? 0 : 1) - (b[0].severity === "fail" ? 0 : 1)
    || (a[0].t0 ?? 1e9) - (b[0].t0 ?? 1e9));
  const html = order.map(([iss, k]) => `
    <li data-k="${k}" class="${iss.severity} ${j.selIssue === k ? "on" : ""}">
      ${iss.frame ? `<img loading="lazy" src="/api/scans/${enc(j.id)}/thumbs/${enc(iss.frame)}" alt="" />` : "<span></span>"}
      <div><div class="k">${esc(ISSUE[iss.issue] || iss.issue)}</div>
        <div class="t">${iss.t0 !== null && iss.t0 !== undefined ? `${iss.t0.toFixed(0)}–${iss.t1.toFixed(0)} s · ` : ""}${iss.severity === "fail" ? "won't map" : "may be weak"}</div>
        <div>${esc(iss.message)}</div></div>
    </li>`).join("") || '<li class="dim" style="display:block;border:0">Nothing so far.</li>';
  if ($("#issues").dataset.html !== html) {
    $("#issues").innerHTML = html;
    $("#issues").dataset.html = html;
    for (const li of $("#issues").querySelectorAll("li[data-k]")) li.onclick = () => selectIssue(+li.dataset.k);
  }
}

function selectIssue(k) {
  cur.selIssue = cur.selIssue === k ? null : k;
  draw();
  $(`#issues li[data-k="${k}"]`)?.scrollIntoView({ block: "nearest" });
}

$("#cancel").onclick = async () => {
  if (!cur || !confirm("Cancel this scan?")) return;
  await fetch(`/api/scans/${enc(cur.id)}/cancel`, { method: "POST" });
};

// ---- new scan ----------------------------------------------------------------------------
const drop = $("#drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) { $("#video").files = e.dataTransfer.files; pickedFile(); }
};
$("#video").onchange = pickedFile;
function pickedFile() {
  const f = $("#video").files[0];
  if (!f) return;
  $("#drop-text").innerHTML = `<b>${esc(f.name)}</b><br /><span class="dim">${(f.size / 1e6).toFixed(0)} MB</span>`;
  if (!$("#scene-id").value) {
    $("#scene-id").value = f.name.replace(/\.[^.]+$/, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "scan";
  }
}

$("#new-scan").onsubmit = (e) => {
  e.preventDefault();
  const f = $("#video").files[0];
  if (!f) { alert("Choose a video first."); return; }
  const q = new URLSearchParams({ scene_id: $("#scene-id").value, name: $("#scene-name").value || $("#scene-id").value, fps: $("#fps").value || "3" });
  if ($("#t0").value) q.set("t0", $("#t0").value);
  if ($("#t1").value) q.set("t1", $("#t1").value);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/scans?${q}`);
  $("#upload").hidden = false;
  $("#start").disabled = true;
  xhr.upload.onprogress = (p) => {
    if (!p.lengthComputable) return;
    $("#upload-bar").style.width = `${(100 * p.loaded / p.total).toFixed(1)}%`;
    $("#upload-text").textContent = `Uploading… ${(p.loaded / 1e6).toFixed(0)} of ${(p.total / 1e6).toFixed(0)} MB`;
  };
  xhr.onload = async () => {
    $("#start").disabled = false;
    if (xhr.status !== 200) {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail; } catch {}
      $("#upload-text").textContent = `Couldn't start: ${typeof msg === "string" ? msg : JSON.stringify(msg)}`;
      return;
    }
    $("#upload-text").textContent = "Uploaded. Scanning…";
    const { id } = JSON.parse(xhr.responseText);
    await loadJobs();
    openJob(id);
  };
  xhr.onerror = () => { $("#start").disabled = false; $("#upload-text").textContent = "Upload failed."; };
  xhr.send(f);
};

// ---- live coach --------------------------------------------------------------------------
const coach = { stream: null, timer: 0, busy: false, session: `live-${Math.random().toString(36).slice(2, 10)}`, t0: 0, history: [], sent: 0 };
const BAD = new Set(["turning", "lost"]);

$("#coach-start").onclick = async () => {
  try {
    coach.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
  } catch (err) {
    $("#coach-banner").className = "coach-banner bad";
    $("#coach-banner").textContent = `No camera: ${err.message}. On a phone this page must be opened over https.`;
    return;
  }
  const v = $("#cam");
  v.srcObject = coach.stream;
  await v.play();
  coach.t0 = performance.now();
  coach.history = [];
  $("#coach-start").hidden = true;
  $("#coach-stop").hidden = false;
  coach.timer = setInterval(sendFrame, 333);
};

$("#coach-stop").onclick = () => {
  clearInterval(coach.timer);
  coach.stream?.getTracks().forEach((t) => t.stop());
  coach.stream = null;
  fetch(`/api/coach/${coach.session}`, { method: "DELETE" });
  coach.session = `live-${Math.random().toString(36).slice(2, 10)}`;
  $("#coach-start").hidden = false;
  $("#coach-stop").hidden = true;
};

const grab = document.createElement("canvas");
async function sendFrame() {
  const v = $("#cam");
  if (coach.busy || !v.videoWidth) return;
  coach.busy = true;
  grab.width = 480;
  grab.height = Math.round(480 * v.videoHeight / v.videoWidth);
  grab.getContext("2d").drawImage(v, 0, 0, grab.width, grab.height);
  const t = (performance.now() - coach.t0) / 1000;
  try {
    const blob = await new Promise((r) => grab.toBlob(r, "image/jpeg", 0.75));
    const r = await fetch(`/api/coach/${coach.session}?t=${t.toFixed(2)}`, { method: "POST", body: blob });
    if (r.ok) showCoach(await r.json());
  } catch {}
  coach.busy = false;
}

function showCoach(c) {
  const flag = c.flags.find((f) => QC[f]) || "ok";
  coach.history.push(flag);
  if (coach.history.length > 120) coach.history.shift();
  const cls = c.status === "ok" ? "ok" : BAD.has(c.status) ? "bad" : "warn";
  $("#coach-banner").className = `coach-banner ${cls}`;
  $("#coach-banner").textContent = c.status === "ok" ? c.advice : `${QC[c.status]?.[1] || c.status}: ${c.advice}`;
  const cv = $("#coach-strip");
  const W = cv.clientWidth, dpr = devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = 18 * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  const w = W / 120;
  coach.history.forEach((f, i) => { g.fillStyle = QC[f]?.[0] || "#999"; g.fillRect(W - (coach.history.length - i) * w, 0, w + 0.5, 18); });
  const share = coach.history.filter((f) => f === "ok").length / coach.history.length;
  $("#coach-stats").innerHTML = `
    <dt>Time</dt><dd>${c.t.toFixed(0)} s</dd>
    <dt>Good frames</dt><dd>${Math.round(100 * share)}% of the last ${coach.history.length}</dd>
    <dt>Turned</dt><dd>${c.rotation ?? "–"}° in the last second</dd>
    <dt>Parallax</dt><dd>${c.parallax ?? "–"}° ${c.parallax !== null && c.parallax < 0.5 ? "(no depth: walk, don't turn)" : ""}</dd>`;
}

// ---- start -------------------------------------------------------------------------------
addEventListener("resize", schedule);
loadJobs().then(() => {
  const id = new URLSearchParams(location.search).get("scan") || jobs.find((j) => j.status === "running")?.id;
  if (id) openJob(id);
});
setInterval(() => { if (!document.hidden) loadJobs(); }, 4000);
