import { api, connect } from "./api.js";
import { Viewport } from "./viewport.js";
import { snapYaw } from "./geometry.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const state = { scene: null, variant: params.get("variant") || "main", objects: [], room: null, variants: [], selected: null };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function toast(msg, kind = "info") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast ${kind}`;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3500);
}

async function guarded(fn) {
  try { return await fn(); } catch (e) { toast(e.message, "error"); }
}

const vp = new Viewport($("#viewport"), {
  fileUrl: api.fileUrl,
  onSelect: (id) => select(id),
  onMoveEnd: (id, { x, y }) => guarded(async () => {
    const r = await api.move(state.scene, state.variant, id, { x, y, snap: true });
    if (r.notes?.length) toast(r.notes.join("; "));
    if (r.violations_caused?.length) toast(r.violations_caused[0].message, "warn");
    await refresh();
  }),
});

// ---- loading ----------------------------------------------------------------
async function loadScenes() {
  const scenes = await api.scenes();
  const sel = $("#scene-select");
  sel.innerHTML = scenes.map((s) => `<option value="${esc(s.id)}">${esc(s.name)}</option>`).join("");
  if (!scenes.length) {
    toast("No scenes yet. Run `mise demo` to create one.", "warn");
    return;
  }
  const want = params.get("scene");
  state.scene = scenes.some((s) => s.id === want) ? want : scenes[0].id;
  sel.value = state.scene;
  await openScene();
}

let disconnect = null;
async function openScene() {
  disconnect?.();
  disconnect = connect(state.scene, onEvent);
  await loadVariants();
  await refresh({ room: true });
}

async function loadVariants() {
  state.variants = await api.variants(state.scene);
  if (!state.variants.some((v) => v.id === state.variant)) state.variant = "main";
  $("#variant-select").innerHTML = state.variants
    .map((v) => `<option value="${esc(v.id)}">${esc(v.name)}${v.created_by === "agent" ? " · AI" : ""}</option>`).join("");
  $("#variant-select").value = state.variant;
}

async function refresh({ room = false } = {}) {
  const s = state.scene, v = state.variant;
  const [objects, check, library, ops] = await Promise.all([api.objects(s, v), api.check(s, v), api.library(s), api.ops(s, v)]);
  if (room || !state.room) {
    state.room = await api.room(s, v);
    vp.setRoom(s, state.room);
  }
  state.objects = objects;
  vp.setObjects(objects);
  renderObjectList();
  renderWarnings(check);
  renderLibrary(library);
  renderHistory(ops);
  await renderProposal();
  if (state.selected && !objects.some((o) => o.id === state.selected)) select(null);
  else renderSelection();
}

// ---- live updates ------------------------------------------------------------
let pending = null;
function onEvent(evt) {
  if (evt.type === "op" && evt.op?.kind === "create_variant") loadVariants();
  // Coalesce bursts (the AI can make many edits quickly).
  clearTimeout(pending);
  pending = setTimeout(() => {
    if (!vp.drag) refresh();
    if (evt.op && evt.op.actor === "agent" && evt.op.variant === state.variant) toast(`AI: ${evt.op.kind} ${evt.op.object_id ?? ""}`);
  }, 120);
}

// ---- panels ------------------------------------------------------------------
function renderObjectList() {
  $("#object-list").innerHTML = state.objects.map((o) => `
    <li data-id="${esc(o.id)}" class="${o.id === state.selected ? "sel" : ""}">
      <span class="num">${o.label}</span>
      <span class="grow">${esc(o.name)}</span>
      ${o.locked ? '<span class="pill">locked</span>' : ""}
      ${o.source === "product" ? '<span class="pill prod">product</span>' : ""}
      ${o.anisotropy && o.anisotropy > 1.1 ? '<span class="pill warn" title="size mismatch vs listing">size ±</span>' : ""}
    </li>`).join("");
  for (const li of $("#object-list").children) li.onclick = () => { select(li.dataset.id); vp.focus(li.dataset.id); };
}

function renderWarnings(check) {
  const vs = check.violations;
  $("#warn-count").textContent = vs.length ? `${check.hard} hard · ${check.soft} soft` : "none";
  $("#warn-count").className = `badge ${check.hard ? "bad" : check.soft ? "warn" : "ok"}`;
  $("#warnings").innerHTML = vs.length
    ? vs.map((v, i) => `<li data-i="${i}" class="${v.severity}"><span class="kind">${v.kind.replace("_", " ")}</span>${esc(v.message)}</li>`).join("")
    : '<li class="ok">No collisions, clear walkways, doors unblocked.</li>';
  for (const li of $("#warnings").children) {
    const v = vs[li.dataset.i];
    if (v?.objects?.length) li.onclick = () => select(v.objects[0]);
  }
  vp.setFlagged(vs.filter((v) => v.severity === "hard").flatMap((v) => v.objects));
}

function renderLibrary(lib) {
  $("#library").innerHTML = lib.length ? lib.map((a) => `
    <li>
      <span class="grow" title="${esc(a.warnings.join("\n"))}">${esc(a.name)}
        <span class="dim">${a.dims.w.toFixed(2)}×${a.dims.d.toFixed(2)}×${a.dims.h.toFixed(2)} m</span>
        ${a.product?.price != null ? `<span class="dim">${a.product.price} ${esc(a.product.currency ?? "")}</span>` : ""}
      </span>
      ${a.status === "warning" ? '<span class="pill warn">size ±</span>' : a.status === "fallback_box" ? '<span class="pill">box</span>' : ""}
      <button data-asset="${esc(a.id)}">Place</button>
    </li>`).join("") : '<li class="muted">Imported products appear here.</li>';
  for (const b of $("#library").querySelectorAll("button[data-asset]")) {
    b.onclick = () => guarded(async () => {
      const r = await api.add(state.scene, state.variant, { asset_id: b.dataset.asset, relation: { relation: "center" } });
      select(r.object_id);
      if (r.violations_caused?.length) toast(r.violations_caused[0].message, "warn");
      await refresh();
    });
  }
}

function renderHistory(ops) {
  $("#history").innerHTML = ops.slice().reverse().map((o) =>
    `<li class="${o.undone ? "undone" : ""}"><span class="pill ${o.actor === "agent" ? "ai" : ""}">${o.actor}</span> ${esc(o.kind)} ${esc(o.object_id ?? "")} <span class="dim">${esc(o.note)}</span></li>`).join("");
}

async function renderProposal() {
  const box = $("#proposal");
  if (state.variant === "main") { box.hidden = true; return; }
  box.hidden = false;
  const [d, shop] = await Promise.all([api.diff(state.scene, "main", state.variant), api.shopping(state.scene, state.variant)]);
  $("#changes").innerHTML = d.changes.length ? d.changes.map((c) => `
    <li>
      <span class="pill ${c.kind}">${c.kind}</span>
      <span class="grow">${esc(c.name)}
        ${c.kind === "move" ? `<span class="dim">${c.distance_m.toFixed(2)} m, ${c.rotation_deg.toFixed(0)}°</span>` : ""}
        ${c.cost ? `<span class="dim">${c.cost > 0 ? "+" : ""}${c.cost} ${esc(c.currency ?? "")}</span>` : ""}
      </span>
      <button data-accept="${esc(c.object_id)}" title="Copy this change into your main layout">Accept</button>
    </li>`).join("") : '<li class="muted">Same as main.</li>';
  for (const b of $("#changes").querySelectorAll("button[data-accept]")) {
    b.onclick = () => guarded(async () => { await api.promote(state.scene, state.variant, [b.dataset.accept]); toast("Accepted into main"); await refresh(); });
  }
  $("#shopping").innerHTML = shop.rows.length ? `
    <table><thead><tr><th>Item</th><th>Price</th><th>W×D×H</th></tr></thead><tbody>
    ${shop.rows.map((r) => `<tr><td><a href="${esc(r.link)}" target="_blank" rel="noopener">${esc(r.item)}</a><div class="dim">${esc(r.retailer)} · checked ${esc((r.date_checked || "").slice(0, 10))}</div></td>
      <td>${r.price ?? "?"} ${esc(r.currency ?? "")}</td><td>${esc(r.listed_w_d_h_m ?? "?")}</td></tr>`).join("")}
    </tbody><tfoot><tr><td>Total</td><td colspan="2">${shop.total} ${esc(shop.currency ?? "")}</td></tr></tfoot></table>`
    : '<p class="muted">Nothing to buy in this layout.</p>';
}

function select(id) {
  state.selected = id;
  vp.select(id);
  renderSelection();
  for (const li of $("#object-list").children) li.classList.toggle("sel", li.dataset.id === id);
}

function renderSelection() {
  const o = state.objects.find((x) => x.id === state.selected);
  $("#selection").hidden = !o;
  if (!o) return;
  $("#sel-info").innerHTML = `
    <div class="title">${esc(o.name)} <span class="dim">${esc(o.id)}</span></div>
    <div class="dim">${o.extent.w.toFixed(2)} × ${o.extent.d.toFixed(2)} × ${o.extent.h.toFixed(2)} m as placed · on ${esc(o.support)}</div>
    <div class="dim">x ${o.pose.x.toFixed(2)} · y ${o.pose.y.toFixed(2)} · yaw ${o.pose.yaw.toFixed(0)}°${o.pose.roll || o.pose.pitch ? ` · tipped ${o.pose.roll ? `roll ${o.pose.roll}°` : ""}${o.pose.pitch ? ` pitch ${o.pose.pitch}°` : ""}` : ""}</div>
    ${o.product ? `<div><a href="${esc(o.product.url)}" target="_blank" rel="noopener">${esc(o.product.retailer || "product")}</a> ${o.product.price ?? ""} ${esc(o.product.currency ?? "")}</div>` : ""}
    ${o.moved_from_scan ? '<div class="pill warn">moved from scan: lighting is baked from its old spot</div>' : ""}
    ${(o.warnings || []).map((w) => `<div class="warnline">${esc(w)}</div>`).join("")}`;
  $("#toggle-lock").textContent = o.locked ? "Unlock" : "Lock";
  $("#tip").hidden = o.upright_required;
  $("#upright-only").checked = o.upright_required;
  $("#upright-only").disabled = state.variant !== "main";
  $("#delete").textContent = state.variant === "main" ? "Delete" : "Hide";
}

// ---- actions ---------------------------------------------------------------------
async function rotate(delta) {
  const o = state.objects.find((x) => x.id === state.selected);
  if (!o) return;
  await guarded(async () => {
    const r = await api.move(state.scene, state.variant, o.id, { yaw: snapYaw(o.pose.yaw + delta), snap: true });
    if (r.violations_caused?.length) toast(r.violations_caused[0].message, "warn");
    await refresh();
  });
}

async function removeSelected() {
  const o = state.objects.find((x) => x.id === state.selected);
  if (!o) return;
  await guarded(async () => {
    if (state.variant === "main") {
      if (!confirm(`Delete ${o.name} from your main layout? (Undo restores it.)`)) return;
      await api.remove(state.scene, state.variant, o.id);
    } else {
      await api.hide(state.scene, state.variant, o.id);
    }
    select(null);
    await refresh();
  });
}

async function tip() {
  const o = state.objects.find((x) => x.id === state.selected);
  if (!o) return;
  if (o.upright_required) return toast(`${o.name} is set to upright only`, "warn");
  await guarded(async () => {
    const r = await api.move(state.scene, state.variant, o.id, { roll: ((o.pose.roll || 0) + 90) % 180, snap: true });
    if (r.violations_caused?.length) toast(r.violations_caused[0].message, "warn");
    await refresh();
  });
}

$("#tip").onclick = tip;
$("#upright-only").onchange = (e) => guarded(async () => {
  const o = state.objects.find((x) => x.id === state.selected);
  if (e.target.checked && (o.pose.roll || o.pose.pitch)) await api.move(state.scene, "main", o.id, { roll: 0, pitch: 0, snap: true });
  await api.update(state.scene, "main", o.id, { upright_only: e.target.checked });
  await refresh();
});
$("#rot-left").onclick = () => rotate(-15);
$("#rot-right").onclick = () => rotate(15);
$("#rot-90").onclick = () => rotate(90);
$("#delete").onclick = removeSelected;
$("#toggle-lock").onclick = () => guarded(async () => {
  const o = state.objects.find((x) => x.id === state.selected);
  if (state.variant !== "main") return toast("Lock objects in the main layout", "warn");
  await api.update(state.scene, "main", o.id, { locked: !o.locked });
  await refresh();
});
$("#undo").onclick = () => guarded(async () => { const r = await api.undo(state.scene, state.variant); if (!r.undone) toast("Nothing to undo"); await refresh(); });
$("#redo").onclick = () => guarded(async () => { const r = await api.redo(state.scene, state.variant); if (!r.redone) toast("Nothing to redo"); await refresh(); });

$("#scene-select").onchange = (e) => { state.scene = e.target.value; state.variant = "main"; state.room = null; openScene(); };
$("#variant-select").onchange = (e) => { state.variant = e.target.value; select(null); refresh(); };
$("#new-variant").onclick = () => guarded(async () => {
  const name = prompt("Name for the new layout", `Layout ${state.variants.length}`);
  if (!name) return;
  const r = await api.createVariant(state.scene, name, state.variant);
  state.variant = r.variant_id;
  await loadVariants();
  await refresh();
});
$("#accept-all").onclick = () => guarded(async () => {
  if (!confirm("Copy every change in this layout into your main layout?")) return;
  await api.promote(state.scene, state.variant, null);
  toast("All changes accepted into main");
  await refresh();
});
$("#discard-variant").onclick = () => guarded(async () => {
  if (!confirm("Delete this layout? Your main layout is not affected.")) return;
  await api.deleteVariant(state.scene, state.variant);
  state.variant = "main";
  await loadVariants();
  await refresh();
});

for (const b of document.querySelectorAll(".seg button")) {
  b.onclick = () => setView(b.dataset.view);
}
function setView(view) {
  for (const x of document.querySelectorAll(".seg button")) x.classList.toggle("on", x.dataset.view === view);
  vp.setMode(view);
  $("#walk-hint").hidden = view !== "walk";
  $("#hint").hidden = view === "walk";
}

$("#import-form").onsubmit = (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const status = $("#import-status");
  guarded(async () => {
    let job = await api.startImport(state.scene, {
      url: f.get("url"), category: f.get("category") || "furniture", dims: f.get("dims") || null,
    });
    status.textContent = "Importing… fetching listing, generating 3D, fitting scale";
    while (job.status === "queued" || job.status === "running") {
      await new Promise((r) => setTimeout(r, 1500));
      job = await api.job(job.id);
    }
    if (job.status === "done") {
      const r = job.result;
      status.textContent = `${r.name}: ${r.status === "fallback_box" ? "shown as a sized box" : `anisotropy ${r.anisotropy}`}`;
      e.target.reset();
    } else if (job.status === "needs_input") {
      status.textContent = job.question;
    } else {
      status.textContent = `Failed: ${job.error}`;
    }
    await refresh();
  });
};

$("#upload-form").onsubmit = (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  guarded(async () => {
    await api.upload(state.scene, { name: f.get("name"), category: f.get("category") || "furniture", w: +f.get("w"), d: +f.get("d"), h: +f.get("h") });
    e.target.reset();
    await refresh();
  });
};

window.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); (e.shiftKey ? $("#redo") : $("#undo")).click(); return; }
  if (vp.mode === "walk") return;
  if (e.key === "q" || e.key === "Q") rotate(-15);
  else if (e.key === "e" || e.key === "E") rotate(15);
  else if (e.key === "r" || e.key === "R") rotate(90);
  else if (e.key === "t" || e.key === "T") tip();
  else if (e.key === "Delete" || e.key === "Backspace") removeSelected();
  else if (e.key === "Escape") select(null);
  else if (e.key === "1") setView("orbit");
  else if (e.key === "2") setView("walk");
  else if (e.key === "3") setView("plan");
});

loadScenes().then(() => { if (params.get("view")) setView(params.get("view")); }).catch((e) => toast(e.message, "error"));
