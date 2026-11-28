import { api, connect } from "./api.js";
import { Viewport } from "./viewport.js";
import { snapYaw } from "./geometry.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const state = { scene: null, variant: params.get("variant") || "main", objects: [], room: null, variants: [], selected: null, space: null,
  fixtures: params.get("fixtures") !== "0" };
// what the viewport and the list show: on the empty-home view, built-ins (fixture objects) are hidden
const visible = () => state.objects.filter((o) => state.fixtures || !o.fixture);
const SPACE_KINDS = ["living", "kitchen", "dining", "bedroom", "bathroom", "office", "hallway", "entry", "closet", "laundry", "storage", "balcony", "other"];

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
  onFloorPick: ({ x, y, axis }) => splitAt(axis, Math.round((axis === "x" ? x : y) * 20) / 20),
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
    await renderFinishes();
  }
  state.objects = objects;
  vp.setObjects(visible());
  renderSpaces();
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
  const roomChanged = ["room", "finish", "promote"].includes(evt.op?.kind);
  // Coalesce bursts (the AI can make many edits quickly).
  clearTimeout(pending);
  pending = setTimeout(() => {
    if (!vp.drag) refresh({ room: roomChanged });
    if (evt.op && evt.op.actor === "agent" && evt.op.variant === state.variant) toast(`AI: ${evt.op.kind} ${evt.op.object_id ?? ""}`);
  }, 120);
}

// ---- finishes: paint, floors, ceilings, trim (per layout) ----------------------
const FINISH_PARTS = [["floor", "Floor"], ["walls", "Walls"], ["ceiling", "Ceiling"], ["trim", "Trim"]];
async function renderFinishes() {
  const box = $("#finish-list");
  if (!box) return;
  let f;
  try { f = await api.finishes(state.scene, state.variant); } catch { box.innerHTML = ""; return; }
  const spaces = f.spaces.filter((sp) => !state.space || sp.space === state.space);
  box.innerHTML = spaces.map((sp) => `
    <div class="finish-space"><div class="dim">${esc(sp.name)}</div>
    ${FINISH_PARTS.map(([part, label]) => {
      const sf = sp[part] || {};
      const changed = sp.changed.includes(part) ? `<span class="pill" title="changed in this layout">new</span>` : "";
      const kinds = f.kinds.map((k) => `<option ${k === (sf.kind || "") ? "selected" : ""}>${k}</option>`).join("");
      return `<div class="row finish-row" data-space="${esc(sp.space)}" data-part="${part}">
        <span class="grow">${label} ${changed}</span>
        <select class="finish-kind" title="material"><option value="">—</option>${kinds}</select>
        <input class="finish-color" type="color" value="${esc(sf.color || "#cccccc")}" title="color" /></div>`;
    }).join("")}
    ${Object.keys(sp.wall_overrides).length ? `<div class="muted">${Object.entries(sp.wall_overrides).map(([w, s]) => `wall ${esc(w)}: ${esc(s.kind || "")}`).join(" · ")}</div>` : ""}
    </div>`).join("");
  for (const row of box.querySelectorAll(".finish-row")) {
    const send = (body) => guarded(async () => {
      await api.setFinish(state.scene, state.variant, { space: row.dataset.space, part: row.dataset.part, ...body });
      toast(`${row.dataset.part} updated`);
      await refresh({ room: true });
    });
    row.querySelector(".finish-color").onchange = (e) => send({ color: e.target.value });
    row.querySelector(".finish-kind").onchange = (e) => e.target.value && send({ kind: e.target.value });
  }
}

// ---- panels ------------------------------------------------------------------
const spaceName = (id) => state.room?.spaces?.find((s) => s.id === id)?.name ?? "outside any space";

function renderSpaces() {
  const spaces = state.room?.spaces || [];
  if (state.space && !spaces.some((s) => s.id === state.space)) state.space = null;
  $("#space-list").innerHTML = spaces.map((sp) => `
    <li data-space="${esc(sp.id)}" class="${sp.id === state.space ? "sel" : ""}">
      <span class="grow">${esc(sp.name)} <span class="dim">${esc(sp.kind)} · ${sp.area_m2.toFixed(1)} m² · ${sp.objects.length} items</span></span>
      ${sp.open_with.length ? `<span class="pill" title="open to ${esc(sp.open_with.map(spaceName).join(", "))}">open</span>` : ""}
    </li>`).join("");
  for (const li of $("#space-list").children) li.onclick = () => selectSpace(li.dataset.space === state.space ? null : li.dataset.space);
  const sp = spaces.find((s) => s.id === state.space);
  $("#space-edit").hidden = !sp;
  syncOutlineEdit(sp);
  if (!sp) return;
  $("#space-name").value = sp.name;
  $("#space-kind").innerHTML = SPACE_KINDS.map((k) => `<option ${k === sp.kind ? "selected" : ""}>${k}</option>`).join("");
  $("#space-ceiling").value = sp.ceiling_height ?? "";
  $("#space-ceiling").placeholder = `${state.room.ceiling_height}`;
  $("#space-merge-with").innerHTML = spaces.filter((s) => s.id !== sp.id).map((s) => `<option value="${esc(s.id)}">${esc(s.name)}</option>`).join("");
  const editable = state.variant === "main";
  for (const el of $("#space-edit").querySelectorAll("input, select, button")) el.disabled = !editable;
  if (spaces.length < 2) $("#space-merge").disabled = $("#space-delete").disabled = $("#space-merge-with").disabled = true;
  $("#space-hint").textContent = editable ? "" : "Spaces are edited in the main layout.";
}

function syncOutlineEdit(sp) {
  const on = !!sp && $("#space-outline").checked && state.variant === "main";
  vp.editOutline(on ? sp : null, (polygon) => roomEdit(() => api.updateSpace(state.scene, sp.id, { polygon }), `${sp.name}: outline updated`));
}

$("#space-outline").onchange = (e) => {
  if (e.target.checked) setView("plan");
  syncOutlineEdit(state.room?.spaces?.find((s) => s.id === state.space));
};

function selectSpace(id) {
  state.space = id;
  vp.focusSpace(state.room.spaces.find((s) => s.id === id) || null);
  renderSpaces();
  renderObjectList();
  renderFinishes();
}

async function roomEdit(fn, msg) {
  await guarded(async () => {
    const r = await fn();
    if (msg) toast(msg);
    const hard = (r?.violations || []).filter((v) => v.severity === "hard");
    if (hard.length) toast(hard[0].message, "warn");
    await refresh({ room: true });
    return r;
  });
}

function splitAt(axis, at) {
  const sp = state.room.spaces.find((s) => s.id === state.space);
  if (!sp) return;
  const name = prompt(`Name for the new part of ${sp.name} (${axis === "x" ? "right of" : "above"} ${axis} = ${at.toFixed(2)} m)`, "Dining area");
  if (!name) return;
  const kind = SPACE_KINDS.find((k) => name.toLowerCase().includes(k.slice(0, 4))) || "other";
  roomEdit(async () => {
    const r = await api.splitSpace(state.scene, sp.id, { axis, at, new_name: name, new_kind: kind });
    state.space = r.space_id;
    return r;
  }, `Split ${sp.name}: ${name} is a zone of the same open floor`);
}

$("#space-all").onclick = () => selectSpace(null);
$("#space-save").onclick = () => roomEdit(() => api.updateSpace(state.scene, state.space, {
  name: $("#space-name").value.trim() || undefined, kind: $("#space-kind").value,
  ceiling_height: $("#space-ceiling").value ? +$("#space-ceiling").value : null,
}), "Saved");
for (const axis of ["x", "y"]) {
  $(`#space-split-${axis}`).onclick = () => {
    setView("plan");
    vp.startPick(axis);
    $("#space-hint").textContent = `Click the plan where the ${axis === "x" ? "vertical" : "horizontal"} zone line goes (Esc cancels).`;
  };
}
$("#space-merge").onclick = () => {
  const other = $("#space-merge-with").value;
  if (!other) return;
  roomEdit(() => api.mergeSpace(state.scene, state.space, other), `Merged ${spaceName(other)} into ${spaceName(state.space)}`);
};
$("#space-delete").onclick = () => {
  if (!confirm(`Delete the space ${spaceName(state.space)}? Its floor leaves the model (undo restores it).`)) return;
  const id = state.space;
  state.space = null;
  roomEdit(() => api.deleteSpace(state.scene, id), "Space deleted");
};

function renderObjectList() {
  const multi = (state.room?.spaces || []).length > 1;
  const shown = visible().filter((o) => !state.space || o.space === state.space);
  const groups = multi && !state.space
    ? [...new Set(shown.map((o) => o.space))].map((sid) => [sid, shown.filter((o) => o.space === sid)])
    : [[null, shown]];
  $("#object-list").innerHTML = groups.map(([sid, objs]) =>
    (multi && !state.space ? `<li class="group">${esc(spaceName(sid))}</li>` : "") + objs.map((o) => `
    <li data-id="${esc(o.id)}" class="${o.id === state.selected ? "sel" : ""}">
      <span class="num">${o.label}</span>
      <span class="grow">${esc(o.name)}</span>
      ${o.fixture ? '<span class="pill fixture" title="Built in: part of the home">built-in</span>' : o.locked ? '<span class="pill">locked</span>' : ""}
      ${o.source === "product" ? '<span class="pill prod">product</span>' : ""}
      ${o.anisotropy && o.anisotropy > 1.1 ? '<span class="pill warn" title="size mismatch vs listing">size ±</span>' : ""}
    </li>`).join("")).join("");
  for (const li of $("#object-list").querySelectorAll("li[data-id]")) li.onclick = () => { select(li.dataset.id); vp.focus(li.dataset.id); };
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
  for (const li of $("#object-list").querySelectorAll("li[data-id]")) li.classList.toggle("sel", li.dataset.id === id);
}

function renderSelection() {
  const o = state.objects.find((x) => x.id === state.selected);
  $("#selection").hidden = !o;
  if (!o) return;
  $("#sel-info").innerHTML = `
    <div class="title">${esc(o.name)} <span class="dim">${esc(o.id)}</span></div>
    <div class="dim">${o.extent.w.toFixed(2)} × ${o.extent.d.toFixed(2)} × ${o.extent.h.toFixed(2)} m as placed · on ${esc(o.support)}${(state.room?.spaces || []).length > 1 ? ` · in ${esc(spaceName(o.space))}` : ""}</div>
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
$("#undo").onclick = () => guarded(async () => { const r = await api.undo(state.scene, state.variant); if (!r.undone) toast("Nothing to undo"); await refresh({ room: true }); });
$("#redo").onclick = () => guarded(async () => { const r = await api.redo(state.scene, state.variant); if (!r.redone) toast("Nothing to redo"); await refresh({ room: true }); });

$("#scene-select").onchange = (e) => { state.scene = e.target.value; state.variant = "main"; state.room = null; state.space = null; openScene(); };
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
  else if (e.key === "Escape") { if (vp.pickMode) { vp.cancelPick(); $("#space-hint").textContent = ""; } else select(null); }
  else if (e.key === "1") setView("orbit");
  else if (e.key === "2") setView("walk");
  else if (e.key === "3") setView("plan");
});

loadScenes().then(() => { if (params.get("view")) setView(params.get("view")); }).catch((e) => toast(e.message, "error"));
if (params.has("debug")) window.__mise = { vp, state }; // for scripted screenshots and poking from the console

// ---- empty home vs with built-ins ---------------------------------------------------
function setFixtures(on) {
  state.fixtures = on;
  for (const b of document.querySelectorAll("#fixtures-seg button")) b.classList.toggle("on", (b.dataset.fixtures === "1") === on);
  const url = new URL(location.href);
  if (on) url.searchParams.delete("fixtures"); else url.searchParams.set("fixtures", "0");
  history.replaceState(null, "", url);
  const sel = state.objects.find((o) => o.id === state.selected);
  if (sel?.fixture && !on) select(null);
  vp.setObjects(visible());
  renderObjectList();
}
for (const b of document.querySelectorAll("#fixtures-seg button")) b.onclick = () => setFixtures(b.dataset.fixtures === "1");
setFixtures(state.fixtures);
