// three.js + Spark viewport. World is z-up, meters, matching scene.json.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { extentOf, footprint, liveConflicts } from "./geometry.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const EYE = 1.6;
const PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f"];

export class Viewport {
  constructor(el, { onSelect, onMoveEnd, fileUrl }) {
    this.el = el;
    this.onSelect = onSelect;
    this.onMoveEnd = onMoveEnd;
    this.fileUrl = fileUrl;
    this.objects = new Map(); // id -> { data, group, box, edges, label }
    this.room = null;
    this.flagged = new Set();
    this.mode = "orbit";

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);
    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = "labels";
    el.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(getComputedStyle(document.body).getPropertyValue("--canvas").trim() || "#ece9e2");
    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7f70, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(3, -2, 6);
    this.scene.add(sun);

    this.persp = new THREE.PerspectiveCamera(60, 1, 0.02, 200);
    this.persp.position.set(-2, -3, 4);
    this.ortho = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
    this.ortho.position.set(0, 0, 30);
    this.ortho.up.set(0, 1, 0);
    this.camera = this.persp;
    this.controls = new OrbitControls(this.persp, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.roomGroup = new THREE.Group();
    this.objGroup = new THREE.Group();
    this.scene.add(this.roomGroup, this.objGroup);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.drag = null;
    this.walk = { yaw: 0, pitch: 0, keys: new Set(), locked: false };
    this.gltf = new GLTFLoader();

    const dom = this.renderer.domElement;
    dom.addEventListener("pointerdown", (e) => this._down(e));
    dom.addEventListener("pointermove", (e) => this._move(e));
    window.addEventListener("pointerup", (e) => this._up(e));
    dom.addEventListener("click", () => { if (this.mode === "walk" && !this.walk.locked) dom.requestPointerLock(); });
    document.addEventListener("pointerlockchange", () => { this.walk.locked = document.pointerLockElement === dom; });
    document.addEventListener("mousemove", (e) => {
      if (this.mode !== "walk" || !this.walk.locked) return;
      this.walk.yaw -= e.movementX * 0.0025;
      this.walk.pitch = Math.max(-1.3, Math.min(1.3, this.walk.pitch - e.movementY * 0.0025));
    });
    window.addEventListener("keydown", (e) => { if (this.mode === "walk") this.walk.keys.add(e.code); });
    window.addEventListener("keyup", (e) => this.walk.keys.delete(e.code));
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    this.timer = new THREE.Timer();
    this.renderer.setAnimationLoop(() => this._tick());
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    this.persp.aspect = w / h;
    this.persp.updateProjectionMatrix();
    this._fitOrtho();
  }

  // ---- room -------------------------------------------------------------------
  setRoom(sceneId, room) {
    this.sceneId = sceneId;
    this.room = room;
    this.roomGroup.traverse((o) => { if (o.isCSS2DObject) o.element.remove(); });
    this.roomGroup.clear();
    const poly = room.floor_polygon;
    const shape = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
    const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape),
      new THREE.MeshStandardMaterial({ color: 0xd8ccb6, roughness: 0.9, transparent: !!room.splat, opacity: room.splat ? 0.0 : 1 }));
    floor.name = "floor";
    this.roomGroup.add(floor);
    this.floor = floor;
    const H = room.ceiling_height;
    for (const w of room.walls) {
      const [ax, ay] = w.a, [bx, by] = w.b;
      const len = Math.hypot(bx - ax, by - ay);
      const wall = new THREE.Mesh(new THREE.PlaneGeometry(len, H),
        new THREE.MeshStandardMaterial({ color: 0xf4f1ea, side: THREE.BackSide, transparent: true, opacity: room.splat ? 0 : 0.85 }));
      wall.position.set((ax + bx) / 2, (ay + by) / 2, H / 2);
      // Plane normal ends up (sin a, -cos a); point it outward so BackSide hides walls between you and the room.
      let ang = Math.atan2(by - ay, bx - ax);
      const n0 = w.inward_normal || [0, 1];
      if (Math.sin(ang) * n0[0] - Math.cos(ang) * n0[1] > 0) ang += Math.PI;
      wall.rotation.set(Math.PI / 2, 0, ang, "ZXY");
      this.roomGroup.add(wall);
      const edge = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(ax, ay, 0.002), new THREE.Vector3(bx, by, 0.002)]),
        new THREE.LineBasicMaterial({ color: 0x333333 }));
      this.roomGroup.add(edge);
      const tag = this._label(`W${w.id} · ${len.toFixed(2)} m`, "wall-label");
      const n = w.inward_normal || [0, 0];
      tag.position.set((ax + bx) / 2 - n[0] * 0.25, (ay + by) / 2 - n[1] * 0.25, 0.01);
      tag.userData.planOnly = true;
      this.roomGroup.add(tag);
    }
    for (const op of room.openings) {
      const pts = op.polygon.map(([x, y, z]) => new THREE.Vector3(x, y, z));
      const color = op.kind === "window" ? 0x3c82c8 : 0xb46e28;
      this.roomGroup.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color })));
      const base = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(pts[0].x, pts[0].y, 0.004), new THREE.Vector3(pts[1].x, pts[1].y, 0.004)]),
        new THREE.LineBasicMaterial({ color, linewidth: 3 }));
      this.roomGroup.add(base);
      const tag = this._label(op.id.replace("_", " "), "op-label");
      tag.position.set((pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2, op.kind === "window" ? op.top : 0.05);
      this.roomGroup.add(tag);
    }
    if (room.splat) {
      const splat = new SplatMesh({ url: this.fileUrl(sceneId, room.splat) });
      this.roomGroup.add(splat);
    }
    const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
    this.bounds = { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
    const cx = (this.bounds.minX + this.bounds.maxX) / 2, cy = (this.bounds.minY + this.bounds.maxY) / 2;
    this.controls.target.set(cx, cy, 0.8);
    if (!this._placedCamera) {
      this.persp.position.set(cx - 0.2 * (this.bounds.maxX - this.bounds.minX), this.bounds.minY - 2.5, 4.2);
      this._placedCamera = true;
    }
    this.walkPos = new THREE.Vector3(cx, cy, EYE);
    this._fitOrtho();
  }

  _fitOrtho() {
    if (!this.bounds) return;
    const { minX, maxX, minY, maxY } = this.bounds;
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    const pad = 0.8, rw = maxX - minX + 2 * pad, rh = maxY - minY + 2 * pad;
    const scale = Math.max(rw / w, rh / h);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    Object.assign(this.ortho, { left: -w * scale / 2, right: w * scale / 2, top: h * scale / 2, bottom: -h * scale / 2 });
    this.ortho.position.set(cx, cy, 30);
    this.ortho.lookAt(cx, cy, 0);
    this.ortho.updateProjectionMatrix();
  }

  _label(text, cls) {
    const div = document.createElement("div");
    div.className = cls;
    div.textContent = text;
    return new CSS2DObject(div);
  }

  // ---- objects ----------------------------------------------------------------
  setObjects(list) {
    const seen = new Set();
    list.forEach((o, i) => {
      seen.add(o.id);
      let rec = this.objects.get(o.id);
      const key = JSON.stringify([o.asset, o.asset_kind, o.dims, o.asset_transform, o.color]);
      if (rec && rec.key !== key) { this._dispose(rec); rec = null; }
      if (!rec) rec = this._build(o, i, key);
      rec.data = o;
      rec.index = i;
      this._applyPose(rec);
      const ext = extentOf(o);
      rec.label.element.innerHTML = `<b>${o.label ?? i + 1}</b> ${o.name}<span class="dim">${ext.w.toFixed(2)}×${ext.d.toFixed(2)}×${ext.h.toFixed(2)} m</span>`
        + (o.warnings?.some((w) => w.startsWith("size mismatch")) ? `<span class="warnbadge" title="${o.warnings.join("\n")}">size ±</span>` : "")
        + (o.locked ? `<span class="lockbadge">locked</span>` : "");
    });
    for (const [id, rec] of this.objects) if (!seen.has(id)) { this._dispose(rec); this.objects.delete(id); }
    this._restyle();
  }

  _dispose(rec) {
    // CSS2D label elements live outside the WebGL scene; remove them explicitly.
    rec.group.traverse((o) => { if (o.isCSS2DObject) o.element.remove(); });
    this.objGroup.remove(rec.group);
  }

  _build(o, i, key) {
    // group: position + yaw.  tip: roll/pitch about the box center.  body: the canonical-frame object.
    const group = new THREE.Group();
    const tip = new THREE.Group();
    const body = new THREE.Group();
    body.position.z = -o.dims.h / 2;
    tip.add(body);
    group.add(tip);
    const geo = new THREE.BoxGeometry(o.dims.w, o.dims.d, o.dims.h);
    geo.translate(0, 0, o.dims.h / 2);
    const isBox = o.asset_kind === "box" || !o.asset;
    const color = new THREE.Color(o.color || PALETTE[i % PALETTE.length]);
    const box = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color, roughness: 0.8, transparent: true, opacity: isBox ? (o.category === "rug" ? 0.6 : 0.92) : 0.0, depthWrite: isBox,
    }));
    box.userData.id = o.id;
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x222222, transparent: true, opacity: isBox ? 0.6 : 0.0 }));
    // front marker: a thin bar on the +y face
    const front = new THREE.Mesh(new THREE.BoxGeometry(o.dims.w * 0.6, 0.02, 0.02), new THREE.MeshBasicMaterial({ color: 0x111111 }));
    front.position.set(0, o.dims.d / 2, 0.012);
    body.add(box, edges, front);
    if (o.asset_kind === "splat" && o.asset) {
      body.add(new SplatMesh({ url: this.fileUrl(this.sceneId, o.asset) }));
    } else if (o.asset_kind === "mesh" && o.asset) {
      this.gltf.load(this.fileUrl(this.sceneId, o.asset), (g) => {
        const holder = new THREE.Group();
        if (o.asset_transform) {
          const m = new THREE.Matrix4();
          m.set(...o.asset_transform.flat()); // row-major
          holder.applyMatrix4(m);
        }
        holder.add(g.scene);
        body.add(holder);
      });
    }
    const label = this._label("", "obj-label");
    label.position.set(0, 0, o.dims.h + 0.08);
    group.add(label);
    this.objGroup.add(group);
    const rec = { data: o, group, tip, box, edges, label, key, isBox };
    this.objects.set(o.id, rec);
    return rec;
  }

  _applyPose(rec) {
    const p = rec.data.pose;
    rec.group.position.set(p.x, p.y, p.z);
    rec.group.rotation.set(0, 0, (p.yaw * Math.PI) / 180);
    const ext = extentOf(rec.data);
    rec.tip.position.z = ext.h / 2;
    rec.tip.rotation.set(((p.pitch || 0) * Math.PI) / 180, ((p.roll || 0) * Math.PI) / 180, 0, "XYZ");
    rec.label.position.z = ext.h + 0.08;
  }

  setFlagged(ids) { this.flagged = new Set(ids); this._restyle(); }

  select(id) { this.selected = id; this._restyle(); }

  _restyle(live = new Set()) {
    for (const [id, rec] of this.objects) {
      const sel = id === this.selected, bad = this.flagged.has(id) || live.has(id);
      rec.edges.material.color.set(bad ? 0xd62828 : sel ? 0x1f6feb : 0x222222);
      rec.edges.material.opacity = sel || bad ? 1 : rec.isBox ? 0.6 : 0.0;
      rec.label.element.classList.toggle("sel", sel);
      rec.label.element.classList.toggle("bad", bad);
    }
  }

  // ---- views ------------------------------------------------------------------
  setMode(mode) {
    this.mode = mode;
    if (document.pointerLockElement) document.exitPointerLock();
    this.camera = mode === "plan" ? this.ortho : this.persp;
    this.controls.enabled = mode === "orbit";
    if (mode === "walk") {
      // Start in the first doorway looking at the room's center, as you would walk in.
      const { minX, maxX, minY, maxY } = this.bounds;
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      const door = this.room.openings.find((o) => o.kind === "door");
      const wall = door && this.room.walls.find((w) => w.id === door.wall);
      if (door && wall) {
        const [x0, y0] = door.polygon[0], [x1, y1] = door.polygon[1];
        const n = wall.inward_normal;
        this.walkPos.set((x0 + x1) / 2 + n[0] * 0.35, (y0 + y1) / 2 + n[1] * 0.35, EYE);
      } else {
        this.walkPos.set(cx, cy, EYE);
      }
      this.walk.yaw = Math.atan2(cy - this.walkPos.y, cx - this.walkPos.x);
      this.walk.pitch = -0.25;
    }
    if (mode === "orbit") {
      this.persp.position.set(this.persp.position.x, this.persp.position.y, Math.max(this.persp.position.z, 2.5));
    }
    this.el.classList.toggle("plan", mode === "plan");
  }

  _insideRoom(v) {
    const poly = this.room.floor_polygon;
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > v.y) !== (yj > v.y) && v.x < ((xj - xi) * (v.y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  _tick() {
    this.timer.update();
    const dt = Math.min(this.timer.getDelta(), 0.05);
    if (this.mode === "walk") {
      const k = this.walk.keys, sp = 1.4 * dt;
      const f = new THREE.Vector3(Math.cos(this.walk.yaw), Math.sin(this.walk.yaw), 0);
      const r = new THREE.Vector3(Math.sin(this.walk.yaw), -Math.cos(this.walk.yaw), 0);
      const next = this.walkPos.clone();
      if (k.has("KeyW") || k.has("ArrowUp")) next.addScaledVector(f, sp);
      if (k.has("KeyS") || k.has("ArrowDown")) next.addScaledVector(f, -sp);
      if (k.has("KeyD") || k.has("ArrowRight")) next.addScaledVector(r, sp);
      if (k.has("KeyA") || k.has("ArrowLeft")) next.addScaledVector(r, -sp);
      if (!this.room || this._insideRoom(next)) this.walkPos.copy(next);
      this.persp.position.copy(this.walkPos);
      const look = new THREE.Vector3(Math.cos(this.walk.yaw) * Math.cos(this.walk.pitch), Math.sin(this.walk.yaw) * Math.cos(this.walk.pitch), Math.sin(this.walk.pitch));
      this.persp.lookAt(this.walkPos.clone().add(look));
    } else if (this.mode === "orbit") {
      this.controls.update();
    }
    for (const rec of this.objects.values()) rec.label.visible = true;
    this.roomGroup.traverse((o) => { if (o.userData.planOnly) o.visible = this.mode === "plan"; });
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }

  // ---- picking and dragging -----------------------------------------------------
  _ray(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
  }

  _pick(e) {
    this._ray(e);
    const boxes = [...this.objects.values()].map((r) => r.box);
    const hits = this.raycaster.intersectObjects(boxes, false);
    // prefer things that are not rugs when stacked
    hits.sort((a, b) => (this.objects.get(a.object.userData.id).data.category === "rug") - (this.objects.get(b.object.userData.id).data.category === "rug") || a.distance - b.distance);
    return hits[0]?.object.userData.id ?? null;
  }

  _floorPoint(e, z = 0) {
    this._ray(e);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -z);
    const p = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(plane, p) ? p : null;
  }

  _down(e) {
    if (this.mode === "walk" || e.button !== 0) return;
    const id = this._pick(e);
    this.onSelect(id);
    if (!id) return;
    const rec = this.objects.get(id);
    if (rec.data.support !== "floor" && !rec.data.support.startsWith("wall")) {
      // stacked objects move with their support; still allow nudging on it
    }
    const p = this._floorPoint(e, rec.data.pose.z);
    if (!p) return;
    this.drag = { id, start: p, pose0: { ...rec.data.pose }, moved: false };
    this.controls.enabled = false;
    this.renderer.domElement.setPointerCapture(e.pointerId);
  }

  _move(e) {
    if (!this.drag) return;
    const rec = this.objects.get(this.drag.id);
    const p = this._floorPoint(e, this.drag.pose0.z);
    if (!p) return;
    rec.data.pose.x = this.drag.pose0.x + (p.x - this.drag.start.x);
    rec.data.pose.y = this.drag.pose0.y + (p.y - this.drag.start.y);
    this.drag.moved = Math.hypot(p.x - this.drag.start.x, p.y - this.drag.start.y) > 0.005;
    this._applyPose(rec);
    // carry dependents visually
    for (const dep of this.objects.values()) {
      if (dep.data.support === rec.data.id) {
        dep.group.position.x = dep.data.pose.x + (rec.data.pose.x - this.drag.pose0.x);
        dep.group.position.y = dep.data.pose.y + (rec.data.pose.y - this.drag.pose0.y);
      }
    }
    const others = [...this.objects.values()].map((r) => r.data);
    const hits = liveConflicts(rec.data, others, this.room.floor_polygon);
    this._restyle(new Set(hits.length ? [rec.data.id, ...hits.map((h) => h.id)] : []));
  }

  _up() {
    if (!this.drag) return;
    const d = this.drag;
    this.drag = null;
    this.controls.enabled = this.mode === "orbit";
    if (d.moved) {
      const rec = this.objects.get(d.id);
      this.onMoveEnd(d.id, { x: rec.data.pose.x, y: rec.data.pose.y });
    }
  }

  focus(id) {
    const rec = this.objects.get(id);
    if (!rec || this.mode !== "orbit") return;
    this.controls.target.set(rec.data.pose.x, rec.data.pose.y, rec.data.pose.z + extentOf(rec.data).h / 2);
  }

  footprintOf(id) {
    const rec = this.objects.get(id);
    return rec ? footprint(rec.data.pose, extentOf(rec.data)) : null;
  }
}
