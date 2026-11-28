// three.js + Spark viewport. World is z-up, meters, matching scene.json.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { SparkRenderer, SplatMesh, SplatEdit, SplatEditSdf } from "@sparkjsdev/spark";
import { extentOf, footprint, inAny, liveConflicts } from "./geometry.js";
import { EXTERIOR, doorLeaves, entryDoor, wallPieces } from "./walls3d.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const EYE = 1.6;
const PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f"];

export class Viewport {
  constructor(el, { onSelect, onMoveEnd, fileUrl, onFloorPick }) {
    this.el = el;
    this.onFloorPick = onFloorPick;
    this.pickMode = null; // { axis: "x" | "y" } while choosing where to split a space
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
    // soft, even interior light (physical intensities: ~pi gives surfaces their own color)
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xd9d2c5, 3.0));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
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
    this.outlineEdit = null;
    const polys = room.floor_polygons || [room.floor_polygon];
    this.polys = polys;
    this.floors = [];
    this.wallMeshes = [];
    const spaces = room.spaces || [];
    const finishOf = (id) => (spaces.find((s) => s.id === id) || {}).finish || {};
    // floors: one per space, in its finish (texture UVs are meters: the shape is in world x, y)
    const floorShapes = spaces.length ? spaces.map((s) => [s.polygon, s.finish?.floor]) : polys.map((p) => [p, null]);
    for (const [poly, surf] of floorShapes) {
      const shape = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
      const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape), this._surface(surf, "#d8ccb6", { transparent: !!room.splat, opacity: room.splat ? 0 : 1 }));
      floor.name = "floor";
      this.roomGroup.add(floor);
      this.floors.push(floor);
      // space outline, dashed: it reads as a zone line wherever no wall covers it
      const pts = [...poly, poly[0]].map(([x, y]) => new THREE.Vector3(x, y, 0.003));
      const dash = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineDashedMaterial({ color: 0x8a8070, dashSize: 0.12, gapSize: 0.08 }));
      dash.computeLineDistances();
      dash.userData.planOnly = true;
      this.roomGroup.add(dash);
    }
    if (spaces.length > 1) {
      for (const sp of spaces) {
        const xs = sp.polygon.map((p) => p[0]), ys = sp.polygon.map((p) => p[1]);
        const tag = this._label(`${sp.name} · ${sp.area_m2.toFixed(1)} m²`, "space-label");
        tag.element.dataset.space = sp.id;
        tag.position.set((Math.min(...xs) + Math.max(...xs)) / 2, Math.max(...ys) - 0.35, 0.02); // top edge, clear of most furniture
        tag.userData.notWalk = true;
        this.roomGroup.add(tag);
      }
      // ceilings, seen only when you walk around inside
      for (const sp of spaces) {
        const shape = new THREE.Shape(sp.polygon.map(([x, y]) => new THREE.Vector2(x, y)));
        const ceil = new THREE.Mesh(new THREE.ShapeGeometry(shape), this._surface(sp.finish?.ceiling, "#f2f0eb", { side: THREE.DoubleSide }));
        ceil.position.z = sp.ceiling || room.ceiling_height;
        ceil.userData.walkOnly = true;
        this.roomGroup.add(ceil);
      }
    }
    // solid walls with the openings cut through, in each space's wall finish
    const wallMats = new Map();
    const matsFor = (spaceId, fullHeight) => {
      // the cut top of a full-height wall reads dark from above (as on a plan); door heads and sills show the finish
      const key = `${spaceId}|${fullHeight}`;
      if (!wallMats.has(key)) {
        const face = this._surface(finishOf(spaceId).walls, "#f4f1ea");
        const top = fullHeight ? new THREE.MeshStandardMaterial({ color: 0x6b665e, roughness: 1 }) : face;
        const solid = [face, face, face, face, top, face];
        const faded = solid.map((m) => Object.assign(m.clone(), { transparent: true, opacity: 0.12, depthWrite: false }));
        wallMats.set(key, { solid, faded });
      }
      return wallMats.get(key);
    };
    const trim = new THREE.MeshStandardMaterial({ color: 0xe8e5de, roughness: 0.6 });
    for (const pc of wallPieces(room)) {
      const len = pc.s1 - pc.s0, h = pc.z1 - pc.z0;
      if (len < 0.005 || h < 0.005) continue;
      const geo = new THREE.BoxGeometry(len, pc.thick, h);
      // UVs in meters (so textures keep their real size) and continuous along the wall
      const uv = geo.attributes.uv, dims = [[h, pc.thick], [h, pc.thick], [len, h], [len, h], [len, pc.thick], [len, pc.thick]];
      dims.forEach(([du, dv], f) => {
        for (let k = f * 4; k < f * 4 + 4; k++) {
          const off = f === 2 || f === 3 ? [pc.s0, pc.z0] : [0, 0];
          uv.setXY(k, uv.getX(k) * du + off[0], uv.getY(k) * dv + off[1]);
        }
      });
      const mats = matsFor(pc.space, pc.z1 >= (room.walls.find((w) => w.id === pc.wall)?.height || room.ceiling_height) - 1e-3);
      const mesh = new THREE.Mesh(geo, room.splat ? mats.faded : mats.solid);
      const mid = (pc.s0 + pc.s1) / 2;
      const cx = pc.a[0] + pc.u[0] * mid - pc.n[0] * pc.thick / 2, cy = pc.a[1] + pc.u[1] * mid - pc.n[1] * pc.thick / 2;
      mesh.position.set(cx, cy, (pc.z0 + pc.z1) / 2);
      mesh.rotation.z = Math.atan2(pc.u[1], pc.u[0]);
      mesh.userData.wall = { out: [-pc.n[0], -pc.n[1]], c: [cx, cy], mats };
      mesh.userData.notPlan = pc.z0 > 1e-3; // plans are cut below door height: openings show as gaps
      this.roomGroup.add(mesh);
      this.wallMeshes.push(mesh);
      // baseboard along the room-side face
      const b0 = Math.max(pc.s0, 0), b1 = Math.min(pc.s1, pc.wallLen);
      if (pc.z0 < 1e-3 && b1 - b0 > 0.02 && !room.splat) {
        const bb = new THREE.Mesh(new THREE.BoxGeometry(b1 - b0, 0.012, 0.09), trim);
        const bm = (b0 + b1) / 2;
        bb.position.set(pc.a[0] + pc.u[0] * bm + pc.n[0] * 0.006, pc.a[1] + pc.u[1] * bm + pc.n[1] * 0.006, 0.045);
        bb.rotation.z = mesh.rotation.z;
        bb.userData.notPlan = true;
        this.roomGroup.add(bb);
      }
    }
    for (const w of room.walls) {
      const [ax, ay] = w.a, [bx, by] = w.b;
      const len = Math.hypot(bx - ax, by - ay);
      const edge = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(ax, ay, 0.004), new THREE.Vector3(bx, by, 0.004)]),
        new THREE.LineBasicMaterial({ color: 0x333333 }));
      this.roomGroup.add(edge);
      const tag = this._label(`W${w.id} · ${len.toFixed(2)} m`, "wall-label");
      const n = w.inward_normal || [0, 0];
      tag.position.set((ax + bx) / 2 - n[0] * 0.25, (ay + by) / 2 - n[1] * 0.25, 0.01);
      tag.userData.planOnly = true;
      this.roomGroup.add(tag);
    }
    // door leaves, open, and the doorways you can walk through between rooms
    for (const lf of doorLeaves(room)) {
      const leaf = new THREE.Mesh(new THREE.BoxGeometry(lf.width, lf.thick, lf.height),
        new THREE.MeshStandardMaterial({ color: lf.kind === "slide" ? 0x5d5953 : 0xeceae4, roughness: 0.55 }));
      leaf.position.set(lf.cx, lf.cy, lf.height / 2);
      leaf.rotation.z = lf.angle;
      leaf.userData.notPlan = true;
      this.roomGroup.add(leaf);
    }
    this.doorways = [];
    for (const op of room.openings) {
      const pts = op.polygon.map(([x, y, z]) => new THREE.Vector3(x, y, z));
      const color = op.kind === "window" ? 0x3c82c8 : 0xb46e28;
      const loop = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color }));
      loop.userData.planOnly = op.kind !== "window";
      this.roomGroup.add(loop);
      const base = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(pts[0].x, pts[0].y, 0.004), new THREE.Vector3(pts[1].x, pts[1].y, 0.004)]),
        new THREE.LineBasicMaterial({ color, linewidth: 3 }));
      base.userData.planOnly = true;
      this.roomGroup.add(base);
      const tag = this._label(op.id.replace("_", " "), "op-label");
      tag.position.set((pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2, op.kind === "window" ? op.top : 0.05);
      tag.userData.planOnly = true;
      this.roomGroup.add(tag);
      if (op.kind === "door") {
        const w = room.walls.find((x) => x.id === op.wall) || {};
        const n = w.inward_normal || [0, 0];
        const [x0, y0] = [pts[0].x, pts[0].y], [x1, y1] = [pts[1].x, pts[1].y], d = 0.4;
        // into the next room, or (outside doors) only across the wall's thickness
        const mid = [(x0 + x1) / 2 - n[0] * d, (y0 + y1) / 2 - n[1] * d];
        const dOut = inAny(mid, polys) ? d : EXTERIOR;
        const rect = [[x0 - n[0] * dOut, y0 - n[1] * dOut], [x1 - n[0] * dOut, y1 - n[1] * dOut], [x1 + n[0] * d, y1 + n[1] * d], [x0 + n[0] * d, y0 + n[1] * d]];
        this.doorways.push(rect);
        // threshold: floor across the wall's thickness, under the room floors elsewhere
        const sill = new THREE.Mesh(new THREE.ShapeGeometry(new THREE.Shape(rect.map(([x, y]) => new THREE.Vector2(x, y)))),
          this._surface(finishOf(w.space).floor, "#d8ccb6"));
        sill.position.z = -0.002;
        this.roomGroup.add(sill);
      }
    }
    this.wallCuts = [];
    if (room.splat) {
      const splat = new SplatMesh({ url: this.fileUrl(sceneId, room.splat) });
      // dollhouse for the splat: from outside, a wall between you and its room is cut away (a box per
      // wall that sets its Gaussians' opacity to 0; Gaussians show from both sides, so fading won't do)
      const edit = new SplatEdit({ rgbaBlendMode: "multiply" });
      for (const w of room.walls || []) {
        const [ax, ay] = w.a, [bx, by] = w.b;
        const len = Math.hypot(bx - ax, by - ay);
        if (len < 0.05) continue;
        const h = w.height || room.ceiling_height;
        const box = new SplatEditSdf({ type: "box", opacity: 0 });
        box.position.set((ax + bx) / 2, (ay + by) / 2, h / 2);
        box.rotation.z = Math.atan2(by - ay, bx - ax);
        box.scale.set(len / 2 - 0.01, 0.07, h / 2 + 0.1); // half sizes
        box.visible = false;
        edit.add(box);
        const n = w.inward_normal || [0, 0];
        this.wallCuts.push({ box, out: [-n[0], -n[1]], c: [(ax + bx) / 2, (ay + by) / 2] });
      }
      splat.add(edit);
      this.roomGroup.add(splat);
    }
    if (room.ceiling_splat) {  // ceilings only when walking through: from above they'd hide the rooms
      const ceil = new SplatMesh({ url: this.fileUrl(sceneId, room.ceiling_splat) });
      ceil.userData.walkOnly = true;
      this.roomGroup.add(ceil);
    }
    const [[minX, minY], [maxX, maxY]] = [room.bounds.min, room.bounds.max];
    this.bounds = { minX, maxX, minY, maxY };
    this.homeBounds = { ...this.bounds };
    this.splitLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0x1f6feb }));
    this.splitLine.visible = false;
    this.roomGroup.add(this.splitLine);
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

  /** A material for a floor, wall or ceiling finish: { color, texture, size_m } (texture in meters). */
  _surface(surf, fallback, extra = {}) {
    const opts = { color: surf?.color || fallback, roughness: 0.92, ...extra };
    if (surf?.texture && this.sceneId) {
      this._textures = this._textures || new Map();
      const key = `${this.sceneId}/${surf.texture}@${surf.size_m || 1}`;
      if (!this._textures.has(key)) {
        const t = new THREE.TextureLoader().load(this.fileUrl(this.sceneId, surf.texture));
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.colorSpace = THREE.SRGBColorSpace;
        t.anisotropy = 8;
        t.repeat.set(1 / (surf.size_m || 1), 1 / (surf.size_m || 1));
        this._textures.set(key, t);
      }
      opts.map = this._textures.get(key);
      opts.color = 0xffffff; // the texture carries the color
    }
    return new THREE.MeshStandardMaterial(opts);
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
      const entry = entryDoor(this.room, (p) => inAny(p, this.polys));
      if (entry) {
        this.walkPos.set(entry.c[0] + entry.n[0] * 0.6, entry.c[1] + entry.n[1] * 0.6, EYE);
        this.walk.yaw = Math.atan2(entry.n[1], entry.n[0]);
      } else {
        this.walkPos.set(cx, cy, EYE);
        this.walk.yaw = 0;
      }
      this.walk.pitch = -0.25;
    }
    if (mode === "orbit") {
      this.persp.position.set(this.persp.position.x, this.persp.position.y, Math.max(this.persp.position.z, 2.5));
    }
    this.el.classList.toggle("plan", mode === "plan");
  }

  _insideRoom(v) {
    return inAny([v.x, v.y], this.polys) || inAny([v.x, v.y], this.doorways || []);
  }

  /** Frame one space (or the whole home with null) in plan and orbit views. */
  focusSpace(space) {
    if (!space) this.bounds = { ...this.homeBounds };
    else {
      const xs = space.polygon.map((p) => p[0]), ys = space.polygon.map((p) => p[1]);
      this.bounds = { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
    }
    const { minX, maxX, minY, maxY } = this.bounds;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, size = Math.max(maxX - minX, maxY - minY);
    this.controls.target.set(cx, cy, 0.8);
    if (this.mode === "orbit") this.persp.position.set(cx - 0.3 * size, cy - 1.1 * size - 1, Math.max(3, size * 1.1));
    this._fitOrtho();
    for (const el of this.el.querySelectorAll(".space-label")) el.classList.toggle("sel", !!space && el.dataset.space === space.id);
  }

  /** Show draggable corner and edge handles for one space's outline (plan view). */
  editOutline(space, onChange) {
    if (this.outlineDrag) return; // a live update arrived mid-drag: keep the handles you hold
    if (this.outlineEdit) { this.roomGroup.remove(this.outlineEdit.group); this.outlineEdit = null; }
    if (!space) return;
    const group = new THREE.Group();
    const poly = space.polygon.map((p) => [...p]);
    const handles = [];
    const mk = (color, size) => new THREE.Mesh(new THREE.BoxGeometry(size, size, 0.02), new THREE.MeshBasicMaterial({ color }));
    poly.forEach((p, i) => {
      const h = mk(0x1f6feb, 0.16);
      h.userData = { kind: "vertex", i };
      handles.push(h);
      group.add(h);
      const e = mk(0xffffff, 0.12);
      e.material.color.set(0x1f6feb); e.material.transparent = true; e.material.opacity = 0.55;
      e.userData = { kind: "edge", i };
      handles.push(e);
      group.add(e);
    });
    const line = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x1f6feb }));
    group.add(line);
    this.roomGroup.add(group);
    this.outlineEdit = { space, poly, handles, line, group, onChange };
    this._layoutOutline();
  }

  _layoutOutline() {
    const { poly, handles, line } = this.outlineEdit;
    for (const h of handles) {
      const { kind, i } = h.userData;
      const a = poly[i], b = poly[(i + 1) % poly.length];
      if (kind === "vertex") h.position.set(a[0], a[1], 0.05);
      else h.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.05);
    }
    line.geometry.setFromPoints(poly.map(([x, y]) => new THREE.Vector3(x, y, 0.04)));
  }

  _outlineDown(e) {
    if (!this.outlineEdit) return false;
    this._ray(e);
    const hit = this.raycaster.intersectObjects(this.outlineEdit.handles, false)[0];
    if (!hit) return false;
    const p = this._floorPoint(e);
    this.outlineDrag = { ...hit.object.userData, start: p, poly0: this.outlineEdit.poly.map((q) => [...q]) };
    this.controls.enabled = false;
    this.renderer.domElement.setPointerCapture(e.pointerId);
    return true;
  }

  _outlineMove(e) {
    const d = this.outlineDrag;
    const p = this._floorPoint(e);
    if (!p) return;
    const snap = (v) => Math.round(v * 20) / 20;
    const poly = this.outlineEdit.poly, n = poly.length;
    const dx = p.x - d.start.x, dy = p.y - d.start.y;
    if (d.kind === "vertex") {
      poly[d.i] = [snap(d.poly0[d.i][0] + dx), snap(d.poly0[d.i][1] + dy)];
    } else {
      // slide the edge along its normal, keeping its direction (walls stay square)
      const a = d.poly0[d.i], b = d.poly0[(d.i + 1) % n];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      const nx = -(b[1] - a[1]) / len, ny = (b[0] - a[0]) / len;
      let t = dx * nx + dy * ny;
      if (Math.abs(nx) < 1e-6 || Math.abs(ny) < 1e-6) t = Math.round(t * 20) / 20;
      poly[d.i] = [a[0] + nx * t, a[1] + ny * t];
      poly[(d.i + 1) % n] = [b[0] + nx * t, b[1] + ny * t];
    }
    this._layoutOutline();
  }

  _outlineUp() {
    const d = this.outlineDrag;
    this.outlineDrag = null;
    this.controls.enabled = this.mode === "orbit";
    const poly = this.outlineEdit.poly;
    const changed = poly.some((q, i) => q[0] !== d.poly0[i][0] || q[1] !== d.poly0[i][1]);
    if (changed) this.outlineEdit.onChange(poly.map(([x, y]) => [Math.round(x * 1000) / 1000, Math.round(y * 1000) / 1000]));
  }

  /** Next click on the floor reports a point; a guide line shows where an x/y split would go. */
  startPick(axis) { this.pickMode = { axis }; this.el.classList.add("picking"); }
  cancelPick() { this.pickMode = null; this.splitLine.visible = false; this.el.classList.remove("picking"); }

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
    this.roomGroup.traverse((o) => {
      if (o.userData.planOnly) o.visible = this.mode === "plan";
      else if (o.userData.walkOnly) o.visible = this.mode === "walk";
      else if (o.userData.notPlan) o.visible = this.mode !== "plan";
      else if (o.userData.notWalk) o.visible = this.mode !== "walk";
    });
    // dollhouse: from outside, walls between you and the rooms fade so you can see in
    const cam = this.camera.position;
    for (const { box, out, c } of this.wallCuts || []) {
      box.visible = this.mode === "orbit" && out[0] * (cam.x - c[0]) + out[1] * (cam.y - c[1]) > 0;
    }
    for (const m of this.wallMeshes || []) {
      const { out, c, mats } = m.userData.wall;
      const near = this.mode === "orbit" && out[0] * (cam.x - c[0]) + out[1] * (cam.y - c[1]) > 0;
      const want = near || this.room?.splat ? mats.faded : mats.solid;
      if (m.material !== want) m.material = want;
    }
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
    if (this.pickMode) {
      const p = this._floorPoint(e);
      const axis = this.pickMode.axis;
      this.cancelPick();
      if (p) this.onFloorPick?.({ x: p.x, y: p.y, axis });
      return;
    }
    if (this._outlineDown(e)) return;
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
    if (this.outlineDrag) { this._outlineMove(e); return; }
    if (this.pickMode) {
      const p = this._floorPoint(e);
      if (!p) return;
      const { minX, maxX, minY, maxY } = this.homeBounds;
      const at = Math.round((this.pickMode.axis === "x" ? p.x : p.y) * 20) / 20; // 5 cm steps
      const a = this.pickMode.axis === "x" ? [at, minY - 0.3] : [minX - 0.3, at];
      const b = this.pickMode.axis === "x" ? [at, maxY + 0.3] : [maxX + 0.3, at];
      this.splitLine.geometry.setFromPoints([new THREE.Vector3(a[0], a[1], 0.02), new THREE.Vector3(b[0], b[1], 0.02)]);
      this.splitLine.visible = true;
      this.pickMode.at = at;
      return;
    }
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
    const hits = liveConflicts(rec.data, others, this.polys);
    this._restyle(new Set(hits.length ? [rec.data.id, ...hits.map((h) => h.id)] : []));
  }

  _up() {
    if (this.outlineDrag) { this._outlineUp(); return; }
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
