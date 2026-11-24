// Solid walls for the 3D view: thickness, openings cut through, door leaves. Pure math, no three.js.
//
// Each space's walls sit on its outline (the room-side face). A wall is extruded away from its
// room by half the gap to the facing wall of the next space (so two rooms' walls meet in the
// middle of the partition), or by EXTERIOR where nothing faces it.

export const EXTERIOR = 0.12;
const MAX_GAP = 0.4;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const near = (a, b, eps = 1e-3) => Math.abs(a[0] - b[0]) < eps && Math.abs(a[1] - b[1]) < eps;

function frame(w) {
  const d = sub(w.b, w.a), len = Math.hypot(d[0], d[1]);
  return { len, u: [d[0] / len, d[1] / len], n: w.inward_normal || [0, 0] };
}

/** How far each wall extends away from its room. */
export function wallThickness(walls) {
  const out = new Map();
  for (const w of walls) {
    const f = frame(w);
    const o = [-f.n[0], -f.n[1]];
    let best = null;
    for (const v of walls) {
      if (v === w || v.space === w.space) continue;
      const g = frame(v);
      if (dot(g.n, o) < 0.99) continue; // the facing wall's room lies beyond it, so its normal points our way out
      const gap = dot(sub(v.a, w.a), o);
      if (gap <= 0.005 || gap > MAX_GAP) continue;
      const s0 = dot(sub(v.a, w.a), f.u), s1 = dot(sub(v.b, w.a), f.u);
      const overlap = Math.min(f.len, Math.max(s0, s1)) - Math.max(0, Math.min(s0, s1));
      if (overlap > 0.3 * f.len && (best === null || gap < best)) best = gap;
    }
    out.set(w.id, best === null ? EXTERIOR : best / 2);
  }
  return out;
}

function convexAt(poly, p, dir) {
  // is p a convex corner of the CCW polygon? dir: +1 at the wall's end (b), -1 at its start (a)
  const k = poly.findIndex((q) => near(q, p));
  if (k < 0) return false;
  const prev = poly[(k - 1 + poly.length) % poly.length], next = poly[(k + 1) % poly.length];
  const e1 = sub(p, prev), e2 = sub(next, p);
  return e1[0] * e2[1] - e1[1] * e2[0] > 1e-9;
}

/**
 * Solid boxes for every wall: [{ wall, space, a, u, n, s0, s1, z0, z1, thick, wallLen }]
 * s along the wall from a (may run past the ends at convex corners), z up from the floor.
 */
export function wallPieces(room) {
  const thick = wallThickness(room.walls);
  const polys = new Map((room.spaces || []).map((s) => [s.id, s.polygon]));
  const fallback = room.floor_polygon || (room.floor_polygons || [])[0];
  const pieces = [];
  for (const w of room.walls) {
    const { len, u, n } = frame(w);
    const t = thick.get(w.id);
    const H = w.height || room.ceiling_height;
    const poly = polys.get(w.space) || fallback || [];
    // close the outside corner: run on by the thickness where the outline turns convex
    const ext0 = convexAt(poly, w.a, -1) ? t : 0, ext1 = convexAt(poly, w.b, 1) ? t : 0;
    const ops = room.openings.filter((o) => o.wall === w.id)
      .map((o) => ({ s0: Math.max(0, o.center - o.width / 2), s1: Math.min(len, o.center + o.width / 2), z0: o.bottom || 0, z1: Math.min(o.top, H) }))
      .sort((p, q) => p.s0 - q.s0);
    const base = { wall: w.id, space: w.space, a: w.a, u, n, thick: t, wallLen: len };
    let s = -ext0;
    for (const o of ops) {
      if (o.s0 > s + 1e-3) pieces.push({ ...base, s0: s, s1: o.s0, z0: 0, z1: H });
      if (o.z0 > 1e-3) pieces.push({ ...base, s0: o.s0, s1: o.s1, z0: 0, z1: o.z0 });
      if (o.z1 < H - 1e-3) pieces.push({ ...base, s0: o.s0, s1: o.s1, z0: o.z1, z1: H });
      s = Math.max(s, o.s1);
    }
    if (len + ext1 > s + 1e-3) pieces.push({ ...base, s0: s, s1: len + ext1, z0: 0, z1: H });
  }
  return pieces;
}

/**
 * Door leaves, one per physical door (a door recorded on both sides of a partition is drawn once):
 * [{ id, kind: "swing" | "slide", cx, cy, angle, width, height, thick }] -- a box centered at (cx, cy),
 * `width` along `angle` (radians from +x), `height` up from the floor.
 */
export function doorLeaves(room) {
  const thick = wallThickness(room.walls);
  const walls = new Map(room.walls.map((w) => [w.id, w]));
  const leaves = [], seen = [];
  for (const op of room.openings) {
    if (op.kind !== "door" || !["in", "out", "sliding"].includes(op.swing)) continue;
    const w = walls.get(op.wall);
    if (!w) continue;
    const { u, n } = frame(w);
    const c = [w.a[0] + u[0] * op.center, w.a[1] + u[1] * op.center];
    if (seen.some((p) => Math.hypot(p[0] - c[0], p[1] - c[1]) < 0.4)) continue;
    seen.push(c);
    const lt = 0.035, h = op.top - 0.01;
    if (op.swing === "sliding") {
      // a barn door parked beside its opening (on the side with more wall), hung just off the room-side face
      const { len } = frame(w);
      const side = len - (op.center + op.width / 2) >= op.center - op.width / 2 ? 1 : -1;
      const along = op.center + side * op.width * 0.55;
      leaves.push({ id: op.id, kind: "slide", cx: w.a[0] + u[0] * along + n[0] * 0.05, cy: w.a[1] + u[1] * along + n[1] * 0.05,
        angle: Math.atan2(u[1], u[0]), width: op.width + 0.08, height: op.top + 0.03, thick: lt });
      continue;
    }
    // hinge as the editor and the checks describe it: seen from inside the room, facing the wall
    const left = [n[1], -n[0]];
    const hd = op.hinge === "left" ? left : [-left[0], -left[1]];
    const sn = op.swing === "out" ? [-n[0], -n[1]] : n;
    // an outswing door hangs on the far face: the whole partition (both rooms' halves) or the outside wall
    const t = thick.get(w.id) ?? EXTERIOR;
    const face = op.swing === "out" ? (t < EXTERIOR - 1e-6 ? 2 * t : t) : 0;
    const hinge = [c[0] + hd[0] * op.width / 2 + sn[0] * face, c[1] + hd[1] * op.width / 2 + sn[1] * face];
    // open 90°: the leaf stands along the swing direction, just inside the jamb
    const leafW = op.width - 0.02;
    leaves.push({ id: op.id, kind: "swing",
      cx: hinge[0] + sn[0] * leafW / 2 - hd[0] * lt / 2, cy: hinge[1] + sn[1] * leafW / 2 - hd[1] * lt / 2,
      angle: Math.atan2(sn[1], sn[0]), width: leafW, height: h, thick: lt });
  }
  return leaves;
}

/** The door whose outside is not any room: where a visitor walks in. */
export function entryDoor(room, inside) {
  const walls = new Map(room.walls.map((w) => [w.id, w]));
  for (const op of room.openings) {
    if (op.kind !== "door") continue;
    const w = walls.get(op.wall);
    if (!w) continue;
    const { u, n } = frame(w);
    const c = [w.a[0] + u[0] * op.center, w.a[1] + u[1] * op.center];
    if (!inside([c[0] - n[0] * 0.5, c[1] - n[1] * 0.5])) return { c, n };
  }
  return null;
}
