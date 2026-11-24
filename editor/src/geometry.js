// Client-side footprint math for live feedback while dragging.
// The backend's check_layout is authoritative; this only colors boxes mid-drag.

export function footprint(pose, dims, pad = 0) {
  const a = (pose.yaw * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  const hw = dims.w / 2 + pad, hd = dims.d / 2 + pad;
  return [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]].map(([x, y]) => [pose.x + c * x - s * y, pose.y + s * x + c * y]);
}

function axes(p) {
  const out = [];
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length];
    const nx = -(y2 - y1), ny = x2 - x1, n = Math.hypot(nx, ny);
    if (n > 1e-12) out.push([nx / n, ny / n]);
  }
  return out;
}

export function penetration(p, q) {
  let best = Infinity;
  for (const [ax, ay] of [...axes(p), ...axes(q)]) {
    const pp = p.map(([x, y]) => x * ax + y * ay), qq = q.map(([x, y]) => x * ax + y * ay);
    best = Math.min(best, Math.min(Math.max(...pp), Math.max(...qq)) - Math.max(Math.min(...pp), Math.min(...qq)));
  }
  return best;
}

export function pointInPolygon([x, y], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

const IGNORED = new Set(["rug", "mat"]);

/** Ids of objects that overlap `obj` (footprint + height) or stick out of the room. */
export function liveConflicts(obj, objects, floorPolygon) {
  const fp = footprint(obj.pose, obj.dims);
  const hits = [];
  if (IGNORED.has(obj.category)) return hits;
  for (const o of objects) {
    if (o.id === obj.id || IGNORED.has(o.category) || o.support === obj.id || obj.support === o.id) continue;
    const zo = Math.min(obj.pose.z + obj.dims.h, o.pose.z + o.dims.h) - Math.max(obj.pose.z, o.pose.z);
    if (zo <= 0.01) continue;
    const d = penetration(fp, footprint(o.pose, o.dims));
    if (d > 0.01) hits.push({ id: o.id, depth: d });
  }
  if (!fp.every((p) => pointInPolygon(p, floorPolygon))) hits.push({ id: "__room__", depth: 0 });
  return hits;
}

export const snapYaw = (yaw, step = 15) => (((Math.round(yaw / step) * step + 180) % 360) + 360) % 360 - 180;
