// Thin client for the scene backend. The editor always acts as the user.
const H = { "Content-Type": "application/json", "X-Actor": "user" };

async function req(method, path, body) {
  const r = await fetch(path, { method, headers: H, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail ?? msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

const enc = encodeURIComponent;
const V = (s, v) => `/api/scenes/${enc(s)}/variants/${enc(v)}`;

export const api = {
  scenes: () => req("GET", "/api/scenes"),
  scene: (s) => req("GET", `/api/scenes/${enc(s)}`),
  room: (s, v) => req("GET", `/api/scenes/${enc(s)}/room?variant=${enc(v)}`),
  variants: (s) => req("GET", `/api/scenes/${enc(s)}/variants`),
  createVariant: (s, name, base) => req("POST", `/api/scenes/${enc(s)}/variants`, { name, base }),
  deleteVariant: (s, v) => req("DELETE", V(s, v)),
  objects: (s, v) => req("GET", `${V(s, v)}/objects`),
  move: (s, v, id, body) => req("POST", `${V(s, v)}/objects/${enc(id)}/move`, body),
  add: (s, v, body) => req("POST", `${V(s, v)}/objects`, body),
  hide: (s, v, id) => req("POST", `${V(s, v)}/objects/${enc(id)}/hide`),
  remove: (s, v, id) => req("DELETE", `${V(s, v)}/objects/${enc(id)}`),
  update: (s, v, id, fields) => req("PATCH", `${V(s, v)}/objects/${enc(id)}`, { fields }),
  undo: (s, v) => req("POST", `${V(s, v)}/undo`),
  redo: (s, v) => req("POST", `${V(s, v)}/redo`),
  check: (s, v) => req("GET", `${V(s, v)}/check`),
  diff: (s, a, b) => req("GET", `/api/scenes/${enc(s)}/diff?a=${enc(a)}&b=${enc(b)}`),
  promote: (s, v, ids) => req("POST", `${V(s, v)}/promote`, { object_ids: ids }),
  shopping: (s, v) => req("GET", `${V(s, v)}/shopping`),
  library: (s) => req("GET", `/api/scenes/${enc(s)}/library`),
  upload: (s, body) => req("POST", `/api/scenes/${enc(s)}/library/upload`, body),
  startImport: (s, body) => req("POST", `/api/scenes/${enc(s)}/imports`, body),
  job: (id) => req("GET", `/api/jobs/${enc(id)}`),
  ops: (s, v) => req("GET", `/api/scenes/${enc(s)}/ops?variant=${enc(v)}&limit=30`),
  fileUrl: (s, path) => `/files/${enc(s)}/${path}`,
};

export function connect(sceneId, onEvent) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let ws, timer, closed = false;
  const open = () => {
    ws = new WebSocket(`${proto}://${location.host}/ws/scenes/${enc(sceneId)}`);
    ws.onmessage = (e) => { try { onEvent(JSON.parse(e.data)); } catch {} };
    ws.onclose = () => { if (!closed) timer = setTimeout(open, 1500); };
    ws.onopen = () => { clearInterval(ws._ping); ws._ping = setInterval(() => ws.readyState === 1 && ws.send("ping"), 20000); };
  };
  open();
  return () => { closed = true; clearTimeout(timer); ws?.close(); };
}
