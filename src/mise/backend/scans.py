"""Scan jobs and the live capture coach.

  POST /api/scans?scene_id=..&name=..   body: the video file (raw bytes) -> {"id": ...}; runs in a thread
  GET  /api/scans                        jobs, newest first
  GET  /api/scans/{id}?since=N           status plus the events after seq N (polling)
  WS   /ws/scans/{id}                    every event so far, then each new one as it happens
  POST /api/scans/{id}/cancel
  GET  /api/scans/{id}/thumbs/{name}     small frame images for the timeline and issue list

  POST /api/coach/{session}?t=SECONDS    body: one JPEG from the phone camera -> flags and advice now

Scans stay on this machine: the video, frames and solve are written under the scans folder.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
FRAME = re.compile(r"^frame_\d{4,6}\.jpg$")
MAX_VIDEO = 4 * 1024**3


class ScanJob:
    def __init__(self, id: str, scene_id: str, name: str, work: Path):
        self.id, self.scene_id, self.name, self.work = id, scene_id, name, work
        self.status = "queued"  # queued | running | done | failed | cancelled
        self.created = time.time()
        self.events: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.ev = None  # the pipeline's ScanEvents (for cancelling)
        self.thread: threading.Thread | None = None  # set when this server runs the job
        self.offset = 0  # bytes of events.jsonl already read (jobs run elsewhere)

    def summary(self) -> dict:
        stages = {}
        for e in self.events:
            if e["kind"] == "stage":
                stages[e["stage"]] = e["status"]
        return {"id": self.id, "scene_id": self.scene_id, "name": self.name, "status": self.status,
                "created": self.created, "stages": stages, "events": len(self.events),
                "result": self.result, "error": self.error}


def register(app: FastAPI, store, scans_dir: Path, get_loop: Callable[[], asyncio.AbstractEventLoop | None],
             runner: Callable | None = None) -> None:
    """runner(video, work, opts, store, ev) runs the pipeline (default: pipeline.scan.run_scan)."""
    jobs: dict[str, ScanJob] = {}
    sockets: dict[str, set[WebSocket]] = {}
    coaches: dict[str, tuple[float, object]] = {}
    scans_dir = Path(scans_dir)

    def refresh(j: ScanJob) -> None:
        """Jobs this server isn't running (earlier runs, or `mise scan` in a terminal): read their log."""
        log = j.work / "events.jsonl"
        if j.thread is not None or not log.exists():
            return
        size = log.stat().st_size
        if size == j.offset:
            return
        with log.open() as f:
            f.seek(j.offset)
            chunk = f.read()
        complete = chunk[: chunk.rfind("\n") + 1]
        j.offset += len(complete.encode())
        for line in complete.splitlines():
            if line.strip():
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                push(j, ev)
        last = j.events[-1]["kind"] if j.events else None
        j.status = {"done": "done", "error": "failed", "cancelled": "cancelled"}.get(last, "running")
        if j.status == "running" and time.time() - log.stat().st_mtime > 3600:
            j.status, j.error = "failed", "stopped without finishing"
        j.result = next((e["result"] for e in reversed(j.events) if e["kind"] == "done"), None)
        j.error = next((e["message"] for e in reversed(j.events) if e["kind"] == "error"), j.error)

    def scan_folder() -> None:
        if not scans_dir.exists():
            return
        for d in sorted(scans_dir.iterdir()):
            meta = d / "job.json"
            if not meta.exists():
                continue
            try:
                m = json.loads(meta.read_text())
                if m["id"] in jobs:
                    continue
                j = ScanJob(m["id"], m["scene_id"], m["name"], d)
                j.created = m.get("created", d.stat().st_mtime)
                jobs[j.id] = j
                refresh(j)
            except (OSError, ValueError, KeyError):
                continue

    def job_or_404(job_id: str) -> ScanJob:
        if job_id not in jobs:
            raise HTTPException(404, "no such scan")
        return jobs[job_id]

    def push(job: ScanJob, ev: dict) -> None:
        if job.events and ev.get("seq", 0) <= job.events[-1].get("seq", -1):
            return
        job.events.append(ev)
        loop = get_loop()
        if not loop:
            return
        msg = json.dumps(ev, default=str)
        for ws in list(sockets.get(job.id, ())):
            asyncio.run_coroutine_threadsafe(_send(job.id, ws, msg), loop)

    async def _send(job_id: str, ws: WebSocket, msg: str) -> None:
        try:
            await ws.send_text(msg)
        except Exception:  # noqa: BLE001 - a closed socket, in any of its forms
            sockets.get(job_id, set()).discard(ws)

    scan_folder()  # jobs from earlier runs, or from `mise scan` in a terminal

    @app.post("/api/scans")
    async def start_scan(request: Request, scene_id: str, name: str = "Scanned home", fps: float = 3.0,
                         t0: float | None = None, t1: float | None = None, scale: float | None = None):
        if not SLUG.match(scene_id):
            raise HTTPException(422, "scene_id: lowercase letters, digits, - and _ (up to 64)")
        if not 0.5 <= fps <= 6:
            raise HTTPException(422, "fps must be between 0.5 and 6")
        if scale is not None and not 0 < scale < 1e6:
            raise HTTPException(422, "scale must be positive")
        if any(j.scene_id == scene_id and j.status in ("queued", "running") for j in jobs.values()):
            raise HTTPException(409, "a scan for this scene is already running")
        jid = f"scan_{uuid.uuid4().hex[:8]}"
        job = ScanJob(jid, scene_id, name[:120], scans_dir / jid)
        job.work.mkdir(parents=True, exist_ok=True)
        video = job.work / "video.mp4"
        size = 0
        with video.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_VIDEO:
                    f.close()
                    video.unlink()
                    raise HTTPException(413, "video too large (4 GB max)")
                f.write(chunk)
        if size < 1024:
            raise HTTPException(422, "send the video file as the request body")
        (job.work / "job.json").write_text(json.dumps({"id": job.id, "scene_id": scene_id, "name": job.name,
                                                        "created": job.created}))
        jobs[job.id] = job

        def work():
            from ..pipeline.scan import Cancelled, ScanEvents, ScanOptions, run_scan

            run = runner or run_scan
            job.ev = ScanEvents(job.work / "events.jsonl", lambda e: push(job, e))
            job.status = "running"
            try:
                opts = ScanOptions(scene_id=scene_id, name=job.name, fps=fps, t0=t0, t1=t1, scale=scale)
                res = run(video, job.work, opts, store, job.ev)
                job.result = res.to_dict() if hasattr(res, "to_dict") else res
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001 - reported to the monitor
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"

        job.thread = threading.Thread(target=work, daemon=True, name=job.id)
        job.thread.start()
        return {"id": job.id, "scene_id": scene_id}

    @app.get("/api/scans")
    def list_scans():
        scan_folder()
        for j in jobs.values():
            if j.status == "running":
                refresh(j)
        return [j.summary() for j in sorted(jobs.values(), key=lambda j: -j.created)]

    @app.get("/api/scans/{job_id}")
    def get_scan(job_id: str, since: int = -1):
        j = job_or_404(job_id)
        refresh(j)
        return {**j.summary(), "events": [e for e in j.events if e["seq"] > since]}

    @app.post("/api/scans/{job_id}/cancel")
    def cancel_scan(job_id: str):
        j = job_or_404(job_id)
        if j.status not in ("queued", "running") or j.ev is None:
            raise HTTPException(409, f"scan is {j.status}" if j.ev else "this scan runs elsewhere (a terminal): stop it there")
        j.ev.cancel.set()
        return {"ok": True}

    @app.get("/api/scans/{job_id}/thumbs/{name}")
    def thumb(job_id: str, name: str):
        j = job_or_404(job_id)
        if not FRAME.match(name):
            raise HTTPException(404)
        p = j.work / "thumbs" / name
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, media_type="image/jpeg")

    @app.websocket("/ws/scans/{job_id}")
    async def ws_scan(ws: WebSocket, job_id: str) -> None:
        await ws.accept()
        j = jobs.get(job_id)
        if j is None:
            await ws.close(code=4404)
            return
        for e in list(j.events):  # catch up, then live
            await ws.send_text(json.dumps(e, default=str))
        sockets.setdefault(job_id, set()).add(ws)
        try:
            while True:
                if j.thread is None and j.status == "running":
                    # run elsewhere: follow its log
                    try:
                        await asyncio.wait_for(ws.receive_text(), timeout=0.7)
                    except TimeoutError:
                        refresh(j)
                else:
                    await ws.receive_text()
        except WebSocketDisconnect:
            sockets[job_id].discard(ws)

    # ---- live coach: one frame at a time from the phone ------------------------------------
    @app.post("/api/coach/{session}")
    async def coach(session: str, request: Request, t: float | None = None):
        import cv2
        import numpy as np

        from ..pipeline import capture_qc

        if not SLUG.match(session):
            raise HTTPException(422, "bad session id")
        body = await request.body()
        if not body or len(body) > 8 * 1024**2:
            raise HTTPException(422, "send one JPEG frame (under 8 MB) as the body")
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(422, "not an image")
        now = time.time()
        for k in [k for k, (ts, _) in coaches.items() if now - ts > 600]:  # forget idle sessions
            del coaches[k]
        if session not in coaches:
            if len(coaches) >= 50:
                raise HTTPException(429, "too many live sessions")
            coaches[session] = (now, capture_qc.Analyzer(width=480))
        _, a = coaches[session]
        coaches[session] = (now, a)
        del a.frames[:-300], a._sharp_ref[:-300]  # a live session only needs its recent past
        start = getattr(a, "_t_start", None)
        if start is None:
            a._t_start = start = now
        fq = a.add(t if t is not None else now - start, img)
        recent = [f for f in a.frames if f.t > fq.t - 3]
        worst = {}
        for f in recent:
            for fl in f.flags:
                worst[fl] = worst.get(fl, 0) + 1
        # advise on what persisted over the last ~3 s, not a single bad frame
        persistent = [k for k, n in sorted(worst.items(), key=lambda kv: -kv[1]) if n >= max(2, len(recent) // 2)]
        return {"t": round(fq.t, 2), "flags": fq.flags, "parallax": None if np.isnan(fq.parallax_deg) else round(fq.parallax_deg, 2),
                "rotation": None if np.isnan(fq.rotation_deg) else round(fq.rotation_deg, 1),
                "status": persistent[0] if persistent else "ok",
                "advice": capture_qc.ADVICE[persistent[0]] if persistent else "Looks good: keep walking slowly along the walls.",
                "frames": len(a.frames)}

    @app.delete("/api/coach/{session}")
    def coach_end(session: str):
        coaches.pop(session, None)
        return {"ok": True}
