"""The scan pipeline's pieces and its jobs API (with a stand-in for the slow solve)."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mise.backend.app import create_app
from mise.pipeline import room_outline as ro
from mise.pipeline.scan import Cancelled, ScanEvents, detect_counters

cv2 = pytest.importorskip("cv2")


# ---- layout helpers ------------------------------------------------------------------------

def _outline(poly):
    poly = np.array(poly, float)
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return ro.Outline(polygon=[tuple(p) for p in poly], area_m2=area, ceiling_m=2.5, free=np.zeros((1, 1)), origin=(0, 0), cell=0.05)


def test_furniture_front_moves_back_to_the_wall():
    rng = np.random.default_rng(0)
    # room 0..4 x 0..3; free space stopped at x = 3.6 along a 0.4 m deep bookcase against the wall at x = 4
    room = _outline([(0, 0), (3.6, 0), (3.6, 3), (0, 3)])
    shelf = np.c_[rng.uniform(3.6, 4.0, 600), rng.uniform(0.5, 2.5, 600), rng.uniform(0.1, 1.3, 600)]
    # the opposite wall (x = 0) has a few points right on it: a real wall, must not move
    wall = np.c_[rng.uniform(-0.03, 0.03, 200), rng.uniform(0.2, 2.8, 200), rng.uniform(0.1, 2.2, 200)]
    fixes = ro.push_furniture_edges([room], np.vstack([shelf, wall]))
    assert len(fixes) == 1 and fixes[0].moved_m == pytest.approx(0.4, abs=0.05)
    xs = sorted({round(p[0], 2) for p in room.polygon})
    assert xs[0] == 0 and xs[-1] == pytest.approx(4.0, abs=0.05)


def test_open_continuation_is_not_furniture():
    rng = np.random.default_rng(1)
    room = _outline([(0, 0), (3, 0), (3, 3), (0, 3)])
    # points keep going past the edge (the room carries on, or an open doorway): leave it
    beyond = np.c_[rng.uniform(3.0, 4.5, 900), rng.uniform(0.5, 2.5, 900), rng.uniform(0.1, 1.3, 900)]
    assert ro.push_furniture_edges([room], beyond) == []


def test_counter_detection():
    rng = np.random.default_rng(2)
    top = np.c_[rng.uniform(1.0, 3.0, 2500), rng.uniform(0.0, 0.65, 2500), rng.normal(0.91, 0.01, 2500)]
    table = np.c_[rng.uniform(5, 6.2, 1500), rng.uniform(2, 2.8, 1500), rng.normal(0.74, 0.01, 1500)]
    clutter = np.c_[rng.uniform(0, 7, 300), rng.uniform(0, 4, 300), rng.uniform(0, 2.4, 300)]
    found = detect_counters(np.vstack([top, table, clutter]), [])
    assert len(found) == 1
    c = found[0]
    assert c["x0"] == pytest.approx(1.0, abs=0.1) and c["x1"] == pytest.approx(3.0, abs=0.1)
    assert c["y1"] - c["y0"] == pytest.approx(0.65, abs=0.12) and c["top"] == pytest.approx(0.91, abs=0.02)


def test_events_log_and_cancel(tmp_path):
    seen = []
    ev = ScanEvents(tmp_path / "events.jsonl", seen.append)
    ev.stage("frames", "running")
    ev.issue("blurry", "slow down", t0=1.0, t1=3.0)
    assert [e["kind"] for e in seen] == ["stage", "issue"]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2
    ev.cancel.set()
    with pytest.raises(Cancelled):
        ev.emit("progress")


# ---- jobs API ------------------------------------------------------------------------------

class Result(dict):
    def to_dict(self):
        return dict(self)


def fake_runner(video, work, opts, store, ev):
    (work / "thumbs").mkdir(exist_ok=True)
    cv2.imwrite(str(work / "thumbs" / "frame_0000.jpg"), np.zeros((10, 10, 3), np.uint8))
    ev.emit("frames", frames=[{"name": "frame_0000.jpg", "t": 0.0}])
    for name in ("frames", "check", "solve"):
        ev.stage(name, "running")
        ev.stage(name, "done")
    if opts.name == "slow":
        for _ in range(500):
            time.sleep(0.01)
            ev.emit("progress", stage="solve")
    ev.issue("turning", "walk, don't turn", "fail", 3.0, 9.0, frame="frame_0000.jpg")
    ev.emit("done", result={"scene_id": opts.scene_id})
    return Result(scene_id=opts.scene_id, rooms=1)


@pytest.fixture
def sclient(tmp_path):
    app = create_app(tmp_path / "scenes", scans_dir=tmp_path / "scans", scan_runner=fake_runner)
    with TestClient(app) as c:
        yield c


def _wait(c, jid, status=("done", "failed", "cancelled")):
    for _ in range(300):
        j = c.get(f"/api/scans/{jid}").json()
        if j["status"] in status:
            return j
        time.sleep(0.02)
    raise AssertionError(j)


def test_scan_job_runs_and_reports(sclient):
    c = sclient
    r = c.post("/api/scans", params={"scene_id": "my-flat", "name": "Flat"}, content=b"\0" * 4096)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    j = _wait(c, jid)
    assert j["status"] == "done" and j["result"]["rooms"] == 1
    assert j["stages"] == {"frames": "done", "check": "done", "solve": "done"}
    issues = [e for e in j["events"] if e["kind"] == "issue"]
    assert issues[0]["issue"] == "turning" and issues[0]["t0"] == 3.0
    later = c.get(f"/api/scans/{jid}", params={"since": issues[0]["seq"]}).json()["events"]
    assert [e["kind"] for e in later] == ["done"]
    assert c.get(f"/api/scans/{jid}/thumbs/frame_0000.jpg").status_code == 200
    assert c.get(f"/api/scans/{jid}/thumbs/..%2Fvideo.mp4").status_code == 404
    assert c.get("/api/scans").json()[0]["id"] == jid
    with c.websocket_connect(f"/ws/scans/{jid}") as ws:  # replays what happened so far
        assert ws.receive_json()["kind"] == "frames"


def test_scan_input_checks(sclient):
    c = sclient
    assert c.post("/api/scans", params={"scene_id": "../x"}, content=b"\0" * 4096).status_code == 422
    assert c.post("/api/scans", params={"scene_id": "ok"}, content=b"").status_code == 422
    assert c.post("/api/scans", params={"scene_id": "ok", "fps": 60}, content=b"\0" * 4096).status_code == 422
    assert c.get("/api/scans/nope").status_code == 404


def test_scan_cancel(sclient):
    c = sclient
    jid = c.post("/api/scans", params={"scene_id": "slow-one", "name": "slow"}, content=b"\0" * 4096).json()["id"]
    _wait(c, jid, ("running",))
    assert c.post(f"/api/scans/{jid}/cancel").status_code == 200
    assert _wait(c, jid)["status"] == "cancelled"


def test_scans_survive_a_restart(tmp_path):
    app = create_app(tmp_path / "scenes", scans_dir=tmp_path / "scans", scan_runner=fake_runner)
    with TestClient(app) as c:
        jid = c.post("/api/scans", params={"scene_id": "flat"}, content=b"\0" * 4096).json()["id"]
        _wait(c, jid)
    app2 = create_app(tmp_path / "scenes", scans_dir=tmp_path / "scans", scan_runner=fake_runner)
    with TestClient(app2) as c:
        j = c.get(f"/api/scans/{jid}").json()
        assert j["status"] == "done" and any(e["kind"] == "issue" for e in j["events"])


def test_live_coach(sclient):
    c = sclient
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (360, 640)).astype(np.uint8), (0, 0), 1.0)
    out = None
    for i in range(6):
        _ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(np.roll(tex, 4 * i, axis=1), cv2.COLOR_GRAY2BGR))
        out = c.post("/api/coach/phone-1", params={"t": i / 3}, content=jpg.tobytes()).json()
    assert out["frames"] == 6 and out["status"] == "ok" and "advice" in out
    dark = cv2.imencode(".jpg", np.full((360, 640, 3), 5, np.uint8))[1].tobytes()
    for i in range(6, 14):
        out = c.post("/api/coach/phone-1", params={"t": i / 3}, content=dark).json()
    assert out["status"] in ("dark", "low_texture") and out["advice"]
    assert c.post("/api/coach/phone-1", content=b"not a jpeg").status_code == 422
    assert c.delete("/api/coach/phone-1").json()["ok"]
