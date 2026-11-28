"""Run learned_recon.py on a Colab GPU from this machine, through the `colab` CLI.

    python scripts/colab/run_learned_recon.py bedroom --t0 134 --t1 182
    python scripts/colab/run_learned_recon.py tour --gpu H100 --model vggt --chunk 150

1. packs the frames of the stretch (picked by mise, see --frames) and uploads them,
2. starts the reconstruction on the VM in the background and follows its log,
3. downloads the outputs to data/scans/learned/NAME (frames.json, sparse/0, depth.npz, meta.json),
4. stops the VM, also when something fails (unless --keep).

Then: mise scan VIDEO --scene ID --poses data/scans/learned/NAME --frames FRAMES

With --splat SOLVE it trains a Gaussian splat instead (splat_train.py) on the frames of that solve
(a learned_recon / join-poses folder) and fetches splat.ply + meta.json, by default to
data/scans/splats/NAME:

    python scripts/colab/run_learned_recon.py home-v3 --splat data/scans/home-v3/learned-run/learned/home \
        --frames data/scans/home-v3/learned-run/picked --gpu G4,H100,A100 --steps 30000

Session state goes to its own file (~/.config/colab-cli/mise-sessions.json) and sessions are
named mise-NAME, so runs never touch sessions other work on this account has open. Several
runs can go at once (different NAMEs): the account allows concurrent GPU sessions, each one
spending compute units.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = Path.home() / ".config" / "colab-cli" / "mise-sessions.json"


def colab(*args: str, timeout: float | None = None, check: bool = True, stdin: str | None = None) -> str:
    p = subprocess.run(["colab", "--config", str(CONFIG), *args], capture_output=True, text=True, input=stdin,
                       timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"colab {' '.join(args)} failed ({p.returncode}): {p.stderr[-2000:]}")
    return p.stdout


def remote(session: str, code: str, timeout: float = 120) -> str:
    return colab("exec", "-s", session, "--timeout", str(timeout), stdin=code, timeout=timeout + 60)


def _base_colour(src: Path, dst: Path) -> Path:
    """A copy of a Gaussian splat .ply without its f_rest_* (view-dependent colour) properties."""
    import numpy as np
    with open(src, "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += fh.readline()
        lines = head.splitlines()
        props = [ln.split()[-1].decode() for ln in lines if ln.startswith(b"property")]
        n = int(next(ln for ln in lines if ln.startswith(b"element vertex")).split()[-1])
        data = np.frombuffer(fh.read(), dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    keep = [i for i, q in enumerate(props) if not q.startswith("f_rest_")]
    hdr = [ln for ln in lines if not (ln.startswith(b"property") and ln.split()[-1].startswith(b"f_rest_"))]
    with open(dst, "wb") as fh:
        fh.write(b"\n".join(hdr) + b"\n")
        fh.write(np.ascontiguousarray(data[:, keep]).tobytes())
    return dst


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="run name: session mise-NAME, outputs in data/scans/learned/NAME")
    ap.add_argument("--frames", type=Path, default=Path("data/scans/tour-frames"),
                    help="folder with frames/ and frames.json picked by mise (extract_sharp_frames)")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--model", default="mapanything",
                    help="mapanything, vggt, or one of learned_recon.EXTERNAL (pi3, pi3x, da3, da3_nested, must3r, ...)")
    ap.add_argument("--gpu", default="A100,G4,H100,L4",
                    help="GPUs to try in order until one is free (T4, L4, G4, A100, H100)")
    ap.add_argument("--wait", type=float, default=60, help="minutes to keep trying for a free GPU")
    ap.add_argument("--keep", action="store_true", help="leave the VM running afterwards")
    ap.add_argument("--out", type=Path, help="default: data/scans/learned/NAME (data/scans/splats/NAME with --splat)")
    ap.add_argument("--splat", type=Path, help="train a splat on this solve's frames instead (splat_train.py)")
    ap.add_argument("--render", type=Path, nargs="+", help="splat .ply files to render side by side (render_splat.py)")
    ap.add_argument("--cameras", type=Path, help="with --render: cameras.json (from splat_train.py)")
    ap.add_argument("--names", nargs="*", help="with --render: frames to render (default 8 spread over the cameras)")
    ap.add_argument("--segment", action="store_true", help="label the frames' pixels instead (segment_frames.py)")
    ap.add_argument("--idol", type=Path, help="product images to 3D splats instead (idol_run.py): a folder with "
                    "images.json ([{name, mode}]) and the images; sends this machine's Hugging Face token")
    ap.add_argument("--labels", type=Path, help="with --splat: segment_frames.py output, to label every Gaussian")
    ap.add_argument("--scene-from", type=Path, help="with --splat: a scan's result.json; train in that scene's frame")
    a, extra = ap.parse_known_args(argv)  # anything else goes to the remote script (--chunk, --steps, ...)
    session = f"mise-{a.name}"
    kind = "idol" if a.idol else "renders" if a.render else "splats" if a.splat else "semantic" if a.segment else "learned"
    out = a.out or Path("data/scans") / kind / a.name
    script = {"splats": "splat_train.py", "semantic": "segment_frames.py", "learned": "learned_recon.py",
              "renders": "render_splat.py", "idol": "idol_run.py"}[kind]
    if a.render:  # the frames to look from
        cams = json.loads(a.cameras.read_text())
        names = a.names or [cams[i]["name"] for i in [round(k * (len(cams) - 1) / 7) for k in range(8)]]
        a.names = names

    if a.idol:  # no frames: the product images, and the token for Idol's gated encoders
        a.frames = a.idol
        kept = []
    else:
        kept = json.loads((a.frames / "frames.json").read_text())
    if a.render:
        kept = [k for k in kept if k["name"] in set(a.names)]
    if a.splat:  # the frames the solve has
        names = {k["name"] for k in json.loads((a.splat / "frames.json").read_text())}
        kept = [k for k in kept if k["name"] in names]
    sel = [k for k in kept if (a.t0 is None or k["t"] >= a.t0) and (a.t1 is None or k["t"] <= a.t1)]
    print(f"{len(sel)} frames in the stretch; packing", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="mise-colab-"))
    pack = tmp / "in.tgz"
    with tarfile.open(pack, "w:gz") as tar:
        for k in sel:
            tar.add(a.frames / "frames" / k["name"], f"in/frames/{k['name']}")
        if a.idol:
            for j in json.loads((a.idol / "images.json").read_text()):
                tar.add(a.idol / j["name"], f"in/images/{j['name']}")
            tar.add(a.idol / "images.json", "in/images.json")
            tok = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
            if tok.exists():  # to the run's own VM only; idol_run.py moves it into the VM's HF cache
                tar.add(tok, "in/hf_token")
        info = tarfile.TarInfo("in/frames.json")
        data = json.dumps(sel).encode()
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        if (a.frames / "intrinsics.json").exists():  # the camera's calibration travels with its frames
            tar.add(a.frames / "intrinsics.json", "in/intrinsics.json")
        if a.splat:
            for f in ("cameras.txt", "images.txt", "points3D.txt"):
                tar.add(a.splat / "sparse" / "0" / f, f"in/solve/sparse/0/{f}")
            if (a.splat / "depth.npz").exists():  # the model's depth maps: splat_train's depth loss
                tar.add(a.splat / "depth.npz", "in/solve/depth.npz")
            if a.labels:
                for f in ("labels.npz", "meta.json"):
                    tar.add(a.labels / f, f"in/labels/{f}")

        if a.render:  # base colour only (no view-dependent terms): a quarter of the size, enough to look
            for k, pth in enumerate(a.render):
                tar.add(_base_colour(pth, tmp / f"splat_{k}.ply"), f"in/splat_{k}.ply")
            tar.add(a.cameras, "in/cameras.json")

    if a.idol:
        args = ["--images", "/content/in/images", "--out", "/content/out", *extra]
    elif a.render:
        args = ["--splats", *[f"/content/in/splat_{k}.ply" for k in range(len(a.render))], "--cameras",
                "/content/in/cameras.json", "--frames", "/content/in/frames", "--out", "/content/out",
                "--names", *a.names, *extra]
    elif a.segment:
        args = ["--frames", "/content/in", "--out", "/content/out", *extra]
    elif a.splat:
        args = ["--frames", "/content/in", "--solve", "/content/in/solve", "--out", "/content/out", *extra]
        if a.scene_from:
            ref = next(pc for pc in json.loads(a.scene_from.read_text())["pieces"] if pc.get("scene_from_solve"))
            args += ["--world", json.dumps(ref["scene_from_solve"])]
        if a.labels:
            args += ["--labels", "/content/in/labels"]
    else:
        args = ["--frames", "/content/in", "--out", "/content/out", "--model", a.model]
        args += [f"--t0={a.t0}"] if a.t0 is not None else []
        args += [f"--t1={a.t1}"] if a.t1 is not None else []
        args += extra
    started = time.time()
    deadline, got_gpu = time.time() + 60 * a.wait, None
    while got_gpu is None:
        for gpu in a.gpu.split(","):
            print(f"starting {session} on {gpu}", flush=True)
            try:
                colab("new", "-s", session, "--gpu", gpu.strip(), timeout=1200)
                got_gpu = gpu
                break
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"  no {gpu}: {str(e).strip().splitlines()[-1][-160:]}", flush=True)
                colab("stop", "-s", session, check=False, timeout=300)
        if got_gpu is None:
            if time.time() > deadline:
                raise SystemExit(f"none of {a.gpu} could be had in {a.wait} min; try again later")
            print("  no GPU free; trying the list again in 2 min", flush=True)
            time.sleep(120)
    try:
        # in pieces: some Colab regions refuse an upload of ~90 MB in one request (400 Bad Request)
        piece, parts = 40 << 20, []
        with open(pack, "rb") as fh:
            while chunk := fh.read(piece):
                part = tmp / f"in.tgz.{len(parts):03d}"
                part.write_bytes(chunk)
                parts.append(part)
        for part in parts:
            for attempt in range(4):
                try:
                    colab("upload", "-s", session, str(part), f"/content/{part.name}", timeout=1800)
                    break
                except (RuntimeError, subprocess.TimeoutExpired) as e:
                    print(f"  (upload of {part.name}, attempt {attempt + 1}, failed: {str(e).strip().splitlines()[-1][-120:]})",
                          flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(30)
        join = ("import glob, subprocess\n"  # safe to run again: once joined, the pieces are gone
                "if glob.glob('/content/in.tgz.[0-9]*'):\n"
                "    subprocess.run('cat /content/in.tgz.[0-9]* > /content/in.tgz.tmp && mv /content/in.tgz.tmp "
                "/content/in.tgz && rm /content/in.tgz.[0-9]*', shell=True, check=True)\n")
        for attempt in range(4):
            try:
                remote(session, join, timeout=300)
                break
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"  (joining the pieces, attempt {attempt + 1}, failed: {str(e).strip().splitlines()[-1][-120:]})",
                      flush=True)
                if attempt == 3:
                    raise
                time.sleep(30)
        for scr in {script, "splat_train.py"} if a.render else {script}:  # render_splat uses splat_train's install
            colab("upload", "-s", session, str(HERE / scr), f"/content/{scr}", timeout=300)
        start = f"""
import os, subprocess, shlex, sys
if os.path.exists("/content/run.sh"):  # an earlier attempt got this far: don't start a second job
    print("already started")
else:
    subprocess.run(["tar", "xzf", "/content/in.tgz", "-C", "/content"], check=True)
    cmd = [sys.executable, "-u", "/content/{script}", *{args!r}]
    open("/content/run.sh.tmp", "w").write(shlex.join(cmd) + " > /content/recon.log 2>&1; echo $? > /content/exit_code\\n")
    os.rename("/content/run.sh.tmp", "/content/run.sh")
    subprocess.Popen(["bash", "/content/run.sh"], start_new_session=True)
    print("started:", shlex.join(cmd))
"""
        for attempt in range(4):  # the exec call itself can time out now and then
            try:
                print("  " + remote(session, start).strip()[-300:], flush=True)
                break
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"  (start attempt {attempt + 1} failed: {str(e).strip().splitlines()[-1][-120:]})", flush=True)
                if attempt == 3:
                    raise
                time.sleep(20)
        seen, code, misses = 0, None, 0
        while code is None:
            time.sleep(30)
            try:  # a status check can hang or fail now and then; the job runs on regardless
                text = remote(session, """
import os
log = open("/content/recon.log").read() if os.path.exists("/content/recon.log") else ""
print(log)
print("@@EXIT@@", open("/content/exit_code").read().strip() if os.path.exists("/content/exit_code") else "")
""", timeout=60)
                misses = 0
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                misses += 1
                print(f"  (status check failed, {misses} in a row: {str(e).strip().splitlines()[-1][-120:]})", flush=True)
                if misses >= 10:
                    raise
                continue
            log, _, tail = text.rpartition("@@EXIT@@")
            lines = log.rstrip("\n").splitlines()
            for line in lines[seen:]:
                if not line.startswith(("  ", "Downloading", "Fetching")) or "[recon]" in line or "[splat]" in line or "[seg]" in line or "[render]" in line or "[idol]" in line or 'File "' in line:
                    print("  " + line[-300:], flush=True)
            seen = len(lines)
            code = int(tail.strip()) if tail.strip() else None
        if code != 0:
            raise RuntimeError(f"reconstruction failed (exit {code}); see the log above")
        for attempt in range(3):  # packing can outlast one exec call; the next attempt just packs again
            try:
                remote(session, 'import subprocess; subprocess.run(["tar", "czf", "/content/out.tgz", "-C", "/content", '
                                '"out"], check=True)', timeout=600)
                break
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"  (packing attempt {attempt + 1} failed: {str(e).strip().splitlines()[-1][-120:]})", flush=True)
                if attempt == 2:
                    raise
                time.sleep(20)
        got = tmp / "out.tgz"
        colab("download", "-s", session, "/content/out.tgz", str(got), timeout=1800)
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(got) as tar:
            for m in tar.getmembers():
                if m.name.startswith("out/") and m.isfile() and ".." not in m.name:
                    m.name = m.name[len("out/"):]
                    tar.extract(m, out, filter="data")
        meta = json.loads((out / "meta.json").read_text())
        print(f"done in {(time.time() - started) / 60:.1f} min: {out}", flush=True)
        print(json.dumps({k: v for k, v in meta.items() if k != "chunks"}, indent=1))
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)  # the packed inputs (with the token, for --idol)
        if a.keep:
            print(f"left {session} running: colab --config {CONFIG} stop -s {session}")
        else:
            colab("stop", "-s", session, check=False, timeout=300)
            print(f"stopped {session}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
