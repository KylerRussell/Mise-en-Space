"""Run learned_recon.py on a Colab GPU from this machine, through the `colab` CLI.

    python scripts/colab/run_learned_recon.py bedroom --t0 134 --t1 182
    python scripts/colab/run_learned_recon.py tour --gpu H100 --model vggt --chunk 150

1. packs the frames of the stretch (picked by mise, see --frames) and uploads them,
2. starts the reconstruction on the VM in the background and follows its log,
3. downloads the outputs to data/scans/learned/NAME (frames.json, sparse/0, depth.npz, meta.json),
4. stops the VM, also when something fails (unless --keep).

Then: mise scan VIDEO --scene ID --poses data/scans/learned/NAME --frames FRAMES

Session state goes to its own file (~/.config/colab-cli/mise-sessions.json) and sessions are
named mise-NAME, so runs never touch sessions other work on this account has open. Several
runs can go at once (different NAMEs): the account allows concurrent GPU sessions, each one
spending compute units.
"""

from __future__ import annotations

import argparse
import io
import json
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="run name: session mise-NAME, outputs in data/scans/learned/NAME")
    ap.add_argument("--frames", type=Path, default=Path("data/scans/tour-frames"),
                    help="folder with frames/ and frames.json picked by mise (extract_sharp_frames)")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--model", choices=["mapanything", "vggt"], default="mapanything")
    ap.add_argument("--gpu", default="A100,G4,H100,L4",
                    help="GPUs to try in order until one is free (T4, L4, G4, A100, H100)")
    ap.add_argument("--wait", type=float, default=60, help="minutes to keep trying for a free GPU")
    ap.add_argument("--keep", action="store_true", help="leave the VM running afterwards")
    ap.add_argument("--out", type=Path, help="default: data/scans/learned/NAME")
    a, extra = ap.parse_known_args(argv)  # anything else goes to learned_recon.py (--chunk, --limit, ...)
    session = f"mise-{a.name}"
    out = a.out or Path("data/scans/learned") / a.name

    kept = json.loads((a.frames / "frames.json").read_text())
    sel = [k for k in kept if (a.t0 is None or k["t"] >= a.t0) and (a.t1 is None or k["t"] <= a.t1)]
    print(f"{len(sel)} frames in the stretch; packing", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="mise-colab-"))
    pack = tmp / "in.tgz"
    with tarfile.open(pack, "w:gz") as tar:
        for k in sel:
            tar.add(a.frames / "frames" / k["name"], f"in/frames/{k['name']}")
        info = tarfile.TarInfo("in/frames.json")
        data = json.dumps(sel).encode()
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

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
        colab("upload", "-s", session, str(pack), "/content/in.tgz", timeout=1800)
        colab("upload", "-s", session, str(HERE / "learned_recon.py"), "/content/learned_recon.py", timeout=300)
        remote(session, f"""
import subprocess, shlex, sys
subprocess.run(["tar", "xzf", "/content/in.tgz", "-C", "/content"], check=True)
cmd = [sys.executable, "-u", "/content/learned_recon.py", *{args!r}]
open("/content/run.sh", "w").write(shlex.join(cmd) + " > /content/recon.log 2>&1; echo $? > /content/exit_code\\n")
subprocess.Popen(["bash", "/content/run.sh"], start_new_session=True)
print("started:", shlex.join(cmd))
""")
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
                if not line.startswith(("  ", "Downloading", "Fetching")) or "[recon]" in line:
                    print("  " + line[-300:], flush=True)
            seen = len(lines)
            code = int(tail.strip()) if tail.strip() else None
        if code != 0:
            raise RuntimeError(f"reconstruction failed (exit {code}); see the log above")
        remote(session, 'import subprocess; subprocess.run(["tar", "czf", "/content/out.tgz", "-C", "/content", "out"], check=True)')
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
        if a.keep:
            print(f"left {session} running: colab --config {CONFIG} stop -s {session}")
        else:
            colab("stop", "-s", session, check=False, timeout=300)
            print(f"stopped {session}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
