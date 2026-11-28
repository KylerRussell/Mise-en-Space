"""Client for Idol image-to-3D (base: Gaussian-splat `.ply`; Max: GLB mesh).

What the project plan states about the API, and what this code relies on:
  * base Idol returns a centered Gaussian-splat `.ply` with no stated up
    axis or units, and its splats carry base color only (no higher-order SH);
  * input takes an ``image_type`` of ``real`` (photos) or ``synthetic`` (renders);
  * Idol Max returns ``model_glb``, upright in +Y-up coordinates, at $0.20/generation.

NOT VERIFIED here (the docs hosts were unreachable when this was written):
the exact endpoint ids, the input field name for the image URL and the
output field names for the `.ply`. They are settings (`IdolConfig`) and
the response is searched for the first `.ply`/`.glb` URL, so a mismatch
shows up as a clear error rather than a silent bad asset. Check them
against the docs before first use:
  https://fal.ai/models/speridlabs/idol/max/llms.txt
  https://www.pixazo.ai/models/idol.md

The transport is fal's generic queue API (submit, poll status, fetch result).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx


@dataclass
class IdolConfig:
    base_model: str = os.environ.get("IDOL_MODEL", "speridlabs/idol")
    max_model: str = os.environ.get("IDOL_MAX_MODEL", "speridlabs/idol/max")
    queue_url: str = os.environ.get("FAL_QUEUE_URL", "https://queue.fal.run")
    image_field: str = os.environ.get("IDOL_IMAGE_FIELD", "image_url")
    api_key: str | None = field(default_factory=lambda: os.environ.get("FAL_KEY"))
    poll_s: float = 2.0
    timeout_s: float = 600.0


@dataclass
class IdolResult:
    kind: Literal["splat", "mesh"]
    path: Path
    up_axis: tuple[float, float, float] | None  # declared up in asset frame, if known
    raw: dict
    cost_usd: float | None = None


class IdolError(RuntimeError):
    pass


def find_asset_url(obj: Any, exts: tuple[str, ...]) -> str | None:
    """Depth-first search of a response for the first URL ending in one of `exts`."""
    if isinstance(obj, str):
        base = obj.split("?")[0].lower()
        return obj if obj.startswith("http") and base.endswith(exts) else None
    if isinstance(obj, dict):
        # prefer conventional keys first
        for k in ("model_glb", "ply", "splat", "model", "file", "output"):
            if k in obj:
                u = find_asset_url(obj[k], exts)
                if u:
                    return u
        for v in obj.values():
            u = find_asset_url(v, exts)
            if u:
                return u
    if isinstance(obj, list):
        for v in obj:
            u = find_asset_url(v, exts)
            if u:
                return u
    return None


class IdolClient:
    def __init__(self, cfg: IdolConfig | None = None, http: httpx.Client | None = None):
        self.cfg = cfg or IdolConfig()
        self.http = http or httpx.Client(timeout=60)

    def _headers(self) -> dict:
        if not self.cfg.api_key:
            raise IdolError("FAL_KEY is not set")
        return {"Authorization": f"Key {self.cfg.api_key}", "Content-Type": "application/json"}

    def _run(self, model: str, payload: dict) -> dict:
        r = self.http.post(f"{self.cfg.queue_url}/{model}", json=payload, headers=self._headers())
        r.raise_for_status()
        sub = r.json()
        status_url = sub.get("status_url") or f"{self.cfg.queue_url}/{model}/requests/{sub['request_id']}/status"
        result_url = sub.get("response_url") or f"{self.cfg.queue_url}/{model}/requests/{sub['request_id']}"
        t0 = time.time()
        while True:
            s = self.http.get(status_url, headers=self._headers())
            s.raise_for_status()
            st = s.json().get("status")
            if st == "COMPLETED":
                break
            if st in ("FAILED", "ERROR", "CANCELLED"):
                raise IdolError(f"generation {st}: {s.json()}")
            if time.time() - t0 > self.cfg.timeout_s:
                raise IdolError("generation timed out")
            time.sleep(self.cfg.poll_s)
        res = self.http.get(result_url, headers=self._headers())
        res.raise_for_status()
        return res.json()

    def generate(self, image_url: str, out_dir: Path, image_type: Literal["real", "synthetic"] = "real",
                 mesh: bool = False) -> IdolResult:
        model = self.cfg.max_model if mesh else self.cfg.base_model
        payload = {self.cfg.image_field: image_url, "image_type": image_type}
        raw = self._run(model, payload)
        exts = (".glb",) if mesh else (".ply", ".splat")
        url = find_asset_url(raw, exts)
        if not url:
            raise IdolError(f"no {'/'.join(exts)} URL in response keys {list(raw)}; check IdolConfig against the docs")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / ("model.glb" if mesh else "model.ply")
        with self.http.stream("GET", url) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return IdolResult(kind="mesh" if mesh else "splat", path=path,
                          up_axis=(0.0, 1.0, 0.0) if mesh else None, raw=raw, cost_usd=0.20 if mesh else None)


class IdolColab:
    """Idol (speridlabs/idol, open weights) run on the owner's Colab GPU instead of a hosted API.

    Each call packs the product image, runs scripts/colab/idol_run.py on a fresh `mise-idol-*`
    session through run_learned_recon.py --idol, and fetches the splat. A few minutes per call,
    most of it starting the VM and installing Idol. Needs a Hugging Face login on this machine
    whose account has accepted the terms of Idol's gated encoders (DINOv3, FLUX.2-dev).
    """

    def __init__(self, repo: Path | None = None, gpu: str = "L4,G4,A100,H100", num_gaussians: int = 262144,
                 runner=None, http: httpx.Client | None = None):
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[3]
        self.gpu, self.num_gaussians = gpu, num_gaussians
        self.http = http or httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        self.run = runner or self._run

    def _run(self, cmd: list[str], log: Path) -> int:
        import subprocess

        with open(log, "w") as fh:
            return subprocess.run(cmd, cwd=self.repo, stdout=fh, stderr=subprocess.STDOUT).returncode

    def generate(self, image_url: str, out_dir: Path, image_type: Literal["real", "synthetic"] = "real",
                 mesh: bool = False) -> IdolResult:
        import json
        import sys
        import uuid

        if mesh:
            raise IdolError("Idol on Colab makes splats only (Idol Max meshes are the hosted API)")
        out_dir = Path(out_dir)
        inp = out_dir / "idol-in"
        inp.mkdir(parents=True, exist_ok=True)
        ext = Path(image_url.split("?")[0]).suffix.lower() if Path(image_url.split("?")[0]).suffix.lower() in (
            ".jpg", ".jpeg", ".png", ".webp") else ".jpg"
        name = f"product{ext}"
        if image_url.startswith(("http://", "https://")):
            r = self.http.get(image_url)
            r.raise_for_status()
            (inp / name).write_bytes(r.content)
        else:
            (inp / name).write_bytes(Path(image_url).read_bytes())
        (inp / "images.json").write_text(json.dumps([{"name": name, "mode": image_type}]))
        run_out = out_dir / "idol-out"
        cmd = [sys.executable, str(self.repo / "scripts" / "colab" / "run_learned_recon.py"),
               f"idol-{uuid.uuid4().hex[:6]}", "--idol", str(inp), "--gpu", self.gpu, "--out", str(run_out),
               "--num-gaussians", str(self.num_gaussians)]
        code = self.run(cmd, out_dir / "idol.log")
        ply_path = run_out / "product.ply"
        if code != 0 or not ply_path.exists():
            tail = (out_dir / "idol.log").read_text()[-600:] if (out_dir / "idol.log").exists() else ""
            raise IdolError(f"Idol on Colab failed (exit {code}): {tail}")
        dest = out_dir / "model.ply"
        dest.write_bytes(ply_path.read_bytes())
        meta = json.loads((run_out / "meta.json").read_text()) if (run_out / "meta.json").exists() else {}
        return IdolResult(kind="splat", path=dest, up_axis=None, raw=meta)
