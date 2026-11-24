"""Helpers for the pre-registered hypotheses H1-H6 (see docs/PLAN.md and docs/EVALUATION.md).

Fix every threshold in `THRESHOLDS` *before* a test runs, record the
prediction, then score it against the result with :func:`score`.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

THRESHOLDS = {
    "H1": {"predict_max_err": 0.01, "kill_err": 0.02},
    "H2": {"metric": "masked LPIPS (lower is better)"},
    "H3": {"predict_chamfer_frac_of_diag": 0.05},
    "H4": {"n": 20, "predict_pass_rate": 0.80, "kill_pass_rate": 0.60, "gate": 1.10},
    "H5": {"predict_win_rate": 0.70, "hard_violations": 0, "held_out_briefs": 5},
    "H6": {"predict_live_rate": 0.90, "kill_live_rate": 0.80, "days": 7},
}


# ---- H1 -----------------------------------------------------------------------

def h1(scale_report: dict) -> dict:
    err = scale_report.get("max_abs_error")
    t = THRESHOLDS["H1"]
    return {"max_abs_error": err, "prediction_held": err is not None and err <= t["predict_max_err"],
            "kill": err is not None and err > t["kill_err"]}


# ---- H2 -----------------------------------------------------------------------

_LPIPS = None

def masked_lpips(img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray) -> float:
    """LPIPS inside the dilated removal mask (needs the `eval` extra). Images HxWx3 in [0,1]."""
    import lpips  # type: ignore
    import torch  # type: ignore

    global _LPIPS
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex", spatial=True)
    fn = _LPIPS
    ta = torch.tensor(img_a).permute(2, 0, 1)[None].float() * 2 - 1
    tb = torch.tensor(img_b).permute(2, 0, 1)[None].float() * 2 - 1
    with torch.no_grad():
        d = fn(ta, tb)[0, 0].numpy()
    m = mask.astype(bool)
    return float(d[m].mean())


def h2(learned: list[float], baseline: list[float]) -> dict:
    """Per-view masked LPIPS on held-out views (same views, both arms)."""
    a, b = np.asarray(learned), np.asarray(baseline)
    wins = int((a < b).sum())
    return {"n_views": len(a), "learned_mean": float(a.mean()), "baseline_mean": float(b.mean()),
            "learned_wins": wins, "prediction_held": bool(a.mean() < b.mean()),
            "switch": "ship the plane baseline" if a.mean() >= b.mean() else None}


# ---- H3 -----------------------------------------------------------------------

def chamfer(a: np.ndarray, b: np.ndarray, max_points: int = 20000, seed: int = 0) -> float:
    """Symmetric mean nearest-neighbor distance (brute force in chunks; fine for <= 20k points)."""
    rng = np.random.default_rng(seed)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]

    def one_way(p, q):
        out = np.empty(len(p))
        for i in range(0, len(p), 2048):
            d = ((p[i:i + 2048, None, :] - q[None, :, :]) ** 2).sum(-1)
            out[i:i + 2048] = np.sqrt(d.min(1))
        return out.mean()

    return float((one_way(a, b) + one_way(b, a)) / 2)


def h3(completed: np.ndarray, scanned_alone: np.ndarray) -> dict:
    diag = float(np.linalg.norm(scanned_alone.max(0) - scanned_alone.min(0)))
    c = chamfer(completed, scanned_alone)
    frac = c / diag
    return {"chamfer_m": c, "diag_m": diag, "frac": frac,
            "prediction_held": frac <= THRESHOLDS["H3"]["predict_chamfer_frac_of_diag"]}


# ---- H4 -----------------------------------------------------------------------

def h4(anisotropies: list[float]) -> dict:
    t = THRESHOLDS["H4"]
    a = np.asarray(anisotropies)
    rate = float((a <= t["gate"]).mean()) if len(a) else 0.0
    return {"n": len(a), "pass_rate": rate, "prediction_held": rate >= t["predict_pass_rate"],
            "switch": "Idol Max meshes or require confirmed dimensions" if rate < t["kill_pass_rate"] else None}


# ---- H5 -----------------------------------------------------------------------

@dataclass
class BlindPair:
    brief_id: str
    left: str  # variant id shown on the left
    right: str
    solver_side: str  # "left" | "right", hidden from the judge
    choice: str | None = None  # "left" | "right" | "tie"


@dataclass
class BlindStudy:
    pairs: list[BlindPair] = field(default_factory=list)

    @staticmethod
    def build(items: list[tuple[str, str, str]], seed: int = 0) -> "BlindStudy":
        """items: (brief_id, variant_with_solver, variant_without_solver). Order and sides are randomized."""
        rng = random.Random(seed)
        pairs = []
        for brief, with_s, without_s in items:
            if rng.random() < 0.5:
                pairs.append(BlindPair(brief, with_s, without_s, "left"))
            else:
                pairs.append(BlindPair(brief, without_s, with_s, "right"))
        rng.shuffle(pairs)
        return BlindStudy(pairs)

    def judge_sheet(self) -> list[dict]:
        """What the judge sees: no solver_side."""
        return [{"i": i, "brief": p.brief_id, "left": p.left, "right": p.right} for i, p in enumerate(self.pairs)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([asdict(p) for p in self.pairs], indent=2))

    @staticmethod
    def load(path: str | Path) -> "BlindStudy":
        return BlindStudy([BlindPair(**d) for d in json.loads(Path(path).read_text())])

    def score(self, held_out_briefs: set[str] | None = None) -> dict:
        ps = [p for p in self.pairs if p.choice and (held_out_briefs is None or p.brief_id in held_out_briefs)]
        wins = sum(p.choice == p.solver_side for p in ps)
        ties = sum(p.choice == "tie" for p in ps)
        n = len(ps) - ties
        rate = wins / n if n else 0.0
        return {"n": len(ps), "ties": ties, "solver_win_rate": rate,
                "prediction_held": rate >= THRESHOLDS["H5"]["predict_win_rate"],
                "switch": "keep the solver only as a checker" if n and rate < 0.5 else None}


# ---- H6 -----------------------------------------------------------------------

def h6(recheck_result: dict) -> dict:
    t = THRESHOLDS["H6"]
    r = recheck_result["rate"]
    return {**recheck_result, "prediction_held": r >= t["predict_live_rate"],
            "switch": "restrict to API-backed retailers" if r < t["kill_live_rate"] else None}
