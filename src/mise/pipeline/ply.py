"""Read, write and transform 3D Gaussian Splatting `.ply` files with numpy only.

Standard 3DGS layout (Inria / Nerfstudio `ns-export gaussian-splat`):
x y z [nx ny nz] f_dc_0..2 f_rest_0..N opacity scale_0..2 rot_0..3
with log scales, logit opacity and quaternions in (w, x, y, z) order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
}
SH_C0 = 0.28209479177387814


@dataclass
class Splat:
    """A set of Gaussians. Arrays are (N, k); `extra` holds any other vertex fields."""

    xyz: np.ndarray
    f_dc: np.ndarray | None = None
    f_rest: np.ndarray | None = None
    opacity: np.ndarray | None = None  # logit, (N,)
    scale: np.ndarray | None = None  # log, (N, 3)
    rot: np.ndarray | None = None  # (w, x, y, z), (N, 4)
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.xyz)

    @property
    def is_gaussian(self) -> bool:
        return self.scale is not None and self.rot is not None

    def subset(self, mask: np.ndarray) -> "Splat":
        def s(a):
            return None if a is None else a[mask]
        return Splat(s(self.xyz), s(self.f_dc), s(self.f_rest), s(self.opacity), s(self.scale), s(self.rot),
                     {k: v[mask] for k, v in self.extra.items()})

    @staticmethod
    def concat(parts: list["Splat"]) -> "Splat":
        def c(name):
            vals = [getattr(p, name) for p in parts]
            if any(v is None for v in vals):
                return None
            return np.concatenate(vals)
        keys = set.intersection(*(set(p.extra) for p in parts)) if parts else set()
        return Splat(c("xyz"), c("f_dc"), c("f_rest"), c("opacity"), c("scale"), c("rot"),
                     {k: np.concatenate([p.extra[k] for p in parts]) for k in keys})

    def rgb(self) -> np.ndarray:
        """Base color in [0, 1] from the SH DC term."""
        if self.f_dc is None:
            return np.full((len(self), 3), 0.5)
        return np.clip(self.f_dc * SH_C0 + 0.5, 0, 1)

    def set_rgb(self, rgb: np.ndarray) -> None:
        self.f_dc = (np.clip(rgb, 0, 1) - 0.5) / SH_C0

    def alpha(self) -> np.ndarray:
        if self.opacity is None:
            return np.ones(len(self))
        return 1 / (1 + np.exp(-self.opacity))


def read_ply(path: str | Path) -> Splat:
    data = Path(path).read_bytes()
    end = data.index(b"end_header") + len(b"end_header")
    while data[end:end + 1] in (b"\r", b"\n"):
        end += 1
    header = data[:end].decode("ascii", errors="replace").splitlines()
    fmt, n, props, in_vertex = None, 0, [], False
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError("list properties in the vertex element are not supported")
            props.append((parts[2], _PLY_TYPES[parts[1]]))
    if fmt == "ascii":
        arr = np.loadtxt(data[end:].decode().splitlines()[:n], dtype=np.float64, ndmin=2)
        rec = {name: arr[:, i] for i, (name, _) in enumerate(props)}
    else:
        order = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(name, order + t) for name, t in props])
        raw = np.frombuffer(data, dtype=dt, count=n, offset=end)
        rec = {name: raw[name].astype(np.float32) for name, _ in props}
    names = [p[0] for p in props]

    def take(prefix: str, count: int | None = None) -> np.ndarray | None:
        cols = sorted((k for k in names if k.startswith(prefix)), key=lambda k: int(k[len(prefix):]))
        if not cols or (count and len(cols) != count):
            return None
        return np.stack([rec[c] for c in cols], axis=1).astype(np.float32)

    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    f_dc = take("f_dc_", 3)
    if f_dc is None and {"red", "green", "blue"} <= set(names):
        rgb = np.stack([rec["red"], rec["green"], rec["blue"]], axis=1) / 255.0
        f_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    used = {"x", "y", "z", "nx", "ny", "nz", "opacity", "red", "green", "blue"} | {
        k for k in names if k.startswith(("f_dc_", "f_rest_", "scale_", "rot_"))}
    return Splat(
        xyz=xyz, f_dc=f_dc, f_rest=take("f_rest_"),
        opacity=rec["opacity"].astype(np.float32) if "opacity" in rec else None,
        scale=take("scale_", 3), rot=take("rot_", 4),
        extra={k: rec[k].astype(np.float32) for k in names if k not in used},
    )


def write_ply(path: str | Path, s: Splat) -> None:
    cols: list[tuple[str, np.ndarray]] = [("x", s.xyz[:, 0]), ("y", s.xyz[:, 1]), ("z", s.xyz[:, 2])]
    if s.is_gaussian:
        zeros = np.zeros(len(s), np.float32)
        cols += [("nx", zeros), ("ny", zeros), ("nz", zeros)]
    if s.f_dc is not None:
        cols += [(f"f_dc_{i}", s.f_dc[:, i]) for i in range(3)]
    if s.f_rest is not None:
        cols += [(f"f_rest_{i}", s.f_rest[:, i]) for i in range(s.f_rest.shape[1])]
    if s.opacity is not None:
        cols.append(("opacity", s.opacity))
    if s.scale is not None:
        cols += [(f"scale_{i}", s.scale[:, i]) for i in range(3)]
    if s.rot is not None:
        cols += [(f"rot_{i}", s.rot[:, i]) for i in range(4)]
    cols += list(s.extra.items())
    dt = np.dtype([(name, "<f4") for name, _ in cols])
    arr = np.empty(len(s), dtype=dt)
    for name, v in cols:
        arr[name] = v
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(s)}\n" + "".join(
        f"property float {name}\n" for name, _ in cols) + "end_header\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


# ---- transforms ---------------------------------------------------------------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z); broadcasts over leading dims."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180°: rotate about any axis orthogonal to a
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def transform(s: Splat, R: np.ndarray | None = None, t: np.ndarray | None = None, scale: float = 1.0,
              drop_sh_rest: bool = False) -> Splat:
    """Apply x' = scale * R x + t to positions, Gaussian rotations and scales.

    Higher-order SH (f_rest) are *not* rotated; for pure yaw or small
    rotations the view-dependent error is minor. Pass ``drop_sh_rest`` to
    zero them when the rotation is large (e.g. Y-up to Z-up).
    """
    R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
    t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64)
    out = Splat(
        xyz=(scale * s.xyz.astype(np.float64) @ R.T + t).astype(np.float32),
        f_dc=None if s.f_dc is None else s.f_dc.copy(),
        f_rest=None if s.f_rest is None else (np.zeros_like(s.f_rest) if drop_sh_rest else s.f_rest.copy()),
        opacity=None if s.opacity is None else s.opacity.copy(),
        scale=None if s.scale is None else (s.scale + np.log(scale)).astype(np.float32),
        rot=None,
        extra={k: v.copy() for k, v in s.extra.items()},
    )
    if s.rot is not None:
        q = rotmat_to_quat(R)
        r = quat_mul(np.broadcast_to(q, s.rot.shape), s.rot.astype(np.float64))
        out.rot = (r / np.linalg.norm(r, axis=1, keepdims=True)).astype(np.float32)
    return out


def gaussian_extent(s: Splat, sigmas: float = 2.0) -> np.ndarray:
    """Per-Gaussian radius (largest axis), in the same units as xyz."""
    if s.scale is None:
        return np.zeros(len(s))
    return sigmas * np.exp(s.scale.max(axis=1))
