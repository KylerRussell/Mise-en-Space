"""Minimal GLB reader: vertex positions of every mesh primitive, with node transforms."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMP = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T  # glTF is column-major
    M = np.eye(4)
    if "scale" in node:
        M = np.diag([*node["scale"], 1.0]) @ M
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        M = T @ M
    if "translation" in node:
        T = np.eye(4)
        T[:3, 3] = node["translation"]
        M = T @ M
    return M


def read_glb_positions(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    magic, _version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("not a GLB file")
    off = 12
    gltf, binchunk = None, b""
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif ctype == 0x004E4942:
            binchunk = chunk
        off += 8 + clen
    if gltf is None:
        raise ValueError("GLB has no JSON chunk")

    def accessor(i: int) -> np.ndarray:
        acc = gltf["accessors"][i]
        bv = gltf["bufferViews"][acc["bufferView"]]
        n = _NCOMP[acc["type"]]
        dt = np.dtype("<" + _COMP[acc["componentType"]])
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride") or dt.itemsize * n
        count = acc["count"]
        raw = np.frombuffer(binchunk, dtype=np.uint8, count=stride * (count - 1) + dt.itemsize * n, offset=start)
        out = np.lib.stride_tricks.as_strided(raw, shape=(count, n * dt.itemsize), strides=(stride, 1)).copy()
        return out.view(dt).reshape(count, n).astype(np.float64)

    pts: list[np.ndarray] = []

    def visit(ni: int, parent: np.ndarray) -> None:
        node = gltf["nodes"][ni]
        M = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                if "POSITION" in prim["attributes"]:
                    p = accessor(prim["attributes"]["POSITION"])
                    pts.append(p @ M[:3, :3].T + M[:3, 3])
        for c in node.get("children", []):
            visit(c, M)

    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)] if gltf.get("scenes") else None
    roots = scene.get("nodes", []) if scene else list(range(len(gltf.get("nodes", []))))
    for r in roots:
        visit(r, np.eye(4))
    if not pts:
        raise ValueError("GLB has no mesh positions")
    return np.vstack(pts)
