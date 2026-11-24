from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mise.backend.app import create_app
from mise.demo import demo_scene
from mise.pipeline import ply
from mise.products.idol import IdolResult
from mise.products.listing import Listing
from mise.scene.store import SceneStore


@pytest.fixture
def scene():
    return demo_scene()


@pytest.fixture
def store(tmp_path):
    st = SceneStore(tmp_path / "scenes")
    st.create(demo_scene())
    return st


def synthetic_sofa_splat(path: Path, size=(1.0, 0.45, 0.4), n=6000, seed=0, y_up=True) -> Path:
    """A box-ish 'sofa' in generator units (y up, front toward +z), like an image-to-3D output."""
    rng = np.random.default_rng(seed)
    w, d, h = size
    seat = np.c_[rng.uniform(-w / 2, w / 2, n), rng.uniform(0, h * 0.5, n), rng.uniform(-d / 2, d / 2, n)]
    back = np.c_[rng.uniform(-w / 2, w / 2, n // 2), rng.uniform(h * 0.5, h, n // 2), rng.uniform(-d / 2, -d / 2 + d * 0.25, n // 2)]
    xyz = np.vstack([seat, back]).astype(np.float32)
    if not y_up:
        xyz = xyz[:, [0, 2, 1]]
    N = len(xyz)
    s = ply.Splat(xyz=xyz, f_dc=np.zeros((N, 3), np.float32), opacity=np.full(N, 3.0, np.float32),
                  scale=np.full((N, 3), np.log(0.005), np.float32), rot=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32))
    ply.write_ply(path, s)
    return path


class FakeGenerator:
    """Stands in for Idol: returns a synthetic splat at an arbitrary scale with a declared y-up."""

    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.calls = []

    def generate(self, image_url, out_dir, image_type="real", mesh=False):
        self.calls.append((image_url, image_type))
        out_dir.mkdir(parents=True, exist_ok=True)
        size = self.sizes[min(len(self.calls) - 1, len(self.sizes) - 1)]
        path = synthetic_sofa_splat(out_dir / "model.ply", size=size, seed=len(self.calls))
        return IdolResult(kind="splat", path=path, up_axis=(0, 1, 0), raw={"ply": {"url": "x"}})


def fake_listing(url="https://shop.example/sofa", dims='86"W x 38"D x 34"H', price=899.0):
    return Listing(url=url, retailer="shop.example", title="Example 3-seat sofa", price=price, currency="USD",
                   raw_dims=dims, images=["https://shop.example/img1.jpg", "https://shop.example/img2.jpg"])


def _no_network(url):
    raise OSError("no network in tests")


@pytest.fixture
def client(tmp_path):
    gen = FakeGenerator([(1.0, 0.44, 0.395)])
    app = create_app(tmp_path / "scenes", generator=gen, listing_fetcher=lambda url: fake_listing(url),
                     image_fetcher=_no_network)
    app.state.store.create(demo_scene())
    with TestClient(app) as c:
        c.generator = gen
        yield c
