import json

import numpy as np
import pytest

from mise.products.dimensions import DimensionParseError, parse_dimensions
from mise.products.importer import ImportRequest, NeedsInput, run_import
from mise.products.listing import parse_listing
from mise.products.orient import orient
from mise.products.scale_fit import fit_scale, gate_for

from .conftest import FakeGenerator, fake_listing


@pytest.mark.parametrize("raw,cat,expect", [
    ('84"W x 38"D x 34"H', "sofa", (2.1336, 0.9652, 0.8636)),
    ("W 220 x D 95 x H 80 cm", "sofa", (2.2, 0.95, 0.8)),
    ("Width: 84 in, Depth: 38 in, Height: 34 in", "sofa", (2.1336, 0.9652, 0.8636)),
    ('Overall: 33.5" H x 84" W x 37.75" D', "sofa", (2.1336, 0.9589, 0.8509)),
    ("7' 2\" W x 3' 2\" D x 2' 10\" H", "sofa", (2.1844, 0.9652, 0.8636)),
    ('35 1/2"W x 20"D x 30"H', "desk", (0.9017, 0.508, 0.762)),
    ("Assembled: 31.5 in L x 15.75 in W x 29.5 in H", "desk", (0.8001, 0.4001, 0.7493)),
    ("2200 x 950 x 800 mm", "sofa", (2.2, 0.95, 0.8)),
])
def test_parse_dimensions(raw, cat, expect):
    p = parse_dimensions(raw, cat)
    assert np.allclose(p.as_tuple(), expect, atol=2e-3), p
    assert p.plausible


def test_unlabeled_is_flagged_and_implausible_rejected():
    p = parse_dimensions("220 x 95 x 80 cm", "sofa")
    assert p.assumed_order
    bad = parse_dimensions("Height 2.5 m, width 0.8 m, depth 0.5 m", "sofa")
    assert not bad.plausible
    with pytest.raises(DimensionParseError):
        parse_dimensions("Free shipping on orders over $50", "sofa")


def test_gate_thresholds():
    assert gate_for(1.05) == "accept" and gate_for(1.10) == "accept"
    assert gate_for(1.2) == "warn" and gate_for(1.3) == "reject"


def test_fit_scale_uses_median_and_reports_anisotropy():
    f = fit_scale((2.2, 0.95, 0.8), (1.0, 0.45, 0.4))
    assert np.isclose(f.scale, np.median([2.2, 0.95 / 0.45, 2.0]))
    assert np.isclose(f.anisotropy, 2.2 / 2.0) and f.gate == "accept"


def test_fit_scale_detects_swapped_front():
    f = fit_scale((2.2, 0.95, 0.8), (0.45, 1.0, 0.4))
    assert f.swapped_wd and f.gate == "accept"


def test_orient_with_up_hint_gives_canonical_dims():
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(-1, 1, 5000), rng.uniform(0, 0.8, 5000), rng.uniform(-0.45, 0.45, 5000)]
    o = orient(pts, up_hint=np.array([0, 1.0, 0]), camera_dir=np.array([0, 0, -1.0]))
    assert np.allclose(o.dims, (2.0, 0.9, 0.8), atol=0.02)
    c = pts @ o.R.T + o.t
    assert abs(c[:, 2].min()) < 0.01  # footprint on z = 0


def test_parse_listing_jsonld():
    html = """<html><head><script type="application/ld+json">{"@context":"https://schema.org","@type":"Product",
    "name":"Linen sofa","image":["/a.jpg"],"offers":{"@type":"Offer","price":"1,299.00","priceCurrency":"USD",
    "availability":"https://schema.org/InStock"},"width":{"value":84,"unitCode":"INH"},
    "depth":{"value":38,"unitCode":"INH"},"height":{"value":34,"unitCode":"INH"}}</script></head>
    <body>Dimensions: 84"W x 38"D x 34"H</body></html>"""
    lst = parse_listing(html, "https://www.shop.example/p/1")
    assert lst.title == "Linen sofa" and lst.price == 1299.0 and lst.currency == "USD" and lst.in_stock
    assert lst.images == ["https://www.shop.example/a.jpg"] and lst.retailer == "shop.example"
    p = parse_dimensions(lst.raw_dims, "sofa")
    assert np.allclose(p.as_tuple(), (2.1336, 0.9652, 0.8636), atol=1e-3)


def test_import_accepts_and_writes_canonical_metric_splat(tmp_path):
    gen = FakeGenerator([(1.0, 0.44, 0.395)])
    res = run_import(ImportRequest(url="u", category="sofa"), tmp_path / "assets", gen, listing=fake_listing())
    a = res.asset
    assert a.status == "ready" and a.asset_kind == "splat" and a.anisotropy <= 1.10
    assert np.allclose(a.dims.as_tuple(), (2.18, 0.965, 0.864), rtol=0.10)  # within 10% per axis
    from mise.pipeline.ply import read_ply
    s = read_ply(tmp_path / a.asset)
    assert abs(s.xyz[:, 2].min()) < 0.03 and abs(s.xyz[:, 0].mean()) < 0.1


def test_import_retries_then_falls_back_to_box(tmp_path):
    gen = FakeGenerator([(1.0, 0.9, 0.2), (1.0, 0.9, 0.2)])  # wrong proportions, anisotropy >> 1.25
    res = run_import(ImportRequest(url="u", category="sofa"), tmp_path / "assets", gen, listing=fake_listing())
    assert len(gen.calls) == 2 and res.asset.status == "fallback_box" and res.asset.asset_kind == "box"
    assert np.allclose(res.asset.dims.as_tuple(), (2.1844, 0.9652, 0.8636), atol=1e-3)


def test_import_asks_when_dimensions_missing(tmp_path):
    with pytest.raises(NeedsInput):
        run_import(ImportRequest(url="u", category="sofa"), tmp_path, FakeGenerator([(1, 1, 1)]),
                   listing=fake_listing(dims=None))


def test_idol_on_colab_runs_the_launcher_and_returns_the_splat(tmp_path):
    import json

    from mise.products.idol import IdolColab

    from .conftest import synthetic_sofa_splat

    img = tmp_path / "chair.jpg"
    img.write_bytes(b"\xff\xd8fake")
    seen = {}

    def runner(cmd, log):
        seen["cmd"] = cmd
        inp = cmd[cmd.index("--idol") + 1]
        seen["jobs"] = json.loads((tmp_path / "gen" / "idol-in" / "images.json").read_text())
        out = tmp_path / "gen" / "idol-out"
        out.mkdir(parents=True)
        synthetic_sofa_splat(out / "product.ply")
        (out / "meta.json").write_text(json.dumps({"items": [{"seconds": 50}]}))
        log.write_text("ok")
        return 0 if inp else 1

    res = IdolColab(repo=tmp_path, runner=runner).generate(str(img), tmp_path / "gen", image_type="synthetic")
    assert res.kind == "splat" and res.path.exists() and res.up_axis is None
    assert seen["jobs"] == [{"name": "product.jpg", "mode": "synthetic"}]
    assert "--num-gaussians" in seen["cmd"] and seen["cmd"][2].startswith("idol-")
