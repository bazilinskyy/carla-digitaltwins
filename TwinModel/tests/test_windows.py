"""Tests for window geometry and interiors (twinmodel.export.windows) and their export (realbuildings.apply)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from twinmodel.export import facades as fc
from twinmodel.export import windows as wm


def _wall(x0, x1, z0, z1, y=0.0):
    """A vertical wall along +x at y, facing -y (south), CCW seen from outside."""
    ring = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], float)
    return ring, np.array([0.0, -1.0, 0.0])


def test_layout_matches_the_painted_grid():
    st = fc.STYLES["brick_postwar"]
    ring, n = _wall(0.0, 12.0, 0.0, 2 * st.storey_m)
    ops = wm.layout(ring, n, 0.0, 2 * st.storey_m, st.storey_m, st)
    assert len(ops) == 4 * 2                      # 12 m / 3.0 m bays, two storeys
    along = fc.wall_frame(n)
    a0 = float((ring @ along).min())
    bay = 12.0 / 4
    o = next(o for o in ops if o.bay == 1 and o.storey == 1)
    x0, x1, y0, y1 = st.window
    assert o.u0 == pytest.approx(a0 + (1 + x0) * bay) and o.u1 == pytest.approx(a0 + (1 + x1) * bay)
    assert o.z0 == pytest.approx(st.storey_m * (1 + y0)) and o.z1 == pytest.approx(st.storey_m * (1 + y1))
    # the same window in the texture's UV space lands on the painted window's pixels (wall_uv / _window_rect)
    assert along == pytest.approx([1.0, 0.0, 0.0])
    uv = fc.wall_uv(np.array([[o.u0, 0, o.z0], [o.u1, 0, o.z1]]), n, 0.0, st.storey_m, st.bay_m, ring=ring)
    px = fc.TEX_PX
    wx0, wx1, wy0, wy1, _ = fc._window_rect(st, 0, 1, 0, px / fc.TEX_BAYS, px / fc.TEX_STOREYS)
    assert (uv[0, 0] % 1.0) * px == pytest.approx(wx0, abs=2)
    assert (uv[1, 0] % 1.0) * px == pytest.approx(wx1, abs=2)


def test_blank_and_sloped_walls_get_no_windows():
    ring, n = _wall(0, 10, 0, 6)
    assert wm.layout(ring, n, 0, 6, 3.0, fc.STYLES["blank_brick"]) == []
    tilted = np.array([0.0, -0.8, 0.6])
    assert wm.layout(ring, tilted, 0, 6, 3.0, fc.STYLES["brick_postwar"]) == []


def test_openings_stay_inside_a_gable():
    st = fc.STYLES["brick_old"]
    # a gable wall: 8 m wide, eaves at 6.4 m, ridge at 10 m
    ring = np.array([[0, 0, 0], [8, 0, 0], [8, 0, 6.4], [4, 0, 10.0], [0, 0, 6.4]], float)
    n = np.array([0.0, -1.0, 0.0])
    ops = wm.layout(ring, n, 0.0, 10.0, st.storey_m, st)
    gable = Polygon([(p[0], p[2]) for p in ring])
    assert ops
    for o in ops:
        assert gable.contains(Polygon([(o.u0, o.z0), (o.u1, o.z0), (o.u1, o.z1), (o.u0, o.z1)]))


def test_shopfront_door_bay_follows_the_texture():
    st = fc.STYLES["shopfront"]
    ring, n = _wall(0, 16, 0, fc.SHOP_FLOOR_M)
    for variant in (0, 1):
        ops = wm.layout(ring, n, 0.0, fc.SHOP_FLOOR_M, st.storey_m, st, variant)
        doors = [o.bay for o in ops if o.door]
        assert doors and all(b % fc.TEX_BAYS == (1 + variant) % fc.TEX_BAYS for b in doors)
        assert all(o.z0 == pytest.approx(0.01 * st.storey_m) for o in ops if o.door)


def test_cut_keeps_the_wall_area_minus_the_openings():
    from twinmodel.export.realbuildings import triangulate
    st = fc.STYLES["brick_modern"]
    ring, n = _wall(0, 11.2, 0, 3 * st.storey_m)
    ops = wm.layout(ring, n, 0, 3 * st.storey_m, st.storey_m, st)
    cut = wm.cut_openings([ring], n, ops)
    area = 0.0
    for poly in cut:
        v, tri, nn = triangulate(poly)
        a = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
        area += 0.5 * np.linalg.norm(a, axis=1).sum()
        assert np.allclose(v[:, 1], 0.0)                       # on the wall plane
        assert nn @ n > 0.99                                     # same facing
    assert area == pytest.approx(11.2 * 3 * st.storey_m - sum(o.width * o.height for o in ops), rel=1e-6)


def test_module_parts_and_local_frame():
    st = fc.STYLES["brick_old"]
    parts = wm.module_parts(st)
    assert set(parts) == {"reveal", "frame", "glass", "sill"}
    w, h = wm.nominal_size(st)
    gv = parts["glass"][0]
    assert gv[:, 1].min() > 0                                   # the glass sits behind the wall plane (+y = in)
    assert gv[:, 0].min() >= -w / 2 - 1e-9 and gv[:, 0].max() <= w / 2 + 1e-9
    sv = parts["sill"][0]
    assert sv[:, 1].min() < 0 and sv[:, 2].max() <= 1e-9        # the sill sticks out, below the opening
    assert parts["glass"][2].min() >= -1e-9 and parts["glass"][2].max() <= 1 + 1e-9   # 0-1 UVs
    door = wm.module_parts(fc.STYLES["shopfront"], door=True)
    assert "sill" not in door and door["glass"][0][:, 2].min() == pytest.approx(0.0)
    # modern glass curtain walls: no reveal deeper than the frame
    assert wm.REVEAL_M["office_glass"] < wm.FRAME_DEPTH_M


@pytest.mark.parametrize("normal", [(0, -1, 0), (1, 0, 0), (-0.6, 0.8, 0), (0.28, 0.96, 0)])
def test_instance_transform_places_the_module_on_the_wall(normal):
    from twinmodel.export.ue import model_to_ue
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    along = fc.wall_frame(n)
    origin = np.array([5.0, -3.0, 2.0])
    ring = np.array([origin, origin + 9 * along, origin + 9 * along + [0, 0, 6], origin + [0, 0, 6]])
    st = fc.STYLES["brick_postwar"]
    ops = wm.layout(ring, n, 2.0, 8.0, st.storey_m, st)
    o = ops[0]
    x, y, z, yaw, sx, sz = wm.instance_transform(o, n, ring[0], st)
    w_nom, h_nom = wm.nominal_size(st)
    assert sx == pytest.approx(o.width / w_nom, rel=1e-3) and sz == pytest.approx(o.height / h_nom, rel=1e-3)
    # a module-local point (model frame: x along, +y into the wall, z up) through the glb axes and the UE yaw
    q = np.array([0.3, 0.1, 0.5])
    local_ue = np.array([q[0] * sx, -q[1], q[2] * sz]) * 100        # glb: (x, -y, z) * 100, scaled
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    world_ue = np.array([c * local_ue[0] - s * local_ue[1], s * local_ue[0] + c * local_ue[1], local_ue[2]]) + [x, y, z]
    centre = origin + ((o.u0 + o.u1) / 2 - origin @ along) * along + [0, 0, o.z0 - origin[2]]
    expect = centre + q[0] * sx * along - q[1] * n + [0, 0, q[2] * sz]
    assert world_ue == pytest.approx(model_to_ue(expect)[0], abs=0.05)


def test_night_data_is_stable_and_spread():
    st = fc.STYLES["brick_postwar"]
    ring, n = _wall(0, 30, 0, 9)
    ops = wm.layout(ring, n, 0, 9, st.storey_m, st)
    a = [wm.night_data("b1", 0, o) for o in ops]
    assert a == [wm.night_data("b1", 0, o) for o in ops]
    ranks = [r for r, _, _ in a]
    assert 0 <= min(ranks) and max(ranks) <= 1 and len(set(ranks)) == len(ranks)
    assert all(0.5 <= b <= 1.0 for _, _, b in a)


def test_interiors_by_use():
    assert wm.interior_of("brick_postwar") == "home"
    assert wm.interior_of("office_glass") == "office"
    assert wm.interior_of("industrial") is None
    assert wm.PROVENANCE_WINDOWS == "inferred" and wm.PROVENANCE_INTERIORS == "generated"


def test_facade_texture_without_windows():
    st = fc.STYLES["brick_postwar"]
    painted, _, _ = fc.synth_facade(st, 0, px=128)
    plain, _, _ = fc.synth_facade(st, 0, px=128, windows=False)
    glass = np.all(np.abs(painted.astype(int) - np.array(st.glass)) < 40, axis=-1).mean()
    glass_plain = np.all(np.abs(plain.astype(int) - np.array(st.glass)) < 40, axis=-1).mean()
    assert glass > 0.05 and glass_plain < 0.01
    assert fc.material_key("brick_postwar", 1, windows=False) == "facadenw_brick_postwar_1"
    assert fc.parse_material_key("facadenw_brick_postwar_1") == ("brick_postwar", 1, False)
    assert fc.parse_material_key("facade_office_glass_0") == ("office_glass", 0, True)


def _apply_nyc(tmp_path, **kw):
    """The NYC cache fixture of tests/test_realbuildings.py, then realbuildings.apply with ``kw``."""
    from tests.test_realbuildings import _box, _manifest, _write, MIDTOWN
    from pyproj import Transformer
    from twinmodel.export import realbuildings as rb
    from twinmodel.ingest import buildings, citygml
    to_file = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    x, y = to_file.transform(-73.984, 40.7549)
    ft = 1 / citygml.US_FT
    raw = []
    for gid, bin_, dx, top in (("g1", "1026885", 0.0, 400.0), ("g2", "1000000", 120.0, 60.0)):
        surf = []
        for rings, kind in _box(x + dx * ft, y, x + (dx + 30) * ft, y + 30 * ft, 50.0, 50.0 + top):
            surf.append([kind, [np.round(r, 3).ravel().tolist() for r in rings]])
        raw.append({"id": gid, "attrs": {"BIN": bin_, "DOITT_ID": gid[1:]}, "surfaces": surf})
    tag = "%.5f_%.5f_%.5f_%.5f" % MIDTOWN
    build = tmp_path / "b"
    _write(build / "nyc3d" / f"citygml_{tag}.json.gz", raw)
    _write(build / "nyc3d" / f"footprints_{tag}.json.gz", {"1026885": ["1010180026", 1930]})
    _write(build / "nyc3d" / f"pluto_{tag}.json.gz", {"1010180026": {
        "bldgclass": "O4", "yearbuilt": "1931", "numfloors": "34", "officearea": "200000", "retailarea": "8000"}})
    man = _manifest()
    blds, src = buildings.load("nyc3d", build, man)
    (build / "ue" / "meshes").mkdir(parents=True)
    return build, man, rb.apply(build, man, blds, src, texture_px=64, roof_px=64, **kw)


def test_apply_dry_run_counts_and_writes_nothing(tmp_path):
    build, man, stats = _apply_nyc(tmp_path, dry_run=True, interiors=True)
    assert stats["windows"] > 100 and stats["window_instances"] >= 3 * stats["windows"]
    assert stats["shop_windows"] > 0 and stats["doors"] > 0
    assert stats.get("interiors_office", 0) + stats.get("interiors_home", 0) > 0
    assert not (build / "ue" / "manifest.json").exists()
    assert not list((build / "ue" / "meshes").glob("*.glb"))


def test_apply_window_geometry_and_interiors(tmp_path):
    build, man, stats = _apply_nyc(tmp_path, window_geometry=True, interiors=True)
    man = json.loads((build / "ue" / "manifest.json").read_text())
    mods, inst = man["window_modules"], man["window_instances"]
    assert mods and inst
    assert all((build / "ue" / m["file"]).exists() and m["provenance"] == "inferred" for m in mods.values())
    parts = {m["part"] for m in mods.values()}
    assert {"frame", "glass", "reveal"} <= parts
    for e in inst:
        assert e["mesh"] in mods and len(e["transforms"]) and all(len(t) == 6 for t in e["transforms"])
        if e["material"].startswith("interior_"):
            assert e["provenance"] == "generated" and len(e["custom"]) == len(e["transforms"])
            assert man["textures"][e["material"]]["interior"] in ("home", "office", "shop")
        else:
            assert e["provenance"] == "inferred"
            assert e["material"] in man["textures"] or e["material"].startswith("facadenw_")
    # walls with windows use the textures without painted windows, which have no night mask
    nw = [k for k in man["textures"] if k.startswith("facadenw_")]
    assert nw and all("windows" not in man["textures"][k] for k in nw)
    assert any(a["material"].startswith("facadenw_") for a in man["assets"])
    assert man["stats"]["window_instances"] == sum(len(e["transforms"]) for e in inst)
    assert man["stats"]["interiors"] is True
