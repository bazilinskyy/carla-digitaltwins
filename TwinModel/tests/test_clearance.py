"""Tests for twinmodel.export.clearance: building meshes cut clear of the road and sidewalk corridor."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from twinmodel.export import clearance as C
from twinmodel.export.ue import read_glb, write_glb


# --------------------------------------------------------------------------- a tiny bake export

def _slab(x0, y0, x1, y1, z=0.0):
    """Horizontal rectangle as two triangles (CCW from above)."""
    v = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    return v, np.array([[0, 1, 2], [0, 2, 3]])


def _box(x0, y0, x1, y1, z0, z1, *, bottom=False, grid=None):
    """A closed box (walls + roof, + bottom face when ``bottom``) as (positions, normals, uvs, faces); walls
    split into ``grid`` m quads so a cut has something to bite."""
    vs, ns, us, fs = [], [], [], []
    n = 0

    def quad(a, b, c, d, nrm):
        nonlocal n
        q = np.array([a, b, c, d], float)
        vs.append(q)
        ns.append(np.repeat(np.array(nrm, float)[None], 4, axis=0))
        us.append(q[:, :2] * 0.1 + q[:, 2:3] * 0.05)
        fs.append(np.array([[0, 1, 2], [0, 2, 3]]) + n)
        n += 4
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1]):
        L = math.hypot(bx - ax, by - ay)
        k = max(1, int(round(L / grid))) if grid else 1
        nrm = ((by - ay) / L, -(bx - ax) / L, 0.0)
        for i in range(k):
            pa = (ax + (bx - ax) * i / k, ay + (by - ay) * i / k)
            pb = (ax + (bx - ax) * (i + 1) / k, ay + (by - ay) * (i + 1) / k)
            quad((*pa, z0), (*pb, z0), (*pb, z1), (*pa, z1), nrm)
    quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1))
    if bottom:
        quad((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (0, 0, -1))
    return np.vstack(vs), np.vstack(ns), np.vstack(us), np.vstack(fs)


def _bld(bid, x0, y0, x1, y1, base, roof, style="render_modern"):
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return {"id": bid, "real": "test", "raw_ring_ue": [[x * 100, -y * 100] for x, y in ring],
            "rings_ue": [[[x * 100, -y * 100] for x, y in ring]], "base_z_cm": base * 100,
            "eaves_z_cm": roof * 100, "roof_z_cm": roof * 100, "style": style}


def make_export(tmp_path, buildings, road=(-60, -3.5, 60, 3.5), road_z=0.0, bottom=False, extra_assets=()):
    """<tmp>/twin/ue with one drivable strip and the given boxes (id, x0, y0, x1, y1, base, roof)."""
    build = tmp_path / "twin"
    out = build / "ue"
    (out / "meshes").mkdir(parents=True)
    assets = []
    v, f = _slab(*road, z=road_z)
    bb = write_glb(out / "meshes/T_L0_drivable_0_0.glb", "T_L0_drivable_0_0", v, np.tile([0, 0, 1.0], (4, 1)),
                   v[:, :2], f, "road")
    assets.append({"file": "meshes/T_L0_drivable_0_0.glb", "asset": "T_L0_drivable_0_0", "layer": 0,
                   "kind": "drivable", "material": "road", "tile": [0, 0], "triangles": 2, "bbox_ue": bb})
    blds = []
    for k, (bid, x0, y0, x1, y1, base, roof) in enumerate(buildings):
        p, n, u, fc = _box(x0, y0, x1, y1, base, roof, bottom=bottom, grid=2.0)
        name = f"T_L0_building_0_0_b{k}"
        bb = write_glb(out / f"meshes/{name}.glb", name, p, n, u, fc, "facade_render_modern_0")
        assets.append({"file": f"meshes/{name}.glb", "asset": name, "layer": 0, "kind": "building_3dbag",
                       "material": "facade_render_modern_0", "texture": "facade_render_modern_0", "tile": [0, 0],
                       "triangles": int(len(fc)), "bbox_ue": bb})
        blds.append(_bld(bid, x0, y0, x1, y1, base, roof))
    assets.extend(extra_assets)
    man = {"name": "T", "tile_m": 250.0, "assets": assets, "buildings": blds, "stats": {},
           "textures": {"facade_render_modern_0": {"base_color": "textures/x.png"}},
           "geo_reference": "+proj=tmerc +lat_0=52 +lon_0=5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"}
    (out / "manifest.json").write_text(json.dumps(man))
    return build


def lanes_along_x(y=0.0, z=0.0, x0=-55, x1=55, kind="Driving", width=3.5, lane=1):
    xs = np.arange(x0, x1 + 0.01, 1.0)
    n = len(xs)
    return C.Lanes(np.column_stack([xs, np.full(n, y), np.full(n, z)]), np.zeros(n), np.full(n, width),
                   np.full(n, 7), np.full(n, lane), xs - x0, np.array([kind] * n, dtype=object))


def merged(lanes_list):
    return C.Lanes(*[np.concatenate([getattr(l, f) for l in lanes_list])
                     for f in ("xyz", "heading", "width", "road", "lane", "s", "kind")])


def building_tris(build):
    man = json.loads((build / "ue/manifest.json").read_text())
    return C.building_triangles(build / "ue", man), man


def volumes(build, shift=(37.0, -21.0, 5.0)):
    """Signed volume of the building surface about two origins: equal only when the surface is closed
    (T-junctions allowed, unlike trimesh's watertight test)."""
    tris, _ = building_tris(build)

    def vol(t):
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
    return vol(tris), vol(tris - np.asarray(shift))


def assert_closed(build, expect=None):
    v0, v1 = volumes(build)
    assert abs(v0 - v1) < 1e-3 * max(1.0, abs(v0)), f"the surface is open: {v0:.2f} vs {v1:.2f}"
    if expect is not None:
        assert abs(v0 - expect) < 0.5, f"volume {v0:.2f}, expected {expect:.2f}"


def blocked_in(tris, poly, z0, z1, n=24):
    """Building triangles with a point strictly inside poly x (z0, z1) (dense barycentric samples)."""
    a, b = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    keep = a + b <= 1.0
    w = np.column_stack([1 - a[keep] - b[keep], a[keep], b[keep]])
    inner = poly.buffer(-0.01)
    k = 0
    for t in tris:
        if t[:, 2].min() >= z1 or t[:, 2].max() <= z0:
            continue
        p = w @ t
        m = (p[:, 2] > z0 + 1e-3) & (p[:, 2] < z1 - 1e-3)
        if m.any() and shapely.contains_xy(inner, p[m, 0], p[m, 1]).any():
            k += 1
    return k


# --------------------------------------------------------------------------- helpers

def test_parse_maxheight():
    assert C.parse_maxheight("3.2") == 3.2
    assert C.parse_maxheight("3,9 m") == 3.9
    assert abs(C.parse_maxheight("12'6\"") - 3.81) < 0.01
    assert abs(C.parse_maxheight("14 ft") - 4.267) < 0.01
    assert C.parse_maxheight("default") is None and C.parse_maxheight("below_default") is None


def test_split_line_and_z_keep_area():
    tri = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [4, 0, 0, 1, 0, 0, 0, 1], [0, 4, 4, 0, 1, 0, 0, 1]], float)
    L, R = C._split_line(tri, np.array([1.0, -5]), np.array([1.0, 5]))
    assert L is not None and R is not None
    assert abs(C._area3(L) + C._area3(R) - C._area3(tri)) < 1e-9
    lo, hi = C._split_z(tri, 2.0)
    assert abs(C._area3(lo) + C._area3(hi) - C._area3(tri)) < 1e-9
    # attributes interpolate linearly: u follows x
    assert np.allclose(L[:, 3], L[:, 0] / 4.0) and np.allclose(hi[:, 4], hi[:, 1] / 4.0)


# --------------------------------------------------------------------------- the cut

def test_passage_through_building_is_clear_and_closed(tmp_path):
    """A 20 m building across the road: below 4.6 m the corridor is empty, the cut is closed (watertight)
    with a soffit at the clearance and walls along the road edges, in the plain wall texture."""
    build = make_export(tmp_path, [("B1", -10, -15, 10, 15, -0.3, 20.0)], bottom=True)
    assert_closed(build, 20 * 30 * 20.3)
    lanes = lanes_along_x()
    before = C.check(build, lanes)
    assert before["road"]["crossings"] >= 2 and before["road"]["blocked"] + before["road"]["inside"] >= 15
    st = C.apply(build, lanes=lanes, log=lambda *a: None)
    assert st["passage"] == 1 and st["trim"] == 0 and st["soffit_m2"] > 100
    tris, man = building_tris(build)
    road = box(-60, -3.5, 60, 3.5)
    assert blocked_in(tris, road.buffer(-0.05), 0.0, 4.5) == 0
    zs = tris[:, :, 2]
    assert np.isclose(zs, 4.6).any()          # the soffit
    assert_closed(build, 20 * 30 * 20.3 - 20 * (7.0 + 2 * C.ROAD_BUFFER_M) * (4.6 + 0.3))
    after = C.check(build, lanes)
    assert after["road"]["blocked"] == 0
    cut = [a for a in man["assets"] if "_cut_" in a["asset"]]
    assert cut and all(a["texture"].startswith("plain_") for a in cut)
    assert all(a["texture"] in man["textures"] for a in cut)
    assert (build / "ue" / man["textures"][cut[0]["texture"]]["base_color"]).exists()


def test_apply_twice_gives_the_same_meshes(tmp_path):
    build = make_export(tmp_path, [("B1", -10, -15, 10, 15, -0.3, 20.0)])
    C.apply(build, lanes=lanes_along_x(), log=lambda *a: None)
    t1, m1 = building_tris(build)
    C.apply(build, lanes=lanes_along_x(), log=lambda *a: None)
    t2, m2 = building_tris(build)
    assert len(t1) == len(t2) and np.allclose(np.sort(t1.reshape(-1)), np.sort(t2.reshape(-1)))
    assert sorted(a["asset"] for a in m1["assets"]) == sorted(a["asset"] for a in m2["assets"])
    assert (build / "ue/meshes/uncut").is_dir()
    # restore gives back the uncut export
    man = json.loads((build / "ue/manifest.json").read_text())
    C.restore(build / "ue", man)
    assert "clearance" not in man and not any("_cut_" in a["asset"] for a in man["assets"])


def test_sliver_is_trimmed_to_the_kerb(tmp_path):
    """A footprint 1 m into the carriageway: cut back over the whole height, no overhang left."""
    build = make_export(tmp_path, [("S", -10, 2.5, 10, 15, -0.3, 15.0)], bottom=True)
    st = C.apply(build, lanes=lanes_along_x(), log=lambda *a: None)
    assert st["trim"] == 1 and st["passage"] == 0
    tris, _ = building_tris(build)
    corridor = box(-60, -3.5, 60, 3.5 + C.ROAD_BUFFER_M)
    assert blocked_in(tris, corridor.buffer(-0.05), -1.0, 100.0) == 0
    assert_closed(build, 20 * 12.5 * 15.3 - 20 * (3.5 + C.ROAD_BUFFER_M - 2.5) * 15.3)


def test_grazing_facade_is_left_alone(tmp_path):
    """A facade exactly on the road edge (the buffer reaches 0.1 m into it): nothing changes."""
    build = make_export(tmp_path, [("G", -10, 3.5, 10, 15, -0.3, 15.0)])
    st = C.apply(build, lanes=lanes_along_x(), log=lambda *a: None)
    assert st["changed_assets"] == 0 and st["added_assets"] == 0


def test_low_building_on_the_road_is_removed(tmp_path):
    """A 3 m kiosk across the carriageway: too low to pass under, removed inside the corridor."""
    build = make_export(tmp_path, [("K", -3, -6, 3, 6, -0.3, 3.0)])
    st = C.apply(build, lanes=lanes_along_x(), log=lambda *a: None)
    assert st["trim"] == 1
    tris, _ = building_tris(build)
    assert blocked_in(tris, box(-60, -3.4, 60, 3.4), -1.0, 50.0) == 0
    assert C.check(build, lanes_along_x())["road"]["blocked"] == 0


def test_sidewalk_lane_gets_a_low_passage(tmp_path):
    """A sidewalk lane running 1 m inside a facade: a 2.5 m arcade notch, the facade above stays."""
    build = make_export(tmp_path, [("W", -10, 5.0, 10, 15, -0.3, 15.0)])
    walk = lanes_along_x(y=5.5, z=0.15, kind="Sidewalk", width=2.0, lane=2)
    lanes = merged([lanes_along_x(), walk])
    assert C.check(build, lanes)["walk"]["blocked"] >= 2
    st = C.apply(build, lanes=lanes, log=lambda *a: None)
    assert st["passage"] == 1
    tris, _ = building_tris(build)
    band = box(-60, 5.5 - C.WALK_HALF_M + 0.05, 60, 5.5 + C.WALK_HALF_M - 0.05)
    assert blocked_in(tris, band, 0.2, 2.6) == 0
    assert (tris[:, :, 2] > 10).any()                       # the facade above is still there
    assert np.isclose(tris[:, :, 2], math.ceil((0.15 + 2.5) * 10 - 1e-9) / 10).any()   # soffit at 2.7 m
    assert C.check(build, lanes)["walk"]["blocked"] == 0


def test_elevated_road_keeps_structure_below(tmp_path):
    """A road deck at 8 m over a building solid (a viaduct in the data): the slab around the deck is cut,
    the structure under it stays, and the cut gets a floor and a soffit."""
    build = make_export(tmp_path, [("V", -10, -15, 10, 15, -0.3, 20.0)], road_z=8.0, bottom=True)
    lanes = lanes_along_x(z=8.0)
    st = C.apply(build, lanes=lanes, log=lambda *a: None)
    assert st["passage"] == 1 and st["floor_m2"] > 100
    tris, _ = building_tris(build)
    road = box(-60, -3.4, 60, 3.4)
    assert blocked_in(tris, road, 7.8, 12.5) == 0
    assert blocked_in(tris, road, 0.5, 7.0) > 0             # walls below the deck stay
    assert_closed(build, 20 * 30 * 20.3 - 20 * (7.0 + 2 * C.ROAD_BUFFER_M) * (12.6 - 7.7))


def test_photo_uvs_survive_the_cut(tmp_path):
    """Cut walls keep their texture mapping: every vertex's uv still follows its position."""
    build = make_export(tmp_path, [("P", -10, -15, 10, 15, -0.3, 20.0)])
    C.apply(build, lanes=lanes_along_x(), log=lambda *a: None)
    g = read_glb(build / "ue/meshes/T_L0_building_0_0_b0.glb")
    from twinmodel.export.ue import gltf_to_model
    p = gltf_to_model(g["positions"])
    assert np.allclose(g["uvs"], p[:, :2] * 0.1 + p[:, 2:3] * 0.05, atol=1e-4)


def test_maxheight_street_gets_a_lower_passage(tmp_path):
    build = make_export(tmp_path, [("M", -10, -15, 10, 15, -0.3, 20.0)])
    streets = [(box(-60, -3.5, 60, 3.5), 3.4, True)]
    lanes = lanes_along_x()
    C.apply(build, lanes=lanes, streets=streets, log=lambda *a: None)
    tris, _ = building_tris(build)
    assert np.isclose(tris[:, :, 2], 3.6).any() and not np.isclose(tris[:, :, 2], 4.6).any()


def test_osm_street_zones_from_model_and_overpass(tmp_path):
    build = tmp_path / "b"
    twin = build / "X.twin"
    twin.mkdir(parents=True)
    (twin / "roads.geojson").write_text(json.dumps({"features": [
        {"properties": {"id": "r1", "osm_way_ids": [11]}}, {"properties": {"id": "r2", "osm_way_ids": [12]}}]}))
    (twin / "surfaces.geojson").write_text(json.dumps({"features": [
        {"properties": {"kind": "drivable", "road_ids": ["r1"]}, "geometry": box(0, 0, 10, 5).__geo_interface__},
        {"properties": {"kind": "drivable", "road_ids": ["r2"]}, "geometry": box(20, 0, 30, 5).__geo_interface__}]}))
    (build / "data").mkdir()
    (build / "data/overpass_x.json").write_text(json.dumps({"elements": [
        {"type": "way", "id": 11, "tags": {"highway": "service", "maxheight": "3.2"}},
        {"type": "way", "id": 12, "tags": {"highway": "footway", "covered": "yes"}}]}))
    z = C.osm_street_zones(build)
    assert len(z) == 2
    assert sorted(((mh or 0.0), cov) for _, mh, cov in z) == [(0.0, True), (3.2, False)]


def test_check_counts_walls_and_overhangs(tmp_path):
    """The check: a wall across the lane blocks, a roof 10 m up does not, an overhang at 3 m does."""
    build = make_export(tmp_path, [("A", -2, -6, 2, 6, -0.3, 10.0)])
    lanes = lanes_along_x()
    r = C.check(build, lanes)
    assert 2 <= r["road"]["crossings"] <= 4 and r["road"]["blocked"] == r["road"]["crossings"]
    assert 2 <= r["road"]["inside"] <= 5
    # an overhang: a horizontal plate at 3 m over the road only
    v, f = _slab(20, -2, 30, 2, z=3.0)
    tris = np.concatenate([C.building_triangles(build / "ue", json.loads((build / "ue/manifest.json").read_text())),
                           v[f]])
    r2 = C.check(build, lanes, tris=tris)
    assert r2["road"]["blocked"] >= r["road"]["blocked"] + 9
    assert all(math.isfinite(h["lat"]) for h in r2["road"]["hits"])


def test_solid_footprints_leave_the_passage_open(tmp_path):
    """<Name>.buildings.json: the building's footprint minus the passage, in CARLA x, y (y flipped)."""
    build = make_export(tmp_path, [("B1", -10, -15, 10, 15, -0.3, 20.0)])
    C.apply(build, lanes=lanes_along_x(), log=lambda *a: None)
    recs = json.loads((build / "T.buildings.json").read_text())
    assert len(recs) == 2 and all(r["id"] == "B1" for r in recs)
    polys = [Polygon(r["p"]) for r in recs]
    assert abs(sum(p.area for p in polys) - (600 - 20 * (7.0 + 2 * C.ROAD_BUFFER_M))) < 1.0
    assert all(not p.contains(shapely.Point(0, 0)) for p in polys)
    assert any(p.contains(shapely.Point(0, -10)) for p in polys)   # model y +10 -> CARLA y -10


def test_footprints_rewritten_without_touching_meshes(tmp_path):
    build = make_export(tmp_path, [("B1", -10, -15, 10, 15, -0.3, 20.0)])
    C.apply(build, lanes=lanes_along_x(), log=lambda *a: None)
    first = json.loads((build / "T.buildings.json").read_text())
    stamp = {p.name: p.stat().st_mtime_ns for p in (build / "ue/meshes").glob("*.glb")}
    (build / "T.buildings.json").unlink()
    assert C.write_footprints(build, lanes=lanes_along_x()) == len(first)
    assert json.loads((build / "T.buildings.json").read_text()) == first
    assert stamp == {p.name: p.stat().st_mtime_ns for p in (build / "ue/meshes").glob("*.glb")}
