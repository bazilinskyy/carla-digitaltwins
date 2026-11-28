"""Tests for real 3D buildings (twinmodel.export.realbuildings, twinmodel.ingest.buildings / citygml /
buildings_nyc, the NYS ortho provider); no network."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from twinmodel.export import facades as fc
from twinmodel.export import realbuildings as rb
from twinmodel.export.ue import gltf_to_model, read_glb
from twinmodel.ingest import buildings, buildings_nyc, citygml, imagery

MIDTOWN = (40.74861, -73.9923, 40.76119, -73.9757)
EHV = (51.4360, 5.4700, 51.4500, 5.4930)
GEO = "+proj=tmerc +lat_0=40.7549 +lon_0=-73.984 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def _box(x0, y0, x1, y1, z0, z1):
    """Surfaces of a box building (counter-clockwise seen from outside), as a provider gives them."""
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = []
    for (ax, ay), (bx, by) in zip(c, c[1:] + c[:1]):
        walls.append(([np.array([[ax, ay, z0], [bx, by, z0], [bx, by, z1], [ax, ay, z1]], float)], "WallSurface"))
    roof = ([np.array([[x, y, z1] for x, y in c], float)], "RoofSurface")
    ground = ([np.array([[x, y, z0] for x, y in reversed(c)], float)], "GroundSurface")
    return walls + [roof, ground]


# --------------------------------------------------------------------------- providers and styles

def test_choose_provider_by_bbox():
    assert buildings.choose(EHV) == "3dbag"
    assert buildings.choose(MIDTOWN) == "nyc3d"
    assert buildings.choose((41.38, 2.15, 41.40, 2.18)) == "catastro"  # Barcelona: the Spanish Catastro
    assert buildings.choose((52.50, 13.38, 52.52, 13.41)) is None       # Berlin: no national model (Overture)
    assert buildings.choose((52.50, 13.38, 52.52, 13.41), include_global=True) == "overture"
    with pytest.raises(KeyError):
        buildings.module("nope")


def test_nys_orthos_decline_outside_new_york():
    assert imagery._fetch_nys(None, EHV, 0.25, None) is None
    assert "nys_orthos" in imagery.PROVIDERS


def test_pluto_uses_and_building_classes():
    lot = {"bldgclass": "O4", "officearea": "100000", "retailarea": "5000", "resarea": "0"}
    u = buildings_nyc.uses_of(lot)
    assert set(u) == {"office", "retail"} and u["office"] == pytest.approx(100000 * buildings_nyc.SQFT)
    # a hotel's floor area is "other" in PLUTO: the building class names it
    assert set(buildings_nyc.uses_of({"bldgclass": "H2", "otherarea": "80000"})) == {"hotel"}
    assert buildings_nyc.uses_of({"bldgclass": "D6"}) == {"residential": 1.0}
    assert buildings_nyc.uses_of({"bldgclass": "V0"}) == {}


@pytest.mark.parametrize("kw,style,shop", [
    (dict(year=1931, uses={"office": 9000, "retail": 300}, height_m=120.0, region="us"), "stone_prewar", True),
    (dict(year=1931, uses={"office": 9000}, height_m=120.0), "office_concrete", False),   # nl: unchanged
    (dict(year=2010, uses={"office": 9000, "retail": 300}, height_m=150.0, region="us"), "office_glass", True),
    (dict(year=1910, uses={"residential": 900}, height_m=15.0, region="us"), "brick_old", False),
    (dict(year=1960, uses={"hotel": 9000}, height_m=60.0, region="us"), "office_concrete", False),
    (dict(year=1950, uses={"storage": 900}, height_m=8.0, region="us"), "industrial", False),
    (dict(year=None, uses={"utility": 1}, height_m=240.0, region="us"), "office_glass", False),   # no metal tower
])
def test_classify_neutral_uses_and_us_region(kw, style, shop):
    base = dict(id="NYC.BIN.1", area_m2=600.0)
    base.update(kw)
    assert fc.classify(fc.BuildingInfo(**base)) == (style, shop)


def test_roof_heights_weighted_percentile():
    surf = _box(0, 0, 10, 10, 5.0, 25.0) + _box(10, 0, 12, 2, 5.0, 60.0)   # a slab and a thin spire
    eaves, top = rb.roof_heights(surf, 5.0)
    assert eaves == pytest.approx(20.0) and top == pytest.approx(60.0)


def test_skirt_reaches_the_lower_terrain(tmp_path):
    z = np.full((20, 20), 10.0)
    z[:, :10] = 8.0                       # the street falls 2 m to the west
    np.savez(tmp_path / "dem.npz", z=z, x0=-20.0, y0=-20.0, dx=2.0, dy=2.0, source="test")
    b = rb.RealBuilding(id="b", surfaces=_box(-5, -5, 5, 5, 10.0, 30.0), poly=Polygon(
        [(-5, -5), (5, -5), (5, 5), (-5, 5)]), ground=10.0, height=20.0)
    flat = rb.RealBuilding(id="f", surfaces=_box(12, 12, 15, 15, 10.0, 20.0), poly=Polygon(
        [(12, 12), (15, 12), (15, 15), (12, 15)]), ground=10.0, height=10.0)
    assert rb.skirt([b, flat], tmp_path / "dem.npz") == 1
    walls = [r for rings, k in b.surfaces if k == "WallSurface" for r in rings]
    assert min(r[:, 2].min() for r in walls) == pytest.approx(7.7)
    roof = next(rings[0] for rings, k in b.surfaces if k == "RoofSurface")
    assert np.allclose(roof[:, 2], 30.0)
    assert min(r[:, 2].min() for rings, k in flat.surfaces if k == "WallSurface" for r in rings) == 10.0


# --------------------------------------------------------------------------- CityGML

GML = """<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel xmlns:core="http://www.opengis.net/citygml/1.0" xmlns:bldg="http://www.opengis.net/citygml/building/1.0"
 xmlns:gml="http://www.opengis.net/gml" xmlns:gen="http://www.opengis.net/citygml/generics/1.0">
<gml:boundedBy><gml:Envelope srsName="EPSG:2263" srsDimension="3">
<gml:lowerCorner>987000 215000 0</gml:lowerCorner><gml:upperCorner>989000 216000 500</gml:upperCorner>
</gml:Envelope></gml:boundedBy>
{members}
</core:CityModel>"""
MEMBER = """<core:cityObjectMember><bldg:Building gml:id="{gid}">
<gen:stringAttribute name="BIN"><gen:value>{bin}</gen:value></gen:stringAttribute>
<gen:stringAttribute name="DOITT_ID"><gen:value>{doitt}</gen:value></gen:stringAttribute>
{surfaces}
</bldg:Building></core:cityObjectMember>"""
SURF = """<bldg:boundedBy><bldg:{kind} gml:id="s{n}">
<gen:stringAttribute name="BIN"><gen:value>0</gen:value></gen:stringAttribute>
<bldg:lod2MultiSurface><gml:MultiSurface srsName="EPSG:2263"><gml:surfaceMember><gml:Polygon><gml:exterior>
<gml:LinearRing><gml:posList>{pos}</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>
</gml:MultiSurface></bldg:lod2MultiSurface></bldg:{kind}></bldg:boundedBy>"""


def _gml_building(gid, bin_, doitt, x0, y0, w, z0, z1):
    surf, n = [], 0
    for rings, kind in _box(x0, y0, x0 + w, y0 + w, z0, z1):
        r = rings[0]
        pos = " ".join("%.3f" % v for p in np.vstack([r, r[:1]]) for v in p)
        surf.append(SURF.format(kind=kind, n=n, pos=pos))
        n += 1
    return MEMBER.format(gid=gid, bin=bin_, doitt=doitt, surfaces="".join(surf))


def test_citygml_envelope_and_buildings(tmp_path):
    inside = _gml_building("g1", "1026885", "110442", 988000, 215500, 60, 70.0, 230.0)
    outside = _gml_building("g2", "1000000", "7", 900000, 100000, 60, 70.0, 130.0)
    path = tmp_path / "DA99_3D_Buildings_Merged.gml"
    path.write_text(GML.format(members=inside + outside))
    srs, env = citygml.envelope(path.read_bytes()[:4000])
    assert srs == "EPSG:2263" and env == (987000.0, 215000.0, 989000.0, 216000.0)
    assert citygml.crs_of(srs) == ("EPSG:2263", False, citygml.US_FT)
    assert citygml.crs_of("http://www.opengis.net/def/crs/EPSG/0/6697")[:2] == ("EPSG:6668", True)

    def box(p):
        return (p[:, 0] > 987000) & (p[:, 0] < 989000) & (p[:, 1] > 215000) & (p[:, 1] < 216000)
    got = list(citygml.buildings(path, box))
    assert len(got) == 1 and got[0]["id"] == "g1" and got[0]["attrs"]["BIN"] == "1026885"
    kinds = sorted(k for k, _ in got[0]["surfaces"])
    assert kinds.count("WallSurface") == 4 and "RoofSurface" in kinds and "GroundSurface" in kinds
    ring = got[0]["surfaces"][0][1][0]
    assert ring.shape == (4, 3)                         # closing point dropped


def test_citygml_members_by_envelope(tmp_path):
    import zipfile
    z = tmp_path / "city.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("x/DA1_3D_Buildings_Merged.gml", GML.format(members=""))
        zf.writestr("x/DA2_3D_Buildings_Merged.gml",
                    GML.format(members="").replace("987000 215000", "1 1").replace("989000 216000", "2 2"))
    got = citygml.members(str(z), lambda n, srs, env: env[0] < 988000 < env[2], cache_root=tmp_path / "cache")
    assert [p.name for p in got] == ["x_DA1_3D_Buildings_Merged.gml"]
    assert json.loads((tmp_path / "cache" / "city_zip" / "_index.json").read_text())["envelopes"]


# --------------------------------------------------------------------------- the NYC provider and apply

def _manifest():
    return {"name": "T", "tile_m": 250.0, "geo_reference": GEO, "bbox_wgs84": list(MIDTOWN),
            "stats": {"buildings": 1}, "assets": [], "buildings": [
                {"id": "w1", "category": "commercial", "rings_ue": [[[0, 0], [100, 0], [100, -100]]]}]}


def _write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def test_nyc_load_from_cache_and_apply(tmp_path):
    from pyproj import Transformer
    to_file = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    x, y = to_file.transform(-73.984, 40.7549)           # the model origin
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
    assert [b.id for b in blds] == ["NYC.BIN.1026885", "NYC.DOITT.2"]       # placeholder BIN -> DOITT id
    a, b = blds
    assert a.year == 1931 and a.levels == 34 and set(a.uses) == {"office", "retail"} and a.region == "us"
    assert a.ground == pytest.approx(50.0 * citygml.US_FT, abs=1e-3)
    assert a.height == pytest.approx(400.0 * citygml.US_FT, abs=0.01)
    assert a.poly.area == pytest.approx(900.0, rel=0.01)                  # 30 m x 30 m footprint
    assert abs(a.poly.centroid.x - 15.0) < 0.5 and abs(a.poly.centroid.y - 15.0) < 0.5
    assert b.uses == {} and b.year is None

    (build / "ue" / "meshes").mkdir(parents=True)
    img = np.full((400, 400, 3), 120, np.uint8)
    ortho = (img, (1.0, 0.0, -200.0, 0.0, -1.0, 200.0))   # 1 m/px, x -200..200, y 200..-200
    stats = rb.apply(build, man, blds, src, ortho=ortho, ortho_label="test photo", texture_px=64, roof_px=64)
    assert stats["buildings"] == 2 and stats["real_buildings"] == "nyc3d"
    assert stats["facade_styles"] == {"stone_prewar": 1, "brick_postwar": 1}   # no year: post-war homes
    assert stats["roof_plain_tall"] == 1                  # the 122 m tower: no photo roof (not a true ortho)
    mats = {x["material"] for x in man["assets"]}
    assert "roof" in mats and "roof_0_0" in mats
    assert any(m.startswith("facade_shopfront") for m in mats)
    assert len({x["asset"] for x in man["assets"]}) == len(man["assets"])   # no name clashes
    assert man["buildings_osm"][0]["id"] == "w1" and len(man["buildings"]) == 2
    assert all(x["real"] == "nyc3d" for x in man["buildings"])
    tower = next(x for x in man["assets"] if x["material"].startswith("facade_stone_prewar"))
    pos = gltf_to_model(read_glb(build / "ue" / tower["file"])["positions"])
    assert pos[:, 2].max() == pytest.approx(450.0 * citygml.US_FT, abs=0.01)
    assert (build / "ue" / man["textures"]["roof_0_0"]["base_color"]).exists()
    # a rerun (e.g. after a code change) replaces the real buildings, the OSM ones stay in buildings_osm
    stats2 = rb.apply(build, json.loads((build / "ue" / "manifest.json").read_text()), blds, src, looks=False)
    assert stats2["buildings"] == 2
    man2 = json.loads((build / "ue" / "manifest.json").read_text())
    assert {x["material"] for x in man2["assets"]} == {"building"}
    assert (build / "ue" / "meshes" / "osm_buildings").is_dir()


# --------------------------------------------------------------------------- PLATEAU

def test_plateau_meshes_and_coverage():
    from twinmodel.ingest import buildings_plateau as pl
    assert pl.mesh3(35.6595, 139.7005) == "53393596"            # Shibuya crossing
    shibuya = (35.655, 139.695, 35.665, 139.705)
    assert {"53393596", "53393586"} <= pl.meshes(shibuya)
    assert buildings.choose(shibuya) == "plateau"
    assert buildings.choose((35.68, 139.76, 35.69, 139.77)) is None   # Chiyoda: no dataset listed yet
    assert pl.USAGE["414"]["retail"] > 0
    st, shop = fc.classify(fc.BuildingInfo(id="j", uses={"residential": 1}, height_m=20, area_m2=300, region="jp"))
    assert st in ("jp_mansion", "jp_tile", "jp_concrete") and not shop   # Japanese tiles and balconies, not brick


def test_japanese_styles_by_use_and_height():
    def styles(**kw):
        return {fc.classify(fc.BuildingInfo(id="J%d" % k, area_m2=400, region="jp", **kw))[0] for k in range(200)}
    flats = styles(uses={"residential": 1}, height_m=24)
    assert flats <= {"jp_mansion", "jp_tile", "jp_concrete"} and "jp_mansion" in flats
    towers = styles(uses={"office": 1}, height_m=60)
    assert {"office_glass", "jp_office"} <= towers and "render_modern" not in towers
    assert styles(uses={"residential": 1}, height_m=7) <= {"jp_tile", "render_modern", "jp_concrete"}
    # a shop floor in Japan is the signboard front; elsewhere the plain shopfront
    assert fc.shop_style("jp") == "shopfront_jp" and fc.shop_style("nl") == "shopfront"
    assert fc.classify(fc.BuildingInfo(id="s", uses={"retail": 1}, height_m=4.0, area_m2=80, region="jp"))[0] == \
        "shopfront_jp"
    assert all(fc.BLANK_OF[s] == "blank_concrete" for s in ("jp_tile", "jp_mansion", "jp_office", "jp_concrete"))
    assert fc.blank_of("office_glass", "jp") == "blank_concrete" and fc.blank_of("office_glass", "nl") == "blank_brick"
    assert fc.classify(fc.BuildingInfo(id="h", height_m=3.0, area_m2=20, region="jp"))[0] == "blank_concrete"


def test_photo_walls_keep_the_photo_only_above_the_street_storeys(tmp_path):
    # PLATEAU's wall photos are oblique aerial images (about 19 cm per texel in the Shibuya export): the street
    # storeys get the building's facade style, the photo stays above photo_walls_above_m, its UVs cut with it
    from PIL import Image
    img = tmp_path / "wall.jpg"
    Image.fromarray(np.full((64, 64, 3), (150, 120, 90), np.uint8)).save(img)
    surf = _box(0, 0, 20, 20, 0.0, 30.0)
    tex = {}
    for si, (rings, kind) in enumerate(surf):
        if kind == "WallSurface":
            r = rings[0]
            # u along the wall's 20 m, v up its 30 m (CityGML: origin bottom left)
            along = np.linalg.norm(r[:, :2] - r[0, :2], axis=1) / 20.0
            tex[si] = (str(img), [np.column_stack([along, r[:, 2] / 30.0])])
    tall = rb.RealBuilding(id="JP.t", surfaces=surf, textures=tex, poly=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
                           ground=0.0, height=30.0, uses={"office": 1}, region="jp")
    surf2 = _box(40, 0, 50, 10, 0.0, 9.0)
    tex2 = {si: (str(img), [np.column_stack([np.linspace(0, 1, 4), rings[0][:, 2] / 9.0])])
            for si, (rings, kind) in enumerate(surf2) if kind == "WallSurface"}
    low = rb.RealBuilding(id="JP.l", surfaces=surf2, textures=tex2, poly=Polygon([(40, 0), (50, 0), (50, 10), (40, 10)]),
                          ground=0.0, height=9.0, uses={"residential": 1}, region="jp")
    build = tmp_path / "b"
    (build / "ue" / "meshes").mkdir(parents=True)
    man = _manifest()
    src = rb.Source(name="plateau", label="PLATEAU test", asset_source="test")
    stats = rb.apply(build, man, [tall, low], src, texture_px=64)
    assert stats["photo_walls_street_styled"] == 4 and stats["photo_walls_all_styled"] == 4
    photo = [a for a in man["assets"] if a["material"].startswith("photo_")]
    styled = [a for a in man["assets"] if a["material"].startswith("facade_")]
    assert photo and styled
    for a in photo:
        g = read_glb(build / "ue" / a["file"])
        pos = gltf_to_model(g["positions"])
        walls = np.abs(g["normals"][:, 1]) < 0.5        # glTF y up: wall vertices
        assert pos[walls][:, 2].min() >= 6.0 - 1e-6     # the photo only above the storey line nearest 8 m
        uv = g["uvs"][walls]
        assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6   # cut UVs stay inside the atlas image
    low_z = []
    for a in styled:
        pos = gltf_to_model(read_glb(build / "ue" / a["file"])["positions"])
        low_z.append(pos[:, 2].max())
    assert min(low_z) <= 10.0 + 1e-6                    # the street storeys are styled
    # photos everywhere, as before, with photo_walls_above_m=0
    man0 = _manifest()
    stats0 = rb.apply(build, man0, [tall, low], src, texture_px=64, photo_walls_above_m=0)
    assert "photo_walls_street_styled" not in stats0 and stats0["photo_textured_surfaces"] == 8


def test_an_export_without_its_builds_terrain_keeps_its_buildings_in_place(tmp_path):
    # restyling an export copied away from its build (ZurichHB, 2026-11-25): an elevation grid from the export's
    # own surfaces over the ground plane, and the buildings it already had keep their base heights by id
    import json
    from shapely.geometry import box
    from twinmodel.export import realbuildings as rb
    from .test_surface_gaps import _export
    ue = _export(tmp_path, [("drivable", -20, 20, -4, 4, 412.0), ("sidewalk", -20, 20, 4, 8, 412.15)],
                 ground_z=411.65)
    man = json.loads((ue / "manifest.json").read_text())
    man["bbox_wgs84"] = [47.37, 8.53, 47.38, 8.54]
    p = rb.dem_from_export(tmp_path, man)
    assert p is not None and rb.find_dem(tmp_path, man["bbox_wgs84"]) == p
    from twinmodel.ingest.buildings_extrude import Ground
    g = Ground(p)
    lo, med = g.sample(box(-2, -2, 2, 2))
    assert abs(med - 412.0) < 0.05                                       # on the road
    lo, med = g.sample(box(30, 30, 34, 34))
    assert abs(med - 412.0) < 0.05                                       # off the surfaces: the plane + its drop
    man["buildings"] = [{"id": "OV.a", "real": "overture", "base_z_cm": 41030.0}]
    ring = [np.array([[0, 0, 400.0], [5, 0, 400.0], [5, 5, 400.0]])]
    moved = rb.RealBuilding(id="OV.a", surfaces=[(ring, "wall")], ground=400.0, height=9.0, roof_z=409.0)
    new = rb.RealBuilding(id="OV.b", surfaces=[([r.copy() for r in ring], "wall")], ground=400.0, height=9.0)
    assert rb.keep_bases([moved, new], man) == 1
    assert moved.ground == 410.3 and abs(moved.surfaces[0][0][0][0, 2] - 410.3) < 1e-9 and moved.roof_z == 419.3
    assert new.ground == 400.0                                           # a new building: the grid places it
