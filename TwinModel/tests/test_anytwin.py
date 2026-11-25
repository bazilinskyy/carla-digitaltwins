"""A twin of any place (twinmodel.regions, the national/global ortho, elevation and building providers,
buildings_extrude, twinmodel.twin); no network: every fetch is monkeypatched."""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from twinmodel import regions, twin
from twinmodel.export import facades as fc
from twinmodel.export import realbuildings as rb
from twinmodel.ingest import (buildings, buildings_bdtopo, buildings_catastro, buildings_extrude as ex,
                              buildings_osm, buildings_overture, elevation, imagery)

SALAMANCA = (40.9600, -5.6700, 40.9690, -5.6580)
ANN_ARBOR = (42.2760, -83.7550, 42.2850, -83.7430)
BERLIN = (52.50, 13.38, 52.52, 13.41)
LIEGE = (50.63, 5.56, 50.65, 5.59)            # Wallonia: outside the Flemish orthophoto
GHENT = (51.05, 3.71, 51.06, 3.73)
GEO_SAL = "+proj=tmerc +lat_0=40.9645 +lon_0=-5.664 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


# --------------------------------------------------------------------------- regions: plans and licences

def test_plans_per_country():
    es = regions.plan_for("ES", SALAMANCA)
    assert es.ortho == ("ign_es", "osm_tiles")                 # ICGC only in Catalonia
    assert es.dem[0] == "ign_wcs" and es.dem[-2:] == ("aws_terrain", "copernicus_aws")
    assert es.buildings[0] == "catastro" and es.facade_region == "south" and not es.lefthand
    bcn = regions.plan_for("ES", (41.38, 2.15, 41.40, 2.18))
    assert bcn.ortho[0] == "icgc" and bcn.dem[0] == "icgc_mdt2m"
    us = regions.plan_for("us", ANN_ARBOR)
    assert us.ortho == ("naip", "osm_tiles") and us.dem[0] == "usgs_3dep"   # NYS orthos only in New York State
    assert us.buildings[:2] == ("nyc3d", "overture") and us.facade_region == "us"
    nl = regions.plan_for("NL", (51.436, 5.47, 51.45, 5.493))
    assert nl.ortho == ("pdok_luchtfoto", "osm_tiles") and nl.dem == ("ahn_wcs", "copernicus_aws")
    assert regions.plan_for("BE", GHENT).ortho[0] == "flanders"
    liege = regions.plan_for("BE", LIEGE)
    assert liege.ortho == ("osm_tiles",) and any("no open aerial photo" in n for n in liege.notes)
    de = regions.plan_for("DE", BERLIN)
    assert de.ortho == ("osm_tiles",) and de.dem == ("aws_terrain", "copernicus_aws")
    assert de.buildings == ("overture", "osm") and de.facade_region == "eu"
    assert regions.plan_for("IT").facade_region == "south"
    jp = regions.plan_for("JP")
    assert jp.lefthand and jp.facade_region == "jp" and any("left-hand" in n for n in jp.notes)
    unknown = regions.plan_for(None)
    assert unknown.iso2 is None and unknown.buildings == ("overture", "osm")


def test_every_provider_has_a_licence_and_exists():
    names = set()
    for cc in list(regions.COUNTRIES) + [None, "DE"]:
        p = regions.plan_for(cc)
        names |= set(p.ortho) | set(p.dem) | set(p.buildings)
        for o in p.ortho:
            assert o in imagery.PROVIDERS
        for d in p.dem:
            assert d in elevation.PROVIDERS
        for b in p.buildings:
            assert b in buildings.PROVIDERS
    names |= set(buildings.PROVIDERS) | set(imagery.PROVIDERS) | set(elevation.PROVIDERS)
    for n in names:
        what, lic, url = regions.SOURCES[n]
        assert what and lic, n
    # nothing that may not be used
    assert not any(bad in " ".join(map(str, regions.SOURCES.values())).lower()
                   for bad in ("esri world imagery", "google satellite", "bing"))


def test_coverage_boxes():
    assert regions.covers("ign_fr", (48.85, 2.34, 48.86, 2.36))
    assert not regions.covers("ign_fr", SALAMANCA)
    assert regions.covers("ign_es", (28.40, -16.30, 28.47, -16.24))   # Tenerife
    assert regions.covers("osm_tiles", BERLIN)                       # no entry: everywhere


def test_attribution_text_lists_sources_with_licences():
    report = {"build": {"imagery": {"source": "ign_es"}}, "elevation": {"source": "ign_wcs"}}
    man = {"stats": {"real_buildings": "catastro", "building_source": "Catastro (10 parts)"}}
    text = regions.attribution_text("SalamancaCentre", report, man, extra=("nominatim",))
    assert "SalamancaCentre" in text and "OpenStreetMap" in text and "ODbL" in text
    assert "PNOA" in text and "MDT05" in text and "Catastro" in text and text.count("CC BY 4.0") >= 3
    assert "Nominatim" in text
    assert regions.used_sources({"build": {"imagery": {"source": "osm_tiles"}}}, {}) == ["osm"]


def test_cli_auto_profile_takes_the_region_plan(monkeypatch, tmp_path):
    from twinmodel import cli
    monkeypatch.setattr(cli, "country_for_bbox", lambda bbox, cache: "ES")
    monkeypatch.setattr(cli, "building_coverage", lambda osm, frame, bbox: 0.4)
    p, meta = cli.select_profile("auto", None, None, SALAMANCA, tmp_path)
    assert p.name == "eu_dense" and p.sources.ortho == ("ign_es", "osm_tiles")
    assert meta["region_plan"]["buildings"][0] == "catastro"
    p2, meta2 = cli.select_profile("nl_urban", None, None, SALAMANCA, tmp_path)   # explicit: its own sources
    assert p2.sources.ortho[0] == "pdok_luchtfoto" and "region_plan" not in meta2


# --------------------------------------------------------------------------- imagery

class _Resp:
    def __init__(self, content=b"", ctype="image/jpeg", status=200, js=None):
        self.content, self.status_code, self.headers, self._js = content, status, {"content-type": ctype}, js
        self.text = content[:200].decode("latin-1") if isinstance(content, bytes) else str(content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._js


def _frame():
    from twinmodel.frame import LocalFrame
    return LocalFrame.from_bbox(*SALAMANCA)


def test_fetch_ortho_skips_providers_outside_their_area_and_blank_answers(monkeypatch, tmp_path):
    called = []

    def fake(name, arr):
        def fn(frame, bbox, res, grid, **_):
            called.append(name)
            return imagery.OrthoImage(arr, x0=0, y0=0, dx=res, dy=res)
        return fn
    white = np.full((10, 10, 3), 255, np.uint8)
    photo = np.random.default_rng(0).integers(30, 200, (10, 10, 3)).astype(np.uint8)
    monkeypatch.setitem(imagery.PROVIDERS, "pdok_luchtfoto", fake("pdok_luchtfoto", photo))
    monkeypatch.setitem(imagery.PROVIDERS, "ign_fr", fake("ign_fr", white))
    monkeypatch.setitem(imagery.PROVIDERS, "ign_es", fake("ign_es", photo))
    img = imagery.fetch_ortho(_frame(), (48.85, 2.34, 48.851, 2.341), cache_dir=tmp_path,
                              sources=("pdok_luchtfoto", "ign_fr", "ign_es"), use_cache=False)
    assert called == ["ign_fr"] and img is None            # PDOK not asked in Paris; IGN's white answer refused
    called.clear()
    img = imagery.fetch_ortho(_frame(), (40.9600, -5.67, 40.9601, -5.6699), cache_dir=tmp_path,
                              sources=("ign_fr", "ign_es"), use_cache=False)
    assert called == ["ign_es"] and img.source == "ign_es"
    assert imagery.mostly_blank(white) and not imagery.mostly_blank(photo)


def test_wms_jpeg_provider_requests(monkeypatch):
    from PIL import Image
    from twinmodel.frame import LocalFrame
    seen = []

    def get(self, url, params=None, timeout=None, **kw):
        seen.append((url, params))
        buf = io.BytesIO()
        Image.fromarray(np.full((int(params["HEIGHT"]), int(params["WIDTH"]), 3), 90, np.uint8)).save(buf, "JPEG")
        return _Resp(buf.getvalue(), "image/jpeg")
    import requests
    monkeypatch.setattr(requests.Session, "get", get)
    bbox = (51.0540, 3.7200, 51.0545, 3.7207)       # Ghent
    frame = LocalFrame.from_bbox(*bbox)
    grid = imagery.model_grid(frame, bbox, 0.5, 0.0)
    img = imagery._fetch_flanders(frame, bbox, 0.5, grid)
    assert seen and seen[0][0] == imagery.FLANDERS_WMS
    assert seen[0][1]["LAYERS"] == "OMWRGB25VL" and seen[0][1]["CRS"] == "EPSG:31370"
    assert img is not None and img.source == "flanders" and int(img.array.mean()) in range(80, 100)


def test_xyz_tiles_and_gsi_decoding():
    x0, y0, x1, y1 = imagery.xyz_tile_range((35.65, 139.69, 35.66, 139.70), 18)
    assert x0 <= x1 and y0 <= y1 and (x1 - x0) < 20
    from PIL import Image
    # GSI: 12.34 m -> x = 1234 = (0, 4, 210); -1.00 m -> 2^24 - 100; no data 2^23
    a = np.zeros((1, 3, 3), np.uint8)
    a[0, 0] = (0, 4, 210)
    neg = 2 ** 24 - 100
    a[0, 1] = (neg >> 16, (neg >> 8) & 255, neg & 255)
    a[0, 2] = (128, 0, 0)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    h = elevation.decode_gsi_png(buf.getvalue())[0, 0]
    assert h[0] == pytest.approx(12.34) and h[1] == pytest.approx(-1.0) and np.isnan(h[2])
    t = np.zeros((1, 1, 3), np.uint8)
    t[0, 0] = (128, 100, 128)            # 128*256 + 100 + 0.5 - 32768 = 100.5
    buf = io.BytesIO()
    Image.fromarray(t).save(buf, "PNG")
    assert elevation.decode_terrarium(buf.getvalue())[0, 0, 0] == pytest.approx(100.5)


# --------------------------------------------------------------------------- elevation

def test_fetch_dem_skips_national_dtms_outside_their_country(monkeypatch, tmp_path):
    called = []
    from twinmodel.model import Elevation

    def fake(name):
        def fn(frame, bbox, grid, spacing):
            called.append(name)
            return Elevation(np.full((3, 3), 800.0), 0, 0, 2, 2, source=name)
        return fn
    for n in ("rge_alti", "swissalti3d", "aws_terrain"):
        monkeypatch.setitem(elevation.PROVIDERS, n, fake(n))
    el = elevation.fetch_dem(_frame(), SALAMANCA, cache_dir=tmp_path, use_cache=False,
                             sources=("rge_alti", "swissalti3d", "aws_terrain"))
    assert called == ["aws_terrain"] and el.source == "aws_terrain"


def test_rge_alti_bil_tiles(monkeypatch):
    def get(self, url, params=None, timeout=None, **kw):
        w, h = int(params["WIDTH"]), int(params["HEIGHT"])
        z = np.full((h, w), 35.5, "<f4")
        z[0, 0] = -99999.0
        return _Resp(z.tobytes(), "image/x-bil;bits=32")
    import requests
    monkeypatch.setattr(requests.Session, "get", get)
    from twinmodel.frame import LocalFrame
    paris = (48.8500, 2.3400, 48.8520, 2.3430)
    frame = LocalFrame.from_bbox(*paris)
    grid = imagery.model_grid(frame, paris, 2.0, 10.0)
    el = elevation._rge_alti(frame, paris, grid)
    assert el is not None and np.allclose(el.z, 35.5)


def test_swissalti3d_newest_tile_per_square(monkeypatch):
    feats = [{"id": "swissalti3d_2019_2683-1247", "assets": {
                "swissalti3d_2019_2683-1247_0.5_2056_5728.tif": {"href": "https://x/2019_05.tif"},
                "swissalti3d_2019_2683-1247_2_2056_5728.tif": {"href": "https://x/2019_2.tif"}}},
             {"id": "swissalti3d_2023_2683-1247", "assets": {
                "swissalti3d_2023_2683-1247_0.5_2056_5728.tif": {"href": "https://x/2023_05.tif"}}},
             {"id": "swissalti3d_2021_2684-1247", "assets": {
                "swissalti3d_2021_2684-1247_0.5_2056_5728.tif": {"href": "https://x/2684.tif"}}}]

    class S:
        def get(self, url, params=None, **kw):
            assert params["bbox"].startswith("8.54,47.37")
            return _Resp(b"", "application/json", js={"features": feats})
    hrefs = elevation.swissalti3d_assets((47.37, 8.54, 47.38, 8.55), session=S())
    assert hrefs == ["https://x/2023_05.tif", "https://x/2684.tif"]


# --------------------------------------------------------------------------- extruded buildings

def _roofs_walls(surfaces):
    return ([r for r, k in surfaces if k == "RoofSurface"], [r for r, k in surfaces if k == "WallSurface"])


@pytest.mark.parametrize("shape,n_roofs,ridge", [(None, 1, 6.0), ("gabled", 2, 9.0), ("hipped", 4, 9.0),
                                                 ("pyramidal", 4, 9.0), ("skillion", 1, 9.0)])
def test_prism_roof_shapes(shape, n_roofs, ridge):
    s = ex.prism(box(0, 0, 10, 6), -0.3, 6.0, 9.0, shape)
    roofs, walls = _roofs_walls(s)
    assert len(roofs) == n_roofs and len(walls) == 4
    area = 0.0
    for rings in roofs:
        assert rb.newell(rings[0])[2] > 0                    # roofs face up
        v, tri, _ = rb.triangulate(rings)
        area += 0.5 * np.abs(np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])[:, 2]).sum()
    assert area == pytest.approx(60.0, abs=0.01)             # the roof covers the footprint exactly once
    assert max(r[0][:, 2].max() for r in roofs) == pytest.approx(ridge)
    south = next(w for w in walls if np.allclose(w[0][:2, 1], 0.0))
    assert rb.newell(south[0]) == pytest.approx([0, -1, 0], abs=1e-9)   # walls face outwards
    assert south[0][:, 2].min() == pytest.approx(-0.3)


def test_prism_gable_walls_close_under_the_ridge_and_courtyards():
    s = ex.prism(box(0, 0, 10, 6), 0.0, 6.0, 9.0, "gabled")
    _, walls = _roofs_walls(s)
    gable = next(w for w in walls if np.allclose(w[0][:, 0], 10.0))     # the east end
    assert gable[0][:, 2].max() == pytest.approx(9.0) and len(gable[0]) == 5   # a vertex at the ridge
    ring = box(0, 0, 20, 20).difference(box(5, 5, 15, 15))
    roofs, walls = _roofs_walls(ex.prism(ring, 0.0, 10.0))
    assert len(walls) == 8 and len(roofs[0]) == 2                          # outer + courtyard ring
    inner = next(w for w in walls if np.allclose(w[0][:2, 1], 5.0))
    assert rb.newell(inner[0])[1] == pytest.approx(1.0)                    # courtyard walls face the courtyard


def test_heights_and_roof_guesses():
    h = ex.heights(100.0, total_m=12.0, roof_m=3.0)
    assert (h.eaves, h.top, h.guessed) == (109.0, 112.0, False)
    h = ex.heights(100.0, levels=4, region="south")
    assert h.eaves == pytest.approx(112.5) and h.top == h.eaves
    assert ex.heights(0.0).guessed
    assert ex.guess_roof("house", 120.0, 6.0, "nl") == "gabled"
    assert ex.guess_roof("house", 120.0, 6.0, "us") == "hipped"
    assert ex.guess_roof("apartments", 900.0, 20.0, "nl") is None
    assert ex.guess_roof(None, 150.0, 6.0, "south") is None     # southern towns: flat roofs unless known
    assert ex.roof_rise(box(0, 0, 10, 6), "gabled", "nl") == pytest.approx(np.tan(np.radians(40)) * 3)
    assert ex.norm_shape("half-hipped") == "hipped" and ex.norm_shape("weird") is None


def test_make_building_parts_and_eaves_percentile():
    parts = [(box(0, 0, 20, 10), {"total_m": 18.6, "levels": 6}), (box(0, 10, 20, 12), {"total_m": 6.6})]
    b, guessed = ex.make_building("ES.x", parts, ex.Ground(None), region="south", year=1890,
                                  uses={"residential": 1}, levels=6)
    assert not guessed and b.parts == 2 and b.poly.area == pytest.approx(240.0)
    assert b.height == pytest.approx(18.6)                      # 70 % of the area is the tall part
    assert b.roof_z == pytest.approx(18.6) and b.region == "south"
    # built before 1900: the southern old-town facade (ashlar with balconies, 2026-11-25; plain ashlar before)
    assert fc.classify(fc.BuildingInfo(id=b.id, year=b.year, uses=b.uses, height_m=b.height,
                                       area_m2=b.poly.area, region=b.region))[0] == "stone_balcony"


def test_south_facades():
    st, _ = fc.classify(fc.BuildingInfo(id="a", year=1975, uses={"residential": 1}, height_m=15, area_m2=300,
                                        region="south"))
    assert st == "render_warm"
    old_uni, _ = fc.classify(fc.BuildingInfo(id="u", year=1600, uses={"government": 1}, height_m=18, area_m2=3000,
                                             region="south"))
    assert old_uni == "stone_old"
    nl, _ = fc.classify(fc.BuildingInfo(id="a", year=1975, uses={"residential": 1}, height_m=15, area_m2=300))
    assert nl in ("brick_late", "concrete_panel")               # the Netherlands: its own eras (facades.py)
    for name in ("stone_old", "render_warm"):
        alb, nrm, orm = fc.synth_facade(fc.STYLES[name], 0, px=64)
        assert alb.shape == (64, 64, 3) and fc.USE_OF_STYLE[name] == "home"


# --------------------------------------------------------------------------- the providers, from cached answers

def _man(bbox=SALAMANCA, geo=GEO_SAL):
    return {"name": "T", "tile_m": 250.0, "geo_reference": geo, "bbox_wgs84": list(bbox), "stats": {}, "assets": [],
            "buildings": []}


def _gz(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


CATASTRO_XML = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2"
  xmlns:bu-core2d="http://inspire.jrc.ec.europa.eu/schemas/bu-core2d/2.0"
  xmlns:bu-ext2d="http://inspire.jrc.ec.europa.eu/schemas/bu-ext2d/2.0">
 <gml:featureMember><bu-ext2d:Building gml:id="ES.SDGC.BU.123TL">
  <bu-core2d:dateOfConstruction><bu-core2d:DateOfEvent><bu-core2d:beginning>1902-01-01T00:00:00</bu-core2d:beginning>
  </bu-core2d:DateOfEvent></bu-core2d:dateOfConstruction>
  <bu-ext2d:geometry><gml:Surface><gml:patches><gml:PolygonPatch><gml:exterior><gml:LinearRing>
   <gml:posList>40.9645 -5.6640 40.9645 -5.6638 40.9647 -5.6638 40.9647 -5.6640 40.9645 -5.6640</gml:posList>
  </gml:LinearRing></gml:exterior></gml:PolygonPatch></gml:patches></gml:Surface></bu-ext2d:geometry>
  <bu-ext2d:currentUse>1_residential</bu-ext2d:currentUse>
  <bu-ext2d:officialArea><bu-ext2d:OfficialArea><bu-ext2d:value uom="m2">2345</bu-ext2d:value></bu-ext2d:OfficialArea>
  </bu-ext2d:officialArea>
 </bu-ext2d:Building></gml:featureMember>
 <gml:featureMember><bu-ext2d:BuildingPart gml:id="ES.SDGC.BU.123TL_part1">
  <bu-ext2d:geometry><gml:Surface><gml:patches><gml:PolygonPatch><gml:exterior><gml:LinearRing>
   <gml:posList>40.9645 -5.6640 40.9645 -5.6638 40.9646 -5.6638 40.9646 -5.6640 40.9645 -5.6640</gml:posList>
  </gml:LinearRing></gml:exterior></gml:PolygonPatch></gml:patches></gml:Surface></bu-ext2d:geometry>
  <bu-ext2d:numberOfFloorsAboveGround>5</bu-ext2d:numberOfFloorsAboveGround>
 </bu-ext2d:BuildingPart></gml:featureMember>
 <gml:featureMember><bu-ext2d:BuildingPart gml:id="ES.SDGC.BU.123TL_part2">
  <bu-ext2d:geometry><gml:Surface><gml:patches><gml:PolygonPatch><gml:exterior><gml:LinearRing>
   <gml:posList>40.9646 -5.6640 40.9646 -5.6638 40.9647 -5.6638 40.9647 -5.6640 40.9646 -5.6640</gml:posList>
  </gml:LinearRing></gml:exterior></gml:PolygonPatch></gml:patches></gml:Surface></bu-ext2d:geometry>
  <bu-ext2d:numberOfFloorsAboveGround>2</bu-ext2d:numberOfFloorsAboveGround>
 </bu-ext2d:BuildingPart></gml:featureMember>
 <gml:featureMember><bu-ext2d:BuildingPart gml:id="ES.SDGC.BU.123TL_part3">
  <bu-ext2d:geometry><gml:Surface><gml:patches><gml:PolygonPatch><gml:exterior><gml:LinearRing>
   <gml:posList>40.9645 -5.6640 40.9645 -5.6638 40.9647 -5.6638 40.9647 -5.6640 40.9645 -5.6640</gml:posList>
  </gml:LinearRing></gml:exterior></gml:PolygonPatch></gml:patches></gml:Surface></bu-ext2d:geometry>
  <bu-ext2d:numberOfFloorsAboveGround>0</bu-ext2d:numberOfFloorsAboveGround>
 </bu-ext2d:BuildingPart></gml:featureMember>
</gml:FeatureCollection>"""


def test_catastro_parse_and_load(monkeypatch, tmp_path):
    blds, parts = buildings_catastro.parse(CATASTRO_XML)
    b = blds["ES.SDGC.BU.123TL"]
    assert b["year"] == 1902 and b["use"] == "1_residential" and b["area"] == 2345.0
    assert b["polys"][0][0][0] == [-5.664, 40.9645]                     # lon, lat
    assert [p["floors"] for p in parts] == [5, 2, 0] and parts[0]["building"] == "ES.SDGC.BU.123TL"
    seen = []

    def fake_get(params, tries=4):
        seen.append(params)
        return CATASTRO_XML
    monkeypatch.setattr(buildings_catastro, "_get", fake_get)
    monkeypatch.setattr(buildings_catastro, "TILE_DEG", 1.0)
    build = tmp_path / "b"
    out, src = buildings.load("catastro", build, _man())
    assert {p["typeNames"] for p in seen} == {"BU:Building", "BU:BuildingPart"}
    assert seen[0]["srsName"] == "EPSG::4326"
    assert len(out) == 1 and out[0].parts == 2 and out[0].year == 1902        # the underground part left out
    assert out[0].uses == {"residential": 2345.0} and out[0].region == "south"
    assert out[0].roof_z == pytest.approx(5 * 3.0 + 0.6, abs=0.01)
    assert src.name == "catastro" and "CC BY 4.0" in src.asset_source
    assert list((build / "catastro").glob("bu_*.json.gz"))                   # cached
    assert buildings_catastro.covers(SALAMANCA) and not buildings_catastro.covers((43.25, -2.95, 43.27, -2.92))


def test_bdtopo_heights_and_load(tmp_path):
    kw = buildings_bdtopo.part_kwargs({"altitude_minimale_sol": 35.0, "altitude_minimale_toit": 53.0,
                                       "altitude_maximale_toit": 58.0, "nombre_d_etages": 6}, 400.0)
    assert kw == {"levels": 6, "total_m": 23.0, "roof_m": 5.0, "shape": "gabled"}
    assert buildings_bdtopo.part_kwargs({"hauteur": 9.0}, 50.0)["total_m"] == 9.0
    assert buildings_bdtopo.year_of({"date_d_apparition": "1885-01-01"}) == 1885
    paris = (48.8500, 2.3400, 48.8520, 2.3430)
    geo = "+proj=tmerc +lat_0=48.851 +lon_0=2.3415 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    raw = [{"id": "BATIMENT1", "props": {"altitude_minimale_sol": 35.0, "altitude_minimale_toit": 53.0,
                                        "altitude_maximale_toit": 58.0, "usage_1": "Résidentiel",
                                        "date_d_apparition": "1870"},
            "polys": [[[[2.3410, 48.8505], [2.3414, 48.8505], [2.3414, 48.8508], [2.3410, 48.8508],
                        [2.3410, 48.8505]]]]}]
    _gz(tmp_path / "bdtopo" / ("batiment_%.5f_%.5f_%.5f_%.5f.json.gz" % paris), raw)
    out, src = buildings.load("bdtopo", tmp_path, _man(paris, geo))
    assert len(out) == 1 and out[0].year == 1870 and out[0].uses == {"residential": 1.0}
    assert out[0].roof_z == pytest.approx(58.0 - 35.0, abs=0.01)          # no DEM: ground 0
    assert sum(1 for _, k in out[0].surfaces if k == "RoofSurface") == 4   # ~970 m2 with a 5 m roof: hipped


def test_overture_release_query_and_load(monkeypatch, tmp_path):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", boom)
    assert buildings_overture.latest_release() == buildings_overture.RELEASE
    q = buildings_overture.query(ANN_ARBOR, "2026-10-21.0")
    assert "release/2026-10-21.0/theme=buildings" in q and "bbox.xmin < -83.743" in q and "roof_shape" in q
    geo = "+proj=tmerc +lat_0=42.2805 +lon_0=-83.749 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    ring = [[-83.7492, 42.2804], [-83.7490, 42.2804], [-83.7490, 42.2806], [-83.7492, 42.2806], [-83.7492, 42.2804]]
    raw = {"release": "r", "buildings": [
        {"id": "a", "height": 30.0, "num_floors": 8, "roof_shape": None, "roof_height": None,
         "roof_orientation": None, "class": "office", "subtype": "commercial", "is_underground": False,
         "dataset": "OpenStreetMap", "polys": [[ring]]},
        {"id": "b", "height": None, "num_floors": None, "roof_shape": "gabled", "roof_height": 3.0,
         "roof_orientation": "across", "class": "house", "subtype": "residential", "is_underground": False,
         "dataset": "OpenStreetMap", "polys": [[[[x + 0.001, y] for x, y in ring]]]},
        {"id": "c", "height": 4.0, "class": "roof", "is_underground": False, "polys": [[ring]]}]}
    _gz(tmp_path / "overture" / ("buildings_%.5f_%.5f_%.5f_%.5f.json.gz" % ANN_ARBOR), raw)
    out, src = buildings.load("overture", tmp_path, _man(ANN_ARBOR, geo), region="us")
    assert [b.id for b in out] == ["OV.a", "OV.b"]                       # the canopy ("roof") left out
    office, house = out
    assert office.roof_z == pytest.approx(30.0, abs=0.01) and office.uses == {"office": 1.0}
    assert house.uses == {"residential": 1.0} and house.roof_z - house.height == pytest.approx(3.0 + 0.3, abs=0.4)
    assert src.extra_stats["buildings_guessed"] == 1                      # the house has no height


def test_osm_provider_from_the_twin(tmp_path):
    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 8], [0, 8],
                                                                                   [0, 0]]]},
              "properties": {"id": "b1", "osm_id": 7, "height": None, "levels": 2,
                             "tags": {"building": "house", "roof:shape": "hipped"}}},
             {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[20, 0], [24, 0], [24, 3], [20, 3],
                                                                                   [20, 0]]]},
              "properties": {"id": "b2", "osm_id": 8, "tags": {"building": "roof"}}}]
    p = tmp_path / "T.twin" / "buildings.geojson"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    out, src = buildings_osm.load(tmp_path, _man(), region="nl")
    assert len(out) == 1 and out[0].id == "OSM.7" and out[0].levels == 2
    assert sum(1 for _, k in out[0].surfaces if k == "RoofSurface") == 4      # hipped
    assert src.name == "osm"


def test_candidates_order_and_fallback_listing():
    assert buildings.candidates(SALAMANCA)[-2:] == ["overture", "osm"]
    assert buildings.candidates(SALAMANCA, order=["catastro", "overture", "osm"]) == ["catastro", "overture", "osm"]
    assert buildings.candidates(BERLIN) == ["overture", "osm"]


def test_real_buildings_tool_falls_back(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cc_real_buildings_tool",
                                                  Path(__file__).resolve().parents[1] / "tools" / "real_buildings.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    build = tmp_path / "b"
    (build / "ue").mkdir(parents=True)
    (build / "ue" / "manifest.json").write_text(json.dumps(_man()))
    (build / "report.json").write_text(json.dumps({"profile": {"iso2": "ES"}}))
    tried, applied = [], {}

    def fake_load(name, b, man, **kw):
        tried.append((name, kw.get("region")))
        if name == "catastro":
            raise RuntimeError("service down")
        return [rb.RealBuilding(id="x", surfaces=[])], rb.Source(name=name, label=name, asset_source=name)
    monkeypatch.setattr(tool.providers, "load", fake_load)
    monkeypatch.setattr(tool.realbuildings, "apply", lambda *a, **k: applied.setdefault("source", a[3].name))
    assert tool.main([str(build), "--plain"]) == 0
    assert tried == [("catastro", "south"), ("overture", "south")] and applied["source"] == "overture"


# --------------------------------------------------------------------------- twinmodel twin

def test_names_boxes_and_cli():
    assert twin.camel_name("Plaza Mayor, Salamanca") == "PlazaMayorSalamanca"
    assert twin.camel_name("渋谷") == "Twin" and twin.camel_name("1st Street") == "Twin1stStreet"
    s, w, n, e = twin.square_bbox(40.965, -5.664, 1.0)
    assert (n - s) * 111.32 == pytest.approx(1.0, abs=0.01)
    from twinmodel import cli
    args = cli._build_parser().parse_args(["twin", "Plaza", "Mayor,", "Salamanca", "--name", "Sal", "--bake",
                                           "--no-nav-build", "--max-rss-gb", "20"])
    assert args.place == ["Plaza", "Mayor,", "Salamanca"] and args.bake and args.no_nav_build
    assert args.max_rss_gb == 20.0 and args.func is twin.main_args


def test_geocode_is_cached(monkeypatch, tmp_path):
    import requests
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append(params["q"])
        assert "User-Agent" in headers
        return _Resp(b"", "application/json", js=[{"lat": "40.965", "lon": "-5.664", "display_name": "Plaza Mayor",
                                                    "address": {"country_code": "es"}}])
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(twin.time, "sleep", lambda s: None)
    g = twin.geocode("Plaza Mayor, Salamanca", tmp_path)
    g2 = twin.geocode("plaza  mayor, salamanca", tmp_path)
    assert g == g2 and g["country_code"] == "ES" and calls == ["Plaza Mayor, Salamanca"]


def test_editor_paths_need_the_toolchain(monkeypatch, tmp_path):
    monkeypatch.setattr(twin, "TOOLCHAIN_FILE", tmp_path / "none.json")
    with pytest.raises(twin.TwinError):
        twin.editor_paths({"carla_root": None, "engine": None})
    root, eng = tmp_path / "carla", tmp_path / "ue"
    (root / "Unreal" / "CarlaUnreal").mkdir(parents=True)
    (root / "Unreal" / "CarlaUnreal" / "CarlaUnreal.uproject").write_text("{}")
    exe = eng / "Engine" / "Binaries" / "Linux" / "UnrealEditor-Cmd"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert twin.editor_paths({"carla_root": str(root), "engine": str(eng)}) == (exe, root / "Unreal" / "CarlaUnreal"
                                                                                / "CarlaUnreal.uproject")


def test_twin_runs_the_steps_and_writes_credits(monkeypatch, tmp_path, capsys):
    out = tmp_path / "sal"
    ran = []

    def fake_run(cmd, log_path, cwd=None, watchdog=None, nice=0):
        cmd = [str(c) for c in cmd]
        ran.append(cmd)
        if "build" in cmd and "-m" in cmd and cmd[cmd.index("-m") + 2] == "build":
            (out / "Sal.twin").mkdir(parents=True, exist_ok=True)
            (out / "Sal.twin" / "model.json").write_text("{}")
            (out / "Sal.xodr").write_text("<OpenDRIVE/>")
            (out / "report.json").write_text(json.dumps({
                "profile": {"name": "eu_dense", "iso2": "ES"}, "elevation": {"source": "ign_wcs", "applied": True,
                                                                             "z_min": 790, "z_max": 810},
                "build": {"imagery": {"source": "ign_es"}}}))
            return {"rc": 1, "seconds": 1.0, "peak_rss_gb": 0.0, "killed": None, "tail": []}
        if "bake-export" in cmd:
            (out / "ue").mkdir(parents=True, exist_ok=True)
            (out / "ue" / "manifest.json").write_text(json.dumps({"name": "Sal", "stats": {"assets": 10}}))
        if cmd[1].endswith("real_buildings.py"):
            m = json.loads((out / "ue" / "manifest.json").read_text())
            m["stats"].update(real_buildings="catastro", buildings=3)
            (out / "ue" / "manifest.json").write_text(json.dumps(m))
        if cmd[1].endswith("ground_photo.py"):
            m = json.loads((out / "ue" / "manifest.json").read_text())
            m["stats"]["ground_photo"] = {"tiles": 16}
            (out / "ue" / "manifest.json").write_text(json.dumps(m))
        return {"rc": 0, "seconds": 1.0, "peak_rss_gb": 0.0, "killed": None, "tail": []}
    monkeypatch.setattr(twin, "run", fake_run)
    from twinmodel import cli
    args = cli._build_parser().parse_args(["twin", "--bbox", *map(str, SALAMANCA), "--name", "Sal", "--out", str(out),
                                           "--cache", str(tmp_path / "data")])
    assert args.func(args) == 0
    steps = [c[c.index("-m") + 2] if "-m" in c else Path(c[1]).name for c in ran]
    assert steps == ["build", "bake-export", "real_buildings.py", "ground_photo.py", "road_clearance.py",
                     "landmarks.py", "surface_gaps.py", "road_paving.py"]
    text = capsys.readouterr().out
    assert "[twin] done build: terrain ign_wcs 790-810 m, photo ign_es" in text
    assert "[twin] skip bake:" in text and "validation findings" in text
    attr = (out / "ue" / "ATTRIBUTION.md").read_text()
    assert "Catastro" in attr and "PNOA" in attr
    assert json.loads((out / "ue" / "manifest.json").read_text())["attribution"] == "ATTRIBUTION.md"
    summ = json.loads((out / "twin.json").read_text())
    assert summ["level"] == "/Game/Carla/Maps/Twins/Sal/Sal" and summ["buildings"] == 3
    assert summ["country"] == "ES" and summ["region_plan"]["buildings"][0] == "catastro"
    # a second run skips the steps done before (the clearance cut and the steps after it are always redone: they
    # start from the uncut meshes)
    ran.clear()
    assert args.func(args) == 0
    assert [Path(c[1]).name for c in ran] == ["road_clearance.py", "landmarks.py", "surface_gaps.py", "road_paving.py"]


def test_twin_stops_when_lanes_stay_blocked(monkeypatch, tmp_path, capsys):
    """The clearance check failing (exit 3) stops the twin before the bake, with the numbers in twin.json."""
    out = tmp_path / "blk"
    (out / "Blk.twin").mkdir(parents=True)
    (out / "Blk.twin" / "model.json").write_text("{}")
    (out / "Blk.xodr").write_text("<OpenDRIVE/>")
    (out / "ue").mkdir()
    (out / "ue" / "manifest.json").write_text(json.dumps({"name": "Blk", "stats": {"real_buildings": "osm"}}))
    (out / "report.json").write_text(json.dumps({"profile": {"iso2": "ES"}, "build": {"imagery": {"source": "x"}}}))

    def fake_run(cmd, log_path, cwd=None, watchdog=None, nice=0):
        if str(cmd[1]).endswith("road_clearance.py"):
            (out / "clearance.json").write_text(json.dumps({"ok": False, "cut": {"passage": 1, "trim": 0}, "after": {
                "road": {"blocked": 3, "samples": 10, "hits": []}, "walk": {"blocked": 0, "samples": 5, "hits": []}}}))
            return {"rc": 3, "seconds": 1.0, "peak_rss_gb": 0.0, "killed": None, "tail": ["clearance: FAILED"]}
        return {"rc": 0, "seconds": 1.0, "peak_rss_gb": 0.0, "killed": None, "tail": []}
    monkeypatch.setattr(twin, "run", fake_run)
    from twinmodel import cli
    args = cli._build_parser().parse_args(["twin", "--bbox", *map(str, SALAMANCA), "--name", "Blk", "--out", str(out),
                                           "--cache", str(tmp_path / "data"), "--skip", "ground"])
    assert args.func(args) == 2
    text = capsys.readouterr().out
    assert "[twin] failed clearance:" in text and "still stand on the lanes" in text
    summ = json.loads((out / "twin.json").read_text())
    assert summ["clearance"]["ok"] is False and summ["clearance"]["after"]["road"]["blocked"] == 3


# --------------------------------------------------------------------------- left-hand traffic by mirroring

def test_mirror_tags():
    from twinmodel import lefthand
    assert lefthand.mirror_tag("turn:lanes", "left|through|through;right") == ("turn:lanes", "through;left|through|right")
    assert lefthand.mirror_tag("sidewalk:left", "separate") == ("sidewalk:right", "separate")
    assert lefthand.mirror_tag("sidewalk", "left") == ("sidewalk", "right")
    assert lefthand.mirror_tag("restriction", "no_left_turn") == ("restriction", "no_right_turn")
    assert lefthand.mirror_tag("width:lanes", "3.0|3.5") == ("width:lanes", "3.5|3.0")
    assert lefthand.mirror_tag("name", "Right Street") == ("name", "Right Street")      # names stay
    assert lefthand.mirror_tag("bright", "x") == ("bright", "x")
    assert lefthand.mirror_heading(0.0) == pytest.approx(np.pi) and lefthand.mirror_heading(np.pi / 2) == \
        pytest.approx(np.pi / 2)


def test_lefthand_lane_graph_and_xodr():
    import copy

    from twinmodel import lefthand, profiles
    from twinmodel.export.xodr import export_xodr
    from twinmodel.frame import LocalFrame
    from twinmodel.ingest.osm import load_fixture
    from twinmodel.lanegraph import build_lanegraph
    fixture = Path(__file__).parent / "fixtures" / "eixample_overpass.json"
    bbox = (41.3905, 2.1630, 41.3945, 2.1690)
    frame = LocalFrame.from_bbox(*bbox)
    osm = load_fixture(fixture)
    mirrored = lefthand.mirror_osm(copy.deepcopy(osm), frame.origin_lon)
    n0 = next(iter(osm.nodes.values()))
    assert mirrored.nodes[n0.id].lon == pytest.approx(2 * frame.origin_lon - n0.lon)
    x, _ = frame.to_local(n0.lon, n0.lat)
    xm, _ = frame.to_local(mirrored.nodes[n0.id].lon, n0.lat)
    assert xm == pytest.approx(-x, abs=1e-6)                     # the frame is symmetric: an exact mirror
    with profiles.use(profiles.JP_URBAN.with_(drives_on="right")):
        model = build_lanegraph(mirrored, frame, bbox, name="lht")
    before = {r.id: (np.asarray(r.reference_line.coords)[:, :2].copy(), sorted(l.id for l in r.lanes))
              for r in model.roads}
    stats = lefthand.mirror_model(model)
    assert stats["roads"] == len(model.roads) and stats["lane_links"] > 0
    two_way = 0
    for r in model.roads:
        xy, ids = before[r.id]
        assert np.allclose(np.asarray(r.reference_line.coords)[:, 0], -xy[:, 0])
        assert sorted(l.id for l in r.lanes) == sorted(-i for i in ids)
        fwd = [l for l in r.lanes if l.type == "driving" and l.direction == "forward"]
        back = [l for l in r.lanes if l.type == "driving" and l.direction == "backward"]
        if fwd and back and not r.junction_id:
            two_way += 1
            assert all(l.id > 0 for l in fwd) and all(l.id < 0 for l in back)   # driving on the left
    assert two_way > 0
    for j in model.junctions:
        for c in j.connections:
            inc, con = model.road(c.incoming_road), model.road(c.connecting_road)
            for ll in c.lane_links:
                assert any(l.id == ll.from_lane for l in inc.lanes) and any(l.id == ll.to_lane for l in con.lanes)
    with profiles.use("jp_urban"):
        xodr = export_xodr(model)
    assert xodr.count('rule="LHT"') == len(model.roads)
    with profiles.use("eu_dense"):
        assert 'rule="LHT"' not in export_xodr(model)


def test_lefthand_countries_get_left_hand_profiles(monkeypatch, tmp_path):
    from twinmodel import cli, profiles
    assert profiles.choose_for_country("JP").name == "jp_urban"
    monkeypatch.setattr(cli, "building_coverage", lambda osm, frame, bbox: 0.4)
    monkeypatch.setattr(cli, "country_for_bbox", lambda bbox, cache: "GB")
    p, meta = cli.select_profile("auto", None, None, (51.50, -0.13, 51.51, -0.12), tmp_path)
    assert p.name == "eu_dense" and p.drives_on == "left" and meta["region_plan"]["lefthand"]
    monkeypatch.setattr(cli, "country_for_bbox", lambda bbox, cache: "JP")
    p, _ = cli.select_profile("auto", None, None, (35.655, 139.695, 35.664, 139.706), tmp_path)
    assert p.name == "jp_urban" and p.drives_on == "left" and p.sources.ortho[0] == "gsi_photo"


# --------------------------------------------------------------------------- a city model's own photo textures

def test_atlas_packing_and_uvs(tmp_path):
    scale, place = rb.pack_atlas([(256, 128), (64, 64), (512, 512)], atlas_px=1024, gutter=2)
    assert scale == 1.0 and all(p is not None for p in place)
    rects = [box(x, y, x + w, y + h) for x, y, w, h in place]
    assert not any(a.intersects(b) and a.intersection(b).area > 0 for i, a in enumerate(rects) for b in rects[i + 1:])
    s2, _ = rb.pack_atlas([(2048, 2048)] * 4, atlas_px=1024)
    assert s2 < 0.5
    uv = rb.atlas_uv(np.array([[0.0, 0.0], [1.0, 1.0]]), (100, 200, 50, 40), atlas_px=1000)
    assert uv[0] == pytest.approx([(100 + 0.5) / 1000, (200 + 40 - 0.5) / 1000])   # bottom left -> v down
    assert uv[1] == pytest.approx([(150 - 0.5) / 1000, (200 + 0.5) / 1000])
    from PIL import Image
    paths = []
    for k, c in enumerate(((255, 0, 0), (0, 255, 0))):
        p = tmp_path / f"t{k}.png"
        Image.new("RGB", (32, 16), c).save(p)
        paths.append(p)
    atlas, place = rb.build_atlas(paths, atlas_px=128)
    x, y, w, h = place[1]
    assert tuple(atlas[y + 2, x + 2]) == (0, 255, 0) and tuple(atlas[y - 1, x]) == (0, 255, 0)   # gutter repeats


def test_textured_surfaces_go_to_the_tile_atlas(tmp_path):
    from PIL import Image
    img = tmp_path / "facade.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(img)
    surf = ex.prism(box(0, 0, 10, 10), 0.0, 12.0)
    walls = [i for i, (_, k) in enumerate(surf) if k == "WallSurface"]
    tex = {walls[0]: (str(img), [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)])}
    b = rb.RealBuilding(id="JP.x", surfaces=surf, poly=box(0, 0, 10, 10), ground=0.0, height=12.0, roof_z=12.0,
                        region="jp", textures=tex)
    build = tmp_path / "b"
    (build / "ue" / "meshes").mkdir(parents=True)
    man = _man()
    stats = rb.apply(build, man, [b], rb.Source(name="plateau", label="p", asset_source="PLATEAU"), texture_px=32)
    assert stats["photo_textured_surfaces"] == 1 and stats["photo_atlases"] == 1
    photo = [a for a in man["assets"] if a["material"].startswith("photo_")]
    assert len(photo) == 1 and photo[0]["asset"].endswith("_photo") and photo[0]["kind"] == "building_3dbag"
    assert (build / "ue" / man["textures"][photo[0]["material"]]["base_color"]).exists()


def test_crossing_ways_become_zebras_with_their_walking_direction():
    from twinmodel import profiles
    from twinmodel.export.ue import crossing_walk_dir, zebra_stripes
    from twinmodel.ingest.osm import parse_osm
    from twinmodel.frame import LocalFrame
    from twinmodel.surfaces import crossing_ways
    raw = {"elements": [
        {"type": "node", "id": 1, "lat": 35.6590, "lon": 139.7000}, {"type": "node", "id": 2, "lat": 35.6600,
                                                                      "lon": 139.7010},
        {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "footway", "footway": "crossing"}},
        {"type": "way", "id": 11, "nodes": [1, 2], "tags": {"highway": "footway"}}]}
    frame = LocalFrame.from_bbox(35.6585, 139.6995, 35.6605, 139.7015)
    ways = crossing_ways(parse_osm(raw), frame)
    assert len(ways) == 1 and ways[0].startswith("LINESTRING")
    assert profiles.JP_URBAN.crossing.from_ways and not profiles.EU_DENSE.crossing.from_ways

    class S:
        tags = {"walk_dir": [0.7071, 0.7071]}
        road_ids = []
    poly = box(0, 0, 10, 10).intersection(Polygon([(0, -2), (12, 10), (10, 12), (-2, 0)]))
    d = crossing_walk_dir(None, S(), poly)
    assert d == pytest.approx([0.7071, 0.7071])
    bars = zebra_stripes(poly, along=d)
    assert len(bars) > 5                                   # diagonal bars across the diagonal walk
