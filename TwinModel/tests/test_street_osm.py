"""Tests for the street side of the facade styles: twinmodel.ingest.street_osm and its use in realbuildings.apply
(shopfront ground floors from OSM shops at the facade, OSM wall materials and colours). No network."""
from __future__ import annotations

import json

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from twinmodel.export import realbuildings as rb
from twinmodel.ingest import street_osm

from .test_realbuildings import _box

GEO = "+proj=tmerc +lat_0=51.443 +lon_0=5.4815 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m"


def test_parse_and_ground_floor():
    data = {"elements": [
        {"type": "node", "id": 1, "lat": 51.44, "lon": 5.48, "tags": {"shop": "bakery", "name": "B", "opening": "x"}},
        {"type": "node", "id": 2, "lat": 51.44, "lon": 5.48, "tags": {"amenity": "cafe", "level": "1"}},
        {"type": "node", "id": 3, "lat": 51.44, "lon": 5.48, "tags": {"amenity": "bench"}},
        {"type": "way", "id": 4, "center": {"lat": 51.441, "lon": 5.481},
         "tags": {"building": "yes", "building:material": "glass", "name": "x"}},
        {"type": "way", "id": 5, "tags": {"building": "yes", "building:colour": "red"}},   # no centre: skipped
    ]}
    out = street_osm.parse(data)
    assert [p["tags"] for p in out["pois"]] == [{"shop": "bakery", "name": "B"}, {"amenity": "cafe", "level": "1"}]
    assert out["buildings"] == [{"lat": 51.441, "lon": 5.481, "tags": {"building": "yes",
                                                                       "building:material": "glass"}}]
    assert street_osm.ground_floor({}) and street_osm.ground_floor({"level": "0;1"})
    assert street_osm.ground_floor({"level": "-1"}) and not street_osm.ground_floor({"level": "2"})
    assert street_osm.ground_floor({"level": "ground"})        # unreadable: assume the street
    m = street_osm.to_model(out, GEO)
    assert len(m["pois"]) == 1                                  # the first-floor cafe is left out
    x, y, _ = m["buildings"][0]
    assert x == pytest.approx(-34.8, abs=1.0) and y == pytest.approx(-222.4, abs=1.0)
    assert street_osm.to_model(None, GEO) is None
    assert "craft" in street_osm.query((1, 2, 3, 4)) and "out tags center" in street_osm.query((1, 2, 3, 4))


def test_fetch_is_cached_and_fails_soft(monkeypatch, tmp_path):
    import requests
    calls = []

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"elements": [{"type": "node", "lat": 1.0, "lon": 2.0, "tags": {"shop": "books"}}]}

    def post(url, data=None, timeout=None, headers=None):
        calls.append(url)
        if len(calls) == 1:
            raise requests.ConnectionError("busy")   # the first mirror is down: the next one answers
        return R()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(street_osm, "ROUND_WAIT_S", 0.0)
    out = street_osm.fetch((1, 2, 3, 4), tmp_path)
    # the first mirror down, the next one answers the street query, then the historic areas (2026-11-25)
    assert out["pois"][0]["tags"] == {"shop": "books"} and len(calls) == 3 and out["version"] == street_osm.VERSION
    assert street_osm.fetch((1, 2, 3, 4), tmp_path) == out and len(calls) == 3    # cached
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    assert street_osm.fetch((5, 6, 7, 8), tmp_path) is None


def test_street_matches_and_shop_walls():
    polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), Polygon([(10.2, 0), (20, 0), (20, 10), (10.2, 10)]),
             Polygon([(40, 0), (100, 0), (100, 60), (40, 60)])]
    tree = STRtree(polys)
    street = {"pois": [(5.0, -2.0, {}),        # on the pavement in front of building 0
                       (15.0, 1.0, {}),        # inside building 1, near its front wall
                       (70.0, 30.0, {}),       # deep inside the mall (building 2): none
                       (30.0, 30.0, {})],      # in the street, far from all
              "buildings": [(15.0, 5.0, {"building:colour": "white"})]}
    shops, tags = rb.street_matches(polys, tree, street)
    assert dict(shops) == {0: [(5.0, -2.0, 2.0)], 1: [(15.0, 1.0, 1.0)]}
    assert tags == {1: {"building:colour": "white"}}
    assert rb.street_matches(polys, tree, None) == ({}, {})
    front = np.array([[0, 0, 0], [10, 0, 0], [10, 0, 9], [0, 0, 9]], float)
    back = np.array([[10, 10, 0], [0, 10, 0], [0, 10, 9], [10, 10, 9]], float)
    side = np.array([[10, 0, 0], [10, 10, 0], [10, 10, 9], [10, 0, 9]], float)
    assert rb.shop_wall(front, [(5.0, -2.0, 2.0)]) and not rb.shop_wall(back, [(5.0, -2.0, 2.0)])
    assert not rb.shop_wall(side, [(5.0, -2.0, 2.0)])          # 5.4 m from the shop: not its front
    assert rb.shop_wall(side, [(9.5, -0.5, 0.5)])              # a corner shop: both walls
    assert rb.shop_wall(back, [])                              # shops known only from the registers: all walls


def test_apply_with_street_osm(tmp_path):
    from twinmodel.export import facades as fc
    blds = [rb.RealBuilding(id="NL.IMBAG.Pand.1", surfaces=_box(0, 0, 10, 20, 0.0, 12.0),
                            poly=Polygon([(0, 0), (10, 0), (10, 20), (0, 20)]), ground=0.0, height=12.0, year=1958,
                            uses={"woonfunctie": 400, "bijeenkomstfunctie": 60}),
            rb.RealBuilding(id="NL.IMBAG.Pand.2", surfaces=_box(30, 0, 40, 20, 0.0, 9.0),
                            poly=Polygon([(30, 0), (40, 0), (40, 20), (30, 20)]), ground=0.0, height=9.0, year=2004,
                            uses={"woonfunctie": 300})]
    src = rb.Source(name="3dbag", label="3DBAG", asset_source="3DBAG")

    def run(street):
        build = tmp_path / ("s" if street else "n")
        (build / "ue" / "meshes").mkdir(parents=True)
        man = {"name": "T", "tile_m": 250.0, "geo_reference": GEO, "bbox_wgs84": [51.44, 5.48, 51.45, 5.49],
               "stats": {"buildings": 0}, "assets": [], "buildings": []}
        stats = rb.apply(build, man, blds, src, texture_px=64, roof_px=64, window_lights=False, street=street)
        return stats, json.loads((build / "ue" / "manifest.json").read_text())

    stats, man = run(None)            # no OSM: the cafe-less assembly unit still makes a shopfront, on all walls
    assert stats["shopfront_buildings"] == 1 and stats["shopfront_walls"] == 4
    assert "street_osm" not in stats
    # OSM: a shop in front of building 2's south wall, building 2 mapped as anthracite brick; building 1's
    # assembly unit has no shop or cafe at its walls
    stats, man = run({"pois": [(35.0, -1.0, {"shop": "bakery"})],
                      "buildings": [(35.0, 10.0, {"building:colour": "#505050", "building:material": "brick"})]})
    assert stats["shopfront_buildings"] == 1 and stats["shopfront_walls"] == 1
    assert stats["street_osm"] == {"pois": 1, "pois_at_buildings": 1, "buildings_with_wall_tags": 1}
    by_id = {b["id"]: b for b in man["buildings"]}
    assert by_id["NL.IMBAG.Pand.1"]["style"] == "brick_wederopbouw" and not by_id["NL.IMBAG.Pand.1"]["shopfront"]
    assert by_id["NL.IMBAG.Pand.2"]["style"] == "brick_contemporary" and by_id["NL.IMBAG.Pand.2"]["shopfront"]
    var = fc.STYLES["brick_contemporary"].colours.index((82, 80, 80))
    assert fc.material_key("brick_contemporary", var) in {a["material"] for a in man["assets"]}


def test_historic_areas_and_old_buildings():
    # an old town as a relation of two outer ways, a district as a closed way (Overpass out geom)
    g = lambda pts: [{"lat": la, "lon": lo} for la, lo in pts]     # noqa: E731
    data = {"elements": [
        {"type": "relation", "tags": {"heritage": "1", "boundary": "protected_area", "name": "Old City"},
         "members": [{"type": "way", "role": "outer", "geometry": g([(0, 0), (0, 1), (1, 1)])},
                     {"type": "way", "role": "outer", "geometry": g([(1, 1), (1, 0), (0, 0)])}]},
        {"type": "way", "tags": {"historic": "district"}, "geometry": g([(2, 2), (2, 3), (3, 3), (2, 2)])},
        {"type": "way", "tags": {"heritage": "2"}, "geometry": g([(5, 5), (5, 6)])}]}       # open: not an area
    areas = street_osm.parse_areas(data)
    assert len(areas) == 2 and areas[0]["tags"]["name"] == "Old City" and len(areas[0]["rings"][0]) == 5
    model = street_osm.to_model({"pois": [], "buildings": [], "areas": areas}, "EPSG:3857")
    union = rb.historic_areas(model)
    assert union is not None and union.area > 0
    assert rb.old_building({"heritage": "2"}) and rb.old_building({"historic": "building"})
    assert rb.old_building({"start_date": "1755"}) and rb.old_building({"start_date": "~1820"})
    assert rb.old_building({"building:architecture": "neoclassicism"}) and rb.old_building({"start_date": "C18"})
    assert not rb.old_building({"start_date": "1962"}) and not rb.old_building({"historic": "no"})
    assert not rb.old_building({})


def test_old_cores_from_old_buildings():
    # seven old buildings 20 m apart form an old core; a lone old church and its modern neighbours do not
    sq = lambda x, y: Polygon([(x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10)])   # noqa: E731
    polys = [sq(20 * i, 0) for i in range(7)] + [sq(60, 30)] + [sq(1000, 0), sq(1020, 0)]
    old = [True] * 7 + [False] + [True, False]
    core = rb.old_cores(polys, old)
    assert core is not None and core.contains(polys[7].representative_point())        # the new block inside it
    assert not core.contains(polys[8].representative_point())                         # the lone church
    assert rb.old_cores(polys, [False] * len(polys)) is None
