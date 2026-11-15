"""Spanish buildings for twinmodel.export.realbuildings: the Catastro's INSPIRE buildings (CC BY 4.0).

The Direccion General del Catastro publishes every building of Spain except the Basque Country and Navarre (their
own cadastres) through its INSPIRE WFS, keyless: the building (``BU:Building``: footprint, year of construction,
current use, gross floor area) and its parts (``BU:BuildingPart``: one footprint per number of floors above ground,
so a block's street front and its lower back wings are separate parts). Each part becomes a prism of its floors
(twinmodel.ingest.buildings_extrude), standing on the twin's terrain; parts with no floor above ground (garages
under a courtyard) are left out. The Catastro gives no roof shapes: roofs are flat (and take the aerial photo).
Requests go in tiles of ``TILE_DEG`` (the service answers large boxes slowly), cached in <build>/catastro/.
"""
from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..export.realbuildings import Source, cache_json
from . import buildings_extrude as ex

WFS = "http://ovc.catastro.meh.es/INSPIRE/wfsBU.aspx"
COVERAGE = (27.60, -18.20, 43.85, 4.40)          # Spain with the Canaries; covers() leaves out the Basque Country
NO_CATASTRO = [(42.45, -3.45, 43.45, -1.70),      # Basque Country (own cadastres): rough boxes
               (41.90, -2.50, 43.32, -0.70)]      # Navarre
TILE_DEG = 0.0035                                 # ~300 m
WORKERS = 4                                       # requests at once
STOREY_M = 3.0
GROUND_FLOOR_EXTRA_M = 0.6
UA = "twinmodel (carla-digitaltwins)"
USES = {"1_residential": "residential", "2_agriculture": "storage", "3_industrial": "industry",
        "4_1_office": "office", "4_2_retail": "retail", "4_3_publicServices": "government"}
NS = {"gml": "http://www.opengis.net/gml/3.2", "bu-core2d": "http://inspire.jrc.ec.europa.eu/schemas/bu-core2d/2.0",
      "bu-ext2d": "http://inspire.jrc.ec.europa.eu/schemas/bu-ext2d/2.0"}


def covers(bbox) -> bool:
    s, w, n, e = bbox
    lat, lon = (s + n) / 2, (w + e) / 2
    return not any(bs <= lat <= bn and bw <= lon <= be for bs, bw, bn, be in NO_CATASTRO)


def _poslist(text: str):
    """'lat lon lat lon ...' (EPSG:4326 axis order) -> [[lon, lat], ...]."""
    v = np.asarray(text.split(), dtype=np.float64).reshape(-1, 2)
    return v[:, ::-1].round(7).tolist()


def _polygons(elem):
    """[[outer, *inner rings as [[lon, lat], ...]]] of the gml surfaces under elem."""
    out = []
    for patch in elem.iter("{%s}PolygonPatch" % NS["gml"]):
        rings = []
        for tag in ("exterior", "interior"):
            for ring in patch.findall("gml:%s" % tag, NS):
                pl = ring.find(".//gml:posList", NS)
                if pl is not None and pl.text:
                    rings.append(_poslist(pl.text))
        if rings:
            out.append(rings)
    return out


def parse(xml_bytes: bytes):
    """(buildings, parts) of a WFS answer: buildings {id: {year, use, area, polys}}, parts [{building, floors, polys}]."""
    buildings, parts = {}, []
    root = ET.fromstring(xml_bytes)
    for b in root.iter("{%s}Building" % NS["bu-ext2d"]):
        gid = b.get("{%s}id" % NS["gml"], "")
        year = None
        beg = b.find(".//bu-core2d:dateOfConstruction//bu-core2d:beginning", NS)
        if beg is not None and beg.text and re.match(r"\d{4}", beg.text):
            year = int(beg.text[:4])
        use = b.findtext("bu-ext2d:currentUse", default="", namespaces=NS)
        area = b.findtext(".//bu-ext2d:officialArea//bu-ext2d:value", default="", namespaces=NS)
        try:
            area = float(area)
        except ValueError:
            area = 0.0
        buildings[gid] = {"year": year if year and 1000 < year < 2100 else None, "use": use, "area": area,
                          "polys": _polygons(b)}
    for p in root.iter("{%s}BuildingPart" % NS["bu-ext2d"]):
        gid = p.get("{%s}id" % NS["gml"], "")
        floors = p.findtext("bu-ext2d:numberOfFloorsAboveGround", default="", namespaces=NS)
        try:
            floors = int(floors)
        except ValueError:
            floors = None
        parts.append({"id": gid, "building": gid.split("_part")[0], "floors": floors, "polys": _polygons(p)})
    return buildings, parts


def _get(params, tries=4):
    import requests
    for k in range(tries):
        try:
            r = requests.get(WFS, params=params, timeout=180, headers={"User-Agent": UA})
            r.raise_for_status()
            if b"FeatureCollection" not in r.content[:3000]:
                raise RuntimeError("Catastro WFS: %s" % r.content[:300])
            return r.content
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(3.0 * (k + 1))


def tiles(bbox, step=TILE_DEG):
    s, w, n, e = bbox
    ny, nx = max(1, math.ceil((n - s) / step)), max(1, math.ceil((e - w) / step))
    for i in range(ny):
        for j in range(nx):
            yield (s + i * (n - s) / ny, w + j * (e - w) / nx, s + (i + 1) * (n - s) / ny, w + (j + 1) * (e - w) / nx)


def fetch(bbox, workers: int = WORKERS):
    """{"buildings": {...}, "parts": [...]} for the bbox (each feature once). The service takes 10-30 s an answer,
    so a few tiles are asked at once."""
    from concurrent.futures import ThreadPoolExecutor
    blds, parts, seen = {}, [], set()
    tl = list(tiles(bbox))

    def one(job):
        (s, w, n, e), tn = job
        return parse(_get({"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": tn,
                           "bbox": f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}", "srsName": "EPSG::4326"}))
    jobs = [(t, tn) for t in tl for tn in ("BU:Building", "BU:BuildingPart")]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for k, (b, p) in enumerate(ex.map(one, jobs)):
            blds.update(b)
            for part in p:
                if part["id"] not in seen:
                    seen.add(part["id"])
                    parts.append(part)
            if k % 2 == 1:
                print("catastro: tile %d of %d: %d buildings, %d parts so far" % (k // 2 + 1, len(tl), len(blds),
                                                                                 len(parts)), flush=True)
    return {"buildings": blds, "parts": parts}


def part_height(floors: int) -> float:
    return floors * STOREY_M + GROUND_FLOOR_EXTRA_M


def load(build: Path, man: dict, looks: bool = True, region: str = "south", **_):
    s, w, n, e = bbox = tuple(man["bbox_wgs84"])
    raw = cache_json(build / "catastro" / ("bu_%.5f_%.5f_%.5f_%.5f.json.gz" % bbox), lambda: fetch(bbox))
    tr = ex.model_transformer(man)
    ground = ex.ground_for(build, man)
    by_bld = {}
    for p in raw["parts"]:
        if not p.get("floors") or p["floors"] <= 0:
            continue
        for rings in p["polys"]:
            poly = ex.to_model(tr, rings)
            if poly is not None:
                by_bld.setdefault(p["building"], []).append((poly, {"total_m": part_height(p["floors"]),
                                                                    "levels": p["floors"]}))
    out, n_parts_only = [], 0
    for bid, info in raw["buildings"].items():
        parts = by_bld.pop(bid, None)
        if not parts:   # a building without parts above ground (rare): its footprint at two floors
            polys = [ex.to_model(tr, r) for r in info.get("polys", [])]
            parts = [(p, {"total_m": part_height(2), "levels": 2}) for p in polys if p is not None]
            if not parts:
                continue
        use = USES.get(info.get("use") or "")
        levels = max(kw["levels"] for _, kw in parts)
        rb, _ = ex.make_building("ES." + bid.split(".")[-1], parts, ground, region=region, year=info.get("year"),
                                 uses={use: info.get("area") or 1.0} if use else {}, levels=levels)
        if rb is not None and rb.poly.intersects(_bbox_poly(tr, bbox)):
            out.append(rb)
    for bid, parts in by_bld.items():   # parts whose building lies outside the bbox tiles' answers
        rb, _ = ex.make_building("ES." + bid.split(".")[-1], parts, ground, region=region,
                                 levels=max(kw["levels"] for _, kw in parts))
        if rb is not None:
            out.append(rb)
            n_parts_only += 1
    print(f"catastro: {len(out)} buildings ({sum(b.parts for b in out)} parts; "
          f"{sum(1 for b in out if b.year)} with a construction year)")
    src = Source(name="catastro", label="Catastro INSPIRE buildings (DG del Catastro)",
                 asset_source="Direccion General del Catastro INSPIRE buildings (CC BY 4.0)", roof_photo_max_m=40.0,
                 extra_stats={"buildings_with_year": sum(1 for b in out if b.year)})
    return out, src


def _bbox_poly(tr, bbox):
    from shapely.geometry import box
    s, w, n, e = bbox
    x0, y0 = tr.transform(w, s)
    x1, y1 = tr.transform(e, n)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
