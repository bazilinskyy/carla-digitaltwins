"""French buildings for twinmodel.export.realbuildings: IGN BD TOPO (Licence Ouverte 2.0), keyless WFS.

BD TOPO's ``batiment`` layer gives every building in France its footprint and, from photogrammetry, its ground
height, the lowest and highest roof points (``altitude_minimale_sol``, ``altitude_minimale_toit``,
``altitude_maximale_toit``, NGF-IGN69 like RGE ALTI), the number of floors, the construction date where known and
the use. The eaves are the lowest roof point; a roof more than 1 m high becomes pitched (gabled on small and long
footprints, hipped on large ones), so Haussmann blocks keep their mansard volume roughly and houses their gables.
Fetched from the Geoplateforme WFS (``BDTOPO_V3:batiment``, GeoJSON in EPSG:4326) in pages of 1000, cached in
<build>/bdtopo/.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from ..export.realbuildings import Source, cache_json
from . import buildings_extrude as ex

WFS = "https://data.geopf.fr/wfs/ows"
LAYER = "BDTOPO_V3:batiment"
COVERAGE = (41.30, -5.20, 51.15, 9.60)   # metropolitan France
UA = "twinmodel (carla-digitaltwins)"
USES = {"Résidentiel": "residential", "Commercial et services": "retail", "Industriel": "industry",
        "Religieux": "assembly", "Agricole": "storage", "Sportif": "assembly", "Annexe": "storage"}
PAGE = 1000


def _get(params, tries=4):
    import requests
    for k in range(tries):
        try:
            r = requests.get(WFS, params=params, timeout=180, headers={"User-Agent": UA})
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(3.0 * (k + 1))


def fetch(bbox):
    s, w, n, e = bbox
    out, start = [], 0
    while True:
        page = _get({"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": LAYER,
                     "outputFormat": "application/json", "SRSNAME": "EPSG:4326",
                     "BBOX": f"{s},{w},{n},{e},urn:ogc:def:crs:EPSG::4326", "COUNT": PAGE, "STARTINDEX": start})
        feats = page.get("features", [])
        for f in feats:
            g, p = f.get("geometry"), f.get("properties") or {}
            if not g:
                continue
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            keep = {k: p.get(k) for k in ("hauteur", "nombre_d_etages", "date_d_apparition", "usage_1",
                                          "altitude_minimale_sol", "altitude_minimale_toit",
                                          "altitude_maximale_toit", "nature")}
            out.append({"id": f.get("id") or p.get("cleabs"), "props": keep,
                        "polys": [[[[round(c[0], 7), round(c[1], 7)] for c in ring] for ring in poly]
                                  for poly in polys]})
        print("bdtopo: %d buildings so far" % len(out), flush=True)
        if len(feats) < PAGE:
            break
        start += PAGE
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def part_kwargs(props: dict, area_m2: float) -> dict:
    """heights() arguments and a roof shape from BD TOPO's attributes."""
    sol, tmin, tmax = (_num(props.get(k)) for k in ("altitude_minimale_sol", "altitude_minimale_toit",
                                                    "altitude_maximale_toit"))
    h = _num(props.get("hauteur"))
    floors = _num(props.get("nombre_d_etages"))
    kw = {"levels": floors}
    if sol is not None and tmax is not None and tmax > sol + 1.0:
        kw["total_m"] = tmax - sol
        if tmin is not None and tmax - tmin > 1.0 and tmin > sol:
            kw["roof_m"] = tmax - tmin
            kw["shape"] = "hipped" if area_m2 > 600.0 else "gabled"
        else:
            kw["shape"] = "flat"
    elif h is not None and h > 0:
        kw["total_m"] = h
    return kw


def year_of(props: dict):
    m = re.match(r"(\d{4})", str(props.get("date_d_apparition") or ""))
    y = int(m.group(1)) if m else None
    return y if y and 1000 < y < 2100 else None


def load(build: Path, man: dict, looks: bool = True, region: str = "south", **_):
    bbox = tuple(man["bbox_wgs84"])
    raw = cache_json(build / "bdtopo" / ("batiment_%.5f_%.5f_%.5f_%.5f.json.gz" % bbox), lambda: fetch(bbox))
    tr = ex.model_transformer(man)
    ground = ex.ground_for(build, man)
    out = []
    for f in raw:
        p = f["props"]
        parts = []
        for rings in f["polys"]:
            poly = ex.to_model(tr, rings)
            if poly is not None:
                parts.append((poly, part_kwargs(p, poly.area)))
        if not parts:
            continue
        use = USES.get(p.get("usage_1") or "")
        rb, _ = ex.make_building("FR." + str(f["id"]), parts, ground, region=region, year=year_of(p),
                                 uses={use: 1.0} if use else {}, levels=int(_num(p.get("nombre_d_etages")) or 0))
        if rb is not None:
            out.append(rb)
    print(f"bdtopo: {len(out)} buildings ({sum(1 for b in out if b.year)} with a construction year)")
    return out, Source(name="bdtopo", label="IGN BD TOPO batiment (Geoplateforme WFS)",
                       asset_source="IGN BD TOPO (Licence Ouverte 2.0)", roof_photo_max_m=40.0)
