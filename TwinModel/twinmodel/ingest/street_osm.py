"""What OpenStreetMap says about the street side of a twin's buildings, for the facade styles.

``fetch(bbox_swne, cache_dir)`` asks Overpass (with the mirrors of twinmodel.ingest.osm) for

* shops and the cafes, restaurants, bars, banks, ... a street front has (``shop=*``, ``amenity=`` in
  SHOP_AMENITIES, ``craft=*``, nodes only: OSM maps them as points at or just inside the facade), and
* the buildings that carry a wall material or colour (``building:material``, ``building:colour``,
  ``building:facade:material``, ``building:facade:colour``) or say they are old (``heritage``, ``historic``,
  ``start_date``, ``building:architecture``), as centre points, and
* the historic areas an old-town facade style goes by (``heritage=*`` areas that are not buildings,
  ``historic=district``, cultural protected areas ``boundary=protected_area`` + ``protect_class=22``), as polygons,

caches the answer in ``cache_dir/street_<bbox>.json`` and returns ``{"pois": [...], "buildings": [...]}`` with
lat/lon and the tags that matter, or None when Overpass cannot be reached (the styles then come from the
building registers alone, as before). ``to_model`` puts them into the twin's model frame for
``realbuildings.apply(street=...)``. OSM data: (c) OpenStreetMap contributors, ODbL 1.0.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SHOP_AMENITIES = ("cafe", "restaurant", "bar", "pub", "fast_food", "ice_cream", "bank", "pharmacy",
                  "bureau_de_change", "post_office", "biergarten", "food_court")
POI_KEYS = ("shop", "amenity", "craft", "level", "name")
BUILDING_KEYS = ("building", "building:material", "building:colour", "building:facade:material",
                 "building:facade:colour", "heritage", "historic", "start_date", "building:architecture")
VERSION = 2   # 2: + the old-building tags and the historic areas (2026-11-25); older caches are fetched again
ROUNDS, ROUND_WAIT_S = 3, 60.0   # all mirrors busy: two more rounds a minute apart (2026-11-25: 502s all afternoon)


def query(bbox_swne) -> str:
    s, w, n, e = bbox_swne
    bb = f"{s},{w},{n},{e}"
    am = "|".join(SHOP_AMENITIES)
    return f"""[out:json][timeout:120];
(
  node["shop"]({bb});
  node["amenity"~"^({am})$"]({bb});
  node["craft"]({bb});
  way["building"]["building:material"]({bb});
  way["building"]["building:colour"]({bb});
  way["building"]["building:facade:material"]({bb});
  way["building"]["building:facade:colour"]({bb});
  relation["building"]["building:material"]({bb});
  relation["building"]["building:colour"]({bb});
  way["building"]["heritage"]({bb});
  way["building"]["historic"]({bb});
  way["building"]["start_date"]({bb});
  way["building"]["building:architecture"]({bb});
  relation["building"]["heritage"]({bb});
);
out tags center;
"""


def area_query(bbox_swne) -> str:
    """The historic areas the twin lies in or touches, whole (an old town is larger than the twin), as geometry:
    the areas around a 3 x 3 grid of points over the bbox (``is_in``), and the areas with a corner inside it."""
    s, w, n, e = bbox_swne
    bb = f"{s},{w},{n},{e}"
    pts = [(s + (n - s) * a, w + (e - w) * b) for a in (0.1, 0.5, 0.9) for b in (0.1, 0.5, 0.9)]
    isin = "".join(f"is_in({la:.6f},{lo:.6f});" for la, lo in pts)
    sel = '["heritage"][!"building"]', '["historic"="district"]', '["boundary"="protected_area"]["protect_class"="22"]'
    around = "".join(f"area.a{t};" for t in sel)
    inside = "".join(f"way{t}({bb});relation{t}({bb});" for t in sel)
    return f"""[out:json][timeout:120];
({isin})->.a;
({around})->.h;
(way(pivot.h);relation(pivot.h);{inside});
out geom;
"""


def parse_areas(data: dict) -> list:
    """Overpass ``out geom`` JSON -> [{"rings": [[[lat, lon], ...], ...], "tags": {...}}] (outer rings only)."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge, polygonize

    out = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        if el.get("type") == "way":
            g = el.get("geometry") or []
            if len(g) >= 4 and (g[0]["lat"], g[0]["lon"]) == (g[-1]["lat"], g[-1]["lon"]):
                out.append({"rings": [[[p["lat"], p["lon"]] for p in g]], "tags": tags})
        elif el.get("type") == "relation":
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in el.get("members", [])
                     if m.get("type") == "way" and m.get("role", "outer") in ("outer", "") and
                     len(m.get("geometry") or []) >= 2]
            rings = [[[lat, lon] for lon, lat in poly.exterior.coords] for poly in polygonize(linemerge(lines))] \
                if lines else []
            if rings:
                out.append({"rings": rings, "tags": tags})
    keep = ("heritage", "historic", "boundary", "protect_class", "name")
    return [{"rings": a["rings"], "tags": {k: a["tags"][k] for k in keep if k in a["tags"]}} for a in out]


def parse(data: dict) -> dict:
    """Overpass JSON -> {"pois": [{lat, lon, tags}], "buildings": [{lat, lon, tags}]}."""
    pois, blds = [], []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        if "building" in tags and el.get("type") != "node":
            blds.append({"lat": lat, "lon": lon, "tags": {k: tags[k] for k in BUILDING_KEYS if k in tags}})
        elif tags.get("shop") or tags.get("craft") or tags.get("amenity") in SHOP_AMENITIES:
            pois.append({"lat": lat, "lon": lon, "tags": {k: tags[k] for k in POI_KEYS if k in tags}})
    return {"pois": pois, "buildings": blds}


def fetch(bbox_swne, cache_dir, force: bool = False, timeout: float = 150.0) -> Optional[dict]:
    """The street-side OSM data of the bbox (cached), or None when no Overpass server answers."""
    import requests

    from .osm import OVERPASS_MIRRORS

    cache_dir = Path(cache_dir)
    key = "_".join(f"{float(v):.5f}" for v in bbox_swne)
    path = cache_dir / f"street_{key}.json"
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if cached.get("version", 1) >= VERSION:
            return cached

    def ask(q, parse_fn):
        import time
        for rnd in range(ROUNDS):
            for url in OVERPASS_MIRRORS:
                try:
                    r = requests.post(url, data={"data": q}, timeout=timeout,
                                      headers={"User-Agent": "twinmodel (carla-digitaltwins)"})
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    return parse_fn(r.json())
                except Exception as exc:  # noqa: BLE001 - busy or unreachable: the next mirror
                    log.warning("street osm: %s failed (%s)", url, exc)
            if rnd + 1 < ROUNDS:
                time.sleep(ROUND_WAIT_S)     # every mirror busy (Overpass at capacity): again in a while
        return None
    out = ask(query(bbox_swne), parse)
    if out is None:
        return None
    areas = ask(area_query(bbox_swne), parse_areas)
    out["areas"] = areas or []
    out["version"] = VERSION if areas is not None else 1    # no answer for the areas: asked again next time
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    return out


def ground_floor(tags: dict) -> bool:
    """Whether a POI is at street level (no ``level`` tag, or level 0 / -1..0 / 0;1)."""
    lv = str(tags.get("level", "0")).replace(",", ";").split(";")
    try:
        return min(float(x) for x in lv if x.strip()) <= 0.0
    except ValueError:
        return True


def to_model(data: Optional[dict], proj: str) -> Optional[dict]:
    """Model-frame points for realbuildings.apply: {"pois": [(x, y, tags)], "buildings": [(x, y, tags)]};
    upper-floor POIs (``level`` > 0) are left out."""
    if data is None:
        return None
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    def pts(items, keep=lambda t: True):
        out = []
        for it in items:
            if keep(it["tags"]):
                x, y = tr.transform(it["lon"], it["lat"])
                out.append((float(x), float(y), it["tags"]))
        return out
    areas = []
    for a in data.get("areas", ()):
        rings = []
        for ring in a.get("rings", ()):
            xs, ys = tr.transform([p[1] for p in ring], [p[0] for p in ring])
            rings.append([(float(x), float(y)) for x, y in zip(xs, ys)])
        areas.append((rings, a.get("tags", {})))
    return {"pois": pts(data.get("pois", []), ground_floor), "buildings": pts(data.get("buildings", [])),
            "areas": areas}
