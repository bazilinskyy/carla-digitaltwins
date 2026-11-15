"""Buildings anywhere for twinmodel.export.realbuildings: Overture Maps (ODbL / CDLA Permissive 2.0), keyless.

Overture's buildings theme merges OpenStreetMap with Esri Community Maps, Microsoft's and Google's footprints and
national agencies' data, with ``height`` (to the top), ``num_floors``, ``roof_shape``, ``roof_height`` and
``roof_orientation`` where a source has them, and a ``class``/``subtype`` (house, apartments, commercial, ...). The
GeoParquet release on S3 is read with DuckDB (``pip install duckdb``; no key, no account): only the row groups
whose bbox meets the twin's are downloaded (about a minute, most of it reading the files' metadata). The newest
release is taken from Overture's STAC catalogue. Footprints become prisms with their roof shapes
(twinmodel.ingest.buildings_extrude); small homes without a roof shape get a pitched roof guessed by region.
Cached in <build>/overture/.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..export.realbuildings import Source, cache_json
from . import buildings_extrude as ex

COVERAGE = (-90.0, -180.0, 90.0, 180.0)
STAC = "https://stac.overturemaps.org/catalog.json"
RELEASE = "2026-10-21.0"     # used when the catalogue does not answer
PARQUET = "s3://overturemaps-us-west-2/release/{rel}/theme=buildings/type=building/*"
SKIP_CLASSES = {"roof", "carport", "canopy", "transportation"}   # roofs on posts, not buildings with walls
USES = {   # class / subtype -> facades.classify use
    "house": "residential", "detached": "residential", "semidetached_house": "residential", "terrace": "residential",
    "apartments": "residential", "residential": "residential", "dormitory": "residential", "bungalow": "residential",
    "commercial": "retail", "retail": "retail", "supermarket": "retail", "kiosk": "retail", "office": "office",
    "industrial": "industry", "warehouse": "storage", "factory": "industry", "manufacture": "industry",
    "school": "education", "university": "education", "college": "education", "kindergarten": "education",
    "education": "education", "hospital": "health", "medical": "health", "hotel": "hotel",
    "civic": "government", "government": "government", "public": "government", "church": "assembly",
    "religious": "assembly", "mosque": "assembly", "temple": "assembly", "entertainment": "assembly",
    "agricultural": "storage", "outbuilding": "storage", "service": "utility", "garage": "storage",
    "garages": "storage", "shed": "storage",
}
COLUMNS = ("id", "height", "num_floors", "roof_shape", "roof_height", "roof_orientation", "class", "subtype",
           "is_underground", "dataset")


def latest_release(timeout: float = 15.0) -> str:
    try:
        import requests
        r = requests.get(STAC, timeout=timeout, headers={"User-Agent": "twinmodel (carla-digitaltwins)"})
        r.raise_for_status()
        return str(r.json().get("latest") or RELEASE)
    except Exception:  # noqa: BLE001 - offline catalogue: the release this code was written against
        return RELEASE


def query(bbox, release: str) -> str:
    s, w, n, e = bbox
    return (f"SELECT id, height, num_floors, roof_shape, roof_height, roof_orientation, class, subtype, "
            f"is_underground, sources[1].dataset AS dataset, ST_AsGeoJSON(geometry) AS g "
            f"FROM read_parquet('{PARQUET.format(rel=release)}', hive_partitioning=1) "
            f"WHERE bbox.xmin < {e} AND bbox.xmax > {w} AND bbox.ymin < {n} AND bbox.ymax > {s}")


def fetch(bbox, release: str = None):
    import duckdb   # ImportError -> the caller falls back to the next provider
    release = release or latest_release()
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
    rows = con.execute(query(bbox, release)).fetchall()
    out = []
    for r in rows:
        rec = dict(zip(COLUMNS, r[:-1]))
        geom = json.loads(r[-1])
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        rec["polys"] = [[[[round(c[0], 7), round(c[1], 7)] for c in ring] for ring in poly] for poly in polys]
        out.append(rec)
    print(f"overture {release}: {len(out)} buildings ({sum(1 for b in out if b['height'])} with a height, "
          f"{sum(1 for b in out if b['roof_shape'])} with a roof shape)", flush=True)
    return {"release": release, "buildings": out}


def part_kwargs(rec: dict) -> dict:
    return {"total_m": rec.get("height"), "roof_m": rec.get("roof_height"), "levels": rec.get("num_floors"),
            "shape": rec.get("roof_shape"), "orientation": rec.get("roof_orientation"),
            "category": rec.get("class") or rec.get("subtype")}


def load(build: Path, man: dict, looks: bool = True, region: str = "eu", **_):
    bbox = tuple(man["bbox_wgs84"])
    raw = cache_json(build / "overture" / ("buildings_%.5f_%.5f_%.5f_%.5f.json.gz" % bbox), lambda: fetch(bbox))
    tr = ex.model_transformer(man)
    ground = ex.ground_for(build, man)
    out, n_guessed, n_height = [], 0, 0
    for rec in raw["buildings"]:
        cls = (rec.get("class") or "").lower()
        if rec.get("is_underground") or cls in SKIP_CLASSES:
            continue
        parts = [(p, part_kwargs(rec)) for p in (ex.to_model(tr, rings) for rings in rec["polys"]) if p is not None]
        if not parts:
            continue
        use = USES.get(cls) or USES.get((rec.get("subtype") or "").lower())
        rb, guessed = ex.make_building("OV." + str(rec["id"]), parts, ground, region=region,
                                       uses={use: 1.0} if use else {}, levels=int(rec.get("num_floors") or 0),
                                       category=cls or None)
        if rb is None:
            continue
        out.append(rb)
        n_guessed += int(guessed)
        n_height += int(bool(rec.get("height") or rec.get("num_floors")))
    print(f"overture: {len(out)} buildings, {n_height} with a measured height or floors, "
          f"{n_guessed} with a guessed height or roof")
    return out, Source(name="overture", label=f"Overture Maps buildings {raw.get('release', '')}",
                       asset_source="Overture Maps buildings (ODbL / CDLA Permissive 2.0)", roof_photo_max_m=40.0,
                       extra_stats={"buildings_measured_height": n_height, "buildings_guessed": n_guessed})
