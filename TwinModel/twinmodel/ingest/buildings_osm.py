"""OpenStreetMap buildings for twinmodel.export.realbuildings: the twin's own OSM footprints, with roof shapes.

The last resort, offline: the footprints the twin was built from (``<name>.twin/buildings.geojson``, OSM, ODbL),
with OSM's ``height`` / ``building:levels`` and ``roof:shape``, become prisms with roofs
(twinmodel.ingest.buildings_extrude) and so get facade styles, party walls and lit windows like the real city
models. Without a height or levels the region's default number of floors is used, as the bake export does.
"""
from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import shape

from ..export.realbuildings import Source
from . import buildings_extrude as ex
from .buildings_overture import USES

COVERAGE = (-90.0, -180.0, 90.0, 180.0)
SKIP = {"roof", "carport", "canopy", "construction", "ruins", "no"}


def twin_buildings(build: Path, man: dict) -> Path:
    p = build / f"{man['name']}.twin" / "buildings.geojson"
    if not p.exists():
        hits = sorted(build.glob("*.twin/buildings.geojson"))
        if not hits:
            raise FileNotFoundError(f"no <name>.twin/buildings.geojson in {build}")
        p = hits[0]
    return p


def load(build: Path, man: dict, looks: bool = True, region: str = "eu", **_):
    data = json.loads(twin_buildings(build, man).read_text())
    ground = ex.ground_for(build, man)
    out, n_guessed = [], 0
    for f in data.get("features", []):
        pr = f.get("properties") or {}
        tags = pr.get("tags") or {}
        cat = str(tags.get("building") or "yes").lower()
        if cat in SKIP:
            continue
        geom = shape(f["geometry"])
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        kw = {"total_m": pr.get("height"), "levels": pr.get("levels"), "shape": tags.get("roof:shape"),
              "category": cat}
        parts = [(p, dict(kw)) for p in polys if p.area >= 2.0]
        if not parts:
            continue
        use = USES.get(cat) or ("retail" if tags.get("shop") else None)
        rb, guessed = ex.make_building("OSM." + str(pr.get("osm_id") or pr.get("id")), parts, ground, region=region,
                                       uses={use: 1.0} if use else {}, levels=int(pr.get("levels") or 0),
                                       category=cat)
        if rb is not None:
            out.append(rb)
            n_guessed += int(guessed)
    print(f"osm: {len(out)} buildings, {n_guessed} with a guessed height or roof")
    return out, Source(name="osm", label="OpenStreetMap buildings (the twin's footprints)",
                       asset_source="OpenStreetMap contributors (ODbL 1.0)", roof_photo_max_m=40.0,
                       extra_stats={"buildings_guessed": n_guessed})
