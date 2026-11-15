"""Real 3D building providers, chosen by where the twin is (twinmodel.export.realbuildings does the rest).

Each provider is a module with ``COVERAGE`` (s, w, n, e; or a ``covers(bbox)`` function when its data come in
pieces) and ``load(build, man, looks, **opts) -> (list[RealBuilding], Source)``. ``choose`` picks the first
provider that covers the twin's bbox; a new country or city is one more module and one line in ``PROVIDERS``.

| name | where | data |
| --- | --- | --- |
| ``3dbag`` | the Netherlands | 3DBAG LoD 2.2 (CC BY 4.0) + BAG uses (CC0) |
| ``nyc3d`` | New York City | NYC 3D Building Model LoD 2 + Building Footprints + PLUTO (NYC Open Data) |
| ``plateau`` | Japanese cities in ``buildings_plateau.DATASETS`` (Shibuya) | PLATEAU CityGML LoD 2 / LoD 1 (MLIT) |
| ``catastro`` | Spain (not the Basque Country, Navarre) | Catastro INSPIRE buildings: floors per part, year, use (CC BY 4.0) |
| ``bdtopo`` | France | IGN BD TOPO: ground, eaves and ridge heights, floors, year, use (Licence Ouverte 2.0) |
| ``overture`` | everywhere | Overture Maps: height, floors, roof shape, class (ODbL / CDLA Permissive 2.0; needs duckdb) |
| ``osm`` | everywhere, offline | the twin's own OSM footprints with height, levels and roof:shape (ODbL) |

The last four have footprints and heights, not 3D models: twinmodel.ingest.buildings_extrude makes prisms with
roof shapes. ``candidates`` lists every provider covering a bbox in order, so a caller can fall back to the next
when one fails (no network, no duckdb, a service down).
"""
from __future__ import annotations

import importlib
from typing import Optional

# name -> module (in this package); the order is the order ``choose`` tries them
PROVIDERS = {
    "3dbag": "buildings_3dbag",
    "nyc3d": "buildings_nyc",
    "plateau": "buildings_plateau",
    "catastro": "buildings_catastro",
    "bdtopo": "buildings_bdtopo",
    "overture": "buildings_overture",
    "osm": "buildings_osm",
}
GLOBAL = ("overture", "osm")   # cover everywhere: never chosen over a national model


def module(name: str):
    try:
        return importlib.import_module("." + PROVIDERS[name], __package__)
    except KeyError:
        raise KeyError(f"unknown building provider {name!r}; known: {sorted(PROVIDERS)}") from None


def covers(name: str, bbox_swne) -> bool:
    mod = module(name)
    s, w, n, e = bbox_swne
    cs, cw, cn, ce = mod.COVERAGE
    inside = cs <= s and n <= cn and cw <= w and e <= ce
    return inside and (mod.covers(bbox_swne) if hasattr(mod, "covers") else True)


def candidates(bbox_swne, order=None) -> list:
    """Every provider covering the bbox, in ``order`` (default PROVIDERS' order, the global ones last)."""
    names = list(order) if order else list(PROVIDERS)
    names = [n for n in names if n in PROVIDERS]
    return [p for p in names if covers(p, bbox_swne)]


def choose(bbox_swne, include_global: bool = False) -> Optional[str]:
    """The first provider whose coverage holds the bbox, or None (the twin keeps its OSM buildings); the global
    providers (Overture, OSM) only with ``include_global``."""
    return next((p for p in candidates(bbox_swne) if include_global or p not in GLOBAL), None)


def load(name: str, build, man: dict, looks: bool = True, **opts):
    return module(name).load(build, man, looks=looks, **opts)
