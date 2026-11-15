"""Left-hand traffic twins by mirroring: the lane graph is built for right-hand traffic in a mirror image of the
place, and the model is mirrored back.

The lane graph (twinmodel.lanegraph) places forward lanes right of the reference line, medians on the left, turn
lanes, junction connections and signals the right-hand way. A mirror image of a left-hand city is a right-hand
city: so for a profile with ``drives_on="left"`` the OSM data is mirrored about the frame's central meridian
(``mirror_osm``: the local transverse Mercator frame is symmetric about it, so x -> -x exactly) with every
left/right in its tags swapped (sidewalk:left, turn:lanes, restriction=no_left_turn, ... and per-lane lists
reversed), the lane graph is built there with a right-hand copy of the profile, and the model is mirrored back
(``mirror_model``): geometry x -> -x, lane ids negated (lane -1 stays right of the reference line, so the forward
lanes become the positive, left ones), lane links and signal validities negated, signal t and headings mirrored,
and the left/right in tags swapped back. The OpenDRIVE then says ``rule="LHT"`` on every road, which CARLA 1.0
reads (RoadParser: is_rht; positive lanes run along the reference line).
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

import shapely
from shapely.geometry.base import BaseGeometry

_LR = re.compile(r"(?<![A-Za-z])(left|right|Left|Right)(?![A-Za-z])")
_SWAP = {"left": "right", "right": "left", "Left": "Right", "Right": "Left"}
# tags whose values name a side or a turn (names, addresses, refs are never touched: "Bright Street" stays)
DIRECTIONAL = {"turn", "sidewalk", "cycleway", "parking", "restriction", "side", "placement", "change", "busway",
               "footway", "shoulder", "lanes", "traffic_sign", "kerb", "crossing", "destination", "overtaking"}


def swap_lr(text: str) -> str:
    """'sidewalk:left' -> 'sidewalk:right', 'slight_right;through' -> 'slight_left;through' (whole words only)."""
    return _LR.sub(lambda m: _SWAP[m.group(0)], text)


def mirror_tag(key: str, value):
    """(key, value) of an OSM tag in the mirror image: left/right swapped in the key and, for side and turn tags,
    in the value; per-lane lists (``*:lanes*``, values separated by '|') listed from the other side."""
    k = swap_lr(key)
    if not isinstance(value, str):
        return k, value
    v = value
    parts = key.split(":")
    if ":lanes" in key or parts[-1] == "lanes" and "|" in v:
        if "|" in v:
            v = "|".join(reversed(v.split("|")))
    if parts[0] in DIRECTIONAL or any(p in DIRECTIONAL for p in parts[1:]):
        v = swap_lr(v)
    return k, v


def mirror_tags(tags: dict) -> dict:
    out = {}
    for key, value in (tags or {}).items():
        k, v = mirror_tag(key, value)
        out[k] = v
    return out


def mirror_osm(osm, lon0: float):
    """Mirror parsed OSM data (twinmodel.ingest.osm.OsmData) in place about the meridian lon0 and swap left/right
    in all tags. Returns it."""
    for n in osm.nodes.values():
        n.lon = 2.0 * lon0 - n.lon
        n.tags = mirror_tags(n.tags)
    for w in osm.ways:
        w.tags = mirror_tags(w.tags)
    for r in osm.relations:
        r.tags = mirror_tags(r.tags)
    return osm


# --------------------------------------------------------------------------- the model back

def _neg_x(coords):
    out = coords.copy()
    out[:, 0] = -out[:, 0]
    return out


def mirror_geom(g):
    if g is None or not isinstance(g, BaseGeometry) or g.is_empty:
        return g
    has_z = g.has_z
    return shapely.transform(g, _neg_x, include_z=has_z)


def _mirror_value(key: str, v: Any):
    if isinstance(v, BaseGeometry):
        return mirror_geom(v)
    if isinstance(v, str) and key.endswith("_wkt"):
        try:
            return shapely.to_wkt(mirror_geom(shapely.from_wkt(v)))
        except Exception:  # noqa: BLE001 - not WKT after all
            return v
    if key in ("centre", "center", "xy") and isinstance(v, (list, tuple)) and len(v) >= 2 and \
            all(isinstance(c, (int, float)) for c in v[:2]):
        return type(v)([-v[0], *v[1:]])
    if isinstance(v, str) and _LR.search(v) and key in ("turn", "median_side", "side", "sidewalk", "cycleway",
                                                       "parking", "placement"):
        return swap_lr(v)
    return v


def _mirror_model_tags(tags: dict) -> dict:
    out = {}
    for k, v in (tags or {}).items():
        nk = swap_lr(k) if _LR.search(k) else k
        out[nk] = _mirror_value(k, v)
    return out


def mirror_heading(h: float) -> float:
    """A heading (radians from +x, counter-clockwise) in the mirror image x -> -x."""
    m = math.pi - h
    return math.atan2(math.sin(m), math.cos(m))


def mirror_model(model) -> dict:
    """Mirror a twin model in place (x -> -x) and turn its right-hand lane graph into a left-hand one. Returns
    counts of what was changed."""
    n = {"roads": 0, "lanes": 0, "lane_links": 0, "signals": 0, "geometries": 0}

    def geom_fields(obj):
        for f in dataclasses.fields(obj):
            if f.name.startswith("_"):
                continue
            v = getattr(obj, f.name)
            if isinstance(v, BaseGeometry):
                setattr(obj, f.name, mirror_geom(v))
                n["geometries"] += 1
            elif f.name == "tags" and isinstance(v, dict):
                setattr(obj, f.name, _mirror_model_tags(v))

    for r in model.roads:
        geom_fields(r)
        n["roads"] += 1
        for lane in r.lanes:
            lane.id = -lane.id
            lane.tags = _mirror_model_tags(lane.tags)
            if lane.marking is not None and lane.marking.geometry is not None:
                lane.marking.geometry = mirror_geom(lane.marking.geometry)
            n["lanes"] += 1
        if r.center_marking is not None and r.center_marking.geometry is not None:
            r.center_marking.geometry = mirror_geom(r.center_marking.geometry)
    for j in model.junctions:
        geom_fields(j)
        for c in j.connections:
            for ll in c.lane_links:
                ll.from_lane, ll.to_lane = -ll.from_lane, -ll.to_lane
                n["lane_links"] += 1
    for s in model.signals:
        geom_fields(s)
        s.t = -s.t
        s.heading = mirror_heading(s.heading)
        s.h_offset = -s.h_offset
        s.validities = [tuple(sorted((-a, -b))) for a, b in s.validities]
        n["signals"] += 1
    for coll in (model.controllers, model.surfaces, model.curbs, model.markings, model.buildings, model.objects):
        for obj in coll:
            if dataclasses.is_dataclass(obj):
                geom_fields(obj)
    if hasattr(model, "_datum"):
        model._datum = None
        model._datum_key = None
    model.metadata["lefthand"] = {"method": "mirror", **n}
    return n
