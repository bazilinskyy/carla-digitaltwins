"""Keep the roads and sidewalks of a bake export clear of buildings.

Open building models are solids extruded down to the ground. Where a road really runs under a building
(a bus terminal, a station hall, a viaduct) or where a footprint overlaps the carriageway (a registration or
data error), the baked walls stand on the driving lanes and the traffic drives through them. ``apply`` cuts
the building meshes of a bake export (``ue/manifest.json`` + ``ue/meshes/*.glb``, any building source: OSM
slabs, 3DBAG, PLATEAU with photo textures, Catastro, Overture/OSM prisms, NYC) so that

* inside the road corridor (the export's drivable, parking and crossing meshes, plus the OpenDRIVE driving,
  parking and biking lanes when the xodr can be read) no building geometry is left between just below the road
  and ``ROAD_CLEARANCE_M`` above it (or the street's OSM ``maxheight`` plus a margin);
* along the sidewalks (the OpenDRIVE sidewalk lanes, ``WALK_HALF_M`` either side of their centre line; the
  sidewalk meshes when the xodr cannot be read) the same up to ``WALK_CLEARANCE_M``.

How a cut looks depends on the building:

* passage: the walls and roofs below the clearance are removed and the cut is closed like a real passage, with
  a ceiling (soffit) at the clearance height and walls where the corridor runs through the footprint, in the
  building's own wall material without windows (``plain_<style>_<variant>`` textures; photo-textured PLATEAU
  buildings get the plain wall of their facade style). The texture coordinates of the cut faces are
  interpolated, so photo atlases and facade bays stay where they were;
* trim: a footprint that only grazes the carriageway (narrower than ``SLIVER_M`` inside it, not split by it) is
  a data error, not a passage: it is cut back to the kerb line over its whole height (no overhang), and so is a
  building too low to pass under (eaves below the clearance);
* nothing: an overlap thinner than ``GRAZE_M`` (a facade on the kerb line) is left alone.

The cut works per (OSM layer of the road mesh) in grid cells of ``CELL_M``; each cell removes the slab
[road z min - ``FLOOR_BELOW_M``, road z max + clearance], so a building under an elevated road keeps its
structure below the deck. The uncut meshes are kept in ``ue/meshes/uncut/`` and the manifest's ``clearance``
record lists what was changed, so a second run starts from the uncut meshes again (the result does not depend
on how often the step ran). ``check`` measures what is left: CARLA's 1 m driving-lane and sidewalk-lane samples
against the building triangles, the same test as the stand-alone checker used before the cut existed.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree

ROAD_CLEARANCE_M = 4.6     # above the road: buses (3.0-4.0 m), lorries (4.0 m) and a margin
WALK_CLEARANCE_M = 2.5     # above a sidewalk lane: pedestrians and cyclists
MAXHEIGHT_MARGIN_M = 0.2   # a street with OSM maxheight gets this much above it ...
MAXHEIGHT_MIN_M = 3.0      # ... but never less than this (cars and vans routed there anyway)
FLOOR_BELOW_M = 0.3        # the cut starts this far below the lowest road point of a cell
CELL_M = 8.0               # grid of the cut: each cell has its own road height
ROAD_BUFFER_M = 0.1        # the road meshes, slightly buffered
WALK_HALF_M = 0.6          # sidewalk corridor: this far either side of the sidewalk lane's centre line
GRAZE_M = 0.3              # a footprint inside a corridor by less than this (width) is left alone
SLIVER_M = 1.8             # a footprint inside the carriageway by less than this, not split by it: trimmed
LOW_EAVES_M = 0.5          # a building with eaves below clearance + this is removed in the corridor
FOOT_BUFFER_M = 0.6        # walls this close to a footprint belong to it (city-model walls stray from footprints)
STEP_M = 0.1               # the clearance heights are rounded up to this
THIN_TOP_M = 0.5           # a building surface less than this above the clearance is cut away too
SOFFIT_GRID_M = 1.0        # the soffit is kept where the building's roof is above it, tested on this grid
WALL_PIECE_M = 0.5         # cut walls are built in pieces this long (their top follows the roof, their neighbour
                           # cut can change along an edge)
ROAD_KINDS = ("drivable", "parking", "crossing")
WALK_KINDS = ("sidewalk",)
BUILDING_KINDS = ("building", "building_3dbag", "roof")
# CARLA lane types: the road corridor and what the check counts as a road sample
ROAD_LANES = ("Driving", "Parking", "Bidirectional", "Restricted")
WALK_LANES = ("Sidewalk", "Biking")   # cut to WALK_CLEARANCE_M (sidewalks in a band, bike lanes whole)
CHECK_BANDS = {"road": (0.5, 4.0), "walk": (0.3, 2.2)}   # m above the lane: a building surface in it blocks
CHECK_LANES = {"road": ("Driving",), "walk": ("Sidewalk",)}
EPS = 1e-6
INF = float("inf")


# --------------------------------------------------------------------------- lanes

@dataclass
class Lanes:
    """1 m lane samples (model frame): x, y, z, heading (rad), width, plus ids and the CARLA lane type."""
    xyz: np.ndarray                 # (n, 3)
    heading: np.ndarray             # (n,)
    width: np.ndarray               # (n,)
    road: np.ndarray                # (n,) int
    lane: np.ndarray                # (n,) int: one id per (road, section, lane)
    s: np.ndarray                   # (n,)
    kind: np.ndarray                # (n,) str: CARLA lane type name

    def select(self, kinds) -> "Lanes":
        m = np.isin(self.kind, list(kinds))
        return Lanes(self.xyz[m], self.heading[m], self.width[m], self.road[m], self.lane[m], self.s[m],
                     self.kind[m])

    def segments(self, max_gap: float = 2.0):
        """(i, j) index pairs of consecutive samples of one lane piece (along s, at most ``max_gap`` apart)."""
        order = np.lexsort((self.s, self.lane))
        a, b = order[:-1], order[1:]
        same = self.lane[a] == self.lane[b]
        d = np.linalg.norm(self.xyz[a, :2] - self.xyz[b, :2], axis=1)
        keep = same & (d <= max_gap) & (d > 1e-3)
        return a[keep], b[keep]

    @classmethod
    def empty(cls) -> "Lanes":
        z = np.zeros(0)
        return cls(np.zeros((0, 3)), z, z, z.astype(int), z.astype(int), z, np.zeros(0, dtype=object))

    def __len__(self):
        return len(self.xyz)


def xodr_lanes(xodr_path, step: float = 1.0) -> Optional[Lanes]:
    """Lane samples of an OpenDRIVE file read by CARLA's own parser (``carla.Map``, no server): the driving
    lanes CARLA generates every ``step`` m and, from each of them, the lanes beside it (sidewalks, parking,
    shoulders, bike lanes). None without the carla module."""
    try:
        import carla
    except Exception:  # noqa: BLE001 - no carla wheel in this interpreter
        return None
    cmap = carla.Map("clearance", Path(xodr_path).read_text())
    rows, seen = [], set()

    def add(w):
        key = (w.road_id, w.section_id, w.lane_id, round(w.s, 1))
        if key in seen:
            return
        seen.add(key)
        loc, rot = w.transform.location, w.transform.rotation
        rows.append((loc.x, -loc.y, loc.z, -math.radians(rot.yaw), w.lane_width, w.road_id,
                     (w.road_id * 1000 + w.section_id) * 100 + w.lane_id + 50, w.s, str(w.lane_type)))
    for w in cmap.generate_waypoints(step):
        add(w)
        for side in ("get_right_lane", "get_left_lane"):
            x = w
            for _ in range(8):
                x = getattr(x, side)()
                if x is None:
                    break
                if str(x.lane_type) != "Driving":
                    add(x)
    if not rows:
        return Lanes.empty()
    a = np.array([r[:8] for r in rows], dtype=np.float64)
    return Lanes(a[:, :3], a[:, 3], a[:, 4], a[:, 5].astype(int), a[:, 6].astype(int), a[:, 7],
                 np.array([r[8] for r in rows], dtype=object))


def lane_quads(lanes: Lanes, half: Optional[float] = None) -> list:
    """[(polygon, z max, z min)] one quad per pair of consecutive samples: the lane's full width, or ``half``
    either side of the centre line. Neighbouring quads share their edge (offsets along each sample's mean
    normal), so the ribbon has no notches on the outside of a bend."""
    out = []
    ia, ib = lanes.segments()
    if not len(ia):
        return out
    d = lanes.xyz[ib, :2] - lanes.xyz[ia, :2]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), EPS)
    nrm = np.zeros((len(lanes), 2))
    np.add.at(nrm, ia, np.column_stack([-d[:, 1], d[:, 0]]))
    np.add.at(nrm, ib, np.column_stack([-d[:, 1], d[:, 0]]))
    nl = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(nl > 1e-6, nrm / np.maximum(nl, EPS), 0.0)
    w = np.full(len(lanes), half) if half is not None else lanes.width / 2.0
    for i, j, dd in zip(ia, ib, d):
        if w[i] <= 0.05 or w[j] <= 0.05:
            continue
        ni = nrm[i] if nrm[i].any() else np.array([-dd[1], dd[0]])
        nj = nrm[j] if nrm[j].any() else np.array([-dd[1], dd[0]])
        p, q = lanes.xyz[i], lanes.xyz[j]
        poly = Polygon([p[:2] + ni * w[i], q[:2] + nj * w[j], q[:2] - nj * w[j], p[:2] - ni * w[i]])
        if not poly.is_valid:
            poly = shapely.make_valid(poly)
            poly = max(_polys(poly), key=lambda g: g.area, default=None)
        if poly is not None and poly.area > 1e-3:
            out.append((poly, max(p[2], q[2]), min(p[2], q[2])))
    # the ends of each lane piece: a disc, so the next section (sampled up to a step later) joins on
    deg = np.bincount(np.concatenate([ia, ib]), minlength=len(lanes))
    for i in np.nonzero(deg == 1)[0]:
        if w[i] > 0.05:
            out.append((Point(lanes.xyz[i, :2]).buffer(w[i], quad_segs=4), lanes.xyz[i, 2], lanes.xyz[i, 2]))
    return out


# --------------------------------------------------------------------------- meshes

def _gltf_to_model(g):
    g = np.asarray(g, dtype=np.float64).reshape(-1, 3)
    return np.column_stack([g[:, 0], -g[:, 2], g[:, 1]])


def load_asset(out: Path, a: dict) -> Optional[dict]:
    """A manifest asset's glb as model-frame arrays: positions, normals, uvs, faces, material, base colour."""
    from twinmodel.export.ue import read_glb
    p = out / a["file"]
    if not p.exists():
        return None
    g = read_glb(p)
    mat = g["gltf"]["materials"][0]
    col = (mat.get("pbrMetallicRoughness") or {}).get("baseColorFactor", [0.5, 0.5, 0.5, 1.0])[:3]
    return {"pos": _gltf_to_model(g["positions"]), "nrm": _gltf_to_model(g["normals"]),
            "uv": np.asarray(g["uvs"], dtype=np.float64), "faces": np.asarray(g["faces"], dtype=np.int64),
            "material": g["material"], "color": tuple(float(c) for c in col)}


class SurfaceZ:
    """Triangles of a set of horizontal-ish meshes (road, sidewalk) with a 2D index: which triangles touch a
    geometry, and their height range there."""

    def __init__(self, tris: np.ndarray):
        self.tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
        self.polys = shapely.polygons(self.tris[:, :, :2]) if len(self.tris) else np.zeros(0, dtype=object)
        if len(self.polys):
            ok = shapely.area(self.polys) > 1e-6
            self.tris, self.polys = self.tris[ok], self.polys[ok]
        self.tree = STRtree(self.polys) if len(self.polys) else None

    def __len__(self):
        return len(self.tris)

    def query(self, geom) -> np.ndarray:
        if self.tree is None:
            return np.zeros(0, dtype=int)
        return self.tree.query(geom, predicate="intersects")

    def zrange(self, geom) -> Optional[tuple]:
        idx = self.query(geom)
        if not len(idx):
            return None
        z = self.tris[idx][:, :, 2]
        return float(z.min()), float(z.max())


class TopZ:
    """Highest building surface above a point (vertical ray against the non-vertical building triangles)."""

    def __init__(self, tris: np.ndarray):
        tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
        if len(tris):
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            flat = np.abs(n[:, 2]) > 0.05 * np.linalg.norm(n, axis=1) + EPS
            tris = tris[flat]
        self.tris = tris
        self.tree = STRtree(shapely.polygons(tris[:, :, :2])) if len(tris) else None

    def __call__(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.full(len(xy), -INF)
        if self.tree is None or not len(xy):
            return out
        qi, ti = self.tree.query(shapely.points(xy[:, 0], xy[:, 1]), predicate="intersects")
        if not len(qi):
            return out
        t = self.tris[ti]
        z = _bary_z(t, xy[qi])
        np.maximum.at(out, qi, z)
        return out


def _bary_z(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """z of the plane of each triangle t (m, 3, 3) at xy (m, 2)."""
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    v0, v1, v2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], xy - a[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    return (1 - l1 - l2) * a[:, 2] + l1 * b[:, 2] + l2 * c[:, 2]


# --------------------------------------------------------------------------- buildings and corridor

@dataclass
class Bld:
    id: str
    poly: Polygon                   # footprint (model xy)
    base: float
    eaves: float
    roof: float
    style: Optional[str] = None
    real: bool = False


def manifest_buildings(man: dict) -> list:
    """The export's footprints (real or OSM) as ``Bld``: the raw ring when there is one (``rings_ue`` of a
    procedural-capable building is simplified)."""
    out = []
    for i, b in enumerate(man.get("buildings") or ()):
        rings = [b["raw_ring_ue"]] if b.get("raw_ring_ue") else (b.get("rings_ue") or [])
        polys = []
        for r in rings:
            if r and len(r) >= 3:
                p = Polygon([(x / 100.0, -y / 100.0) for x, y in r])
                if not p.is_valid:
                    p = p.buffer(0)
                polys.extend(_polys(p))
        polys = [p for p in polys if p.area > 1.0]
        if not polys:
            continue
        poly = max(polys, key=lambda p: p.area) if len(polys) == 1 else shapely.union_all(polys)
        base = float(b.get("base_z_cm", 0.0)) / 100.0
        roof = float(b.get("roof_z_cm", b.get("base_z_cm", 0.0))) / 100.0
        eaves = float(b.get("eaves_z_cm", b.get("roof_z_cm", 0.0))) / 100.0
        real = bool(b.get("real")) or str(b.get("id", "")).startswith("NL.IMBAG")
        out.append(Bld(str(b.get("id", i)), poly, base, eaves, roof, b.get("style"), real))
    return out


def _polys(g) -> list:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if hasattr(g, "geoms"):
        out = []
        for x in g.geoms:
            out.extend(_polys(x))
        return out
    return []


def parse_maxheight(v) -> Optional[float]:
    """OSM maxheight in metres ('3.2', '3.2 m', "12'6\"", '12 ft'); None for 'default', 'none', 'below_default'."""
    if v is None:
        return None
    s = str(v).strip().lower().replace(",", ".")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(m)?$", s)
    if m:
        return float(m.group(1))
    m = re.match(r"^(\d+)\s*'\s*(?:(\d+(?:\.\d+)?)\s*\")?$", s)
    if m:
        return float(m.group(1)) * 0.3048 + float(m.group(2) or 0) * 0.0254
    m = re.match(r"^(\d+(?:\.\d+)?)\s*ft$", s)
    if m:
        return float(m.group(1)) * 0.3048
    return None


def osm_street_zones(build: Path, osm_paths: Iterable = ()) -> list:
    """[(polygon, maxheight m or None, covered)] for the twin's street surfaces whose OSM ways carry a
    ``maxheight`` or ``covered`` tag: read from the model (<build>/*.twin: roads.geojson osm_way_ids,
    surfaces.geojson road_ids) and the build's Overpass cache (overpass_*.json in <build>/data, or ``osm_paths``).
    Empty when either is missing."""
    twin = next(iter(sorted(Path(build).glob("*.twin"))), None)
    files = [Path(p) for p in osm_paths] + sorted(Path(build).glob("data/overpass_*.json"))
    if twin is None or not files or not (twin / "roads.geojson").exists():
        return []
    tags_of = {}
    for f in files:
        try:
            for e in json.loads(f.read_text()).get("elements", ()):
                t = e.get("tags") or {}
                if e.get("type") == "way" and ("maxheight" in t or "covered" in t):
                    tags_of[int(e["id"])] = t
        except (OSError, ValueError):
            continue
    if not tags_of:
        return []
    road_tag = {}
    for f in json.loads((twin / "roads.geojson").read_text()).get("features", ()):
        pr = f.get("properties") or {}
        mh, cov = None, False
        for w in pr.get("osm_way_ids") or ():
            t = tags_of.get(int(w))
            if not t:
                continue
            h = parse_maxheight(t.get("maxheight"))
            if h is not None:
                mh = h if mh is None else min(mh, h)
            cov = cov or t.get("covered") not in (None, "no")
        if mh is not None or cov:
            road_tag[pr.get("id")] = (mh, cov)
    out = []
    if road_tag and (twin / "surfaces.geojson").exists():
        from shapely.geometry import shape
        for f in json.loads((twin / "surfaces.geojson").read_text()).get("features", ()):
            pr = f.get("properties") or {}
            if pr.get("kind") not in ROAD_KINDS:
                continue
            hits = [road_tag[r] for r in pr.get("road_ids") or () if r in road_tag]
            if not hits:
                continue
            mhs = [h for h, _ in hits if h is not None]
            try:
                g = shapely.force_2d(shape(f["geometry"]))
            except Exception:  # noqa: BLE001
                continue
            out.append((g, min(mhs) if mhs else None, any(c for _, c in hits)))
    return out


@dataclass
class Region:
    """A slab of building geometry to remove: ``poly`` (model xy) x [floor, clear] on one road layer."""
    poly: object
    layer: int
    floor: float
    clear: float                    # INF: the whole height (trim)
    bld: int                        # index into the building list
    group: str                      # road | walk
    mode: str                       # passage | trim


@dataclass
class Corridor:
    """Per (layer, group): the corridor meshes (SurfaceZ) and lane quads [(poly, zmax, zmin)]."""
    meshes: dict = field(default_factory=dict)      # (layer, group) -> SurfaceZ
    quads: dict = field(default_factory=dict)       # (layer, group) -> (STRtree, [(poly, zmax, zmin)])
    heights: dict = field(default_factory=lambda: {"road": ROAD_CLEARANCE_M, "walk": WALK_CLEARANCE_M})
    streets: list = field(default_factory=list)     # osm_street_zones
    street_tree: object = None

    def layers(self) -> list:
        return sorted({k[0] for k in list(self.meshes) + list(self.quads)})

    def local(self, layer: int, group: str, geom) -> tuple:
        """(corridor polygon near geom, [z values] there)."""
        parts, zs = [], []
        sz = self.meshes.get((layer, group))
        if sz is not None and len(sz):
            idx = sz.query(geom)
            if len(idx):
                parts.extend(sz.polys[idx])
                zs.append(sz.tris[idx][:, :, 2].ravel())
        q = self.quads.get((layer, group))
        if q is not None:
            idx = q[0].query(geom, predicate="intersects")
            for i in idx:
                poly, zmax, zmin = q[1][i]
                parts.append(poly)
                zs.append(np.array([zmax, zmin]))
        if not parts:
            return None, None
        u = shapely.union_all(parts)
        if group == "road":
            u = u.buffer(ROAD_BUFFER_M, join_style="mitre", mitre_limit=2.0)
        return u, np.concatenate(zs)

    def zvals(self, layer: int, group: str, geom) -> Optional[np.ndarray]:
        """Heights of the corridor meshes / lane samples touching geom (None: none)."""
        zs = []
        sz = self.meshes.get((layer, group))
        if sz is not None and len(sz):
            idx = sz.query(geom)
            if len(idx):
                zs.append(sz.tris[idx][:, :, 2].ravel())
        q = self.quads.get((layer, group))
        if q is not None:
            for i in q[0].query(geom, predicate="intersects"):
                zs.append(np.array(q[1][i][1:]))
        return np.concatenate(zs) if zs else None

    def street_at(self, pt) -> tuple:
        """(maxheight, covered) of the tagged street under a point, or (None, False)."""
        if self.street_tree is None:
            return None, False
        idx = self.street_tree.query(pt, predicate="intersects")
        if not len(idx):
            return None, False
        mhs = [self.streets[i][1] for i in idx if self.streets[i][1] is not None]
        return (min(mhs) if mhs else None), any(self.streets[i][2] for i in idx)


def build_corridor(out: Path, man: dict, lanes: Optional[Lanes] = None, streets: Optional[list] = None,
                   road_m: float = ROAD_CLEARANCE_M, walk_m: float = WALK_CLEARANCE_M) -> Corridor:
    """The corridor of an export: its road meshes per layer (+ the xodr's road lanes), the sidewalk lanes'
    centre bands (or, without lanes, the sidewalk meshes shrunk to their middle)."""
    cor = Corridor(heights={"road": road_m, "walk": walk_m})
    tris = defaultdict(list)
    for a in man["assets"]:
        group = "road" if a["kind"] in ROAD_KINDS else ("walk" if a["kind"] in WALK_KINDS else None)
        if group is None:
            continue
        m = load_asset(out, a)
        if m is None or not len(m["faces"]):
            continue
        tris[(int(a.get("layer") or 0), group)].append(m["pos"][m["faces"]])
    road_layers = {}
    for key, ts in tris.items():
        sz = SurfaceZ(np.concatenate(ts))
        if key[1] == "road":
            cor.meshes[key] = sz
            road_layers[key[0]] = sz
        else:
            road_layers.setdefault(("walk", key[0]), sz)
    have_walk_lanes = False
    items_of = defaultdict(list)
    if lanes is not None and len(lanes):
        # sidewalks: a band either side of the centre line (a facade on a wide pavement stays); bike lanes: whole
        for group, kinds, half in (("road", ROAD_LANES, None), ("walk", ("Sidewalk",), WALK_HALF_M),
                                   ("walk", ("Biking",), None)):
            q = lane_quads(lanes.select(kinds), half=half)
            if not q:
                continue
            have_walk_lanes |= group == "walk"
            cen = shapely.points([p.centroid.coords[0] for p, _, _ in q])
            zq = np.array([zmax for _, zmax, _ in q])
            layer_of = np.zeros(len(q), dtype=int)
            best = np.full(len(q), INF)
            # a lane on a layer's road (or sidewalk) mesh at about its height belongs to that layer
            for key, sz in road_layers.items():
                lay = key if isinstance(key, int) else key[1]
                if sz.tree is None:
                    continue
                qi, ti = sz.tree.query(cen, predicate="intersects")
                if not len(qi):
                    continue
                zt = _bary_z(sz.tris[ti], np.array([c.coords[0] for c in cen[qi]]))
                dz = np.abs(zt - zq[qi])
                for i, d in zip(qi, dz):
                    if d < 1.5 and d < best[i]:
                        best[i], layer_of[i] = d, lay
            for i, item in enumerate(q):
                items_of[(int(layer_of[i]), group)].append(item)
    for key, items in items_of.items():
        cor.quads[key] = (STRtree([p for p, _, _ in items]), items)
    if not have_walk_lanes:   # no xodr lanes: the middle of the sidewalk meshes
        for key, ts in tris.items():
            if key[1] != "walk":
                continue
            sz = SurfaceZ(np.concatenate(ts))
            cor.meshes[key] = sz
    cor.streets = streets or []
    if cor.streets:
        cor.street_tree = STRtree([g for g, _, _ in cor.streets])
    return cor


def _ceil_step(z: float) -> float:
    return math.ceil(z / STEP_M - 1e-9) * STEP_M


def _grid_cells(geom, cell: float = CELL_M) -> list:
    out = []
    if geom is None or geom.is_empty:
        return out
    minx, miny, maxx, maxy = geom.bounds
    for x in np.arange(math.floor(minx / cell) * cell, maxx, cell):
        for y in np.arange(math.floor(miny / cell) * cell, maxy, cell):
            b = box(x, y, x + cell, y + cell)
            if not b.intersects(geom):
                continue
            out.extend(p for p in _polys(geom.intersection(b)) if p.area > 1e-4)
    return out


def plan(blds: list, cor: Corridor, graze_m: float = GRAZE_M, sliver_m: float = SLIVER_M,
         top: Optional[Callable] = None) -> tuple:
    """The regions to cut ([Region], disjoint where their slabs overlap) and the decision per building and
    group. ``top`` (TopZ of the uncut buildings): a cell whose building surface would end less than
    ``THIN_TOP_M`` above the clearance is cut up to that surface (no thin plate floats over the road)."""
    regions, decisions = [], []
    for bi, b in enumerate(blds):
        fb = b.poly.buffer(FOOT_BUFFER_M, join_style="mitre", mitre_limit=2.0)
        zones = []
        for layer in cor.layers():
            for group in ("road", "walk"):
                zone, zs = cor.local(layer, group, fb)
                if zone is None:
                    continue
                if group == "walk" and (layer, "walk") not in cor.quads:
                    # sidewalk meshes without lanes: keep only their middle (a facade on the edge is fine)
                    zone = zone.buffer(-1.0)
                # a sidewalk band overlapping a road corridor at the same height gives way to the road's taller cut
                # in the overlap resolution below; one under a road deck (an underpass) keeps its own
                zone = zone.intersection(fb)
                if zone.is_empty:
                    continue
                inside = zone.intersection(b.poly)
                if not inside.is_empty:
                    zones.append((layer, group, zone, inside))
        # a building only grazed by every corridor is left alone; one that is cut anywhere is cut everywhere the
        # corridors touch it (a sidewalk lane running into a road passage's side wall)
        if not any(not ins.buffer(-graze_m / 2.0).is_empty for _, _, _, ins in zones):
            continue
        for layer, group, zone, inside in zones:
            if True:   # noqa: SIM108 - keeps the cell code at its indentation
                cells = []
                for c in _grid_cells(zone):
                    zr = cor.zvals(layer, group, c.buffer(ROAD_BUFFER_M + 0.1))
                    if zr is None or not len(zr):
                        continue
                    h = cor.heights[group]
                    mh, covered = cor.street_at(c.representative_point()) if group == "road" else (None, False)
                    if mh is not None:
                        h = max(mh + MAXHEIGHT_MARGIN_M, MAXHEIGHT_MIN_M)
                    clear = _ceil_step(float(zr.max()) + h)
                    if top is not None:
                        pts = [c.representative_point().coords[0], c.centroid.coords[0]]
                        tz = top(np.array(pts))
                        tz = tz[np.isfinite(tz) & (tz > clear) & (tz < clear + THIN_TOP_M)]
                        if len(tz):
                            clear = _ceil_step(float(tz.max()) + 0.05)
                    cells.append((c, float(zr.min()) - FLOOR_BELOW_M, clear, covered))
                if not cells:
                    continue
                clear_max = max(c[2] for c in cells)
                covered = any(c[3] for c in cells)
                mode, why = "passage", ""
                if b.eaves < clear_max + LOW_EAVES_M and b.roof < clear_max + 2.0:
                    mode, why = "trim", "low"
                elif group == "road" and not covered and inside.buffer(-sliver_m / 2.0).is_empty \
                        and len(_polys(b.poly.difference(zone))) <= 1:
                    mode, why = "trim", "sliver"
                decisions.append({"building": b.id, "layer": layer, "group": group, "mode": mode,
                                  "why": why, "area_m2": round(inside.area, 1)})
                for c, floor, clear, _ in cells:
                    regions.append(Region(c, layer, floor, INF if mode == "trim" else clear, bi, group, mode))
    # disjoint where the slabs overlap in height (whatever their layer: a lane of one layer can lie on another's
    # mesh): trims and higher cuts first; where a lower slab reaches further down, the higher one is extended
    regions.sort(key=lambda r: (-r.clear, r.layer, r.bld))
    out = []
    for r in regions:
        if out:
            tree = STRtree([a.poly for a in out])
            hits = [i for i in tree.query(r.poly, predicate="intersects")
                    if out[i].floor <= r.clear + 0.01 and out[i].clear >= r.floor - 0.01]
            extra = []
            for i in hits:
                x = out[i]
                if r.floor < x.floor - 0.01:
                    ov = x.poly.intersection(r.poly)
                    ov = [p for p in _polys(ov) if p.area > 1e-3]
                    if ov:
                        rest = x.poly.difference(r.poly)
                        out[i] = dataclasses.replace(x, poly=rest if not rest.is_empty else Polygon())
                        extra.extend(dataclasses.replace(x, poly=p, floor=r.floor) for p in ov)
            if hits:
                r.poly = r.poly.difference(shapely.union_all([out[i].poly for i in hits] +
                                                             [e.poly for e in extra]))
            out = [x for x in out if not x.poly.is_empty]
            for x in extra:
                out.extend(dataclasses.replace(x, poly=p) for p in _polys(x.poly) if p.area > 1e-3)
        for p in _polys(r.poly):
            if p.area > 1e-3:
                out.append(dataclasses.replace(r, poly=p))
    out = [dataclasses.replace(x, poly=p) for x in out for p in _polys(x.poly) if p.area > 1e-3]
    return out, decisions


# --------------------------------------------------------------------------- clipping

def _split_line(poly: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Split a convex polygon (k, D: x y z ...) by the vertical plane through segment a-b: (left, right),
    either may be None."""
    d = b - a
    n = np.array([-d[1], d[0]])
    nn = np.linalg.norm(n)
    if nn < EPS:
        return poly, None
    s = ((poly[:, :2] - a) @ n) / nn
    if (s >= -1e-7).all():
        return poly, None
    if (s <= 1e-7).all():
        return None, poly
    left, right = [], []
    k = len(poly)
    for i in range(k):
        p, q = poly[i], poly[(i + 1) % k]
        sp, sq = s[i], s[(i + 1) % k]
        if sp >= -1e-7:
            left.append(p)
        if sp <= 1e-7:
            right.append(p)
        if (sp > 1e-7 and sq < -1e-7) or (sp < -1e-7 and sq > 1e-7):
            c = p + (sp / (sp - sq)) * (q - p)
            left.append(c)
            right.append(c)
    L = np.array(left) if len(left) >= 3 else None
    R = np.array(right) if len(right) >= 3 else None
    return L, R


def _split_z(poly: np.ndarray, z: float):
    """(below, above) parts of a convex polygon at the plane Z = z."""
    s = poly[:, 2] - z
    if (s >= -1e-7).all():
        return None, poly
    if (s <= 1e-7).all():
        return poly, None
    lo, hi = [], []
    k = len(poly)
    for i in range(k):
        p, q = poly[i], poly[(i + 1) % k]
        sp, sq = s[i], s[(i + 1) % k]
        if sp <= 1e-7:
            lo.append(p)
        if sp >= -1e-7:
            hi.append(p)
        if (sp > 1e-7 and sq < -1e-7) or (sp < -1e-7 and sq > 1e-7):
            c = p + (sp / (sp - sq)) * (q - p)
            lo.append(c)
            hi.append(c)
    return (np.array(lo) if len(lo) >= 3 else None), (np.array(hi) if len(hi) >= 3 else None)


def _area3(poly: np.ndarray) -> float:
    p = poly[:, :3]
    return 0.5 * float(np.linalg.norm(np.cross(p[1:-1] - p[0], p[2:] - p[0]).sum(axis=0))) if len(p) >= 3 else 0.0


def _foot2d(tri: np.ndarray):
    """2D footprint of a 3D triangle: a polygon, or for a vertical one the segment between its two farthest
    points."""
    xy = tri[:, :2]
    a2 = 0.5 * abs((xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1]) - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1]))
    if a2 > 1e-4:
        return Polygon(xy)
    d = [(np.sum((xy[i] - xy[j]) ** 2), i, j) for i, j in ((0, 1), (1, 2), (0, 2))]
    _, i, j = max(d)
    if d[0][0] + d[1][0] + d[2][0] < 1e-8:
        return Point(xy[0])
    return LineString([xy[i], xy[j]])


def _inside(region_poly, piece: np.ndarray, normal: np.ndarray, others=()) -> bool:
    """Is a clipped piece inside a region? Its centroid; a vertical piece when one side of it is in the region
    and the other side in it or in one of ``others`` (a party wall between two cut cells goes; a wall exactly
    on the corridor edge, e.g. a facade on the kerb line, stays)."""
    c = piece[:, :2].mean(axis=0)
    hz = math.hypot(normal[0], normal[1])
    if abs(normal[2]) < 0.05 and hz > 0.5:
        n2 = np.array(normal[:2]) / hz * 0.03
        a, b = c + n2, c - n2
        ina, inb = bool(shapely.contains_xy(region_poly, *a)), bool(shapely.contains_xy(region_poly, *b))
        if ina and inb:
            return True
        if not (ina or inb):
            return False
        other = b if ina else a
        return any(bool(shapely.contains_xy(o, *other)) for o in others)
    return bool(shapely.contains_xy(region_poly, c[0], c[1]))


def _segments(poly) -> np.ndarray:
    segs = []
    for ring in [poly.exterior, *poly.interiors]:
        c = np.asarray(ring.coords)[:, :2]
        segs.append(np.stack([c[:-1], c[1:]], axis=1))
    return np.concatenate(segs) if segs else np.zeros((0, 2, 2))


def clip_mesh(pos, nrm, uv, faces, regions: list, tree: Optional[STRtree],
              removed_by: Optional[dict] = None, exclude: Optional[int] = None) -> tuple:
    """Remove the regions' slabs from a triangle mesh. Returns (pos, nrm, uv, faces, removed triangle area);
    attributes of new vertices are interpolated (texture coordinates included). ``removed_by`` (region index
    -> m2) collects what each region removed."""
    if tree is None or not len(faces):
        return pos, nrm, uv, faces, 0.0
    tri = pos[faces]
    tb = shapely.box(tri[:, :, 0].min(1) - 0.05, tri[:, :, 1].min(1) - 0.05, tri[:, :, 0].max(1) + 0.05,
                     tri[:, :, 1].max(1) + 0.05)     # a little wider: the cut beside a wall counts (_inside)
    ti, ri = tree.query(tb, predicate="intersects")
    if not len(ti):
        return pos, nrm, uv, faces, 0.0
    cand = defaultdict(list)
    for t, r in zip(ti, ri):
        if r != exclude:
            cand[int(t)].append(int(r))
    keep = np.ones(len(faces), dtype=bool)
    new_v, new_f = [], []
    removed = 0.0
    nbase = len(pos)
    seg_cache = {}
    for t, rs in cand.items():
        f = faces[t]
        attrs = np.hstack([pos[f], uv[f], nrm[f]])      # (3, 8)
        fn = np.cross(pos[f[1]] - pos[f[0]], pos[f[2]] - pos[f[0]])
        fl = np.linalg.norm(fn)
        if fl < 1e-12:
            continue
        fn = fn / fl
        g = _foot2d(pos[f])
        hit = []
        cross = []
        for r in rs:
            rp = regions[r].poly
            if not rp.intersects(g):
                continue
            hit.append(r)
            if not rp.contains(g):
                cross.append(r)
        if not hit:
            continue
        zlo = min(regions[r].floor for r in hit)
        zhi = max(regions[r].clear for r in hit)
        if attrs[:, 2].min() >= zhi or attrs[:, 2].max() <= zlo - 1e-3:
            continue
        pieces = [attrs]
        if cross:
            x0, y0 = attrs[:, :2].min(0) - 0.01
            x1, y1 = attrs[:, :2].max(0) + 0.01
            for r in cross:
                if r not in seg_cache:
                    seg_cache[r] = _segments(regions[r].poly)
                segs = seg_cache[r]
                m = ((np.maximum(segs[:, 0, 0], segs[:, 1, 0]) >= x0) & (np.minimum(segs[:, 0, 0], segs[:, 1, 0]) <= x1)
                     & (np.maximum(segs[:, 0, 1], segs[:, 1, 1]) >= y0) & (np.minimum(segs[:, 0, 1], segs[:, 1, 1]) <= y1))
                for a, b in segs[m]:
                    nxt = []
                    for p in pieces:
                        L, R = _split_line(p, a, b)
                        nxt.extend(x for x in (L, R) if x is not None)
                    pieces = nxt
        out = []
        for p in pieces:
            parts = [p]
            for r in hit:
                reg = regions[r]
                if not _inside(reg.poly, p, fn, [regions[o].poly for o in rs if o != r]):
                    continue
                a_before = sum(_area3(q) for q in parts)
                nxt = []
                for q in parts:
                    lo, rest = _split_z(q, reg.floor - 1e-3)
                    if lo is not None:
                        nxt.append(lo)
                    if rest is not None and reg.clear < INF:
                        _, hi = _split_z(rest, reg.clear)
                        if hi is not None:
                            nxt.append(hi)
                parts = nxt
                if removed_by is not None:
                    removed_by[r] = removed_by.get(r, 0.0) + a_before - sum(_area3(q) for q in parts)
            out.extend(parts)
        a_in = _area3(attrs)
        a_out = sum(_area3(p) for p in out)
        if a_in - a_out < 1e-6:
            continue
        keep[t] = False
        removed += max(0.0, a_in - a_out)
        for p in out:
            if _area3(p) < 1e-7:
                continue
            k = len(p)
            new_v.append(p)
            new_f.append(np.array([[0, i, i + 1] for i in range(1, k - 1)]) + nbase)
            nbase += k
    if not new_v and keep.all():
        return pos, nrm, uv, faces, 0.0
    faces2 = faces[keep]
    if new_v:
        V = np.vstack(new_v)
        pos = np.vstack([pos, V[:, :3]])
        uv = np.vstack([uv, V[:, 3:5]])
        nn = V[:, 5:8]
        nn = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        nrm = np.vstack([nrm, nn])
        faces2 = np.vstack([faces2, np.vstack(new_f)])
    # drop unused vertices
    used = np.unique(faces2)
    remap = -np.ones(len(pos), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return pos[used], nrm[used], uv[used], remap[faces2], removed


# --------------------------------------------------------------------------- closing the cut

def plain_key(style: Optional[str], bid: str) -> tuple:
    """(texture key, facades Style) of the windowless wall of a building's facade style: glass and shopfront
    walls get the concrete of an office ceiling."""
    from twinmodel.export import facades
    st = facades.STYLES.get(style or "") or facades.STYLES["render_modern"]
    if st.wall == "glass" or st.shop:
        st = facades.STYLES["office_concrete"]
    var = facades.variant_of(st.name, bid)
    return "plain_%s_%d" % (st.name, var), st


class _Faces:
    def __init__(self):
        self.pos, self.nrm, self.uv, self.faces, self.n = [], [], [], [], 0

    def add(self, v, normal, uv, tri):
        v = np.asarray(v, dtype=np.float64)
        tri = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
        # winding: counter-clockwise seen from the normal's side
        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        flip = (np.cross(b - a, c - a) @ np.asarray(normal, dtype=np.float64)) < 0
        tri = tri.copy()
        tri[flip] = tri[flip][:, ::-1]
        self.pos.append(v)
        self.nrm.append(np.repeat(np.asarray(normal, dtype=np.float64)[None, :], len(v), axis=0))
        self.uv.append(np.asarray(uv, dtype=np.float64))
        self.faces.append(tri + self.n)
        self.n += len(v)

    def arrays(self):
        return np.vstack(self.pos), np.vstack(self.nrm), np.vstack(self.uv), np.vstack(self.faces)


def _flat_uv(v, st):
    from twinmodel.export import facades
    if st is None:
        return v[:, :2].copy()
    return np.column_stack([v[:, 0] / (facades.TEX_BAYS * st.bay_m), -v[:, 1] / (facades.TEX_STOREYS * st.storey_m)])


def _wall_uv(v, normal, st):
    from twinmodel.export import facades
    along = facades.wall_frame(normal)
    if st is None:
        return np.column_stack([v @ along, v[:, 2]])
    return facades.blank_uv(v, normal, st)


def close_cut(regions: list, blds: list, top: TopZ, plain: bool, tile_m: float = 250.0,
              cut_blds: Optional[set] = None, tree: Optional[STRtree] = None) -> tuple:
    """Soffits, floors and cut walls that close the cut regions of the buildings in ``cut_blds`` (those that
    lost geometry; default all): {(tile, material key, real): _Faces}, the keys used, and stats."""
    from twinmodel.export.ue import _tile_index
    from twinmodel.export.mesh import triangulate_polygon
    out = defaultdict(_Faces)
    keys = {}
    stats = {"soffit_m2": 0.0, "wall_m2": 0.0, "floor_m2": 0.0}
    all_tree = STRtree([r.poly for r in regions]) if regions else None
    for ri, r in enumerate(regions):
        if cut_blds is not None and r.bld not in cut_blds:
            continue
        b = blds[r.bld]
        P = r.poly.intersection(b.poly)
        for p in _polys(P):
            if p.area < 0.05:
                continue
            key, st = ("building", None) if plain else plain_key(b.style, b.id)
            keys[key] = st
            cen = p.centroid
            dst = out[(_tile_index(cen.x, cen.y, tile_m), key, b.real)]
            sink = _Faces()
            rp = p.representative_point()
            top_rp = top([[rp.x, rp.y]])[0]
            if not math.isfinite(top_rp):
                top_rp = b.roof     # the footprint reaches beyond the mesh: the manifest's roof height
            if b.base >= (r.clear if r.clear < INF else top_rp) - 0.05 or top_rp <= r.floor + 0.05:
                continue        # the building does not reach into this slab
            # soffit: where the building is above the clearance (a footprint can reach beyond its roof)
            if r.clear < INF:
                cells = _grid_cells(p, SOFFIT_GRID_M)
                if cells:
                    # a cell is kept when the building is above any of its corners or its middle
                    pts, own = [], []
                    for ci, c in enumerate(cells):
                        q = [c.representative_point().coords[0]] + list(np.asarray(c.exterior.coords)[:-1])
                        pts.extend(q)
                        own.extend([ci] * len(q))
                    cz = top(np.array(pts))
                    cmax = np.full(len(cells), -INF)
                    np.maximum.at(cmax, np.array(own), np.where(np.isfinite(cz), cz, -INF))
                    keep = [c for c, z in zip(cells, cmax) if z > r.clear + 0.001]
                    sof = shapely.union_all(keep).buffer(0) if keep else None
                    for sp in _polys(sof):
                        v2, f = triangulate_polygon(orient(sp, 1.0))
                        if len(f):
                            v = np.column_stack([v2, np.full(len(v2), r.clear)])
                            sink.add(v, (0.0, 0.0, -1.0), _flat_uv(v, st), f)
                            stats["soffit_m2"] += sp.area
            # floor (structure under an elevated road, or the base below the road)
            if b.base < r.floor - 0.5:
                v2, f = triangulate_polygon(orient(p, 1.0))
                if len(f):
                    v = np.column_stack([v2, np.full(len(v2), r.floor)])
                    sink.add(v, (0.0, 0.0, 1.0), _flat_uv(v, st), f)
                    stats["floor_m2"] += p.area
            # walls where the cut meets uncut building (or a lower cut)
            fb = b.poly.boundary
            po = orient(p, 1.0)
            for ring in [po.exterior, *po.interiors]:
                c = np.asarray(ring.coords)[:, :2]
                for a, e in zip(c[:-1], c[1:]):
                    d = e - a
                    L = float(np.linalg.norm(d))
                    if L < 0.02:
                        continue
                    m = (a + e) / 2.0
                    if fb.distance(Point(m)) < 0.02:
                        continue        # on the facade: the opening of the passage
                    n_in = np.array([-d[1], d[0]]) / L      # interior (the cut) on the left
                    normal = np.array([n_in[0], n_in[1], 0.0])
                    # pieces of at most WALL_PIECE_M: the neighbour can change along a long edge
                    k = max(1, int(math.ceil(L / WALL_PIECE_M)))
                    ts = np.linspace(0.0, 1.0, k + 1)
                    pts = a[None, :] + ts[:, None] * d[None, :]
                    tz = top(pts)
                    tz = np.where(np.isfinite(tz), tz, b.roof)
                    for i in range(k):
                        mi = (pts[i] + pts[i + 1]) / 2.0
                        lo_hi = [(r.floor, r.clear)]
                        # every cut beside this one, on any layer: its slab needs no wall (looked for a little
                        # way out, over the slivers two independent cell outlines leave between them)
                        nb = []
                        for off in (0.05, 0.15, 0.3):
                            q = Point(mi - n_in * off)
                            nb = [regions[j] for j in all_tree.query(q, predicate="intersects")]
                            nb = [x for x in nb if x is not r and x.poly.contains(q)]
                            if nb:
                                break
                        for x in nb:
                            nxt = []
                            for lo, hi in lo_hi:
                                if x.clear <= lo or x.floor >= hi:
                                    nxt.append((lo, hi))
                                    continue
                                if x.floor > lo:
                                    nxt.append((lo, x.floor))
                                if x.clear < hi:
                                    nxt.append((x.clear, hi))
                            lo_hi = nxt
                        for lo, hi in lo_hi:
                            lo = max(lo, b.base)
                            z1a, z1b = min(hi, tz[i]), min(hi, tz[i + 1])
                            if not (math.isfinite(z1a) and math.isfinite(z1b)):
                                continue
                            if z1a <= lo + 0.05 and z1b <= lo + 0.05:
                                continue
                            z1a, z1b = max(z1a, lo), max(z1b, lo)
                            v = np.array([[*pts[i], lo], [*pts[i + 1], lo], [*pts[i + 1], z1b], [*pts[i], z1a]])
                            sink.add(v, normal, _wall_uv(v, normal, st), [[0, 1, 2], [0, 2, 3]])
                            stats["wall_m2"] += L / k * ((z1a + z1b) / 2 - lo)
            if sink.faces:
                # what another cut (an overlapping layer, a neighbour's cell) removes is not put back
                pos, nrm, uv, faces = sink.arrays()
                pos, nrm, uv, faces, _ = clip_mesh(pos, nrm, uv, faces, regions, tree, exclude=ri)
                if len(faces):
                    dst.pos.append(pos)
                    dst.nrm.append(nrm)
                    dst.uv.append(uv)
                    dst.faces.append(faces + dst.n)
                    dst.n += len(pos)
    return out, keys, {k: round(v, 1) for k, v in stats.items()}


# --------------------------------------------------------------------------- footprints for simulation tools

OPEN_M = 2.0     # a cut at least this tall leaves room for a walker or a car: not part of the solid footprint


def _no_holes(poly) -> list:
    """A polygon split into pieces without holes (cut through each hole's centre)."""
    todo, out = [poly], []
    while todo:
        p = todo.pop()
        if not p.interiors:
            out.append(p)
            continue
        hx = Polygon(p.interiors[0]).centroid.x
        minx, miny, maxx, maxy = p.bounds
        for half in (box(minx - 1, miny - 1, hx, maxy + 1), box(hx, miny - 1, maxx + 1, maxy + 1)):
            todo.extend(q for q in _polys(p.intersection(half)) if q.area > 0.05)
    return out


def solid_footprints(blds: list, regions: list) -> list:
    """The buildings' footprints minus the passages cut through them (where at least OPEN_M of height was
    cleared), as clipcheck/buildings.json records in CARLA x, y: [{id, p: [[x, y], ...], z0, z1, s}], holes
    split away (the readers take one ring)."""
    tall = [r for r in regions if r.clear - r.floor >= OPEN_M]
    tree = STRtree([r.poly for r in tall]) if tall else None
    out = []
    for b in blds:
        g = b.poly
        if tree is not None:
            hit = tree.query(g, predicate="intersects")
            if len(hit):
                g = g.difference(shapely.union_all([tall[i].poly for i in hit]))
        for p in _polys(shapely.make_valid(g.buffer(0))):
            for q in _no_holes(p):
                # one valid ring per record: the readers use an even-odd test on the exterior alone
                parts = [x for x in _polys(shapely.make_valid(Polygon(q.exterior).simplify(0.02))) if x.area >= 0.5]
                parts = [y for x in parts for y in _no_holes(x)]
                for r in parts:
                    if not r.is_valid or r.area < 0.5:
                        continue
                    ring = np.asarray(r.exterior.coords)
                    out.append({"id": b.id, "p": [[round(float(x), 2), round(float(-y), 2)] for x, y in ring],
                                "z0": round(b.base, 2), "z1": round(b.roof, 2), "s": "twin"})
    return out


def write_footprints(build: Path, man: Optional[dict] = None, *, lanes: Optional[Lanes] = None, streets=None,
                     graze_m: float = GRAZE_M, sliver_m: float = SLIVER_M) -> int:
    """Rewrite <build>/<Name>.buildings.json of an export cut before (its plan redone on the uncut meshes kept in
    meshes/uncut/, nothing else written). Returns the number of records."""
    build = Path(build)
    out = build / "ue"
    man = man if man is not None else json.loads((out / "manifest.json").read_text())
    rec = man.get("clearance") or {}
    p0 = rec.get("params") or {}
    added = set(rec.get("added_assets", ()))
    uncut = {a["asset"]: a for a in rec.get("uncut_assets", ())}
    tops = []
    for a in [x for x in man["assets"] if x["asset"] not in added] + \
             [a for n, a in uncut.items() if n not in {x["asset"] for x in man["assets"]}]:
        if a["kind"] not in BUILDING_KINDS:
            continue
        aa = dict(a)
        if a["asset"] in uncut and (out / "meshes" / "uncut" / Path(a["file"]).name).exists():
            aa["file"] = "meshes/uncut/" + Path(a["file"]).name
        m = load_asset(out, aa)
        if m is not None and len(m["faces"]):
            tops.append(m["pos"][m["faces"]])
    top = TopZ(np.concatenate(tops) if tops else np.zeros((0, 3, 3)))
    blds = manifest_buildings(man)
    cor = build_corridor(out, man, lanes, streets, p0.get("road_m", ROAD_CLEARANCE_M),
                         p0.get("walk_m", WALK_CLEARANCE_M))
    regions, _ = plan(blds, cor, graze_m, sliver_m, top)
    recs = solid_footprints(blds, regions)
    (build / f"{man['name']}.buildings.json").write_text(json.dumps(recs))
    return len(recs)


# --------------------------------------------------------------------------- the step

def restore(out: Path, man: dict) -> bool:
    """Undo a previous cut: the uncut meshes back from meshes/uncut/, the added assets and textures removed.
    Returns True when there was one."""
    rec = man.get("clearance")
    if not rec:
        return False
    names = {a["asset"] for a in rec.get("uncut_assets", ())} | set(rec.get("added_assets", ()))
    for a in man["assets"]:
        if a["asset"] in set(rec.get("added_assets", ())):
            (out / a["file"]).unlink(missing_ok=True)
    assets = [a for a in man["assets"] if a["asset"] not in names]
    for a in rec.get("uncut_assets", ()):
        src = out / "meshes" / "uncut" / Path(a["file"]).name
        if src.exists():
            shutil.move(str(src), str(out / a["file"]))
        assets.append(a)
    man["assets"] = assets
    for k in rec.get("added_textures", ()):
        man.get("textures", {}).pop(k, None)
    man.pop("clearance", None)
    return True


def apply(build: Path, man: Optional[dict] = None, *, lanes: Optional[Lanes] = None, streets=None,
          road_m: float = ROAD_CLEARANCE_M, walk_m: float = WALK_CLEARANCE_M, graze_m: float = GRAZE_M,
          sliver_m: float = SLIVER_M, write: bool = True, log: Callable = print) -> dict:
    """Cut the building meshes of the bake export <build>/ue so the corridor is clear (module docstring);
    writes the meshes and the manifest. Returns the stats (also in manifest ``clearance``)."""
    import time
    from twinmodel.export import facades
    from twinmodel.export.ue import write_glb
    t0 = time.time()
    build = Path(build)
    out = build / "ue"
    man = man if man is not None else json.loads((out / "manifest.json").read_text())
    restore(out, man)
    name = man["name"]
    blds = manifest_buildings(man)
    cor = build_corridor(out, man, lanes, streets, road_m, walk_m)
    # all building triangles (uncut) for the roof heights over the cut
    bassets = [a for a in man["assets"] if a["kind"] in BUILDING_KINDS]
    meshes = {}
    tops = []
    for a in bassets:
        m = load_asset(out, a)
        if m is None:
            continue
        meshes[a["asset"]] = m
        if len(m["faces"]):
            tops.append(m["pos"][m["faces"]])
    top = TopZ(np.concatenate(tops) if tops else np.zeros((0, 3, 3)))
    regions, decisions = plan(blds, cor, graze_m, sliver_m, top)
    stats = {"buildings": len(blds), "regions": len(regions),
             "passage": len({d["building"] for d in decisions if d["mode"] == "passage"}),
             "trim": len({d["building"] for d in decisions if d["mode"] == "trim"}),
             "road_m": road_m, "walk_m": walk_m, "lanes": int(len(lanes)) if lanes is not None else 0,
             "osm_streets": len(cor.streets)}
    log(f"clearance: {len(blds)} buildings, {len(regions)} cut cells; passages {stats['passage']}, "
        f"trims {stats['trim']} buildings")
    tree = STRtree([r.poly for r in regions]) if regions else None
    uncut, removed_m2, n_changed = [], 0.0, 0
    removed_by = {}
    new_assets = []
    for a in man["assets"]:
        if a["asset"] not in meshes or tree is None:
            new_assets.append(a)
            continue
        m = meshes[a["asset"]]
        pos, nrm, uv, faces, rem = clip_mesh(m["pos"], m["nrm"], m["uv"], m["faces"], regions, tree, removed_by)
        if faces is m["faces"]:
            new_assets.append(a)
            continue
        n_changed += 1
        removed_m2 += rem
        uncut.append(dict(a))
        if write:
            (out / "meshes" / "uncut").mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / a["file"], out / "meshes" / "uncut" / Path(a["file"]).name)
        if not len(faces):
            if write:
                (out / a["file"]).unlink(missing_ok=True)
            continue
        b = dict(a)
        if write:
            b["bbox_ue"] = write_glb(out / a["file"], a["asset"], pos, nrm, uv, faces, m["material"], m["color"])
        b["vertices"], b["triangles"] = int(len(pos)), int(len(faces))
        new_assets.append(b)
    # closing faces
    plain = not man.get("textures") or not any(k.startswith(("facade_", "photo_")) for k in man.get("textures", {}))
    lost = defaultdict(float)
    for r, a_ in removed_by.items():
        lost[regions[r].bld] += a_
    # every building with a cut gets it closed, also where its mesh lost nothing there (the lane runs under its
    # roof between walls that stand outside the corridor)
    cut_blds = {r.bld for r in regions}
    # the lowest point of each cut building's mesh: walls extended down to the terrain (skirts) reach below the
    # manifest's ground height
    if tops and cut_blds:
        allt = np.concatenate(tops)
        cen = allt[:, :, :2].mean(axis=1)
        ctree = STRtree(shapely.points(cen[:, 0], cen[:, 1]))
        for bi in cut_blds:
            idx = ctree.query(blds[bi].poly.buffer(0.5), predicate="intersects")
            if len(idx):
                blds[bi].base = min(blds[bi].base, float(allt[idx][:, :, 2].min()))
    sinks, keys, cstats = close_cut(regions, blds, top, plain, float(man.get("tile_m", 250.0)), cut_blds, tree)
    added, added_tex = [], []
    textures = man.setdefault("textures", {}) if not plain else man.get("textures", {})
    for (tile, key, real), fb in sorted(sinks.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if not fb.faces:
            continue
        pos, nrm, uv, faces = fb.arrays()
        tx, ty = tile
        asset = f"{name}_L0_building_{tx}_{ty}_cut_{key}" + ("" if real else "_osm")
        asset = asset.replace("-", "m")
        bbox = write_glb(out / "meshes" / f"{asset}.glb", asset, pos, nrm, uv, faces, key) if write else None
        entry = {"file": f"meshes/{asset}.glb", "asset": asset, "layer": 0,
                 "kind": "building_3dbag" if real else "building", "material": key,
                 "semantic": "Building", "tile": [tx, ty], "vertices": int(len(pos)), "triangles": int(len(faces)),
                 "bbox_ue": bbox, "source": "clearance cut (twinmodel.export.clearance)"}
        if not plain:
            entry["texture"] = key
        new_assets.append(entry)
        added.append(asset)
    if not plain and write:
        tex_dir = out / "textures"
        tex_dir.mkdir(exist_ok=True)
        for key, st in sorted(keys.items()):
            if key in textures or st is None:
                continue
            var = int(key.rsplit("_", 1)[1])
            alb, nrm_, orm = facades.synth_facade(dataclasses.replace(st, blank=True), var, px=facades.TEX_PX // 2)
            entry = {}
            for suffix, arr in (("d", alb), ("n", nrm_), ("orm", orm)):
                rel = f"textures/T_{name}_{key}_{suffix}.png"
                facades.save_png(out / rel, arr)
                entry[{"d": "base_color", "n": "normal", "orm": "orm"}[suffix]] = rel
            entry.update(style=st.name, variant=var, bay_m=st.bay_m, storey_m=st.storey_m, plain=True)
            textures[key] = entry
            added_tex.append(key)
    man["assets"] = new_assets
    stats.update(cut_buildings=len([bi for bi, a_ in lost.items() if a_ > 0.05]), changed_assets=n_changed, removed_m2=round(removed_m2, 1), added_assets=len(added), **cstats,
                 seconds=round(time.time() - t0, 1))
    man["clearance"] = {"version": 1, "stats": stats, "uncut_assets": uncut, "added_assets": added,
                        "added_textures": added_tex,
                        "params": {"road_m": road_m, "walk_m": walk_m, "graze_m": graze_m, "sliver_m": sliver_m,
                                   "cell_m": CELL_M, "walk_half_m": WALK_HALF_M, "floor_below_m": FLOOR_BELOW_M}}
    man["stats"]["triangles"] = int(sum(a.get("triangles", 0) for a in new_assets))
    man["stats"]["assets"] = len(new_assets)
    if write:
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        (build / "clearance_cuts.json").write_text(json.dumps(decisions, indent=0))
        # the solid footprints, for the simulation's own checks (coupled-carla tools/trial_buildings.py,
        # clipcheck): copied next to the twin's OpenDRIVE as <Name>.buildings.json
        (build / f"{name}.buildings.json").write_text(json.dumps(solid_footprints(blds, regions)))
    log(f"clearance: {n_changed} meshes cut ({removed_m2:.0f} m2 of building surface removed), {len(added)} "
        f"closing meshes (soffits {cstats['soffit_m2']:.0f} m2, walls {cstats['wall_m2']:.0f} m2) in "
        f"{stats['seconds']:.0f} s")
    return stats


# --------------------------------------------------------------------------- the check

def building_triangles(out: Path, man: dict) -> np.ndarray:
    ts = []
    for a in man["assets"]:
        if a["kind"] not in BUILDING_KINDS:
            continue
        m = load_asset(out, a)
        if m is not None and len(m["faces"]):
            ts.append(m["pos"][m["faces"]])
    return np.concatenate(ts) if ts else np.zeros((0, 3, 3))


def check(build: Path, lanes: Lanes, man: Optional[dict] = None, bands: Optional[dict] = None,
          tris: Optional[np.ndarray] = None, streets: Optional[list] = None) -> dict:
    """Lane samples blocked by buildings. ``crossings`` (the stand-alone checker's measure): for each pair of
    consecutive 1 m samples of a driving (sidewalk) lane, a building triangle that crosses the segment between
    ``bands`` m above it; walls (vertical triangles) count over their height range, other triangles at their
    height over the segment's ends and middle. ``blocked`` adds the samples under a building surface inside the
    band (an overhang a vehicle would hit). ``inside`` (information only, not in ``blocked``): samples a vertical
    ray from the band's bottom shows inside a building solid (an odd number of building surfaces above); open
    source meshes (canopies, overlapping parts, no ground faces) make it approximate. Returns {group: {samples, blocked, crossings, roads, worst, hits: [...]}} (model x, y, z
    and lat/lon of each blocked sample). On a street with an OSM ``maxheight`` (``streets``, see
    osm_street_zones) the road band ends just under that street's clearance."""
    build = Path(build)
    out = build / "ue"
    man = man if man is not None else json.loads((out / "manifest.json").read_text())
    bands = bands or CHECK_BANDS
    tris = building_triangles(out, man) if tris is None else tris
    res = {}
    if len(tris):
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nl = np.linalg.norm(n, axis=1)
        ok = nl > 1e-9
        tris, n, nl = tris[ok], n[ok], nl[ok]
        vertical = np.abs(n[:, 2]) < 0.3 * nl
        foot = [_foot2d(t) for t in tris]
        tree = STRtree(foot)
        zlo, zhi = tris[:, :, 2].min(1), tris[:, :, 2].max(1)
    to_ll = _to_lonlat(man)
    for group, kinds in CHECK_LANES.items():
        lo_b, hi_b = bands[group]
        L = lanes.select(kinds)
        ia, ib = L.segments()
        blocked = np.zeros(len(L), dtype=bool)
        hi_s = np.full(len(L), float(hi_b))     # band top per sample
        if group == "road" and streets and len(L):
            st_tree = STRtree([g for g, _, _ in streets])
            qi, si_ = st_tree.query(shapely.points(L.xyz[:, 0], L.xyz[:, 1]), predicate="intersects")
            for q, k in zip(qi, si_):
                mh = streets[k][1]
                if mh is not None:
                    hi_s[q] = min(hi_s[q], max(mh + MAXHEIGHT_MARGIN_M, MAXHEIGHT_MIN_M) - 0.15)
        if len(tris) and len(ia):
            segs = shapely.linestrings(np.stack([L.xyz[ia, :2], L.xyz[ib, :2]], axis=1))
            si, ti = tree.query(segs, predicate="intersects")
            for s, t in zip(si, ti):
                i, j = ia[s], ib[s]
                if blocked[i]:
                    continue
                z = 0.5 * (L.xyz[i, 2] + L.xyz[j, 2])
                hi_b = min(hi_s[i], hi_s[j])
                if vertical[t]:
                    hit = False
                    if zlo[t] < z + hi_b and zhi[t] > z + lo_b:
                        x = foot[t].intersection(segs[s])
                        if not x.is_empty:
                            zi = _vertical_z_range(tris[t], np.asarray(x.representative_point().coords[0]))
                            hit = zi is not None and zi[0] < z + hi_b and zi[1] > z + lo_b
                else:
                    pts = np.array([L.xyz[i, :2], L.xyz[j, :2], 0.5 * (L.xyz[i, :2] + L.xyz[j, :2])])
                    inside = shapely.contains_xy(foot[t].buffer(1e-6), pts[:, 0], pts[:, 1])
                    if not inside.any():
                        pts = np.array([np.asarray(foot[t].intersection(segs[s]).representative_point().coords[0])])
                    else:
                        pts = pts[inside]
                    zt = _bary_z(np.repeat(tris[t][None], len(pts), axis=0), pts)
                    hit = bool(((zt > z + lo_b) & (zt < z + hi_b)).any())
                if hit:
                    blocked[i] = True
        crossings = int(blocked.sum())
        why = np.where(blocked, "wall", "").astype(object)
        # inside a building solid: a vertical ray from lo above the sample crosses an odd number of building
        # surfaces (the exports have no bottom faces: roofs and soffits are what a ray meets)
        if len(tris) and len(L):
            flat = ~vertical
            if flat.any():
                ftree = STRtree(shapely.polygons(tris[flat][:, :, :2]))
                ft = tris[flat]
                qi, ti = ftree.query(shapely.points(L.xyz[:, 0], L.xyz[:, 1]), predicate="intersects")
                if len(qi):
                    zt = _bary_z(ft[ti], L.xyz[qi, :2])
                    above = zt > L.xyz[qi, 2] + lo_b
                    # a point on an edge shared by two triangles meets the same surface twice: once per height
                    uq = np.unique(np.column_stack([qi[above], np.round(zt[above], 3)]), axis=0)
                    cnt = np.bincount(uq[:, 0].astype(int), minlength=len(L))
                    band = above & (zt < L.xyz[qi, 2] + hi_s[qi])
                    why[qi[band]] = np.where(why[qi[band]] == "", "surface", why[qi[band]])
                    blocked[qi[band]] = True
                    enclosed = ((cnt % 2) == 1) & ~blocked
                    why[enclosed] = "inside"
        hits = []
        for i in np.nonzero(blocked | (why == "inside"))[0]:
            x, y, z = L.xyz[i]
            lon, lat = to_ll(x, y)
            hits.append({"road": int(L.road[i]), "why": str(why[i]), "x": round(float(x), 2), "y": round(float(y), 2),
                         "z": round(float(z), 2), "lat": round(lat, 6), "lon": round(lon, 6)})
        roads = defaultdict(int)
        for h in hits:
            if h["why"] != "inside":
                roads[h["road"]] += 1
        n_in = sum(1 for h in hits if h["why"] == "inside")
        res[group] = {"samples": int(len(L)), "blocked": len(hits) - n_in, "crossings": crossings, "inside": n_in,
                      "roads": len(roads),
                      "worst": [{"road": r, "blocked": k,
                                 "lat": next(h["lat"] for h in hits if h["road"] == r and h["why"] != "inside"),
                                 "lon": next(h["lon"] for h in hits if h["road"] == r and h["why"] != "inside")}
                                for r, k in sorted(roads.items(), key=lambda kv: -kv[1])[:12]],
                      "hits": hits}
    return res


def _vertical_z_range(t: np.ndarray, p: np.ndarray) -> Optional[tuple]:
    """(z min, z max) of a vertical triangle on the vertical line through p (on its 2D footprint)."""
    xy = t[:, :2]
    d = [(np.sum((xy[i] - xy[j]) ** 2), i, j) for i, j in ((0, 1), (1, 2), (0, 2))]
    _, i, j = max(d)
    ax = xy[j] - xy[i]
    L = float(np.linalg.norm(ax))
    if L < 1e-9:
        return float(t[:, 2].min()), float(t[:, 2].max())
    ax /= L
    u = (xy - xy[i]) @ ax
    up = float((p - xy[i]) @ ax)
    zs = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ua, ub = u[a], u[b]
        if abs(ub - ua) < 1e-9:
            if abs(up - ua) < 1e-6:
                zs.extend([t[a, 2], t[b, 2]])
            continue
        f = (up - ua) / (ub - ua)
        if -1e-6 <= f <= 1 + 1e-6:
            zs.append(t[a, 2] + f * (t[b, 2] - t[a, 2]))
    return (float(min(zs)), float(max(zs))) if zs else None


def _to_lonlat(man: dict):
    geo = man.get("geo_reference")
    if not geo:
        return lambda x, y: (float("nan"), float("nan"))
    try:
        from pyproj import Transformer
        proj = " ".join(t for t in geo.split() if not t.startswith(("+geoidgrids", "+vunits")))
        tr = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
        return lambda x, y: tr.transform(x, y)
    except Exception:  # noqa: BLE001
        return lambda x, y: (float("nan"), float("nan"))
