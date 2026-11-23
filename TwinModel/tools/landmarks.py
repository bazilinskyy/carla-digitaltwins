"""Landmarks for a twin's ``ue/`` export: hand-described detail for named buildings and rooftop objects.

Runs on the export (``manifest.json`` + ``meshes/`` + ``textures/``) after real_buildings, ground_photo and
road_clearance, so the build directory is not needed. Each landmark:

* ``curtain_wall``: a building (3DBAG id) whose generated walls are cut out of the shared wall meshes and
  rebuilt from their own 3DBAG geometry as a curtain wall: glass with per-panel tint, roughness and a slight
  tilt (so reflections break up like real glazing), lit at night through the facade night mask; aluminium
  mullions on every bay; dark floor-edge spandrels on every storey; vertical glass fins on the mullions over
  double-height zones; an aluminium coping around each roof. Roofs keep the aerial photo.
* ``roof_letters``: extruded letters (a logo) standing on a building's roof, with a steel frame behind.

The bake needs no change: new assets have kind ``landmark`` and a ``texture`` key (bake_level.asset_key).
Re-running restores the export from ``landmarks_backup/`` first, so the step is idempotent.

TU/e campus (TUeCampus), the default set:

* Atlas (former Hoofgebouw, S.J. van Embden 1958-63; renovated 2014-19 by Team V / Team RSVP): the whole
  building is a glass curtain wall; vertical fins on the mullions act as a safety barrier, mark the
  double-height spaces and add depth (architect's description). Bay 1.8 m and storey 3.6 m are assumed
  (not published); the massing is 3DBAG LoD 2.2.
* The red "TU/e" letters on the roof of Vertigo (Cursor, 2018 and 2023: "the logo on the roof of Vertigo";
  capital T and U, a slash, a lowercase e, red). Height and position are not published: 5.5 m letters on
  the west edge of the tower roof, facing west towards the station (user's choice, 2026-11-22), placed along
  the edge where no plant room is in the way.

Every other twin gets the generic set: the OSM detail of its buildings (tools/osm_detail.py), picked by the
export's name (SPECS).

Usage: python tools/landmarks.py <build>/ue [--letters-facing north|south|east|west] [--restore]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mapbox_earcut as earcut  # noqa: E402
from shapely import contains_xy  # noqa: E402
from shapely.affinity import scale as shp_scale  # noqa: E402
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from twinmodel.export import facades  # noqa: E402
from twinmodel.export.ue import _tile_index, gltf_to_model, read_glb, write_glb  # noqa: E402

WALL_KINDS = ("building_3dbag", "building")
KIND = "landmark"
BACKUP = "landmarks_backup"
PROVENANCE = "landmark: 3DBAG LoD 2.2 massing, detail described (tools/landmarks.py)"

TUE_CAMPUS = {
    # De Zaale passes under the north end of the Atlas tower at street level (OSM: layer=-1 on the passage). The twin
    # read it as underground: road 277 dives 6 m and De Zaale stops at the tower, behind a wall. Level it, join it to
    # road 201 and pave the gap; walls over it start 4.6 m up (the clearance pass's road height).
    "dezaale_passage": {"type": "passage", "road": "277", "to": "201", "z": 16.2575,
                        "drop": {"layer": -1, "box": [-285.5, -278.5, -43.0, -26.0]},
                        "fill": {"x": [-282.4, -259.6], "axis_y": -34.9, "drive_half": 3.5, "walk_m": 2.0,
                                 "kerb_m": 0.15},
                        "clearance_m": 4.6, "soffit_under": "NL.IMBAG.Pand.0772100001010279"},
    # the ground floors, entrances and steps of every building from OSM (tools/osm_detail.py); Atlas's three entrances
    # have revolving doors at the top of stairs (user, 2026-11-22: OSM tags only the west one and has no stairs at the
    # north one, in the passage under the tower; 6 steps assumed there, as at the west entrance)
    "osm": {"type": "osm_detail", "revolving": [[-269.5, -52.9], [-269.5, -140.4], [-285.5, -92.8]],
            # Atlas's entrance stairs span the entrance fronts (user: "much wider" than 2.5-6 m; estimated from the
            # fronts between the cores: north and south 10.5 m, west 9 m)
            "extra_steps": [{"line": [[-269.5, -49.4], [-269.5, -51.6]], "step_count": 6, "width": 10.5,
                             "incline": "up", "handrail": "yes"}],
            "step_widths": {"1292493545": 10.5, "1078555767": 9.0}},
    "atlas": {"type": "curtain_wall", "building": "NL.IMBAG.Pand.0772100001010279",
              "bay_m": 1.8, "storey_m": 3.6, "fins_min_height_m": 20.0},
    "tue_letters": {"type": "roof_letters", "building": "NL.IMBAG.Pand.0772100001003373",
                    "text": "TU/e", "height_m": 5.5, "depth_m": 0.45, "setback_m": 4.0, "facing": "west"},
}

# every other twin: the OSM detail of its buildings only (open ground floors, entrances, steps); hand-described
# landmarks are per twin, by the export's name
GENERIC = {"osm": {"type": "osm_detail"}}
SPECS = {"TUeCampus": TUE_CAMPUS}


def spec_for(name: str) -> dict:
    return SPECS.get(name, GENERIC)


# --------------------------------------------------------------------------- materials

def _orm(rough, metal, px=4):
    o = np.zeros((px, px, 4), dtype=np.uint8)
    o[..., 0] = 255
    o[..., 1] = np.clip(np.round(np.asarray(rough) * 255), 0, 255)
    o[..., 2] = int(round(metal * 255))
    return o


def _solid(out: Path, name: str, textures: dict, key: str, rgb, rough: float, metal: float) -> None:
    rel = f"textures/T_{name}_{key}_d.png"
    facades.save_png(out / rel, np.full((4, 4, 3), rgb, dtype=np.uint8))
    nrel = f"textures/T_{name}_flat_n.png"
    facades.save_png(out / nrel, facades.flat_normal())
    orel = f"textures/T_{name}_{key}_orm.png"
    facades.save_png(out / orel, _orm(rough, metal))
    textures[key] = {"base_color": rel, "normal": nrel, "orm": orel, "provenance": PROVENANCE}


GLASS_STYLE = facades.Style("landmark_glass", 1.8, 3.6, "glass", ((60, 76, 88),),
                            window=(0.02, 0.98, 0.04, 0.98), frame=(150, 155, 160), frame_m=0.05,
                            sill=False, glass=(52, 70, 84))


def glass_textures(bay_m: float, storey_m: float, seed: int = 0, px: int = 1024):
    """Albedo, normal and ORM of a curtain wall texture tile (TEX_BAYS x TEX_STOREYS panels, row 0 = top):
    per panel a slightly different blue-grey tint, a roughness of 0.03-0.09, a tilt of up to ~1.2 degrees
    (the normal), lowered blinds behind about a quarter of the panels, and a thin frame line."""
    rng = np.random.default_rng(seed)
    nb, ns = facades.TEX_BAYS, facades.TEX_STOREYS
    pw, ph = px // nb, px // ns
    alb = np.zeros((px, px, 3))
    rough = np.zeros((px, px))
    nrm = np.zeros((px, px, 3))
    yy = np.linspace(0.0, 1.0, ph)[:, None]
    for sy in range(ns):
        for bx in range(nb):
            tint = np.array([44, 60, 74]) + rng.normal(0, 3.0, 3) + rng.normal(0, 2.0)
            cell = np.broadcast_to(tint, (ph, pw, 3)).copy()
            cell += (1.0 - yy[..., None]) * 6.0          # a little lighter towards the head (sky)
            if rng.random() < 0.12:                        # blinds lowered part way
                drop = int(ph * rng.uniform(0.15, 0.7))
                cell[:drop] = cell[:drop] * 0.7 + np.array([140, 140, 136]) * 0.3
            fpx = max(2, int(0.05 / bay_m * pw))
            cell[:fpx] = cell[-fpx:] = 150
            cell[:, :fpx] = cell[:, -fpx:] = 150
            y0, x0 = sy * ph, bx * pw
            alb[y0:y0 + ph, x0:x0 + pw] = cell
            rough[y0:y0 + ph, x0:x0 + pw] = rng.uniform(0.03, 0.09)
            tx, ty = rng.normal(0, 0.012, 2)
            n = np.array([tx, ty, 1.0])
            nrm[y0:y0 + ph, x0:x0 + pw] = n / np.linalg.norm(n)
            rough[y0:y0 + fpx, x0:x0 + pw] = rough[y0 + ph - fpx:y0 + ph, x0:x0 + pw] = 0.35
    alb_u8 = np.clip(np.round(alb), 0, 255).astype(np.uint8)
    nrm_u8 = np.clip(np.round((nrm * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    orm = _orm(rough, 0.5, px=px)
    return alb_u8, nrm_u8, orm


# --------------------------------------------------------------------------- mesh building

class Mesh:
    def __init__(self):
        self.v, self.n, self.uv, self.f = [], [], [], []
        self.count = 0

    def add(self, v, n, uv, f):
        v = np.asarray(v, float)
        f = np.asarray(f, np.int64).reshape(-1, 3)
        if np.ndim(n) == 1:   # one normal for all: wind every triangle CCW seen from that side
            nv = np.asarray(n, float)
            cr = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            flip = cr @ nv < 0
            f = f.copy()
            f[flip] = f[flip][:, ::-1]
            n = np.broadcast_to(nv, v.shape).copy()
        self.v.append(v)
        self.n.append(np.asarray(n, float))
        self.uv.append(np.asarray(uv, float))
        self.f.append(f + self.count)
        self.count += len(v)

    def quad(self, a, b, c, d, n, uv=None):
        """Quad a-b-c-d, CCW seen from the side ``n`` points to."""
        q = np.array([a, b, c, d], float)
        if np.dot(np.cross(q[1] - q[0], q[2] - q[0]), n) < 0:
            q = q[::-1]
        if uv is None:
            uv = _planar_uv(q, n)
        self.add(q, n, uv, [[0, 1, 2], [0, 2, 3]])

    def box(self, centre, ax, ay, az, hx, hy, hz):
        """Box with half sizes hx, hy, hz along the orthonormal axes ax, ay, az."""
        c, ax, ay, az = (np.asarray(t, float) for t in (centre, ax, ay, az))
        for s in (-1, 1):
            for (u, w, h, a, b) in ((ax, ay, az, hx, (hy, hz)), (ay, az, ax, hy, (hz, hx)), (az, ax, ay, hz, (hx, hy))):
                p = c + s * a * u
                e1, e2 = w * b[0], h * b[1]
                self.quad(p - e1 - e2, p + e1 - e2, p + e1 + e2, p - e1 + e2, s * u)

    def arrays(self):
        if not self.v:
            return None
        return (np.concatenate(self.v), np.concatenate(self.n), np.concatenate(self.uv), np.concatenate(self.f))


def _planar_uv(q, n):
    n = np.asarray(n, float)
    if abs(n[2]) > 0.7:
        return q[:, :2]
    t = np.array([-n[1], n[0], 0.0])
    t /= max(np.linalg.norm(t), 1e-9)
    return np.column_stack([q @ t, -q[:, 2]])


def _triangulate(poly: Polygon):
    rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
    verts = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = earcut.triangulate_float64(verts, ends).reshape(-1, 3)
    return verts, tri


def _polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def extrude(poly2d: Polygon, origin, ax, ay, depth, mesh: Mesh):
    """Extrude a polygon given in the plane (origin, ax, ay) by ``depth`` along ax x ay (towards the front)."""
    o, ax, ay = (np.asarray(t, float) for t in (origin, ax, ay))
    nz = np.cross(ax, ay)
    for p in _polys(poly2d.buffer(0)):
        p = orient(p, 1.0)   # exterior CCW, holes CW: edge x normal points away from the solid
        verts, tri = _triangulate(p)
        to3 = lambda uv, off: o + np.outer(uv[:, 0], ax) + np.outer(uv[:, 1], ay) + off * nz  # noqa: E731
        front = to3(verts, depth)
        back = to3(verts, 0.0)
        mesh.add(front, nz, verts, tri)
        mesh.add(back, -nz, verts, tri[:, ::-1])
        for ring in [p.exterior] + list(p.interiors):
            c = np.asarray(ring.coords)
            for i in range(len(c) - 1):
                a, b = c[i:i + 1], c[i + 1:i + 2]
                pa, pb = to3(a, 0.0)[0], to3(b, 0.0)[0]
                e = pb - pa
                n = np.cross(e, nz)
                n /= max(np.linalg.norm(n), 1e-9)
                mesh.quad(pa, pb, pb + depth * nz, pa + depth * nz, n)


# --------------------------------------------------------------------------- the TU/e logo

def logo_tue() -> Polygon:
    """The TU/e word mark as polygons, cap height 1 (x right, y up): T, U, a slash, a lowercase e."""
    st = 0.19
    t = box(0.0, 1.0 - st, 0.64, 1.0).union(box(0.32 - st / 2, 0.0, 0.32 + st / 2, 1.0))
    ux0, ux1, r = 0.74, 1.40, 0.33
    cx, cy = (ux0 + ux1) / 2, r
    lower = box(ux0 - 1, -1, ux1 + 1, cy)
    u_outer = box(ux0, cy, ux1, 1.0).union(Point(cx, cy).buffer(r, 64).intersection(lower))
    u_inner = box(ux0 + st, cy, ux1 - st, 1.01).union(Point(cx, cy).buffer(r - st, 64).intersection(lower))
    u = u_outer.difference(u_inner)
    slash = Polygon([(1.46, 0.0), (1.64, 0.0), (1.98, 1.0), (1.80, 1.0)])
    ex, R = 2.38, 0.33
    ecx, ecy = ex, R
    ring = Point(ecx, ecy).buffer(R, 64).difference(Point(ecx, ecy).buffer(R - 0.16, 64))
    bar = box(ecx - R + 0.05, ecy - 0.03, ecx + R - 0.05, ecy + 0.07)
    gap = Polygon([(ecx, ecy - 0.03), (ecx + R + 0.1, ecy - 0.03), (ecx + R + 0.1, ecy - 0.24)])
    e = ring.union(bar).difference(gap)
    return unary_union([t, u, slash, e])


LOGOS = {"TU/e": logo_tue}


# --------------------------------------------------------------------------- geometry helpers

def wall_planes(v, f, nrm):
    """Group wall triangles (model coords) into planes: list of (n_xy, d, polygon in (s, z))."""
    planes = []
    for k, tri in enumerate(f):
        n = nrm[k].copy()
        n[2] = 0.0
        ln = np.linalg.norm(n)
        if ln < 1e-6:
            continue
        n /= ln
        p = v[tri]
        d = float(np.mean(p[:, :2] @ n[:2]))
        for pl in planes:
            if np.dot(pl["n"], n) > 0.999 and abs(pl["d"] - d) < 0.25:
                pl["tris"].append(p)
                break
        else:
            planes.append({"n": n, "d": d, "tris": [p]})
    out = []
    for pl in planes:
        n = pl["n"]
        t = np.array([-n[1], n[0], 0.0])
        polys = []
        for p in pl["tris"]:
            sz = np.column_stack([p @ t, p[:, 2]])
            q = Polygon(sz)
            if q.area > 1e-4:
                polys.append(q.buffer(0.01, join_style=2))
        u = unary_union(polys).buffer(-0.01, join_style=2) if polys else Polygon()
        for q in _polys(u):
            q = q.simplify(0.02)
            if q.area > 0.5:
                out.append((n, pl["d"], q))
    return out


def _to3(n, d, s, z):
    t = np.array([-n[1], n[0], 0.0])
    base = n * d
    return base[None, :] + np.outer(s, t) + np.outer(z, [0, 0, 1.0])


def curtain_wall(planes, base_z, bay, storey, fins_min_z, bid, meshes):
    """Fill wall planes with glass, mullions, spandrels and fins into meshes[glass|alu|spandrel|fin]."""
    stats = {"panels_m2": 0.0, "mullions": 0, "spandrels": 0, "fins": 0}
    up = np.array([0, 0, 1.0])
    for w, (n, d, poly) in enumerate(planes):
        t = np.array([-n[1], n[0], 0.0])
        ou, ov = facades.light_offset(bid, w)
        # glass on the wall plane
        verts, tri = _triangulate(poly)
        p3 = _to3(n, d, verts[:, 0], verts[:, 1])
        uv = np.column_stack([verts[:, 0] / bay / facades.TEX_BAYS + ou,
                              -(verts[:, 1] - base_z) / storey / facades.TEX_STOREYS + ov])
        meshes["glass"].add(p3, n, uv, tri)
        stats["panels_m2"] += poly.area
        s0, z0, s1, z1 = poly.bounds
        if s1 - s0 < 1.5:     # a column facet or a sliver: glass only (bands and fins on it float in the air)
            continue
        # mullions every bay: 7 cm wide, 18 cm deep
        k0 = math.ceil((s0 + 0.15) / bay)
        for k in range(k0, int(math.floor((s1 - 0.15) / bay)) + 1):
            s = k * bay
            seg = poly.intersection(LineString([(s, z0 - 1), (s, z1 + 1)]))
            for g in getattr(seg, "geoms", [seg]):
                if g.is_empty or g.length < 0.5:
                    continue
                za, zb = g.bounds[1], g.bounds[3]
                c = _to3(n, d, [s], [(za + zb) / 2])[0] + n * 0.09
                meshes["alu"].box(c, t, up, n, 0.035, (zb - za) / 2, 0.09)
                stats["mullions"] += 1
                tall = zb >= fins_min_z
                if tall:
                    # double-height zones (two storeys by six bays) get fins on their mullions
                    for j in range(int(math.floor((za - base_z) / storey)), int(math.ceil((zb - base_z) / storey))):
                        if j < 1 or j % 2:
                            continue
                        zone = zlib.crc32(f"{bid}-{w}-{j}-{k // 6}".encode()) % 5
                        if zone:
                            continue
                        fz0, fz1 = max(za, base_z + j * storey), min(zb, base_z + (j + 2) * storey)
                        if fz1 - fz0 < 2.0:
                            continue
                        cf = _to3(n, d, [s], [(fz0 + fz1) / 2])[0] + n * (0.18 + 0.22)
                        meshes["fin"].box(cf, t, up, n, 0.0125, (fz1 - fz0) / 2 - 0.05, 0.22)
                        stats["fins"] += 1
        # floor-edge spandrels every storey: 35 cm high, 6 cm proud
        j0 = max(1, math.ceil((z0 + 0.3 - base_z) / storey))
        for j in range(j0, int(math.floor((z1 - 0.3 - base_z) / storey)) + 1):
            z = base_z + j * storey
            seg = poly.intersection(LineString([(s0 - 1, z), (s1 + 1, z)]))
            for g in getattr(seg, "geoms", [seg]):
                if g.is_empty or g.length < 0.5:
                    continue
                sa, sb = g.bounds[0], g.bounds[2]
                c = _to3(n, d, [(sa + sb) / 2], [z - 0.12])[0] + n * 0.03
                meshes["spandrel"].box(c, t, up, n, (sb - sa) / 2, 0.175, 0.03)
                stats["spandrels"] += 1
    stats["panels_m2"] = round(stats["panels_m2"], 1)
    return stats


def roof_levels(v, f, nrm, min_area=40.0, gap=0.6):
    """Flat roof parts: list of (polygon xy, z). Roof triangles are sorted by height and split where two
    consecutive heights differ by more than ``gap`` (3DBAG roofs come in slivers a few cm apart), then the
    triangles of each height cluster are unioned."""
    items = []
    for k, tri in enumerate(f):
        if abs(nrm[k][2]) < 0.7:
            continue
        p = v[tri]
        q = Polygon(p[:, :2])
        if q.area > 1e-4:
            items.append((float(p[:, 2].mean()), q))
    items.sort(key=lambda t: t[0])
    clusters, cur = [], []
    for z, q in items:
        if cur and z - cur[-1][0] > gap:
            clusters.append(cur)
            cur = []
        cur.append((z, q))
    if cur:
        clusters.append(cur)
    out = []
    for cl in clusters:
        u = unary_union([q.buffer(0.05, join_style=2) for _, q in cl]).buffer(-0.05, join_style=2)
        z = float(np.median([z for z, _ in cl]))
        for q in _polys(u):
            q = q.simplify(0.05)
            if q.area >= min_area:
                out.append((q, z))
    return out


def coping(levels, mesh, height=0.9, thick=0.25, min_area=100.0):
    for poly, z in levels:
        if poly.area < min_area:
            continue
        ring = np.asarray(poly.exterior.coords)
        for i in range(len(ring) - 1):
            a, b = ring[i], ring[i + 1]
            e = b - a
            ln = float(np.linalg.norm(e))
            if ln < 0.3:
                continue
            ax = np.array([e[0] / ln, e[1] / ln, 0.0])
            nout = np.array([ax[1], -ax[0], 0.0])
            if not poly.exterior.is_ccw:
                nout = -nout
            c = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z + height / 2]) - nout * thick / 2
            mesh.box(c, ax, np.array([0, 0, 1.0]), nout, ln / 2, height / 2, thick / 2)


# --------------------------------------------------------------------------- passages

def patch_xodr(src: Path, dst: Path, spec: dict = TUE_CAMPUS) -> list:
    """Level and join the roads of each passage in an OpenDRIVE file: road ``road`` becomes a straight line from its
    start to the start of road ``to``, flat at ``z``, linked to it lane by lane."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(src)
    root = tree.getroot()
    roads = {r.get("id"): r for r in root.findall("road")}
    done = []
    for lm, s in spec.items():
        if s["type"] != "passage":
            continue
        a, b = roads[s["road"]], roads[s["to"]]
        ga = a.find("planView").findall("geometry")[0]
        gb = b.find("planView").findall("geometry")[0]
        x0, y0 = float(ga.get("x")), float(ga.get("y"))
        x1, y1 = float(gb.get("x")), float(gb.get("y"))
        length = math.hypot(x1 - x0, y1 - y0)
        pv = a.find("planView")
        for g in list(pv):
            pv.remove(g)
        g = ET.SubElement(pv, "geometry", s="0", x=f"{x0:.7f}", y=f"{y0:.7f}",
                          hdg=f"{math.atan2(y1 - y0, x1 - x0):.12f}", length=f"{length:.9f}")
        ET.SubElement(g, "line")
        a.set("length", f"{length:.9f}")
        ep = a.find("elevationProfile")
        for e in list(ep):
            ep.remove(e)
        zb = float(b.find("elevationProfile").findall("elevation")[0].get("a"))
        ET.SubElement(ep, "elevation", s="0", a=f"{s['z']:.6f}", b=f"{(zb - s['z']) / length:.9f}", c="0", d="0")
        for road, kind, other, cp in ((a, "successor", s["to"], "start"), (b, "predecessor", s["road"], "end")):
            link = road.find("link")
            old = link.find(kind)
            if old is not None:
                link.remove(old)
            el = ET.SubElement(link, kind, elementType="road", elementId=other, contactPoint=cp)
            if kind == "predecessor":          # predecessor first, as the schema orders them
                link.remove(el)
                link.insert(0, el)
            sec = road.find("lanes").findall("laneSection")[-1 if kind == "successor" else 0]
            for lane in sec.iter("lane"):
                if lane.get("id") == "0":
                    continue
                ll = lane.find("link")
                for o in ll.findall(kind):
                    ll.remove(o)
                ET.SubElement(ll, kind, id=lane.get("id"))
        done.append({"road": s["road"], "to": s["to"], "length_m": round(length, 2), "z": s["z"]})
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return done


def _fit_uv(v, uv):
    a = np.column_stack([v[:, 0], v[:, 1], np.ones(len(v))])
    coef, *_ = np.linalg.lstsq(a, uv, rcond=None)
    return coef


def _append(ue: Path, man: dict, kind: str, layer: int, x: float, y: float, verts, faces, normals):
    """Append triangles (model coords) to the export's asset of that kind and layer on the tile at (x, y)."""
    tile = list(_tile_index(x, y, man["tile_m"]))
    a = next(e for e in man["assets"] if e["kind"] == kind and e["layer"] == layer and list(e["tile"]) == tile)
    _backup(ue, a["file"])
    g = read_glb(ue / a["file"])
    v = gltf_to_model(g["positions"].astype(float))
    nn = gltf_to_model(g["normals"].astype(float))
    uv = g["uvs"].astype(float)
    coef = _fit_uv(v, uv)
    verts = np.asarray(verts, float)
    nuv = np.column_stack([verts[:, 0], verts[:, 1], np.ones(len(verts))]) @ coef
    f = np.concatenate([g["faces"].astype(np.int64), np.asarray(faces, np.int64) + len(v)])
    bbox = write_glb(ue / a["file"], a["asset"], np.concatenate([v, verts]), np.concatenate([nn, normals]),
                     np.concatenate([uv, nuv]), f, g["material"])
    a.update(vertices=int(len(v) + len(verts)), triangles=int(len(f)), bbox_ue=bbox)


def _drop(ue: Path, man: dict, layer: int, bx) -> int:
    """Remove the triangles of a layer whose centre lies in the box (x0, x1, y0, y1)."""
    n = 0
    for a in man["assets"]:
        if a["layer"] != layer or a["kind"] in WALL_KINDS + ("roof", KIND):
            continue
        g = read_glb(ue / a["file"])
        v = gltf_to_model(g["positions"].astype(float))
        f = g["faces"].astype(np.int64)
        c = v[f].mean(axis=1)
        m = (c[:, 0] > bx[0]) & (c[:, 0] < bx[1]) & (c[:, 1] > bx[2]) & (c[:, 1] < bx[3])
        if not m.any():
            continue
        _backup(ue, a["file"])
        rest = f[~m]
        n += int(m.sum())
        if len(rest) == 0:
            rest = np.zeros((1, 3), np.int64)   # one degenerate triangle keeps the asset valid (a kept real one
            #                                     showed: the sunk road's last triangle dived 6 m in the passage)
        bbox = write_glb(ue / a["file"], a["asset"], v, gltf_to_model(g["normals"].astype(float)),
                         g["uvs"].astype(float), rest, g["material"])
        a.update(triangles=int(len(rest)), bbox_ue=bbox)
    return n


def _top_z(ue: Path, man: dict, x: float, y: float, kinds=("drivable", "crossing", "parking")):
    """The highest road surface (layer 0) at (x, y) in the export, or None."""
    best = None
    tile = list(_tile_index(x, y, man["tile_m"]))
    for a in man["assets"]:
        if a["kind"] not in kinds or a["layer"] != 0 or list(a["tile"]) != tile:
            continue
        g = read_glb(ue / a["file"])
        v = gltf_to_model(g["positions"].astype(float))
        for tri in g["faces"]:
            p = v[tri]
            (x1, y1), (x2, y2), (x3, y3) = p[0, :2], p[1, :2], p[2, :2]
            den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if abs(den) < 1e-12:
                continue
            l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / den
            l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / den
            l3 = 1 - l1 - l2
            if min(l1, l2, l3) >= -1e-6:
                zz = l1 * p[0, 2] + l2 * p[1, 2] + l3 * p[2, 2]
                best = zz if best is None else max(best, zz)
    return best


def passage(ue: Path, man: dict, s: dict) -> dict:
    """Pave a passage at street level (drivable lanes, raised sidewalks with kerbs) after dropping the wrongly
    sunk layer. Returns the clearance area (polygon, z of the opening's top)."""
    dropped = _drop(ue, man, s["drop"]["layer"], s["drop"]["box"])
    fx0, fx1 = s["fill"]["x"]
    yc, hd, wk, kb, z = s["fill"]["axis_y"], s["fill"]["drive_half"], s["fill"]["walk_m"], s["fill"]["kerb_m"], s["z"]
    up = np.array([0, 0, 1.0])
    cx, cy = (fx0 + fx1) / 2, yc

    # the fill reaches the street's own surface at both ends (a 1.6 m slot was left where the street mesh ended
    # short of the given x, 2026-11-23), at the street's height there so the joint has no step (a 2 cm step at
    # the crossing shook the film camera)
    def street_edge(x, step):
        for k in range(0, 41):                      # up to 8 m out
            zz = _top_z(ue, man, x + step * k * 0.2, yc)
            if zz is not None:
                return x + step * k * 0.2, zz
        return x, z
    fx0, z_w = street_edge(fx0, -1.0)
    fx1, z_e = street_edge(fx1, 1.0)

    def quad(y0, y1, dz):
        return ([[fx0, y0, z_w + dz], [fx1, y0, z_e + dz], [fx1, y1, z_e + dz], [fx0, y1, z_w + dz]],
                [[0, 1, 2], [0, 2, 3]])
    v, f = quad(yc - hd, yc + hd, 0.002)
    _append(ue, man, "drivable", 0, cx, cy, v, f, np.tile(up, (4, 1)))
    vs, fs, ns = [], [], []
    for (y0, y1) in ((yc - hd - wk, yc - hd), (yc + hd, yc + hd + wk)):
        q, qf = quad(y0, y1, kb)
        fs += [[i + len(vs) for i in tri] for tri in qf]
        vs += q
        ns += [up] * 4
    _append(ue, man, "sidewalk", 0, cx, cy, vs, fs, np.array(ns))
    rv, rf, rn = [], [], []
    for yk, sgn in ((yc - hd, 1.0), (yc + hd, -1.0)):    # kerb faces towards the road
        q = [[fx0, yk, z_w], [fx1, yk, z_e], [fx1, yk, z_e + kb], [fx0, yk, z_w + kb]]
        nrm = np.array([0, sgn, 0.0])
        tri = [[0, 1, 2], [0, 2, 3]] if sgn < 0 else [[0, 2, 1], [0, 3, 2]]
        rf += [[i + len(rv) for i in t3] for t3 in tri]
        rv += q
        rn += [nrm] * 4
    _append(ue, man, "riser", 0, cx, cy, rv, rf, np.array(rn))
    area = box(fx0 - 1.0, yc - hd - wk, fx1 + 1.0, yc + hd + wk)
    return {"dropped_triangles": dropped, "area": area, "z_top": max(z_w, z_e) + s["clearance_m"],
            "fill": {"x0": fx0, "x1": fx1, "yc": yc, "hd": hd, "wk": wk, "kb": kb, "z_w": z_w, "z_e": z_e},
            "report": {"dropped_triangles": dropped, "paved_m2": round(area.area, 1),
                       "z_west": round(z_w, 3), "z_east": round(z_e, 3)}}


def passage_fronts(meshes, x0, x1, fill, ztop, bay, storey):
    """Entrance fronts on both sides of a passage under a building, facing into it: glazing from the pavement to a
    dark fascia under the soffit, aluminium mullions every bay, a transom at door height and two sliding-door frames
    per side. Without them the passage shows the building's empty inside through the backs of its outer walls."""
    yc, hd, wk, kb = fill["yc"], fill["hd"], fill["wk"], fill["kb"]
    up = np.array([0, 0, 1.0])
    span = x1 - x0
    if span < 3.0:
        return

    def zfoot(x):        # the pavement's top along the passage (the fill slopes from z_w to z_e)
        u = (x - fill["x0"]) / max(fill["x1"] - fill["x0"], 1e-6)
        return fill["z_w"] + (fill["z_e"] - fill["z_w"]) * u + kb
    for y, n in ((yc - hd - wk, np.array([0, 1.0, 0])), (yc + hd + wk, np.array([0, -1.0, 0]))):
        t = np.array([1.0, 0, 0])
        za, zb = zfoot(x0), zfoot(x1)
        zg = ztop - 0.7                                          # glazing top, the fascia above it
        quad = np.array([[x0, y, za], [x1, y, zb], [x1, y, zg], [x0, y, zg]])
        uv = np.column_stack([(quad[:, 0] - x0) / bay / facades.TEX_BAYS,
                              -(quad[:, 2] - za) / storey / facades.TEX_STOREYS])
        meshes["glass"].add(quad, n, uv, [[0, 1, 2], [0, 2, 3]])
        zf = (zg + ztop) / 2
        meshes["spandrel"].box(np.array([(x0 + x1) / 2, y, zf]) + n * 0.04, t, up, n, span / 2, 0.35, 0.04)
        zm = (za + zb) / 2
        for k in range(1, int(span / bay) + 1):              # mullions
            x = x0 + k * bay
            if x > x1 - 0.2:
                break
            meshes["alu"].box(np.array([x, y, (zfoot(x) + zg) / 2]) + n * 0.06, t, up, n, 0.035, (zg - zfoot(x)) / 2, 0.06)
        meshes["alu"].box(np.array([(x0 + x1) / 2, y, zm + 2.6]) + n * 0.05, t, up, n, span / 2, 0.04, 0.05)  # transom
        for frac in (1 / 3, 2 / 3):                          # sliding doors: posts, head and the meeting stile
            xc = x0 + span * frac
            zd = zfoot(xc)
            for dx in (-1.2, 1.2):
                meshes["alu"].box(np.array([xc + dx, y, zd + 1.3]) + n * 0.08, t, up, n, 0.06, 1.3, 0.08)
            meshes["alu"].box(np.array([xc, y, zd + 1.3]) + n * 0.07, t, up, n, 0.025, 1.3, 0.07)
            meshes["alu"].box(np.array([xc, y, zd + 2.62]) + n * 0.08, t, up, n, 1.26, 0.06, 0.08)


def clip_planes(planes, clear: list):
    """Cut each wall plane polygon away below a clearance area's z_top where the wall's foot crosses it."""
    out = []
    for n, d, poly in planes:
        t = np.array([-n[1], n[0], 0.0])
        s0, z0, s1, z1 = poly.bounds
        q = poly
        for area, ztop in clear:
            a = n[:2] * d + t[:2] * (s0 - 1)
            b = n[:2] * d + t[:2] * (s1 + 1)
            seg = LineString([tuple(a), tuple(b)]).intersection(area.buffer(0.3))
            for g in getattr(seg, "geoms", [seg]):
                if g.is_empty or g.length < 0.05:
                    continue
                ss = [float(np.dot(np.array(c), t[:2])) for c in g.coords]
                q = q.difference(box(min(ss), z0 - 1.0, max(ss), ztop))
        for p in _polys(q):
            if p.area > 0.3:
                out.append((n, d, p))
    return out


# --------------------------------------------------------------------------- export plumbing

def _backup(ue: Path, rel: str):
    src, dst = ue / rel, ue / BACKUP / rel
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def restore(ue: Path) -> bool:
    bk = ue / BACKUP
    if not bk.exists():
        return False
    man = json.loads((ue / "manifest.json").read_text())
    for rel in (man.get("landmarks") or {}).get("files", []):
        (ue / rel).unlink(missing_ok=True)
    for p in bk.rglob("*"):
        if p.is_file():
            dst = ue / p.relative_to(bk)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    return True


def _write_asset(ue, man, name, lm, part, key, tile_m, arrays, texture_key, extra=None):
    v, n, uv, f = arrays
    c = v.mean(axis=0)
    tile = _tile_index(float(c[0]), float(c[1]), tile_m)
    asset = f"{name}_LM_{lm}_{part}".replace("/", "_")
    rel = f"meshes/landmarks/{asset}.glb"
    bbox = write_glb(ue / rel, asset, v, n, uv, f, key)
    entry = {"file": rel, "asset": asset, "layer": 0, "kind": KIND, "material": "building",
             "semantic": "Building", "tile": list(tile), "vertices": int(len(v)), "triangles": int(len(f)),
             "bbox_ue": bbox, "texture": texture_key, "landmark": lm, "provenance": PROVENANCE}
    if extra:
        entry.update(extra)
    man["assets"].append(entry)
    return rel


def cut_building(ue: Path, man: dict, bid: str):
    """Remove one building's wall triangles from the shared wall meshes; return them (model coords)."""
    b = next(x for x in man["buildings"] if x["id"] == bid)
    poly = Polygon([(x / 100.0, -y / 100.0) for x, y in b["rings_ue"][0]]).buffer(0.3)
    walls_v, walls_n = [], []
    roofs_v, roofs_n = [], []
    removed = {}
    keep_assets = []
    for a in man["assets"]:
        if a["kind"] not in WALL_KINDS + ("roof",):
            keep_assets.append(a)
            continue
        g = read_glb(ue / a["file"])
        v = gltf_to_model(g["positions"].astype(float))
        nn = gltf_to_model(g["normals"].astype(float))
        f = g["faces"].astype(np.int64)
        cen = v[f].mean(axis=1)
        inside = contains_xy(poly, cen[:, 0], cen[:, 1])
        if not inside.any():
            keep_assets.append(a)
            continue
        fn = nn[f[:, 0]]
        if a["kind"] == "roof":
            roofs_v.append(v[f[inside]])
            roofs_n.append(fn[inside])
            keep_assets.append(a)
            continue
        walls_v.append(v[f[inside]])
        walls_n.append(fn[inside])
        removed[a["asset"]] = int(inside.sum())
        _backup(ue, a["file"])
        rest = f[~inside]
        if len(rest) == 0:
            continue          # nothing left of this asset: drop it from the manifest
        used = np.unique(rest)
        remap = -np.ones(len(v), np.int64)
        remap[used] = np.arange(len(used))
        bbox = write_glb(ue / a["file"], a["asset"], v[used], nn[used], g["uvs"][used].astype(float),
                         remap[rest], g["material"])
        a.update(vertices=int(len(used)), triangles=int(len(rest)), bbox_ue=bbox)
        keep_assets.append(a)
    man["assets"] = keep_assets

    def pack(vs, ns):
        if not vs:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3))
        tv = np.concatenate(vs)
        return tv.reshape(-1, 3), np.arange(len(tv) * 3).reshape(-1, 3), np.concatenate(ns)
    return b, pack(walls_v, walls_n), pack(roofs_v, roofs_n), removed


def apply(ue: Path, spec: dict | None = None, facing: str | None = None) -> dict:
    ue = Path(ue)
    restore(ue)
    _backup(ue, "manifest.json")
    man = json.loads((ue / "manifest.json").read_text())
    spec = spec if spec is not None else spec_for(man["name"])
    name, tile_m = man["name"], man["tile_m"]
    textures = man["textures"]
    files, report, clear = [], {}, []

    # materials shared by all landmarks
    _solid(ue, name, textures, "lm_alu", (172, 176, 180), 0.3, 0.9)
    _solid(ue, name, textures, "lm_spandrel", (54, 58, 64), 0.35, 0.3)
    _solid(ue, name, textures, "lm_fin", (150, 172, 174), 0.05, 0.4)
    _solid(ue, name, textures, "lm_red", (199, 33, 37), 0.45, 0.0)     # TU/e red #C72125
    _solid(ue, name, textures, "lm_steel", (62, 64, 68), 0.5, 0.7)
    _solid(ue, name, textures, "lm_soffit", (186, 184, 178), 0.8, 0.0)
    _solid(ue, name, textures, "lm_core", (168, 166, 160), 0.85, 0.0)       # exposed concrete
    _solid(ue, name, textures, "lm_steps", (150, 148, 142), 0.9, 0.0)
    for k in ("lm_alu", "lm_spandrel", "lm_fin", "lm_red", "lm_steel", "lm_soffit", "lm_core", "lm_steps"):
        files += [textures[k]["base_color"], textures[k]["orm"]]

    for lm, s in spec.items():
        if s["type"] == "osm_detail":
            import osm_detail
            parts = {k: Mesh() for k in ("soffit", "core", "steps", "rail", "door_glass", "door_frame")}
            walls = {}

            def append_wall(b, a2, b2, n, za, zb, ztop, walls=walls):
                """A wall of a building's own facade (its 3DBAG style), from the ground to ztop, facing n."""
                style = b.get("style") or "office_glass"
                key = next((k for k in textures if k.startswith(f"facade_{style}_")), None) or \
                    next(k for k in textures if k.startswith("facade_"))
                tex = textures[key]
                bay, storey = tex.get("bay_m", 1.8), tex.get("storey_m", 3.6)
                mid = (np.asarray(a2) + np.asarray(b2)) / 2
                tile = _tile_index(float(mid[0]), float(mid[1]), tile_m)
                m = walls.setdefault((tile, key), Mesh())
                q = np.array([[a2[0], a2[1], za], [b2[0], b2[1], zb], [b2[0], b2[1], ztop], [a2[0], a2[1], ztop]])
                tdir = np.array([-n[1], n[0], 0.0])
                base = b["base_z_cm"] / 100.0
                uv = np.column_stack([q @ tdir / bay / facades.TEX_BAYS,
                                      -(q[:, 2] - base) / storey / facades.TEX_STOREYS])
                m.add(q, np.array([n[0], n[1], 0.0]), uv, [[0, 1, 2], [0, 2, 3]])
            rep = osm_detail.apply(ue, man, s, parts, append_wall)
            keys = {"soffit": "lm_soffit", "core": "lm_core", "steps": "lm_steps", "rail": "lm_alu",
                    "door_glass": "lm_fin", "door_frame": "lm_alu"}
            for part, mesh in parts.items():
                arr = mesh.arrays()
                if arr is not None:
                    files.append(_write_asset(ue, man, name, lm, part, keys[part], tile_m, arr, keys[part]))
            for (tile, key), mesh in walls.items():
                arr = mesh.arrays()
                if arr is None:
                    continue
                files.append(_write_asset(ue, man, name, lm, f"wall_{tile[0]}_{tile[1]}_{key}".replace("-", "m"), key,
                                          tile_m, arr, key, extra={"kind": "building_3dbag", "tile": list(tile)}))
            report[lm] = rep
        elif s["type"] == "passage":
            pr = passage(ue, man, s)
            clear.append((pr["area"], pr["z_top"], s.get("soffit_under"), pr["fill"]))
            report[lm] = pr["report"]
        elif s["type"] == "curtain_wall":
            b, (wv, wf, wn), (rv, rf, rn), removed = cut_building(ue, man, s["building"])
            base_z = b["base_z_cm"] / 100.0
            planes = clip_planes(wall_planes(wv, wf, wn), [(a, z) for a, z, _, _ in clear])
            meshes = {k: Mesh() for k in ("glass", "alu", "spandrel", "fin")}
            st = curtain_wall(planes, base_z, s["bay_m"], s["storey_m"], base_z + s["fins_min_height_m"],
                              b["id"], meshes)
            levels = roof_levels(rv, rf, rn)
            coping(levels, meshes["alu"])
            # the underside of the building over a passage (a concrete soffit at the opening's top)
            # glass texture with the night mask
            gkey = f"lm_{lm}_glass"
            alb, nrm, orm = glass_textures(s["bay_m"], s["storey_m"], seed=zlib.crc32(lm.encode()))
            ent = {}
            for suf, arr, fld in (("d", alb, "base_color"), ("n", nrm, "normal"), ("orm", orm, "orm")):
                rel = f"textures/T_{name}_{gkey}_{suf}.png"
                facades.save_png(ue / rel, arr)
                ent[fld] = rel
                files.append(rel)
            style = facades.Style("landmark_glass", s["bay_m"], s["storey_m"], "glass", ((60, 76, 88),),
                                  window=GLASS_STYLE.window, frame=GLASS_STYLE.frame, frame_m=GLASS_STYLE.frame_m,
                                  sill=False, glass=GLASS_STYLE.glass)
            lights = facades.synth_window_lights(style, 0)
            rel = f"textures/T_{name}_{gkey}_w.png"
            facades.save_png(ue / rel, lights)
            files.append(rel)
            ent.update(windows=rel, use="office", windows_uv_scale=facades.MASK_UV_SCALE, style="landmark_glass",
                       variant=0, bay_m=s["bay_m"], storey_m=s["storey_m"], provenance=PROVENANCE)
            textures[gkey] = ent
            keys = {"glass": gkey, "alu": "lm_alu", "spandrel": "lm_spandrel", "fin": "lm_fin", "soffit": "lm_soffit"}
            for part, mesh in meshes.items():
                arr = mesh.arrays()
                if arr is not None:
                    files.append(_write_asset(ue, man, name, lm, part, keys[part], tile_m, arr, keys[part]))
            b["landmark"] = lm
            st.update(walls=len(planes), roof_levels=len(levels), removed_triangles=removed)
            report[lm] = st
        elif s["type"] == "roof_letters":
            b = next(x for x in man["buildings"] if x["id"] == s["building"])
            _, _, (rv, rf, rn), _ = cut_building_readonly(ue, man, s["building"])
            levels = roof_levels(rv, rf, rn, min_area=200.0)
            zmax = max(z for _, z in levels)
            poly, z = max((pz for pz in levels if pz[1] > zmax - 6.0), key=lambda pz: pz[0].area)
            poly = max(_polys(poly.buffer(-0.6, join_style=2).buffer(0.6, join_style=2)), key=lambda p: p.area)  # no slivers
            mesh_red, mesh_steel = Mesh(), Mesh()
            facing_dir = facing or s.get("facing", "north")
            fwd = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}[facing_dir]
            fwd = np.array([fwd[0], fwd[1], 0.0])
            right = np.array([fwd[1], -fwd[0], 0.0]) * -1   # viewer facing -fwd: their right
            # the edge of the roof part on the facing side
            cx, cy = poly.centroid.x, poly.centroid.y
            proj = np.asarray(poly.exterior.coords) @ fwd[:2]
            edge = float(proj.max())
            centre = np.array([cx, cy, z]) + fwd * (edge - (np.array([cx, cy]) @ fwd[:2]) - s["setback_m"])
            shape = LOGOS[s["text"]]()
            minx, miny, maxx, maxy = shape.bounds
            h = s["height_m"]
            shape = shp_scale(shape, h, h, origin=(0, 0))
            w = (maxx - minx) * h
            # slide along the edge to the nearest spot clear of plant rooms (higher roof parts) and on the roof
            obstacles = unary_union([p for p, zz in levels if zz > z + 0.8] or [Polygon()])
            inner = poly.buffer(-0.3)
            found = None
            setbacks = [sb for sb in np.arange(s["setback_m"], 0.99, -0.25)] or [s["setback_m"]]
            for sb in setbacks:            # closest to the wanted setback first, then nearer the edge
                c1 = centre + fwd * (s["setback_m"] - sb)
                for k in [0] + [sg * i * 0.5 for i in range(1, 120) for sg in (1, -1)]:
                    c2 = c1 + right * k
                    strip = LineString([tuple((c2 - right * w / 2)[:2]),
                                        tuple((c2 + right * w / 2)[:2])]).buffer(0.4)
                    if inner.contains(strip) and not strip.intersects(obstacles):
                        found = (c2, float(sb))
                        break
                if found:
                    break
            if not found:
                raise RuntimeError(f"{lm}: no place on the roof edge clear of plant rooms")
            centre, setback = found
            origin = centre - right * (w / 2) + np.array([0, 0, 0.35])
            # letters in the plane (right, up), extruded towards the viewer (fwd)
            extrude(shape, origin - fwd * s["depth_m"] / 2, right, np.array([0, 0, 1.0]), s["depth_m"], mesh_red)
            # steel frame behind: posts every ~2 m and two rails
            back = origin - fwd * (s["depth_m"] / 2 + 0.6)
            up = np.array([0, 0, 1.0])
            for xk in np.arange(0.3, w, 2.0):
                mesh_steel.box(back + right * xk + up * (h * 0.5), right, up, fwd, 0.08, h * 0.5 + 0.2, 0.08)
                mesh_steel.box(back + right * xk + fwd * 0.3 + up * 0.15, right, up, fwd, 0.06, 0.15, 0.4)
            for zr in (0.25 * h, 0.8 * h):
                mesh_steel.box(back + right * (w / 2) + up * zr, right, up, fwd, w / 2, 0.06, 0.06)
            files.append(_write_asset(ue, man, name, lm, "letters", "lm_red", tile_m, mesh_red.arrays(), "lm_red"))
            files.append(_write_asset(ue, man, name, lm, "frame", "lm_steel", tile_m, mesh_steel.arrays(), "lm_steel"))
            report[lm] = {"roof_z": round(z, 2), "width_m": round(w, 2), "height_m": h, "facing": facing_dir,
                          "setback_m": round(setback, 2),
                          "centre_model": [round(float(c), 2) for c in centre]}
    man["landmarks"] = {"version": 1, "report": report, "files": sorted(set(files))}
    (ue / "manifest.json").write_text(json.dumps(man, indent=1))
    return report


def cut_building_readonly(ue: Path, man: dict, bid: str):
    """The roof triangles of a building without changing anything (for rooftop objects)."""
    b = next(x for x in man["buildings"] if x["id"] == bid)
    poly = Polygon([(x / 100.0, -y / 100.0) for x, y in b["rings_ue"][0]]).buffer(0.3)
    vs, ns = [], []
    for a in man["assets"]:
        if a["kind"] != "roof":
            continue
        g = read_glb(ue / a["file"])
        v = gltf_to_model(g["positions"].astype(float))
        nn = gltf_to_model(g["normals"].astype(float))
        f = g["faces"].astype(np.int64)
        cen = v[f].mean(axis=1)
        inside = contains_xy(poly, cen[:, 0], cen[:, 1])
        if inside.any():
            vs.append(v[f[inside]])
            ns.append(nn[f[:, 0]][inside])
    tv = np.concatenate(vs).reshape(-1, 3)
    return b, None, (tv, np.arange(len(tv)).reshape(-1, 3), np.concatenate(ns)), {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ue", type=Path, help="the twin's ue/ export folder (manifest.json)")
    ap.add_argument("--letters-facing", choices=["north", "south", "east", "west"], default=None)
    ap.add_argument("--restore", action="store_true", help="only restore the export from landmarks_backup/")
    ap.add_argument("--xodr", nargs=2, type=Path, metavar=("IN", "OUT"),
                    help="also write OUT: the OpenDRIVE IN with the passages levelled and joined")
    args = ap.parse_args(argv)
    if args.restore:
        print("restored" if restore(args.ue) else "nothing to restore")
        return 0
    rep = apply(args.ue, facing=args.letters_facing)
    if args.xodr:
        spec = spec_for(json.loads((args.ue / "manifest.json").read_text())["name"])
        if any(s["type"] == "passage" for s in spec.values()):
            rep["xodr"] = patch_xodr(args.xodr[0], args.xodr[1], spec)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
