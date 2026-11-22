"""OSM detail for a twin's buildings: what 3DBAG's extruded footprints leave out, from OpenStreetMap's 3D tagging.

Run by tools/landmarks.py on a ``ue/`` export (before the landmark models, which then rebuild their walls from the
carved ones). From the OSM data of the twin's box (building:part ways with min_height / height, entrance nodes,
highway=steps ways; Overpass, cached in ``ue/osm_detail.json``):

* open ground floors: where a building's parts start above the ground (min_height > 2 m: a tower on pilotis, a
  passage, a cantilever) and no ground-level part stands, the building's walls are cut away below that height, the
  underside gets a soffit, the ground-level parts inside (cores, columns) stand as solids, and where the open area
  meets an enclosed ground floor, that floor gets its own wall in the building's facade (3DBAG has none there);
* entrances: a door at each OSM entrance on a building's outline (also on the edge of an open area), facing out:
  a revolving door (door=revolving, or listed in ``revolving``) as a glass drum with four wings and a canopy, else
  a glass sliding door with its frame; standing on the floor its steps lead up to;
* steps: each highway=steps way as a flight (step_count x 16 cm, width tag, ``step_widths`` by way id, or 3 m; incline, else the end at an
  entrance or the higher ground is the top), with handrails where handrail=yes, and a landing from the top step to
  the door it leads to; ``extra_steps`` adds flights OSM does not have.

All of it is measured or tagged except the step rise (16 cm), the door sizes and the default width.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

from twinmodel.export import facades
from twinmodel.export.ue import _tile_index, gltf_to_model, read_glb, write_glb

STEP_RISE = 0.16
STEP_RUN_MAX = 0.40
DOOR_W, DOOR_H = 2.4, 2.6
DEFAULT_WIDTH = 3.0          # a flight without a width tag
DRUM_R, DRUM_H = 1.15, 2.5
SURFACE_KINDS = ("drivable", "sidewalk", "crossing", "parking", "island", "median", "verge", "ground", "groundplane")
WALL_KINDS = ("building_3dbag", "building")


def _num(v, default=0.0):
    try:
        return float(str(v).split()[0].replace(",", "."))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- OSM

def fetch(man: dict, cache: Path, log=print) -> dict:
    """The OSM elements of the twin's box (cached)."""
    if cache.exists():
        return json.loads(cache.read_text())
    import requests
    s, w, n, e = man["bbox_wgs84"]
    q = (f'[out:json][timeout:150];(way["building:part"]({s},{w},{n},{e});node["entrance"]({s},{w},{n},{e});'
         f'way["highway"]({s},{w},{n},{e}););out tags geom;')
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q},
                      headers={"User-Agent": "carla-digitaltwins/osm_detail"}, timeout=200)
    r.raise_for_status()
    data = r.json()
    data["fetched"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cache.write_text(json.dumps(data))
    log(f"osm_detail: {len(data['elements'])} OSM elements fetched")
    return data


def project(man: dict):
    import re
    from pyproj import Transformer
    proj = re.sub(r"\s*\+geoidgrids=\S+|\s*\+vunits=\S+", "", man["geo_reference"])
    tr = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    return lambda lat, lon: tr.transform(lon, lat)


def parse(data: dict, to_xy) -> dict:
    parts, entrances, steps, ways, below = [], [], [], [], []
    for e in data["elements"]:
        t = e.get("tags", {})
        if e["type"] == "node" and "entrance" in t:
            entrances.append({"id": e["id"], "xy": np.array(to_xy(e["lat"], e["lon"])), "tags": t})
        elif e["type"] == "way" and "geometry" in e:
            pts = [to_xy(p["lat"], p["lon"]) for p in e["geometry"]]
            if "building:part" in t and len(pts) >= 4:
                poly = Polygon(pts).buffer(0)
                if poly.area > 0.5:
                    lv = _num(t.get("building:levels"), 0)
                    h = _num(t.get("height"), lv * 3.2 if lv else 0)
                    mh = _num(t.get("min_height"), _num(t.get("building:min_level"), 0) * 3.2)
                    parts.append({"id": e["id"], "poly": poly, "h": h, "min_h": mh, "tags": t})
            elif "highway" in t and len(pts) >= 2:
                indoor = t.get("indoor") in ("yes", "corridor", "room", "area") or \
                    t.get("level", "0").split(";")[0].strip() not in ("0", "")
                if t["highway"] == "steps" and not indoor:
                    steps.append({"id": e["id"], "line": np.array(pts), "tags": t})
                if not indoor:                      # corridors inside a building are not a way under it
                    ways.append(LineString(pts))
                if _num(t.get("level", "0").split(";")[0], 0) < 0 or t.get("location") == "underground":
                    below.append(LineString(pts))
    return {"parts": parts, "entrances": entrances, "steps": steps, "ways": ways, "below": below}


# --------------------------------------------------------------------------- the ground

class Ground:
    """Heights of the export's walkable / drivable surfaces (layer 0), by tile."""

    def __init__(self, ue: Path, man: dict):
        self.ue, self.man, self.tiles = ue, man, {}

    def _tile(self, tile):
        if tile not in self.tiles:
            tris = []
            for a in self.man["assets"]:
                if a["kind"] in SURFACE_KINDS and a.get("layer", 0) == 0 and tuple(a["tile"]) == tile:
                    g = read_glb(self.ue / a["file"])
                    v = gltf_to_model(g["positions"].astype(float))
                    tris.append(v[g["faces"]])
            self.tiles[tile] = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        return self.tiles[tile]

    def z(self, x, y, default=None):
        T = self._tile(_tile_index(x, y, self.man["tile_m"]))
        if not len(T):
            return default
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        l1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / den
        l2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / den
        l3 = 1 - l1 - l2
        m = ok & (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not m.any():
            return default
        zz = l1[m] * a[m, 2] + l2[m] * b[m, 2] + l3[m] * c[m, 2]
        return float(zz.max())


# --------------------------------------------------------------------------- geometry

def prism(mesh, poly: Polygon, z0, z1, top=True, bottom=False):
    from landmarks import _polys, _triangulate
    for p in _polys(poly):
        ring = np.asarray(p.exterior.coords)
        if not p.exterior.is_ccw:
            ring = ring[::-1]
        for i in range(len(ring) - 1):
            a, b = ring[i], ring[i + 1]
            e = b - a
            if np.hypot(*e) < 0.02:
                continue
            n = np.array([e[1], -e[0], 0.0]) / np.hypot(*e)
            mesh.quad([a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1], n)
        verts, tri = _triangulate(p)
        if top:
            mesh.add(np.column_stack([verts, np.full(len(verts), z1)]), np.array([0, 0, 1.0]), verts, tri)
        if bottom:
            mesh.add(np.column_stack([verts, np.full(len(verts), z0)]), np.array([0, 0, -1.0]), verts, tri)


def oriented_box(mesh, centre, fwd, w, d, h0, h1):
    """A box standing from h0 to h1, d deep along fwd (unit xy), w wide across it."""
    f = np.array([fwd[0], fwd[1], 0.0])
    r = np.array([f[1], -f[0], 0.0])
    c = np.array([centre[0], centre[1], (h0 + h1) / 2])
    mesh.box(c, r, np.array([0, 0, 1.0]), f, w / 2, (h1 - h0) / 2, d / 2)


def outward(poly: Polygon, xy):
    """The outward normal (unit xy) of the polygon edge nearest xy."""
    ring = np.asarray(poly.exterior.coords)
    best, n_out = 1e9, np.array([1.0, 0.0])
    ccw = poly.exterior.is_ccw
    for i in range(len(ring) - 1):
        a, b = ring[i], ring[i + 1]
        d = LineString([a, b]).distance(Point(xy))
        if d < best:
            e = (b - a) / max(np.hypot(*(b - a)), 1e-9)
            n = np.array([e[1], -e[0]]) if ccw else np.array([-e[1], e[0]])
            best, n_out = d, n
    return n_out, best


# --------------------------------------------------------------------------- the passes

def open_ground(ue, man, osm, ground, meshes, append_wall, log=print):
    """Open ground floors under raised building parts. Returns {building id: [(region, z_top)]} and stats."""
    from landmarks import _backup, _polys
    regions, st = {}, {"buildings": 0, "regions_m2": 0.0, "cores": 0, "cut_triangles": 0, "new_wall_m2": 0.0}
    by_id = {}
    for b in man["buildings"]:
        fp = Polygon([(x / 100.0, -y / 100.0) for x, y in b["rings_ue"][0]]).buffer(0)
        parts = [p for p in osm["parts"] if fp.buffer(1.0).contains(p["poly"].representative_point())]
        raised = [p for p in parts if p["min_h"] > 2.0]
        if not raised:
            continue
        low = unary_union([p["poly"] for p in parts if p["min_h"] <= 0.5] or [Polygon()])
        cores = [p for p in parts if p["min_h"] <= 0.5 and p["poly"].area < 80.0]
        floors = unary_union([p["poly"] for p in parts if p["min_h"] <= 0.5 and p["poly"].area >= 80.0] or [Polygon()])
        base = b["base_z_cm"] / 100.0
        out = []
        raised_union = unary_union([p["poly"] for p in raised])
        for r in _polys(raised_union.intersection(fp).difference(floors)):
            over = [p for p in raised if p["poly"].intersection(r).area > 0.5]
            if not over:
                continue
            p = min(over, key=lambda q: q["min_h"])           # the lowest part over it is the soffit
            if r.area < 6.0:
                continue
            # open only where people or traffic pass underneath: an OSM way (footway, steps, road) runs
            # through it (a raised part over an unmapped ground floor, e.g. MetaForum's hall, stays closed)
            inner = r.buffer(-0.5)
            if inner.is_empty or not any(w.intersection(inner).length > 2.0 for w in osm.get("ways", ())
                                         if w.intersects(inner)):
                continue
            c = r.representative_point()
            z0 = ground.z(c.x, c.y, base)
            ztop = max(base + p["min_h"], z0 + 2.6)
            out.append((r, ztop, z0))
        if not out:
            continue
        st["buildings"] += 1
        regions[b["id"]] = out
        by_id[b["id"]] = (b, fp, out, cores, floors)
    # cut the walls in the open regions below their soffit, in the shared wall meshes
    if by_id:
        for a in man["assets"]:
            if a["kind"] not in WALL_KINDS:
                continue
            g = read_glb(ue / a["file"])
            v = gltf_to_model(g["positions"].astype(float))
            nn = gltf_to_model(g["normals"].astype(float))
            uv = g["uvs"].astype(float)
            f = g["faces"].astype(np.int64)
            cen = v[f].mean(axis=1)
            cut = np.zeros(len(f), bool)
            ztop_of = np.full(len(f), np.nan)
            for _, (b, fp, out, _, _) in by_id.items():
                for r, ztop, _ in out:
                    m = contains_xy(r.buffer(0.35), cen[:, 0], cen[:, 1]) & (v[f][:, :, 2].min(axis=1) < ztop - 0.01)
                    m &= ~cut
                    cut |= m
                    ztop_of[m] = ztop
            if not cut.any():
                continue
            _backup(ue, a["file"])
            keep_v, keep_n, keep_uv, keep_f = [v], [nn], [uv], [f[~cut]]
            nv = len(v)
            for k in np.where(cut)[0]:
                tri = f[k]
                p = v[tri]
                # uv is affine in position on a planar wall: fit it on the triangle
                A = np.column_stack([p, np.ones(3)])
                coef = np.linalg.lstsq(A, uv[tri], rcond=None)[0]
                (_, _), (va, fa) = facades.clip_triangles(p, np.array([[0, 1, 2]]), ztop_of[k])
                if len(fa) == 0:
                    continue
                keep_v.append(va)
                keep_n.append(np.repeat(nn[tri[:1]], len(va), axis=0))
                keep_uv.append(np.column_stack([va, np.ones(len(va))]) @ coef)
                keep_f.append(fa + nv)
                nv += len(va)
            st["cut_triangles"] += int(cut.sum())
            V, N, UV, F = (np.concatenate(x) for x in (keep_v, keep_n, keep_uv, keep_f))
            bbox = write_glb(ue / a["file"], a["asset"], V, N, UV, F, g["material"])
            a.update(vertices=int(len(V)), triangles=int(len(F)), bbox_ue=bbox)
    from landmarks import _triangulate
    for bid, (b, fp, out, cores, floors) in by_id.items():
        for r, ztop, z0 in out:
            st["regions_m2"] += r.area
            for p in _polys(r):                                          # the soffit
                verts, tri = _triangulate(p)
                meshes["soffit"].add(np.column_stack([verts, np.full(len(verts), ztop)]), np.array([0, 0, -1.0]),
                                     verts, tri)
            for c in cores:                                              # cores and columns
                if c["poly"].intersects(r.buffer(0.5)):
                    cc = c["poly"].representative_point()
                    prism(meshes["core"], c["poly"], ground.z(cc.x, cc.y, z0) - 0.2, ztop, top=False)
                    st["cores"] += 1
            # where the open area meets an enclosed ground floor: that floor's own wall, facing into the open area
            inner = r.boundary.difference(fp.exterior.buffer(0.4))
            for seg in getattr(inner, "geoms", [inner]):
                if seg.is_empty or seg.length < 1.0 or not floors.buffer(0.6).intersects(seg):
                    continue
                cs = np.asarray(seg.coords)
                for i in range(len(cs) - 1):
                    a2, b2 = cs[i], cs[i + 1]
                    L = np.hypot(*(b2 - a2))
                    if L < 0.3:
                        continue
                    mid = (a2 + b2) / 2
                    e = (b2 - a2) / L
                    n = np.array([e[1], -e[0]])
                    if not r.contains(Point(mid + n * 0.5)):
                        n = -n
                    za, zb = ground.z(a2[0], a2[1], z0), ground.z(b2[0], b2[1], z0)
                    append_wall(b, a2, b2, n, za, zb, ztop)
                    st["new_wall_m2"] += L * (ztop - z0)
    st["regions_m2"] = round(st["regions_m2"], 1)
    st["new_wall_m2"] = round(st["new_wall_m2"], 1)
    return regions, st


def flights(osm, ground, entrances, meshes, extra=(), widths=None, log=print):
    """Steps (and their landings to the doors they lead to). Returns {entrance id: floor z} and stats."""
    st = {"flights": 0, "steps": 0, "landings": 0, "rails": 0}
    floor = {}
    widths = widths or {}
    ways = list(osm["steps"]) + [{"id": f"extra{i}", "line": np.asarray(s["line"], float),
                                  "tags": {k: str(v) for k, v in s.items() if k != "line"}} for i, s in enumerate(extra)]
    for s in ways:
        line = s["line"]
        t = s["tags"]
        a, b = line[0], line[-1]
        L = float(np.sum(np.hypot(*np.diff(line, axis=0).T)))
        if L < 0.5:
            continue
        counted = int(_num(t.get("step_count"), 0))
        n = counted or max(2, min(8, int(round(L / 0.32))))       # uncounted: at most 8 (a long way is mostly flat)
        width = float(widths.get(str(s["id"]), _num(t.get("width"), DEFAULT_WIDTH)))
        # which end is the top: incline, else the end at an entrance, else the higher ground
        ent_a = min(entrances, key=lambda e: np.hypot(*(e["xy"] - a)), default=None)
        ent_b = min(entrances, key=lambda e: np.hypot(*(e["xy"] - b)), default=None)
        da = np.hypot(*(ent_a["xy"] - a)) if ent_a else 1e9
        db = np.hypot(*(ent_b["xy"] - b)) if ent_b else 1e9
        inc = t.get("incline", "")
        if inc == "up":
            top_b = True
        elif inc == "down":
            top_b = False
        elif min(da, db) < 6.0:
            top_b = db < da
        else:
            top_b = (ground.z(*b, 0) or 0) >= (ground.z(*a, 0) or 0)
        # steps down to a basement entrance (level -1, bicycle cellars) cannot be dug into the ground plate: left out
        near = [e for e in entrances if min(np.hypot(*(e["xy"] - a)), np.hypot(*(e["xy"] - b))) < 6.0]
        if any(_num(e["tags"].get("level"), 0) < 0 for e in near) or _num(t.get("level"), 0) < 0 or \
                any(min(w.distance(Point(a)), w.distance(Point(b))) < 2.5 for w in osm.get("below", ())):
            continue
        lo, hi = (a, b) if top_b else (b, a)
        fwd = (hi - lo) / max(np.hypot(*(hi - lo)), 1e-9)
        z0 = ground.z(lo[0], lo[1], None)
        if z0 is None:
            continue
        run = min(L / n, STEP_RUN_MAX)
        flight = run * n
        door = ent_b if top_b else ent_a
        ddoor = db if top_b else da
        # the flight ends at the top end of the way (a long way: an approach, then the steps)
        start = hi - fwd * flight
        for k in range(n):
            c = start + fwd * (run * (k + 0.5))
            oriented_box(meshes["steps"], c, fwd, width, run, z0 - 0.3, z0 + STEP_RISE * (k + 1))
        ztop = z0 + STEP_RISE * n
        st["flights"] += 1
        st["steps"] += n
        if door is not None and ddoor < 12.0:                            # the landing to the door
            dvec = door["xy"] - hi
            dl = float(np.hypot(*dvec))
            if dl > 0.3:
                oriented_box(meshes["steps"], hi + dvec / 2, dvec / dl, max(width, DOOR_W + 0.6), dl + 0.3,
                             z0 - 0.3, ztop)
                st["landings"] += 1
            floor[door["id"]] = ztop
        if t.get("handrail", "no") in ("yes", "both", "left", "right"):
            up = np.array([fwd[0], fwd[1], STEP_RISE / run])
            up /= np.linalg.norm(up)
            r = np.array([fwd[1], -fwd[0], 0.0])
            length = np.sqrt(flight ** 2 + (STEP_RISE * n) ** 2)
            for side in (-1, 1):
                mid = start + fwd * flight / 2 + r[:2] * side * (width / 2 - 0.05)
                c = np.array([mid[0], mid[1], z0 + STEP_RISE * n / 2 + 0.9])
                meshes["rail"].box(c, up, np.cross(r, up), r, length / 2, 0.025, 0.025)
                for end, zz in ((start, z0), (hi, ztop)):
                    pe = end + r[:2] * side * (width / 2 - 0.05)
                    meshes["rail"].box(np.array([pe[0], pe[1], zz + 0.45]), r, np.array([0, 0, 1.0]),
                                       np.array([fwd[0], fwd[1], 0.0]), 0.02, 0.45, 0.02)
                st["rails"] += 1
    return floor, st


def doors(osm, ground, buildings_fp, regions, floor, meshes, revolving=(), log=print):
    st = {"revolving": 0, "sliding": 0, "skipped": 0}
    edges = [fp for fp in buildings_fp]
    for b_regions in regions.values():
        edges += [r for r, _, _ in b_regions]
    for e in osm["entrances"]:
        xy = e["xy"]
        best = None
        for poly in edges:
            n, d = outward(poly, xy)
            if d < 1.5 and (best is None or d < best[1]):
                best = (n, d, poly)
        if best is None:
            st["skipped"] += 1
            continue
        n = best[0]
        if any(np.hypot(*(xy - np.asarray(r))) < 1.0 for r in revolving) or e["tags"].get("door") == "revolving":
            kind = "revolving"
        else:
            kind = "sliding"
        z = floor.get(e["id"])
        for k in (1.0, 2.5, 5.0):
            if z is not None:
                break
            z = ground.z(*(xy + n * k), None)
        if z is None:
            st["skipped"] += 1
            continue
        if kind == "revolving":
            revolving_door(meshes, xy + n * 0.55, n, z)
        else:
            sliding_door(meshes, xy, n, z)
        st[kind] += 1
    return st


def revolving_door(meshes, centre, n, z):
    """A revolving door: glass drum (two curved glass walls), four glass wings on a post, a canopy and a floor."""
    segs = 24
    cx, cy = centre
    ang0 = math.atan2(n[1], n[0])
    for a0, a1 in ((ang0 + math.pi * 0.25, ang0 + math.pi * 0.75), (ang0 - math.pi * 0.75, ang0 - math.pi * 0.25)):
        for i in range(segs // 4):
            t0 = a0 + (a1 - a0) * i / (segs // 4)
            t1 = a0 + (a1 - a0) * (i + 1) / (segs // 4)
            p0 = np.array([cx + DRUM_R * math.cos(t0), cy + DRUM_R * math.sin(t0)])
            p1 = np.array([cx + DRUM_R * math.cos(t1), cy + DRUM_R * math.sin(t1)])
            mid = (t0 + t1) / 2
            nn = np.array([math.cos(mid), math.sin(mid), 0.0])
            q = [[p0[0], p0[1], z], [p1[0], p1[1], z], [p1[0], p1[1], z + DRUM_H], [p0[0], p0[1], z + DRUM_H]]
            meshes["door_glass"].quad(*q, nn)
            meshes["door_glass"].quad(*q[::-1], -nn)
    for k in range(4):                                                   # the wings, at 45 degrees to the facade
        a = ang0 + math.pi / 4 + k * math.pi / 2
        d = np.array([math.cos(a), math.sin(a)])
        c = np.array([cx, cy]) + d * DRUM_R / 2
        meshes["door_glass"].box(np.array([c[0], c[1], z + DRUM_H / 2 - 0.05]), np.array([d[0], d[1], 0.0]),
                                 np.array([0, 0, 1.0]), np.array([-d[1], d[0], 0.0]), DRUM_R / 2 - 0.03,
                                 DRUM_H / 2 - 0.1, 0.015)
    meshes["rail"].box(np.array([cx, cy, z + DRUM_H / 2]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                       np.array([0, 0, 1.0]), 0.05, 0.05, DRUM_H / 2)
    from shapely.geometry import Point as P
    from landmarks import _triangulate
    disc = P(cx, cy).buffer(DRUM_R + 0.05, 24)
    verts, tri = _triangulate(disc)
    meshes["door_frame"].add(np.column_stack([verts, np.full(len(verts), z + DRUM_H + 0.25)]), np.array([0, 0, 1.0]),
                             verts, tri)
    meshes["door_frame"].add(np.column_stack([verts, np.full(len(verts), z + DRUM_H)]), np.array([0, 0, -1.0]),
                             verts, tri)
    ring = np.asarray(disc.exterior.coords)
    for i in range(len(ring) - 1):
        a, b = ring[i], ring[i + 1]
        e = b - a
        nn = np.array([e[1], -e[0], 0.0]) / max(np.hypot(*e), 1e-9)
        meshes["door_frame"].quad([a[0], a[1], z + DRUM_H], [b[0], b[1], z + DRUM_H], [b[0], b[1], z + DRUM_H + 0.25],
                                  [a[0], a[1], z + DRUM_H + 0.25], nn)
    verts2, tri2 = _triangulate(P(cx, cy).buffer(DRUM_R, 24))
    meshes["door_frame"].add(np.column_stack([verts2, np.full(len(verts2), z + 0.01)]), np.array([0, 0, 1.0]),
                             verts2, tri2)


def sliding_door(meshes, xy, n, z):
    """A glass sliding door in an aluminium frame, set 6 cm proud of the wall."""
    f = np.array([n[0], n[1], 0.0])
    r = np.array([n[1], -n[0], 0.0])
    up = np.array([0, 0, 1.0])
    base = np.array([xy[0], xy[1], z]) + f * 0.06
    meshes["door_glass"].quad(base - r * DOOR_W / 2, base + r * DOOR_W / 2, base + r * DOOR_W / 2 + up * DOOR_H,
                              base - r * DOOR_W / 2 + up * DOOR_H, f)
    for dx in (-DOOR_W / 2, DOOR_W / 2, 0.0):
        meshes["door_frame"].box(base + r * dx + up * DOOR_H / 2 + f * 0.03, r, up, f, 0.06 if dx else 0.025,
                                 DOOR_H / 2, 0.05)
    meshes["door_frame"].box(base + up * (DOOR_H + 0.06) + f * 0.03, r, up, f, DOOR_W / 2 + 0.06, 0.06, 0.06)


def apply(ue, man, spec, meshes, append_wall, log=print) -> dict:
    """The whole pass; meshes: Mesh per part key (soffit, core, steps, rail, door_glass, door_frame)."""
    data = fetch(man, Path(ue) / "osm_detail.json", log)
    osm = parse(data, project(man))
    ground = Ground(Path(ue), man)
    regions, st_open = open_ground(Path(ue), man, osm, ground, meshes, append_wall, log)
    floor, st_steps = flights(osm, ground, osm["entrances"], meshes, spec.get("extra_steps", ()),
                              spec.get("step_widths"), log)
    fps = [Polygon([(x / 100.0, -y / 100.0) for x, y in b["rings_ue"][0]]).buffer(0) for b in man["buildings"]]
    st_doors = doors(osm, ground, fps, regions, floor, meshes, spec.get("revolving", ()), log)
    return {"osm": {"parts": len(osm["parts"]), "entrances": len(osm["entrances"]), "steps": len(osm["steps"]),
                    "fetched": data.get("fetched")},
            "open_ground": st_open, "steps": st_steps, "doors": st_doors,
            "regions": {k: [(round(r.area, 1), round(z, 2)) for r, z, _ in v] for k, v in regions.items()}}
