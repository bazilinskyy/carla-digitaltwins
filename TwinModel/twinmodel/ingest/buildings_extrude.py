"""Buildings from footprints and heights (no 3D model): prisms with roof shapes, for twinmodel.export.realbuildings.

Most of the world has no open 3D city model, but it has footprints with a height or a number of floors (Overture
Maps, OpenStreetMap, the Spanish Catastro, IGN's BD TOPO) and, for some buildings, a roof shape (Overture's and
OSM's ``roof:shape``). ``prism`` turns one footprint into the surfaces ``RealBuilding`` carries: walls from the
ground to the roof line and the roof faces, with semantic kinds, so the facade styles, party walls, photo roofs and
lit windows of realbuildings.apply work on them as on 3DBAG or PLATEAU buildings.

A roof is the lower envelope of a few planes over the footprint's minimum rotated rectangle: two for a gabled
roof (ridge along the long side, or across with ``orientation="across"``), four for a hipped or pyramidal one, one
for a skillion. Each plane's share of the footprint is one planar roof face (the footprint clipped by half-planes,
so courtyards and odd shapes work), and the walls get a vertex wherever the roof line changes plane, so gables
are closed. Unknown roofs are guessed only for small homes (``guess_roof``): pitched by the region's habit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# roof shapes of OSM/Overture -> the planes used here
SHAPE_ALIASES = {"gabled": "gabled", "saltbox": "gabled", "gambrel": "gabled", "round": "gabled",
                 "hipped": "hipped", "half-hipped": "hipped", "half_hipped": "hipped", "mansard": "hipped",
                 "side_hipped": "hipped", "pyramidal": "pyramidal", "dome": "pyramidal", "onion": "pyramidal",
                 "cone": "pyramidal", "skillion": "skillion", "lean_to": "skillion", "flat": "flat"}
# typical pitch (degrees) of a guessed home roof per facade region, and its roof shape
REGION_ROOF = {"nl": ("gabled", 40.0), "eu": ("gabled", 38.0), "us": ("hipped", 25.0), "south": ("hipped", 22.0),
               "jp": ("hipped", 25.0)}
HOME_CATEGORIES = {"house", "detached", "semidetached_house", "semi", "terrace", "bungalow", "cabin", "farm",
                   "residential", "static_caravan", "villa", "hut"}
STOREY_M = {"nl": 3.0, "eu": 3.1, "us": 3.3, "south": 3.0, "jp": 3.0}
DEFAULT_LEVELS = {"nl": 3, "eu": 3, "us": 2, "south": 4, "jp": 2}


@dataclass
class Heights:
    ground: float          # lowest terrain along the footprint (walls start just below it)
    eaves: float           # absolute eaves height
    top: float             # absolute top of the roof
    guessed: bool = False  # the height or the roof was guessed


def norm_shape(shape: Optional[str]) -> Optional[str]:
    if not shape:
        return None
    return SHAPE_ALIASES.get(str(shape).strip().lower().replace(" ", "_"))


def guess_roof(category: Optional[str], area_m2: float, eaves_h: float, region: str) -> Optional[str]:
    """A pitched roof for what is clearly a small home (OSM category or a small, low footprint), else None (flat)."""
    cat = (category or "").lower()
    if area_m2 <= 0 or eaves_h <= 0:
        return None
    small = area_m2 < 220.0 and eaves_h <= 10.0
    if cat in HOME_CATEGORIES and area_m2 < 400.0 and eaves_h <= 12.0:
        return REGION_ROOF.get(region, REGION_ROOF["eu"])[0]
    if cat in ("", "yes") and small and region in ("nl", "eu", "us"):
        return REGION_ROOF.get(region, REGION_ROOF["eu"])[0]
    return None


def roof_rise(poly, shape: Optional[str], region: str) -> float:
    """Height of a guessed roof: the region's pitch over half the short side of the rectangle, at most 6 m."""
    if not shape or shape == "flat":
        return 0.0
    _, _, hl, hw = frame_of(poly)
    pitch = REGION_ROOF.get(region, REGION_ROOF["eu"])[1]
    run = min(hl, hw) if shape in ("gabled", "hipped", "pyramidal") else 2 * min(hl, hw)
    return float(min(6.0, math.tan(math.radians(pitch)) * run))


def heights(ground: float, *, total_m: Optional[float] = None, roof_m: Optional[float] = None,
            levels: Optional[float] = None, region: str = "eu", shape: Optional[str] = None,
            guessed_rise: float = 0.0) -> Heights:
    """Absolute eaves and top from what a source gives: a total height (to the top, as OSM/Overture ``height``),
    a roof height, a number of floors; else the region's default floors (``guessed``)."""
    storey = STOREY_M.get(region, 3.0)
    guessed = False
    rise = roof_m if roof_m is not None and roof_m > 0 else (guessed_rise if shape and shape != "flat" else 0.0)
    if total_m is not None and total_m > 0:
        top = ground + float(total_m)
        rise = min(rise, 0.6 * float(total_m))
        eaves = top - rise
    else:
        n = levels if levels is not None and levels > 0 else None
        if n is None:
            n, guessed = DEFAULT_LEVELS.get(region, 3), True
        eaves = ground + float(n) * storey + 0.5
        top = eaves + rise
    return Heights(ground=ground, eaves=eaves, top=max(top, eaves), guessed=guessed)


# --------------------------------------------------------------------------- geometry

def frame_of(poly):
    """(centre, u, half length along u, half width along v) of the minimum rotated rectangle; u is the long axis."""
    rect = poly.minimum_rotated_rectangle
    pts = np.asarray(rect.exterior.coords)[:4] if rect.geom_type == "Polygon" else None
    if pts is None or len(pts) < 4:
        c = np.asarray(poly.centroid.coords[0])
        return c, np.array([1.0, 0.0]), 0.5, 0.5
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    u = e1 / l1 if l1 >= l2 and l1 > 0 else (e2 / l2 if l2 > 0 else np.array([1.0, 0.0]))
    c = pts.mean(axis=0)
    return c, u, max(l1, l2) / 2.0, max(1e-3, min(l1, l2) / 2.0)


def roof_planes(poly, shape: Optional[str], orientation: Optional[str] = None):
    """[(ka, kb, k0)] with f = ka*a + kb*b + k0 in the rectangle frame (a along u, b along v); roof = rise * min f.
    Also returns (centre, u, v). Empty list: flat."""
    c, u, hl, hw = frame_of(poly)
    if orientation == "across":
        u = np.array([-u[1], u[0]])
        hl, hw = hw, hl
    v = np.array([-u[1], u[0]])
    s = norm_shape(shape)
    if s == "gabled":
        planes = [(0.0, -1.0 / hw, 1.0), (0.0, 1.0 / hw, 1.0)]
    elif s == "hipped":
        planes = [(0.0, -1.0 / hw, 1.0), (0.0, 1.0 / hw, 1.0), (-1.0 / hw, 0.0, hl / hw), (1.0 / hw, 0.0, hl / hw)]
    elif s == "pyramidal":
        planes = [(-1.0 / hl, 0.0, 1.0), (1.0 / hl, 0.0, 1.0), (0.0, -1.0 / hw, 1.0), (0.0, 1.0 / hw, 1.0)]
    elif s == "skillion":
        planes = [(0.0, 0.5 / hw, 0.5)]
    else:
        planes = []
    return planes, (c, u, v)


def _f(plane, a, b):
    return plane[0] * a + plane[1] * b + plane[2]


def _halfplane(frame, ka, kb, k0, big):
    """Polygon (model xy) of {p : ka*a + kb*b + k0 <= 0} in a box of half-size ``big`` around the centre."""
    from shapely.geometry import Polygon, box
    c, u, v = frame
    sq = box(-big, -big, big, big)
    n = math.hypot(ka, kb)
    if n < 1e-12:
        return sq if k0 <= 0 else Polygon()
    # points of the square in (a, b), clipped by the half plane (Sutherland-Hodgman on one edge)
    pts = [(-big, -big), (big, -big), (big, big), (-big, big)]
    out = []
    for i in range(4):
        p, q = pts[i], pts[(i + 1) % 4]
        fp, fq = ka * p[0] + kb * p[1] + k0, ka * q[0] + kb * q[1] + k0
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    if len(out) < 3:
        return Polygon()
    xy = [tuple(c + a * u + b * v) for a, b in out]
    return Polygon(xy)


def _polys(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and g.area > 1e-4]


def _ring3(coords, zfn):
    pts = np.asarray(coords, dtype=np.float64)[:-1]
    if len(pts) < 3:
        return None
    return np.column_stack([pts, zfn(pts)])


def prism(poly, base: float, eaves: float, top: Optional[float] = None, shape: Optional[str] = None,
          orientation: Optional[str] = None) -> list:
    """Surfaces [(rings, kind)] of a building on footprint ``poly`` (shapely, model xy): walls from ``base`` to the
    roof line and the roof faces; ``top`` is the ridge (None or <= eaves: flat). Rings are (n, 3) arrays without
    the closing point, outer rings counter-clockwise seen from outside."""
    from shapely.geometry.polygon import orient
    if poly is None or poly.is_empty or poly.area < 1.0:
        return []
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type != "Polygon":
            parts = _polys(poly)
            if not parts:
                return []
            poly = max(parts, key=lambda p: p.area)
    poly = orient(poly.simplify(0.05, preserve_topology=True), 1.0)
    rise = max(0.0, (top or eaves) - eaves)
    planes, frame = roof_planes(poly, shape if rise > 0.05 else None, orientation)
    c, u, v = frame

    def ab(pts):
        d = pts[:, :2] - c
        return d @ u, d @ v

    def roof_z(pts):
        if not planes:
            return np.full(len(pts), eaves)
        a, b = ab(pts)
        return eaves + rise * np.min([_f(p, a, b) for p in planes], axis=0).clip(min=0.0)

    out = []
    # roof faces: the footprint's share of each plane
    if not planes:
        rings = [_ring3(poly.exterior.coords, lambda p: np.full(len(p), eaves))]
        rings += [_ring3(r.coords, lambda p: np.full(len(p), eaves)) for r in poly.interiors]
        rings = [r for r in rings if r is not None]
        if rings:
            out.append((rings, "RoofSurface"))
    else:
        big = 4.0 * max(1.0, math.sqrt(poly.area) + poly.length)
        for i, pi in enumerate(planes):
            region = poly
            for j, pj in enumerate(planes):
                if j == i:
                    continue
                hp = _halfplane(frame, pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2], big)
                region = region.intersection(hp)
                if region.is_empty:
                    break
            for piece in _polys(region):
                piece = orient(piece, 1.0)

                def zf(p, pl=pi):
                    a, b = ab(p)
                    return eaves + rise * np.clip(_f(pl, a, b), 0.0, None)
                rings = [_ring3(piece.exterior.coords, zf)] + [_ring3(r.coords, zf) for r in piece.interiors]
                rings = [r for r in rings if r is not None]
                if rings:
                    out.append((rings, "RoofSurface"))
    # walls: one per footprint edge, with a vertex wherever the roof line changes plane
    pairs = [(pi, pj) for k, pi in enumerate(planes) for pj in planes[k + 1:]]
    for ring in [poly.exterior, *poly.interiors]:
        pts = np.asarray(ring.coords, dtype=np.float64)
        for p, q in zip(pts[:-1], pts[1:]):
            if float(np.hypot(*(q - p))) < 0.01:
                continue
            ts = [0.0, 1.0]
            if pairs:
                pa, pb = ab(np.vstack([p, q]))
                for pi, pj in pairs:
                    g0 = _f(pi, pa[0], pb[0]) - _f(pj, pa[0], pb[0])
                    g1 = _f(pi, pa[1], pb[1]) - _f(pj, pa[1], pb[1])
                    if (g0 < 0 < g1) or (g1 < 0 < g0):
                        ts.append(g0 / (g0 - g1))
            ts = sorted(set(round(t, 6) for t in ts))
            line = np.array([p + t * (q - p) for t in ts])
            tops = roof_z(line)
            wall = [np.array([p[0], p[1], base]), np.array([q[0], q[1], base])]
            wall += [np.array([x, y, z]) for (x, y), z in zip(line[::-1], tops[::-1])]
            out.append(([np.asarray(wall)], "WallSurface"))
    return out


# --------------------------------------------------------------------------- terrain under a footprint

class Ground:
    """The twin's cached elevation grid (data/dem_*.npz), sampled along footprints: (lowest, median)."""

    def __init__(self, dem_path=None):
        self.z = None
        if dem_path is not None:
            d = np.load(dem_path, allow_pickle=True)
            self.z, self.x0, self.y0 = d["z"], float(d["x0"]), float(d["y0"])
            self.dx, self.dy = float(d["dx"]), float(d["dy"])

    def __bool__(self):
        return self.z is not None

    def sample(self, poly, step_m: float = 1.0):
        if self.z is None or poly is None or poly.is_empty:
            return 0.0, 0.0
        ring = poly.exterior
        pts = np.asarray([ring.interpolate(t).coords[0] for t in np.arange(0.0, max(ring.length, 0.1), step_m)])
        c = np.clip(np.round((pts[:, 0] - self.x0) / self.dx).astype(int), 0, self.z.shape[1] - 1)
        r = np.clip(np.round((pts[:, 1] - self.y0) / self.dy).astype(int), 0, self.z.shape[0] - 1)
        v = self.z[r, c]
        v = v[np.isfinite(v)]
        if not len(v):
            return 0.0, 0.0
        return float(v.min()), float(np.median(v))


def ground_for(build, man) -> "Ground":
    from ..export.realbuildings import find_dem
    dp = find_dem(build, man["bbox_wgs84"])
    if dp is None:
        print("extrude: no cached elevation grid; buildings stand at height 0")
        return Ground(None)
    return Ground(dp)


def model_transformer(man):
    from pyproj import Transformer
    from ..export.realbuildings import model_proj
    return Transformer.from_crs("EPSG:4326", model_proj(man), always_xy=True)


def to_model(tr, rings_lonlat):
    """[[lon, lat], ...] rings (outer first) -> shapely Polygon in the model frame, or None."""
    from shapely.geometry import Polygon
    rr = []
    for ring in rings_lonlat:
        a = np.asarray(ring, dtype=np.float64)
        if len(a) < 3:
            continue
        x, y = tr.transform(a[:, 0], a[:, 1])
        rr.append(list(zip(x, y)))
    if not rr:
        return None
    p = Polygon(rr[0], rr[1:])
    if not p.is_valid:
        p = p.buffer(0)
        parts = _polys(p)
        p = max(parts, key=lambda g: g.area) if parts else None
    return p if p is not None and p.area >= 2.0 else None


def eaves70(area_eaves: list) -> float:
    """Area-weighted 70th percentile of the parts' eaves (as 3DBAG's b3_h_dak_70p): a tall corner part does not make
    a whole block a tower, a low annex does not make it a shed."""
    items = sorted(area_eaves, key=lambda t: t[1])
    areas = np.cumsum([max(a, 1e-6) for a, _ in items])
    return float(items[int(np.searchsorted(areas, 0.7 * areas[-1]))][1])


def make_building(bid: str, parts: list, ground: "Ground", *, region: str = "eu", year=None, uses=None,
                  levels: int = 0, category: Optional[str] = None, base_offset: float = 0.3):
    """A RealBuilding from parts [(poly, heights_kwargs)] (heights_kwargs for ``heights``: total_m, roof_m, levels,
    shape, orientation), standing on the twin's terrain. Returns (RealBuilding or None, guessed: bool)."""
    from shapely.ops import unary_union
    from ..export.realbuildings import RealBuilding
    polys = [p for p, _ in parts if p is not None and not p.is_empty]
    if not polys:
        return None, False
    foot = unary_union(polys)
    foot_parts = _polys(foot)
    if not foot_parts:
        return None, False
    foot = max(foot_parts, key=lambda g: g.area) if foot.geom_type != "Polygon" else foot
    low, med = ground.sample(foot)
    surfaces, guessed, eaves_all, top_all = [], False, [], []
    for poly, kw in parts:
        kw = dict(kw)
        orientation = kw.pop("orientation", None)
        shape = norm_shape(kw.pop("shape", None))
        cat = kw.pop("category", category)
        guessed_shape = False
        h0 = heights(med, region=region, **{k: kw.get(k) for k in ("total_m", "roof_m", "levels")})
        if shape is None and kw.get("roof_m") is None:
            shape = guess_roof(cat, poly.area, h0.eaves - med, region)
            guessed_shape = shape is not None
        rise = roof_rise(poly, shape, region) if shape else 0.0
        h = heights(med, region=region, shape=shape, guessed_rise=rise,
                    **{k: kw.get(k) for k in ("total_m", "roof_m", "levels")})
        guessed |= h.guessed or guessed_shape
        surfaces += prism(poly, low - base_offset, h.eaves, h.top, shape, orientation)
        eaves_all.append((poly.area, h.eaves))
        top_all.append(h.top)
    if not surfaces:
        return None, guessed
    eaves = eaves70(eaves_all)
    return RealBuilding(id=bid, surfaces=surfaces, poly=foot, ground=low, height=max(0.0, eaves - low),
                        roof_z=max(top_all), year=year, uses=dict(uses or {}), levels=int(levels or 0),
                        parts=len(parts), region=region), guessed
