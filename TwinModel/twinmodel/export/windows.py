"""Real window geometry for the textured 3DBAG / PLATEAU / Catastro walls (facades.py), and interiors behind it.

facades.py paints each style's windows into a tiling texture laid out by ``wall_uv``: a whole number of bays
across every wall and a whole number of storeys up to the eaves. This module uses that same grid to build the
windows as geometry instead:

* ``layout`` lists the openings of one wall (bay, storey, rectangle in the wall plane), with the window
  fractions of the style (``Style.window``) and the shopfront door rule of ``facades._window_rect``, so a
  window sits exactly where the texture painted it;
* ``cut_openings`` cuts those rectangles out of the wall polygon (the wall keeps its UVs; its texture is drawn
  without windows: ``facades.synth_facade(..., windows=False)``);
* ``module_parts`` builds one window module per style at its nominal size (reveal, frame with mullions and
  transom, sill, glass), in a local frame: x along the wall, -y outwards, z up, origin at the bottom centre of
  the opening on the wall plane. Every opening is an instance of its style's module, scaled in x and z to its
  real size (depth unscaled), so the bake places them as instanced Nanite meshes (one instanced component per
  module part and tile);
* ``instance_transform`` gives an opening's transform in Unreal (cm, yaw in degrees);
* interiors: the glass of a window can show a room (an interior-mapping material picked by the building use:
  home, office, shop; ``INTERIOR_OF_USE``) instead of dark glass. Interiors are invented detail: provenance
  ``generated``; the window geometry follows the style rules: provenance ``inferred``.

Pure numpy + shapely; unit tested in tests/test_windows.py.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from twinmodel.export import facades

# reveal depth (wall plane to the glass) per style, metres: deep masonry before the war, shallow modern frames
REVEAL_M = {"brick_old": 0.22, "stone_prewar": 0.25, "stone_old": 0.28, "brick_wederopbouw": 0.14,
            "brick_late": 0.12, "brick_postwar": 0.14, "brick_modern": 0.12, "brick_contemporary": 0.16,
            "render_modern": 0.14, "render_warm": 0.18, "concrete_panel": 0.10, "office_brick": 0.12,
            "office_concrete": 0.10, "office_glass": 0.03, "industrial": 0.05, "shopfront": 0.10,
            "stone_balcony": 0.30, "stucco_old": 0.24, "render_central": 0.14, "jp_tile": 0.10,
            "jp_mansion": 0.08, "jp_office": 0.05, "jp_concrete": 0.14, "shopfront_jp": 0.08}
FRAME_DEPTH_M = 0.07     # frame profile depth
GLASS_BACK_M = 0.03      # glass behind the frame's outer face
SILL_H_M = 0.05          # sill thickness
SILL_OUT_M = 0.05        # how far a sill sticks out of the wall
MIN_OPENING_M = 0.25     # openings narrower or lower than this are left painted (not cut)
EDGE_M = 0.02            # an opening must lie this far inside its wall polygon
MAX_TILT = 0.15          # walls whose normal has a larger vertical part (sloped walls) keep painted windows

# building use -> interior room kind (the interior-mapping material the bake picks; None: dark glass)
INTERIOR_OF_USE = {"home": "home", "office": "office", "shop": "shop", "industry": None}
PROVENANCE_WINDOWS = "inferred"
PROVENANCE_INTERIORS = "generated"


@dataclass(frozen=True)
class Opening:
    u0: float          # along the wall (metres, wall frame: facades.wall_frame), left edge
    u1: float
    z0: float          # model z (metres), bottom
    z1: float
    bay: int
    storey: int        # 0 = lowest storey this layout covers
    door: bool
    style: str

    @property
    def width(self) -> float:
        return self.u1 - self.u0

    @property
    def height(self) -> float:
        return self.z1 - self.z0


def _crc(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def wall_axes(normal: np.ndarray):
    """(along, up): unit vectors in the wall plane; along horizontal (facades.wall_frame), up = normal x along
    pointing upwards."""
    along = facades.wall_frame(normal)
    n = np.asarray(normal, float)
    up = np.cross(n / max(np.linalg.norm(n), 1e-12), along)
    if up[2] < 0:
        up = -up
    return along, up / max(np.linalg.norm(up), 1e-12)


def _plane_2d(pts: np.ndarray, along: np.ndarray, up: np.ndarray, origin: np.ndarray) -> np.ndarray:
    d = np.asarray(pts, float) - origin
    return np.column_stack([d @ along, d @ up])


def layout(ring: np.ndarray, normal: np.ndarray, z_lo: float, z_hi: float, storey_h: float,
           style: facades.Style, variant: int = 0, poly_rings: Optional[list] = None) -> list[Opening]:
    """Openings of one wall between z_lo (the floor of its first storey) and z_hi (its top), on the grid
    facades.wall_uv lays the style's texture on: ``ring`` is the wall's outer ring (model metres), the bays
    span its horizontal extent. Openings that do not lie inside the wall polygon (``poly_rings``: the part of
    the wall this layout is for, default ``ring``) by EDGE_M are left out."""
    from shapely.geometry import Polygon, box

    if style.blank or style.window is None:
        return []
    n = np.asarray(normal, float)
    if abs(n[2]) > MAX_TILT:
        return []
    along, up = wall_axes(n)
    a = np.asarray(ring, float) @ along
    a0, a1 = float(a.min()), float(a.max())
    width = a1 - a0
    if width <= 1e-6 or storey_h <= 0:
        return []
    bays = max(1, int(round(width / style.bay_m)))
    bay = width / bays
    origin = np.asarray(ring, float)[0]
    rings = poly_rings or [ring]
    p2 = [_plane_2d(r, along, up, origin) for r in rings]
    try:
        wall = Polygon(p2[0], p2[1:]).buffer(0)
    except Exception:  # noqa: BLE001 - degenerate wall
        return []
    if wall.is_empty:
        return []
    inner = wall.buffer(-EDGE_M)
    o_a = float(origin @ along)
    x0f, x1f, y0f, y1f = style.window
    out = []
    n_st = int(math.floor((z_hi - z_lo) / storey_h + 0.05))
    for s in range(max(0, n_st)):
        floor = z_lo + s * storey_h
        for b in range(bays):
            # the shopfront's door: facades._window_rect puts it in bay (row + variant) % TEX_BAYS of each tile,
            # row 1 for the one-storey shop floor (wall_uv's v from the ground: the lower row of the tile)
            door = style.shop and b % facades.TEX_BAYS == (1 + variant) % facades.TEX_BAYS
            u0, u1 = a0 + (b + x0f) * bay, a0 + (b + x1f) * bay
            z0 = floor + (0.01 if door else y0f) * storey_h
            z1 = floor + y1f * storey_h
            if u1 - u0 < MIN_OPENING_M or z1 - z0 < MIN_OPENING_M:
                continue
            # the rectangle in the plane frame (u relative to the origin, w along `up`)
            w0 = (z0 - origin[2]) / max(up[2], 1e-6)
            w1 = (z1 - origin[2]) / max(up[2], 1e-6)
            rect = box(u0 - o_a, w0, u1 - o_a, w1)
            if not inner.contains(rect):
                continue
            out.append(Opening(u0, u1, z0, z1, b, s, bool(door), style.name))
    return out


def cut_openings(rings: list, normal: np.ndarray, openings: list[Opening]) -> list[list[np.ndarray]]:
    """The wall polygon (outer ring + holes, model metres) with the openings cut out: a list of polygons, each
    [outer ring, holes...] as (n, 3) arrays on the wall plane, for realbuildings.triangulate."""
    from shapely.geometry import MultiPolygon, Polygon, box
    from shapely.ops import unary_union

    if not openings:
        return [list(rings)]
    n = np.asarray(normal, float)
    along, up = wall_axes(n)
    origin = np.asarray(rings[0], float)[0]
    o_a = float(origin @ along)
    p2 = [_plane_2d(r, along, up, origin) for r in rings]
    wall = Polygon(p2[0], p2[1:]).buffer(0)
    holes = unary_union([box(o.u0 - o_a, (o.z0 - origin[2]) / up[2], o.u1 - o_a, (o.z1 - origin[2]) / up[2])
                         for o in openings])
    rest = wall.difference(holes)
    polys = list(rest.geoms) if isinstance(rest, MultiPolygon) else ([rest] if not rest.is_empty else [])

    def back(xy):
        xy = np.asarray(xy, float)[:-1]
        return origin + xy[:, :1] * along + xy[:, 1:2] * up
    from shapely.geometry.polygon import orient
    # keep the wall's winding (facing outwards): the outer ring turns as the original did in the plane frame
    sign = 1.0 if Polygon(p2[0]).exterior.is_ccw else -1.0
    out = []
    for p in polys:
        if p.geom_type != "Polygon" or p.area < 1e-6:
            continue
        p = orient(p.simplify(0), sign)   # drop collinear points; outer ring as the original
        out.append([back(p.exterior.coords)] + [back(h.coords) for h in p.interiors])
    return out


# --------------------------------------------------------------------------- modules

def _box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box: (verts, normals, uvs, faces), 24 vertices (flat shading), CCW from outside."""
    faces_def = [  # normal, 4 corners (CCW seen from outside)
        ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        ((-1, 0, 0), [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]),
        ((0, 1, 0), [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]),
        ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        ((0, 0, -1), [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)]),
    ]
    return _quads(faces_def)


def _quads(faces_def):
    v, nrm, uv, f = [], [], [], []
    for normal, quad in faces_def:
        base = len(v)
        q = np.asarray(quad, float)
        nn = np.asarray(normal, float)
        # planar UVs in metres on the face's two largest axes
        keep = [i for i in range(3) if abs(nn[i]) < 0.5][:2]
        for p in q:
            v.append(p)
            nrm.append(nn)
            uv.append((p[keep[0]], -p[keep[1]]))
        f += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.asarray(v), np.asarray(nrm), np.asarray(uv), np.asarray(f, dtype=np.int64)


def _merge(parts):
    parts = [p for p in parts if p is not None and len(p[3])]
    if not parts:
        return None
    v, n, uv, f, off = [], [], [], [], 0
    for pv, pn, puv, pf in parts:
        v.append(pv)
        n.append(pn)
        uv.append(puv)
        f.append(pf + off)
        off += len(pv)
    return np.vstack(v), np.vstack(n), np.vstack(uv), np.vstack(f)


def nominal_size(style: facades.Style, door: bool = False) -> tuple[float, float]:
    """(width, height) of a style's opening at its nominal bay and storey (metres)."""
    x0, x1, y0, y1 = style.window
    return (x1 - x0) * style.bay_m, ((y1 - 0.01) if door else (y1 - y0)) * style.storey_m


def module_parts(style: facades.Style, door: bool = False) -> dict:
    """Window module of a style at its nominal size: {part: (verts, normals, uvs, faces)} for the parts
    reveal (the wall's own material), frame, sill, glass. Local frame: x along the wall, y INTO the wall
    (-y is outside), z up; origin at the bottom centre of the opening on the wall plane."""
    w, h = nominal_size(style, door)
    d = REVEAL_M.get(style.name, 0.12)
    hw = w / 2
    fm = max(0.03, style.frame_m)
    fd = min(FRAME_DEPTH_M, max(0.02, d))
    yf0 = max(0.0, d - fd - GLASS_BACK_M)      # the frame's outer face (behind the wall plane)
    yf1 = yf0 + fd
    yg = yf0 + GLASS_BACK_M                     # glass plane
    parts = {}
    # reveal: the four inner faces of the hole, from the wall plane to the frame (facing into the opening)
    if yf0 > 1e-3:
        parts["reveal"] = _quads([
            ((1, 0, 0), [(-hw, yf0, 0), (-hw, 0, 0), (-hw, 0, h), (-hw, yf0, h)]),       # left jamb faces +x
            ((-1, 0, 0), [(hw, 0, 0), (hw, yf0, 0), (hw, yf0, h), (hw, 0, h)]),         # right jamb faces -x
            ((0, 0, -1), [(-hw, 0, h), (hw, 0, h), (hw, yf0, h), (-hw, yf0, h)]),       # head faces down
            ((0, 0, 1), [(-hw, yf0, 0), (hw, yf0, 0), (hw, 0, 0), (-hw, 0, 0)]),        # bottom faces up
        ])
    # frame: four bars around the glass, mullions and transom
    bars = [_box(-hw, -hw + fm, yf0, yf1, 0, h), _box(hw - fm, hw, yf0, yf1, 0, h),
            _box(-hw + fm, hw - fm, yf0, yf1, h - fm, h)]
    if not door:
        bars.append(_box(-hw + fm, hw - fm, yf0, yf1, 0, fm))
    bar = max(0.02, fm / 2)
    for m in range(1, style.mullions + 1):
        xm = -hw + w * m / (style.mullions + 1)
        bars.append(_box(xm - bar / 2, xm + bar / 2, yf0 + 0.01, yf1 - 0.01, fm, h - fm))
    if style.transom and not door:
        zt = h - (h - 2 * fm) / 3 - fm
        bars.append(_box(-hw + fm, hw - fm, yf0 + 0.01, yf1 - 0.01, zt - bar / 2, zt + bar / 2))
    parts["frame"] = _merge(bars)
    # glass: one quad facing out (-y)
    gz0 = 0.0 if door else fm
    gv, gn, _, gf = _quads([((0, -1, 0), [(-hw + fm, yg, gz0), (hw - fm, yg, gz0), (hw - fm, yg, h - fm),
                                          (-hw + fm, yg, h - fm)])])
    # 0-1 across the pane (glTF v down): what an interior-mapping material expects of a window
    guv = np.column_stack([(gv[:, 0] + hw - fm) / max(w - 2 * fm, 1e-6), 1.0 - (gv[:, 2] - gz0) / max(h - fm - gz0, 1e-6)])
    parts["glass"] = (gv, gn, guv, gf)
    if style.sill and not door:
        parts["sill"] = _box(-hw - 0.05, hw + 0.05, -SILL_OUT_M, yf0, -SILL_H_M, 0.0)
    return parts


def module_key(style: str, door: bool) -> str:
    return f"win_{style}" + ("_door" if door else "")


# --------------------------------------------------------------------------- instances

def instance_transform(o: Opening, normal: np.ndarray, ring_origin: np.ndarray, style: facades.Style):
    """Unreal transform of an opening's module instance: (x_cm, y_cm, z_cm, yaw_deg, scale_x, scale_z).
    The module's local x maps to the wall's along axis, local +y (into the wall) to the inward normal."""
    along, up = wall_axes(normal)
    n = np.asarray(normal, float)
    origin = np.asarray(ring_origin, float)
    if origin.ndim == 2:     # the wall's ring: its first point is the plane origin layout() used
        origin = origin[0]
    o_a = float(origin @ along)
    uc = (o.u0 + o.u1) / 2 - o_a
    w0 = (o.z0 - origin[2]) / up[2]
    p = origin + uc * along + w0 * up
    # UE: (x, -y, z) * 100; the glb writer maps the module's local frame the same way, so a yaw that turns
    # UE x onto the wall's along axis (in UE) puts the module in place
    yaw = math.degrees(math.atan2(-along[1], along[0]))
    w_nom, h_nom = nominal_size(facades.STYLES[o.style], o.door)
    return (round(float(p[0]) * 100, 2), round(float(-p[1]) * 100, 2), round(float(p[2]) * 100, 2),
            round(yaw, 3), round(o.width / w_nom, 4), round(o.height / h_nom, 4))


def night_data(building_id: str, wall: int, o: Opening) -> tuple[float, float, float]:
    """Per-instance custom data for the interior's night light: (rank 0-1: lit when below the use's lit share
    x NightFactor, warm/cool 0/1, brightness 0.5-1), stable per building, wall and window."""
    import hashlib
    # blake2b, not crc32: the crc of near-identical strings has correlated low bits (repeated ranks)
    r = int.from_bytes(hashlib.blake2b(f"win|{building_id}|{wall}|{o.bay}|{o.storey}".encode(),
                                       digest_size=8).digest(), "little")
    rank = (r & 0xFFFF) / 65535.0
    cool = 1.0 if ((r >> 16) & 0xFF) / 255.0 < facades.LIGHT_COOL_SHARE else 0.0
    bright = 0.5 + 0.5 * ((r >> 24) & 0xFF) / 255.0
    return round(rank, 4), cool, round(bright, 3)


def interior_of(style: str) -> Optional[str]:
    return INTERIOR_OF_USE.get(facades.USE_OF_STYLE.get(style, "home"))


# --------------------------------------------------------------------------- hidden openings

HIDE_OUT_M = 0.75     # look this far out of the wall (beyond roof overhangs) for building in front of a window
HIDE_ABOVE_M = 0.2    # building whose top is this much above the opening's middle there hides it


def roof_top(buildings, triangulate):
    """clearance.TopZ over the non-vertical surfaces (roofs, floors, bases) of every building: the top of the
    building volume above a point."""
    from twinmodel.export.clearance import TopZ
    tris = []
    for b in buildings:
        for rings, kind in b.surfaces:
            if not rings or kind in ("WallSurface", "ClosureSurface"):
                continue
            res = triangulate(rings)
            if res is None:
                continue
            v, t, n = res
            if abs(n[2]) < 0.3:
                continue
            tris.append(v[t])
    return TopZ(np.concatenate(tris) if tris else np.zeros((0, 3, 3)))


def visible(ops: list, normal: np.ndarray, ring: np.ndarray, top) -> np.ndarray:
    """Per opening: False when a building stands in front of it (the top of the building volume HIDE_OUT_M out of
    the wall, at the opening's centre, is above the opening's middle): the walls between the parts of one building
    and walls against a neighbour that party-wall detection missed."""
    if not ops:
        return np.zeros(0, dtype=bool)
    along, up = wall_axes(normal)
    n = np.asarray(normal, float)
    nh = n[:2] / max(np.linalg.norm(n[:2]), 1e-9)
    origin = np.asarray(ring, float)[0]
    o_a = float(origin @ along)
    xy, zmid = [], []
    for o in ops:
        c = origin + ((o.u0 + o.u1) / 2 - o_a) * along
        xy.append(c[:2] + HIDE_OUT_M * nh)
        zmid.append((o.z0 + o.z1) / 2)
    tz = top(np.asarray(xy))
    return ~(tz > np.asarray(zmid) + HIDE_ABOVE_M)


# --------------------------------------------------------------------------- the road clearance cut

CUT_SIDE_M = 0.3     # an opening counts as cut when its line on the wall, widened this much to both sides, meets a cut


def drop_cut(manifest: dict, regions: list) -> dict:
    """Remove the window instances on wall parts the road clearance cut takes away (export/clearance.py: passages
    under buildings, trimmed overhangs), so no window floats in a passage. ``regions``: [(polygon in model xy,
    floor z, clear z)]. Every part of one window shares its transform, so a window goes as a whole. Edits
    ``manifest["window_instances"]`` in place; returns {"windows_dropped": n, "instances_dropped": n}."""
    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    inst = manifest.get("window_instances") or []
    mods = manifest.get("window_modules") or {}
    out = {"windows_dropped": 0, "instances_dropped": 0}
    if not inst or not regions:
        return out
    polys = [r[0] for r in regions]
    tree = STRtree(polys)
    cut_tf = {}      # transform tuple -> cut?

    def is_cut(tf, mesh):
        key = tuple(tf)
        if key in cut_tf:
            return cut_tf[key]
        m = mods.get(mesh, {})
        st = facades.STYLES.get(m.get("style", ""))
        if st is None:
            cut_tf[key] = False
            return False
        w_nom, h_nom = nominal_size(st, bool(m.get("door")))
        x, y, z, yaw, sx, sz = tf
        cx, cy, z0 = x / 100.0, -y / 100.0, z / 100.0
        w, h = w_nom * sx, h_nom * sz
        ax, ay = math.cos(math.radians(yaw)), -math.sin(math.radians(yaw))   # the wall's along axis, model frame
        line = LineString([(cx - ax * w / 2, cy - ay * w / 2), (cx + ax * w / 2, cy + ay * w / 2)]).buffer(CUT_SIDE_M)
        hit = False
        for i in tree.query(line):
            poly, floor, clear = regions[int(i)]
            if z0 < clear and z0 + h > floor and poly.intersects(line):
                hit = True
                break
        cut_tf[key] = hit
        return hit
    kept = []
    for e in inst:
        keep = [i for i, tf in enumerate(e["transforms"]) if not is_cut(tf, e["mesh"])]
        out["instances_dropped"] += len(e["transforms"]) - len(keep)
        if not keep:
            continue
        if len(keep) != len(e["transforms"]):
            e["transforms"] = [e["transforms"][i] for i in keep]
            if e.get("custom"):
                e["custom"] = [e["custom"][i] for i in keep]
        kept.append(e)
    manifest["window_instances"] = kept
    out["windows_dropped"] = sum(1 for v in cut_tf.values() if v)
    st = manifest.setdefault("stats", {})
    if "window_instances" in st:
        st["window_instances"] = sum(len(e["transforms"]) for e in kept)
    st["windows_cut_by_clearance"] = out["windows_dropped"]
    return out
