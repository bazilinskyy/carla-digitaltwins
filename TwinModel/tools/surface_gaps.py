"""Close the gaps in a twin's road and pavement surfaces: slots and holes where no surface mesh covers the ground
between two pieces of road or pavement, so the ground plane (about 0.35 m lower) or nothing shows through as a dark
slit across the street (the joint where a landmark's passage fill met the street, mesh seams of the export).

Runs on a ``ue/`` export after tools/landmarks.py and before tools/road_paving.py (which then paves the patches like
the road they extend). Generic: no per-place numbers.

* The export's surfaces (drivable, parking, crossing, sidewalk, island, median, verge, ground, curb) are rasterised
  on a 0.2 m grid.
* A gap is a cell no surface covers with road or pavement on two opposite sides within ``REACH`` m (2.5 m). Wider open
  ground (lawns, squares on the ground plane's aerial photo) is never a gap.
* A crack (at most ``CRACK`` m across) or a gap under a building (a passage) is closed with the surfaces around it:
  each cell takes the kind of the nearest road or pavement, and each patch is that surface extended: its vertices
  lie on the plane of the nearest triangle of that kind (so kerbs stay kerbs and slopes continue), 3 mm under it and
  reaching 15 cm (half a cell and 5 cm) under the edges it meets (no z-fighting, no seam, no crack).
  So is a wider slot with the same surface at the same height on both sides (a seam across a street).
* A wider gap in the open between different surfaces (an unmodelled verge or path, up to 2 x ``REACH`` m) is raised
  ground: the aerial-photo ground of the tile (what is really there, grass or paving) at the height of the surfaces
  around it.
* Where a patch meets a lower surface (a cycle path beside the raised footpath it crosses) it stops at that
  surface's edge instead of reaching over it, and where a patch and its neighbour differ by more than ``RISE`` m a
  kerb face (a ``riser`` asset, the export's kerb stone) closes the step: a raised patch's 0.2 m cell staircase
  showed black open risers on the De Zaale kerbs (2026-11-24). Two kinds of patch in one gap meet on a straightened
  line, not on the cell staircase.
* The patches are new assets of that kind (``meshes/gaps/``, ``gap_fill``: true) with the material, texture and UV
  mapping of the neighbouring asset of that kind on the tile. Re-running removes them first (idempotent).

Usage: python tools/surface_gaps.py <build>/ue [--report-only] [--png gaps.png]
  (--report-only: the gaps left, counting patches already made; changes nothing)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mapbox_earcut as earcut  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy import ndimage  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon, box  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

import shapely  # noqa: E402
import shapely.ops  # noqa: E402
from shapely.affinity import affine_transform  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402

from twinmodel.export.ue import _curb_band_uv, _tile_index, gltf_to_model, read_glb, write_glb  # noqa: E402

RES = 0.2              # m per grid cell
REACH = 2.5            # m: a gap has road or pavement on two opposite sides within this distance
CRACK = 1.2            # m: thicker gaps in the open get the photo ground instead of the neighbouring surface
MIN_M2 = 0.1           # smaller specks are rasterisation noise
OVERLAP = RES / 2 + 0.05  # m the patch reaches past its cells, under the surfaces it meets: a surface's edge
                       # can lie half a cell from the cell edge (with 5 cm, cracks up to 5 cm stayed open)
SINK = 0.003           # m under the surface it extends
STEEP_NZ = 0.5         # triangles steeper than 60 degrees (kerb faces, risers) give their top edge, not their plane
PLANE_CLAMP = 0.4      # m a patch vertex may lie above or below the heights of the triangle it extends
GRADE = 0.12           # past a triangle's edge its plane continues at this grade at most: a sloping sliver along a
                       # pavement's edge (40 degrees) otherwise lifts the patch 0.25 m over the pavement (TUeCampus)
RISE = 0.03            # m: a step higher than this between a patch and its neighbour gets a kerb face
RISER_FOOT = 0.02      # m the kerb face reaches under the lower surface
SKIRT = 0.5            # m a patch's free edge reaches down (under the ground plane, like the export's plates)
STRAIGHT = 1.5         # cells: tolerance straightening the line between two kinds of patch in one gap
COVER = ("drivable", "parking", "crossing", "sidewalk", "island", "median", "verge", "ground", "curb")
FILL = ("drivable", "parking", "crossing", "sidewalk", "island", "median")
PROVENANCE = "inferred: surface gap closed with the neighbouring surface (tools/surface_gaps.py)"


def _load(ue: Path, man: dict, fills=False):
    """Surface triangles: (n, 3, 3) model coords, their kind index, layer and source asset index."""
    tris, kind, layer, src = [], [], [], []
    for i, a in enumerate(man["assets"]):
        if a["kind"] not in COVER or (a.get("gap_fill") and not fills):
            continue
        g = read_glb(ue / a["file"])
        f = g["faces"].astype(np.int64)
        if len(f) == 0:
            continue
        p = gltf_to_model(g["positions"].astype(float))[f]
        e1, e2 = p[:, 1, :2] - p[:, 0, :2], p[:, 2, :2] - p[:, 0, :2]
        area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        p = p[area > 1e-6]
        if not len(p):
            continue
        tris.append(p)
        kind += [COVER.index(a["kind"])] * len(p)
        layer += [a["layer"]] * len(p)
        src += [i] * len(p)
    return np.concatenate(tris), np.array(kind), np.array(layer), np.array(src)


class Grid:
    def __init__(self, tris):
        self.x0 = float(tris[:, :, 0].min()) - REACH - 1
        self.y0 = float(tris[:, :, 1].min()) - REACH - 1
        self.w = int((tris[:, :, 0].max() + REACH + 1 - self.x0) / RES) + 1
        self.h = int((tris[:, :, 1].max() + REACH + 1 - self.y0) / RES) + 1

    def raster_ids(self, tris, ids):
        """int32 grid: id + 1 of the triangle covering each cell centre (0: none); later triangles win."""
        im = Image.new("I", (self.w, self.h), 0)
        d = ImageDraw.Draw(im)
        q = (tris[:, :, :2] - [self.x0, self.y0]) / RES - 0.5
        for t, i in zip(q, ids):
            d.polygon([(t[0, 0], t[0, 1]), (t[1, 0], t[1, 1]), (t[2, 0], t[2, 1])], fill=int(i) + 1)
        return np.array(im, np.int32)

    def xy(self, r, c):
        return self.x0 + (np.asarray(c) + 0.5) * RES, self.y0 + (np.asarray(r) + 0.5) * RES


def _near(mask, n, dr, dc):
    out = np.zeros_like(mask)
    for s in range(1, n + 1):
        out |= np.roll(mask, (-dr * s, -dc * s), (0, 1))
    return out


def find(ue: Path, man: dict, fills=False, within=None, grid=None):
    """The gap cells and what they need: returns (grid, gap mask, labels, tris, kind, layer, src, fill ids)."""
    tris, kind, layer, src = _load(ue, man, fills)
    grid = grid or Grid(tris)          # later passes keep the first pass's grid (patches reach a little further)
    order = np.argsort(np.array([FILL.index(COVER[k]) if COVER[k] in FILL else -1 for k in kind]), kind="stable")
    cid = grid.raster_ids(tris[order], order)               # what covers the ground (road/pavement on top)
    cover = cid > 0
    road_ids = np.nonzero(np.isin(kind, [COVER.index(k) for k in FILL]))[0]
    road_ids = road_ids[np.argsort(tris[road_ids, :, 2].mean(axis=1), kind="stable")]   # the top surface wins
    rid = grid.raster_ids(tris[road_ids], road_ids)          # road/pavement triangle id + 1 per cell
    road = rid > 0
    n = int(round(REACH / RES))
    both = (_near(road, n, 0, 1) & _near(road, n, 0, -1)) | (_near(road, n, 1, 0) & _near(road, n, -1, 0))
    gap = ~cover & both
    if within is not None:        # later passes: only what is left of the first pass's gaps (patches must not grow)
        gap &= within
    lab, nlab = ndimage.label(gap)
    sizes = ndimage.sum(gap, lab, index=np.arange(1, nlab + 1)) * RES * RES
    small = np.zeros(nlab + 1, bool)
    small[1:] = sizes < MIN_M2
    gap &= ~small[lab]
    lab, nlab = ndimage.label(gap)
    # building footprints (a gap under a building is a passage: never photo ground, which shows the roof)
    im = Image.new("1", (grid.w, grid.h), 0)
    d = ImageDraw.Draw(im)
    for b in man.get("buildings", []) + man.get("buildings_osm", []):
        for ring in b.get("rings_ue", [])[:1]:
            pts = [((x / 100.0 - grid.x0) / RES - 0.5, (-y / 100.0 - grid.y0) / RES - 0.5) for x, y in ring]
            if len(pts) > 2:
                d.polygon(pts, fill=1)
    inb = np.array(im, bool)
    idx = np.arange(1, nlab + 1)
    thick = ndimage.maximum(ndimage.distance_transform_edt(gap), lab, index=idx) * 2 * RES if nlab else []
    under = ndimage.mean(inb, lab, index=idx) if nlab else []
    extend = np.zeros(nlab + 1, bool)
    extend[1:] = (np.asarray(thick) <= CRACK) | (np.asarray(under) >= 0.5)
    pad = int(REACH / RES) + 2
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if not extend[i]:     # the same surface at the same height on both sides: a seam in it, not a verge
            r0, c0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
            r1, c1 = min(sl[0].stop + pad, gap.shape[0]), min(sl[1].stop + pad, gap.shape[1])
            extend[i] = _continuous(rid[r0:r1, c0:c1], lab[r0:r1, c0:c1] == i, tris, kind, grid, r0, c0) >= 0.5
    return grid, gap, lab, nlab, tris, kind, layer, src, rid, extend, cid


def _continuous(win, cells, tris, kind, grid, r0, c0):
    """Share of the gap cells with the same kind of surface at the same height (10 cm) on two opposite sides
    within REACH, along the rows or the columns of the grid."""
    road = win > 0
    n = int(REACH / RES)
    ok = np.zeros(cells.shape, bool)
    for axis in (0, 1):
        r = road if axis == 1 else road.T
        c = cells if axis == 1 else cells.T
        w = win if axis == 1 else win.T
        idx = np.arange(r.shape[1])[None, :].repeat(r.shape[0], 0)
        left = np.maximum.accumulate(np.where(r, idx, -1), axis=1)
        right = np.flip(np.minimum.accumulate(np.flip(np.where(r, idx, r.shape[1] + 1), 1), axis=1), 1)
        rows, cols = np.nonzero(c)
        lf, rt = left[rows, cols], right[rows, cols]
        good = (lf >= 0) & (rt < r.shape[1]) & (cols - lf <= n) & (rt - cols <= n)
        res = np.zeros(len(rows), bool)
        for j in np.nonzero(good)[0]:
            ta, tb = w[rows[j], lf[j]] - 1, w[rows[j], rt[j]] - 1
            if kind[ta] != kind[tb]:
                continue
            if axis == 1:
                ya = yb = grid.y0 + (r0 + rows[j] + 0.5) * RES
                xa, xb = grid.x0 + (c0 + lf[j] + 0.5) * RES, grid.x0 + (c0 + rt[j] + 0.5) * RES
            else:
                xa = xb = grid.x0 + (c0 + rows[j] + 0.5) * RES
                ya, yb = grid.y0 + (r0 + lf[j] + 0.5) * RES, grid.y0 + (r0 + rt[j] + 0.5) * RES
            res[j] = abs(_plane_z(tris[ta], xa, ya) - _plane_z(tris[tb], xb, yb)) < 0.10
        okk = np.zeros(c.shape, bool)
        okk[rows, cols] = res
        ok |= okk if axis == 1 else okk.T
    return float(ok[cells].mean()) if cells.any() else 0.0


def _plane_z(t, x, y):
    """z of triangle t's plane at (x, y), within PLANE_CLAMP of the triangle's own heights. A steep triangle (a kerb
    face, a step riser) gives its top edge: extending its plane sent patches kilometres into the ground (2026-11-23,
    Helsinki -6.5 km, EhvCentre -282 m)."""
    a, b, c = t
    nrm = np.cross(b - a, c - a)
    lo, hi = float(min(a[2], b[2], c[2])), float(max(a[2], b[2], c[2]))
    n = float(np.linalg.norm(nrm))
    if n < 1e-12 or abs(nrm[2]) < STEEP_NZ * n:
        return hi
    z = float(a[2] - (nrm[0] * (x - a[0]) + nrm[1] * (y - a[1])) / nrm[2])
    (px, py, pz), d = _nearest_on(t, x, y)
    if d > 0:                          # outside the triangle: the slope goes on, but no steeper than GRADE
        z = min(max(z, pz - GRADE * d), pz + GRADE * d)
    return min(max(z, lo - PLANE_CLAMP), hi + PLANE_CLAMP)


def _nearest_on(t, x, y):
    """The point of triangle t (seen from above) nearest to (x, y), with its height, and the distance to it."""
    a, b, c = (np.asarray(q, float) for q in t)
    p = np.array([x, y], float)
    ab, ac = b[:2] - a[:2], c[:2] - a[:2]
    det = ab[0] * ac[1] - ab[1] * ac[0]
    if abs(det) > 1e-12:
        ap = p - a[:2]
        u = (ap[0] * ac[1] - ap[1] * ac[0]) / det
        w = (ab[0] * ap[1] - ab[1] * ap[0]) / det
        if u >= 0 and w >= 0 and u + w <= 1:
            return (x, y, float(a[2] + u * (b[2] - a[2]) + w * (c[2] - a[2]))), 0.0
    best = None
    for q, r in ((a, b), (b, c), (c, a)):
        e = r[:2] - q[:2]
        L = float(e @ e)
        s_ = 0.0 if L < 1e-18 else min(max(float((p - q[:2]) @ e) / L, 0.0), 1.0)
        m = q + s_ * (r - q)
        d = float(np.hypot(*(p - m[:2])))
        if best is None or d < best[1]:
            best = ((float(m[0]), float(m[1]), float(m[2])), d)
    return best


def _polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def _mask_poly(m, r0, c0, grid):
    """Polygon of the True cells of m (window origin r0, c0) from row runs."""
    boxes = []
    for r in range(m.shape[0]):
        row = m[r]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            x0 = grid.x0 + (c0 + a) * RES
            x1 = grid.x0 + (c0 + b) * RES
            y0 = grid.y0 + (r0 + r) * RES
            boxes.append(box(x0, y0, x1, y0 + RES))
    return unary_union(boxes)


def _cell_polys(masks, r0, c0, grid, straighten):
    """{key: polygon in metres} of the True cells of each mask; with straighten, the lines where two masks meet are
    simplified together (a coverage in whole cells, so the pieces still meet exactly) instead of the cell staircase."""
    keys, polys = list(masks), []
    for k in keys:
        boxes = []
        m = masks[k]
        for r in range(m.shape[0]):
            d = np.diff(np.concatenate([[0], m[r].astype(np.int8), [0]]))
            for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                boxes.append(box(int(a), r, int(b), r + 1))
        polys.append(shapely.segmentize(unary_union(boxes), 1.0) if boxes else Polygon())
    if straighten and len(keys) > 1:
        try:
            arr = np.array(polys, dtype=object)
            if shapely.coverage_is_valid(arr):
                polys = list(shapely.coverage_simplify(arr, STRAIGHT, simplify_boundary=False))
        except Exception:      # an odd coverage keeps the cell outline
            pass
    tr = [RES, 0, 0, RES, grid.x0 + c0 * RES, grid.y0 + r0 * RES]
    return {k: affine_transform(p, tr) for k, p in zip(keys, polys)}


def _lines(g):
    if g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    out = []
    for p in getattr(g, "geoms", []):
        out += _lines(p)
    return out


def _wall(line, z_hi, z_lo, upper):
    """Kerb faces along ``line``: [(v, f, st, normals)] for the stretches where z_hi(x, y) - z_lo(x, y) > RISE, from
    RISER_FOOT under z_lo up to z_hi, looking away from the polygon ``upper`` (the higher surface)."""
    out = []
    for ln in _lines(line):
        ln = ln.segmentize(0.5)
        xy = np.asarray(ln.coords)[:, :2]
        if len(xy) < 2 or ln.length < 0.05:
            continue
        top = np.array([z_hi(x, y) for x, y in xy])
        bot = np.array([z_lo(x, y) for x, y in xy])
        tall = (top - bot) > RISE
        for j in range(len(xy) - 1):
            if not (tall[j] or tall[j + 1]):
                continue
            a, b = xy[j], xy[j + 1]
            d = b - a
            n = float(np.hypot(*d))
            if n < 1e-4:
                continue
            nrm = np.array([d[1], -d[0]]) / n
            if upper.contains(shapely.Point(*(0.5 * (a + b) + 0.03 * nrm))):
                nrm = -nrm
            zt = [max(top[j], bot[j] + RISE), max(top[j + 1], bot[j + 1] + RISE)]
            zb = [bot[j] - RISER_FOOT, bot[j + 1] - RISER_FOOT]
            v = np.array([[a[0], a[1], zb[0]], [b[0], b[1], zb[1]], [b[0], b[1], zt[1]], [a[0], a[1], zt[0]]])
            f = np.array([[0, 1, 2], [0, 2, 3]])
            if np.dot(np.cross(v[1] - v[0], v[2] - v[0])[:2], nrm) < 0:
                f = f[:, [0, 2, 1]]
            st = np.array([[0.0, zt[0], zb[0]], [n, zt[1], zb[1]], [n, zt[1], zt[1]], [0.0, zt[0], zt[0]]])
            out.append((v, f, st, np.tile([nrm[0], nrm[1], 0.0], (4, 1))))
    return out


def _mesh_z(pieces):
    """z(x, y) on a patch's own triangles (points on its edge lie on one)."""
    tri = [v[f] for v, f in pieces for f in f]
    if not tri:
        return None
    tree = STRtree([Polygon(t[:, :2]) for t in tri])

    def z(x, y):
        t = tri[int(np.asarray(tree.query_nearest(shapely.Point(x, y))).ravel()[0])]
        a, b, c = t
        nrm = np.cross(b - a, c - a)
        if abs(nrm[2]) < 1e-12:
            return float(t[:, 2].mean())
        return float(a[2] - (nrm[0] * (x - a[0]) + nrm[1] * (y - a[1])) / nrm[2])
    return z


def patches(grid, gap, lab, nlab, tris, kind, layer, src, rid, extend, cid=None):
    """[(kind, layer, src asset or -1: photo ground, vertices (n,3), faces (m,3)[, (st, normals): a kerb face])]
    closing every gap."""
    out = []
    pad = int(REACH / RES) + 3
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        r0, r1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, gap.shape[0])
        c0, c1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, gap.shape[1])
        win = rid[r0:r1, c0:c1]
        cells = lab[r0:r1, c0:c1] == i
        # nearest road/pavement triangle for every cell of the window
        _, (ir, ic) = ndimage.distance_transform_edt(win == 0, return_indices=True)
        near_id = win[ir, ic] - 1
        nk = kind[near_id] if extend[i] else np.full(near_id.shape, -1)
        ks = [int(k) for k in np.unique(nk[cells]) if (cells & (nk == k)).sum() * RES * RES >= MIN_M2]
        look = {}
        for k in ks:
            # nearest triangle of THIS kind (photo ground: of any road or pavement), for the vertex heights, looked up
            # from the patch's own nearest cell (a cell outside it can pick a far sloping triangle: 0.4 m jumps)
            kwin = np.where((win > 0) & (kind[np.maximum(win - 1, 0)] == k), win, 0) if k >= 0 else win
            if k >= 0:
                _, (kr, kc) = ndimage.distance_transform_edt(kwin == 0, return_indices=True)
            else:
                kr, kc = ir, ic
            _, (mr, mc) = ndimage.distance_transform_edt(~(cells & (nk == k)), return_indices=True)
            look[k] = (kwin, kr, kc, mr, mc)

        def tri_at(k, x, y):
            kwin, kr, kc, mr, mc = look[k]
            rr = np.clip(np.round((np.asarray(y) - grid.y0) / RES - 0.5).astype(int) - r0, 0, win.shape[0] - 1)
            cc = np.clip(np.round((np.asarray(x) - grid.x0) / RES - 0.5).astype(int) - c0, 0, win.shape[1] - 1)
            rr, cc = mr[rr, cc], mc[rr, cc]
            return kwin[kr[rr, cc], kc[rr, cc]] - 1

        def patch_z(k):
            return lambda x, y: _plane_z(tris[tri_at(k, x, y)], x, y) - SINK

        region = _cell_polys({k: cells & (nk == k) for k in ks}, r0, c0, grid, straighten=bool(extend[i]))
        bpoly = {k: region[k].buffer(OVERLAP, join_style=2) for k in ks}
        # what covers the ground around the gap: lower surfaces are not covered, higher ones get a kerb face down to
        # the patch, edges on nothing a skirt
        cwin = cid[r0:r1, c0:c1] if cid is not None else win
        ids = np.unique(cwin[cwin > 0]) - 1
        if len(ids):
            nrm = np.cross(tris[ids, 1] - tris[ids, 0], tris[ids, 2] - tris[ids, 0])
            ids = ids[np.abs(nrm[:, 2]) >= STEEP_NZ * np.linalg.norm(nrm, axis=1)]
        tpoly = [Polygon(tris[t][:, :2]) for t in ids]
        tree = STRtree(tpoly) if tpoly else None

        def tri_z(sub):
            sub_tree = STRtree([tpoly[q] for q in sub])

            def z(x, y):
                j = sub[int(np.asarray(sub_tree.query_nearest(shapely.Point(x, y))).ravel()[0])]
                return _plane_z(tris[ids[j]], x, y)
            return z

        def union_of(qs, pad_m=0.0):
            g = unary_union([tpoly[q] for q in qs]) if qs else Polygon()
            return g.buffer(pad_m, join_style=2) if pad_m and not g.is_empty else g

        # pass 1: each patch's outline (stopping at lower surfaces and lower patches) and its triangles
        plan = {}
        for k in ks:
            zk = patch_z(k)
            lower, higher, level = [], [], []
            for q in ([int(q) for q in tree.query(bpoly[k].buffer(0.01))] if tree is not None else []):
                # the step where the two meet (a large sloping triangle's mean height says little)
                pa, _ = shapely.ops.nearest_points(tpoly[q], region[k])
                d = _plane_z(tris[ids[q]], pa.x, pa.y) - zk(pa.x, pa.y)
                (lower if d < -RISE else higher if d > RISE else level).append(q)
            lowk = []
            for k2 in ks:
                if k2 == k or region[k2].distance(region[k]) > 0.01:
                    continue
                edge = region[k2].boundary.intersection(region[k].buffer(0.002))
                px, py = (edge if not edge.is_empty else region[k2]).representative_point().coords[0]
                if patch_z(k2)(px, py) < zk(px, py) - RISE:
                    lowk.append(k2)
            poly = bpoly[k].difference(unary_union([union_of(lower)] + [bpoly[k2] for k2 in lowk]))
            pieces = []
            for pp in _polys(poly.simplify(0.02)):
                if pp.area < MIN_M2:
                    continue
                pp = orient(pp.segmentize(0.6))
                rings = [np.asarray(pp.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in pp.interiors]
                xy = np.concatenate(rings)
                ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
                faces = np.asarray(earcut.triangulate_float64(xy, ends), np.int64).reshape(-1, 3)
                if not len(faces):
                    continue
                tid = tri_at(k, xy[:, 0], xy[:, 1])
                z = np.array([_plane_z(tris[t], x, y) for t, (x, y) in zip(tid, xy)]) - SINK
                v = np.column_stack([xy, z])
                # counter-clockwise from above: normals up
                e1, e2 = v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]
                flip = np.cross(e1, e2)[:, 2] < 0
                faces[flip] = faces[flip][:, [0, 2, 1]]
                pieces.append((v, faces, int(np.bincount(tid).argmax())))
            plan[k] = (poly, pieces, lower, higher, level, lowk)
        # pass 2: the kerb faces, their tops and feet on the patches' own triangles
        mz = {k: _mesh_z([(v, f) for v, f, _ in plan[k][1]]) for k in ks}
        for k in ks:
            poly, pieces, lower, higher, level, lowk = plan[k]
            for v, faces, t0 in pieces:
                if k >= 0:
                    out.append((COVER[k], int(layer[t0]), int(src[t0]), v, faces))
                else:
                    out.append(("ground", int(layer[t0]), -1, v, faces))
            zk = mz[k]
            if zk is None:
                continue
            walls = []
            own = union_of(higher + level, 0.01)        # edges between two of the export's surfaces have faces
            if lower:
                low = union_of(lower)
                walls += _wall(poly.boundary.intersection(low.buffer(0.002, join_style=2)).difference(own),
                               zk, tri_z(lower), poly)
            for k2 in lowk:
                if mz[k2] is None:
                    continue
                line = poly.boundary.intersection(bpoly[k2].buffer(0.002, join_style=2))
                if lower:
                    line = line.difference(union_of(lower, 0.002))
                walls += _wall(line.difference(own), zk, mz[k2], poly)
            if higher:
                hi = union_of(higher)
                walls += _wall(hi.boundary.intersection(poly).difference(union_of(lower + level, 0.01)),
                               tri_z(higher), zk, hi)
            held = unary_union([union_of(lower + higher + level)] + [bpoly[k2] for k2 in ks if k2 != k])
            free = poly.boundary.difference(held.buffer(0.01, join_style=2)) if not held.is_empty else poly.boundary
            walls += _wall(free, zk, lambda x, y: zk(x, y) - SKIRT + RISER_FOOT, poly)
            if walls:
                t0 = pieces[0][2] if pieces else 0
                for v, f, st, nrm in walls:
                    out.append(("riser", int(layer[t0]), -2, v, f, (st, nrm)))
    return out


def _fit_uv(v, uv):
    a = np.column_stack([v[:, 0], v[:, 1], np.ones(len(v))])
    coef, *_ = np.linalg.lstsq(a, uv, rcond=None)
    return coef


def remove(ue: Path, man: dict) -> int:
    keep, n = [], 0
    for a in man.get("assets", []):
        if a.get("gap_fill"):
            (ue / a["file"]).unlink(missing_ok=True)
            n += 1
        else:
            keep.append(a)
    if "assets" in man:
        man["assets"] = keep
    return n


def _write(ue: Path, man: dict, pts, made: list):
    """New assets for the patches, grouped by (kind, layer, tile, neighbouring asset)."""
    groups = {}
    for k, ly, s, v, f, *extra in pts:
        c = v.mean(axis=0)
        tile = tuple(_tile_index(float(c[0]), float(c[1]), man["tile_m"]))
        groups.setdefault((k, ly, tile, s), []).append((v, f, extra[0] if extra else None))
    name, new = man["name"], []
    for (k, ly, tile, s), parts in sorted(groups.items()):
        if s == -2:     # kerb faces: the export's riser material, (along, height) UVs on the curb texture band
            vs, fs, uvs, ns, off = [], [], [], [], 0
            for v, f, (st, nrm) in parts:
                vs.append(v)
                fs.append(f + off)
                off += len(v)
                uvs.append([_curb_band_uv(a, zt, z) for (a, zt, _), z in zip(st, v[:, 2])])
                ns.append(nrm)
            v, f = np.concatenate(vs), np.concatenate(fs)
            asset = f"{name}_gap_{k}_{tile[0]}_{tile[1]}_{len(made) + len(new)}".replace("-", "m")
            rel = f"meshes/gaps/{asset}.glb"
            (ue / rel).parent.mkdir(parents=True, exist_ok=True)
            bbox = write_glb(ue / rel, asset, v, np.concatenate(ns), np.array(np.concatenate(uvs), float), f, "riser")
            new.append(dict(material="riser", semantic="SideWalk", kind=k, file=rel, asset=asset, layer=ly,
                            tile=list(tile), vertices=int(len(v)), triangles=int(len(f)), bbox_ue=bbox,
                            gap_fill=True, provenance=PROVENANCE))
            continue
        if s < 0:       # photo ground: the tile's ground (or ground plane) asset, for its photo and UVs
            cand = [a for a in man["assets"] if a["kind"] in ("ground", "groundplane") and list(a["tile"]) == list(tile)
                    and not a.get("gap_fill")]
            cand.sort(key=lambda a: (a["kind"] != "ground", -a.get("triangles", 0)))
            if not cand:
                continue
            sib = cand[0]
        else:
            sib = man["assets"][s]
        g = read_glb(ue / sib["file"])
        sv = gltf_to_model(g["positions"].astype(float))
        coef = _fit_uv(sv, g["uvs"].astype(float))
        vs, fs, off = [], [], 0
        for v, f, _ in parts:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        v, f = np.concatenate(vs), np.concatenate(fs)
        uv = np.column_stack([v[:, 0], v[:, 1], np.ones(len(v))]) @ coef
        nrm = np.tile([0.0, 0.0, 1.0], (len(v), 1))
        asset = f"{name}_gap_{k}_{tile[0]}_{tile[1]}_{len(made) + len(new)}".replace("-", "m")
        rel = f"meshes/gaps/{asset}.glb"
        (ue / rel).parent.mkdir(parents=True, exist_ok=True)
        bbox = write_glb(ue / rel, asset, v, nrm, uv, f, g["material"])
        e = {kk: vv for kk, vv in sib.items() if kk in ("material", "semantic", "texture")}
        e.update(kind=k, file=rel, asset=asset, layer=ly, tile=list(tile), vertices=int(len(v)),
                 triangles=int(len(f)), bbox_ue=bbox, gap_fill=True, provenance=PROVENANCE)
        new.append(e)
    man["assets"].extend(new)
    made.extend(new)


def _area(pts, area):
    for k, ly, s, v, f, *_ in pts:
        e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
        area[k] = area.get(k, 0.0) + float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())


def apply(ue: Path, report_only=False, png: Path | None = None, passes=4, log=print) -> dict:
    """Close the gaps (up to ``passes`` rounds: a patch's edge can leave a sliver); with report_only, only report
    the gaps left (counting existing patches)."""
    ue = Path(ue)
    man = json.loads((ue / "manifest.json").read_text())
    removed = 0 if report_only else remove(ue, man)
    if not any(a.get("kind") in COVER for a in man.get("assets", [])):
        return {"gaps": 0, "gap_m2": 0.0, "patched_m2": {}, "removed_old": removed, "note": "no surfaces"}
    made, area, rep, first, grid = [], {}, {"removed_old": removed}, None, None
    for k in range(1 if report_only else passes):
        grid, gap, lab, nlab, tris, kind, layer, src, rid, extend, cid = find(ue, man, fills=report_only or k > 0,
                                                                         within=first, grid=grid if k else None)
        if k == 0:
            first = gap.copy()
            rep.update(gaps=int(nlab), gap_m2=round(float(gap.sum()) * RES * RES, 1))
            if png:
                img = np.zeros(gap.shape + (3,), np.uint8)
                img[rid > 0] = (90, 90, 90)
                img[gap] = (255, 40, 40)
                Image.fromarray(img[::-1]).save(png)
            if report_only:
                rep["where"] = [[round(float(c), 1) for c in grid.xy((s[0].start + s[0].stop) / 2,
                                                                     (s[1].start + s[1].stop) / 2)]
                                for s in ndimage.find_objects(lab)]
                return rep
        if k > 0:
            sizes = ndimage.sum(gap, lab, index=np.arange(1, nlab + 1)) * RES * RES if nlab else [0.0]
            rep["left"], rep["left_max_m2"] = int(nlab), round(float(np.max(sizes)), 2)
        if not nlab or k == passes - 1 and k > 0:
            break
        pts = patches(grid, gap, lab, nlab, tris, kind, layer, src, rid, extend, cid)
        _area(pts, area)
        _write(ue, man, pts, made)
        rep[f"pass{k + 1}"] = int(nlab)
    rep["patched_m2"] = {k: round(v, 1) for k, v in sorted(area.items())}
    rep["assets"] = len(made)
    man["surface_gaps"] = rep
    (ue / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"surface gaps: {rep['gaps']} gaps ({rep['gap_m2']} m2) closed in {len(made)} assets {rep['patched_m2']}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ue", type=Path)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--png", type=Path)
    a = ap.parse_args(argv)
    rep = apply(a.ue, a.report_only, a.png)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
