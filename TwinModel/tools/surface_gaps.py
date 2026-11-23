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
  reaching 5 cm under the edges it meets (no z-fighting, no seam).
  So is a wider slot with the same surface at the same height on both sides (a seam across a street).
* A wider gap in the open between different surfaces (an unmodelled verge or path, up to 2 x ``REACH`` m) is raised
  ground: the aerial-photo ground of the tile (what is really there, grass or paving) at the height of the surfaces
  around it.
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

from twinmodel.export.ue import _tile_index, gltf_to_model, read_glb, write_glb  # noqa: E402

RES = 0.2              # m per grid cell
REACH = 2.5            # m: a gap has road or pavement on two opposite sides within this distance
CRACK = 1.2            # m: thicker gaps in the open get the photo ground instead of the neighbouring surface
MIN_M2 = 0.1           # smaller specks are rasterisation noise
OVERLAP = 0.05         # m the patch reaches under the surfaces it meets
SINK = 0.003           # m under the surface it extends
STEEP_NZ = 0.5         # triangles steeper than 60 degrees (kerb faces, risers) give their top edge, not their plane
PLANE_CLAMP = 0.4      # m a patch vertex may lie above or below the heights of the triangle it extends
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
    cover = grid.raster_ids(tris[order], order) > 0          # everything that covers the ground
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
    return grid, gap, lab, nlab, tris, kind, layer, src, rid, extend


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
    return min(max(z, lo - PLANE_CLAMP), hi + PLANE_CLAMP)


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


def patches(grid, gap, lab, nlab, tris, kind, layer, src, rid, extend):
    """[(kind, layer, src asset or -1: photo ground, vertices (n,3), faces (m,3))] closing every gap."""
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
        for k in np.unique(nk[cells]):
            mk = cells & (nk == k)
            if mk.sum() * RES * RES < MIN_M2:
                continue
            # nearest triangle of THIS kind (photo ground: of any road or pavement), for the vertex heights
            kwin = np.where((win > 0) & (kind[np.maximum(win - 1, 0)] == k), win, 0) if k >= 0 else win
            if k >= 0:
                _, (kr, kc) = ndimage.distance_transform_edt(kwin == 0, return_indices=True)
            else:
                kr, kc = ir, ic
            poly = _mask_poly(mk, r0, c0, grid).buffer(OVERLAP, join_style=2).simplify(0.02)
            for p in _polys(poly):
                if p.area < MIN_M2:
                    continue
                p = orient(p.segmentize(0.6))
                rings = [np.asarray(p.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in p.interiors]
                xy = np.concatenate(rings)
                ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
                faces = np.asarray(earcut.triangulate_float64(xy, ends), np.int64).reshape(-1, 3)
                if not len(faces):
                    continue
                rr = np.clip(((xy[:, 1] - grid.y0) / RES - 0.5).round().astype(int) - r0, 0, win.shape[0] - 1)
                cc = np.clip(((xy[:, 0] - grid.x0) / RES - 0.5).round().astype(int) - c0, 0, win.shape[1] - 1)
                tid = kwin[kr[rr, cc], kc[rr, cc]] - 1
                z = np.array([_plane_z(tris[t], x, y) for t, (x, y) in zip(tid, xy)]) - SINK
                v = np.column_stack([xy, z])
                # counter-clockwise from above: normals up
                e1, e2 = v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]
                flip = np.cross(e1, e2)[:, 2] < 0
                faces[flip] = faces[flip][:, [0, 2, 1]]
                t0 = int(np.bincount(tid).argmax())
                if k >= 0:
                    out.append((COVER[k], int(layer[t0]), int(src[t0]), v, faces))
                else:
                    out.append(("ground", int(layer[t0]), -1, v, faces))
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
    for k, ly, s, v, f in pts:
        c = v.mean(axis=0)
        tile = tuple(_tile_index(float(c[0]), float(c[1]), man["tile_m"]))
        groups.setdefault((k, ly, tile, s), []).append((v, f))
    name, new = man["name"], []
    for (k, ly, tile, s), parts in sorted(groups.items()):
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
        for v, f in parts:
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
    for k, ly, s, v, f in pts:
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
        grid, gap, lab, nlab, tris, kind, layer, src, rid, extend = find(ue, man, fills=report_only or k > 0,
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
        pts = patches(grid, gap, lab, nlab, tris, kind, layer, src, rid, extend)
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
