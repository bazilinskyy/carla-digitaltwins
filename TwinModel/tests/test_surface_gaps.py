import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import surface_gaps  # noqa: E402
from twinmodel.export.ue import read_glb, gltf_to_model, write_glb  # noqa: E402


def _quad(x0, x1, y0, y1, z):
    v = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    return v, np.array([[0, 1, 2], [0, 2, 3]])


def _export(tmp_path, pieces, ground_z=0.0):
    """A tiny export: road/pavement quads (kind, x0, x1, y0, y1, z) on one tile over a ground plane."""
    ue = tmp_path / "ue"
    (ue / "meshes").mkdir(parents=True)
    assets = []
    allp = list(pieces) + [("groundplane", -50, 50, -50, 50, ground_z)]
    for i, (kind, x0, x1, y0, y1, z) in enumerate(allp):
        v, f = _quad(x0, x1, y0, y1, z)
        rel = f"meshes/T_{kind}_{i}.glb"
        bbox = write_glb(ue / rel, f"T_{kind}_{i}", v, np.tile([0, 0, 1.0], (4, 1)), v[:, :2] / 4.0, f, kind)
        assets.append({"file": rel, "asset": f"T_{kind}_{i}", "layer": 0, "kind": kind, "material": kind,
                       "semantic": "Road", "tile": [0, 0], "vertices": 4, "triangles": 2, "bbox_ue": bbox})
    (ue / "manifest.json").write_text(json.dumps({"name": "T", "tile_m": 200, "assets": assets, "buildings": []}))
    return ue


def test_a_slot_across_the_street_is_closed_at_road_height(tmp_path):
    # a road and its pavement end 1.6 m short of the next piece (the Atlas passage, 2026-11-23); ground 0.4 m below
    ue = _export(tmp_path, [("drivable", -20, -0.8, -3.5, 3.5, 16.2), ("drivable", 0.8, 20, -3.5, 3.5, 16.2),
                            ("sidewalk", -20, -0.8, 3.5, 5.5, 16.35), ("sidewalk", 0.8, 20, 3.5, 5.5, 16.35)],
                 ground_z=15.8)
    rep = surface_gaps.apply(ue, log=lambda *a: None)
    assert rep["gaps"] >= 1 and rep["patched_m2"].get("drivable", 0) > 7 and rep["patched_m2"].get("sidewalk", 0) > 2
    man = json.loads((ue / "manifest.json").read_text())
    fills = [a for a in man["assets"] if a.get("gap_fill")]
    zs = {}
    for a in fills:
        v = gltf_to_model(read_glb(ue / a["file"])["positions"].astype(float))
        zs.setdefault(a["kind"], []).extend(v[:, 2].tolist())
    assert abs(np.mean(zs["drivable"]) - 16.2) < 0.02        # the road extended, not the ground
    assert abs(np.mean(zs["sidewalk"]) - 16.35) < 0.02       # the kerb stays a kerb
    assert surface_gaps.apply(ue, report_only=True)["gaps"] == 0
    # idempotent: a second run replaces its patches
    surface_gaps.apply(ue, log=lambda *a: None)
    man2 = json.loads((ue / "manifest.json").read_text())
    assert len([a for a in man2["assets"] if a.get("gap_fill")]) == len(fills)


def test_open_ground_is_not_a_gap(tmp_path):
    # two roads 8 m apart with a lawn between: no patch
    ue = _export(tmp_path, [("drivable", -20, 20, -10, -4, 16.2), ("drivable", -20, 20, 4, 10, 16.2)])
    assert surface_gaps.apply(ue, log=lambda *a: None)["gaps"] == 0


def test_a_steep_neighbour_does_not_send_the_patch_into_the_ground():
    # nearest triangle a nearly vertical kerb face: its plane at 2 m reaches -284 m (Helsinki, 2026-11-23)
    kerb = np.array([[0, 0, 16.35], [1, 0, 16.35], [0, 0.001, 16.2]], float)
    assert abs(surface_gaps._plane_z(kerb, 0.5, 2.0) - 16.35) < 1e-9
    # a gentle slope extends normally, a far extrapolation is clamped to 0.4 m of the triangle
    ramp = np.array([[0, 0, 10.0], [1, 0, 10.0], [0, 1, 10.1]], float)
    assert abs(surface_gaps._plane_z(ramp, 0.5, 2.0) - 10.2) < 1e-9
    assert abs(surface_gaps._plane_z(ramp, 0.5, 50.0) - 10.5) < 1e-9
    # a 40 degree sliver along a pavement's edge continues at 12 % at most past its edge, not at its own slope
    # (TUeCampus: a patch 0.25 m over the pavement beside it)
    sliver = np.array([[0, 0, 10.0], [1, 0, 10.0], [0, 0.2, 10.17]], float)
    (_, _, pz), d = surface_gaps._nearest_on(sliver, 0.5, 0.5)
    assert d > 0.3 and surface_gaps._plane_z(sliver, 0.5, 0.5) <= pz + 0.12 * d + 1e-9
    assert abs(surface_gaps._plane_z(sliver, 0.2, 0.05) - 10.0425) < 1e-9          # inside: its own plane


def _fills(ue):
    man = json.loads((ue / "manifest.json").read_text())
    out = {}
    for a in man["assets"]:
        if a.get("gap_fill"):
            g = read_glb(ue / a["file"])
            out.setdefault(a["kind"], []).append((gltf_to_model(g["positions"].astype(float)), g["faces"].astype(int),
                                                  a))
    return out


def _open_steps(ue, x0, x1, y0, y1, res=0.05):
    """Metres of step (> 3 cm between the top surfaces of neighbouring cells) with no kerb face on it, in a box."""
    man = json.loads((ue / "manifest.json").read_text())
    tris, kind, _, _ = surface_gaps._load(ue, man, fills=True)
    xs, ys = np.arange(x0, x1, res) + res / 2, np.arange(y0, y1, res) + res / 2
    X, Y = np.meshgrid(xs, ys)
    top = np.full(X.shape, np.nan)
    for t in tris:
        a, b, c = t
        d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(d) < 1e-9:
            continue
        u = ((X - a[0]) * (c[1] - a[1]) - (Y - a[1]) * (c[0] - a[0])) / d
        w = ((b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])) / d
        ins = (u >= -1e-6) & (w >= -1e-6) & (u + w <= 1 + 1e-6)
        z = a[2] + u * (b[2] - a[2]) + w * (c[2] - a[2])
        top = np.where(ins & ~(top >= z), z, top)
    segs = []
    for a in man["assets"]:
        if a["kind"] == "riser":
            g = read_glb(ue / a["file"])
            v = gltf_to_model(g["positions"].astype(float))
            for f in g["faces"].astype(int):
                q = v[f]
                segs.append((q[:, :2].min(0), q[:, :2].max(0), q[:, 2].min(), q[:, 2].max()))
    open_m = 0.0
    for axis in (0, 1):
        za, zb = (top[:, :-1], top[:, 1:]) if axis == 1 else (top[:-1], top[1:])
        step = np.abs(za - zb) > 0.03
        for r, c in zip(*np.nonzero(step)):
            mx = X[r, c] + (res / 2 if axis == 1 else 0)
            my = Y[r, c] + (res / 2 if axis == 0 else 0)
            lo, hi = min(za[r, c], zb[r, c]), max(za[r, c], zb[r, c])
            if not any(mn[0] - 0.06 <= mx <= mxx[0] + 0.06 and mn[1] - 0.06 <= my <= mxx[1] + 0.06
                       and zmin <= lo + 0.01 and zmax >= hi - 0.01 for mn, mxx, zmin, zmax in segs):
                open_m += res
    return open_m


def test_a_raised_patch_beside_a_lower_surface_has_no_open_step(tmp_path):
    # De Zaale, 2026-11-24: a footpath crosses a cycle path; the pavements on both sides stop 2.4 m short of each other
    # and the cycle path's mesh bulges into the gap (15 cm lower). The patch met the bulge on a 0.2 m cell staircase
    # with no kerb face: black open risers. Now a kerb face (riser material) closes every step, on a straight line.
    ue = _export(tmp_path, [("drivable", -20, 20, -4, 0, 15.82), ("drivable", -1.0, 0.6, 0, 0.7, 15.82),
                            ("sidewalk", -20, -1.2, 0, 2.0, 15.97), ("sidewalk", 1.2, 20, 0, 2.0, 15.97)],
                 ground_z=15.6)
    surface_gaps.apply(ue, log=lambda *a: None)
    f = _fills(ue)
    assert all(a["material"] == "riser" for _, _, a in f["riser"])
    pts = np.concatenate([v for v, _, _ in f["riser"]])
    assert pts[:, 2].max() > 15.95 and pts[:, 2].min() < 15.81
    assert _open_steps(ue, -1.4, 1.4, 0.05, 1.95) < 0.15
    # the pavement patch covers nothing of the cycle path's bulge (no raised lip over it)
    from shapely.geometry import Polygon, box as sbox
    from shapely.ops import unary_union
    walk = unary_union([Polygon(v[t, :2]) for v, faces, _ in f["sidewalk"] for t in faces])
    assert walk.intersection(sbox(-1.0, 0, 0.6, 0.7)).area < 0.02
    assert surface_gaps.apply(ue, report_only=True)["gaps"] == 0


def test_a_gap_between_two_pavements_is_pavement_not_a_road_notch(tmp_path):
    # De Zaale, 2026-11-24: a footpath crosses a cycle path and leaves a 2.4 m gap between two pavements. Its cells
    # nearest the cycle path took the cycle path's kind: a 1.3 m road-height notch with kerb faces in the pavement.
    # The pavement lies on both sides of the gap at one height, so the gap is pavement up to the cycle path's edge.
    ue = _export(tmp_path, [("drivable", -20, 20, -4, 0, 15.82),
                            ("sidewalk", -20, -1.2, 0, 2.0, 15.97), ("sidewalk", 1.2, 20, 0, 2.0, 15.97)],
                 ground_z=15.6)
    surface_gaps.apply(ue, log=lambda *a: None)
    f = _fills(ue)
    from shapely.geometry import Polygon, box as sbox
    from shapely.ops import unary_union
    road = unary_union([Polygon(v[t, :2]) for v, faces, _ in f.get("drivable", []) for t in faces])
    walk = unary_union([Polygon(v[t, :2]) for v, faces, _ in f["sidewalk"] for t in faces])
    assert road.intersection(sbox(-1.2, 0.4, 1.2, 2.0)).area < 0.02           # no notch into the crossing
    assert walk.intersection(sbox(-1.2, 0.4, 1.2, 2.0)).area > 0.95 * 2.4 * 1.6
    assert _open_steps(ue, -1.1, 1.1, -0.3, 1.95) < 0.15       # the kerb along the cycle path is closed

def test_two_kinds_in_one_gap_meet_on_a_straight_kerb(tmp_path):
    ue = _export(tmp_path, [("drivable", -20, -0.8, -3.5, 3.5, 16.2), ("drivable", 0.8, 20, -3.5, 3.5, 16.2),
                            ("sidewalk", -20, -0.8, 3.5, 5.5, 16.35), ("sidewalk", 0.8, 20, 3.5, 5.5, 16.35)],
                 ground_z=15.8)
    surface_gaps.apply(ue, log=lambda *a: None)
    pts = np.concatenate([v for v, _, _ in _fills(ue)["riser"]])
    # the kerb face stands on the line between the road and the pavement (within a cell), 15 cm plus its foot high
    mid = pts[(pts[:, 1] > -3.0) & (pts[:, 1] < 5.0)]
    assert len(mid) and np.all(np.abs(mid[:, 1] - 3.5) < 0.25)
    assert abs(mid[:, 2].max() - 16.347) < 0.01 and 16.15 < mid[:, 2].min() < 16.18
    # the ends of the patches, on nothing in this export, get a skirt under the ground like the export's plates
    ends = pts[(pts[:, 1] < -3.0) | (pts[:, 1] > 5.0)]
    assert len(ends) and ends[:, 2].min() < 15.8
    assert _open_steps(ue, -0.75, 0.75, -3.4, 5.4) < 0.15     # the fixture's own kerb outside the gap has no face
