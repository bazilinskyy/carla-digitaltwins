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
