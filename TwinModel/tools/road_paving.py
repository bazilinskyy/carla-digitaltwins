"""Brick and sett roads in a twin's ``ue/`` export: the road surface itself, not overlays on it.

OSM tags many carriageways ``surface=paving_stones`` (Dutch klinkers), ``bricks`` or ``sett``. The export paints
every road with asphalt; this splits the road meshes (kinds drivable, parking, crossing) by the surface of the road
each triangle belongs to and gives the paved pieces their own textured material (tools/paving_textures.py):

* ``clinker``: red-brown fired clay bricks in herringbone (paving_stones / bricks / clinker in the Netherlands);
* ``pavers``: grey concrete paving blocks in herringbone (paving_stones elsewhere, or ``colour=grey``);
* ``sett``: granite setts in courses (sett, cobblestone).

The geometry is the export's own road surface, so the paving is as smooth as the asphalt it replaces (overlays laid
on the road showed their edges, steps and seams). A triangle belongs to the highest-class car road whose carriageway
(its estimated half width + 0.5 m) holds its centre, else to the nearest car road within 12 m: an asphalt main road
keeps its surface through a junction and a brick side street starts at its edge, as in the street.

UVs are metres / TILE_M (2 m) in world axes; herringbone reads the same in every direction. The bake needs no change:
the new assets keep their kind, material and semantic and carry a ``texture`` key (bake_level.asset_key). With
``--xodr IN OUT`` the OpenDRIVE header gets ``<userData code="twinPaving" value="clinker,sett"/>`` so a simulator
that lays its own paving overlays (coupled-carla) knows the level has them built in.

Run after tools/landmarks.py (its passage fill is part of the road meshes); re-running landmarks restores the road
meshes it touched, so run this again after it. Re-running this restores from ``paving_backup/`` first.

    python tools/road_paving.py <build>/ue [--region NL] [--xodr IN OUT] [--restore]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shapely import STRtree  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402

import osm_detail  # noqa: E402
import paving_textures  # noqa: E402
from twinmodel.export.ue import gltf_to_model, read_glb, write_glb  # noqa: E402

BACKUP = "paving_backup"
ROAD_KINDS = ("drivable", "parking", "crossing")
TILE_M = paving_textures.TILE_M
PROVENANCE = "road paving: OSM surface of the carriageway, procedural texture (tools/road_paving.py)"

# car roads by class (higher wins in a junction) and their default half width (m) when OSM has no lanes/width
CAR_ROADS = {"motorway": (9, 7.0), "trunk": (8, 6.0), "primary": (7, 5.0), "secondary": (6, 4.5),
             "tertiary": (5, 3.8), "unclassified": (4, 3.2), "residential": (4, 3.2), "busway": (4, 3.2),
             "living_street": (3, 3.0), "service": (2, 2.8), "pedestrian": (1, 3.0), "road": (3, 3.0),
             "track": (0, 2.0)}
for _k in ("motorway", "trunk", "primary", "secondary", "tertiary"):
    CAR_ROADS[_k + "_link"] = (CAR_ROADS[_k][0], CAR_ROADS[_k][1] * 0.7)
BRICKS = {"paving_stones", "bricks", "brick", "clinker", "paved:bricks", "paving_stones:30"}
SETT = {"sett", "cobblestone", "unhewn_cobblestone", "cobblestone:flattened"}


def style_of(tags: dict, region: str) -> str | None:
    """The paving style of a road from its OSM tags, or None (asphalt, concrete, unpaved, untagged)."""
    s = (tags.get("surface") or "").strip()
    if s in SETT:
        return "sett"
    if s in BRICKS:
        colour = (tags.get("surface:colour") or tags.get("colour") or "").lower()
        if colour in ("grey", "gray", "#808080", "black", "white"):
            return "pavers"
        if colour in ("red", "brown", "#a52a2a") or region == "NL" or s in ("bricks", "brick", "clinker"):
            return "clinker"
        return "pavers"
    return None


def half_width(tags: dict, default: float) -> float:
    w = osm_detail._num(tags.get("width"), 0.0)
    if w > 0:
        return w / 2
    lanes = osm_detail._num(tags.get("lanes"), 0.0)
    return max(default, lanes * 1.6) if lanes else default


def car_roads(data: dict, to_xy, region: str) -> list:
    """[{line, style, rank, half, layer}] of the car roads in the OSM elements."""
    out = []
    for e in data["elements"]:
        t = e.get("tags", {})
        hw = t.get("highway")
        if e["type"] != "way" or hw not in CAR_ROADS or len(e.get("geometry") or []) < 2:
            continue
        if t.get("area") == "yes":
            continue
        rank, dh = CAR_ROADS[hw]
        line = LineString([to_xy(p["lat"], p["lon"]) for p in e["geometry"]])
        out.append({"id": e["id"], "line": line, "style": style_of(t, region), "rank": rank,
                    "half": half_width(t, dh), "layer": int(osm_detail._num(t.get("layer"), 0))})
    return out


def classify(points: np.ndarray, layer: int, roads: list, tree: STRtree, reach: float = 12.0) -> list:
    """The paving style (or None) at each point (model x, y) of a road mesh on this OSM layer."""
    out = []
    for x, y in points:
        p = Point(x, y)
        cand = [roads[i] for i in tree.query(p.buffer(reach))]
        if not cand:
            out.append(None)
            continue
        same = [r for r in cand if r["layer"] == layer] or cand     # a level fill over a layer -1 way: any layer
        d = [(r["line"].distance(p), r) for r in same]
        inside = [(r["rank"], -dd, r) for dd, r in d if dd <= r["half"] + 0.5]
        if inside:
            out.append(max(inside, key=lambda z: (z[0], z[1]))[2]["style"])
        else:
            dd, r = min(d, key=lambda z: z[0])
            out.append(r["style"] if dd <= reach else None)
    return out


def _backup(ue: Path, rel: str):
    src, dst = ue / rel, ue / BACKUP / rel
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def restore(ue: Path) -> bool:
    """Undo a previous run: the road meshes from the backup, paving assets and textures out of the manifest."""
    bk = ue / BACKUP
    mp = ue / "manifest.json"
    man = json.loads(mp.read_text())
    if not bk.exists() and not man.get("paving"):
        return False
    if bk.exists():
        for p in bk.rglob("*"):
            if p.is_file() and p.name != "manifest.json":
                dst = ue / p.relative_to(bk)
                shutil.copy2(p, dst)
        saved = bk / "manifest.json"
        if saved.exists():
            man = json.loads(saved.read_text())
    for a in [a for a in man["assets"] if a.get("paving")]:
        (ue / a["file"]).unlink(missing_ok=True)
    man["assets"] = [a for a in man["assets"] if not a.get("paving")]
    for k in [k for k in man.get("textures", {}) if k.startswith("paving_")]:
        del man["textures"][k]
    man.pop("paving", None)
    mp.write_text(json.dumps(man, indent=1))
    if bk.exists():
        shutil.rmtree(bk)
    return True


def _subset(v, n, f, keep):
    """Vertices, normals and re-indexed faces of the kept triangles."""
    ff = f[keep]
    used, inv = np.unique(ff.reshape(-1), return_inverse=True)
    return v[used], n[used], inv.reshape(-1, 3)


def apply(ue: Path, region: str = "NL", log=print) -> dict:
    ue = Path(ue)
    restore(ue)
    mp = ue / "manifest.json"
    man = json.loads(mp.read_text())
    (ue / BACKUP).mkdir(exist_ok=True)
    shutil.copy2(mp, ue / BACKUP / "manifest.json")
    data = osm_detail.fetch(man, ue / "osm_detail.json", log)
    roads = car_roads(data, osm_detail.project(man), region)
    paved = [r for r in roads if r["style"]]
    if not paved:
        log("road paving: no paved car roads in OSM")
        return {"paved_m2": {}}
    tree = STRtree([r["line"] for r in roads])
    name = man["name"]
    styles_used, area, new_assets = set(), {}, []
    for a in [a for a in man["assets"] if a["kind"] in ROAD_KINDS]:
        g = read_glb(ue / a["file"])
        v = gltf_to_model(g["positions"].astype(float))
        nrm = gltf_to_model(g["normals"].astype(float))
        f = g["faces"].astype(np.int64)
        if len(f) == 0:
            continue
        c = v[f].mean(axis=1)
        # quick reject: no paved road near this mesh at all
        x0, y0 = c[:, 0].min() - 15, c[:, 1].min() - 15
        x1, y1 = c[:, 0].max() + 15, c[:, 1].max() + 15
        if not any(r["line"].bounds[0] < x1 and r["line"].bounds[2] > x0 and r["line"].bounds[1] < y1
                   and r["line"].bounds[3] > y0 for r in paved):
            continue
        sty = np.array([s or "" for s in classify(c[:, :2], a["layer"], roads, tree)])
        if not (sty != "").any():
            continue
        _backup(ue, a["file"])
        tri_area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
        for s in sorted(str(x) for x in set(sty) - {""}):
            keep = sty == s
            vs, ns, fs = _subset(v, nrm, f, keep)
            uv = vs[:, :2] / TILE_M
            asset = f"{a['asset']}_pv_{s}"
            rel = f"meshes/paving/{asset}.glb"
            key = f"paving_{s}"
            (ue / rel).parent.mkdir(parents=True, exist_ok=True)
            bbox = write_glb(ue / rel, asset, vs, ns, uv, fs, key)
            new_assets.append({"file": rel, "asset": asset, "layer": a["layer"], "kind": a["kind"],
                               "material": a.get("material", "road"), "semantic": a.get("semantic", "Road"),
                               "tile": list(a["tile"]), "vertices": int(len(vs)), "triangles": int(len(fs)),
                               "bbox_ue": bbox, "texture": key, "paving": s, "provenance": PROVENANCE})
            styles_used.add(s)
            area[s] = area.get(s, 0.0) + float(tri_area[keep].sum())
        rest = sty == ""
        if rest.any():
            vr, nr, fr = _subset(v, nrm, f, rest)
            uvr = g["uvs"].astype(float)[np.unique(f[rest].reshape(-1))]
        else:       # every triangle is paved: keep one degenerate triangle so the asset stays valid
            vr, nr, fr, uvr = v[:1], nrm[:1], np.zeros((1, 3), np.int64), g["uvs"].astype(float)[:1]
        bbox = write_glb(ue / a["file"], a["asset"], vr, nr, uvr, fr, g["material"])
        a.update(vertices=int(len(vr)), triangles=int(len(fr)), bbox_ue=bbox)
    man["assets"].extend(new_assets)
    tex_dir = ue / "textures"
    for s in sorted(styles_used):
        man.setdefault("textures", {})[f"paving_{s}"] = paving_textures.write(s, tex_dir, name)
    report = {"paved_m2": {k: round(v_, 1) for k, v_ in sorted(area.items())}, "assets": len(new_assets),
              "region": region, "roads": {s: sum(1 for r in paved if r["style"] == s) for s in sorted(styles_used)}}
    man["paving"] = {"version": 1, "report": report, "files": [a["file"] for a in new_assets]}
    mp.write_text(json.dumps(man, indent=1))
    log(f"road paving: {report}")
    return report


def mark_xodr(src: Path, dst: Path, styles) -> None:
    """Copy the OpenDRIVE with <userData code="twinPaving" value="..."/> in its header."""
    text = Path(src).read_text(encoding="utf-8")
    text = re.sub(r'[ \t]*<userData code="twinPaving"[^>]*/>\r?\n?', "", text)
    tag = f'<userData code="twinPaving" value="{",".join(sorted(styles))}"/>'
    m = re.search(r"<header\b[^>]*?(/?)>", text)
    if m is None:
        raise ValueError("no <header> in " + str(src))
    if m.group(1) == "/":       # <header .../> -> <header ...>tag</header>
        text = text[:m.start()] + m.group(0)[:-2] + ">\n        " + tag + "\n    </header>" + text[m.end():]
    else:
        text = text[:m.end()] + "\n        " + tag + text[m.end():]
    Path(dst).write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ue")
    ap.add_argument("--region", default="NL", help="ISO country of the twin (NL: paving_stones are red klinkers)")
    ap.add_argument("--xodr", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args(argv)
    if args.restore:
        print("restored" if restore(Path(args.ue)) else "nothing to restore")
        return
    rep = apply(Path(args.ue), args.region)
    if args.xodr and rep.get("paved_m2"):
        mark_xodr(Path(args.xodr[0]), Path(args.xodr[1]), rep["paved_m2"].keys())
        print("xodr marked:", args.xodr[1])


if __name__ == "__main__":
    main()
