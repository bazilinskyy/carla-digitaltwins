"""Keep a twin's roads and sidewalks clear of buildings, then check that they are.

    python tools/road_clearance.py build/<name>                 # after real_buildings.py (twin step "clearance")
    python tools/road_clearance.py build/<name> --check-only    # measure only
    python tools/road_clearance.py build/<name> --before        # also measure the uncut meshes first

The cut (twinmodel.export.clearance.apply) removes the building geometry that stands in the road corridor below
the clearance (4.6 m above the road, the street's OSM maxheight when it has one; 2.5 m along the sidewalk lanes)
and closes it as a passage, or trims a footprint that only grazes the carriageway back to the kerb. It starts
from the uncut meshes each time (ue/meshes/uncut/), so it can run again after any change.

The check reads the twin's OpenDRIVE with CARLA's own parser (the carla wheel in this interpreter) and counts the
1 m driving-lane and sidewalk-lane samples that are blocked by a building (a wall across the lane or a surface
less than 4.0 m / 2.2 m above it, or inside a building). The result goes to <build>/clearance.json (every blocked
sample with lat/lon) and to twin.json ``clearance``. Exit code 3 when more samples are blocked than --max-road /
--max-walk allow (default 0): the twin is not ready to bake.
"""
import argparse
import json
import sys
from pathlib import Path

from twinmodel.export import clearance


def find_xodr(build: Path, man: dict):
    p = man.get("xodr")
    if p and Path(p).exists():
        return Path(p)
    hits = sorted(build.glob("*.xodr"))
    return hits[0] if hits else None


def summary(res: dict) -> dict:
    return {g: {k: v for k, v in r.items() if k != "hits"} for g, r in res.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build_dir")
    ap.add_argument("--xodr", help="OpenDRIVE of the twin (default: the manifest's, else <build>/*.xodr)")
    ap.add_argument("--osm", action="append", default=[], help="Overpass JSON with the streets' maxheight/covered "
                    "tags (default: <build>/data/overpass_*.json)")
    ap.add_argument("--check-only", action="store_true", help="measure, do not cut")
    ap.add_argument("--before", action="store_true", help="measure the uncut meshes before cutting")
    ap.add_argument("--road-m", type=float, default=clearance.ROAD_CLEARANCE_M)
    ap.add_argument("--walk-m", type=float, default=clearance.WALK_CLEARANCE_M)
    ap.add_argument("--max-road", type=int, default=0, help="blocked driving-lane samples allowed (default 0)")
    ap.add_argument("--max-walk", type=int, default=0, help="blocked sidewalk-lane samples allowed (default 0)")
    args = ap.parse_args(argv)
    build = Path(args.build_dir)
    out = build / "ue"
    man = json.loads((out / "manifest.json").read_text())
    xodr = Path(args.xodr) if args.xodr else find_xodr(build, man)
    lanes = clearance.xodr_lanes(xodr) if xodr else None
    if lanes is None:
        print("clearance: no carla module or no xodr: the corridor is the road meshes only and nothing can be "
              "checked", flush=True)
    else:
        print(f"clearance: {len(lanes)} lane samples from {xodr.name}", flush=True)
    rec = {"xodr": str(xodr) if xodr else None}
    if args.before and lanes is not None:
        uncut = json.loads(json.dumps(man))
        if clearance.restore(out, uncut):
            # restore moved the uncut meshes back: measure, then cut again below
            man = uncut
        rec["before"] = clearance.check(build, lanes, man, streets=clearance.osm_street_zones(build, args.osm))
        print("clearance: before " + _line(rec["before"]), flush=True)
        if args.check_only:
            (out / "manifest.json").write_text(json.dumps(man, indent=1))
    if not args.check_only:
        streets = clearance.osm_street_zones(build, args.osm)
        rec["cut"] = clearance.apply(build, man, lanes=lanes, streets=streets, road_m=args.road_m,
                                     walk_m=args.walk_m)
        man = json.loads((out / "manifest.json").read_text())
    ok = True
    if lanes is not None:
        rec["after"] = clearance.check(build, lanes, man, streets=clearance.osm_street_zones(build, args.osm))
        a = rec["after"]
        ok = a["road"]["blocked"] <= args.max_road and a["walk"]["blocked"] <= args.max_walk
        print(("clearance: " if ok else "clearance: FAILED, ") + _line(a), flush=True)
        for g in ("road", "walk"):
            for w in a[g]["worst"][:5]:
                print(f"    {g} road {w['road']}: {w['blocked']} samples at {w['lat']},{w['lon']}", flush=True)
    rec["ok"] = ok
    rec["limits"] = {"road": args.max_road, "walk": args.max_walk}
    (build / "clearance.json").write_text(json.dumps(rec, indent=1))
    tj = build / "twin.json"
    try:
        twin = json.loads(tj.read_text()) if tj.exists() else None
    except ValueError:
        twin = None
    if twin is not None:
        twin["clearance"] = {"ok": ok, "cut": rec.get("cut"),
                             **{k: summary(rec[k]) for k in ("before", "after") if k in rec}}
        for k in ("before", "after"):
            for g in twin["clearance"].get(k, {}).values():
                g.pop("worst", None)
        tj.write_text(json.dumps(twin, indent=1, ensure_ascii=False, default=str))
    return 0 if ok else 3


def _line(r: dict) -> str:
    return (f"{r['road']['blocked']} of {r['road']['samples']} driving-lane samples blocked on {r['road']['roads']} "
            f"roads ({r['road']['crossings']} wall crossings), {r['walk']['blocked']} of {r['walk']['samples']} "
            f"sidewalk samples on {r['walk']['roads']} roads")


if __name__ == "__main__":
    sys.exit(main())
