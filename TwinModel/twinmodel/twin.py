"""One command from a place to a baked CARLA 1.0 digital twin.

    python -m twinmodel twin "Plaza Mayor, Salamanca" --name SalamancaCentre --size 1.0
    python -m twinmodel twin --bbox 40.960 -5.670 40.969 -5.658 --name SalamancaCentre --bake \\
        --carla-root ~/carla58/carla --engine ~/carla58/UnrealEngine5_carla --carla-python ~/carla58/venv/bin/python

Runs the whole pipeline with the best open data of the place's country (twinmodel.regions):

1. place      find the place with OpenStreetMap's Nominatim (cached; or take --bbox), a square of --size km
2. build      ``twinmodel build --profile auto``: OSM roads, the national terrain model and aerial photo
3. export     ``twinmodel bake-export``: per-tile meshes and the manifest for Unreal
4. buildings  ``tools/real_buildings.py``: the national 3D or footprint model (3DBAG, PLATEAU, NYC, Catastro,
              BD TOPO), else Overture Maps, else OSM; facade styles, photo roofs, lit windows
5. ground     ``tools/ground_photo.py``: the aerial photo on open ground (where the country has an open one)
5b. clearance ``tools/road_clearance.py``: building geometry standing on the driving lanes (below 4.6 m) and
              sidewalk lanes (below 2.5 m) is cut out and closed as a passage, or trimmed to the kerb; then
              CARLA's lane samples are checked, and a twin with blocked lanes stops here (twin.json clearance)
6. credits    ATTRIBUTION.md (every source with its licence), copied next to the level by the bake
7. bake       ``ue/bake_level.py`` in the CARLA 1.0 editor (--bake; needs a CARLA ue58-dev source build with
              the editor), under a memory watchdog; the level is /Game/Carla/Maps/Twins/<Name>/<Name>
8. nav        with --no-nav-build: the two World Partition navigation builders, one editor at a time (less
              memory at once than the bake running them itself)

Progress is printed in plain words, one line per step (``[twin] ...``; coupled-carla's map maker reads them); each
step's full log goes to <out>/twin.log, and <out>/twin.json sums up the twin (numbers, sources, timings, level).
Steps already done are redone only with --force (or skipped with --skip). The toolchain paths can also come from
TWINMODEL_CARLA_ROOT, TWINMODEL_UE_ENGINE, TWINMODEL_CARLA_PYTHON or ~/.config/twinmodel/toolchain.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

PKG = Path(__file__).resolve().parent
TOOLS = PKG.parent / "tools"
UE_DIR = PKG.parent / "ue"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = {"User-Agent": "twinmodel (carla-digitaltwins)"}
STEPS = [
    ("place", "Finding the place"),
    ("build", "Building roads, terrain and aerial photo from open data"),
    ("export", "Cutting the meshes for Unreal"),
    ("buildings", "Putting in the real buildings with facade styles"),
    ("ground", "Laying the aerial photo on the ground"),
    ("clearance", "Keeping the roads and sidewalks clear of buildings"),
    ("credits", "Writing the data credits"),
    ("bake", "Baking the CARLA 1.0 level in the Unreal editor"),
    ("nav", "Building the pedestrian navigation (two more editor runs)"),
]
NAV_BUILDERS = ("CarlaCrosswalkNavBuilder", "WorldPartitionNavigationDataBuilder")
TOOLCHAIN_FILE = Path.home() / ".config" / "twinmodel" / "toolchain.json"


class TwinError(Exception):
    def __init__(self, message, advice=""):
        super().__init__(message)
        self.advice = advice


def say(kind: str, step: str, text: str = ""):
    """One progress line: ``[twin] <kind> <step>: <text>`` (kind: start, done, note, skip, failed)."""
    print(f"[twin] {kind} {step}: {text}".rstrip(), flush=True)


# --------------------------------------------------------------------------- place

def camel_name(text: str, fallback: str = "Twin") -> str:
    """'Plaza Mayor, Salamanca' -> 'PlazaMayorSalamanca' (ASCII letters and digits, starts with a letter, <= 32)."""
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = "".join(c for c in t if not unicodedata.combining(c)).encode("ascii", "ignore").decode()
    words = [w for w in re.split(r"[^A-Za-z0-9]+", t) if w]
    name = "".join(w[:1].upper() + w[1:] for w in words)[:32]
    if not name or not name[0].isalpha():
        name = fallback + name
    return name[:32]


def square_bbox(lat: float, lon: float, km: float) -> list:
    """[s, w, n, e] of a km x km square (as coupled-carla's map maker)."""
    half = km / 2.0
    dlat = half / 111.32
    dlon = half / (111.32 * math.cos(math.radians(lat)))
    return [round(lat - dlat, 5), round(lon - dlon, 5), round(lat + dlat, 5), round(lon + dlon, 5)]


def geocode(place: str, cache_dir: Path) -> dict:
    """{lat, lon, name, country_code} of the best Nominatim match (cached in <cache>/geocode.json). Only the typed
    place is sent."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / "geocode.json"
    try:
        cache = json.loads(cp.read_text())
    except (OSError, ValueError):
        cache = {}
    key = " ".join(place.lower().split())
    if key not in cache:
        import requests
        time.sleep(1.1)   # Nominatim: at most one request a second
        r = requests.get(NOMINATIM, params={"q": place, "format": "jsonv2", "limit": 1, "addressdetails": 1,
                                            "accept-language": "en"}, headers=UA, timeout=30)
        r.raise_for_status()
        res = r.json()
        if not res:
            raise TwinError(f'I could not find "{place}".', "Check the spelling, add the town or country, or give "
                                                            "--bbox S W N E.")
        a = res[0].get("address") or {}
        cache[key] = {"lat": float(res[0]["lat"]), "lon": float(res[0]["lon"]),
                      "name": res[0].get("display_name", place),
                      "country_code": (a.get("country_code") or "").upper() or None}
        cp.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return cache[key]


def seed_country(cache_dir: Path, bbox, iso2: Optional[str]):
    """Write the build's country cache (ingest.osm.country_for_bbox) from the geocoder's answer: one Overpass
    query less."""
    if not iso2:
        return
    from .ingest.osm import _country_cache_path
    s, w, n, e = bbox
    p = _country_cache_path(cache_dir, (s + n) / 2.0, (w + e) / 2.0)
    if not p.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"iso2": iso2, "source": "nominatim"}))


# --------------------------------------------------------------------------- toolchain

def toolchain(args) -> dict:
    """CARLA 1.0 source tree, UE 5.8 engine and the python with the carla wheel: arguments, environment, then
    ~/.config/twinmodel/toolchain.json."""
    try:
        cfg = json.loads(TOOLCHAIN_FILE.read_text())
    except (OSError, ValueError):
        cfg = {}
    tc = {"carla_root": args.carla_root or os.environ.get("TWINMODEL_CARLA_ROOT") or cfg.get("carla_root"),
          "engine": args.engine or os.environ.get("TWINMODEL_UE_ENGINE") or cfg.get("engine"),
          "carla_python": args.carla_python or os.environ.get("TWINMODEL_CARLA_PYTHON") or cfg.get("carla_python")}
    return {k: (str(Path(v).expanduser()) if v else None) for k, v in tc.items()}


def editor_paths(tc: dict):
    """(UnrealEditor-Cmd, CarlaUnreal.uproject) or a TwinError saying what is missing."""
    if not tc.get("carla_root") or not tc.get("engine"):
        raise TwinError("The bake needs a CARLA 1.0 source build with the Unreal editor.",
                        "Give --carla-root (a built ue58-dev checkout) and --engine (its UE 5.8 engine), or set "
                        "them in ~/.config/twinmodel/toolchain.json; without them the twin stops before the bake "
                        "(its manifest can be baked later on a machine that has the editor).")
    root, eng = Path(tc["carla_root"]), Path(tc["engine"])
    uproject = root / "Unreal" / "CarlaUnreal" / "CarlaUnreal.uproject"
    exe = next((p for p in (eng / "Engine/Binaries/Linux/UnrealEditor-Cmd",
                            eng / "Engine/Binaries/Win64/UnrealEditor-Cmd.exe",
                            eng / "Engine/Binaries/Mac/UnrealEditor-Cmd") if p.exists()), None)
    if exe is None or not uproject.exists():
        raise TwinError(f"No CARLA 1.0 editor at {root} with engine {eng}.",
                        "Build CARLA ue58-dev with the editor (carla-unreal-editor target) first.")
    return exe, uproject


# --------------------------------------------------------------------------- running steps

def mem_available_gb() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    except OSError:
        return None
    return None


def group_rss_gb(sid: int) -> float:
    """Total RSS of the processes in session ``sid`` (Linux /proc), GB."""
    total = 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if os.getsid(int(d.name)) != sid:
                continue
            for line in (d / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except (OSError, ProcessLookupError, ValueError):
            continue
    return total / 1048576.0


def run(cmd, log_path: Path, cwd=None, watchdog: Optional[dict] = None, nice: int = 0) -> dict:
    """Run a step, its output appended to the log. ``watchdog`` {max_rss_gb, min_avail_gb} (Linux): the step's
    whole process session is killed when its RSS passes max_rss_gb or the machine's available memory falls
    under min_avail_gb. Returns {rc, seconds, peak_rss_gb, killed, tail}."""
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
        logf.flush()
        pre = None
        if nice and hasattr(os, "nice"):
            pre = lambda: os.nice(nice)  # noqa: E731
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=logf, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True,
                                preexec_fn=pre if os.name == "posix" else None)
        peak, killed = 0.0, None
        while proc.poll() is None:
            time.sleep(5.0)
            if watchdog and sys.platform.startswith("linux"):
                rss = group_rss_gb(proc.pid)
                peak = max(peak, rss)
                avail = mem_available_gb()
                if rss > watchdog["max_rss_gb"] or (avail is not None and avail < watchdog["min_avail_gb"]):
                    killed = f"watchdog: RSS {rss:.1f} GB, {avail:.1f} GB available"
                    logf.write(killed + "\n")
                    logf.flush()
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        rc = proc.wait()
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "peak_rss_gb": round(peak, 2), "killed": killed,
            "tail": tail}


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- the command

def twin(args) -> int:
    t_start = time.time()
    cache = Path(args.cache).expanduser()
    summary = {"steps": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    # 1. place
    say("start", "place", STEPS[0][1])
    iso2 = None
    if args.bbox:
        bbox = [float(v) for v in args.bbox]
        label = args.name or "twin"
    else:
        if not args.place:
            raise TwinError("Which place?", 'Give a place ("Plaza Mayor, Salamanca") or --bbox S W N E.')
        g = geocode(" ".join(args.place), cache)
        bbox = square_bbox(g["lat"], g["lon"], args.size)
        iso2, label = g["country_code"], g["name"]
    s, w, n, e = bbox
    if not (s < n and w < e) or (n - s) * 111.32 > 4.0:
        raise TwinError(f"The box {bbox} is empty or larger than 4 km.", "Use a square of 0.5 to 2 km.")
    name = args.name or camel_name(" ".join(args.place or []))
    out = Path(args.out or Path("build") / re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "twin.log"
    seed_country(cache, bbox, iso2)
    from . import regions
    plan = regions.plan_for(iso2, bbox) if iso2 else None
    say("done", "place", f"{label} -> {name}, box {bbox}" + (f" ({iso2})" if iso2 else ""))
    for note in (plan.notes if plan else ()):
        say("note", "place", note)
    summary.update(name=name, place=label, bbox=bbox, country=iso2, out=str(out))
    skip = set((args.skip or "").split(",")) - {""}
    py = sys.executable

    def step(key, cmd, done_test, text_fn, watchdog=None, nice=0, ok_rc=(0,), outputs=None):
        title = dict(STEPS)[key]
        if key in skip or (done_test() and not args.force):
            say("skip", key, "done before" if key not in skip else "skipped (--skip)")
            before = ((_json(out / "twin.json") or {}).get("steps") or {}).get(key) or {}
            summary["steps"][key] = dict(before, skipped=True)   # the numbers of the run that did it
            return True
        say("start", key, title)
        res = run(cmd, log_path, watchdog=watchdog, nice=nice)
        summary["steps"][key] = {k: res[k] for k in ("rc", "seconds", "peak_rss_gb", "killed")}
        missing = [str(p) for p in (outputs or ()) if not Path(p).exists()]
        if res["rc"] not in ok_rc or res["killed"] or missing:
            why = res["killed"] or (f"exit code {res['rc']}" + (f", no {missing[0]}" if missing else ""))
            say("failed", key, why + f"; the log is {log_path}")
            for line in res["tail"][-12:]:
                print("    " + line, flush=True)
            return False
        say("done", key, text_fn())
        return True

    twin_dir, xodr, man_path = out / f"{name}.twin", out / f"{name}.xodr", out / "ue" / "manifest.json"
    # 2. build (exit 1 = the OpenDRIVE has validation findings; the twin is still written and usable)
    build_cmd = [py, "-m", "twinmodel", "build", "--bbox", *map(str, bbox), "--name", name, "--out", out,
                 "--cache", cache, "--profile", args.profile]
    if not step("build", build_cmd, lambda: xodr.exists() and (twin_dir / "model.json").exists(),
                lambda: _build_text(out), ok_rc=(0, 1), outputs=(xodr, twin_dir / "model.json")):
        raise TwinError("The road network could not be built.", f"See {log_path}.")
    if summary["steps"]["build"].get("rc") == 1:
        say("note", "build", "the OpenDRIVE has validation findings (report.json); it loads in CARLA")
    link_cache(cache, out, bbox)
    rep = _json(out / "report.json") or {}
    iso2 = iso2 or (rep.get("profile") or {}).get("iso2")
    plan = regions.plan_for(iso2, bbox)
    summary["country"], summary["region_plan"] = iso2, plan.as_dict()
    # 3. export
    if not step("export", [py, "-m", "twinmodel", "bake-export", out, name], man_path.exists,
                lambda: _export_text(man_path)):
        raise TwinError("The meshes for Unreal could not be written.", f"See {log_path}.")
    # 4. buildings
    man = _json(man_path) or {}
    bcmd = [py, TOOLS / "real_buildings.py", out] + (["--no-global"] if args.no_global_buildings else [])
    if not step("buildings", bcmd, lambda: bool((_json(man_path) or {}).get("stats", {}).get("real_buildings")),
                lambda: _buildings_text(man_path)):
        say("note", "buildings", "the OSM building blocks stay")
    # 5. ground photo
    img = ((rep.get("build") or {}).get("imagery") or {}).get("source")
    photo = img and img != "osm_tiles"
    if photo or plan.iso2 == "NL":
        if not step("ground", [py, TOOLS / "ground_photo.py", out],
                    lambda: bool((_json(man_path) or {}).get("stats", {}).get("ground_photo")),
                    lambda: _ground_text(man_path)):
            say("note", "ground", "CARLA's ground materials stay")
    else:
        say("skip", "ground", "no open aerial photo here: CARLA's ground materials stay")
    # 5b. clearance (always redone: it starts from the uncut meshes, about a minute)
    if man_path.exists():
        ccmd = [py, TOOLS / "road_clearance.py", out, "--max-road", str(args.clearance_max_road),
                "--max-walk", str(args.clearance_max_walk)]
        ok = step("clearance", ccmd, lambda: False, lambda: _clearance_text(out))
        rec = _json(out / "clearance.json") or {}
        if rec:
            summary["clearance"] = {"ok": rec.get("ok"), "cut": rec.get("cut"),
                                    "after": {g: {k: v for k, v in r.items() if k not in ("hits", "worst")}
                                              for g, r in (rec.get("after") or {}).items()}}
        if not ok:
            _write_summary(out, summary, man_path, t_start)
            raise TwinError("Buildings still stand on the lanes after the clearance cut.",
                            f"See {out / 'clearance.json'} (every blocked lane sample with lat/lon); raise "
                            "--clearance-max-road/--clearance-max-walk or --skip clearance to bake anyway.")
    # 6. credits
    say("start", "credits", STEPS[6][1])
    man = _json(man_path) or {}
    text = regions.attribution_text(name, rep, man, extra=("nominatim",) if args.place else ())
    (out / "ATTRIBUTION.md").write_text(text, encoding="utf-8")
    (out / "ue" / "ATTRIBUTION.md").write_text(text, encoding="utf-8")
    # the place record: travels with the level (bake_level.copy_place), read by tools/stage_sidecars.py for the
    # level's country (the OpenDRIVE's own signal stamp was the street profile's on twins baked before 2026-11-20)
    place = {k: summary.get(k) for k in ("name", "place", "bbox", "country")}
    (out / "ue" / "place.json").write_text(json.dumps(place, indent=1, ensure_ascii=False))
    if man:
        man["attribution"] = "ATTRIBUTION.md"
        man["place"] = "place.json"
        man_path.write_text(json.dumps(man, indent=1))
    say("done", "credits", f"{len(regions.used_sources(rep, man))} sources -> {out / 'ATTRIBUTION.md'}")
    # 7. bake
    level = f"/Game/Carla/Maps/Twins/{name}/{name}"
    summary.update(level=level, xodr=str(xodr), manifest=str(man_path))
    if args.bake:
        tc = toolchain(args)
        exe, uproject = editor_paths(tc)
        if sys.platform.startswith("linux") and args.min_start_gb:
            waited = 0
            while (mem_available_gb() or 0) < args.min_start_gb:
                if waited >= args.wait_mem_min * 60:
                    raise TwinError(f"Not enough free memory for the bake ({mem_available_gb():.0f} GB available, "
                                    f"{args.min_start_gb:g} GB needed).", "Try again later or with --no-nav-build.")
                if waited == 0:
                    say("note", "bake", f"waiting for {args.min_start_gb:g} GB of free memory")
                time.sleep(30)
                waited += 30
        script = (f"{UE_DIR / 'bake_level.py'} --manifest {man_path.resolve()} --name {name} "
                  f"--report {(out / 'bake_report.json').resolve()}")
        if tc.get("carla_python"):
            script += f" --python-bin {tc['carla_python']}"
        if args.no_nav_build:
            script += " --no-nav-build"
        cmd = [exe, uproject, "-run=pythonscript", f"-script={script}", "-unattended", "-nosplash", "-nullrhi",
               "-nosound"]
        wd = {"max_rss_gb": args.max_rss_gb, "min_avail_gb": args.min_avail_gb}
        if not step("bake", cmd, lambda: False, lambda: _bake_text(out), watchdog=wd, nice=10):
            summary["steps"]["bake"]["failed"] = True
            _write_summary(out, summary, man_path, t_start)
            raise TwinError("The bake in the Unreal editor failed.", f"See {log_path}.")
        if args.no_nav_build and "nav" not in skip:
            say("start", "nav", dict(STEPS)["nav"])
            nav = {}
            for b in NAV_BUILDERS:
                editor = exe.with_name("UnrealEditor" + exe.suffix)
                res = run([editor if editor.exists() else exe, uproject, level, "-run=WorldPartitionBuilderCommandlet",
                           f"-Builder={b}", "-SCCProvider=None", "-nullrhi", "-unattended", "-stdout",
                           "-FullStdOutLogOutput"], log_path, watchdog=wd, nice=10)
                # the editor can exit non-zero in its shutdown handler after the builder saved: trust its result line
                ok = not res["killed"] and (res["rc"] == 0 or any("finished execution (result 0)" in t
                                                                    for t in res["tail"]))
                nav[b] = {"ok": ok, "seconds": res["seconds"], "peak_rss_gb": res["peak_rss_gb"],
                          "killed": res["killed"]}
            summary["steps"]["nav"] = nav
            if all(v["ok"] for v in nav.values()):
                say("done", "nav", "crosswalks and navigation mesh built")
            else:
                say("failed", "nav", "a navigation builder failed (walkers then cannot find paths); the log is "
                                     f"{log_path}")
    else:
        say("skip", "bake", "no --bake: bake the manifest later with ue/bake_level.py in the CARLA 1.0 editor")
    _write_summary(out, summary, man_path, t_start)
    say("done", "twin", f"{name} ready in {(time.time() - t_start) / 60:.1f} min: {out / 'twin.json'}")
    return 0


def link_cache(cache: Path, out: Path, bbox) -> int:
    """Link the build's cached terrain grid, aerial photo and Overpass extract of this bbox into <out>/data/, where the
    buildings, ground-photo and clearance tools look for them (realbuildings.find_dem / find_ortho search the build
    folder and its parents)."""
    s, w, n, e = bbox
    stem = "%.5f_%.5f_%.5f_%.5f_" % (s, w, n, e)
    dst = out / "data"
    if cache.resolve() == dst.resolve():
        return 0
    k = 0
    for src in list(cache.glob("dem_" + stem + "*.npz")) + list(cache.glob("ortho_" + stem + "*.tif")) + \
            list(cache.glob("overpass_" + stem + "*.json")):    # the streets' maxheight/covered tags (clearance)
        dst.mkdir(parents=True, exist_ok=True)
        link = dst / src.name
        if not link.exists():
            try:
                link.symlink_to(src.resolve())
            except OSError:
                import shutil
                shutil.copyfile(src, link)
            k += 1
    return k


def _build_text(out: Path) -> str:
    rep = _json(out / "report.json") or {}
    b = rep.get("build") or {}
    el = rep.get("elevation") or {}
    prof = rep.get("profile") or {}
    parts = []
    if el.get("applied"):
        parts.append(f"terrain {el.get('source')} {el.get('z_min', 0):.0f}-{el.get('z_max', 0):.0f} m")
    img = (b.get("imagery") or {}).get("source")
    parts.append(f"photo {img or 'none'}")
    parts.append(f"profile {prof.get('name')}")
    return ", ".join(parts)


def _export_text(man_path: Path) -> str:
    st = (_json(man_path) or {}).get("stats", {})
    return f"{st.get('assets', '?')} meshes, {st.get('spawn_points', '?')} spawn points"


def _buildings_text(man_path: Path) -> str:
    st = (_json(man_path) or {}).get("stats", {})
    return f"{st.get('buildings', 0):,} buildings from {st.get('real_buildings')}, styles {st.get('facade_styles')}"


def _ground_text(man_path: Path) -> str:
    gp = (_json(man_path) or {}).get("stats", {}).get("ground_photo") or {}
    return f"{gp.get('tiles', 0)} ground tiles with the aerial photo"


def _clearance_text(out: Path) -> str:
    rec = _json(out / "clearance.json") or {}
    a, cut = rec.get("after") or {}, rec.get("cut") or {}
    if not a:
        return f"{cut.get('passage', 0)} passages, {cut.get('trim', 0)} trims (not checked: no carla module)"
    return (f"{cut.get('passage', 0)} passages, {cut.get('trim', 0)} trims; blocked lane samples: "
            f"{a['road']['blocked']} driving, {a['walk']['blocked']} sidewalk")


def _bake_text(out: Path) -> str:
    rep = _json(out / "bake_report.json") or {}
    return (f"{rep.get('actors_placed', '?')} actors, {rep.get('textured_materials', '?')} textured materials, "
            f"{rep.get('content_total_mb', '?')} MB of content")


def _write_summary(out: Path, summary: dict, man_path: Path, t_start: float):
    man = _json(man_path) or {}
    st = man.get("stats", {})
    rep = _json(out / "report.json") or {}
    bake = _json(out / "bake_report.json") or {}
    tex = man.get("textures") or {}
    summary.update(
        roads=len(_json(out / f"{summary['name']}.twin" / "roads.geojson").get("features", []))
        if (out / f"{summary['name']}.twin" / "roads.geojson").exists() else None,
        buildings=st.get("buildings"), building_source=st.get("real_buildings"),
        facade_styles=st.get("facade_styles"), textures=len(tex),
        texture_mb=round(sum((out / "ue" / v["base_color"]).stat().st_size for v in tex.values()
                             if isinstance(v, dict) and v.get("base_color")
                             and (out / "ue" / v["base_color"]).exists()) / 1e6, 1),
        meshes=st.get("assets"), triangles=st.get("triangles"), spawn_points=st.get("spawn_points"),
        imagery=((rep.get("build") or {}).get("imagery") or {}).get("source"),
        elevation=(rep.get("elevation") or {}).get("source"),
        ground_photo_tiles=(st.get("ground_photo") or {}).get("tiles"),
        bake={k: bake.get(k) for k in ("actors_placed", "textured_materials", "content_total_mb", "seconds",
                                       "level_saved", "tm_cache")} if bake else None,
        seconds=round(time.time() - t_start, 1))
    (out / "twin.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))


def add_parser(sub):
    t = sub.add_parser("twin", help="a place -> a baked CARLA 1.0 digital twin, with the best open data of its "
                                    "country (twinmodel.twin)")
    t.add_argument("place", nargs="*", help='a place ("Plaza Mayor, Salamanca"); or --bbox')
    t.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    t.add_argument("--size", type=float, default=1.0, help="side of the square around the place, km (default 1)")
    t.add_argument("--name", help="level name (CamelCase; default from the place)")
    t.add_argument("--out", help="build directory (default build/<name>)")
    t.add_argument("--cache", default="data", help="download cache (Overpass, photo, terrain)")
    t.add_argument("--profile", default="auto", help="region profile (default auto: by country)")
    t.add_argument("--skip", help="comma-separated steps to skip: build,export,buildings,ground,clearance,bake,nav")
    t.add_argument("--force", action="store_true", help="redo steps done before")
    t.add_argument("--no-global-buildings", action="store_true",
                   help="where no national building model exists keep the OSM blocks (no Overture/OSM prisms)")
    t.add_argument("--clearance-max-road", type=int, default=0,
                   help="driving-lane samples a building may still block after the clearance cut (default 0)")
    t.add_argument("--clearance-max-walk", type=int, default=0,
                   help="sidewalk-lane samples a building may still block after the clearance cut (default 0)")
    t.add_argument("--bake", action="store_true", help="bake the level (needs the CARLA 1.0 editor)")
    t.add_argument("--carla-root", help="CARLA ue58-dev source tree, built with the editor")
    t.add_argument("--engine", help="its UE 5.8 engine root (holds Engine/Binaries)")
    t.add_argument("--carla-python", help="python with the carla 1.0 wheel (the Traffic Manager cache)")
    t.add_argument("--no-nav-build", action="store_true", help="bake without the two navigation builders "
                                                                "(less memory; run them later)")
    t.add_argument("--max-rss-gb", type=float, default=24.0, help="watchdog: kill the bake above this RSS")
    t.add_argument("--min-avail-gb", type=float, default=8.0, help="watchdog: kill the bake below this free memory")
    t.add_argument("--min-start-gb", type=float, default=0.0, help="wait for this much free memory before baking")
    t.add_argument("--wait-mem-min", type=float, default=30.0, help="... for at most this many minutes")
    t.set_defaults(func=main_args)
    return t


def main_args(args) -> int:
    try:
        return twin(args)
    except TwinError as err:
        print(f"[twin] failed twin: {err}", flush=True)
        if err.advice:
            print(f"[twin] advice: {err.advice}", flush=True)
        return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="twinmodel twin", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    add_parser(sub)
    args = ap.parse_args(["twin", *(sys.argv[1:] if argv is None else argv)])
    return args.func(args)


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
