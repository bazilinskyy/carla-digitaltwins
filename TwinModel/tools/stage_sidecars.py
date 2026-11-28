"""Stage the twins' OpenDRIVE files for a package build, per level, and write the package's level manifest.

    python tools/stage_sidecars.py --content C:/cc58/carla/Unreal/CarlaUnreal/Content --levels levels.json \
        --manifest C:/cc58/carla/Unreal/CarlaUnreal/Content/Carla/Maps/OpenDrive/coupled_levels.json

For every twin level (Content/Carla/Maps/Twins/<Name>/OpenDrive/<Name>.xodr) a copy goes to
Content/Carla/Maps/OpenDrive/<Name>.xodr, where UOpenDrive::FindPathToXODRFile looks first and CARLA's own
DefaultGame.ini stages loose files into the pak (no per-twin staging lines). In that server copy:

* sign signals (274 speed limit, 205 yield, 206 stop, 306 priority road) are left out wherever CARLA's stock plates
  are not the country's design (ue/server_signs.py: everywhere but the US-design countries), so CARLA spawns no US
  "SPEED LIMIT" plates in Eindhoven, Salamanca or Tokyo; the client draws the real signs (NDW in the Netherlands,
  OSM elsewhere, in the country's design);
* with ``light_actors: none`` the traffic-light and pedestrian-light signals (1000001, 1000002) are left out too:
  CARLA then spawns no traffic-light actors (EhvCity: 2,687, ~10 ms of every server tick), and the simulator's own
  lights (SUMO's programmes, coupled-carla's signal heads) are the only ones; CARLA's Traffic Manager and agents see
  no lights on that level.

The levels file says per level {"light_actors": "carla" | "none", "signs": "auto" | "none" | "stock",
"country": "NL"}; levels it does not name get DEFAULTS. The country comes, first match wins, from the levels file,
the twin's place record (Twins/<Name>/place.json, written by ``twinmodel twin`` and copied next to the level by
the bake), the place at the OpenDRIVE's geoReference origin (twinmodel.ingest.osm.country_for_bbox: cached, one
Overpass query on a miss), LEVEL_DEFAULTS, and last the OpenDRIVE's own signal stamp, which on twins baked before
the place's country was stamped is the street profile's calibration country (eu_dense: ES, so HelsinkiSenate and
ZurichHB read ES). The manifest records, per level, what was staged: light_actors, the sign policy, the country
and where it came from, and the dropped signal counts, for analyses and the docs.

The level's own copy (Twins/<Name>/OpenDrive/<Name>.xodr, what an editor -game server reads) loses the same sign
signals (light signals stay), the full file kept outside Content (<project>/Saved/xodr_full); any other .xodr in a
level's OpenDrive folder is moved there too: the editor server reads the first *.xodr of that folder, and a full
backup copy kept beside the level's own put every stock US sign back into the twin clips (24-28 Nov 2026).
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ue"))
import server_signs  # noqa: E402
import xodr_drop_signals  # noqa: E402

LIGHT_TYPES = ("1000001", "1000002")
DEFAULTS = {"light_actors": "carla", "signs": "auto", "country": None}
# light_actors none where CARLA's light actors cost more than 2 ms per server tick: EhvCity (2,687 actors, ~10 ms;
# city twin 2026-11-21: tick 15 -> 4.4 ms, RTF 0.986-0.992). TUeCampus (94 light signals) saved 0.6 ms in the
# 2026-11-21 switch-off trials (carla_tick 23.1 -> 22.5 ms): CARLA's lights stay there and on the other twins.
LEVEL_DEFAULTS = {"EhvCity": {"light_actors": "none", "country": "NL"},
                  "TUeCampus": {"country": "NL"}, "TUeCampusWin": {"country": "NL"},
                  "EhvCentre": {"country": "NL"}, "AmsterdamDam": {"country": "NL"}}


GEO = re.compile(r"\+lat_0=([-+0-9.eE]+).*?\+lon_0=([-+0-9.eE]+)", re.S)


def xodr_origin(path, head=65536):
    """(lat, lon) of the OpenDRIVE's geoReference origin (+lat_0 / +lon_0), or None."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read(head)
    m = GEO.search(text[text.find("<geoReference"):] if "<geoReference" in text else "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def twinmodel_lookup(lat, lon):
    """ISO2 of the country at (lat, lon) by twinmodel's own lookup (cached in TwinModel/data), or None."""
    try:
        sys.path.insert(0, str(HERE.parent))
        from twinmodel.ingest import osm
        return osm.country_for_bbox((lat, lon, lat, lon), HERE.parent / "data")
    except Exception:  # noqa: BLE001 - no twinmodel here, no network: the next source decides
        return None


def place_country(level_dir, xodr, lookup=twinmodel_lookup):
    """(ISO2, source) of the twin's place: its place record, else the country at its OpenDRIVE origin."""
    rec = Path(level_dir) / "place.json"
    if rec.exists():
        try:
            c = (json.loads(rec.read_text(encoding="utf-8")).get("country") or "").upper()
            if c:
                return c, "place record"
        except ValueError:
            pass
    o = xodr_origin(xodr)
    if o:
        c = (lookup(*o) or "").upper()
        if c:
            return c, "OpenDRIVE origin"
    return None, None


def plan(name, cfg, xodr_country, place=(None, None)):
    """(signal types to drop, the manifest record) for one level. place: place_country()'s answer."""
    cfg = cfg or {}
    c = {**DEFAULTS, **LEVEL_DEFAULTS.get(name, {}), **cfg}
    if c["light_actors"] not in ("carla", "none"):
        raise ValueError(f"{name}: light_actors {c['light_actors']!r} (carla or none)")
    for country, source in ((cfg.get("country"), "levels file"), place,
                            (LEVEL_DEFAULTS.get(name, {}).get("country"), "level default"),
                            (xodr_country, "OpenDRIVE signals")):
        if country:
            country = country.upper()
            break
    else:
        country, source = None, None
    drop = set(server_signs.types_to_drop(c["signs"], country))
    if c["light_actors"] == "none":
        drop |= set(LIGHT_TYPES)
    rec = {"light_actors": c["light_actors"], "signs": "client" if set(server_signs.SIGN_TYPES) <= drop else "carla",
           "country": country, "country_source": source, "dropped_types": sorted(drop)}
    return drop, rec


def tidy_level_folder(folder, name, aside):
    """Moves every .xodr but <name>.xodr out of a level's OpenDrive folder into ``aside``; returns their names.
    CARLA's editor server reads the FIRST *.xodr of that folder (UOpenDrive::LoadXODR): a full backup copy kept
    beside the level's own (<name>.prebake_signs.xodr, 24-28 Nov 2026) was what it loaded, with every stock US
    sign."""
    moved = []
    for p in sorted(Path(folder).glob("*.xodr")):
        if p.name.lower() != f"{name}.xodr".lower():
            Path(aside).mkdir(parents=True, exist_ok=True)
            p.replace(Path(aside) / p.name)
            moved.append(p.name)
    return moved


def has_types(path, types):
    """True when the OpenDRIVE file has a signal of one of these types."""
    if not types:
        return False
    pat = re.compile(r'<signal\s[^>]*?\btype="(%s)"' % "|".join(re.escape(t) for t in types))
    with open(path, encoding="utf-8", errors="replace") as f:
        return any(pat.search(line) for line in f)


def stage(content, levels, dry_run=False, lookup=twinmodel_lookup, level_copies=True, aside=None):
    """Writes each twin's package copy (Carla/Maps/OpenDrive/<Name>.xodr) and, with ``level_copies``, leaves the
    stock sign signals out of the level's own copy too (what an editor -game server reads; light signals stay
    there), the full file kept in ``aside`` (default <project>/Saved/xodr_full, outside Content: never beside the
    level's copy, see tidy_level_folder)."""
    content = Path(content)
    twins = content / "Carla" / "Maps" / "Twins"
    out_dir = content / "Carla" / "Maps" / "OpenDrive"
    aside = Path(aside) if aside else content.parent / "Saved" / "xodr_full"
    manifest = {}
    for d in sorted(p for p in twins.iterdir() if p.is_dir()) if twins.is_dir() else []:
        src = d / "OpenDrive" / f"{d.name}.xodr"
        if not src.exists():
            continue
        drop, rec = plan(d.name, levels.get(d.name), server_signs.xodr_country(src), place_country(d, src, lookup))
        dst = out_dir / f"{d.name}.xodr"
        if not dry_run:
            rec["moved_aside"] = tidy_level_folder(src.parent, d.name, aside)
            out_dir.mkdir(parents=True, exist_ok=True)
            counts = xodr_drop_signals.drop_signals(str(src), str(dst), drop) if drop else {}
            if not drop:
                dst.write_bytes(src.read_bytes())
            rec["dropped"] = dict(counts)
            signs = [t for t in drop if t in server_signs.SIGN_TYPES]
            if level_copies and has_types(src, signs):
                aside.mkdir(parents=True, exist_ok=True)
                full = aside / f"{d.name}.xodr"
                full.write_bytes(src.read_bytes())
                tmp = aside / f"{d.name}.level.tmp"
                rec["level_dropped"] = dict(xodr_drop_signals.drop_signals(str(full), str(tmp), set(signs)))
                tmp.replace(src)
        manifest[d.name] = rec
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--content", required=True, help="the project's Content folder")
    ap.add_argument("--levels", default=None, help="JSON {level: {light_actors, signs, country}}")
    ap.add_argument("--manifest", default=None, help="write the level manifest here (default: print it)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-level-copies", action="store_true",
                    help="leave the levels' own OpenDRIVE copies (editor servers) as they are")
    ap.add_argument("--aside", default=None, help="where full copies go (default <project>/Saved/xodr_full)")
    a = ap.parse_args(argv)
    levels = json.loads(Path(a.levels).read_text()) if a.levels else {}
    man = {"levels": stage(a.content, levels, a.dry_run, level_copies=not a.no_level_copies, aside=a.aside),
           "note": "light_actors none: no CARLA traffic-light actors on that level (OpenDRIVE light signals left out "
                   "of the server copy); signs client: no CARLA sign actors, the client draws the real signs"}
    text = json.dumps(man, indent=1)
    if a.manifest and not a.dry_run:
        Path(a.manifest).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
