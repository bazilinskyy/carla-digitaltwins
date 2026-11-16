"""Replace a twin's extruded OSM buildings with real 3D buildings from an open city or national model.

    python tools/real_buildings.py build/<name>                  # after `twinmodel bake-export`
    python tools/real_buildings.py build/<name> --source nyc3d   # or 3dbag; default: by the twin's bbox

The provider is chosen by where the twin is (twinmodel.ingest.buildings): 3DBAG LoD 2.2 in the Netherlands
(with BAG uses), the NYC 3D Building Model LoD 2 in New York City (with PLUTO uses), PLATEAU in the listed Japanese
wards, the Catastro in Spain, BD TOPO in France; elsewhere (or when the national service fails) Overture Maps,
then the twin's own OSM footprints, as prisms with roof shapes (--no-global keeps the bake export's OSM slabs
there instead). The facade region (brick, warm render and sandstone, US, Japanese) comes from the twin's country
(twinmodel.regions). Every building gets a
facade style from its construction year, uses, height and size, roofs get the twin's aerial photo, and the
manifest's building assets and footprints are swapped for the real ones (twinmodel.export.realbuildings).
The OSM building meshes are kept in meshes/osm_buildings/. Fetches are cached in <build>/<provider>/.
"""
import argparse
import json
import sys
from pathlib import Path

from twinmodel.export import facades, realbuildings
from twinmodel.ingest import buildings as providers


def main(argv=None, default_source="auto"):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build_dir")
    ap.add_argument("--source", default=default_source, choices=["auto", *providers.PROVIDERS],
                    help="building provider (default: the one covering the twin's bbox)")
    ap.add_argument("--lod", default="2.2", help="3DBAG level of detail")
    ap.add_argument("--plain", action="store_true",
                    help="one material for every building (no facade styles, no roof photo)")
    ap.add_argument("--ortho", default=None, help="ortho GeoTIFF in the model frame (default: the "
                    "twin pipeline's cached data/ortho_<bbox>_*.tif)")
    ap.add_argument("--no-uses", action="store_true", help="do not fetch the use registers (BAG units, PLUTO): "
                    "styles from year, height and OSM")
    ap.add_argument("--roof-photo-max-m", type=float, default=None,
                    help="buildings taller than this get a plain roof instead of the aerial photo "
                         "(default: the provider's; NYC 25 m, the Netherlands none)")
    ap.add_argument("--no-skirt", action="store_true",
                    help="do not extend the walls down to the twin's terrain (providers with one ground height "
                         "per building, e.g. NYC)")
    ap.add_argument("--texture-px", type=int, default=facades.TEX_PX)
    ap.add_argument("--no-window-lights", action="store_true",
                    help="no night masks (bake_level.py then gives the facades no lit windows)")
    ap.add_argument("--roof-px", type=int, default=1024, help="aerial photo texture per tile")
    ap.add_argument("--no-source-textures", action="store_true",
                    help="ignore a city model's own photo textures (PLATEAU LoD 2): facade styles instead")
    ap.add_argument("--no-global", action="store_true",
                    help="no Overture/OSM prisms where no national model covers the twin (the OSM slabs stay)")
    ap.add_argument("--no-street-osm", action="store_true",
                    help="do not ask OSM for the shops and cafes at the facades and mapped wall materials/colours "
                         "(shopfront ground floors then come from the use registers alone, on every street wall)")
    ap.add_argument("--region", default=None, help="facade region (nl, eu, south, us, jp; default: by country)")
    ap.add_argument("--procedural-min-height", type=float, default=facades.PROCEDURAL_MIN_HEIGHT_M,
                    help="buildings at least this high to the eaves (m) may take bake_level.py's "
                         "--buildings procedural facades: their walls below the eaves are written apart")
    args = ap.parse_args(argv)
    build = Path(args.build_dir)
    man = json.loads((build / "ue" / "manifest.json").read_text())
    from twinmodel import regions
    iso2 = twin_country(build)
    plan = regions.plan_for(iso2, man["bbox_wgs84"])
    region = args.region or plan.facade_region
    if args.source != "auto":
        names = [args.source]
    else:
        names = providers.candidates(man["bbox_wgs84"], order=list(plan.buildings) + list(providers.PROVIDERS))
        if args.no_global:
            names = [n for n in names if n not in providers.GLOBAL]
    if not names:
        print("real buildings: no provider covers %s (known: %s); the OSM buildings stay" % (
            man["bbox_wgs84"], sorted(providers.PROVIDERS)))
        return 0
    looks = not args.plain
    blds = source = None
    for name in names:
        try:
            blds, source = providers.load(name, build, man, looks=looks, lod=args.lod, uses=not args.no_uses,
                                          region=region, textures=not args.no_source_textures)
        except Exception as exc:  # noqa: BLE001 - offline, a service down, duckdb missing: the next provider
            print(f"real buildings: {name} failed ({type(exc).__name__}: {exc}); trying the next provider",
                  flush=True)
            continue
        if blds:
            break
        print(f"real buildings: {name} gave no buildings; trying the next provider", flush=True)
    if not blds:
        print("real buildings: no provider gave buildings (%s); the OSM buildings stay" % ", ".join(names))
        return 0
    print(f"real buildings: {source.name} for {iso2 or 'an unknown country'} (facade region {region})")
    if args.roof_photo_max_m is not None:
        source.roof_photo_max_m = args.roof_photo_max_m
    if source.skirt and not args.no_skirt:
        dp = realbuildings.find_dem(build, man["bbox_wgs84"])
        if dp:
            print(f"skirt: {realbuildings.skirt(blds, dp)} buildings' walls extended down to the terrain ({dp.name})")
        else:
            print("skirt: no cached elevation grid found; walls stay at the model's ground heights")
    ortho = None
    if looks:
        op = Path(args.ortho) if args.ortho else realbuildings.find_ortho(build, man["bbox_wgs84"])
        if op and op.exists():
            ortho = realbuildings.load_ortho(op, realbuildings.model_proj(man))
            if ortho:
                print(f"ortho: {op.name} {ortho[0].shape[1]}x{ortho[0].shape[0]} px")
        else:
            print("ortho: none found; roofs get a plain roof material")
    street = None
    if looks and not args.no_street_osm:
        from twinmodel.ingest import street_osm
        s, w, n, e = man["bbox_wgs84"]
        try:
            raw = street_osm.fetch((s, w, n, e), build / "osm_street")
        except Exception as exc:  # noqa: BLE001 - the facades stand without it
            print(f"street osm: failed ({type(exc).__name__}: {exc})")
            raw = None
        street = street_osm.to_model(raw, realbuildings.model_proj(man))
        if street is None:
            print("street osm: no Overpass server answered; shopfronts from the use registers alone")
        else:
            print(f"street osm: {len(street['pois'])} ground-floor shops and cafes, "
                  f"{len(street['buildings'])} buildings with a mapped wall material or colour")
    realbuildings.apply(build, man, blds, source, looks=looks, ortho=ortho,
                        ortho_label=realbuildings.ortho_credit(build), texture_px=args.texture_px,
                        roof_px=args.roof_px, procedural_min_height=args.procedural_min_height,
                        window_lights=not args.no_window_lights, street=street)
    return 0


def twin_country(build: Path):
    """ISO alpha-2 of the twin (its report.json profile record), or None."""
    try:
        rep = json.loads((build / "report.json").read_text())
        return (rep.get("profile") or (rep.get("build") or {}).get("profile") or {}).get("iso2")
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    sys.exit(main())
