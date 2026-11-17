"""Far-distance detail for a city-sized twin (EhvCity): HLOD layers on its World Partition level, then the HLOD build.
Editor Python, run headless after the chunked bake (ue/bake_city.py) has finished (its log ends in CITYDONE):

    UnrealEditor-Cmd <CarlaUnreal.uproject> -run=pythonscript \\
        -script="/abs/ue/hlod_city.py --name EhvCity [--dry-run]"
    # then, in its own editor run (needs a GPU: the merged layer bakes materials; a GPU window on win5090):
    UnrealEditor-Cmd <CarlaUnreal.uproject> /Game/Carla/Maps/Twins/EhvCity/EhvCity \\
        -run=WorldPartitionBuilderCommandlet -Builder=WorldPartitionHLODsBuilder -SetupHLODs -BuildHLODs \\
        -AllowCommandletRendering -SCCProvider=None -unattended

What it does:

1. two HLOD layers next to the level, as CARLA's own large maps have them (Town12/13/15 <Map>_HLODLayer_*):
   ``<Name>_HLODLayer_Instanced`` (instancing: the window modules' and other instanced components keep their
   meshes, drawn as instances from far) and ``<Name>_HLODLayer_Merged`` (one simplified, material-baked mesh per HLOD
   cell for terrain, roads and the 3DBAG buildings);
2. every spatially loaded actor of the twin gets its layer (``layer_for``: the window hosts the instanced one, the
   twin's meshes the merged one; spawn points, the sky, PlayerStart and other always-loaded actors none);
3. the World Settings' default HLOD layer is the merged one, so actors added later (a re-bake of a chunk) get it.

Runtime streaming itself needs no change on CARLA 1.0: every vehicle and walker is a streaming source and the
ego's ring is ``tile_stream_distance`` (Docs/core_large_maps.md); beyond the loaded cells the HLODs stand in, so
a camera sees the skyline instead of empty ground. ``--dry-run`` reports what it would assign and changes nothing.
"""
import argparse
import json
import sys

LAYER_INSTANCED = "Instanced"
LAYER_MERGED = "Merged"
NO_LAYER_LABELS = ("PlayerStart", "SpawnPoint", "VehicleSpawnPoint", "BP_Carla_Sky", "Sky", "NavMesh",
                   "RecastNavMesh", "NavMeshBoundsVolume", "WorldDataLayers", "WorldPartitionMiniMap")
MERGE_KINDS = ("building", "building_3dbag", "roof", "drivable", "sidewalk", "curb", "riser", "ground",
               "groundplane", "terrain", "marking_white", "marking_yellow", "parking", "island", "cycleway",
               "verge", "rail")


def layer_for(label: str, folder: str = "", spatially_loaded: bool = True):
    """HLOD layer kind (LAYER_INSTANCED, LAYER_MERGED or None) of an actor of a baked twin, by its label and outliner
    folder (bake_level.py: "SM_<asset>" in "Twin/<kind>", "Windows_<x>_<y>" in "Twin/Windows")."""
    if not spatially_loaded:
        return None
    if any(label.startswith(p) for p in NO_LAYER_LABELS):
        return None
    folder = folder or ""
    if label.startswith("Windows_") or folder.endswith("/Windows"):
        return LAYER_INSTANCED
    if folder.startswith("Twin/"):
        kind = folder.split("/", 1)[1]
        if kind == "boundary":      # the invisible collision wall
            return None
        return LAYER_MERGED if kind in MERGE_KINDS or label.startswith("SM_") else LAYER_INSTANCED
    if label.startswith("SM_"):
        return LAYER_MERGED
    return None


def plan(actors):
    """[(label, folder, spatially_loaded)] -> {"Instanced": n, "Merged": n, "none": n}."""
    out = {LAYER_INSTANCED: 0, LAYER_MERGED: 0, "none": 0}
    for label, folder, sl in actors:
        out[layer_for(label, folder, sl) or "none"] += 1
    return out


def _ensure_layer(unreal, path, kind):
    eal = unreal.EditorAssetLibrary
    if eal.does_asset_exist(path):
        return eal.load_asset(path)
    folder, name = path.rsplit("/", 1)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    factory = None
    for fname in ("HLODLayerFactory", "WorldPartitionHLODLayerFactory"):
        if hasattr(unreal, fname):
            factory = getattr(unreal, fname)()
            break
    layer = tools.create_asset(name, folder, unreal.HLODLayer, factory)
    if layer is None:
        raise RuntimeError("cannot create HLOD layer %s" % path)
    types = getattr(unreal, "HLODLayerType", None) or getattr(unreal, "WorldPartitionHLODLayerType", None)
    want = {LAYER_INSTANCED: ("INSTANCING",), LAYER_MERGED: ("MESH_MERGE", "MESH_SIMPLIFY", "MESH_APPROXIMATE")}[kind]
    for t in want:
        if types is not None and hasattr(types, t):
            layer.set_editor_property("layer_type", getattr(types, t))
            break
    # spatially loaded HLODs on their own coarse grid: 1 km cells, loaded within 5 km of a streaming source
    for prop, val in (("is_spatially_loaded", True), ("cell_size", 100000), ("loading_range", 500000)):
        try:
            layer.set_editor_property(prop, val)
        except Exception:  # noqa: BLE001 - property names move between engine versions
            pass
    eal.save_asset(path, only_if_is_dirty=False)
    return layer


def main(argv):
    import unreal
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--map-root", default="/Game/Carla/Maps/Twins")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    level = "%s/%s/%s" % (args.map_root, args.name, args.name)
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not les.load_level(level):
        raise RuntimeError("cannot open " + level)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    wpl = getattr(unreal, "WorldPartitionBlueprintLibrary", None)
    report = {"level": level, "dry_run": args.dry_run, "plan": {LAYER_INSTANCED: 0, LAYER_MERGED: 0, "none": 0}}
    layers = None
    if not args.dry_run:
        base = "%s/%s/%s_HLODLayer_" % (args.map_root, args.name, args.name)
        layers = {k: _ensure_layer(unreal, base + k, k) for k in (LAYER_INSTANCED, LAYER_MERGED)}
    n_set = 0
    seen = set()

    def process(actors):
        nonlocal n_set
        for a in actors:
            key = a.get_path_name()
            if key in seen:
                continue
            seen.add(key)
            try:
                sl = bool(a.get_editor_property("is_spatially_loaded"))
            except Exception:  # noqa: BLE001
                sl = True
            kind = layer_for(a.get_actor_label(), str(a.get_folder_path()), sl)
            report["plan"][kind or "none"] += 1
            if kind is not None and layers is not None:
                a.set_editor_property("hlod_layer", layers[kind])
                n_set += 1
    # a city does not fit in memory: load the level's actors in batches through their descriptors, assign, save,
    # unload (without the descriptor API: the actors the editor has loaded)
    descs = wpl.get_actor_descs() if wpl is not None and hasattr(wpl, "get_actor_descs") else None
    if descs:
        guids = [d.get_editor_property("guid") for d in descs]
        report["actor_descs"] = len(guids)
        for i in range(0, len(guids), 2000):
            batch = guids[i:i + 2000]
            wpl.load_actors(batch)
            process(eas.get_all_level_actors())
            if layers is not None:
                unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
            wpl.unload_actors(batch)
    else:
        process(eas.get_all_level_actors())
    unreal.log("[hlod_city] plan %s" % json.dumps(report))
    if layers is not None:
        ws = world.get_world_settings()
        try:
            ws.set_editor_property("default_hlod_layer", layers[LAYER_MERGED])
        except Exception as exc:  # noqa: BLE001
            unreal.log_warning("[hlod_city] default HLOD layer: %s" % exc)
        report["layers_set"] = n_set
        report["saved"] = bool(unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    unreal.log("[hlod_city] done %s" % json.dumps(report))
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
