"""Window geometry and interiors of a twin (manifest ``window_modules`` / ``window_instances``, written by
tools/real_buildings.py --window-geometry; twinmodel/export/windows.py) in the baked level. Editor Python,
called by bake_level.py:

* ``material_keys``: the materials the instances use: the walls' windowless facades (``facadenw_*``), the
  solid frames, sills and glass (manifest ``textures``, made by bake_level's textured instances) and the
  interiors (``interior_<room>``, made here on CARLA's fake-interior materials, INTERIOR_PARENTS);
* ``place_windows``: imports each module part once (Nanite on, complex-as-simple collision: lidar and ray casts
  hit the glass, not the hollow building behind the hole in the wall), then per tile one host actor with one
  hierarchical instanced static mesh component per (part, material), instances in world space, and the
  glass's per-instance custom data (night rank, warm/cool, brightness) for a night-aware interior material.

Provenance: windows ``inferred``, interiors ``generated`` (manifest, kept in the report).
"""
import os
import time

import unreal

INTERIOR_PARENTS = {   # CARLA content: interior-mapping instances of M_FakeInterior, by room kind
    "home": "/Game/Carla/Static/Building/Buildings/BuildingComplete/00_GenericComponents/Apt32_to_35/"
            "MI_Apt_FakeInterior.MI_Apt_FakeInterior",
    "office": "/Game/Carla/Static/Building/Buildings/BuildingPieces/00_GenericComponents/"
              "MI_Skysc_FakeInterior_A.MI_Skysc_FakeInterior_A",
    "shop": "/Game/Carla/Static/Building/Buildings/BuildingPieces/00_IndividualComponents/Shop01/"
            "MI_Shop01_Btt_Door_FakeInterior.MI_Shop01_Btt_Door_FakeInterior",
}
CUSTOM_FLOATS = 3


def _log(msg):
    unreal.log("[bake_windows] " + msg)
    print("[bake_windows] " + msg, flush=True)


def material_keys(manifest):
    """(texture keys for bake_level's textured instances, interior room kinds) the instances use."""
    tex = manifest.get("textures", {})
    keys, rooms = set(), set()
    for e in manifest.get("window_instances", ()):
        m = e["material"]
        if m.startswith("interior_"):
            rooms.add(m[len("interior_"):])
        elif m in tex:
            keys.add(m)
    return keys, rooms


def make_interiors(name, rooms, mat_dir):
    """MI_<Name>_interior_<room> on the room's CARLA fake-interior instance; room -> [MIC]."""
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    unreal.EditorAssetLibrary.make_directory(mat_dir)
    out = {}
    for room in sorted(rooms):
        parent_path = INTERIOR_PARENTS.get(room)
        parent = unreal.EditorAssetLibrary.load_asset(parent_path) if parent_path else None
        if parent is None:
            _log("interior parent for %s missing (%s): dark glass instead" % (room, parent_path))
            continue
        mi_name = "MI_%s_interior_%s" % (name, room)
        mi_path = "%s/%s" % (mat_dir, mi_name)
        mic = (unreal.EditorAssetLibrary.load_asset(mi_path) if unreal.EditorAssetLibrary.does_asset_exist(mi_path)
               else tools.create_asset(mi_name, mat_dir, unreal.MaterialInstanceConstant,
                                       unreal.MaterialInstanceConstantFactoryNew()))
        unreal.MaterialEditingLibrary.set_material_instance_parent(mic, parent)
        unreal.MaterialEditingLibrary.update_material_instance(mic)
        unreal.EditorAssetLibrary.save_asset(mi_path, only_if_is_dirty=False)
        # the parameters the parent offers (for wiring the night custom data in a later round)
        try:
            names = [str(p) for p in unreal.MaterialEditingLibrary.get_scalar_parameter_names(parent)]
            _log("interior %s on %s; scalar parameters: %s" % (room, parent_path.split(".")[-1], names[:20]))
        except Exception:  # noqa: BLE001
            pass
        out["interior_" + room] = [mic]
    return out


def import_modules(manifest, dest, finish_mesh, first_material, interchange_options, nanite=True):
    """Import each module part glb into ``dest`` (bake_level's interchange_options: Nanite on);
    returns mesh key -> StaticMesh."""
    base = os.path.dirname(os.path.abspath(manifest["_path"]))
    mods = manifest.get("window_modules", {})
    tasks = []
    for key, m in sorted(mods.items()):
        t = unreal.AssetImportTask()
        t.set_editor_property("filename", os.path.join(base, m["file"]))
        t.set_editor_property("destination_path", dest)
        t.set_editor_property("destination_name", m["asset"])
        t.set_editor_property("automated", True)
        t.set_editor_property("replace_existing", True)
        t.set_editor_property("save", False)
        t.set_editor_property("options", interchange_options())
        tasks.append(t)
    t0 = time.time()
    if tasks:
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    out = {}
    for key, m in sorted(mods.items()):
        path = "%s/%s" % (dest, m["asset"])
        if not unreal.EditorAssetLibrary.does_asset_exist(path):
            found = [p for p in unreal.EditorAssetLibrary.list_assets(dest, recursive=False)
                     if p.split("/")[-1].split(".")[0].startswith(m["asset"])]
            if not found:
                _log("module %s not found after import" % path)
                continue
            path = found[0].split(".")[0]
        mesh = unreal.EditorAssetLibrary.load_asset(path)
        mic = first_material.get(key)
        if mic is not None:
            finish_mesh(mesh, mic, "window", nanite=nanite)
        unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
        out[key] = mesh
    _log("imported %d window module parts in %.1f s" % (len(out), time.time() - t0))
    return out


def _add_hism(actor, sds):
    """A new HierarchicalInstancedStaticMeshComponent on an actor instance (SubobjectDataSubsystem)."""
    handles = sds.k2_gather_subobject_data_for_instance(actor)
    params = unreal.AddNewSubobjectParams(parent_handle=handles[0],
                                          new_class=unreal.HierarchicalInstancedStaticMeshComponent,
                                          blueprint_context=None)
    handle, fail = sds.add_new_subobject(params)
    lib = unreal.SubobjectDataBlueprintFunctionLibrary
    if not lib.is_handle_valid(handle):
        raise RuntimeError("add_new_subobject failed: %s" % fail)
    data = unreal.SubobjectDataBlueprintFunctionLibrary.get_data(handle)
    return unreal.SubobjectDataBlueprintFunctionLibrary.get_object(data)


def place_windows(world, manifest, name, mats, spawn, mesh_dest, finish_mesh, interchange_options, nanite=True):
    """Instances of every window module part, one host actor per tile. ``mats``: material key -> [MIC]
    (bake_level's textured instances plus make_interiors); ``spawn``: bake_level.spawn."""
    inst = manifest.get("window_instances", ())
    report = {"instances": 0, "components": 0, "tiles": 0, "missing_material": 0, "missing_mesh": 0,
              "custom_data": 0, "provenance": {}}
    if not inst:
        return report
    first = {}
    for e in inst:
        mic = (mats.get(e["material"]) or [None])[0]
        if mic is not None:
            first.setdefault(e["mesh"], mic)
    meshes = import_modules(manifest, mesh_dest, finish_mesh, first, interchange_options, nanite)
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    hosts = {}
    t0 = time.time()
    for e in inst:
        mesh = meshes.get(e["mesh"])
        mic = (mats.get(e["material"]) or [None])[0]
        if mesh is None:
            report["missing_mesh"] += len(e["transforms"])
            continue
        if mic is None:
            report["missing_material"] += len(e["transforms"])
        tile = tuple(e["tile"])
        host = hosts.get(tile)
        if host is None:
            host = spawn(world, unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0),
                         "Windows_%d_%d" % tile, "Twin/Windows")
            if host is None:
                continue
            host.get_editor_property("static_mesh_component").set_editor_property(
                "mobility", unreal.ComponentMobility.STATIC)
            hosts[tile] = host
        comp = _add_hism(host, sds)
        comp.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
        comp.set_static_mesh(mesh)
        if mic is not None:
            comp.set_material(0, mic)
        tfs = []
        for x, y, z, yaw, sx, sz in e["transforms"]:
            tfs.append(unreal.Transform(unreal.Vector(x, y, z), unreal.Rotator(0.0, 0.0, yaw),
                                        unreal.Vector(sx, 1.0, sz)))
        comp.add_instances(tfs, False, True)
        if e.get("custom"):
            try:
                comp.set_num_custom_data_floats(CUSTOM_FLOATS)
            except Exception:  # noqa: BLE001 - older API: the property
                comp.set_editor_property("num_custom_data_floats", CUSTOM_FLOATS)
            for i, row in enumerate(e["custom"]):
                for j, v in enumerate(row[:CUSTOM_FLOATS]):
                    comp.set_custom_data_value(i, j, float(v), False)
            report["custom_data"] += len(e["custom"])
        report["instances"] += len(tfs)
        report["components"] += 1
        p = e.get("provenance", "inferred")
        report["provenance"][p] = report["provenance"].get(p, 0) + len(tfs)
    report["tiles"] = len(hosts)
    report["seconds"] = round(time.time() - t0, 1)
    _log("placed %d window instances in %d components on %d tiles (%.1f s)" % (
        report["instances"], report["components"], report["tiles"], report["seconds"]))
    return report
