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
INTERIOR_SOURCE_MASTER = "/Game/Carla/Static/GenericMaterials/000_Masters/Buildings/M_FakeInterior"


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


INTERIOR_NIGHT_MASTER = "/Game/CoupledCarla/Materials/M_TwinInteriorNight"
INTERIOR_NIGHT_DIR = "/Game/CoupledCarla/Materials"


def _build_interior_night_master(source_master, night_mpc, colours, nits_default):
    """M_TwinInteriorNight: a copy of CARLA's M_FakeInterior whose emissive (its room lit at night through
    NightFactor) is gated per pane by the instance's custom data, as the lit-window facades are (twin_materials,
    "lit windows at night"):

        lit       = saturate((Window Lit Share x NightFactor - rank) x 512)      rank: custom data 0
        emissive  = old emissive x lit x brightness x lerp(Window Warm, Window Cool, cool)
                                                                                  cool, brightness: custom data 1, 2
    (without an emissive of its own: lit x brightness x tint x Window Light). So at night only the panes whose rank
    is below the use's lit share at that hour glow, warm or cool, at their own brightness. Generated detail."""
    mel = unreal.MaterialEditingLibrary
    mat = unreal.EditorAssetLibrary.duplicate_asset(source_master, INTERIOR_NIGHT_MASTER)
    if mat is None:
        raise RuntimeError("cannot copy %s" % source_master)
    emissive = unreal.MaterialProperty.MP_EMISSIVE_COLOR
    old = mel.get_material_property_input_node(mat, emissive)
    old_pin = mel.get_material_property_input_node_output_name(mat, emissive) if old is not None else ""

    def node(cls, dx, dy):
        return mel.create_material_expression(mat, cls, -2400 + dx, 2400 + dy)

    def link(a, a_out, b, b_in):
        for n in {"": ("", "Input")}.get(b_in, (b_in,)):
            if mel.connect_material_expressions(a, a_out, b, n):
                return
        raise RuntimeError("cannot connect %s.%s -> %s.%s" % (a.get_name(), a_out, b.get_name(), b_in))

    data = []
    for i in range(CUSTOM_FLOATS):
        d = node(unreal.MaterialExpressionPerInstanceCustomData, 0, 200 * i)
        d.set_editor_property("data_index", i)
        data.append(d)
    rank, cool, bright = data
    nf = node(unreal.MaterialExpressionCollectionParameter, 250, -300)
    nf.set_editor_property("collection", unreal.EditorAssetLibrary.load_asset(night_mpc.split(".")[0]))
    nf.set_editor_property("parameter_name", "NightFactor")
    share = node(unreal.MaterialExpressionScalarParameter, 250, -450)
    share.set_editor_property("parameter_name", "Window Lit Share")
    share.set_editor_property("default_value", 0.0)
    thr = node(unreal.MaterialExpressionMultiply, 550, -400)
    link(share, "", thr, "A")
    link(nf, "", thr, "B")
    diff = node(unreal.MaterialExpressionSubtract, 750, -300)
    link(thr, "", diff, "A")
    link(rank, "", diff, "B")
    gain = node(unreal.MaterialExpressionMultiply, 950, -300)
    link(diff, "", gain, "A")
    gain.set_editor_property("const_b", 512.0)
    lit = node(unreal.MaterialExpressionSaturate, 1150, -300)
    link(gain, "", lit, "")
    cols = {}
    for k, (cname, rgb) in enumerate(sorted(colours.items())):
        v = node(unreal.MaterialExpressionVectorParameter, 600, 450 + 200 * k)
        v.set_editor_property("parameter_name", cname)
        v.set_editor_property("default_value", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
        cols[cname] = v
    tint = node(unreal.MaterialExpressionLinearInterpolate, 900, 500)
    link(cols["Window Warm"], "", tint, "A")
    link(cols["Window Cool"], "", tint, "B")
    link(cool, "", tint, "Alpha")
    lb = node(unreal.MaterialExpressionMultiply, 1350, -100)
    link(lit, "", lb, "A")
    link(bright, "", lb, "B")
    lt = node(unreal.MaterialExpressionMultiply, 1550, 100)
    link(lb, "", lt, "A")
    link(tint, "", lt, "B")
    total = node(unreal.MaterialExpressionMultiply, 1950, 0)
    if old is not None:
        link(old, old_pin, total, "A")
    else:
        nits = node(unreal.MaterialExpressionScalarParameter, 1750, 300)
        nits.set_editor_property("parameter_name", "Window Light")
        nits.set_editor_property("default_value", nits_default)
        link(nits, "", total, "A")
    link(lt, "", total, "B")
    if not mel.connect_material_property(total, "", emissive):
        raise RuntimeError("cannot connect the emissive input")
    try:
        mat.set_editor_property("used_with_instanced_static_meshes", True)
        mat.set_editor_property("used_with_nanite", True)
    except Exception as exc:  # noqa: BLE001
        _log("usage flags: %s" % exc)
    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset(INTERIOR_NIGHT_MASTER, only_if_is_dirty=False)
    _log("interior night master %s: %d expressions (old emissive %s)" % (
        INTERIOR_NIGHT_MASTER, mel.get_num_material_expressions(mat), "kept" if old is not None else "none"))
    return mat


def _night_room_parent(room, parent_path, master):
    """The room's CARLA fake-interior instance copied onto M_TwinInteriorNight (its textures and switches kept)."""
    mel = unreal.MaterialEditingLibrary
    path = "%s/MI_TwinInteriorNight_%s" % (INTERIOR_NIGHT_DIR, room)
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.EditorAssetLibrary.load_asset(path)
    mi = unreal.EditorAssetLibrary.duplicate_asset(parent_path.split(".")[0], path)
    if mi is None:
        raise RuntimeError("cannot copy %s" % parent_path)
    before = {str(n): mel.get_material_instance_static_switch_parameter_value(mi, n)
              for n in mel.get_static_switch_parameter_names(mi)}
    mel.set_material_instance_parent(mi, master)
    for n, v in before.items():
        if mel.get_material_instance_static_switch_parameter_value(mi, n) != v:
            mel.set_material_instance_static_switch_parameter_value(mi, n, v)
    mel.update_material_instance(mi)
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    return mi


def make_interiors(name, rooms, mat_dir, window_hour=None):
    """MI_<Name>_interior_<room>: on the room's CARLA fake-interior instance, or (``window_hour`` given) on that
    instance moved onto M_TwinInteriorNight with the use's lit share at that hour (the panes' night data decides
    which glow); room -> [MIC]."""
    import twin_materials
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    unreal.EditorAssetLibrary.make_directory(mat_dir)
    master = None
    if window_hour is not None:
        try:
            unreal.EditorAssetLibrary.make_directory(INTERIOR_NIGHT_DIR)
            master = (unreal.EditorAssetLibrary.load_asset(INTERIOR_NIGHT_MASTER)
                      if unreal.EditorAssetLibrary.does_asset_exist(INTERIOR_NIGHT_MASTER) else
                      _build_interior_night_master(INTERIOR_SOURCE_MASTER, twin_materials.NIGHT_MPC,
                                                   twin_materials.WINDOW_COLOURS, twin_materials.WINDOW_NITS["home"]))
        except Exception as exc:  # noqa: BLE001 - CARLA's own interiors (every pane lit at night) instead
            _log("interior night master unavailable (%s); CARLA's fake interiors as they are" % exc)
            master = None
    out = {}
    for room in sorted(rooms):
        parent_path = INTERIOR_PARENTS.get(room)
        parent = unreal.EditorAssetLibrary.load_asset(parent_path) if parent_path else None
        if parent is None:
            _log("interior parent for %s missing (%s): dark glass instead" % (room, parent_path))
            continue
        if master is not None:
            try:
                parent = _night_room_parent(room, parent_path, master)
            except Exception as exc:  # noqa: BLE001
                _log("night interior %s: %s; CARLA's instance as it is" % (room, exc))
        mi_name = "MI_%s_interior_%s" % (name, room)
        mi_path = "%s/%s" % (mat_dir, mi_name)
        mic = (unreal.EditorAssetLibrary.load_asset(mi_path) if unreal.EditorAssetLibrary.does_asset_exist(mi_path)
               else tools.create_asset(mi_name, mat_dir, unreal.MaterialInstanceConstant,
                                       unreal.MaterialInstanceConstantFactoryNew()))
        unreal.MaterialEditingLibrary.set_material_instance_parent(mic, parent)
        if master is not None:
            sc = []
            for pname, val in sorted(twin_materials.window_scalars(room, window_hour).items()):
                e = unreal.ScalarParameterValue()
                e.set_editor_property("parameter_info", unreal.MaterialParameterInfo(name=pname))
                e.set_editor_property("parameter_value", float(val))
                sc.append(e)
            mic.set_editor_property("scalar_parameter_values", sc)
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
