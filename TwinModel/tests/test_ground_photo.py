"""Tests for the aerial-photo ground (twinmodel.export.groundphoto, tools/ground_photo.py); no network."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from twinmodel.export import groundphoto as gp
from twinmodel.export.ue import gltf_to_model, read_glb, write_glb

TOOL = Path(__file__).resolve().parents[1] / "tools" / "ground_photo.py"


def _tool():
    spec = importlib.util.spec_from_file_location("ground_photo_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tile_uv_matches_the_roofs():
    uv = gp.tile_uv(np.array([[250.0, 500.0], [500.0, 250.0], [375.0, 375.0]]), (1, 1), 250.0)
    # north-west corner -> (0, 0), south-east -> (1, 1): row 0 of the texture is north
    assert np.allclose(uv, [[0, 0], [1, 1], [0.5, 0.5]])


def test_object_mask_threshold_buildings_and_thin_lines():
    px_m = 0.125
    nd = np.zeros((200, 200), np.float32)
    nd[50:90, 50:90] = 8.0          # a 5 m crown
    nd[150:200, 10:13] = 6.0        # a wire / wall edge, 0.4 m wide
    nd[10:20, 150:190] = 1.0        # a hedge, low
    nd[120:140, 120:140] = 9.0      # inside a building
    nd[0:5, 0:5] = np.nan
    bld = np.zeros_like(nd, bool)
    bld[115:145, 115:145] = True
    m = gp.object_mask(nd, bld, px_m)
    assert m[70, 70] and m[48, 70]                  # the crown, grown at its edge
    assert not m[170, 11]                           # thin line dropped
    assert not m[15, 170]                           # below the threshold
    assert not m[bld].any()                         # never inside a footprint
    assert not m[2, 2]


def test_pull_push_fill_takes_the_colour_around_the_hole():
    img = np.zeros((64, 64, 3), np.float32)
    img[:, :32] = (40, 120, 40)     # grass
    img[:, 32:] = (40, 120, 40)
    img[20:44, 20:44] = (10, 60, 10)   # a tree crown
    valid = np.ones((64, 64), bool)
    valid[20:44, 20:44] = False
    out = gp.pull_push_fill(img, valid)
    assert np.allclose(out[32, 32], (40, 120, 40), atol=1.0)
    assert np.array_equal(out[valid], img[valid])


def test_lift_shadows_brightens_shadow_only():
    rng = np.random.default_rng(1)
    img = np.full((120, 120, 3), (110, 120, 100), np.float32) + rng.normal(0, 3, (120, 120, 3))
    img[30:70, 30:70] = (35, 42, 48) + rng.normal(0, 2, (40, 40, 3))   # bluish cast shadow
    valid = np.ones((120, 120), bool)
    out, sh, gain = gp.lift_shadows(img, valid, 0.125)
    assert sh[50, 50] and not sh[100, 100]
    assert out[50, 50].mean() > 1.6 * img[50, 50].mean()
    assert gain[0] > gain[2]                         # the blue cast is evened out
    assert np.allclose(out[100, 100], img[100, 100], atol=0.5)


def test_process_tile_fills_objects_from_visible_ground_and_keeps_water():
    px = 128
    photo = np.zeros((px, px, 3), np.uint8)
    photo[:] = (60, 130, 60)                 # lawn
    photo[:, 96:] = (90, 90, 90)             # road (covered by the road mesh)
    photo[40:70, 40:70] = (20, 70, 20)       # tree crown
    photo[100:, :30] = (30, 35, 40)          # water
    nd = np.zeros((px, px), np.float32)
    nd[40:70, 40:70] = 10.0
    nd[100:, :30] = np.nan
    cov = np.zeros((px, px), bool)
    cov[:, 96:] = True
    out, st = gp.process_tile(photo, 0.25, nd, None, cov, seed=3)
    assert abs(int(out[55, 55, 1]) - 130) < 15       # crown -> lawn green, not road grey
    assert np.allclose(out[115, 10], (30, 35, 40), atol=2)   # water not brightened
    assert 0.03 < st.masked_frac < 0.15
    # without a surface model nothing is filled
    out2, st2 = gp.process_tile(photo, 0.25, None, None, cov, shadows=False)
    assert np.array_equal(out2[55, 55], photo[55, 55]) and st2.notes


def test_resample_keeps_nan():
    a = np.ones((4, 4), np.float32)
    a[0, 0] = np.nan
    r = gp.resample(a, (16, 16))
    assert r.shape == (16, 16) and np.isnan(r[0, 0]) and np.isfinite(r[15, 15])


def test_apply_rewrites_uvs_and_manifest(tmp_path):
    tool = _tool()
    out = tmp_path / "ue"
    (out / "meshes").mkdir(parents=True)
    pos = np.array([[260.0, 260.0, 15.0], [490.0, 260.0, 15.0], [260.0, 490.0, 15.0]])
    nrm = np.tile([0.0, 0.0, 1.0], (3, 1))
    write_glb(out / "meshes/g.glb", "g", pos, nrm, np.column_stack([pos[:, 0], -pos[:, 1]]),
              np.array([[0, 1, 2]]), "ground")
    man = {"name": "T", "tile_m": 250.0, "stats": {},
           "textures": {"roof_1_1": {"base_color": "textures/r.png"}},
           "assets": [{"file": "meshes/g.glb", "asset": "g", "kind": "groundplane", "material": "ground",
                       "tile": [1, 1]},
                      {"file": "meshes/r.glb", "asset": "r", "kind": "drivable", "material": "road", "tile": [1, 1]}]}
    entry = {"base_color": "textures/T_T_ground_1_1_d.jpg"}
    n = tool.apply(man, out, {(1, 1): entry}, 250.0)
    assert n == 1 and man["assets"][0]["texture"] == "ground_1_1" and "texture" not in man["assets"][1]
    assert man["textures"]["ground_1_1"] == entry and "roof_1_1" in man["textures"]
    g = read_glb(out / "meshes/g.glb")
    assert np.allclose(g["uvs"], [[0.04, 0.96], [0.96, 0.96], [0.04, 0.04]], atol=1e-5)
    assert np.allclose(gltf_to_model(g["positions"]), pos, atol=1e-4)
    # undo: metric UVs and no ground textures
    tool.apply(man, out, {}, 250.0)
    assert "texture" not in man["assets"][0] and list(man["textures"]) == ["roof_1_1"]
    assert np.allclose(read_glb(out / "meshes/g.glb")["uvs"], [[260, -260], [490, -260], [260, -490]])
    json.dumps(man)


def test_in_nl():
    tool = _tool()
    assert tool.in_nl((51.436, 5.47, 51.45, 5.493))
    assert not tool.in_nl((41.38, 2.15, 41.40, 2.18))


def test_ground_tiles_takes_verges_only_when_asked():
    tool = _tool()
    man = {"assets": [{"kind": "ground", "tile": [0, 0]}, {"kind": "verge", "tile": [0, 0]},
                      {"kind": "sidewalk", "tile": [0, 0]}, {"kind": "groundplane", "tile": [1, 0]}]}
    plain = tool.ground_tiles(man)
    assert sorted(a["kind"] for a in plain[(0, 0)]) == ["ground"]
    lawns = tool.ground_tiles(man, tool.KINDS + ("verge",))
    assert sorted(a["kind"] for a in lawns[(0, 0)]) == ["ground", "verge"]


def test_an_export_without_its_build_cache_fetches_the_profiles_photo(tmp_path, monkeypatch):
    # NycMidtown, 2026-11-24: its export copied to another machine had no data/ortho_*.tif, so --verges found no photo
    # for the lawns. The profile's providers (NYS orthos, then NAIP) fetch it again into <build>/data; the rendered
    # map tiles (osm_tiles) are never taken for a photo.
    from twinmodel.export import realbuildings
    from twinmodel.ingest import imagery
    man = {"profile": "us_urban", "origin": {"lat": 40.7549, "lon": -73.984},
           "bbox_wgs84": [40.74861, -73.9923, 40.76119, -73.9757]}
    asked = {}

    def fake_fetch(frame, bbox, cache_dir, sources, **kw):
        asked["sources"] = list(sources)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "ortho_40.74861_-73.99230_40.76119_-73.97570_0.25m_abcd1234.tif").write_bytes(b"tif")
        return type("Img", (), {"source": "nys_orthos"})()
    monkeypatch.setattr(imagery, "fetch_ortho", fake_fetch)
    monkeypatch.setattr(realbuildings, "twin_iso2", lambda build, man: None)    # offline: the profile's providers
    path, src = realbuildings.fetch_twin_ortho(tmp_path, man)
    assert src == "nys_orthos" and path is not None and path.parent == tmp_path / "data"
    assert "osm_tiles" not in asked["sources"] and asked["sources"][0] == "nys_orthos"
    assert realbuildings.ORTHO_CREDITS[src] == "NYS ITS orthoimagery (public domain)"
    assert realbuildings.find_ortho(tmp_path, man["bbox_wgs84"]) == path      # found as the build's own next time


def test_a_finnish_twin_takes_helsinkis_photo_then_mmls(tmp_path, monkeypatch):
    # HelsinkiSenate, 2026-11-25: the twin's ground had no photo (no Finnish provider; its street profile eu_dense
    # lists Spain's). The country's providers that cover the twin are asked: Helsinki's 5 cm city photo, then the
    # National Land Survey's 0.5 m one (keyless mirror); elsewhere in Finland only the national one.
    from twinmodel import regions
    from twinmodel.export import realbuildings
    from twinmodel.ingest import imagery
    hel = [60.16451, 24.93897, 60.17349, 24.95703]
    assert regions.plan_for("FI", hel).ortho == ("helsinki_ortho", "mml_ortho", "osm_tiles")
    assert regions.plan_for("FI", [61.49, 23.74, 61.51, 23.79]).ortho == ("mml_ortho", "osm_tiles")    # Tampere
    for p in ("helsinki_ortho", "mml_ortho"):
        assert p in imagery.PROVIDERS and p in regions.SOURCES and "CC BY 4.0" in realbuildings.ORTHO_CREDITS[p]
    man = {"profile": "eu_dense", "origin": {"lat": 60.169, "lon": 24.948}, "bbox_wgs84": hel}
    asked = {}

    def fake_fetch(frame, bbox, cache_dir, sources, **kw):
        asked["sources"] = list(sources)
        return None
    monkeypatch.setattr(imagery, "fetch_ortho", fake_fetch)
    monkeypatch.setattr(realbuildings, "twin_iso2", lambda build, man: "FI")
    assert realbuildings.fetch_twin_ortho(tmp_path, man) == (None, None)
    assert asked["sources"] == ["helsinki_ortho", "mml_ortho"]


def test_twin_iso2_reads_the_report_before_asking_the_map(tmp_path, monkeypatch):
    import json
    from twinmodel.export import realbuildings
    from twinmodel.ingest import osm
    monkeypatch.setattr(osm, "country_for_bbox", lambda *a, **k: "FI")
    man = {"bbox_wgs84": [60.16, 24.93, 60.17, 24.95]}
    assert realbuildings.twin_iso2(tmp_path, man) == "FI"                   # no report: the map's answer
    (tmp_path / "report.json").write_text(json.dumps({"profile": {"iso2": "EE"}}))
    assert realbuildings.twin_iso2(tmp_path, man) == "EE"
    assert realbuildings.twin_iso2(tmp_path, dict(man, profile={"name": "eu_dense", "iso2": "SE"})) == "SE"


def test_a_fetched_photo_is_credited_in_the_levels_attribution(tmp_path):
    # the build wrote ATTRIBUTION.md before the photo was fetched: its provider (CC BY 4.0) must be listed
    import importlib.util
    spec = importlib.util.spec_from_file_location("gp", Path(__file__).parent.parent / "tools" / "ground_photo.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    att = tmp_path / "ATTRIBUTION.md"
    att.write_text("# X: data sources\n\nCredit:\n\n- OpenStreetMap contributors. Licence: ODbL 1.0.\n\nBuildings: x\n")
    added = gp.credit_photo(tmp_path, ["helsinki_ortho", "osm_tiles_unknown"])
    assert len(added) == 1 and "City of Helsinki" in added[0] and "CC BY 4.0" in added[0]
    lines = att.read_text().splitlines()
    assert lines.index(added[0]) == lines.index("- OpenStreetMap contributors. Licence: ODbL 1.0.") + 1
    assert gp.credit_photo(tmp_path, ["helsinki_ortho"]) == []          # once only

