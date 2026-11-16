"""Tests for twinmodel.export.facades (facade styles, wall/roof UVs, clipping, textures)."""
from __future__ import annotations

import numpy as np
import pytest

from twinmodel.export import facades as fc


def _info(**kw):
    base = dict(id="NL.IMBAG.Pand.1", year=1960, uses={"woonfunctie": 100}, height_m=9.0, area_m2=120.0)
    base.update(kw)
    return fc.BuildingInfo(**base)


@pytest.mark.parametrize("kw,style,shop", [
    (dict(year=1920), "brick_old", False),
    (dict(year=1955), "brick_wederopbouw", False),
    (dict(year=1972), "brick_late", False),
    (dict(year=1985), "brick_modern", False),
    (dict(year=None), "brick_postwar", False),
    (dict(year=1960, uses={"woonfunctie": 300, "winkelfunctie": 120}, height_m=10.0), "brick_wederopbouw", True),
    (dict(year=1960, uses={"woonfunctie": 300, "winkelfunctie": 120}, height_m=10.0, region="eu"),
     "brick_postwar", True),                                                         # outside nl: as before
    (dict(year=1992, uses={"kantoorfunctie": 2000}, height_m=15.0), "office_brick", False),
    (dict(year=1992, uses={"onderwijsfunctie": 2000}, height_m=15.0), "office_glass", False),
    (dict(year=1970, uses={"kantoorfunctie": 2000}, height_m=20.0), "office_concrete", False),
    (dict(year=2005, uses={"kantoorfunctie": 2000}, height_m=40.0), "office_glass", False),
    (dict(year=1990, uses={"industriefunctie": 900}), "industrial", False),
    (dict(year=1975, uses={"winkelfunctie": 300}, height_m=4.5), "shopfront", False),
    (dict(year=1980, uses={}, height_m=2.5, area_m2=15.0), "blank_brick", False),
    (dict(year=1980, uses={}, osm_category="warehouse", area_m2=800.0), "industrial", False),
])
def test_classify(kw, style, shop):
    assert fc.classify(_info(**kw)) == (style, shop)


def test_classify_render_is_deterministic_share_of_modern_homes():
    picks = [fc.classify(_info(id="b%d" % i, year=2005, region="eu"))[0] for i in range(300)]
    share = picks.count("render_modern") / len(picks)
    assert set(picks) <= {"render_modern", "brick_modern"}
    assert 0.2 < share < 0.45
    assert picks == [fc.classify(_info(id="b%d" % i, year=2005, region="eu"))[0] for i in range(300)]
    # the Netherlands from 1995: a quarter rendered, the rest today's light, anthracite or red brick
    nl = [fc.classify(_info(id="b%d" % i, year=2005))[0] for i in range(300)]
    assert set(nl) == {"render_modern", "brick_contemporary"}
    assert 0.15 < nl.count("render_modern") / 300 < 0.35


def test_nl_flats_of_the_1960s_and_70s_are_partly_concrete_panels():
    picks = [fc.classify(_info(id="f%d" % i, year=1968, height_m=14.0))[0] for i in range(300)]
    assert set(picks) == {"concrete_panel", "brick_late"}
    assert 0.55 < picks.count("concrete_panel") / 300 < 0.8
    assert {fc.classify(_info(id="f%d" % i, year=1968, height_m=7.0))[0] for i in range(50)} == {"brick_late"}


@pytest.mark.parametrize("kw,shop", [
    (dict(uses={"woonfunctie": 300, "winkelfunctie": 120}, height_m=10.0, shop_pois=0), True),   # BAG shop unit
    (dict(uses={"woonfunctie": 300}, height_m=10.0, shop_pois=2), True),          # OSM shop, units say homes
    (dict(uses={"woonfunctie": 300, "bijeenkomstfunctie": 80}, height_m=10.0), True),   # unknown POIs: as before
    (dict(uses={"woonfunctie": 300, "bijeenkomstfunctie": 80}, height_m=10.0, shop_pois=0), False),
    (dict(uses={"woonfunctie": 300, "bijeenkomstfunctie": 80}, height_m=10.0, shop_pois=1), True),   # a cafe
])
def test_shop_floor_from_osm_shops(kw, shop):
    assert fc.classify(_info(**kw))[1] is shop


def test_assembly_halls_without_shops_are_not_shopfronts():
    # a one-storey lecture hall (bijeenkomstfunctie) on the TU/e campus: with OSM shops known, no shopfront
    hall = dict(year=1963, uses={"bijeenkomstfunctie": 6000, "industriefunctie": 50}, height_m=4.2, area_m2=5000.0)
    assert fc.classify(_info(**hall))[0] == "shopfront"                 # register alone: as before
    assert fc.classify(_info(shop_pois=0, **hall)) == ("office_concrete", False)
    assert fc.classify(_info(shop_pois=1, **hall))[0] == "shopfront"   # a mapped shop or cafe: a real shop


def test_campus_buildings_without_units_are_not_sheds():
    big = dict(year=1991, uses={}, height_m=4.0, area_m2=13000.0)
    assert fc.classify(_info(**big))[0] == "industrial"
    assert fc.classify(_info(osm_category="hospital", **big))[0] == "office_glass"
    assert fc.classify(_info(osm_category="university", **big))[0] == "office_glass"


@pytest.mark.parametrize("material,style", [
    ("glass", "office_glass"), ("concrete", "concrete_panel"), ("plaster", "render_modern"),
    ("brick", "brick_late"), ("stone", "stone_old"), ("wood", "brick_late"), ("metal", "industrial"),
])
def test_osm_material_overrides_the_wall(material, style):
    assert fc.classify(_info(year=1972, height_m=7.0, material=material))[0] == style


def test_osm_material_for_offices_and_colour_variants():
    office = dict(year=1975, uses={"kantoorfunctie": 2000}, height_m=15.0)
    assert fc.classify(_info(material="brick", **office))[0] == "office_brick"
    assert fc.classify(_info(material="concrete", **office))[0] == "office_concrete"
    # building:colour picks the nearest colour variant
    cols = fc.STYLES["brick_contemporary"].colours
    assert fc.variant_of("brick_contemporary", "x", fc.parse_colour("#525050")) == cols.index((82, 80, 80))
    assert fc.variant_of("brick_contemporary", "x", fc.parse_colour("tan")) == cols.index((186, 164, 134))
    assert fc.variant_of("brick_contemporary", "x") == fc.variant_of("brick_contemporary", "x", None)
    assert fc.parse_colour("#fff") == (255, 255, 255) and fc.parse_colour("Light Grey") == (200, 200, 200)
    assert fc.parse_colour("nonsense") is None and fc.parse_colour("#12") is None and fc.parse_colour("") is None


def test_floor_bands_are_drawn():
    st = fc.STYLES["brick_wederopbouw"]
    alb, _, _ = fc.synth_facade(st, 0, px=256)
    band_rows = alb[256 - 3:256, :, :].reshape(-1, 3).mean(axis=0)    # the band just above the floor
    assert np.allclose(band_rows, st.band, atol=12)
    assert all(fc.BLANK_OF.get(n, "blank_brick") in fc.STYLES for n in fc.STYLES)
    assert fc.PROCEDURAL_STYLES <= set(fc.STYLES)


def test_storeys_fit_the_eaves():
    n, h = fc.storeys_for(9.3, 2.9)
    assert n == 3 and abs(n * h - 9.3) < 1e-9
    assert fc.storeys_for(0.0, 3.0) == (1, 3.0)
    # a 1.2 m wall would make a 1.2 m storey: keep the nominal one
    assert fc.storeys_for(1.2, 3.0) == (1, 3.0)


def _wall(x0, x1, z0, z1, y=0.0):
    """A vertical wall facing -y (outside towards -y), CCW from outside."""
    v = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], float)
    return v, np.array([0.0, -1.0, 0.0])


def test_wall_uv_whole_bays_and_storeys():
    v, nrm = _wall(10.0, 17.3, 5.0, 14.0)
    uv = fc.wall_uv(v, nrm, base_z=5.0, storey_h=3.0, bay_m=2.4)
    bays = round(7.3 / 2.4)
    # u spans a whole number of bays (in texture tiles of TEX_BAYS)
    assert np.isclose(np.ptp(uv[:, 0]) * fc.TEX_BAYS, bays)
    assert np.isclose(uv[:, 0].min(), 0.0)
    # v: 0 at the floor, 3 storeys up at the eaves, growing downwards (glTF)
    assert np.isclose(uv[0, 1], 0.0)
    assert np.isclose(uv[2, 1], -3.0 / fc.TEX_STOREYS)


def test_wall_uv_pieces_share_the_mapping_of_their_ring():
    v, nrm = _wall(0.0, 6.0, 0.0, 9.0)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    (vb, _), (va, _) = fc.clip_triangles(v, tris, 4.0)
    full = fc.wall_uv(v, nrm, 0.0, 3.0, 3.0, ring=v)
    upper = fc.wall_uv(va, nrm, 0.0, 3.0, 3.0, ring=v)
    assert np.isclose(upper[:, 0].min(), full[:, 0].min())
    assert np.isclose(upper[:, 0].max(), full[:, 0].max())


def _area(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()


def test_clip_triangles_keeps_area_and_winding():
    v, nrm = _wall(0.0, 5.0, 0.0, 8.0)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    (vb, tb), (va, ta) = fc.clip_triangles(v, tris, 3.0)
    assert np.isclose(_area(vb, tb), 15.0) and np.isclose(_area(va, ta), 25.0)
    assert vb[:, 2].max() <= 3.0 + 1e-9 and va[:, 2].min() >= 3.0 - 1e-9
    for vv, tt in ((vb, tb), (va, ta)):
        n = np.cross(vv[tt[:, 1]] - vv[tt[:, 0]], vv[tt[:, 2]] - vv[tt[:, 0]])
        assert (n @ np.cross(v[1] - v[0], v[2] - v[0]) > 0).all()


def test_clip_triangles_plane_outside():
    v, _ = _wall(0.0, 5.0, 0.0, 8.0)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    (vb, tb), (va, ta) = fc.clip_triangles(v, tris, -1.0)
    assert len(tb) == 0 and len(ta) == 2


def test_roof_uv_north_up():
    v = np.array([[250.0, 500.0, 20.0], [500.0, 250.0, 20.0]])
    uv = fc.roof_uv(v, (1, 1), 250.0)
    assert np.allclose(uv, [[0.0, 0.0], [1.0, 1.0]])


@pytest.mark.parametrize("name", sorted(fc.STYLES))
def test_synth_facade_tiles_and_is_deterministic(name):
    st = fc.STYLES[name]
    a, n, o = fc.synth_facade(st, 0, px=256)
    assert a.shape == (256, 256, 3) and n.shape == (256, 256, 3) and o.shape == (256, 256, 4)
    assert a.dtype == np.uint8
    b, _, _ = fc.synth_facade(st, 0, px=256)
    assert np.array_equal(a, b)
    # seamless: the wrap-around step is no larger than the steps inside the texture (brick joints)
    for axis in (0, 1):
        x = np.moveaxis(a.astype(int), axis, 0)
        inner = np.abs(np.diff(x, axis=0)).mean(axis=(1, 2)).max()
        wrap = np.abs(x[0] - x[-1]).mean()
        assert wrap <= 1.1 * inner + 1
    if not st.blank:
        # windows: some glossy pixels in the roughness channel
        assert (o[..., 1] < 40).any()


def test_variants_differ():
    st = fc.STYLES["brick_postwar"]
    a0, _, _ = fc.synth_facade(st, 0, px=128)
    a1, _, _ = fc.synth_facade(st, 1, px=128)
    assert not np.array_equal(a0, a1)
    assert fc.variant_of("brick_postwar", "x") in (0, 1)
    assert fc.material_key("brick_postwar", 1) == "facade_brick_postwar_1"


def test_ortho_tile_and_mean_colour():
    # 100 m x 100 m photo at 1 m, x in [0, 100], y in [0, 100], north up; left half red
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :50] = (200, 0, 0)
    img[:, 50:] = (0, 0, 200)
    tf = (1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    t = fc.ortho_tile(img, tf, (0, 0), 100.0, 10)
    assert t.shape == (10, 10, 3)
    assert tuple(t[0, 0]) == (200, 0, 0) and tuple(t[9, 9]) == (0, 0, 200)
    assert fc.ortho_tile(img, tf, (5, 5), 100.0, 10) is None
    ring = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
    assert fc.mean_colour(img, tf, ring) == (200, 0, 0)


def test_height_to_normal_flat():
    n = fc.height_to_normal(np.zeros((8, 8)))
    assert (n == (128, 128, 255)).all()


def test_procedural_capable_buildings():
    assert fc.procedural_ok("brick_postwar", 12.0)
    assert not fc.procedural_ok("brick_postwar", 6.0)          # too low for door + window + top rows
    assert not fc.procedural_ok("industrial", 20.0)            # no warehouse kit
    assert not fc.procedural_ok("blank_brick", 20.0)
    assert not fc.procedural_ok("brick_postwar", 9.55)         # Skysc21 needs 5.08 + 3.5 + 0.99 m
    assert fc.procedural_family("office_glass", 30.0) == "Skysc19"
    assert fc.procedural_family("office_glass", 12.0) == "Skysc21"   # too low for 5 m office storeys
    assert fc.procedural_ok("office_glass", 12.0)
    assert fc.procedural_family("brick_old", 30.0) == "Skysc21"


# --------------------------------------------------------------------------- lit windows at night

def test_wall_uv_tile_offset_moves_by_whole_tiles():
    v, nrm = _wall(10.0, 17.3, 5.0, 14.0)
    a = fc.wall_uv(v, nrm, base_z=5.0, storey_h=3.0, bay_m=2.4)
    b = fc.wall_uv(v, nrm, base_z=5.0, storey_h=3.0, bay_m=2.4, tile_offset=(3, 2))
    assert np.allclose(b - a, [[3.0, 2.0]])


def test_light_offset_is_deterministic_and_spread():
    offs = [fc.light_offset("NL.IMBAG.Pand.%d" % i, w) for i in range(400) for w in range(3)]
    assert offs == [fc.light_offset("NL.IMBAG.Pand.%d" % i, w) for i in range(400) for w in range(3)]
    assert {o[0] for o in offs} == set(range(fc.MASK_OFFSETS[0]))
    assert {o[1] for o in offs} == set(range(fc.MASK_OFFSETS[1]))
    # v per building: every wall of a building keeps its storeys' rows
    assert len({fc.light_offset("b", w)[1] for w in range(10)}) == 1
    assert fc.MASK_UV_SCALE * fc.MASK_OFFSETS[0] == 1.0


def test_blank_styles_have_no_window_lights():
    assert fc.synth_window_lights(fc.STYLES["blank_brick"]) is None
    assert set(fc.USE_OF_STYLE) == {n for n, s in fc.STYLES.items() if not s.blank}


@pytest.mark.parametrize("name", sorted(n for n, s in fc.STYLES.items() if not s.blank))
def test_window_lights_mask(name):
    st = fc.STYLES[name]
    m = fc.synth_window_lights(st, 0, px_bay=32)
    assert m.shape == (fc.MASK_STOREYS * 32, fc.MASK_BAYS * 32, 3) and m.dtype == np.uint8
    assert np.array_equal(m, fc.synth_window_lights(st, 0, px_bay=32))
    assert not np.array_equal(m, fc.synth_window_lights(st, 1, px_bay=32))
    # the lit glass lies inside the facade texture's glass (AO 0.85 behind the frame, glazing bars
    # included), in every one of its tiles, and fills most of it
    _, _, orm = fc.synth_facade(st, 0, px=64)
    glass_tex = orm[..., 0] < 240
    glass_mask = m[..., 0] > 0
    for ty in range(fc.MASK_OFFSETS[1]):
        for tx in range(fc.MASK_OFFSETS[0]):
            tile = glass_mask[ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64]
            assert not (tile & ~glass_tex).any()
            assert tile.sum() > 0.6 * glass_tex.sum()
    # ranks: about the share of windows is lit at any share, nested as the share drops
    for share in (0.1, 0.4, 0.7):
        assert abs(fc.lit_fraction(m, share) - share) < 0.04
    cells = m[16::32, 16::32, 1]
    assert ((cells < 0.1 * 255) <= (cells < 0.4 * 255)).all()
    assert cells.min() > 0 and cells.max() < 255      # never lit by day, never always on
    # a rank is uniform over its cell (filtering never mixes two windows' ranks inside the glass)
    c = m[:32, :32, 1]
    assert (c == c[0, 0]).all()
