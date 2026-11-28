"""Facade styles, wall/roof UVs and synthetic facade textures for real 3D buildings.

3DBAG (LoD 2.2) gives every Dutch building its real footprint, eaves and roof shape, but no
appearance, and so do the other city models (the NYC 3D Building Model, PLATEAU; see
twinmodel.export.realbuildings). This module turns their attributes into a look:

* ``classify`` picks a facade style from the construction year (BAG ``oorspronkelijkbouwjaar``),
  the dominant use of the building's units (BAG ``gebruiksdoel``, weighted by floor area, or the
  neutral names other providers map their registers to: residential, office, retail, ...),
  its height and footprint: pre-war brick, post-war brick, modern brick or render for homes,
  concrete or glass for offices, metal cladding for industry, a shopfront ground floor where
  the building has shops or cafes, a plain wall for party walls and sheds.
* ``wall_uv`` maps a wall polygon so the texture's window grid lines up with the building:
  a whole number of bays across each wall and a whole number of storeys up to the eaves.
* ``synth_facade`` draws the tiling textures (base colour, normal, AO/roughness/metallic) of a
  style with numpy, so the pipeline needs no texture download: 2 bays x 2 storeys a tile.
* ``synth_window_lights`` draws the night mask of a style: which windows are lit and how (a rank
  per window that the material compares with the lit share of the building use, a brightness and
  a warm/cool tint), over MASK_BAYS x MASK_STOREYS windows so the lit pattern does not repeat every
  2 x 2 windows; ``light_offset`` shifts it per building and wall by whole facade tiles.
* ``roof_uv`` and ``ortho_tile`` texture the roofs with the aerial photo (PDOK Luchtfoto,
  CC BY 4.0, the twin's ortho): one texture per World Partition tile, planar from the top.

Pure numpy (PIL only to write PNGs); unit tested in tests/test_facades.py.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

TEX_BAYS = 2      # one facade texture covers 2 bays ...
TEX_STOREYS = 2   # ... by 2 storeys
TEX_PX = 1024
SHOP_FLOOR_M = 4.0   # shopfront ground floor height
PARTY_TOL_M = 0.3    # a wall whose foot lies this close to another footprint is a party wall


@dataclass(frozen=True)
class Style:
    name: str
    bay_m: float                # module width
    storey_m: float             # nominal storey height (the UVs fit a whole number to the eaves)
    wall: str                   # brick | render | concrete | metal | glass
    colours: tuple              # wall colour per variant (sRGB 0-255)
    window: tuple = (0.25, 0.75, 0.30, 0.85)   # x0, x1 (of the bay), y0, y1 (of the storey, from the floor)
    frame: tuple = (235, 235, 230)
    frame_m: float = 0.06
    mullions: int = 0           # vertical glazing bars per window
    transom: bool = False       # horizontal bar at 2/3 height
    sill: bool = True
    lintel: bool = False        # darker soldier course above the window (brick)
    glass: tuple = (54, 60, 60)   # neutral grey-green glass (it was (52, 62, 72): renders read blue, B/G 1.2)
    mortar: tuple = (190, 185, 175)
    brick_m: tuple = (0.22, 0.0625)   # brick length incl. joint, course height
    shop: bool = False          # shopfront: sign band, door
    blank: bool = False         # no windows at all
    band: Optional[tuple] = None   # a horizontal band at every floor (concrete floor edge, stone string course)
    band_m: float = 0.25           # its height
    surround: Optional[tuple] = None   # a moulded window surround (stone or stucco architrave) of this colour
    surround_m: float = 0.14           # its width around the frame
    balcony: bool = False          # a wrought-iron railing and a stone slab across each (floor-length) window
    joint_m: float = 0.010         # brick / tile joint width
    parapet: Optional[tuple] = None   # Japanese apartment balcony: a solid parapet panel of this colour across the
                                      # whole bay in front of floor-length sliding doors, on a slab edge
    spandrel: Optional[tuple] = None  # curtain wall: an opaque glass band of this colour over each floor slab
    ties: bool = False             # exposed concrete: form-tie holes on a 0.6 x 0.45 m grid, formwork panel joints
    sign_band: bool = False        # Japanese shop floor: a tall signboard band, one coloured panel per bay
    glass_tint: float = 0.08       # per-window spread of the glass brightness (reflections, tint)


STYLES = {s.name: s for s in (
    # pre-war Dutch brick: tall windows, white frames, transom, darker lintels, a light string course per floor
    # (Eindhoven's Vestdijk, Kerkstraat, Stratumseind: orange-red and dark brown-grey brick, street photos
    # 2025; the first two colours were darker before 2026-11-16)
    Style("brick_old", 2.4, 3.2, "brick", ((152, 74, 52), (124, 60, 44), (98, 82, 72)),
          window=(0.28, 0.72, 0.22, 0.86), frame=(240, 238, 230), mullions=1, transom=True, lintel=True,
          band=(214, 206, 188), band_m=0.14),
    # Dutch reconstruction 1940-1964 (wederopbouw: the Eindhoven centre after 1942-44): buff and yellow-brown
    # brick, wide steel windows with white frames, concrete floor edges
    Style("brick_wederopbouw", 3.0, 3.0, "brick", ((178, 140, 98), (156, 112, 78), (190, 164, 124)),
          window=(0.10, 0.90, 0.28, 0.84), frame=(228, 226, 218), mullions=2, band=(188, 184, 174), band_m=0.25),
    # 1965-1979: dark brown brick, wide windows in dark wooden frames, concrete floor edges
    Style("brick_late", 3.0, 2.8, "brick", ((104, 72, 56), (122, 84, 62), (86, 66, 58)),
          window=(0.08, 0.92, 0.30, 0.86), frame=(66, 48, 36), mullions=2, sill=False, band=(176, 172, 164),
          band_m=0.30),
    # 1960-1979 flats (galerijflats, system building): storey-high grey concrete panels, wide windows
    Style("concrete_panel", 2.7, 2.8, "concrete", ((178, 174, 166), (160, 156, 148)),
          window=(0.12, 0.88, 0.32, 0.86), frame=(225, 225, 220), mullions=1, sill=False),
    # homes from 1995: buff, anthracite, red or light grey brick, tall windows in anthracite aluminium frames
    Style("brick_contemporary", 2.7, 3.0, "brick", ((186, 164, 134), (82, 80, 80), (150, 82, 60), (200, 196, 188)),
          window=(0.14, 0.86, 0.10, 0.90), frame=(52, 54, 58), mullions=1, sill=False),
    # offices 1985-1999 below 25 m: brick spandrels between ribbon windows
    Style("office_brick", 3.6, 3.4, "brick", ((132, 72, 54), (170, 146, 112)),
          window=(0.05, 0.95, 0.30, 0.88), frame=(90, 92, 96), mullions=2, sill=False),
    # post-war reconstruction (most of the Eindhoven centre): wide windows, darker frames
    Style("brick_postwar", 3.0, 2.9, "brick", ((150, 98, 66), (128, 82, 58)),
          window=(0.12, 0.88, 0.30, 0.82), frame=(70, 60, 52), mullions=2),
    Style("brick_modern", 2.8, 2.9, "brick", ((140, 64, 44), (88, 70, 62)),
          window=(0.18, 0.82, 0.18, 0.86), frame=(60, 62, 64), mullions=1),
    Style("render_modern", 3.0, 3.0, "render", ((228, 224, 214), (205, 200, 190)),
          window=(0.15, 0.85, 0.15, 0.88), frame=(55, 57, 60), mullions=1, sill=False),
    # offices: ribbon windows between concrete spandrels / a curtain wall
    Style("office_concrete", 3.6, 3.4, "concrete", ((172, 168, 160), (198, 196, 190)),
          window=(0.04, 0.96, 0.32, 0.86), frame=(80, 82, 85), mullions=2, sill=False),
    # curtain wall: neutral grey-green glass (it was blue, (58, 76, 90)), dark spandrel glass over each floor slab,
    # aluminium mullions and transoms; the panes' brightness varies (reflections, blinds, lit and dark rooms)
    Style("office_glass", 1.8, 3.6, "glass", ((86, 90, 92), (120, 122, 120), (64, 68, 70)),
          window=(0.03, 0.97, 0.08, 0.97), frame=(150, 155, 160), frame_m=0.05, sill=False,
          glass=(62, 70, 70), transom=True, spandrel=(44, 48, 50), glass_tint=0.14),
    # industry and warehouses: profiled metal sheet, one strip of high windows
    Style("industrial", 6.0, 4.5, "metal", ((168, 174, 178), (120, 128, 132)),
          window=(0.10, 0.90, 0.70, 0.86), frame=(90, 92, 95), mullions=3, sill=False),
    # ground floor of a building with shops or cafes
    Style("shopfront", 4.0, SHOP_FLOOR_M, "render", ((60, 60, 62), (110, 95, 80)),
          window=(0.06, 0.94, 0.03, 0.74), frame=(40, 40, 42), frame_m=0.08, mullions=1, sill=False,
          glass=(50, 56, 56), shop=True),
    # pre-war masonry high-rise (New York: limestone and buff brick, punched windows)
    Style("stone_prewar", 1.8, 3.4, "brick", ((196, 184, 160), (172, 150, 118), (150, 112, 86)),
          window=(0.26, 0.74, 0.24, 0.84), frame=(225, 222, 212), mullions=0, transom=True,
          mortar=(205, 200, 188), brick_m=(0.60, 0.30)),
    # southern Europe and Latin America, before 1940: sandstone or limestone ashlar, tall balcony doors
    # (Salamanca's golden Villamayor stone, Paris and Madrid limestone, grey granite)
    Style("stone_old", 2.2, 3.4, "brick", ((206, 170, 112), (214, 200, 172), (168, 160, 150)),
          window=(0.30, 0.70, 0.06, 0.82), frame=(92, 70, 52), mullions=1, transom=False,
          mortar=(196, 186, 166), brick_m=(0.70, 0.35)),
    # southern homes and blocks after 1940: warm render (cream, ochre, terracotta), shutters' dark frames
    Style("render_warm", 3.0, 3.0, "render", ((232, 214, 178), (214, 168, 112), (196, 128, 92)),
          window=(0.22, 0.78, 0.12, 0.86), frame=(70, 66, 58), mullions=1, sill=True),
    # old towns of southern Europe (historic area or before 1900): ashlar in the local stone, moulded surrounds,
    # a cornice at every floor, floor-length balcony doors with wrought-iron balconies (Salamanca's golden
    # Villamayor sandstone and its Plaza Mayor; also Madrid, Seville, Lisbon limestone)
    Style("stone_balcony", 2.4, 3.6, "brick", ((206, 166, 104), (196, 152, 92), (214, 182, 124), (188, 170, 140)),
          window=(0.31, 0.69, 0.03, 0.80), frame=(72, 54, 40), mullions=1, transom=False, sill=False,
          mortar=(200, 170, 120), brick_m=(0.80, 0.40), band=(184, 146, 92), band_m=0.30,
          surround=(222, 194, 140), surround_m=0.16, balcony=True),
    # old towns of central and northern Europe (historic area or before 1900): stucco in pale ochre, yellow, white,
    # grey, salmon or green, white moulded surrounds and cornices, tall two-pane windows (Helsinki's Senate Square
    # and Kruununhaka, Zurich's Altstadt, Vienna, Prague)
    Style("stucco_old", 2.6, 3.5, "render", ((226, 204, 140), (234, 230, 218), (214, 180, 124), (204, 200, 190),
                                             (220, 174, 142), (196, 204, 176)),
          window=(0.30, 0.70, 0.16, 0.82), frame=(242, 240, 232), mullions=1, transom=True, sill=True,
          band=(238, 234, 222), band_m=0.22, surround=(242, 238, 228), surround_m=0.12),
    # central and northern European homes and blocks after 1900 or of unknown age: light plaster, not brick
    Style("render_central", 2.8, 3.0, "render", ((230, 228, 220), (212, 208, 196), (198, 194, 186), (224, 214, 194)),
          window=(0.20, 0.80, 0.16, 0.86), frame=(84, 84, 86), mullions=1, sill=True),
    # Japan (PLATEAU wards; street photos of Shibuya and Tokyo apartment and office blocks as reference only):
    # ceramic tile cladding (95 x 45 mm "nikoguchi" tiles) in beige, brown, off-white and grey, aluminium sliding
    # windows without sills
    Style("jp_tile", 3.0, 3.1, "tile", ((212, 198, 172), (176, 140, 106), (224, 222, 214), (132, 126, 118),
                                        (192, 180, 162)),
          window=(0.12, 0.88, 0.30, 0.86), frame=(178, 180, 180), frame_m=0.05, mullions=1, sill=False,
          mortar=(204, 200, 192), brick_m=(0.10, 0.05), joint_m=0.005),
    # Japanese apartment blocks ("mansions"): a balcony across every bay with a solid parapet panel (light concrete
    # or frosted glass) in front of floor-length sliding doors, tiled walls behind
    Style("jp_mansion", 3.4, 2.95, "tile", ((226, 222, 212), (206, 196, 180), (186, 170, 150), (156, 146, 134)),
          window=(0.08, 0.92, 0.02, 0.86), frame=(176, 178, 178), frame_m=0.05, mullions=1, sill=False,
          mortar=(210, 206, 198), brick_m=(0.10, 0.05), joint_m=0.005, parapet=(232, 230, 222)),
    # Japanese offices and mixed blocks: ribbon windows between aluminium spandrel panels
    Style("jp_office", 3.2, 3.6, "metal_panel", ((178, 180, 180), (150, 152, 154), (206, 204, 198), (118, 120, 122)),
          window=(0.02, 0.98, 0.30, 0.88), frame=(104, 106, 108), frame_m=0.05, mullions=2, sill=False,
          glass=(60, 66, 66), glass_tint=0.12),
    # exposed concrete (uchippanashi): formwork panel joints and form-tie holes, steel windows
    Style("jp_concrete", 2.7, 3.2, "concrete", ((170, 168, 162), (150, 148, 144)),
          window=(0.16, 0.84, 0.26, 0.88), frame=(62, 64, 66), frame_m=0.05, mullions=1, sill=False, ties=True),
    # Japanese shop floor: glazed fronts under a tall signboard band, one coloured panel per bay (no lettering:
    # the shop signs themselves are props, coupled-carla streetprops)
    Style("shopfront_jp", 3.6, SHOP_FLOOR_M, "render", ((52, 52, 54), (92, 88, 84)),
          window=(0.05, 0.95, 0.03, 0.70), frame=(44, 44, 46), frame_m=0.07, mullions=1, sill=False,
          glass=(50, 56, 56), shop=True, sign_band=True),
    # party walls, narrow walls, sheds: plain wall at true brick scale
    Style("blank_brick", 2.5, 3.0, "brick", ((132, 60, 44), (110, 80, 62)), blank=True, sill=False),
    # the plain wall of the light and buff brick styles
    Style("blank_brick_light", 2.5, 3.0, "brick", ((178, 148, 110), (160, 150, 140)), blank=True, sill=False),
    # the plain wall of the stucco and plaster styles
    Style("blank_render", 2.5, 3.0, "render", ((214, 206, 190), (196, 190, 178)), blank=True, sill=False),
    # the plain wall of the Japanese styles: concrete
    Style("blank_concrete", 2.5, 3.0, "concrete", ((176, 174, 168), (158, 156, 150)), blank=True, sill=False),
)}

# styles drawn at twice the texture size: the street-level storeys, seen from a few metres (TEX_PX already gives
# 0.5-0.8 cm per texel on the upper storeys; a shop floor seen from the pavement needs about twice that)
STREET_STYLES = {"shopfront", "shopfront_jp"}
STREET_PX_MAX = 2048


def texture_px_for(style: str, px: int = TEX_PX) -> int:
    """Texture size of a style's facade: the street-level styles at twice ``px`` (at most STREET_PX_MAX)."""
    return min(STREET_PX_MAX, 2 * px) if style in STREET_STYLES else px

# style -> the plain wall its party walls get
BLANK_OF = {"stone_prewar": "blank_brick", "stone_old": "blank_brick", "render_warm": "blank_brick",
            "render_modern": "blank_brick", "office_concrete": "blank_brick", "office_glass": "blank_brick",
            "industrial": "industrial", "shopfront": "blank_brick", "brick_wederopbouw": "blank_brick_light",
            "brick_contemporary": "blank_brick_light", "concrete_panel": "blank_brick", "office_brick": "blank_brick",
            "stone_balcony": "blank_render", "stucco_old": "blank_render", "render_central": "blank_render",
            "jp_tile": "blank_concrete", "jp_mansion": "blank_concrete", "jp_office": "blank_concrete",
            "jp_concrete": "blank_concrete", "shopfront_jp": "blank_concrete"}


def _crc(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


def material_key(style: str, variant: int, windows: bool = True) -> str:
    """Material (and texture) key of a style's facade; ``windows=False``: the same wall without painted windows
    ("facadenw_..."), for walls whose windows are geometry."""
    return ("facade_%s_%d" if windows else "facadenw_%s_%d") % (style, variant)


def parse_material_key(key: str) -> tuple[str, int, bool]:
    """(style, variant, windows painted) of a facade material key."""
    windows = key.startswith("facade_")
    sname, var = key.split("_", 1)[1].rsplit("_", 1)
    return sname, int(var), windows


def variant_of(style: str, building_id: str, colour: Optional[tuple] = None) -> int:
    """The colour variant of a style for a building: the one nearest the mapped wall colour (OSM
    building:colour) when there is one, else a stable pick from the building id."""
    cols = STYLES[style].colours
    if colour is not None and len(cols) > 1:
        return int(min(range(len(cols)), key=lambda k: sum((a - b) ** 2 for a, b in zip(cols[k], colour))))
    return _crc(style, building_id) % len(cols)


# OSM colour values (building:colour, building:facade:colour): CSS names people use, and #rrggbb
NAMED_COLOURS = {
    "white": (240, 240, 236), "black": (30, 30, 30), "grey": (128, 128, 128), "gray": (128, 128, 128),
    "lightgrey": (200, 200, 200), "lightgray": (200, 200, 200), "darkgrey": (90, 90, 90),
    "darkgray": (90, 90, 90), "silver": (192, 192, 192), "red": (150, 60, 45), "darkred": (110, 40, 35),
    "maroon": (110, 40, 35), "brown": (120, 80, 60), "sienna": (160, 82, 45), "orange": (200, 120, 60),
    "yellow": (220, 200, 120), "beige": (225, 210, 180), "tan": (210, 180, 140), "wheat": (225, 205, 165),
    "khaki": (200, 190, 140), "ivory": (240, 236, 220), "cream": (236, 226, 200), "pink": (230, 180, 175),
    "salmon": (220, 140, 115), "green": (90, 120, 80), "blue": (80, 100, 140), "lightblue": (170, 200, 220),
    "brick": (150, 70, 50), "sand": (215, 195, 155), "terracotta": (190, 100, 70)}


def parse_colour(text) -> Optional[tuple]:
    """(r, g, b) of an OSM colour value (#rgb, #rrggbb or a name), or None."""
    if not text:
        return None
    t = str(text).strip().lower().split(";")[0].replace(" ", "").replace("_", "")
    if t.startswith("#"):
        h = t[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return None
        return None
    return NAMED_COLOURS.get(t)


# --------------------------------------------------------------------------- classification

# uses: BAG gebruiksdoel names (the Netherlands) and the neutral names the other providers map their
# registers to (New York: PLUTO floor areas and building classes)
OFFICE_USES = {"kantoorfunctie", "onderwijsfunctie", "gezondheidszorgfunctie", "logiesfunctie", "celfunctie",
               "office", "education", "health", "hotel", "government"}
SHOP_USES = {"winkelfunctie", "bijeenkomstfunctie", "retail", "assembly"}
RETAIL_USES = {"winkelfunctie", "retail"}   # shops proper; assembly (bijeenkomst) also covers halls and auditoria
INDUSTRY_USES = {"industriefunctie", "industry", "storage", "utility"}
INDUSTRIAL_MAX_M = 30.0   # taller "industrial" buildings are towers on oddly registered lots: not metal sheds
OSM_INDUSTRY = {"industrial", "warehouse", "hangar", "factory", "manufacture"}
OSM_SHED = {"shed", "garage", "garages", "carport", "hut", "roof"}
OSM_OFFICE = {"office", "commercial", "public", "civic", "school", "university", "college", "hospital", "hotel"}


@dataclass
class BuildingInfo:
    """What classify needs to know about a building."""
    id: str
    year: Optional[int] = None
    uses: dict = field(default_factory=dict)   # gebruiksdoel -> floor area m2
    height_m: float = 0.0                      # ground to eaves
    area_m2: float = 0.0                       # footprint
    osm_category: Optional[str] = None
    region: str = "nl"                         # "us": pre-war high-rise is stone; "jp": homes are rendered;
                                               # "south": stone before 1940, warm render after (no brick)
    shop_pois: Optional[int] = None            # ground-floor shops, cafes, ... (OSM) at the facade; None: unknown
    material: Optional[str] = None             # OSM building:material / building:facade:material
    historic: bool = False                     # in an old town: a historic area (OSM heritage, historic district,
                                               # protected area) or the building itself tagged historic/heritage


def dominant_use(uses: dict) -> Optional[str]:
    if not uses:
        return None
    return max(sorted(uses), key=lambda k: uses[k])


# OSM building:material -> the wall it means
MATERIAL_WALL = {"glass": "glass", "concrete": "concrete", "reinforced_concrete": "concrete", "brick": "brick",
                 "bricks": "brick", "clinker": "brick", "plaster": "render", "render": "render", "stucco": "render",
                 "plastered": "render", "metal": "metal", "steel": "metal", "aluminium": "metal",
                 "metal_plates": "metal", "stone": "stone", "sandstone": "stone", "limestone": "stone",
                 "granite": "stone", "marble": "stone", "tile": "tile", "tiles": "tile", "ceramic": "tile",
                 "ceramic_tiles": "tile"}
CAMPUS_CATS = {"university", "college", "school", "hospital", "public", "civic", "government"}


def _nl_home(b: BuildingInfo, year: int) -> str:
    """The Dutch facade of a home or mixed block by its construction year (BAG oorspronkelijkbouwjaar)."""
    if 0 < year < 1940:
        return "brick_old"
    if year == 0:
        return "brick_postwar"
    if year < 1965:
        return "brick_wederopbouw"
    if year < 1980:
        # flats of four storeys and more: a share are concrete system building, the rest brick
        if b.height_m >= 11.5 and year >= 1960 and _crc("panel", b.id) % 3 < 2:
            return "concrete_panel"
        return "brick_late"
    if year < 1995:
        return "brick_modern"
    if _crc("render", b.id) % 4 == 0:
        return "render_modern"
    return "brick_contemporary"


def _by_material(b: BuildingInfo, style: str, home: bool) -> str:
    """The style an OSM building:material asks for, given the style the year and use chose."""
    wall = MATERIAL_WALL.get((b.material or "").strip().lower().split(";")[0])
    if wall is None or STYLES[style].wall == wall or style in ("shopfront", "shopfront_jp", "blank_brick",
                                                                "industrial"):
        return style
    if wall == "tile":
        return "jp_tile" if b.region == "jp" else style
    if wall == "glass":
        return "office_glass"
    if wall == "concrete" and b.region == "jp":
        return "jp_concrete"
    if wall == "concrete":
        return "concrete_panel" if home else "office_concrete"
    if wall == "metal":
        return "industrial" if b.height_m < INDUSTRIAL_MAX_M else "office_glass"
    if wall == "stone":
        return "stone_prewar" if b.region == "us" else "stone_old"
    if wall == "render":
        return "render_warm" if b.region == "south" else "stucco_old" if b.region == "central" and b.historic else \
            "render_central" if b.region == "central" else "render_modern"
    if wall == "brick":
        if not home:
            return "office_brick"
        return "brick_contemporary" if b.region == "nl" and style == "render_modern" else \
            ("brick_modern" if style.startswith(("render", "stone")) else style)
    return style


RELIGIOUS = {"church", "cathedral", "chapel", "basilica", "monastery", "convent", "mosque", "synagogue", "temple",
             "religious", "shrine"}   # OSM building values: never office glass, whatever their height
OLD_TOWN_YEAR = 1900   # built before this: an old-town facade in the south and central regions, area or not
OLD_TOWN_MAX_M = 30.0  # taller buildings in an old town keep their own style (towers, modern blocks)


def shop_style(region: str) -> str:
    """The shop-floor style of a region: Japanese signboard fronts in Japan, else the plain shopfront."""
    return "shopfront_jp" if region == "jp" else "shopfront"


JP_TALL_M = 31.0   # Japan's old 31 m height limit: taller blocks are modern towers (curtain walls, panels)


def _jp(b: BuildingInfo, use, cat) -> str:
    """The facade of a Japanese building (PLATEAU usage and height; Shibuya and Tokyo street photos as reference):
    offices, hotels and big shops are curtain walls above JP_TALL_M, else aluminium panels, tiles or exposed
    concrete; homes and mixed blocks are tiled walls, apartment blocks of four storeys and more mostly with balcony
    parapets across every bay. A stable share per building id, so neighbours differ."""
    pick = _crc("jp", b.id) % 10
    office = use in OFFICE_USES or use in SHOP_USES or (use is None and cat in OSM_OFFICE)
    if office:
        if b.height_m >= JP_TALL_M:
            return "office_glass" if pick < 5 else "jp_office" if pick < 8 else "jp_tile"
        return "jp_office" if pick < 4 else "jp_tile" if pick < 7 else "jp_concrete" if pick < 9 else "office_glass"
    if b.height_m >= 11.0:
        return "jp_mansion" if pick < 6 else "jp_tile" if pick < 9 else "jp_concrete"
    return "jp_tile" if pick < 4 else "render_modern" if pick < 9 else "jp_concrete"


def classify(b: BuildingInfo) -> tuple[str, bool]:
    """(facade style, shopfront ground floor) for a building."""
    style, shop = _classify(b)
    home = USE_OF_STYLE.get(style) == "home"
    return _by_material(b, style, home), shop


def _classify(b: BuildingInfo) -> tuple[str, bool]:
    use = dominant_use(b.uses)
    retail = any(u in RETAIL_USES for u in b.uses)
    if b.shop_pois is None:      # no OSM shops at hand: the units' uses alone, as before
        shops = any(u in SHOP_USES for u in b.uses)
    else:                        # a shop unit (BAG), or a shop or cafe mapped at the facade
        shops = retail or b.shop_pois > 0
    year = b.year or 0
    cat = (b.osm_category or "").lower()
    small = b.area_m2 < 40.0 and b.height_m < 4.0
    if small or (not b.uses and cat in OSM_SHED):
        return "blank_brick", False
    big_low = use is None and b.area_m2 > 1500.0 and b.height_m < 10.0 and cat not in CAMPUS_CATS
    if (use in INDUSTRY_USES or (not b.uses and cat in OSM_INDUSTRY) or big_low) and b.height_m < INDUSTRIAL_MAX_M:
        return "industrial", False
    if use in INDUSTRY_USES:   # a tower on a lot registered as utility or storage (New York): an office look
        use = "office"
    shop_floor = shops and b.height_m >= SHOP_FLOOR_M + 2.5
    if b.region == "us" and 0 < year < 1946 and b.height_m >= 20.0:
        return "stone_prewar", shop_floor   # Manhattan's pre-war towers and apartment blocks
    if b.region in ("south", "central") and (cat in RELIGIOUS or use == "religious"):
        # a church or cathedral: the region's stone or stucco (HelsinkiSenate's cathedral, 2026-11-25: tall and
        # without a year, it had the glass office look)
        return ("stone_old" if b.region == "south" else "stucco_old"), False
    old_town = b.region in ("south", "central") and (b.historic or 0 < year < OLD_TOWN_YEAR) and \
        b.height_m < OLD_TOWN_MAX_M
    if old_town and b.region == "south":
        # Catastro's year is often the last refit, so the historic area decides: ashlar and balconies for homes
        # and mixed blocks, plain ashlar for public buildings, churches and offices
        home_like = use not in OFFICE_USES and not (use is None and cat in OSM_OFFICE)
        return ("stone_balcony" if home_like else "stone_old"), shop_floor
    if old_town:
        return "stucco_old", shop_floor
    if b.region == "south" and 0 < year < 1940 and b.height_m < 40.0:
        return "stone_old", shop_floor      # old public buildings, churches, colleges: stone, not office glass
    one_storey_shop = b.height_m < SHOP_FLOOR_M + 2.5 and (
        use in SHOP_USES if b.shop_pois is None else (use in RETAIL_USES or (use in SHOP_USES and b.shop_pois > 0)))
    if one_storey_shop:
        return shop_style(b.region), False   # a one-storey shop: shopfront all the way up
    if b.region == "jp":
        return _jp(b, use, cat), shop_floor
    if b.region == "nl" and 0 < year < 1940 and b.height_m < 30.0:
        # before 1940 shops, cafes, offices and halls were brick buildings too (Eindhoven's Demer and
        # Vestdijk: 1900-1924 cafes and offices had the concrete office look before 2026-11-16)
        return "brick_old", shop_floor
    if use in OFFICE_USES or (use is None and cat in OSM_OFFICE) or \
            (use in SHOP_USES and b.height_m >= 12.0) or \
            (use in SHOP_USES and use not in RETAIL_USES and b.shop_pois == 0):   # a hall, an auditorium
        if year >= 1985 or (year == 0 and b.height_m >= 25.0):
            if b.region == "nl" and year < 2000 and b.height_m < 25.0 and \
                    use in ("kantoorfunctie", "logiesfunctie", "office", "hotel"):
                return "office_brick", shop_floor     # Dutch offices of the late 1980s and 1990s
            return "office_glass", shop_floor
        return "office_concrete", shop_floor
    # homes and mixed blocks
    if b.region == "south":
        if 0 < year < 1940:
            return "stone_old", shop_floor
        if year == 0 and _crc("south", b.id) % 4 == 0:
            return "stone_old", shop_floor   # no year (Overture, OSM): a quarter of the old stock
        return "render_warm", shop_floor
    if b.region == "nl":
        return _nl_home(b, year), shop_floor
    if b.region == "central":       # plaster, not brick (Helsinki, Zurich: Overture gives no year)
        return ("render_modern" if year >= 1995 and _crc("render", b.id) % 3 == 0 else "render_central"), shop_floor
    if 0 < year < 1940:
        return "brick_old", shop_floor
    if year < 1980:
        return "brick_postwar", shop_floor
    if year >= 1995 and _crc("render", b.id) % 3 == 0:
        return "render_modern", shop_floor
    return "brick_modern", shop_floor


# --buildings procedural (bake_level.py): carla-digitaltwins' modular atom kits need door + window + top
# rows (measured in the content, 2026-11-14: Skysc21 apartment kit 508 + 350 + 99 cm, Skysc19 glass office
# kit 533 + 502 + 502 cm), and they are apartment / office kits, not sheds, shops or warehouses
PROCEDURAL_FAMILY_MIN_M = {"Skysc21": 9.6, "Skysc19": 15.4}
PROCEDURAL_MIN_HEIGHT_M = min(PROCEDURAL_FAMILY_MIN_M.values())
PROCEDURAL_SIMPLIFY_M = 1.0   # footprint simplification before the generator (its modules are 5 m wide)
PROCEDURAL_STYLES = {"brick_old", "brick_postwar", "brick_modern", "render_modern", "office_concrete", "office_glass",
                     "brick_wederopbouw", "brick_late", "concrete_panel", "brick_contemporary", "office_brick"}


def procedural_family(style: str, height_m: float = 1e9) -> str:
    """bake_level.FAMILIES member for a building: the glass office kit for offices tall enough for its
    5 m storeys, else the apartment kit."""
    if style in ("office_glass", "office_concrete", "office_brick") and height_m >= PROCEDURAL_FAMILY_MIN_M["Skysc19"]:
        return "Skysc19"
    return "Skysc21"


def procedural_ok(style: str, height_m: float, min_height_m: float = PROCEDURAL_MIN_HEIGHT_M) -> bool:
    fam = procedural_family(style, height_m)
    return style in PROCEDURAL_STYLES and height_m >= max(min_height_m, PROCEDURAL_FAMILY_MIN_M[fam])


def storeys_for(height_m: float, storey_m: float) -> tuple[int, float]:
    """A whole number of storeys up to the eaves: (n, actual storey height)."""
    if height_m <= 0:
        return 1, storey_m
    n = max(1, int(round(height_m / storey_m)))
    h = height_m / n
    if h < 0.75 * storey_m or h > 1.35 * storey_m:   # an odd wall: keep the nominal storey
        return n, storey_m
    return n, h


# --------------------------------------------------------------------------- UVs

def wall_frame(normal: np.ndarray) -> np.ndarray:
    """Unit horizontal axis along a vertical wall (left to right seen from outside)."""
    nx, ny = float(normal[0]), float(normal[1])
    h = math.hypot(nx, ny)
    if h < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return np.array([-ny / h, nx / h, 0.0])


def wall_uv(verts: np.ndarray, normal: np.ndarray, base_z: float, storey_h: float, bay_m: float,
            ring: Optional[np.ndarray] = None, z0: float = 0.0, tile_offset: tuple = (0, 0)) -> np.ndarray:
    """glTF UVs (v grows downwards) for points of a wall polygon.

    u: a whole number of bays across the wall's horizontal extent (taken from ``ring``, the
    polygon's own vertices, so every piece of a clipped wall shares the mapping);
    v: storeys from ``base_z + z0`` (the floor of the first storey this texture covers).
    One texture tile is TEX_BAYS x TEX_STOREYS modules. ``tile_offset`` adds whole texture tiles
    (integers: the facade textures look the same, the night mask, which repeats every
    MASK_OFFSETS tiles, shows another part of itself; see light_offset)."""
    along = wall_frame(normal)
    ref = verts if ring is None else ring
    a_ref = ref @ along
    a0, a1 = float(a_ref.min()), float(a_ref.max())
    width = max(a1 - a0, 1e-6)
    bays = max(1, int(round(width / bay_m)))
    bay = width / bays
    u = (verts @ along - a0) / bay / TEX_BAYS + int(tile_offset[0])
    v = -(verts[:, 2] - (base_z + z0)) / storey_h / TEX_STOREYS + int(tile_offset[1])
    return np.column_stack([u, v])


def blank_uv(verts: np.ndarray, normal: np.ndarray, style: Style) -> np.ndarray:
    """Plain walls: metric, so the bricks keep their size whatever the wall."""
    along = wall_frame(normal)
    return np.column_stack([verts @ along / (TEX_BAYS * style.bay_m),
                            -verts[:, 2] / (TEX_STOREYS * style.storey_m)])


def roof_uv(verts: np.ndarray, tile: tuple[int, int], tile_m: float) -> np.ndarray:
    """Planar top-down UVs into the tile's ortho texture (image row 0 = north edge)."""
    x0, y1 = tile[0] * tile_m, (tile[1] + 1) * tile_m
    return np.column_stack([(verts[:, 0] - x0) / tile_m, (y1 - verts[:, 1]) / tile_m])


# --------------------------------------------------------------------------- clipping

def clip_triangles(verts: np.ndarray, tris: np.ndarray, z: float):
    """Split a triangle mesh at the plane Z = z. Returns ((verts, tris) below, (verts, tris) above);
    new vertices are interpolated on the cut, winding is preserved."""
    out = {True: ([], []), False: ([], [])}
    for t in tris:
        p = verts[t]
        side = p[:, 2] < z
        if side.all() or (~side).all():
            vs, fs = out[bool(side[0])]
            base = sum(len(a) for a in vs)
            vs.append(p)
            fs.append(np.array([[0, 1, 2]]) + base)
            continue
        poly = {True: [], False: []}
        for i in range(3):
            a, b = p[i], p[(i + 1) % 3]
            sa, sb = bool(a[2] < z), bool(b[2] < z)
            poly[sa].append(a)
            if sa != sb:
                f = (z - a[2]) / (b[2] - a[2])
                c = a + f * (b - a)
                poly[True].append(c)
                poly[False].append(c)
        for s in (True, False):
            pts = poly[s]
            if len(pts) < 3:
                continue
            vs, fs = out[s]
            base = sum(len(a) for a in vs)
            vs.append(np.asarray(pts))
            fs.append(np.array([[0, i, i + 1] for i in range(1, len(pts) - 1)]) + base)

    def pack(s):
        vs, fs = out[s]
        if not vs:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return np.vstack(vs), np.vstack(fs).astype(np.int64)
    return pack(True), pack(False)


# --------------------------------------------------------------------------- textures

def _periodic_noise(shape, scale_px: float, rng) -> np.ndarray:
    """Tileable smooth noise in [-1, 1] (white noise low-passed in the Fourier domain)."""
    h, w = shape
    n = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    filt = np.exp(-(f * scale_px) ** 2 * 4.0)
    out = np.real(np.fft.ifft2(np.fft.fft2(n) * filt))
    out -= out.mean()
    m = np.abs(out).max()
    return out / m if m > 0 else out


def _brick_layer(h, w, mw, mh, style: Style, colour, rng, tone_sd: float = 0.09):
    """Running-bond brick wall over a (mw x mh metre) tile: colour image and height map."""
    bl, ch = style.brick_m
    nb = max(4, int(round(mw / bl)))
    nc = max(4, int(round(mh / ch)))
    nc += nc % 2   # even number of courses so the half-brick offset tiles
    ys, xs = np.mgrid[0:h, 0:w]
    cy = ys * nc / h
    course = np.floor(cy).astype(int)
    cx = xs * nb / w + 0.5 * (course % 2)
    brick = np.floor(cx).astype(int) % nb
    fy, fx = cy - course, cx - np.floor(cx)
    jy = style.joint_m / ch      # 1 cm joints (tiles: 5 mm)
    jx = style.joint_m / bl
    mortar = (fy < jy) | (fx < jx)
    tone = rng.normal(0.0, tone_sd, size=(nc, nb))
    shade = 1.0 + tone[course % nc, brick]
    col = np.asarray(colour, float)[None, None, :] * shade[..., None]
    col[mortar] = style.mortar
    height = np.where(mortar, -0.35, 0.0)
    return col, height


SIGN_COLOURS = ((236, 236, 230), (196, 44, 38), (234, 192, 44), (36, 78, 150), (38, 122, 74), (28, 28, 30),
                (232, 120, 30), (210, 210, 214))   # Japanese signboards: white, red, yellow, blue, green, black, ...
CURTAIN_COLOURS = ((226, 220, 204), (204, 196, 180), (150, 140, 125), (180, 170, 150), (120, 128, 132))


def _occluded_rows(style: Style, gy0: int, gy1: int, floor: float, sh: float):
    """Rows [a, b) of a window's glass that are not seen as glass: the curtain wall's spandrel over the floor slab
    and the parapet of a Japanese balcony in front of the sliding doors. Shared by synth_facade and
    synth_window_lights, so the night mask never lights a spandrel or a parapet."""
    out = []
    if style.spandrel is not None:
        out.append((max(gy0, gy1 - int(round(0.85 / style.storey_m * sh))), gy1))
    if style.parapet is not None:
        out.append((max(gy0, int(round(floor - 1.1 / style.storey_m * sh))), gy1))
    return out


def _tile_layer(h, w, mw, mh, style: Style, colour, rng):
    """Ceramic tile cladding: like _brick_layer at tile scale, a smaller tone spread and glazed sheen."""
    col, height = _brick_layer(h, w, mw, mh, style, colour, rng, tone_sd=0.045)
    return col, height


def _concrete_ties(col, height, mw, mh, style: Style, rng):
    """Exposed concrete: formwork panel joints (1.82 x 0.91 m panels, fitted to the tile) and six form-tie holes
    per panel, a tone per panel."""
    h, w = height.shape
    nx, ny = max(1, int(round(mw / 1.82))), max(1, int(round(mh / 0.91)))
    ys, xs = np.mgrid[0:h, 0:w]
    px_, py_ = xs * nx / w, ys * ny / h
    pi, pj = np.floor(px_).astype(int), np.floor(py_).astype(int)
    tone = rng.normal(0.0, 0.035, size=(ny, nx))
    col *= (1.0 + tone[pj % ny, pi % nx])[..., None]
    fx, fy = px_ - pi, py_ - pj
    jx, jy = 0.004 * nx / mw, 0.004 * ny / mh
    joint = (fx < jx) | (fy < jy)
    col[joint] *= 0.82
    height[joint] = -0.25
    r = 0.013   # tie hole radius (m)
    for hx in (1 / 6, 1 / 2, 5 / 6):
        for hy in (1 / 4, 3 / 4):
            dx = (fx - hx) * mw / nx
            dy = (fy - hy) * mh / ny
            hole = dx * dx + dy * dy < r * r
            col[hole] *= 0.55
            height[hole] = -0.6
    return col, height


def _metal_panels(col, height, mw, mh, style: Style, rng):
    """Aluminium spandrel panels: vertical joints every half bay, a joint at every storey, a soft sheen."""
    h, w = height.shape
    nx = 2 * TEX_BAYS
    ys, xs = np.mgrid[0:h, 0:w]
    fx = (xs * nx / w) % 1.0
    fy = (ys * TEX_STOREYS / h) % 1.0
    joint = (fx < 0.008 * nx / mw) | (fy < 0.008 * TEX_STOREYS / mh)
    sheen = 1.0 + 0.05 * np.cos(2 * np.pi * fy)   # a soft vertical gradient on each panel
    col *= sheen[..., None]
    col[joint] *= 0.6
    height[joint] = -0.4
    return col, height


def synth_facade(style: Style, variant: int = 0, px: int = TEX_PX, seed: int = 0, windows: bool = True):
    """(base colour uint8 HxWx3, normal uint8 HxWx3, ORM uint8 HxWx4) for one tile of a style:
    TEX_BAYS x TEX_STOREYS modules, seamless in both directions, row 0 = top. ``windows=False``: the wall
    without its painted windows (floor bands stay), for walls whose windows are geometry (export/windows.py).

    Windows (2026-11-28, Shibuya facades had a ninth of the photos' edge detail): a frame with a shadow line and
    a bevel, the glass recessed behind it (a shadow along its top and left edges, darker AO), a brightness and
    roughness per pane, and behind the glass blinds, curtains, a lit ceiling or a dark room; curtain walls have
    spandrel glass over the floor slabs, Japanese apartment blocks balcony parapets, Japanese shop floors a
    signboard band."""
    rng = np.random.default_rng(_crc(style.name, variant, seed))
    h = w = px
    mw, mh = TEX_BAYS * style.bay_m, TEX_STOREYS * style.storey_m
    colour = style.colours[variant % len(style.colours)]
    if style.wall == "brick":
        col, height = _brick_layer(h, w, mw, mh, style, colour, rng)
    elif style.wall == "tile":
        col, height = _tile_layer(h, w, mw, mh, style, colour, rng)
    else:
        col = np.ones((h, w, 3)) * np.asarray(colour, float)
        height = np.zeros((h, w))
        if style.wall == "concrete" and style.ties:
            col, height = _concrete_ties(col, height, mw, mh, style, rng)
        elif style.wall == "concrete":   # panel joints at every bay and storey
            ys, xs = np.mgrid[0:h, 0:w]
            jx = (xs * TEX_BAYS / w) % 1.0
            jy = (ys * TEX_STOREYS / h) % 1.0
            seam = (jx < 0.012 / style.bay_m * 2) | (jy < 0.012 / style.storey_m * 2)
            col[seam] *= 0.7
            height[seam] = -0.4
        elif style.wall == "metal_panel":
            col, height = _metal_panels(col, height, mw, mh, style, rng)
        elif style.wall == "metal":    # trapezoidal sheet, ribs every 20 cm
            xs = np.arange(w)[None, :] * mw / w
            rib = (xs % 0.2) / 0.2
            prof = np.clip(np.abs(rib - 0.5) * 4 - 0.6, 0, 1)
            height = np.repeat(prof, h, axis=0) * 0.8
            col = col * (0.92 + 0.08 * np.repeat(prof, h, axis=0))[..., None]
    # dirt / tone: tileable low-frequency variation
    col *= (1.0 + 0.06 * _periodic_noise((h, w), 48, rng))[..., None]
    col *= (1.0 + 0.03 * _periodic_noise((h, w), 6, rng))[..., None]
    rough = np.full((h, w), {"brick": 0.88, "render": 0.88, "concrete": 0.86, "tile": 0.55,
                             "metal_panel": 0.42}.get(style.wall, 0.45))
    rough *= 1.0 + 0.06 * _periodic_noise((h, w), 24, rng)
    metal = np.full((h, w), 0.5 if style.wall == "metal" else 0.35 if style.wall == "metal_panel" else 0.0)
    ao = np.ones((h, w))
    if style.band is not None:   # floor edge / string course: a band just above every storey's floor
        bpx = max(2, int(round(style.band_m / style.storey_m * h / TEX_STOREYS)))
        for sy in range(TEX_STOREYS):
            floor = int(round((sy + 1) * h / TEX_STOREYS))
            col[floor - bpx:floor] = np.asarray(style.band, float) * rng.uniform(0.96, 1.04)
            height[floor - bpx:floor] = 0.25
            rough[floor - bpx:floor] = 0.85
            col[floor:min(h, floor + max(1, bpx // 4))] *= 0.85   # its shadow on the wall below
    if not style.blank and windows:
        bw, sh = w / TEX_BAYS, h / TEX_STOREYS
        fpx = _frame_px(style, bw)
        for sy in range(TEX_STOREYS):
            for bx in range(TEX_BAYS):
                floor = (sy + 1) * sh
                wx0, wx1, wy0, wy1, door = _window_rect(style, variant, bx, sy, bw, sh)
                if style.surround is not None:    # moulded architrave around the window, a little proud of the wall
                    spx = max(2, int(round(style.surround_m / style.bay_m * bw)))
                    sy0, sy1 = max(0, wy0 - spx), min(h, wy1 + (0 if style.balcony else spx))
                    col[sy0:sy1, max(0, wx0 - spx):min(w, wx1 + spx)] = np.asarray(style.surround, float) * \
                        rng.uniform(0.97, 1.03)
                    height[sy0:sy1, max(0, wx0 - spx):min(w, wx1 + spx)] = 0.35
                    rough[sy0:sy1, max(0, wx0 - spx):min(w, wx1 + spx)] = 0.8
                # frame (outer rectangle) with a shadow line where it meets the wall, then the glass inset
                col[wy0:wy1, wx0:wx1] = style.frame
                height[wy0:wy1, wx0:wx1] = -0.3
                rough[wy0:wy1, wx0:wx1] = 0.5
                metal[wy0:wy1, wx0:wx1] = 0.0
                line = max(1, fpx // 3)
                col[wy0:wy0 + line, wx0:wx1] *= 0.72
                col[wy0:wy1, wx0:wx0 + line] *= 0.8
                gy0, gy1, gx0, gx1 = wy0 + fpx, wy1 - fpx, wx0 + fpx, wx1 - fpx
                if gy1 <= gy0 or gx1 <= gx0:
                    continue
                col[gy1:gy1 + line, gx0:gx1] = np.minimum(col[gy1:gy1 + line, gx0:gx1] * 1.15, 255)  # bevel light
                gh = gy1 - gy0
                grad = np.linspace(1.22, 0.82, gh)[:, None, None]   # sky reflection: brighter at the top
                tint = float(np.clip(1.0 + style.glass_tint * rng.standard_normal(), 0.7, 1.4))
                hue = 1.0 + rng.uniform(-0.03, 0.03, 3)                  # a little colour per pane, no blue bias
                g = np.asarray(style.glass, float)[None, None, :] * grad * tint * hue[None, None, :]
                col[gy0:gy1, gx0:gx1] = g
                height[gy0:gy1, gx0:gx1] = -1.0
                rough[gy0:gy1, gx0:gx1] = rng.uniform(0.03, 0.12)
                ao[gy0:gy1, gx0:gx1] = 0.85
                # what is behind the glass
                u = rng.random()
                gw = gx1 - gx0
                if u < 0.22:      # blinds, from the top down
                    yb = gy0 + int(gh * rng.uniform(0.2, 0.9))
                    slat = np.asarray((198, 196, 190), float) * rng.uniform(0.85, 1.05)
                    col[gy0:yb, gx0:gx1] = slat
                    pitch = max(2, int(round(0.035 / style.storey_m * sh)))
                    col[gy0:yb:pitch, gx0:gx1] *= 0.72
                    rough[gy0:yb, gx0:gx1] = 0.7
                elif u < 0.42 and not style.shop:   # curtains drawn to both sides
                    cw = int(gw * rng.uniform(0.15, 0.35))
                    curtain = np.asarray(CURTAIN_COLOURS[int(rng.integers(len(CURTAIN_COLOURS)))], float)
                    folds = 1.0 - 0.12 * (np.sin(np.arange(cw) * 2 * np.pi / max(3, cw / 3)) > 0)
                    for c0 in (gx0, gx1 - cw):
                        col[gy0:gy1, c0:c0 + cw] = curtain[None, None, :] * folds[None, :, None]
                        rough[gy0:gy1, c0:c0 + cw] = 0.9
                elif u < 0.52:    # a lit room: the ceiling's lights (fading downwards) and a paler back wall
                    band = max(2, gh // 8)
                    fade = np.linspace(1.6, 1.15, band)[:, None, None]
                    col[gy0 + band:gy0 + 2 * band, gx0:gx1] = np.minimum(
                        col[gy0 + band:gy0 + 2 * band, gx0:gx1] * fade + 18, 255)
                    col[gy0 + 2 * band:gy0 + gh // 2, gx0:gx1] *= 1.12
                # the frame's shadow on the recessed glass, along its top and left edges
                rec = max(1, int(round(0.05 / style.storey_m * sh)))
                col[gy0:gy0 + rec, gx0:gx1] *= 0.62
                col[gy0:gy1, gx0:gx0 + max(1, rec // 2)] *= 0.75
                ao[gy0:gy0 + rec, gx0:gx1] = 0.62
                bar = max(1, fpx // 2)
                for m in range(1, style.mullions + 1):
                    xm = gx0 + (gx1 - gx0) * m // (style.mullions + 1)
                    col[gy0:gy1, xm - bar:xm + bar] = style.frame
                    col[gy0:gy1, xm + bar - 1:xm + bar] *= 0.75
                    height[gy0:gy1, xm - bar:xm + bar] = -0.3
                if style.transom:
                    yt = gy0 + gh // 3
                    col[yt - bar:yt + bar, gx0:gx1] = style.frame
                    col[yt + bar - 1:yt + bar, gx0:gx1] *= 0.75
                    height[yt - bar:yt + bar, gx0:gx1] = -0.3
                for a, b in _occluded_rows(style, gy0, gy1, floor, sh):
                    if style.spandrel is not None and style.parapet is None:   # opaque glass over the slab
                        col[a:b, gx0:gx1] = np.asarray(style.spandrel, float) * rng.uniform(0.95, 1.05)
                        rough[a:b, gx0:gx1] = 0.2
                        ao[a:b, gx0:gx1] = 1.0
                        col[a:a + bar, gx0:gx1] = style.frame
                if style.sill and not door:
                    s0 = wy1
                    s1 = min(int(floor), wy1 + max(2, int(0.07 / style.storey_m * sh)))
                    col[s0:s1, wx0 - fpx:wx1 + fpx] = (200, 198, 190)
                    height[s0:s1, wx0 - fpx:wx1 + fpx] = 0.3
                    d1 = min(int(floor), s1 + max(1, (s1 - s0) // 2))
                    col[s1:d1, wx0 - fpx:wx1 + fpx] *= 0.82   # its drip shadow
                if style.lintel:
                    l1 = wy0
                    l0 = max(0, wy0 - int(0.22 / style.storey_m * sh))
                    col[l0:l1, wx0:wx1] *= 0.8
                if style.parapet is not None and not door:   # a balcony across the bay: parapet panel on a slab
                    floor_px = int(round(floor))
                    xa, xb = int(round(bx * bw)), int(round((bx + 1) * bw))
                    p0 = max(0, int(round(floor - 1.1 / style.storey_m * sh)))
                    slab = max(2, int(round(0.16 / style.storey_m * sh)))
                    frosted = variant % 2 == 1
                    pc = np.asarray((202, 206, 202) if frosted else style.parapet, float) * rng.uniform(0.96, 1.03)
                    col[p0:floor_px, xa:xb] = pc
                    rough[p0:floor_px, xa:xb] = 0.3 if frosted else 0.85
                    ao[p0:floor_px, xa:xb] = 1.0
                    metal[p0:floor_px, xa:xb] = 0.0
                    height[p0:floor_px, xa:xb] = 0.6
                    rail = max(1, int(round(0.03 / style.storey_m * sh)))
                    col[p0:p0 + rail, xa:xb] = (170, 172, 172)                       # aluminium handrail
                    metal[p0:p0 + rail, xa:xb] = 0.6
                    col[p0 + rail:p0 + 2 * rail, xa:xb] *= 0.8
                    col[floor_px - slab:floor_px, xa:xb] = np.asarray((214, 212, 206), float)   # slab edge
                    col[floor_px - slab:floor_px - slab + 1, xa:xb] *= 0.8
                    col[p0:floor_px, xa:xa + max(1, rail // 2)] *= 0.78              # panel joint at the bay edge
                if style.balcony and not door:   # wrought-iron railing on a stone slab across the floor-length window
                    floor_px = int(round(floor))
                    slab = max(2, int(0.12 / style.storey_m * sh))
                    rail_h = int(0.95 / style.storey_m * sh)
                    over = int(0.16 / style.bay_m * bw)
                    xa, xb = max(0, wx0 - over), min(w, wx1 + over)
                    s0 = max(0, floor_px - slab)
                    col[s0:floor_px, xa:xb] = np.asarray(style.surround or style.band or (200, 190, 170), float)
                    height[s0:floor_px, xa:xb] = 0.6
                    r0 = max(0, s0 - rail_h)
                    iron = np.asarray((38, 36, 34), float)
                    bar = max(1, int(0.018 / style.bay_m * bw))
                    col[r0:r0 + 2 * bar, xa:xb] = iron                       # top rail
                    col[s0 - 2 * bar:s0, xa:xb] = iron                       # bottom rail
                    step = max(3 * bar, int(0.11 / style.bay_m * bw))
                    for xx in range(xa, xb, step):                           # balusters
                        col[r0:s0, xx:xx + bar] = iron
                    for yy in (r0 + rail_h // 3, r0 + 2 * rail_h // 3):     # a little scrollwork: two thin bars
                        col[yy:yy + bar, xa:xb] = iron
                    rough[r0:s0, xa:xb] = np.where((col[r0:s0, xa:xb] == iron).all(-1), 0.55, rough[r0:s0, xa:xb])
                    metal[r0:s0, xa:xb] = np.where((col[r0:s0, xa:xb] == iron).all(-1), 0.6, metal[r0:s0, xa:xb])
                    height[r0:s0, xa:xb] = np.where((col[r0:s0, xa:xb] == iron).all(-1), 0.5, height[r0:s0, xa:xb])
                if style.shop:   # sign band above the shop window
                    xa, xb = int(bx * bw), int((bx + 1) * bw)
                    if style.sign_band:   # Japanese signboard: a coloured panel per bay, a light frame, its shadow
                        b0 = max(0, int(floor - 0.97 * sh))
                        b1 = max(b0 + 1, int(floor - 0.74 * sh))
                        pc = SIGN_COLOURS[int(rng.integers(len(SIGN_COLOURS)))]
                        col[b0:b1, xa:xb] = pc
                        edge = max(1, int(round(0.03 / style.storey_m * sh)))
                        col[b0:b0 + edge, xa:xb] = (226, 226, 222)
                        col[b1 - edge:b1, xa:xb] = (226, 226, 222)
                        col[b0:b1, xa:xa + edge] = (226, 226, 222)
                        col[b1:b1 + edge, xa:xb] *= 0.7
                        height[b0:b1, xa:xb] = 0.5
                        rough[b0:b1, xa:xb] = 0.3
                    else:
                        b0 = max(0, int(floor - 0.97 * sh))
                        b1 = max(b0 + 1, int(floor - 0.80 * sh))
                        col[b0:b1, xa:xb] = (30, 32, 36) if variant == 0 else (120, 30, 28)
                        rough[b0:b1, xa:xb] = 0.4
    albedo = np.clip(col, 0, 255).astype(np.uint8)
    normal = height_to_normal(height, strength=3.0)
    orm = np.stack([ao, np.clip(rough, 0.0, 1.0), metal, np.zeros_like(ao)], axis=-1)
    orm = np.clip(orm * 255.0, 0, 255).astype(np.uint8)
    return albedo, normal, orm


def _frame_px(style: Style, bay_px: float) -> int:
    return max(2, int(round(style.frame_m / style.bay_m * bay_px)))


def _window_rect(style: Style, variant: int, bx: int, sy: int, bw: float, sh: float):
    """Window (outer frame) of bay ``bx`` in storey row ``sy`` (row 0 = top) of a grid of bays
    ``bw`` px wide and storeys ``sh`` px high: (x0, x1, y0, y1, door). The door of a shopfront sits
    in bay (sy + variant) % TEX_BAYS of each facade tile, so a larger grid repeats it per tile."""
    x0, x1, y0, y1 = style.window
    floor = (sy + 1) * sh   # storey floor at the bottom of its band (row grows downwards)
    wx0, wx1 = int(bx * bw + x0 * bw), int(bx * bw + x1 * bw)
    wy0, wy1 = int(floor - y1 * sh), int(floor - y0 * sh)
    door = style.shop and bx % TEX_BAYS == (sy % TEX_STOREYS + variant) % TEX_BAYS
    if door:
        wy1 = int(floor - 0.01 * sh)
    return wx0, wx1, wy0, wy1, door


# --------------------------------------------------------------------------- lit windows at night

MASK_BAYS = 8        # one night mask covers 8 bays ...
MASK_STOREYS = 8     # ... by 8 storeys = 4 x 4 facade texture tiles
MASK_PX_PER_BAY = 128
MASK_UV_SCALE = TEX_BAYS / MASK_BAYS                                  # the material samples it at UV x this
MASK_OFFSETS = (MASK_BAYS // TEX_BAYS, MASK_STOREYS // TEX_STOREYS)   # whole facade tiles per mask
LIGHT_COOL_SHARE = 0.12   # lit windows with cool white light (LED, screens) instead of warm

# facade style -> building use whose lit share the material gets (ue/twin_materials.py WINDOW_LIT_SHARE).
# The styles come from the BAG use of the units (classify): homes and mixed blocks, offices (also
# schools, hospitals, hotels), the shopfront ground floor, industry; plain walls have no windows.
USE_OF_STYLE = {"stone_prewar": "home", "stone_old": "home", "render_warm": "home", "brick_old": "home", "brick_postwar": "home", "brick_modern": "home", "render_modern": "home",
                "brick_wederopbouw": "home", "brick_late": "home", "concrete_panel": "home",
                "brick_contemporary": "home", "office_concrete": "office", "office_glass": "office",
                "office_brick": "office", "shopfront": "shop", "industrial": "industry", "stone_balcony": "home",
                "stucco_old": "home", "render_central": "home", "jp_tile": "home", "jp_mansion": "home",
                "jp_office": "office", "jp_concrete": "office", "shopfront_jp": "shop"}


def light_offset(building_id: str, wall: int) -> tuple[int, int]:
    """Whole facade tiles a wall's UVs are shifted by (wall_uv tile_offset), so buildings and their
    walls show different parts of the night mask: u per wall, v per building (the storeys of one
    building keep their order around its corners)."""
    return _crc("lights-u", building_id, wall) % MASK_OFFSETS[0], _crc("lights-v", building_id) % MASK_OFFSETS[1]


def synth_window_lights(style: Style, variant: int = 0, px_bay: int = MASK_PX_PER_BAY, seed: int = 0):
    """Night mask of a style (uint8 HxWx3, MASK_BAYS x MASK_STOREYS windows, row 0 = top, linear), or
    None for a style without windows. Per window (one cell of bay x storey):

    * R: the glass of the window (same rectangle, frame and glazing bars as synth_facade) times its
      brightness (0.6-1.0, a little brighter towards the ceiling); 0 on the wall;
    * G: the window's rank in (0, 1), uniform over the cell: the material lights the window when
      rank < lit share x NightFactor, so a share of 0.4 lights about 40 % of the windows and a
      lower share (later at night, less dark) lights a subset of them;
    * B: its tint, 0 warm (incandescent / warm LED) to 1 cool white (LIGHT_COOL_SHARE of them).

    Seamless in both directions; the facade texture repeats MASK_OFFSETS times inside it."""
    if style.blank:
        return None
    rng = np.random.default_rng(_crc("lights", style.name, variant, seed))
    bw = sh = px_bay
    h, w = MASK_STOREYS * sh, MASK_BAYS * bw
    r = np.zeros((h, w))
    g = np.zeros((h, w))
    b = np.zeros((h, w))
    fpx = _frame_px(style, bw)
    bar = max(1, fpx // 2)
    n = MASK_BAYS * MASK_STOREYS
    # ranks: a shuffled even spread, so any share lights very nearly that fraction of the windows
    ranks = (rng.permutation(n) + rng.uniform(0.15, 0.85, n)) / n
    for k in range(n):
        sy, bx = divmod(k, MASK_BAYS)
        cy0, cx0 = sy * sh, bx * bw
        g[cy0:cy0 + sh, cx0:cx0 + bw] = ranks[k]
        cool = rng.random() < LIGHT_COOL_SHARE
        b[cy0:cy0 + sh, cx0:cx0 + bw] = rng.uniform(0.6, 1.0) if cool else rng.uniform(0.0, 0.25)
        bright = rng.uniform(0.6, 1.0)
        wx0, wx1, wy0, wy1, door = _window_rect(style, variant, bx, sy, bw, sh)
        gy0, gy1, gx0, gx1 = wy0 + fpx, wy1 - fpx, wx0 + fpx, wx1 - fpx
        if gy1 <= gy0 or gx1 <= gx0:
            continue
        r[gy0:gy1, gx0:gx1] = bright * np.linspace(1.0, 0.8, gy1 - gy0)[:, None]
        for m in range(1, style.mullions + 1):
            xm = gx0 + (gx1 - gx0) * m // (style.mullions + 1)
            r[gy0:gy1, xm - bar:xm + bar] = 0.0
        if style.transom:
            yt = gy0 + (gy1 - gy0) // 3
            r[yt - bar:yt + bar, gx0:gx1] = 0.0
        for a, b_ in _occluded_rows(style, gy0, gy1, (sy + 1) * sh, sh):   # spandrel, balcony parapet
            r[a:b_, gx0:gx1] = 0.0
    out = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def lit_fraction(mask: np.ndarray, share: float) -> float:
    """Fraction of the windows (cells) of a night mask lit at a lit share (what the material does at
    NightFactor 1). For tests and reports."""
    ch, cw = mask.shape[0] // MASK_STOREYS, mask.shape[1] // MASK_BAYS
    cells = mask[ch // 2::ch, cw // 2::cw, 1] / 255.0
    return float((cells < share).mean())


def height_to_normal(height: np.ndarray, strength: float = 3.0) -> np.ndarray:
    """Tangent-space normal map (DirectX green, as UE expects) from a height map, wrapped."""
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5 * strength
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5 * strength
    n = np.stack([-dx, dy, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip(np.round((n * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)


def flat_normal(px: int = 4) -> np.ndarray:
    return np.full((px, px, 3), (128, 128, 255), dtype=np.uint8)


def constant_orm(roughness: float, px: int = 4) -> np.ndarray:
    o = np.zeros((px, px, 4), dtype=np.uint8)
    o[..., 0] = 255
    o[..., 1] = int(round(roughness * 255))
    return o


# --------------------------------------------------------------------------- aerial photo

def ortho_tile(img: np.ndarray, transform, tile: tuple[int, int], tile_m: float, px: int) -> Optional[np.ndarray]:
    """The tile's square of a north-up ortho (HxWx3, affine ``transform`` = (a, b, c, d, e, f) as in
    rasterio: x = a*col + c, y = e*row + f), resampled to px x px (nearest of a box average).
    None when the tile is outside the photo."""
    a, _, c, _, e, f = transform[:6]
    x0, x1 = tile[0] * tile_m, (tile[0] + 1) * tile_m
    y0, y1 = tile[1] * tile_m, (tile[1] + 1) * tile_m
    h, w = img.shape[:2]
    # pixel centres of the output grid
    xs = x0 + (np.arange(px) + 0.5) * (x1 - x0) / px
    ys = y1 - (np.arange(px) + 0.5) * (y1 - y0) / px
    cols = np.floor((xs - c) / a).astype(int)
    rows = np.floor((ys - f) / e).astype(int)
    inside_c = (cols >= 0) & (cols < w)
    inside_r = (rows >= 0) & (rows < h)
    if not inside_c.any() or not inside_r.any():
        return None
    out = np.zeros((px, px, 3), dtype=np.uint8)
    mean = img.reshape(-1, img.shape[2])[:, :3].mean(axis=0).astype(np.uint8)
    out[:] = mean
    rr, cc = np.meshgrid(np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1), indexing="ij")
    ok = inside_r[:, None] & inside_c[None, :]
    out[ok] = img[rr[ok], cc[ok], :3]
    return out


def mean_colour(img: np.ndarray, transform, ring_xy: np.ndarray) -> Optional[tuple]:
    """Average ortho colour inside a footprint ring (model metres), or None."""
    a, _, c, _, e, f = transform[:6]
    h, w = img.shape[:2]
    cols = (ring_xy[:, 0] - c) / a
    rows = (ring_xy[:, 1] - f) / e
    r0, r1 = int(max(0, math.floor(rows.min()))), int(min(h, math.ceil(rows.max()) + 1))
    c0, c1 = int(max(0, math.floor(cols.min()))), int(min(w, math.ceil(cols.max()) + 1))
    if r1 <= r0 or c1 <= c0:
        return None
    rr, cc = np.mgrid[r0:r1, c0:c1]
    inside = _points_in_ring(cc.ravel() + 0.5, rr.ravel() + 0.5, cols, rows)
    if not inside.any():
        return None
    px = img[r0:r1, c0:c1, :3].reshape(-1, 3)[inside]
    return tuple(int(v) for v in np.median(px, axis=0))


def _points_in_ring(x, y, rx, ry) -> np.ndarray:
    inside = np.zeros(len(x), dtype=bool)
    n = len(rx)
    j = n - 1
    for i in range(n):
        cond = ((ry[i] > y) != (ry[j] > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (rx[j] - rx[i]) * (y - ry[i]) / (ry[j] - ry[i]) + rx[i]
        inside ^= cond & (x < xint)
        j = i
    return inside


def save_png(path, arr: np.ndarray) -> None:
    from PIL import Image
    mode = {3: "RGB", 4: "RGBA"}[arr.shape[2]]
    Image.fromarray(arr, mode).save(str(path), optimize=True)
