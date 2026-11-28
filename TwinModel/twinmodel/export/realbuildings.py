"""Swap a twin's extruded OSM buildings for real 3D buildings from an open city or national model.

The providers (twinmodel.ingest.buildings: 3DBAG for the Netherlands, the NYC 3D Building Model for New
York City, PLATEAU for Japan) turn their data into ``RealBuilding`` records in the twin's model frame:
surfaces (rings of 3D points with a semantic kind: RoofSurface, WallSurface, ...), the footprint, the
ground and eaves heights, the construction year and the uses of the floor area. ``apply`` does the rest,
whatever the source:

* every building gets a facade style (twinmodel.export.facades) from its year, uses, height and size:
  walls are written per style with UVs that fit whole bays and storeys, party walls get a plain wall,
  buildings with shops a shopfront ground floor; the synthetic facade textures go to ue/textures/;
* roofs are textured with the twin's aerial photo, one texture per tile (or a plain roof material for
  buildings taller than ``roof_photo_max_m``: a city's aerial photo is not a true ortho, tall buildings
  lean in it);
* one glb per (tile, material) is written in the bake export's format and swapped in for the manifest's
  ``building`` assets (the OSM ones are kept in meshes/osm_buildings/); the manifest's ``buildings``
  become the real footprints with ground, eaves and roof heights (the OSM ones move to ``buildings_osm``),
  so ``bake_level.py --buildings procedural`` builds on them.
"""
from __future__ import annotations

import gzip
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# credits of the twin's ortho providers (report.json build.imagery.source), for the roof textures
ORTHO_CREDITS = {
    "pdok_luchtfoto": "PDOK Luchtfoto (CC BY 4.0)",
    "nys_orthos": "NYS ITS orthoimagery (public domain)",
    "naip": "USGS NAIP (public domain)",
    "icgc": "ICGC orthophoto (CC BY 4.0)",
    "ign_es": "IGN PNOA (CC BY 4.0)",
    "osm_tiles": "OpenStreetMap tiles (ODbL)",
    "ign_fr": "IGN BD ORTHO (Licence Ouverte 2.0)",
    "swissimage": "swisstopo SWISSIMAGE (swisstopo open data)",
    "flanders": "Digitaal Vlaanderen orthophoto (Modellicentie Gratis Hergebruik)",
    "gsi_photo": "GSI Japan seamless photo (GSI terms, CC BY 4.0 compatible)",
    "helsinki_ortho": "City of Helsinki orthophoto (CC BY 4.0)",
    "mml_ortho": "National Land Survey of Finland orthophoto (CC BY 4.0)",
}


@dataclass
class RealBuilding:
    """One building of a real 3D model, in the twin's model frame (metres, z = the source's heights,
    which match the twin's DEM: NAP with AHN, NAVD88 with USGS 3DEP)."""
    id: str
    surfaces: list                     # [(rings: [np.ndarray (n, 3)], kind: str | None)]
    poly: object = None                # shapely Polygon footprint (model xy) or None
    ground: Optional[float] = None     # ground height at the building
    height: float = 0.0                # ground to eaves (m)
    roof_z: Optional[float] = None     # highest roof point
    year: Optional[int] = None
    uses: dict = field(default_factory=dict)   # use -> floor area m2 (facades.classify vocabulary)
    levels: int = 0
    roof_type: Optional[str] = None
    parts: int = 1
    centre: Optional[tuple] = None     # (x, y) that picks the tile; default the mean of the surface points
    region: str = "nl"                 # facades.BuildingInfo.region
    # the source's own photo textures: surface index -> (image file, [(n, 2) texture coordinates per ring],
    # CityGML convention: origin bottom left); apply packs them into one atlas per tile
    textures: dict = field(default_factory=dict)


@dataclass
class Source:
    """What a provider tells apply about its data."""
    name: str                          # provider name ("3dbag", "nyc3d", "plateau")
    label: str                         # stats building_source, e.g. "3DBAG 2.2 via api.3dbag.nl"
    asset_source: str                  # per asset, with the licence: "3DBAG LoD 2.2 (CC BY 4.0)"
    roof_photo_max_m: Optional[float] = None   # taller buildings get the plain roof (None: all photo)
    skirt: bool = False                # extend the walls down to the twin's terrain where it is lower (skirt())
    extra_stats: dict = field(default_factory=dict)


SKIP_SURFACES = {"GroundSurface", "FloorSurface", "ClosureSurface"}


# --------------------------------------------------------------------------- geometry

def newell(pts):
    n = np.zeros(3)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else None


def triangulate(rings):
    """3D polygon (outer ring + holes, each an (n, 3) array) -> (vertices, triangles, normal)."""
    import mapbox_earcut as earcut
    normal = newell(rings[0])
    if normal is None:
        return None
    drop = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != drop]
    verts = np.vstack(rings)
    flat = verts[:, keep]
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = np.asarray(earcut.triangulate_float64(flat, ends), dtype=np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return None
    # keep the winding of the outer ring (counter-clockwise seen from outside)
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    tn = np.cross(b - a, c - a)
    flip = (tn @ normal) < 0
    tri[flip] = tri[flip][:, ::-1]
    return verts, tri, normal


def roof_heights(surfaces, ground: float):
    """(eaves height above ground, highest roof point) from the roof surfaces: the eaves are the
    area-weighted 70th percentile of the roof heights, like 3DBAG's b3_h_dak_70p."""
    zs, ws, top = [], [], None
    for rings, kind in surfaces:
        if kind != "RoofSurface" or not rings:
            continue
        r = rings[0]
        nrm = np.zeros(3)
        for a, b in zip(r, np.roll(r, -1, axis=0)):
            nrm += np.cross(a, b)
        area = 0.5 * float(np.linalg.norm(nrm))
        zs.append(float(r[:, 2].mean()))
        ws.append(max(area, 1e-6))
        m = float(r[:, 2].max())
        top = m if top is None else max(top, m)
    if not zs:
        return 0.0, None
    order = np.argsort(zs)
    cw = np.cumsum(np.asarray(ws)[order])
    z70 = float(np.asarray(zs)[order][np.searchsorted(cw, 0.7 * cw[-1])])
    return max(0.0, z70 - ground), top


# --------------------------------------------------------------------------- caches and fetches

def cache_json(path: Path, fetch_fn: Callable):
    """JSON (gzip) cache beside the build: fetched once, reused by every rerun."""
    if path.exists():
        with gzip.open(path, "rt") as f:
            return json.load(f)
    data = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(data, f)
    tmp.replace(path)
    return data


def get_json(url, params, tries=4, user_agent="twinmodel (carla-digitaltwins)"):
    """GET with retries (open-data APIs answer the odd page with a 500)."""
    import time
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120, headers={"User-Agent": user_agent})
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))


def find_ortho(build_dir: Path, bbox_wgs84) -> Optional[Path]:
    """The twin pipeline's cached ortho for this bbox (data/ortho_<s>_<w>_<n>_<e>_<res>m_*.tif)."""
    s, w, n, e = bbox_wgs84
    stem = "ortho_%.5f_%.5f_%.5f_%.5f_" % (s, w, n, e)
    for d in [build_dir, *build_dir.resolve().parents][:4]:
        hits = sorted((d / "data").glob(stem + "*.tif")) if (d / "data").is_dir() else []
        if hits:
            # the finest resolution first
            return sorted(hits, key=lambda p: float(p.name[len(stem):].split("m_")[0]))[0]
    return None


PHOTO_SOURCES_EXCLUDED = ("osm_tiles",)    # a rendered map, not a photo: never the ground or a roof


def fetch_twin_ortho(build_dir: Path, man: dict):
    """(path, provider) of the twin's aerial photo fetched again when its build's cache is not at hand (an export
    copied to another machine): the providers of the twin's country that cover it (twinmodel.regions: e.g. NYS
    orthos then NAIP in the US, Helsinki's city photo then MML's in Finland), else of its street profile, into
    <build_dir>/data, where find_ortho looks. (None, None) when none answers."""
    from ..frame import LocalFrame
    from .. import profiles
    from ..ingest.imagery import fetch_ortho
    from .. import regions
    bbox = tuple(man["bbox_wgs84"])
    iso2 = twin_iso2(Path(build_dir), man)
    if iso2:            # the country's providers that cover the twin (twinmodel.regions), as the build chose them
        sources = list(regions.plan_for(iso2, bbox).ortho)
    else:               # unknown country: the street profile's providers
        name = (man.get("profile") or {}).get("name") if isinstance(man.get("profile"), dict) else man.get("profile")
        prof = profiles.by_name(name) if name in profiles.PROFILES else profiles.get()
        sources = list(prof.sources.ortho)
    sources = [x for x in sources if x not in PHOTO_SOURCES_EXCLUDED]
    if not sources:
        return None, None
    o = man["origin"]
    img = fetch_ortho(LocalFrame(o["lat"], o["lon"]), bbox, cache_dir=Path(build_dir) / "data", sources=sources)
    if img is None:
        return None, None
    path = find_ortho(Path(build_dir), bbox)
    if path is not None and img.source:   # the provider beside the photo: a later run that finds it cached credits it
        path.with_suffix(".source").write_text(str(img.source))
    return path, img.source


def ortho_provider(path) -> Optional[str]:
    """The provider of a cached photo fetched by fetch_twin_ortho (its .source file), or None."""
    try:
        return Path(path).with_suffix(".source").read_text().strip() or None
    except (OSError, TypeError):
        return None


def twin_iso2(build_dir: Path, man: dict):
    """ISO alpha-2 of a twin: the manifest's or report.json's profile record, else the country under its bbox
    (OpenStreetMap, cached in <build_dir>/data), else None."""
    import json
    for rec in (man.get("profile"), man.get("country")):
        if isinstance(rec, dict) and rec.get("iso2"):
            return rec["iso2"]
        if isinstance(rec, str) and len(rec) == 2:
            return rec.upper()
    try:
        rep = json.loads((Path(build_dir) / "report.json").read_text())
        iso = (rep.get("profile") or (rep.get("build") or {}).get("profile") or {}).get("iso2")
        if iso:
            return iso
    except Exception:  # noqa: BLE001 - an export copied away from its build has no report
        pass
    try:
        from ..ingest.osm import country_for_bbox
        return country_for_bbox(tuple(man["bbox_wgs84"]), cache_dir=Path(build_dir) / "data")
    except Exception:  # noqa: BLE001 - offline: the profile's providers
        return None


def load_ortho(path: Path, proj: str):
    """(image HxWx3, transform) of an ortho GeoTIFF in the twin's model frame, or None when the
    file is in another frame (the twin origin must land on model (0, 0))."""
    import rasterio
    from pyproj import CRS, Transformer
    with rasterio.open(path) as src:
        img = np.moveaxis(src.read()[:3], 0, -1)
        tf = src.transform
        crs = CRS.from_wkt(src.crs.to_wkt())
    lat0 = float(proj.split("+lat_0=")[1].split()[0])
    lon0 = float(proj.split("+lon_0=")[1].split()[0])
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon0, lat0)
    if abs(x) > 1.0 or abs(y) > 1.0:
        print(f"ortho {path.name}: not in the model frame (origin at {x:.1f}, {y:.1f}); roofs untextured")
        return None
    return img, (tf.a, tf.b, tf.c, tf.d, tf.e, tf.f)


def find_dem(build_dir: Path, bbox_wgs84) -> Optional[Path]:
    """The twin pipeline's cached elevation grid for this bbox (data/dem_<s>_<w>_<n>_<e>_<res>m.npz)."""
    s, w, n, e = bbox_wgs84
    stem = "dem_%.5f_%.5f_%.5f_%.5f_" % (s, w, n, e)
    for d in [build_dir, *build_dir.resolve().parents][:4]:
        hits = sorted((d / "data").glob(stem + "*.npz")) if (d / "data").is_dir() else []
        if hits:
            return sorted(hits, key=lambda p: float(p.name[len(stem):].split("m")[0]))[0]
    return None


DEM_EXPORT_RES_M = 2.0


def keep_bases(buildings: list, man: dict, tol_m: float = 0.01) -> int:
    """Rebuilding an export's buildings without the build's elevation grid (restyling an export copied away from
    its build): each building the export already had (same provider id) is moved back onto the base height it had
    there, so it stands where it stood; the export's own surfaces only place the new ones (dem_from_export, whose
    20 m terrain is off by more than 3 m for 7 % of ZurichHB's buildings). Returns the buildings moved."""
    old = {b["id"]: b["base_z_cm"] / 100.0 for b in man.get("buildings", ())
           if b.get("real") and b.get("base_z_cm") is not None}
    n = 0
    for b in buildings:
        z0 = old.get(b.id)
        if z0 is None or b.ground is None or abs(z0 - b.ground) < tol_m:
            continue
        dz = z0 - b.ground
        b.surfaces = [([np.asarray(r, float) + [0.0, 0.0, dz] for r in rings], kind) for rings, kind in b.surfaces]
        b.ground = z0
        if b.roof_z is not None:
            b.roof_z += dz
        n += 1
    return n


def dem_from_export(build_dir: Path, man: dict, res_m: float = DEM_EXPORT_RES_M) -> Optional[Path]:
    """An elevation grid from the export's own road, pavement and ground surfaces, written where find_dem looks
    (<build_dir>/data/dem_<bbox>_<res>m.npz), for an export copied away from its build (whose cached grid stayed
    behind: SalamancaPlaza, HelsinkiSenate, ZurichHB, 2026-11-25): without it the buildings would stand at height
    0. Each cell takes the top surface over it; cells under buildings take the nearest surface. None when the
    export has no surfaces."""
    import sys as _sys
    from scipy import ndimage
    tools = Path(__file__).resolve().parents[2] / "tools"
    if str(tools) not in _sys.path:
        _sys.path.insert(0, str(tools))
    import surface_gaps as sg
    from .ue import GROUND_PLANE_DROP, gltf_to_model, read_glb
    ue = Path(build_dir) / "ue"
    try:
        tris, _kind, _layer, _src = sg._load(ue, man, fills=True)
    except ValueError:      # no road, pavement or ground surface at all
        tris = np.zeros((0, 3, 3))
    # the ground plane follows the terrain (GROUND_PLANE_DROP under it) over the whole twin and past its surfaces:
    # the bottom layer, under every road, pavement and ground tile
    plane = []
    for a in man["assets"]:
        if a["kind"] == "groundplane":
            g = read_glb(ue / a["file"])
            if len(g["faces"]):
                plane.append(gltf_to_model(g["positions"].astype(float))[g["faces"].astype(np.int64)])
    plane = np.concatenate(plane) + [0.0, 0.0, GROUND_PLANE_DROP] if plane else np.zeros((0, 3, 3))
    if not len(tris) and not len(plane):
        return None
    tris = np.concatenate([plane, tris]) if len(plane) else tris
    grid = sg.Grid(tris)
    np_ = len(plane)
    order = np.concatenate([np.arange(np_), np_ + np.argsort(tris[np_:, :, 2].mean(axis=1), kind="stable")])
    rid = grid.raster_ids(tris[order], order)                          # the surfaces over the plane, the top wins
    # each triangle's plane z = ax + by + c (steep ones: their top), clamped to its own heights, for every cell
    a0, b0, c0 = tris[:, 0], tris[:, 1], tris[:, 2]
    nrm = np.cross(b0 - a0, c0 - a0)
    flat = np.abs(nrm[:, 2]) >= sg.STEEP_NZ * np.linalg.norm(nrm, axis=1)
    nz = np.where(flat, nrm[:, 2], 1.0)
    pa, pb = np.where(flat, -nrm[:, 0] / nz, 0.0), np.where(flat, -nrm[:, 1] / nz, 0.0)
    pc = np.where(flat, a0[:, 2] - pa * a0[:, 0] - pb * a0[:, 1], tris[:, :, 2].max(axis=1))
    lo, hi = tris[:, :, 2].min(axis=1), tris[:, :, 2].max(axis=1)
    z = np.full(rid.shape, np.nan)
    rr, cc = np.nonzero(rid > 0)
    t = rid[rr, cc] - 1
    xs = grid.x0 + (cc + 0.5) * sg.RES
    ys = grid.y0 + (rr + 0.5) * sg.RES
    z[rr, cc] = np.clip(pa[t] * xs + pb[t] * ys + pc[t], lo[t], hi[t])
    if not np.isfinite(z).any():
        return None
    _, (ir, ic) = ndimage.distance_transform_edt(~np.isfinite(z), return_indices=True)
    z = z[ir, ic]
    k = max(1, int(round(res_m / sg.RES)))
    zz = z[: z.shape[0] // k * k, : z.shape[1] // k * k].reshape(z.shape[0] // k, k, z.shape[1] // k, k).min(axis=(1, 3))
    s, w, n, e = man["bbox_wgs84"]
    out = Path(build_dir) / "data" / ("dem_%.5f_%.5f_%.5f_%.5f_%gm.npz" % (s, w, n, e, res_m))
    out.parent.mkdir(parents=True, exist_ok=True)
    x0, y0 = grid.x0 + 0.5 * sg.RES, grid.y0 + 0.5 * sg.RES
    np.savez_compressed(out, z=zz.astype(np.float32), x0=x0 + (k - 1) * sg.RES / 2, y0=y0 + (k - 1) * sg.RES / 2,
                        dx=k * sg.RES, dy=k * sg.RES, source="export surfaces")
    return out


def skirt(buildings: list, dem_path: Path, below_m: float = 0.3, step_m: float = 1.0) -> int:
    """Extend each building's walls down to the twin's terrain where it lies below the building's ground.

    A city model stands every building on one ground height (the NYC model: a flat GroundSurface), while the
    twin's terrain (USGS 3DEP, AHN, ...) follows the street: on a slope the building would float above the
    low side. The wall vertices at the base move down to ``below_m`` under the lowest terrain along the
    footprint, so the walls reach into the ground; where the terrain is higher they stay hidden under it.
    Returns the number of buildings extended."""
    d = np.load(dem_path, allow_pickle=True)
    z, x0, y0, dx, dy = d["z"], float(d["x0"]), float(d["y0"]), float(d["dx"]), float(d["dy"])
    n = 0
    for b in buildings:
        if b.poly is None or b.poly.is_empty or b.ground is None:
            continue
        ring = b.poly.exterior
        pts = np.asarray([ring.interpolate(t).coords[0] for t in np.arange(0.0, ring.length, step_m)])
        if not len(pts):
            continue
        c = np.clip(np.round((pts[:, 0] - x0) / dx).astype(int), 0, z.shape[1] - 1)
        r = np.clip(np.round((pts[:, 1] - y0) / dy).astype(int), 0, z.shape[0] - 1)
        low = float(np.nanmin(z[r, c]))
        if not np.isfinite(low) or low >= b.ground - 0.05:
            continue
        base = low - below_m
        for rings, kind in b.surfaces:
            if kind not in ("WallSurface", None):
                continue
            if kind is None:   # untyped faces: walls only
                nrm = newell(rings[0])
                if nrm is None or abs(nrm[2]) > 0.3:
                    continue
            for rr in rings:
                rr[rr[:, 2] <= b.ground + 0.05, 2] = base
        n += 1
    return n


def ortho_credit(build_dir: Path) -> str:
    """Credit of the ortho the twin was built with (its report.json), for the roof textures."""
    try:
        src = json.loads((build_dir / "report.json").read_text())["build"]["imagery"]["source"]
    except Exception:  # noqa: BLE001
        return "the twin's aerial photo"
    return ORTHO_CREDITS.get(src, src)


def model_proj(man: dict) -> str:
    """The manifest's geo reference without the vertical datum terms pyproj cannot always resolve."""
    return " ".join(t for t in man["geo_reference"].split() if not t.startswith(("+geoidgrids", "+vunits")))


def is_real(b: dict) -> bool:
    """A manifest building written by apply (any provider; 3DBAG ids before the field existed)."""
    return bool(b.get("real")) or str(b.get("id", "")).startswith("NL.IMBAG")


# --------------------------------------------------------------------------- the street side (OSM)

SHOP_POI_M = 6.0    # a shop or cafe this close to a building's walls (outside, or inside near the wall) is its own
SHOP_WALL_SLACK_M = 1.5   # the walls nearest one of its shops (to this much more than the nearest) get a shopfront


HISTORIC_ARCH = ("baroque", "gothic", "renaissance", "romanesque", "classicism", "neoclassicism", "classical",
                 "neoclassical", "historicism", "art_nouveau", "jugendstil", "empire", "biedermeier", "plateresque",
                 "mudejar", "rococo", "eclectic", "neo-gothic", "neogothic", "georgian", "victorian")


def old_building(tags: dict) -> bool:
    """Whether a building's own OSM tags say it is old: heritage or historic (not "no"), a historic
    architecture, or a start_date before facades.OLD_TOWN_YEAR."""
    from twinmodel.export import facades
    if str(tags.get("heritage", "")).strip().lower() not in ("", "no"):
        return True
    if str(tags.get("historic", "")).strip().lower() not in ("", "no"):
        return True
    arch = str(tags.get("building:architecture", "")).lower()
    if any(a in arch for a in HISTORIC_ARCH):
        return True
    sd = str(tags.get("start_date", "")).strip().lstrip("~C").split("-")[0].split("..")[0]
    try:
        y = int(float(sd[:4])) if sd[:4].isdigit() else None
    except ValueError:
        y = None
    if y is None and sd[:2].isdigit() and "c" in str(tags.get("start_date", "")).lower():
        y = (int(sd[:2]) - 1) * 100      # "C18" / "18th century"
    return y is not None and 0 < y < facades.OLD_TOWN_YEAR


def historic_areas(street: Optional[dict]):
    """The union of the street data's historic areas (model frame), or None."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = []
    for rings, _tags in (street or {}).get("areas", ()):
        for ring in rings:
            if len(ring) >= 4:
                p = Polygon(ring).buffer(0)
                if not p.is_empty and p.area > 0:
                    polys.append(p)
    return unary_union(polys) if polys else None


OLD_CORE_BUFFER_M = 40.0   # old buildings this close together form an old core ...
OLD_CORE_MIN = 6           # ... of at least this many of them


def old_cores(polys: list, old: list):
    """The old cores of a twin where OSM maps no historic area: the union of OLD_CORE_BUFFER_M around the
    buildings that are old by their own data (a register year or OSM tags), in clusters of at least OLD_CORE_MIN
    of them (one listed church in a modern block is not an old town). None when there is no such cluster."""
    from shapely.ops import unary_union
    pts = [(p, k) for k, p in enumerate(polys) if old[k] and p is not None and not p.is_empty]
    if len(pts) < OLD_CORE_MIN:
        return None
    blobs = unary_union([p.buffer(OLD_CORE_BUFFER_M) for p, _ in pts])
    keep = []
    for blob in getattr(blobs, "geoms", [blobs]):
        n = sum(1 for p, _ in pts if blob.contains(p.representative_point()))
        if n >= OLD_CORE_MIN:
            keep.append(blob)
    return unary_union(keep) if keep else None


def street_matches(polys: list, tree, street: Optional[dict]):
    """({building index: [(x, y, distance to its walls) of its ground-floor shops]}, {building index: OSM wall
    tags}) for
    street_osm.to_model's points: a shop goes to the building that contains it, else the nearest within
    SHOP_POI_M; one deep inside a large building (a mall's inner shops) counts for none."""
    import shapely

    shops, tags = defaultdict(list), {}
    if not street:
        return shops, tags
    for x, y, _t in street.get("pois", ()):
        pt = shapely.Point(x, y)
        best, best_d = None, SHOP_POI_M
        for j in tree.query(pt.buffer(SHOP_POI_M)):
            poly = polys[j]
            if poly is None or poly.is_empty:
                continue
            d = poly.exterior.distance(pt)
            inside = poly.contains(pt)
            if inside and d <= SHOP_POI_M:
                best, best_d, wall_d = j, -1.0, d     # inside wins over any outside neighbour
                break
            if not inside and d < best_d:
                best, best_d, wall_d = j, d, d
        if best is not None:
            shops[int(best)].append((x, y, float(wall_d)))
    for x, y, t in street.get("buildings", ()):
        hit = tree.query(shapely.Point(x, y), predicate="within")
        if len(hit):
            tags[int(hit[0])] = t
    return shops, tags


def shop_wall(ring: np.ndarray, shops: list) -> bool:
    """Whether a wall (its outer ring, model frame) gets the shopfront: always when the building's shops are
    not known by position (BAG uses alone), else when it is one of the walls nearest a shop (within
    SHOP_WALL_SLACK_M of that shop's distance to the footprint: a corner shop gets both of its walls)."""
    if not shops:
        return True
    foot = ring[ring[:, 2] <= ring[:, 2].min() + 0.5][:, :2]
    if len(foot) == 0:
        return True
    # the foot line: a wall is a vertical plane, so its lowest points lie on one segment
    seg = foot[np.argmax(np.linalg.norm(foot - foot[0], axis=1))] - foot[0]
    for x, y, d in shops:
        p = np.array([x, y])
        t = 0.0 if not seg.any() else float(np.clip((p - foot[0]) @ seg / (seg @ seg), 0.0, 1.0))
        if np.linalg.norm(foot[0] + t * seg - p) <= d + SHOP_WALL_SLACK_M:
            return True
    return False


# --------------------------------------------------------------------------- meshes and manifest

class _Builder:
    def __init__(self):
        self.pos, self.nrm, self.uv, self.faces, self.n = [], [], [], [], 0

    def add(self, v, normal, uv, tri):
        self.pos.append(v)
        self.nrm.append(np.repeat(np.asarray(normal)[None, :], len(v), axis=0))
        self.uv.append(uv)
        self.faces.append(tri + self.n)
        self.n += len(v)

    def arrays(self):
        return np.vstack(self.pos), np.vstack(self.nrm), np.vstack(self.uv), np.vstack(self.faces)


PHOTO_WALLS_ABOVE_M = 8.0   # a city model's wall photos only above this height (the street storeys get a style)
PHOTO_MIN_SPAN_M = 4.0      # a building lower than PHOTO_WALLS_ABOVE_M plus this: the style on every wall
_PHOTO_MEAN = {}


def photo_colour(paths: list, max_images: int = 8) -> Optional[tuple]:
    """Median of the mean colours of a building's wall photos (sRGB 0-255), or None: the colour variant of the
    style its street storeys get. Dark glass and sky reflections pull single photos; the median of several holds."""
    from PIL import Image
    means = []
    for pth in sorted(set(paths))[:max_images]:
        if pth not in _PHOTO_MEAN:
            try:
                im = Image.open(pth).convert("RGB")
                im.thumbnail((32, 32))
                _PHOTO_MEAN[pth] = tuple(float(c) for c in np.asarray(im, float).reshape(-1, 3).mean(0))
            except Exception:  # noqa: BLE001 - a broken file: no colour from it
                _PHOTO_MEAN[pth] = None
        if _PHOTO_MEAN[pth] is not None:
            means.append(_PHOTO_MEAN[pth])
    if not means:
        return None
    return tuple(int(round(c)) for c in np.median(np.asarray(means), axis=0))


ATLAS_PX = 4096      # one photo-texture atlas per tile (PLATEAU's LoD 2 facade and roof photos)
ATLAS_GUTTER = 2


def pack_atlas(sizes: list, atlas_px: int = ATLAS_PX, gutter: int = ATLAS_GUTTER):
    """Shelf-pack images of ``sizes`` [(w, h)] into a square atlas: (scale, [(x, y, w, h)] placements in atlas
    pixels). The images are scaled down together until they fit."""
    if not sizes:
        return 1.0, []
    area = sum((w + 2 * gutter) * (h + 2 * gutter) for w, h in sizes)
    scale = min(1.0, math.sqrt(0.85 * atlas_px * atlas_px / max(area, 1)))
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    for _ in range(40):
        place = [None] * len(sizes)
        x = y = shelf = 0
        ok = True
        for i in order:
            w = max(1, int(sizes[i][0] * scale))
            h = max(1, int(sizes[i][1] * scale))
            if x + w + 2 * gutter > atlas_px:
                x, y, shelf = 0, y + shelf, 0
            if y + h + 2 * gutter > atlas_px or w + 2 * gutter > atlas_px:
                ok = False
                break
            place[i] = (x + gutter, y + gutter, w, h)
            x += w + 2 * gutter
            shelf = max(shelf, h + 2 * gutter)
        if ok:
            return scale, place
        scale *= 0.9
    raise ValueError("cannot pack %d images into %d px" % (len(sizes), atlas_px))


def build_atlas(paths: list, atlas_px: int = ATLAS_PX, gutter: int = ATLAS_GUTTER):
    """(atlas image HxWx3 uint8, [(x, y, w, h)] per path): the images packed, with their edges repeated into
    the gutter so mipmaps do not bleed."""
    from PIL import Image
    imgs = []
    for pth in paths:
        try:
            imgs.append(Image.open(pth).convert("RGB"))
        except Exception:  # noqa: BLE001 - a broken file: a grey square
            imgs.append(Image.new("RGB", (8, 8), (128, 128, 128)))
    scale, place = pack_atlas([im.size for im in imgs], atlas_px, gutter)
    atlas = np.zeros((atlas_px, atlas_px, 3), np.uint8)
    for im, (x, y, w, h) in zip(imgs, place):
        a = np.asarray(im.resize((w, h), Image.BILINEAR) if (w, h) != im.size else im)
        a = np.pad(a, ((gutter, gutter), (gutter, gutter), (0, 0)), mode="edge")
        atlas[y - gutter:y + h + gutter, x - gutter:x + w + gutter] = a
    return atlas, place


def atlas_uv(uv: np.ndarray, rect, atlas_px: int = ATLAS_PX) -> np.ndarray:
    """CityGML texture coordinates (origin bottom left, 0-1 over the image) -> glTF UVs (origin top left) in the
    atlas, kept half a pixel inside the image's rectangle."""
    x, y, w, h = rect
    u = np.clip(uv[:, 0], 0.5 / w, 1 - 0.5 / w)
    v = np.clip(uv[:, 1], 0.5 / h, 1 - 0.5 / h)
    return np.column_stack([(x + u * w) / atlas_px, (y + (1.0 - v) * h) / atlas_px])


def apply(build: Path, man: dict, buildings: list, source: Source, *, looks: bool = True,
          ortho=None, ortho_label: str = "", texture_px: Optional[int] = None, roof_px: int = 1024,
          procedural_min_height: Optional[float] = None, window_lights: bool = True,
          street: Optional[dict] = None, window_geometry: bool = False, interiors: bool = False,
          dry_run: bool = False, photo_walls_above_m: float = None) -> dict:
    """Mesh, style and texture ``buildings`` and swap them into the bake export ``build``/ue (its
    manifest ``man`` is updated in place and written). Returns the manifest's stats.

    ``street`` (twinmodel.ingest.street_osm.to_model, None: unknown) adds what OSM maps at the street side:
    ground-floor shops and cafes within SHOP_POI_M of a building's walls give it a shopfront ground floor on
    the walls near them (without it the BAG units' uses decide, on every street wall), and a mapped
    building:material / building:colour picks the wall and its colour.

    ``window_geometry``: the windows the textures would paint become geometry (export/windows.py): holes cut in
    the walls, the walls drawn without painted windows, one instanced module per style (reveal, frame, sill,
    glass) listed in the manifest (``window_modules``, ``window_instances``; provenance "inferred").
    ``interiors``: the glass shows a room picked by the building use (provenance "generated"), else dark glass.
    ``dry_run``: count the windows and interiors per kind and return those stats; nothing is written.

    ``photo_walls_above_m`` (default PHOTO_WALLS_ABOVE_M; 0: the source's photos everywhere, as before 2026-11-28):
    a city model's own wall photos (PLATEAU LoD 2) are aerial oblique images at about 19 cm per texel in the
    exports, blurred, shadowed and sky-tinted at the street; walls keep them only above this height over the
    building's ground, the storeys below get the building's facade style (shop floor included) with the wall
    colour taken from its photos. Buildings lower than this plus PHOTO_MIN_SPAN_M get the style on every wall."""
    import shapely
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    from twinmodel.export import facades
    from twinmodel.export import windows as winmod
    from twinmodel.export.ue import _tile_index, write_glb
    if dry_run:
        window_geometry = True
    # window geometry: tile -> (mesh key, material key) -> [(transform, custom data)]
    win_inst = defaultdict(lambda: defaultdict(list))
    win_meshes = {}          # mesh key -> (style, door, part)
    win_stats = defaultdict(int)

    from twinmodel.export import clearance
    clearance.restore(build / "ue", man)   # a road clearance cut is redone on the new meshes (clearance.apply)
    texture_px = texture_px or facades.TEX_PX
    photo_walls_above_m = PHOTO_WALLS_ABOVE_M if photo_walls_above_m is None else float(photo_walls_above_m)
    n_photo_band = n_photo_styled = 0
    procedural_min_height = facades.PROCEDURAL_MIN_HEIGHT_M if procedural_min_height is None \
        else procedural_min_height
    out = build / "ue"
    name, tile_m = man["name"], float(man["tile_m"])

    # ---- the OSM buildings (their categories help the styles)
    already = bool(man["stats"].get("real_buildings")) or \
        str(man["stats"].get("building_source", "")).startswith("3DBAG")
    osm_blds = man.get("buildings_osm")
    if osm_blds is None:   # the bake export's OSM footprints (also after a --plain run)
        osm_blds = [b for b in man.get("buildings", []) if not is_real(b)]
    osm_polys, osm_cat = [], []
    for b in osm_blds or ():
        ring = b.get("raw_ring_ue") or (b.get("rings_ue") or [None])[0]
        if ring and len(ring) >= 3:
            poly = Polygon([(x / 100.0, -y / 100.0) for x, y in ring])
            if poly.is_valid and poly.area > 0:
                osm_polys.append(poly)
                osm_cat.append(b.get("category"))
    osm_tree = STRtree(osm_polys) if osm_polys else None

    # ---- styles
    polys = [b.poly if b.poly is not None else Polygon() for b in buildings]
    tree = STRtree(polys)
    shop_xy, wall_tags = street_matches(polys, tree, street)
    old_area = historic_areas(street)
    if old_area is None:    # no historic area in OSM here: the old cores of the old buildings themselves
        own_old = [old_building(wall_tags.get(k, {})) or (b.year is not None and 0 < b.year < facades.OLD_TOWN_YEAR)
                   for k, b in enumerate(buildings)]
        old_area = old_cores(polys, own_old)
    from shapely.prepared import prep
    old_area_p = prep(old_area) if old_area is not None else None
    n_historic = 0
    styles_n = defaultdict(int)
    info = []
    for k, b in enumerate(buildings):
        cat = None
        if osm_tree is not None and b.poly is not None and not b.poly.is_empty:
            hit = osm_tree.query(b.poly.representative_point(), predicate="within")
            cat = osm_cat[hit[0]] if len(hit) else None
        tags = wall_tags.get(k, {})
        historic = old_building(tags) or (old_area_p is not None and b.poly is not None and not b.poly.is_empty and
                                          old_area_p.contains(b.poly.representative_point()))
        n_historic += historic
        bi_ = facades.BuildingInfo(id=b.id, year=b.year, uses=b.uses, height_m=b.height,
                                   area_m2=b.poly.area if b.poly is not None else 0.0, osm_category=cat,
                                   region=b.region, shop_pois=None if street is None else len(shop_xy.get(k, ())),
                                   material=tags.get("building:facade:material") or tags.get("building:material"),
                                   historic=historic)
        style, shop = facades.classify(bi_) if looks else ("plain", False)
        colour = facades.parse_colour(tags.get("building:facade:colour") or tags.get("building:colour"))
        info.append({"style": style, "shop": shop, "osm_category": cat, "colour": colour,
                     "shop_xy": shop_xy.get(k, []) if shop else []})
        styles_n[style] += 1

    # ---- meshes
    tiles = defaultdict(lambda: defaultdict(_Builder))   # tile -> (material key, low) -> builder
    photo = defaultdict(lambda: {"images": {}, "parts": []})   # tile -> the source's photo-textured surfaces
    n_textured = 0
    used_keys = set()
    n_buildings = n_parts = n_tris = n_party = n_plain_roof = n_shop_walls = 0
    boundaries = [p.exterior if p is not None and not p.is_empty else None for p in polys]
    top = winmod.roof_top(buildings, triangulate) if window_geometry else None
    for k, b in enumerate(buildings):
        i = info[k]
        allv = np.vstack([r for rings, _ in b.surfaces for r in rings]) if b.surfaces else None
        if allv is None or not len(allv):
            continue
        cx, cy = b.centre or (float(np.mean(allv[:, 0])), float(np.mean(allv[:, 1])))
        tile = _tile_index(cx, cy, tile_m)
        ground = float(b.ground) if b.ground is not None else float(allv[:, 2].min())
        photo_roof = ortho is not None and (source.roof_photo_max_m is None or b.height <= source.roof_photo_max_m)
        st = facades.STYLES.get(i["style"]) if looks else None
        if st is not None:
            colour = i["colour"]
            if colour is None and b.textures and photo_walls_above_m > 0:
                colour = photo_colour([b.textures[si][0] for si, (_, kd) in enumerate(b.surfaces)
                                       if si in b.textures and kd == "WallSurface"])
            var = facades.variant_of(st.name, b.id, colour)
            low_h = facades.SHOP_FLOOR_M if i["shop"] else 0.0
            n_st, storey_h = facades.storeys_for(b.height - low_h, st.storey_m)
            blank = facades.STYLES[facades.blank_of(st.name, b.region)]
            blank_key = facades.material_key(blank.name, facades.variant_of(blank.name, b.id) if
                                             blank.name != st.name else var)
            shop_name = facades.shop_style(b.region)
            shop_st = facades.STYLES[shop_name]
            shop_key = facades.material_key(shop_name, facades.variant_of(shop_name, b.id))
            key = facades.material_key(st.name, var)
            eaves_z = ground + b.height
            i["procedural"] = facades.procedural_ok(st.name, b.height, procedural_min_height)
            near = []
            if b.poly is not None and not b.poly.is_empty:
                near = [boundaries[j] for j in tree.query(b.poly.buffer(facades.PARTY_TOL_M))
                        if j != k and boundaries[j] is not None]
        used = False
        n_wall = 0
        for si, (rings, kind) in enumerate(b.surfaces):
            if kind in SKIP_SURFACES or not rings:
                continue
            res = triangulate(rings)
            if res is None:
                continue
            v, tri, normal = res
            n_tris += len(tri)
            tb = tiles[tile]
            used = True
            photo_cut = False
            if looks and si in b.textures:   # the source's own photo: into the tile's atlas
                img, uvs = b.textures[si]
                uv = np.vstack(uvs)
                wall_kind = not (kind == "RoofSurface" or abs(normal[2]) > 0.7)
                if wall_kind and st is not None and photo_walls_above_m > 0:
                    # the street storeys get the facade style; the photo only above photo_walls_above_m
                    if b.height < photo_walls_above_m + PHOTO_MIN_SPAN_M:
                        n_photo_styled += 1
                        photo_cut = True
                    else:
                        (vb, tb_), (va, ta) = facades.clip_triangles(np.hstack([v, uv]), tri,
                                                                     ground + photo_walls_above_m)
                        if len(ta):
                            ph = photo[tile]
                            idx = ph["images"].setdefault(img, len(ph["images"]))
                            ph["parts"].append((va[:, :3], normal, va[:, 3:5], idx, ta))
                            n_textured += 1
                        if not len(tb_):
                            continue
                        v, tri = vb[:, :3], tb_
                        n_photo_band += 1
                        photo_cut = True
                if not photo_cut:
                    ph = photo[tile]
                    idx = ph["images"].setdefault(img, len(ph["images"]))
                    ph["parts"].append((v, normal, uv, idx, tri))
                    n_textured += 1
                    continue

            def add_wall(mkey, v, tri, uvfn):
                """Walls below the eaves of a procedural-capable building go apart (kind
                'building', which --buildings procedural replaces); the rest is kept."""
                parts_ = [(v, tri, False)]
                if i.get("procedural"):
                    (vb, tb_), (va, ta) = facades.clip_triangles(v, tri, eaves_z)
                    parts_ = [(vb, tb_, True), (va, ta, False)]
                for vv, tt, low in parts_:
                    if len(tt):
                        tb[(mkey, low)].add(vv, normal, uvfn(vv), tt)
                used_keys.add(mkey)
            if not looks:
                if abs(normal[2]) > 0.7:
                    uv = v[:, :2]
                else:
                    along = np.array([-normal[1], normal[0], 0.0]) / max(1e-9, math.hypot(normal[0], normal[1]))
                    uv = np.column_stack([v @ along, v[:, 2]])
                tb[("building", True)].add(v, normal, uv, tri)
                used_keys.add("building")
                continue
            if kind == "RoofSurface" or abs(normal[2]) > 0.7:
                rkey = "roof_%d_%d" % tile if photo_roof else "roof"
                tb[(rkey, False)].add(v, normal, facades.roof_uv(v, tile, tile_m), tri)
                used_keys.add(rkey)
                continue
            # walls
            along = facades.wall_frame(normal)
            width = float(np.ptp(rings[0] @ along))
            foot = rings[0][rings[0][:, 2] <= rings[0][:, 2].min() + 0.5]
            mid = shapely.Point(float(foot[:, 0].mean()), float(foot[:, 1].mean()))
            party = any(bd.distance(mid) < facades.PARTY_TOL_M for bd in near)
            if party or width < 0.6 * st.bay_m or st.blank:
                n_party += int(party)
                add_wall(blank_key, v, tri, lambda vv: facades.blank_uv(vv, normal, blank))
                continue
            ring0 = rings[0]
            off = facades.light_offset(b.id, n_wall)   # each wall shows another part of the night mask
            wall_no = n_wall
            n_wall += 1
            is_shop_wall = bool(i["shop"] and shop_wall(ring0, i["shop_xy"]))
            wkey, wshop_key = key, shop_key
            if window_geometry and not photo_cut:   # (a wall with the photo above keeps painted windows)
                shop_var = facades.variant_of(shop_name, b.id)
                # the storey grid of wall_uv: storeys from ground + low_h; a wall below that (the non-shop walls of
                # a shop building) repeats the grid downwards, whole storeys
                z_lo = ground + low_h
                if not is_shop_wall and low_h > 0:
                    z_lo -= storey_h * math.floor(low_h / storey_h + 1e-6)
                ops_up = winmod.layout(ring0, normal, z_lo, ground + b.height, storey_h, st, var, poly_rings=rings)
                if is_shop_wall:
                    # the upper storeys' grid starts at the shop floor's top only when the building has a shop
                    # floor (low_h); a wall of a shop building always gets it
                    ops_shop = winmod.layout(ring0, normal, ground, ground + facades.SHOP_FLOOR_M,
                                             shop_st.storey_m, shop_st, shop_var, poly_rings=rings)
                else:
                    ops_shop = []
                ops = ops_up + ops_shop
                if ops and top is not None:
                    # a window with building in front of it (another part of the same building, a neighbour whose
                    # footprint touches: Catastro parts, 3DBAG/Overture blocks) is never seen: no window there
                    seen = winmod.visible(ops, normal, ring0, top)
                    win_stats["windows_hidden"] += len(ops) - int(seen.sum())
                    ops = [o for o, v in zip(ops, seen) if v]
                    ops_up = [o for o in ops_up if o in set(ops)]
                    ops_shop = [o for o in ops_shop if o in set(ops)]
                if ops:
                    cut = winmod.cut_openings(rings, normal, ops)
                    pieces = [triangulate(c) for c in cut]
                    pieces = [pc for pc in pieces if pc is not None]
                    if pieces:
                        vs, ts, base = [], [], 0
                        for pv, pt, _ in pieces:
                            vs.append(pv)
                            ts.append(pt + base)
                            base += len(pv)
                        v, tri = np.vstack(vs), np.vstack(ts)
                    if ops_up:
                        wkey = facades.material_key(st.name, var, windows=False)
                    if ops_shop:
                        wshop_key = facades.material_key(shop_name, shop_var, windows=False)
                    use = facades.USE_OF_STYLE.get(st.name, "home")
                    for o in ops:
                        ost = facades.STYLES[o.style]
                        tf = winmod.instance_transform(o, normal, ring0, ost)
                        mk = winmod.module_key(o.style, o.door)
                        o_shop = facades.STYLES[o.style].shop
                        o_use = "shop" if o_shop else use
                        room = winmod.interior_of(o.style) if not o_shop else "shop"
                        glass = ("interior_" + room) if (interiors and room) else "glass_" + o.style
                        wall_mat = facades.material_key(o.style, shop_var if o_shop else var,
                                                        windows=False)
                        for part, mat in (("reveal", wall_mat), ("frame", "frame_" + o.style),
                                          ("sill", "sill"), ("glass", glass)):
                            if part == "sill" and (o.door or not ost.sill):
                                continue
                            if part == "reveal" and winmod.REVEAL_M.get(o.style, 0.12) <= winmod.FRAME_DEPTH_M:
                                continue
                            mesh = f"{mk}_{part}"
                            win_meshes[mesh] = (o.style, o.door, part)
                            win_inst[tile][(mesh, mat)].append(
                                (tf, winmod.night_data(b.id, wall_no, o) if part == "glass" else None))
                        win_stats["windows"] += 1
                        win_stats["doors"] += int(o.door)
                        win_stats["shop_windows"] += int(o_shop)
                        win_stats["interiors_" + (room or "none")] += int(bool(interiors and room))
                        win_stats["use_" + o_use] += 1
                    win_stats["walls_with_windows"] += 1
            if is_shop_wall:
                n_shop_walls += 1
                (vb, tb_), (va, ta) = facades.clip_triangles(v, tri, ground + facades.SHOP_FLOOR_M)
                if len(tb_):
                    add_wall(wshop_key, vb, tb_, lambda vv: facades.wall_uv(
                        vv, normal, ground, shop_st.storey_m, shop_st.bay_m, ring=ring0, tile_offset=off))
                v, tri = va, ta
                if not len(tri):
                    continue
            add_wall(wkey, v, tri, lambda vv: facades.wall_uv(vv, normal, ground, storey_h, st.bay_m,
                                                              ring=ring0, z0=low_h, tile_offset=off))
        if used:
            n_parts += b.parts
            n_buildings += 1
            n_plain_roof += int(looks and ortho is not None and not photo_roof)

    if dry_run:
        stats = {"buildings": n_buildings, "window_tiles": len(win_inst),
                 "window_meshes": len(win_meshes),
                 "window_instances": sum(len(t) for groups in win_inst.values() for t in groups.values()),
                 **dict(sorted(win_stats.items()))}
        print("windows (dry run): " + json.dumps(stats))
        return stats

    # ---- the source's photo textures: one atlas per tile
    atlases = {}
    for tile, ph in sorted(photo.items()):
        if not ph["parts"]:
            continue
        paths = [p for p, _ in sorted(ph["images"].items(), key=lambda t: t[1])]
        atlas, place = build_atlas(paths)
        key = "photo_%d_%d" % tile
        for v, normal, uv, idx, tri in ph["parts"]:
            tiles[tile][(key, False)].add(v, normal, atlas_uv(uv, place[idx]), tri)
        used_keys.add(key)
        atlases[key] = atlas

    # ---- textures
    tex_dir = out / "textures"
    # entries of other tools stay (tools/ground_photo.py: ground_<x>_<y>)
    textures = {k: v for k, v in man.get("textures", {}).items() if k.startswith("ground_")}
    n_lights = 0
    if looks:
        tex_dir.mkdir(exist_ok=True)
        for key in sorted(k for k in used_keys if k.startswith(("facade_", "facadenw_"))):
            sname, var, painted = facades.parse_material_key(key)
            alb, nrm_, orm = facades.synth_facade(facades.STYLES[sname], int(var),
                                                  px=facades.texture_px_for(sname, texture_px), windows=painted)
            entry = {}
            for suffix, arr in (("d", alb), ("n", nrm_), ("orm", orm)):
                rel = f"textures/T_{name}_{key}_{suffix}.png"
                facades.save_png(out / rel, arr)
                entry[{"d": "base_color", "n": "normal", "orm": "orm"}[suffix]] = rel
            st = facades.STYLES[sname]
            entry.update(style=sname, variant=int(var), bay_m=st.bay_m, storey_m=st.storey_m)
            # lit windows at night (bake_level.py puts a style with a mask on the night parent)
            lights = facades.synth_window_lights(st, int(var)) if (window_lights and painted) else None
            if lights is not None:
                rel = f"textures/T_{name}_{key}_w.png"
                facades.save_png(out / rel, lights)
                entry.update(windows=rel, use=facades.USE_OF_STYLE.get(sname, "home"),
                             windows_uv_scale=facades.MASK_UV_SCALE)
                n_lights += 1
            textures[key] = entry
        for key, atlas in sorted(atlases.items()):
            from PIL import Image
            rel = f"textures/T_{name}_{key}_d.jpg"
            Image.fromarray(atlas).save(out / rel, quality=90)
            facades.save_png(out / f"textures/T_{name}_flat_n.png", facades.flat_normal())
            facades.save_png(out / f"textures/T_{name}_photo_orm.png", facades.constant_orm(0.85))
            textures[key] = {"base_color": rel, "normal": f"textures/T_{name}_flat_n.png",
                             "orm": f"textures/T_{name}_photo_orm.png", "source": source.asset_source}
        if ortho:
            facades.save_png(out / f"textures/T_{name}_flat_n.png", facades.flat_normal())
            facades.save_png(out / f"textures/T_{name}_roof_orm.png", facades.constant_orm(0.8))
            for key in sorted(k for k in used_keys if k.startswith("roof_")):
                tx, ty = (int(t) for t in key.split("_")[1:3])
                img = facades.ortho_tile(ortho[0], ortho[1], (tx, ty), tile_m, roof_px)
                if img is None:
                    continue
                rel = f"textures/T_{name}_{key}_d.png"
                facades.save_png(out / rel, img)
                textures[key] = {"base_color": rel, "normal": f"textures/T_{name}_flat_n.png",
                                 "orm": f"textures/T_{name}_roof_orm.png", "source": ortho_label}

    # ---- window modules (export/windows.py): one glb per module part, solid-colour materials, the instances
    window_modules, window_instances = {}, []
    if window_geometry and win_meshes:
        wdir = out / "meshes" / "windows"
        wdir.mkdir(parents=True, exist_ok=True)
        tex_dir.mkdir(exist_ok=True)
        facades.save_png(out / f"textures/T_{name}_flat_n.png", facades.flat_normal())

        def solid(mkey, rgb, rough):
            if mkey in textures:
                return
            rel = f"textures/T_{name}_{mkey}_d.png"
            facades.save_png(out / rel, np.full((4, 4, 3), rgb, dtype=np.uint8))
            orel = f"textures/T_{name}_{mkey}_orm.png"
            facades.save_png(out / orel, facades.constant_orm(rough))
            textures[mkey] = {"base_color": rel, "normal": f"textures/T_{name}_flat_n.png", "orm": orel,
                              "provenance": winmod.PROVENANCE_WINDOWS}
        parts_cache = {}
        for mesh, (sname, door, part) in sorted(win_meshes.items()):
            sty = facades.STYLES[sname]
            if (sname, door) not in parts_cache:
                parts_cache[(sname, door)] = winmod.module_parts(sty, door)
            geom = parts_cache[(sname, door)].get(part)
            if geom is None:
                continue
            pv, pn, puv, pf = geom
            if part == "reveal":   # the wall's texture at its own scale (one tile = TEX_BAYS x TEX_STOREYS)
                puv = puv / np.array([facades.TEX_BAYS * sty.bay_m, facades.TEX_STOREYS * sty.storey_m])
            asset = f"{name}_W_{mesh}".replace("-", "m")
            bbox = write_glb(wdir / f"{asset}.glb", asset, pv, pn, puv, pf, part)
            window_modules[mesh] = {"file": f"meshes/windows/{asset}.glb", "asset": asset, "style": sname,
                                    "door": bool(door), "part": part, "triangles": int(len(pf)),
                                    "bbox_ue": bbox, "semantic": "Building",
                                    "provenance": winmod.PROVENANCE_WINDOWS}
            if part == "frame":
                solid("frame_" + sname, sty.frame, 0.5)
            elif part == "sill":
                solid("sill", (200, 198, 190), 0.8)
            elif part == "glass":
                solid("glass_" + sname, sty.glass, 0.06)
        rooms = set()
        for (tx, ty), groups in sorted(win_inst.items()):
            for (mesh, mat), rows in sorted(groups.items()):
                if mesh not in window_modules:
                    continue
                if mat.startswith("interior_"):
                    rooms.add(mat[len("interior_"):])
                entry = {"tile": [tx, ty], "mesh": mesh, "material": mat,
                         "provenance": winmod.PROVENANCE_INTERIORS if mat.startswith("interior_")
                         else winmod.PROVENANCE_WINDOWS,
                         "transforms": [list(t) for t, _ in rows]}
                if any(c is not None for _, c in rows):
                    entry["custom"] = [list(c) for _, c in rows]
                window_instances.append(entry)
        for room in sorted(rooms):
            # bake_level.py puts these on CARLA's fake-interior materials (twin_materials.INTERIOR_PARENT)
            textures.setdefault("interior_" + room, {"interior": room,
                                                    "provenance": winmod.PROVENANCE_INTERIORS})

    # ---- glb files and the manifest
    mesh_dir = out / "meshes"
    old = [a for a in man["assets"] if a["kind"] in ("building", "building_3dbag", "roof")]
    template = next((a for a in old), None) or {"material": "building", "semantic": "Building", "layer": 0}
    if not already:
        backup = mesh_dir / "osm_buildings"
        backup.mkdir(exist_ok=True)
        for a in old:
            src = out / a["file"]
            if src.exists():
                shutil.move(str(src), str(backup / src.name))
    else:
        for a in old:
            (out / a["file"]).unlink(missing_ok=True)
    assets = [a for a in man["assets"] if a["kind"] not in ("building", "building_3dbag", "roof")]
    for (tx, ty), mats in sorted(tiles.items()):
        for (key, low), bld in sorted(mats.items()):
            if not bld.faces:
                continue
            pos, nrm, uv, faces = bld.arrays()
            is_roof = key.startswith("roof")
            if key == "building":
                asset = f"{name}_L0_building_{tx}_{ty}"
            elif key.startswith("photo_"):
                asset = f"{name}_L0_building_{tx}_{ty}_photo"
            else:
                asset = f"{name}_L0_{'roof' if is_roof else 'building'}_{tx}_{ty}" + \
                        ("" if is_roof else "_" + key[len("facade_"):] + ("_p" if low else ""))
                if key == "roof":
                    asset += "_plain"
            asset = asset.replace("-", "m")
            bbox = write_glb(mesh_dir / f"{asset}.glb", asset, pos, nrm, uv, faces, key)
            # kind 'building': what --buildings procedural replaces; 'building_3dbag': the real walls it keeps
            kind = "roof" if is_roof else ("building" if low else "building_3dbag")
            a = {"file": f"meshes/{asset}.glb", "asset": asset, "layer": template.get("layer", 0),
                 "kind": kind, "material": key,
                 "semantic": "Building", "tile": [tx, ty],
                 "vertices": int(len(pos)), "triangles": int(len(faces)), "bbox_ue": bbox,
                 "source": source.asset_source}
            if key in textures:
                a["texture"] = key
            assets.append(a)
    man["assets"] = assets
    if window_modules:
        man["window_modules"] = window_modules
        man["window_instances"] = window_instances
        man["stats"]["windows"] = dict(sorted(win_stats.items()))
        man["stats"]["window_instances"] = sum(len(e["transforms"]) for e in window_instances)
        man["stats"]["interiors"] = bool(interiors)
    else:
        man.pop("window_modules", None)
        man.pop("window_instances", None)
    if textures:
        man["textures"] = textures
    else:
        man.pop("textures", None)
    if not already:
        man["stats"]["buildings_osm_replaced"] = man["stats"].get("buildings", 0)
    if looks and "buildings_osm" not in man:
        man["buildings_osm"] = osm_blds
    if looks:
        man["buildings"] = []
        for b, i in zip(buildings, info):
            poly = b.poly
            if poly is None or poly.is_empty or b.ground is None:
                continue
            ring = np.asarray(poly.exterior.coords)[:-1]
            rgb = facades.mean_colour(ortho[0], ortho[1], ring) if ortho else None
            gen_ring = ring
            if i.get("procedural"):
                # the generator fills every segment shorter than one 5 m module with a plain quad:
                # drop the footprint's small jogs (and zero-length edges) first
                simple = poly.simplify(facades.PROCEDURAL_SIMPLIFY_M, preserve_topology=True)
                if simple.geom_type == "Polygon" and simple.is_valid and len(simple.exterior.coords) >= 4:
                    gen_ring = np.asarray(simple.exterior.coords)[:-1]
            man["buildings"].append({
                "id": b.id, "real": source.name,
                "rings_ue": [[[round(x * 100, 1), round(-y * 100, 1)] for x, y in gen_ring]],
                "raw_ring_ue": [[round(x * 100, 1), round(-y * 100, 1)] for x, y in ring],
                "base_z_cm": round(float(b.ground) * 100, 1),
                "eaves_z_cm": round((float(b.ground) + b.height) * 100, 1),
                "roof_z_cm": round(float(b.roof_z if b.roof_z is not None else b.ground + b.height) * 100, 1),
                "height_m": round(b.height, 2), "levels": b.levels or 0,
                "category": i["osm_category"] or facades.dominant_use(b.uses) or "yes",
                "year": b.year, "uses": {k: round(v) for k, v in b.uses.items()},
                "roof_type": b.roof_type, "style": i["style"], "shopfront": bool(i["shop"]),
                "roof_rgb": list(rgb) if rgb else None, "procedural": bool(i.get("procedural")),
                "family": facades.procedural_family(i["style"], b.height)})
    man["stats"]["buildings"] = n_buildings
    man["stats"]["real_buildings"] = source.name
    man["stats"]["building_source"] = f"{source.label} ({n_parts} parts, {n_tris} triangles)"
    if looks:
        man["stats"]["facade_styles"] = dict(sorted(styles_n.items()))
        man["stats"]["historic_buildings"] = int(n_historic)   # in an old-town area or tagged old (facades.classify)
        man["stats"]["party_walls"] = n_party
        man["stats"]["roof_photo"] = bool(ortho)
        man["stats"]["window_light_masks"] = n_lights
        if source.roof_photo_max_m is not None and ortho is not None:
            man["stats"]["roof_plain_tall"] = n_plain_roof
        man["stats"]["procedural_capable"] = sum(1 for i in info if i.get("procedural"))
        man["stats"]["shopfront_buildings"] = sum(1 for i in info if i["shop"] or facades.STYLES.get(
            i["style"], facades.STYLES["blank_brick"]).shop)
        man["stats"]["shopfront_walls"] = n_shop_walls
        if street is not None:
            man["stats"]["street_osm"] = {"pois": len(street.get("pois", ())),
                                          "pois_at_buildings": sum(len(v) for v in shop_xy.values()),
                                          "buildings_with_wall_tags": len(wall_tags)}
        if n_textured:
            man["stats"]["photo_textured_surfaces"] = n_textured
            man["stats"]["photo_atlases"] = len(atlases)
        if n_photo_band or n_photo_styled:   # street storeys styled under the source's wall photos
            man["stats"]["photo_walls_above_m"] = photo_walls_above_m
            man["stats"]["photo_walls_street_styled"] = n_photo_band
            man["stats"]["photo_walls_all_styled"] = n_photo_styled
        man["stats"].update(source.extra_stats)
    man["stats"]["triangles"] = int(sum(a["triangles"] for a in assets))
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    n_assets = sum(1 for a in assets if a["kind"] in ("building", "building_3dbag", "roof"))
    print(f"{source.name}: {n_buildings} buildings, {n_parts} parts, {n_tris} triangles in {len(tiles)} tiles, "
          f"{n_assets} assets, {len(textures)} textures; styles {dict(styles_n)}; {n_party} party walls"
          + (f"; {n_plain_roof} tall buildings with a plain roof" if n_plain_roof else ""))
    return man["stats"]
