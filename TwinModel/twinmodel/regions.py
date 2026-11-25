"""Which open data to use where: a twin of any place, from the best keyless, openly licensed data of its country.

``plan_for(iso2, bbox)`` gives a ``RegionPlan``: the orthophoto and elevation providers to try in order (names of
twinmodel.ingest.imagery / twinmodel.ingest.elevation ``PROVIDERS``), the real-building providers
(twinmodel.ingest.buildings ``PROVIDERS``), the facade region (twinmodel.export.facades ``classify``) and the driving
side. National services come first; every provider declines outside its own coverage (``COVERAGE``), so a list can
name a service that covers only part of a country (Flanders, New York State, Catalonia) and the next one takes over.
The global fallbacks are the AWS Terrain Tiles and Copernicus GLO-30 for the ground's heights and Overture Maps (or
OpenStreetMap) for the buildings. There is no openly licensed global aerial photo at street scale (ESRI, Google and
Bing imagery may not be used; Sentinel-2's 10 m pixels are too coarse for a street), so outside the countries listed
here the ground keeps CARLA's materials and roofs a plain roof material; the OSM map tiles stay a refinement aid only.

``SOURCES`` holds each provider's label, licence and home page; ``attribution_text`` turns a twin's report and bake
manifest into the ATTRIBUTION file that travels with the baked level (bake_level.py copies it next to the OpenDRIVE).

| country | aerial photo | elevation | buildings |
| --- | --- | --- | --- |
| NL | PDOK Luchtfoto 8 cm | AHN 0.5 m | 3DBAG LoD 2.2 |
| ES | ICGC 25 cm (Catalonia), IGN PNOA 25 cm | ICGC 2 m, IGN MDT05 | Catastro (floors per part, year, use) |
| FR | IGN BD ORTHO 20 cm | IGN RGE ALTI 1 m | IGN BD TOPO (height, floors, year, use) |
| BE | Flanders orthophoto 25 cm (Flanders only) | global | Overture |
| CH | swisstopo SWISSIMAGE 10 cm | swisstopo swissALTI3D 0.5 m | Overture |
| US | NYS orthos 15 cm (New York State), USGS NAIP 60 cm | USGS 3DEP 1 m | NYC 3D model (NYC), Overture |
| JP | GSI seamless photo | GSI DEM 5 m | PLATEAU (listed wards), Overture |
| FI | City of Helsinki 5 cm (Helsinki), MML 0.5 m (Kapsi mirror) | global | Overture |
| elsewhere | none (CARLA's ground) | AWS Terrain Tiles, Copernicus GLO-30 | Overture, else OSM |
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional

# ISO 3166-1 alpha-2 of the countries that drive on the left (the same list as coupled-carla's map maker)
LEFT_HAND = frozenset("""
    GB IE IM JE GG MT CY IN PK BD LK NP BT MV JP MY SG BN ID TH TL HK MO
    ZA NA BW ZW ZM MW MZ LS SZ KE UG TZ MU SC SH AU NZ FJ PG SB WS TO TV KI NR CK NU NF CX CC PN TK
    JM BS BB TT GY SR AG DM GD KN LC VC AI BM VG KY MS TC VI FK
""".split())

# facade region (twinmodel.export.facades.classify): "south" = warm render and sandstone instead of brick
SOUTH = frozenset("ES PT IT GR MT CY HR FR MC AD SM VA MX AR CL UY PE CO BR EC BO PY VE MA TN DZ IL LB TR".split())

# (s, w, n, e) boxes each provider covers; outside them it is skipped without a request
COVERAGE = {
    "pdok_luchtfoto": [(50.70, 3.20, 53.60, 7.30)],
    "ahn_wcs": [(50.70, 3.20, 53.60, 7.30)],
    "icgc": [(40.52, 0.15, 42.87, 3.34)],
    "icgc_mdt2m": [(40.52, 0.15, 42.87, 3.34)],
    "icgc_terr": [(40.52, 0.15, 42.87, 3.34)],
    "ign_es": [(35.90, -9.40, 43.85, 4.40), (27.60, -18.20, 29.50, -13.40)],
    "ign_wcs": [(35.90, -9.40, 43.85, 4.40), (27.60, -18.20, 29.50, -13.40)],
    "ign_fr": [(41.30, -5.20, 51.15, 9.60)],
    "rge_alti": [(41.30, -5.20, 51.15, 9.60)],
    "swissimage": [(45.80, 5.95, 47.81, 10.50)],
    "swissalti3d": [(45.80, 5.95, 47.81, 10.50)],
    "flanders": [(50.68, 2.53, 51.51, 5.92)],
    "gsi_photo": [(20.0, 122.0, 46.0, 154.0)],
    "gsi_dem": [(20.0, 122.0, 46.0, 154.0)],
    "naip": [(24.4, -125.0, 49.5, -66.9)],
    "usgs_3dep": [(24.4, -125.0, 49.5, -66.9), (51.0, -180.0, 72.0, -129.0), (18.8, -160.5, 22.3, -154.7)],
    "nys_orthos": [(40.49, -79.77, 45.02, -71.85)],
    "helsinki_ortho": [(60.08, 24.78, 60.30, 25.27)],     # the city of Helsinki, islands included
    "mml_ortho": [(59.45, 19.0, 70.10, 31.60)],           # Finland
}


def covers(provider: str, bbox_swne) -> bool:
    """True when ``provider`` has data for the whole bbox (providers without a COVERAGE entry: everywhere)."""
    boxes = COVERAGE.get(provider)
    if not boxes:
        return True
    s, w, n, e = bbox_swne
    return any(bs <= s and n <= bn and bw <= w and e <= be for bs, bw, bn, be in boxes)


@dataclass(frozen=True)
class RegionPlan:
    iso2: Optional[str]
    ortho: tuple
    dem: tuple
    buildings: tuple
    facade_region: str = "eu"
    lefthand: bool = False
    notes: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {"iso2": self.iso2, "ortho": list(self.ortho), "dem": list(self.dem),
                "buildings": list(self.buildings), "facade_region": self.facade_region,
                "lefthand": self.lefthand, "notes": list(self.notes)}


GLOBAL_DEM = ("aws_terrain", "copernicus_aws")
GLOBAL_BUILDINGS = ("overture", "osm")
# country -> (ortho, dem, buildings, facade region); osm_tiles stays last in ortho: the refinement's road mask only
COUNTRIES = {
    "NL": (("pdok_luchtfoto", "osm_tiles"), ("ahn_wcs", "copernicus_aws"), ("3dbag",) + GLOBAL_BUILDINGS, "nl"),
    "ES": (("icgc", "ign_es", "osm_tiles"), ("icgc_mdt2m", "ign_wcs") + GLOBAL_DEM, ("catastro",) + GLOBAL_BUILDINGS,
           "south"),
    "FR": (("ign_fr", "osm_tiles"), ("rge_alti",) + GLOBAL_DEM, ("bdtopo",) + GLOBAL_BUILDINGS, "south"),
    "BE": (("flanders", "osm_tiles"), GLOBAL_DEM, GLOBAL_BUILDINGS, "nl"),
    "CH": (("swissimage", "osm_tiles"), ("swissalti3d",) + GLOBAL_DEM, GLOBAL_BUILDINGS, "eu"),
    "US": (("nys_orthos", "naip", "osm_tiles"), ("usgs_3dep",) + GLOBAL_DEM, ("nyc3d",) + GLOBAL_BUILDINGS, "us"),
    "CA": (("osm_tiles",), GLOBAL_DEM, GLOBAL_BUILDINGS, "us"),
    "FI": (("helsinki_ortho", "mml_ortho", "osm_tiles"), GLOBAL_DEM, GLOBAL_BUILDINGS, "eu"),
    "JP": (("gsi_photo", "osm_tiles"), ("gsi_dem",) + GLOBAL_DEM, ("plateau",) + GLOBAL_BUILDINGS, "jp"),
}


def plan_for(iso2: Optional[str], bbox_swne=None) -> RegionPlan:
    """The data plan for a twin in country ``iso2`` (None: unknown, the global fallbacks). With ``bbox_swne``
    the providers that do not cover it are left out of the lists (osm_tiles and the global ones always stay)."""
    cc = (iso2 or "").upper() or None
    ortho, dem, blds, region = COUNTRIES.get(cc or "", (("osm_tiles",), GLOBAL_DEM, GLOBAL_BUILDINGS, None))
    if region is None:
        region = "south" if cc in SOUTH else "us" if cc in ("US", "CA") else "jp" if cc == "JP" else "eu"
    notes = []
    if bbox_swne is not None:
        ortho = tuple(p for p in ortho if covers(p, bbox_swne))
        dem = tuple(p for p in dem if covers(p, bbox_swne))
    if not any(p != "osm_tiles" for p in ortho):
        notes.append("no open aerial photo here: the ground keeps CARLA's materials and roofs a plain material")
    left = cc in LEFT_HAND
    if left:
        notes.append("left-hand traffic: the lane graph is built in the mirror image and mirrored back "
                     "(twinmodel.lefthand); the OpenDRIVE says rule=LHT, which CARLA 1.0 reads")
    return RegionPlan(cc, tuple(ortho), tuple(dem), tuple(blds), region, left, tuple(notes))


# --------------------------------------------------------------------------- licences and attribution

# provider -> (what, licence, home page); every provider any plan can name is listed (tests check that)
SOURCES = {
    # road network
    "osm": ("OpenStreetMap contributors (roads, crossings, signs, building footprints)", "ODbL 1.0",
            "https://www.openstreetmap.org/copyright"),
    "nominatim": ("Place search: OpenStreetMap Nominatim", "ODbL 1.0", "https://nominatim.org"),
    # aerial photos
    "pdok_luchtfoto": ("PDOK Luchtfoto RGB, Beeldmateriaal Nederland", "CC BY 4.0", "https://www.pdok.nl"),
    "icgc": ("ICGC orthophoto of Catalonia", "CC BY 4.0", "https://www.icgc.cat"),
    "ign_es": ("IGN Spain PNOA maxima actualidad orthophoto (scne.es)", "CC BY 4.0", "https://pnoa.ign.es"),
    "ign_fr": ("IGN France BD ORTHO (Geoplateforme)", "Licence Ouverte 2.0 (Etalab)", "https://geoservices.ign.fr"),
    "swissimage": ("swisstopo SWISSIMAGE 10 cm", "swisstopo open data (free use with attribution)",
                   "https://www.swisstopo.admin.ch"),
    "flanders": ("Digitaal Vlaanderen orthophoto mosaic (OMW)", "Modellicentie Gratis Hergebruik 1.0",
                 "https://www.vlaanderen.be/digitaal-vlaanderen"),
    "naip": ("USDA/USGS NAIP orthoimagery", "public domain", "https://www.usgs.gov"),
    "nys_orthos": ("NYS ITS Geospatial Services orthoimagery", "public domain", "https://orthos.its.ny.gov"),
    "gsi_photo": ("GSI Japan seamless photo tiles (Chiriin tiles)", "GSI terms of use (compatible with CC BY 4.0)",
                  "https://maps.gsi.go.jp/development/ichiran.html"),
    "helsinki_ortho": ("City of Helsinki orthophoto (Helsingin kaupunki, kartta.hel.fi)", "CC BY 4.0",
                       "https://kartta.hel.fi"),
    "mml_ortho": ("National Land Survey of Finland orthophoto (Maanmittauslaitos), via the kartat.kapsi.fi mirror",
                  "CC BY 4.0", "https://www.maanmittauslaitos.fi/en/opendata-licence-cc40"),
    "osm_tiles": ("OpenStreetMap standard tiles (refinement aid only, never a texture)", "ODbL 1.0 / CC BY-SA 2.0",
                  "https://www.openstreetmap.org/copyright"),
    # elevation
    "ahn_wcs": ("AHN DTM 0.5 m via PDOK (Rijkswaterstaat)", "CC0 1.0", "https://www.ahn.nl"),
    "icgc_mdt2m": ("ICGC terrain model 2 m", "CC BY 4.0", "https://www.icgc.cat"),
    "icgc_terr": ("ICGC lidar terrain model 50 cm", "CC BY 4.0", "https://www.icgc.cat"),
    "ign_wcs": ("IGN Spain MDT05 (CNIG)", "CC BY 4.0", "https://www.ign.es"),
    "rge_alti": ("IGN France RGE ALTI 1 m (Geoplateforme)", "Licence Ouverte 2.0 (Etalab)",
                 "https://geoservices.ign.fr"),
    "swissalti3d": ("swisstopo swissALTI3D 0.5 m", "swisstopo open data (free use with attribution)",
                    "https://www.swisstopo.admin.ch"),
    "gsi_dem": ("GSI Japan DEM 5 m / 10 m tiles", "GSI terms of use (compatible with CC BY 4.0)",
                "https://maps.gsi.go.jp/development/ichiran.html"),
    "usgs_3dep": ("USGS 3D Elevation Program (3DEP)", "public domain", "https://www.usgs.gov/3d-elevation-program"),
    "aws_terrain": ("AWS Terrain Tiles (Mapzen/Tilezen: SRTM, 3DEP, EU-DEM, ETOPO1, national lidar, ...)",
                    "open data, attribution per source", "https://github.com/tilezen/joerd/blob/master/docs/attribution.md"),
    "copernicus_aws": ("Copernicus DEM GLO-30 (DLR, Airbus; ESA)", "Copernicus DEM licence (free, attribution)",
                       "https://registry.opendata.aws/copernicus-dem/"),
    "opentopo": ("OpenTopography global DEM (needs a key; skipped without one)", "per dataset",
                 "https://opentopography.org"),
    # buildings
    "3dbag": ("3DBAG LoD 2.2 (TU Delft 3D geoinformation; BAG + AHN)", "CC BY 4.0", "https://3dbag.nl"),
    "nyc3d": ("NYC 3-D Building Model, Building Footprints and PLUTO (NYC OTI, DCP)", "NYC Open Data terms of use",
              "https://opendata.cityofnewyork.us"),
    "plateau": ("Project PLATEAU 3D city models (MLIT Japan)", "PLATEAU terms of use (compatible with CC BY 4.0)",
                "https://www.mlit.go.jp/plateau/"),
    "catastro": ("Direccion General del Catastro, INSPIRE buildings (footprints, floors per part, year, use)",
                 "CC BY 4.0", "https://www.catastro.hacienda.gob.es/webinspire/index.html"),
    "bdtopo": ("IGN France BD TOPO buildings (height, floors, year, use)", "Licence Ouverte 2.0 (Etalab)",
               "https://geoservices.ign.fr/bdtopo"),
    "overture": ("Overture Maps Foundation buildings (OpenStreetMap, Esri Community Maps, Microsoft, Google, "
                 "national agencies)", "ODbL 1.0 (OSM-derived) / CDLA Permissive 2.0",
                 "https://docs.overturemaps.org/attribution/"),
}


def source_line(name: str) -> str:
    what, lic, url = SOURCES.get(name, (name, "see the provider", ""))
    return f"- {what}. Licence: {lic}." + (f" {url}" if url else "")


def used_sources(report: Optional[dict], manifest: Optional[dict]) -> list:
    """Provider names a twin was made from, from its report.json and ue/manifest.json."""
    used = ["osm"]
    rep = report or {}
    b = rep.get("build") if isinstance(rep.get("build"), dict) else rep
    img = (b or {}).get("imagery") or {}
    if img.get("source") and img["source"] != "osm_tiles":
        used.append(img["source"])
    el = (rep.get("elevation") or (b or {}).get("elevation") or {})
    if isinstance(el, dict) and el.get("source"):
        used.append(el["source"])
    st = (manifest or {}).get("stats") or {}
    if st.get("real_buildings"):
        used.append(st["real_buildings"])
    gp = st.get("ground_photo") or {}
    if (gp.get("sources") or {}).get("pdok"):
        used.append("pdok_luchtfoto")
    if gp.get("credit") and any("AHN" in c for c in gp["credit"]):
        used.append("ahn_wcs")
    out = []
    for u in used:
        if u not in out:
            out.append(u)
    return out


def attribution_text(name: str, report: Optional[dict], manifest: Optional[dict], extra: tuple = ()) -> str:
    """The ATTRIBUTION file of a twin (Markdown): every data source it was made from, with its licence."""
    lines = [f"# {name}: data sources and licences", "",
             f"This CARLA level was generated on {_dt.date.today().isoformat()} by the carla-digitaltwins TwinModel "
             "pipeline from open data. Credit these sources when you publish pictures, videos or results made "
             "with it:", ""]
    for s in list(used_sources(report, manifest)) + [e for e in extra if e]:
        lines.append(source_line(s))
    st = (manifest or {}).get("stats") or {}
    if st.get("building_source"):
        lines += ["", f"Buildings: {st['building_source']}."]
    lines += ["", "Facade textures, window lights and road materials are synthetic or CARLA's own (MIT / CC BY)."]
    return "\n".join(lines) + "\n"
