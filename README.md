# CARLA Digital Twins

Build a drivable CARLA map from OpenStreetMap data.

The command line pipeline in `TwinModel/` is the supported way to make a twin: it runs headless, produces the OpenDRIVE, the meshes and the baked Unreal level, and needs no editor interaction.
An Unreal editor plugin is also included and is documented at the end of this file.

---

## Requirements

* Python 3.11 or newer, with the `twinmodel` package installed from `TwinModel/`
* A CARLA UE 5.8 checkout, built, for the baking stage
* The matching CARLA Python wheel, used to validate the generated OpenDRIVE

```sh
cd TwinModel
pip install -e .
```

---

## The pipeline

A twin is built in three stages. The first two are pure Python and need no Unreal; the third runs inside the editor as a commandlet.

```
OSM  ──1──>  <name>.twin + <name>.xodr  ──2──>  manifest.json + *.glb  ──3──>  Unreal level
      build                              bake-export                    bake_level.py
```

### 1. Build the model

Fetches the OpenStreetMap extract for a bounding box and turns it into the twin model and its OpenDRIVE.

```sh
python -m twinmodel build --bbox <south> <west> <north> <east> --name MyCity --out build/mycity
```

Useful options:

| option | effect |
| --- | --- |
| `--quick` | skip the OBJ mesh and preview PNGs; the twin and the OpenDRIVE only |
| `--no-imagery`, `--no-dem` | skip orthophoto and elevation acquisition |
| `--no-refine` | skip the imagery-driven refinement pass |
| `--fixture <file>` | use a cached Overpass JSON instead of fetching |
| `--profile <name>` | region profile: `nl_urban`, `eu_dense`, `us_urban`, `us_suburban`, `jp_urban`, `auto` (by country, with the country's open data: see [A twin of any place](#a-twin-of-any-place)) |

The stage ends by parsing the result with `carla.Map` and writing `report.json`; a non-zero exit means the OpenDRIVE did not validate.

### 2. Export the bake input

Turns the model into the per-tile geometry the baker imports.

```sh
python -m twinmodel bake-export build/mycity MyCity
```

Writes `build/mycity/ue/manifest.json` and one `.glb` per (layer, kind, tile) under `build/mycity/ue/meshes/`, plus the spawn points.

#### The Netherlands: national open data

`--profile nl_urban` (also chosen by `auto` when the country lookup returns NL) uses the Dutch national datasets
instead of the global ones:

| Layer | Provider | Source |
| --- | --- | --- |
| elevation | `ahn_wcs` | [AHN](https://www.ahn.nl) DTM, 0.5 m, NAP heights, from the PDOK WCS in 1 km tiles (CC0) |
| orthophoto | `pdok_luchtfoto` | [PDOK Luchtfoto](https://www.pdok.nl) RGB, current year, 8 cm, from the WMS (CC BY 4.0) |

After the bake export, the extruded OSM buildings can be swapped for [3DBAG](https://3dbag.nl) LoD 2.2
buildings (BAG footprints with AHN lidar roof shapes, CC BY 4.0); the OSM meshes are kept in
`meshes/osm_buildings/`:

```sh
python TwinModel/tools/threedbag_buildings.py build/mycity
```

The ground (open lots, courtyards, gardens, block interiors and the apron: the `ground` and `groundplane`
assets) can then be textured with the 8 cm aerial photo, one 2048 px texture per 250 m tile (12 cm a pixel),
like the 3DBAG roofs; roads, sidewalks, kerbs and markings keep CARLA's materials. Trees, shelters and other
objects standing more than 2 m above the terrain (AHN 0.5 m surface minus terrain model, CC0) are replaced by
the visible ground around them, since the scenery places the trees in 3D, and cast shadows are brightened.
Outside the Netherlands the twin's cached ortho is used, without the object fill:

```sh
python TwinModel/tools/ground_photo.py build/mycity            # --undo puts CARLA's ground back
```

Central Eindhoven (1.6 x 1.6 km): 2,094 roads, 242 junctions, AHN terrain 12.1-23.9 m NAP, 3,654 3DBAG
buildings (274k triangles); its 64 ground tiles take 4 minutes and 64 MB of JPEG textures, with 9.8 % of the
ground replaced (standing objects) and 2.2 % brightened (shadows). Public Overpass servers often time out on an area this size; the OSM stage then
fetches it in four quarters and merges them.

#### A twin of any place

`python -m twinmodel twin` goes from a place to a baked level in one command, with the best keyless, openly
licensed data of the place's country (`twinmodel/regions.py`; every provider is skipped outside its coverage and
the next one takes over, so a failing or absent service never stops the twin):

```sh
python -m twinmodel twin "Plaza Mayor, Salamanca" --name SalamancaPlaza --size 1.0          # stages 1-2 and the extras
python -m twinmodel twin --bbox 35.6555 139.6955 35.6635 139.7055 --name ShibuyaCrossing \
    --bake --carla-root ~/carla58/carla --engine ~/carla58/UnrealEngine5_carla \
    --carla-python ~/carla58/venv/bin/python --no-nav-build                                # ... and stage 3
```

It finds the place with Nominatim (or takes `--bbox`), runs `build --profile auto`, `bake-export`,
`tools/real_buildings.py`, `tools/ground_photo.py` where the country has an open aerial photo, writes
`ATTRIBUTION.md` (every source with its licence; the bake copies it next to the level's `OpenDrive/` and `TM/`
folders), and with `--bake` runs `bake_level.py` in the CARLA 1.0 editor under a memory watchdog (`--max-rss-gb`,
`--min-avail-gb`, `--min-start-gb`), then (with `--no-nav-build`) the two navigation builders one at a time. Each
step prints one plain line (`[twin] done buildings: 1,221 buildings from catastro, ...`); the logs go to
`<out>/twin.log` and the numbers to `<out>/twin.json`. Steps done before are skipped (`--force` redoes them). The
bake needs a CARLA 1.0 (ue58-dev) source build with the editor; without `--bake` the twin stops at the manifest,
which can be baked later on a machine that has one. Paths can also come from `TWINMODEL_CARLA_ROOT`,
`TWINMODEL_UE_ENGINE`, `TWINMODEL_CARLA_PYTHON` or `~/.config/twinmodel/toolchain.json`.

| Country | Aerial photo (ground, roofs) | Terrain | Buildings |
| --- | --- | --- | --- |
| NL | PDOK Luchtfoto 8 cm (CC BY 4.0) | AHN 0.5 m (CC0) | 3DBAG LoD 2.2 (CC BY 4.0) |
| ES | ICGC 25 cm in Catalonia, else IGN PNOA 25 cm (CC BY 4.0) | ICGC 2 m, IGN MDT05 (CC BY 4.0) | Catastro INSPIRE buildings: floors per part, year, use (CC BY 4.0; not the Basque Country, Navarre) |
| FR | IGN BD ORTHO 20 cm (Etalab 2.0) | IGN RGE ALTI 1 m (Etalab 2.0) | IGN BD TOPO: ground, eaves and ridge heights, floors, year, use (Etalab 2.0) |
| BE | Digitaal Vlaanderen orthophoto 25 cm, Flanders only (Modellicentie Gratis Hergebruik) | global | Overture |
| CH | swisstopo SWISSIMAGE 10 cm | swisstopo swissALTI3D 0.5 m | Overture |
| US | NYS orthos 15 cm in New York State, else USDA NAIP 60 cm (public domain) | USGS 3DEP 1 m (public domain) | NYC 3D model in NYC, else Overture |
| JP | GSI seamless photo (GSI terms, CC BY 4.0 compatible) | GSI DEM 5 m | PLATEAU LoD 2 with its own photo textures (listed wards), else Overture |
| elsewhere | none: CARLA's ground materials, plain roofs | AWS Terrain Tiles, Copernicus GLO-30 | Overture Maps (ODbL / CDLA Permissive 2.0), else OSM |

There is no openly licensed global aerial photo at street scale (ESRI, Google and Bing imagery may not be used;
Sentinel-2's 10 m pixels are too coarse for a street), so outside these countries the ground keeps CARLA's
materials. Overture is read from its GeoParquet release on S3 with DuckDB (`pip install duckdb`, no key); where it
fails the twin's own OSM footprints are used. Footprint models (Catastro, BD TOPO, Overture, OSM) become prisms
with gabled, hipped, pyramidal or skillion roofs where the source gives a roof shape (or BD TOPO a ridge above the
eaves); small homes without one get a pitched roof by the region's habit. Facade styles follow the region:
Dutch and north European brick, southern sandstone (before 1940) and warm render, US pre-war stone, Japanese render.

**Left-hand traffic.** The lane graph is right-hand by construction, so in a left-hand country (`jp_urban`, or any
profile in a country of `regions.LEFT_HAND`) it is built in the mirror image of the place (OSM mirrored about the
frame's central meridian, which the transverse Mercator frame keeps exact, with left and right swapped in side and
turn tags) and mirrored back with its lane ids negated (`twinmodel/lefthand.py`); every road carries
`rule="LHT"`, which CARLA 1.0 reads. On the Shibuya twin CARLA 1.0's map has oncoming traffic on the driver's
right at all 3,288 innermost lanes checked, none on the left. `jp_urban` also draws zebras along OSM's crossing
ways inside junctions (`CrossingRules.from_ways`): the scramble's legs and diagonal.

**PLATEAU's photo textures.** Where a PLATEAU city model has LoD 2 appearances (Shibuya 2023: 98 % of the
surfaces), the textures are taken from the CityGML (`citygml.textures`, the images fetched from the zip by range
requests) and packed into one 4096 px atlas per 250 m tile (`realbuildings.pack_atlas`), so the buildings carry
their own photographed facades instead of a synthetic style (`real_buildings.py --no-source-textures` for the
styles). They come from oblique aerial photos: right in colour and layout, soft at street level.

**Tested (2026-11-15, lin5080, the CARLA 1.0 editor headless with `-nullrhi`):**

| Twin | Data | Roads | Buildings | Python stages | Bake | Content |
| --- | --- | --- | --- | --- | --- | --- |
| `SalamancaPlaza`, 1 km around Plaza Mayor, Salamanca | PNOA 25 cm, MDT05 (781-815 m), Catastro | 568 | 1,221 (5,615 parts, 1,215 with a year), 36 photo ground tiles | 9 min | 4.8 min, peak 7.8 GB; navigation 50 s | 90 MB, 568 actors, 82 textured materials |
| `ShibuyaCrossing`, 0.9 km around the scramble, Tokyo (left-hand) | GSI photo, GSI DEM 5 m (9-36 m), PLATEAU LoD 2 with photo textures | 1,399 | 1,648 (54,501 photo-textured surfaces, 18 atlases), 16 photo ground tiles | 38 min, 5 with the downloads cached | 3.0 min, peak 7.5 GB; navigation 55 s | 117 MB, 354 actors, 54 textured materials |

Both levels load in a CARLA 1.0 server (371 and 400 spawn points, 20 of 20 navigation-mesh queries answered) and
drive with SUMO traffic through Coupled CARLA: 60 s headless trials at RTF 1.16 (Salamanca) and 1.18 (Shibuya), no
collisions. The Shibuya driver (the autopilot) drove the left-hand lanes with oncoming traffic on its right for the
first 34 s, then left the lanes on the multi-level roads by the station (ramps at 32-39 m) and circled on a paved
area outside the OpenDRIVE: Shibuya's stacked roads and plazas need a look. Their looks still need a rendering
server. Coupled CARLA's map maker ran the whole chain for a third place
(`--twin`, Grote Markt in Leuven, 0.5 km, without the bake): the Flemish photo and 849 Overture buildings.

### 3. Bake the Unreal level

Run inside the CARLA editor as a commandlet.

```sh
UnrealEditor-Cmd <CarlaUnreal.uproject> -run=pythonscript \
    -script="<repo>/TwinModel/ue/bake_level.py --manifest <abs>/build/mycity/ue/manifest.json --name MyCity" \
    -unattended -nosplash -nullrhi
```

This imports the meshes with Nanite and complex-as-simple collision, places one actor per asset in a World Partition level, copies the OpenDRIVE next to the map, cooks the Traffic Manager cache, builds the pedestrian navigation, and writes `bake_report.json`.

| option | effect |
| --- | --- |
| `--buildings procedural` | use the modular facade generator instead of the manifest's extruded slabs |
| `--map-root`, `--mesh-root` | destination package paths, for baking into a content pack mount |
| `--skip-import` | reuse meshes already imported |
| `--no-navmesh` | skip the nav bounds volume, the RecastNavMesh and both World Partition navigation builders |
| `--no-nav-build` | place the nav bounds volume and the RecastNavMesh but skip the two builders (each a second editor); run them afterwards with `-run=WorldPartitionBuilderCommandlet -Builder=CarlaCrosswalkNavBuilder` and `-Builder=WorldPartitionNavigationDataBuilder` on the level when memory is short |
| `--no-tm-cache` | skip the Traffic Manager cache |
| `--python-bin` | interpreter holding the CARLA wheel, when the editor's own Python does not |
| `--no-nanite` | import without Nanite |

The level is then loadable in the editor and in a CARLA server.

---

## Shipping the map

Two routes, both verified.

**Cook it into a CARLA package.** Add the map to the cook scope and build the package:

```sh
cmake -S . -B Build/Release -DCARLA_MAPS_TO_COOK=/Game/Carla/Maps/Twins/MyCity/MyCity
cmake --build Build/Release --target package
```

**Ship it as a content pack**, with no repackaging of CARLA:

```sh
carla-pack init MyCityPack
carla-pack add MyCityPack --map /Game/Carla/Maps/Twins/MyCity/MyCity
carla-pack build MyCityPack --base <release-metadata.tar.gz>
carla-pack install MyCityPack-1.0.0-<release>.tar.gz --server /opt/carla/Linux
```

The pack carries the map's OpenDRIVE and Traffic Manager data as sidecar files, so a packaged server picks them up when it mounts the pack.

---

## Project CLI and web editor

For durable map projects with vegetation and street furniture authoring, and build/apply controls:

```sh
python -m twinmodel project --help
python -m twinmodel project edit projects/eixample --port 8791
```

See [project CLI usage](TwinModel/docs/PROJECT_CLI.md) and the [project workflow design](TwinModel/docs/PROJECT_CLI_DESIGN.md).
Unreal apply needs a compatible CARLA UE 5.8 checkout with the native vegetation and furniture tools.
Opening the web editor does not reload or replace the running CARLA map.

---

## Other commands

| command | purpose |
| --- | --- |
| `twinmodel validate <twin_dir> <xodr>` | re-run the OpenDRIVE validation on an existing build |
| `twinmodel compare` | OSM tiles, orthophoto, mesh top view and difference rasters |

---

# Unreal editor plugin

The repository also ships a Digital Twins editor plugin, which generates a map from inside Unreal rather than from the command line.
The rest of this file documents building and using it.

---

# Build Instructions for CARLA Digital Twins Plugin (UE 5)

## Prerequisites

- **Unreal Engine 5 installed**
- **Git installed**
- **Operating System**: Linux or Windows
- **Enable Python Editor Script Plugin**  
  To enable:
  1. Open **Unreal Engine 5**.
  2. Go to **Edit** > **Plugins**.
  3. In the **Scripting** section, find **"Python Editor Script Plugin"**.
  4. Check the box to **enable** it.
  5. Restart the editor when prompted.

---

## Common Setup Steps (Linux & Windows)

1. Create a new Unreal Engine 5 project. It **must** be a blank C++ project. We recommend removing starter content.
2. Close the Unreal Editor.
3. Open your project directory in your file explorer.
4. Inside the project root, create a folder named `Plugins` if it doesn't already exist.
5. Open a terminal (Linux) or CMD/PowerShell (Windows), navigate to the `Plugins` folder, and clone the Digital Twins plugin:

   ```bash
   git clone -b ue5-digitaltwins https://github.com/carla-simulator/carla-digitaltwins.git
   ```

6. Navigate into the cloned folder:

   ```bash
   cd carla-digitaltwins
   ```

7. Run the setup script:

   - **On Linux:**
     ```bash
     ./Setup.sh
     ```
   - **On Windows (on an X64 Native Tools Command Prompt for VS 2022):**
     ```cmd
     Setup.bat
     ```

---

## Linux Build Instructions

1. From the project root directory, run:

   ```bash
   make <ProjectName>Editor
   ```

   > Replace `<ProjectName>` with the actual name of your Unreal project.

2. Once compiled, double click on ProjectName.uproject

### Mitsuba support

We provide a road mesh optimization pipeline using [Mitsuba](https://mitsuba.readthedocs.io/en/stable/), an inverse rendering library integrated in DigitalTwins for road refinement. This option is currently only available for Ubuntu. It requires the use of aerial images, whose processing is provided through the library [GDAL](https://github.com/OSGeo/gdal). To install it, run the following commands:

```sh
sudo apt install libgdal-dev
pip3 install gdal[numpy]=="$(gdal-config --version).*"
```

To enable the Mitsuba road optimization option, enable it in the OpenDriveToMap blueprint:

<img width="311" height="67" alt="mitsuba_flag" src="https://github.com/user-attachments/assets/2de7166c-132d-42d2-a508-3c75ab1597a6" />

Alternatively, set the `bUseMitsuba` flag to true in [OpenDriveToMap.h](https://github.com/carla-simulator/carla-digitaltwins/blob/ue5-digitaltwins/Source/CarlaDigitalTwinsTool/Public/Generation/OpenDriveToMap.h) (line 187):

```cpp
bool bUseMitsuba = true;
```

and compile the project.

---

## Windows Build Instructions

1. Right-click the `.uproject` file and select **"Generate Visual Studio project files"**.
2. Open the generated `.sln` file with **Visual Studio 2022** or any compatible version for UE5.
3. Build the solution using **Development Editor** configuration and **Win64** platform.
4. Once compiled, run the project from Visual Studio (F5) or by double-clicking the `.uproject` file.

---

# How to Use the CARLA Digital Twins Tool in Unreal Engine

## Prerequisite

Make sure the **Digital Twins plugin** is correctly installed and built in your Unreal Engine 5 project, as described in the build instructions.

---

## Steps to Launch the Digital Twins Tool

1. **Open your Unreal Engine project** where the Digital Twins plugin is installed.

2. In the **Content Browser**, go to the bottom-right corner and click the **eye icon** to enable "Show Plugin Content".

   <img width="1022" alt="ContentBrowser" src="https://github.com/user-attachments/assets/acd1f2df-dac6-43ee-a1e4-904e26b9f4ee" />
   <img width="173" alt="ShowPluginContent" src="https://github.com/user-attachments/assets/0cdfd612-2e91-412f-abd6-5509ae2e8b8f" />

3. On the **left panel**, a new section will appear labeled `DigitalTwins Content`. Expand it.

   <img width="239" alt="CarlaDigitalTwinsContent" src="https://github.com/user-attachments/assets/c401ed92-fac0-4c31-94d0-9212ae742e27" />

4. Inside that folder, locate the file named:  
   **`UW_DigitalTwins`**

5. **Right-click** on `UW_DigitalTwins` and select:

   ```
   Run Editor Utility Widget
   ```

   <img width="290" alt="RunUtilityWidget" src="https://github.com/user-attachments/assets/1476d92e-0287-4ca7-b8b6-6bf817fef831" />

---

## Importing a Real Map from OpenStreetMap

6. Once the tool launches, a UI will appear with **three sections**:

   - **Filename** – your custom name for the map
   - **OSM URL** – the URL from OpenStreetMap
   - **LocalFilePath** – optional local saving path

7. Go to [https://www.openstreetmap.org](https://www.openstreetmap.org).

8. Search and zoom into the **area** you want to replicate.

9. Click the **Export** button on the top menu, the one in the upper part of the window which is between other buttons as shown in the following screenshot:
   ![image](https://github.com/user-attachments/assets/e6bbc00b-b30c-48f8-80ab-34a6419b3555)

10. On the left side of the window screen, find the text:
    **“Overpass API”**  
    Right-click the link and select **“Copy link address”**.

    ![image](https://github.com/user-attachments/assets/a51d849a-55e3-49ca-95c8-d96c75692e9d)

11. Go back to the Digital Twins tool in Unreal:

    - Paste the copied URL into the **OSMURL** field.
    - Enter a desired name into the **Filename** field.

12. Click the **Generate** button.

---

## Generation and Preview

13. Unreal Engine might seem frozen — **don't worry**, it's processing the data.

    - You can check the progress in the **Output Log**.

14. Once the generation is complete, click **Play** to explore your generated digital twin of the map inside the Unreal environment.

---

## Importing the Generated Map into Another Unreal Project

Once the map generation process is complete:

1. **Close the Unreal Editor**.

2. Open your **file explorer** and navigate to the `Plugins` folder of the project where you originally generated the map.

3. Inside the `Plugins` folder, you will see a plugin folder named after the **map name** you entered in the Digital Twins UI.

4. **Copy that folder**.

5. Go to the target Unreal project (the one where you want to use the map) and paste the copied folder into its own `Plugins` directory.

6. Launch the target project. Unreal Engine will detect the plugin automatically and compile it if necessary.

> Note: **Make sure the target project is also using the same Unreal Engine version and build to avoid compatibility issues.**
