# Third-party assets used by realism.py and city.py

Nothing large is committed. `realism.py` downloads textures, skies and scanned objects on first use
into `~/.cache/jev-drone-realism` (or `$JEV_REALISM_CACHE`). `python realism.py fetch` gets all of them
(about 80 MB unpacked). The only third-party data in the repo is `city_vesterbro.json`.

| what | from | licence | used as |
|---|---|---|---|
| textures (diffuse maps, 1k JPG cached at 512 px PNG) | Poly Haven, https://polyhaven.com (via https://api.polyhaven.com) | CC0 1.0 | floor, wall, beam, facade, roof, bark textures (`realism.TEXTURES`) |
| HDRIs (2k .hdr, tone-mapped into 512 px cube maps) | Poly Haven, https://polyhaven.com | CC0 1.0 | skyboxes (`realism.SKIES`) |
| scanned objects (mesh + texture) | Google Scanned Objects (Google Research), converted to MJCF by kevinzakka/mujoco_scanned_objects, https://github.com/kevinzakka/mujoco_scanned_objects (commit 6ff8d275) | meshes and textures CC-BY 4.0 (Google LLC); the MJCF conversion MIT (c) 2022 Kevin Zakka | clutter and occluders (`realism.OBJECTS`) |
| city footprints, heights, streets, trees (`city_vesterbro.json`) | OpenStreetMap, https://www.openstreetmap.org (via api.openstreetmap.org, bbox 12.54719,55.67096,12.55169,55.67346) | (c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright | the `city` course (city.py) |

The attribution for Google Scanned Objects: "Google Scanned Objects: A High-Quality Dataset of 3D Scanned
Household Items", Downs et al., ICRA 2022, https://app.gazebosim.org/GoogleResearch. The objects are
scaled up to street-furniture size; their textures are only downscaled. The city JSON is a Derivative Database of
OpenStreetMap under the ODbL: projected to metres and simplified (Douglas-Peucker 0.3 m). Any image
rendered from it should credit "(c) OpenStreetMap contributors".
