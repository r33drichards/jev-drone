"""Real-world appearance for the MuJoCo courses, opt-in: textures, a real sky, scanned clutter.

Nothing here changes a scene unless it is asked for. A course gets an appearance two ways:

  * a course-name suffix, "<course>@<spec>", which courses.make (and so run.episode, rover_data,
    flightgif and every modal_laya.py entry point that takes a course name) understands:
        mixed@real   no-climb@real   town@real   city@real   mixed@tex+sky   town@real+neg+s3
  * run.episode(..., appearance=<spec or dict>) / run.py --appearance <spec>, on top of any course.

A spec is '+'-separated tokens (no commas, so it survives modal_laya's --courses a,b,c lists):

    real      = tex + sky + c10 (the preset)
    tex       randomise every scenery material with a Poly Haven texture (CC0) and a neutral tint
    sky       a Poly Haven HDRI (CC0), tone-mapped into a MuJoCo cube-map skybox
    cN        N scanned objects (Google Scanned Objects via mujoco_scanned_objects, CC-BY 4.0) as
              clutter/occluders with collision, placed off the rover's path and off the corridor centre
    clutter   = c10
    neg       hard negatives: also draw red / orange / brown textures and red objects for scenery
              (off by default: the rover's red stays the only red thing in view)
    sun       randomise the sun (directional light) direction and strength (implied by tex)
    sN        appearance seed N (default: the course seed, so every flight seed looks different)

or the same as a dict: {"textures": "polyhaven", "sky": "polyhaven", "clutter": 10, "hard_negatives":
False, "seed": None}. JSON of that dict is accepted wherever a string is.

Assets are fetched on first use into CACHE (~/.cache/jev-drone-realism, outside the repo; override with
$JEV_REALISM_CACHE) from polyhaven.com (api.polyhaven.com / dl.polyhaven.org) and
raw.githubusercontent.com (kevinzakka/mujoco_scanned_objects, pinned commit). `python realism.py fetch`
pre-fetches all of them (~25 MB). Only realism_assets/ATTRIBUTION.md and the small city data are committed.

The Eye (flight.py) segments by geom id, so textures and sky never change what the code sees; clutter
does (it is an obstacle, like any non-floor geom). The floor stays one geom named "floor".
"""
import io, json, os, re, shutil, subprocess, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "realism_assets")
# outside the repo on purpose: modal_laya.py mounts the repo into every container, and ~60 MB of textures
# there would ride along with every launch; a Modal container fetches what it needs on first use instead
CACHE = os.environ.get("JEV_REALISM_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "jev-drone-realism")
UA = "jev-drone-realism/1.0 (MuJoCo drone sim; asset fetch)"
TEX_PX = 512                 # textures are cached at 512 px: the onboard frame is 512x384 at a 110 deg lens
SKY_FACE = 512               # cube face size of the skybox
GSO_COMMIT = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"   # kevinzakka/mujoco_scanned_objects main, 2026-09
GSO_RAW = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/%s/models/%%s/%%s" % GSO_COMMIT

# ------------------------------------------------------------------------------------------------ catalog
# Poly Haven texture ids per pool, chosen from the API listing by category and screened for red: fewer
# than 2% of thumbnail pixels red/orange/brown (hue <= 45 or >= 335 deg, saturation > 0.35), re-checked on
# the downloaded 1k diffuse (red_fraction; anything over RED_MAX is dropped at fetch time).
TEXTURES = {
    "ground": ["aerial_asphalt_01", "asphalt_02", "asphalt_04", "asphalt_06", "clean_asphalt",
               "concrete_pavement_02", "cobblestone_floor_001", "cobblestone_floor_08", "concrete_floor_worn_001",
               "hangar_concrete_floor", "gravel_concrete_03", "granular_concrete", "patterned_cobblestone",
               "gravel_floor_02"],
    "wall": ["beige_wall_001", "blue_plaster_wall", "concrete_wall_001", "concrete_wall_004",
             "concrete_block_wall_02", "concrete_slab_wall", "painted_plaster_wall", "plaster_grey_04",
             "plastered_wall_02", "white_plaster_02", "exterior_wall_cladding_03",
             "concrete_tile_facade", "rectangular_facade_tiles", "corrugated_iron_02"],
    "low": ["blue_metal_plate", "painted_metal_shutter", "worn_shutter", "corrugated_iron", "green_metal_rust",
            "container_side", "concrete_block_wall"],
    "wood": ["wood_planks_grey", "white_planks_clean", "blue_painted_planks", "green_rough_planks",
             "grey_oak_veneer_01", "black_painted_planks", "washed_grey_oak_veneer", "dark_planks"],
    "metal": ["blue_metal_plate", "painted_metal_shutter", "corrugated_iron_03", "worn_corrugated_iron",
              "container_side", "green_metal_rust"],
    "roof": ["bitumen", "grey_roof_01", "grey_roof_tiles_02", "roof_slates_03"],
    "asphalt": ["aerial_asphalt_01", "asphalt_02", "asphalt_04", "asphalt_06", "clean_asphalt"],
    "pavement": ["concrete_pavement_02", "cobblestone_floor_001", "cobblestone_floor_08", "patterned_cobblestone",
                 "hangar_concrete_floor", "granular_concrete"],
    "bark": ["palm_tree_bark", "japanese_sycamore"],
    "leaf": ["moss_wood"],
}
# the hard negatives ("neg"): real red-brown scenery, the confusion Laya v2 showed on the town map
HARD_NEGATIVE_TEXTURES = {
    "ground": ["red_laterite_soil_stones", "terracotta_floor_tiles", "brick_pavement"],
    "wall": ["large_red_bricks", "red_bricks_04", "brick_wall_006", "medieval_red_brick"],
    "low": ["rusty_metal_03", "rusty_painted_metal"],
    "wood": [],
    "metal": ["rusty_metal_03", "rusty_corrugated_iron"],
    "roof": ["clay_roof_tiles_02", "roof_tiles"],
    "asphalt": [],
    "pavement": ["brick_pavement", "terracotta_floor_tiles"],
    "bark": ["bark_brown_01"],
    "leaf": [],
}
# Poly Haven HDRIs: sky-only ("puresky") daylight, no sunsets
SKIES = ["kloofendal_48d_partly_cloudy_puresky", "kloofendal_43d_clear_puresky", "kloofendal_overcast_puresky",
         "qwantani_noon_puresky", "kloppenheim_05_puresky", "syferfontein_18d_clear_puresky"]
# course material -> texture pool; anything unlisted is "wall"; "rover" is never touched; KEEP stay as they are
MATERIAL_POOL = {"grid": "ground", "wall": "wall", "low": "low", "blue": "wall", "yellow": "wall",
                 "roofE": "roof", "roofW": "roof", "roof": "roof", "door": "wood", "fence": "wood",
                 "bus": "metal", "truck": "metal", "cab": "metal", "burnt": "metal", "crate": "wood",
                 "shed": "wood", "bark": "bark", "leaf": "leaf", "facade": "wall", "street": "asphalt",
                 "pavement": "pavement"}
KEEP = {"rover", "window", "cityfloor"}
# metres per texture repeat, per pool (texuniform: the texture keeps its real-world scale on every face)
TILE_M = {"ground": 4.0, "wall": 3.0, "low": 2.0, "wood": 2.0, "metal": 2.5, "roof": 3.0, "bark": 1.0, "leaf": 1.5,
          "asphalt": 4.0, "pavement": 2.5}
# neutral tints (multiplied into the texture): greys, creams, cool and green hues -- never red
TINTS = [(1, 1, 1), (.92, .92, .92), (.8, .8, .82), (1, .97, .88), (.88, .93, 1), (.85, .95, .88), (.95, .95, .8),
         (.75, .8, .9), (.7, .7, .7)]

# Google Scanned Objects (MJCF by kevinzakka/mujoco_scanned_objects): name -> size of the largest side (m)
# once scaled up to street furniture. Dense meshes (> 30k vertices) were left out.
OBJECTS = {
    "Hefty_Waste_Basket_Decorative_Bronze_85_liter": 1.1, "Embark_Lunch_Cooler_Blue": 0.9,
    "Ecoforms_Planter_Pot_GP12AAvocado": 0.9, "Ecoforms_Planter_Pot_QP6Ebony": 1.0,
    "Ecoforms_Pot_Nova_6_Turquoise": 0.9, "Ecoforms_Garden_Pot_GP16ATurquois": 0.9,
    "Down_To_Earth_Orchid_Pot_Ceramic_Lime": 0.8, "3D_Dollhouse_Refrigerator": 1.8, "3D_Dollhouse_Sink": 1.1,
    "3D_Dollhouse_Lamp": 1.9, "Spritz_Easter_Basket_Plastic_Teal": 0.9, "RJ_Rabbit_Easter_Basket_Blue": 0.9,
    "Jansport_School_Backpack_Blue_Streak": 1.0, "Schleich_African_Black_Rhino": 1.9, "Elephant": 2.2,
    "Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black": 1.0,
    "JBL_Charge_Speaker_portable_wireless_wired_Green": 1.0, "Threshold_Porcelain_Pitcher_White": 1.0,
    "Granimals_20_Wooden_ABC_Blocks_Wagon": 1.6, "Great_Dinos_Triceratops_Toy": 2.2, "Dino_3": 2.0, "Dino_5": 2.0,
}
# red / orange / brown by red_fraction (> RED_MAX, measured on the fetched textures): only drawn with "neg"
HARD_NEGATIVE_OBJECTS = {"FIRE_TRUCK": 2.6, "Sapota_Threshold_4_Ceramic_Round_Planter_Red": 0.9,
                         "Down_To_Earth_Orchid_Pot_Ceramic_Red": 0.8, "Cole_Hardware_Flower_Pot_1025": 1.0,
                         "SCHOOL_BUS": 3.2, "Sonny_School_Bus": 2.4, "Vtech_Cruise_Learn_Car_25_Years": 2.2,
                         "Design_Ideas_Drawer_Store_Organizer": 1.4, "Central_Garden_Flower_Pot_Goo_425": 0.8,
                         "Ecoforms_Plant_Pot_GP9_SAND": 0.8, "3D_Dollhouse_Sofa": 2.0, "Dino_4": 2.0,
                         "Schleich_Spinosaurus_Action_Figure": 2.4}
RED_MAX = 0.03               # most red pixels a non-negative texture/object may have (red_fraction)
PRESETS = {"real": {"textures": "polyhaven", "sky": "polyhaven", "clutter": 10, "sun": True}}
LAYOUT = ".U..LFRB.D.."      # MuJoCo skybox gridlayout for our 4x3 cross (faces checked by rendering)


# ------------------------------------------------------------------------------------------ colour checks
def _hsv(rgb):
    a = np.asarray(rgb, dtype=float)[..., :3] / 255.0
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0.0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.where(mx == r, ((g - b) / np.maximum(d, 1e-9)) % 6, np.where(mx == g, (b - r) / np.maximum(d, 1e-9) + 2,
                                                                   (r - g) / np.maximum(d, 1e-9) + 4)) * 60.0
    return np.where(d > 0, h, 0.0), s, mx


def red_fraction(rgb):
    """Share of pixels that are red, orange or brown (hue <= 45 or >= 335 deg, saturation > 0.35, value > 0.15):
    the colours Laya could mistake for the rover (0.95, 0.15, 0.10)."""
    h, s, v = _hsv(rgb)
    return float((((h <= 45) | (h >= 335)) & (s > 0.35) & (v > 0.15)).mean())


def rover_like(rgb):
    """Mask of pixels coloured like the rover as rendered (hue within 15 deg of red, saturated, not dark)."""
    h, s, v = _hsv(rgb)
    return ((h <= 15) | (h >= 345)) & (s > 0.55) & (v > 0.25)


# --------------------------------------------------------------------------------------------- fetching
def _download(url, path):
    """url -> path (atomic). urllib with a User-Agent (Poly Haven's CDN refuses Python's default); curl if that
    fails (e.g. a proxy only curl is set up for)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        rc = subprocess.run(["curl", "-sSfL", "-A", UA, "--max-time", "300", "-o", tmp, url]).returncode \
            if shutil.which("curl") else 1
        if rc != 0:
            raise RuntimeError("could not download %s (%r)" % (url, e))
    os.replace(tmp, path)
    return path


def _json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception:
        out = subprocess.run(["curl", "-sSfL", "-A", UA, "--max-time", "60", url], capture_output=True, check=True)
        return json.loads(out.stdout)


def _meta_path(kind, name):
    return os.path.join(CACHE, kind, name + ".json")


def texture(tid):
    """Cached 512 px PNG of Poly Haven texture `tid`'s diffuse map, and its metadata (red_fraction)."""
    png = os.path.join(CACHE, "textures", "%s_%d.png" % (tid, TEX_PX))
    meta = _meta_path("textures", tid)
    if not (os.path.exists(png) and os.path.exists(meta)):
        from PIL import Image
        files = _json("https://api.polyhaven.com/files/" + tid)
        url = files["Diffuse"]["1k"]["jpg"]["url"]
        raw = _download(url, os.path.join(CACHE, "textures", "raw", tid + "_1k.jpg"))
        im = Image.open(raw).convert("RGB").resize((TEX_PX, TEX_PX), Image.LANCZOS)
        im.save(png)
        os.remove(raw)
        json.dump({"id": tid, "source": url, "licence": "CC0 (polyhaven.com)",
                   "red_fraction": red_fraction(np.asarray(im))}, open(meta, "w"))
    return png, json.load(open(meta))


FACADE_STYLES = [  # (window x0, x1, y0, y1 as fractions of a bay / storey, frame rgb)
    (0.28, 0.72, 0.25, 0.78, (215, 215, 210)), (0.22, 0.78, 0.22, 0.72, (60, 60, 62)),
    (0.33, 0.67, 0.20, 0.80, (235, 232, 220))]


def facade(tid, style=0):
    """Cached facade texture: texture `tid` as one window bay (BAY_W x one storey, city.py) with a window drawn
    on: dark blue-grey glass, a frame and a sill (no red)."""
    png = os.path.join(CACHE, "facades", "%s_w%d.png" % (tid, style))
    if not os.path.exists(png):
        from PIL import Image, ImageDraw
        base, _ = texture(tid)
        im = Image.open(base).convert("RGB")
        n = im.size[0]
        x0, x1, y0, y1, frame = FACADE_STYLES[style % len(FACADE_STYLES)]
        dr = ImageDraw.Draw(im)
        box = [int(x0 * n), int(y0 * n), int(x1 * n), int(y1 * n)]
        dr.rectangle([box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10], fill=frame)
        dr.rectangle(box, fill=(48, 58, 70))
        mid = (box[0] + box[2]) // 2
        dr.rectangle([mid - 5, box[1], mid + 5, box[3]], fill=frame)              # mullion
        dr.rectangle([box[0], (box[1] + box[3]) // 2 - 4, box[2], (box[1] + box[3]) // 2 + 4], fill=frame)
        g = np.asarray(im).astype(float)                     # a faint sky reflection in the upper panes
        yy = np.arange(n)[:, None]
        pane = (yy > box[1]) & (yy < (box[1] + box[3]) // 2)
        g[box[1]:box[3], box[0]:box[2]] += np.where(pane[box[1]:box[3]], 18.0, 0.0)[..., None]
        im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        dr.rectangle([box[0] - 16, box[3] + 10, box[2] + 16, box[3] + 22], fill=(190, 190, 185))   # sill
        os.makedirs(os.path.dirname(png), exist_ok=True)
        im.save(png)
    return png


def _read_hdr(path):
    """Radiance .hdr (RGBE, new-style RLE) -> float32 (H, W, 3)."""
    data = open(path, "rb").read()
    i = data.index(b"\n\n") + 2
    j = data.index(b"\n", i)
    dims = data[i:j].split()
    H, W = int(dims[1]), int(dims[3])
    p = j + 1
    out = np.zeros((H, W, 4), np.uint8)
    buf = np.frombuffer(data, np.uint8)
    for y in range(H):
        if buf[p] == 2 and buf[p + 1] == 2 and (int(buf[p + 2]) << 8 | int(buf[p + 3])) == W:
            p += 4
            for c in range(4):
                x = 0
                while x < W:
                    n = int(buf[p]); p += 1
                    if n > 128:
                        n -= 128
                        out[y, x:x + n, c] = buf[p]; p += 1
                    else:
                        out[y, x:x + n, c] = buf[p:p + n]; p += n
                    x += n
        else:                               # flat scanline
            out[y] = buf[p:p + 4 * W].reshape(W, 4); p += 4 * W
    e = out[..., 3].astype(np.float32)
    scale = np.where(out[..., 3] > 0, np.ldexp(1.0, (e - 136).astype(np.int32)), 0.0)
    return out[..., :3].astype(np.float32) * scale[..., None]


def _cube_dirs(face, n):
    """Unit view directions for every pixel of a skybox face, as MuJoCo shows them (checked by rendering a
    direction-coded cube map and sampling the centre pixel of views in 10 directions)."""
    g = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(g, g)
    one = np.ones_like(u)
    x, y, z = {"R": (one, u, -v), "B": (-u, one, -v), "L": (-one, -u, -v), "F": (u, -one, -v),
               "U": (u, -v, one), "D": (u, v, -one)}[face]
    d = np.stack([x, y, z], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def sky(hid):
    """Cached skybox PNG (4x3 cross, LAYOUT) made from Poly Haven HDRI `hid` (2k .hdr): exposure set so the
    sky's median is mid-grey, Reinhard tone map, gamma 2.2; the equirectangular map sampled per cube face."""
    png = os.path.join(CACHE, "sky", "%s_cube%d.png" % (hid, SKY_FACE))
    if not os.path.exists(png):
        from PIL import Image
        files = _json("https://api.polyhaven.com/files/" + hid)
        url = files["hdri"]["2k"]["hdr"]["url"]
        raw = _download(url, os.path.join(CACHE, "sky", "raw", hid + "_2k.hdr"))
        eq = _read_hdr(raw)
        H, W, _ = eq.shape
        lum = eq @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        eq = eq * (0.45 / max(float(np.median(lum[: H // 2])), 1e-6))      # upper hemisphere median -> 0.45
        eq = eq / (1.0 + eq)
        eq = np.clip(eq, 0, 1) ** (1 / 2.2)
        cross = np.zeros((3 * SKY_FACE, 4 * SKY_FACE, 3), np.float32)
        for k, ch in enumerate(LAYOUT):
            if ch == ".":
                continue
            d = _cube_dirs(ch, SKY_FACE)
            lon = np.arctan2(d[..., 1], d[..., 0])            # azimuth, +y = 90 deg
            lat = np.arcsin(np.clip(d[..., 2], -1, 1))
            # equirectangular: azimuth 0 at the centre column, increasing to the left (a sky has no preferred
            # heading, so which world axis the centre faces does not matter; up and handedness do)
            px = ((0.5 - lon / (2 * np.pi)) % 1.0) * W
            py = (0.5 - lat / np.pi) * H
            xi = np.clip(px.astype(int), 0, W - 1)
            yi = np.clip(py.astype(int), 0, H - 1)
            r, c = divmod(k, 4)
            cross[r * SKY_FACE:(r + 1) * SKY_FACE, c * SKY_FACE:(c + 1) * SKY_FACE] = eq[yi, xi]
        Image.fromarray((cross * 255).astype(np.uint8)).save(png, optimize=True)
        json.dump({"id": hid, "source": url, "licence": "CC0 (polyhaven.com)"}, open(_meta_path("sky", hid), "w"))
        os.remove(raw)
    return png


def scanned_object(name):
    """Cached (obj path, texture path, metadata) of a Google Scanned Object from mujoco_scanned_objects.
    metadata: bbox (min, max) in the object's own metres (z up, base at z = 0) and red_fraction of its texture."""
    d = os.path.join(CACHE, "objects", name)
    obj, tex, meta = os.path.join(d, "model.obj"), os.path.join(d, "texture_%d.png" % TEX_PX), os.path.join(d, "meta.json")
    if not (os.path.exists(obj) and os.path.exists(tex) and os.path.exists(meta)):
        from PIL import Image
        _download(GSO_RAW % (name, "model.obj"), obj)
        raw = _download(GSO_RAW % (name, "texture.png"), os.path.join(d, "texture.png"))
        im = Image.open(raw).convert("RGB")
        im.resize((TEX_PX, TEX_PX), Image.LANCZOS).save(tex)
        os.remove(raw)
        v = np.array([list(map(float, ln.split()[1:4])) for ln in open(obj) if ln.startswith("v ")])
        json.dump({"name": name, "bbox": [v.min(0).tolist(), v.max(0).tolist()], "n_vertices": len(v),
                   "red_fraction": red_fraction(np.asarray(im.resize((128, 128)))),
                   "source": GSO_RAW % (name, ""), "licence": "CC-BY 4.0 (Google Scanned Objects); MJCF MIT"},
                  open(meta, "w"))
    return obj, tex, json.load(open(meta))


def decimated(obj, cells=22):
    """A light copy of a scanned object's mesh (model_lo.obj next to it, made once): vertex clustering on a grid
    of `cells` along the largest side. Positions in a cell merge to their mean; each face corner keeps its own
    texture coordinate (OBJ's separate v/vt indices), so the texture still lands where it did; faces that
    collapse are dropped. ~10x fewer faces: the software renderer draws every one of them every frame."""
    out = obj[:-4] + "_lo%d.obj" % cells
    if os.path.exists(out):
        return out
    V, VT, F = [], [], []
    for ln in open(obj):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("vt "):
            VT.append(ln.split()[1:3])
        elif ln.startswith("f "):
            F.append([tuple(int(i) if i else 0 for i in (c.split("/") + ["", ""])[:2]) for c in ln.split()[1:]])
    V = np.array(V)
    lo, hi = V.min(0), V.max(0)
    h = float((hi - lo).max()) / cells
    key = np.floor((V - lo) / h).astype(np.int64)
    _, cid, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    cid = cid.ravel()
    P = np.zeros((counts.size, 3))
    np.add.at(P, cid, V)
    P /= counts[:, None]
    tris = set()
    lines = []
    for f in F:
        for k in range(1, len(f) - 1):
            c = (f[0], f[k], f[k + 1])
            ids = tuple(int(cid[v - 1]) for v, _ in c)
            if len(set(ids)) < 3 or tuple(sorted(ids)) in tris:
                continue
            tris.add(tuple(sorted(ids)))
            lines.append("f " + " ".join("%d/%d" % (i + 1, t) for i, (_, t) in zip(ids, c)))
    with open(out + ".part", "w") as fh:
        fh.write("".join("v %.5f %.5f %.5f\n" % tuple(p) for p in P))
        fh.write("".join("vt %s %s\n" % tuple(t) for t in VT))
        fh.write("\n".join(lines) + "\n")
    os.replace(out + ".part", out)
    return out


def fetch(verbose=True, negatives=True):
    """Pre-fetch everything the catalogue names (what `apply` would otherwise fetch on first use)."""
    ok, bad = [], []
    jobs = [("texture", t) for pool in TEXTURES.values() for t in pool]
    if negatives:
        jobs += [("texture", t) for pool in HARD_NEGATIVE_TEXTURES.values() for t in pool]
        jobs += [("object", o) for o in HARD_NEGATIVE_OBJECTS]
    jobs += [("sky", s) for s in SKIES] + [("object", o) for o in OBJECTS]
    for kind, name in dict.fromkeys(jobs):
        try:
            {"texture": texture, "sky": sky, "object": scanned_object}[kind](name)
            ok.append(name)
            if verbose:
                print("ok  ", kind, name, flush=True)
        except Exception as e:
            bad.append((name, repr(e)[:200]))
            if verbose:
                print("FAIL", kind, name, repr(e)[:200], flush=True)
    return ok, bad


# ------------------------------------------------------------------------------------------------- specs
def parse(spec, seed=0):
    """A spec (None, dict, JSON string or '+'-token string; module docstring) -> a normalised dict, or None
    for 'off'. `seed` (the course seed) is the appearance seed unless the spec names one."""
    if spec is None or spec is False or spec == "" or spec == "off" or spec == "none":
        return None
    if isinstance(spec, str) and spec.strip().startswith("{"):
        spec = json.loads(spec)
    out = {"textures": None, "sky": None, "clutter": 0, "hard_negatives": False, "sun": False, "seed": None}
    if isinstance(spec, dict):
        out.update({k: v for k, v in spec.items() if k in out})
    else:
        for tok in str(spec).split("+"):
            tok = tok.strip()
            if not tok:
                continue
            if tok in PRESETS:
                out.update(PRESETS[tok])
            elif tok in ("tex", "textures"):
                out["textures"], out["sun"] = "polyhaven", True
            elif tok == "sky":
                out["sky"] = "polyhaven"
            elif tok == "sun":
                out["sun"] = True
            elif tok == "neg":
                out["hard_negatives"] = True
            elif tok == "clutter":
                out["clutter"] = 10
            elif re.fullmatch(r"c\d+", tok):
                out["clutter"] = int(tok[1:])
            elif re.fullmatch(r"s\d+", tok):
                out["seed"] = int(tok[1:])
            else:
                raise ValueError("unknown appearance token %r in %r (see realism.py)" % (tok, spec))
    if out["seed"] is None:
        out["seed"] = int(seed)
    return out


def tag(spec):
    """A short file-name-safe tag for a spec: the token string itself, or a hash of a dict / JSON spec."""
    if isinstance(spec, str) and re.fullmatch(r"[A-Za-z0-9+]+", spec):
        return spec
    import hashlib
    return "a" + hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]


def split_name(name):
    """'mixed@real' -> ('mixed', 'real'); 'mixed' -> ('mixed', None)."""
    base, _, spec = name.partition("@")
    return base, (spec or None)


def make(name, seed=0):
    """courses.make for '<course>@<spec>': the base course with its appearance set (xml() applies it)."""
    base, spec = split_name(name)
    import courses
    c = courses.make(base, seed)
    set_appearance(c, spec, seed)
    c.name = name                   # its own .course_<name>_<seed>.xml, so plain and real never share a file
    return c


def set_appearance(course, spec, seed=None):
    """Give `course` an appearance. A course with a base_appearance (city.py: always textured, with a sky)
    keeps it, with the spec's settings on top."""
    seed = course.seed if seed is None else seed
    app = parse(spec, seed)
    base = getattr(course, "base_appearance", None)
    if base:
        merged = parse(dict(base), seed)
        for k, v in (app or {}).items():
            if v and k != "seed":
                merged[k] = v
        merged["seed"] = app["seed"] if app else seed
        app = merged
    course.appearance = app
    return course


# ------------------------------------------------------------------------------------------------ apply
_MAT = re.compile(r'( *)<material name="([^"]+)"([^>]*)/>\n')


def apply(xml, course):
    """The course's MJCF with its appearance (course.appearance, a parse() dict) applied. None: unchanged."""
    app = getattr(course, "appearance", None)
    if not app:
        return xml
    rng = np.random.default_rng([int(app["seed"]), 9173])
    neg = bool(app.get("hard_negatives"))
    extra_assets = []
    textured = {}                      # material -> metres per texture repeat, for _boxes_to_meshes
    if app.get("textures"):
        used = {}
        def pick(pool):
            cands = list(TEXTURES.get(pool, TEXTURES["wall"]))
            if neg and HARD_NEGATIVE_TEXTURES.get(pool) and rng.random() < 0.35:
                cands = list(HARD_NEGATIVE_TEXTURES[pool])
            fresh = [t for t in cands if t not in used.get(pool, ())] or cands
            for _ in range(len(fresh)):
                t = fresh[int(rng.integers(len(fresh)))]
                try:
                    png, meta = texture(t)
                except Exception:
                    fresh = [x for x in fresh if x != t] or cands
                    continue
                if meta["red_fraction"] > RED_MAX and t not in HARD_NEGATIVE_TEXTURES.get(pool, ()):
                    fresh = [x for x in fresh if x != t] or cands
                    continue
                used.setdefault(pool, set()).add(t)
                return t, png
            return None, None

        def sub(mo):
            ind, name, rest = mo.group(1), mo.group(2), mo.group(3)
            if name in KEEP or name.startswith("keep_"):
                return mo.group(0)
            uv = name.startswith("uv_")        # a mesh with texture coordinates (city.py): a 2d texture
            base = re.sub(r"\d+$", "", name[3:] if uv else name)
            pool = MATERIAL_POOL.get(name, MATERIAL_POOL.get(base, "wall"))
            tid, png = pick(pool)
            if tid is None:
                return mo.group(0)
            tint = TINTS[int(rng.integers(len(TINTS)))] if pool not in ("bark", "leaf") else (1, 1, 1)
            rep = 1.0 / TILE_M.get(pool, 3.0)
            # MuJoCo maps a 2d texture on a primitive by its x-y coordinates, which streaks vertical faces; boxes,
            # cylinders and spheres get the same image as a cube texture instead. The floor is a plane: 2d.
            if uv and base == "facade":       # city facades: the wall texture with a window per bay and storey
                png = facade(tid, int(rng.integers(3)))
            kind = "2d" if (uv or name in ("grid", "street")) else "cube"
            if kind == "cube":                # boxes become textured meshes below (2d); others keep the cube
                extra_assets.append('%s<texture name="rt2_%s" type="2d" file="%s"/>\n' % (ind, name, png))
                extra_assets.append('%s<material name="%s_uv" texture="rt2_%s" rgba="%.2f %.2f %.2f 1" '
                                    'reflectance="0" specular="0.1"/>\n' % (ind, name, name, tint[0], tint[1], tint[2]))
            if uv:
                rep = 1.0                     # the mesh's texture coordinates are already in texture repeats
            else:
                textured[name] = TILE_M.get(pool, 3.0)
            extra_assets.append('%s<texture name="rt_%s" type="%s" file="%s"/>\n' % (ind, name, kind, png))
            return ('%s<material name="%s" texture="rt_%s" texrepeat="%.3f %.3f" texuniform="%s" '
                    'rgba="%.2f %.2f %.2f 1" reflectance="0" specular="0.1"/>\n'
                    % (ind, name, name, rep, rep, "false" if uv else "true", tint[0], tint[1], tint[2]))
        xml = _MAT.sub(sub, xml)
        xml, meshes = _boxes_to_meshes(xml, textured)
        extra_assets += meshes
    if app.get("sky"):
        hid = SKIES[int(rng.integers(len(SKIES)))]
        try:
            png = sky(hid)
            xml = re.sub(r'<texture type="skybox"[^>]*/>',
                         '<texture type="skybox" file="%s" gridsize="3 4" gridlayout="%s"/>' % (png, LAYOUT), xml, count=1)
        except Exception as e:
            print("realism: sky %s unavailable (%r); keeping the gradient" % (hid, e), file=sys.stderr)
    if app.get("sun"):
        el = np.deg2rad(rng.uniform(45, 80))
        az = rng.uniform(0, 2 * np.pi)
        dvec = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        lvl = rng.uniform(0.6, 0.85)
        # a dimmer headlight so the sun's shading and shadows show (the plain scenes light everything from the camera)
        amb = rng.uniform(0.3, 0.4)
        xml = re.sub(r'<headlight diffuse="[^"]+" ambient="[^"]+"/>',
                     '<headlight diffuse=".3 .3 .3" ambient="%.2f %.2f %.2f"/>' % (amb, amb, amb), xml, count=1)
        xml = re.sub(r'<light pos="([^"]+)" dir="0 0 -1" directional="true" diffuse="[^"]+"/>',
                     lambda mo: '<light pos="%s" dir="%.3f %.3f %.3f" directional="true" diffuse="%.2f %.2f %.2f"/>'
                     % (mo.group(1), dvec[0], dvec[1], dvec[2], lvl, lvl, lvl * 0.97), xml, count=1)
    n = int(app.get("clutter") or 0)
    geoms = []
    if n > 0:
        assets, geoms = _clutter(course, n, rng, neg)
        extra_assets += assets
    if extra_assets:
        xml = xml.replace("  </asset>\n", "".join(extra_assets) + "  </asset>\n", 1)
    if geoms:
        k = xml.index('    <body name="rover"')
        xml = xml[:k] + "".join(geoms) + xml[k:]
    return xml


_BOX = re.compile(r'<geom name="([^"]+)" type="box" material="([^"]+)" size="([^"]+)" pos="([^"]+)"/>')


def _box_mesh(name, sx, sy, sz, tile):
    """An inline MJCF mesh of a box of half-sizes (sx, sy, sz), 4 vertices per face so every face has its own
    texture coordinates, in metres / tile (the texture keeps its real-world scale on every face)."""
    V, T, F = [], [], []
    for ax in range(3):
        for sgn in (1, -1):
            a, b = [k for k in range(3) if k != ax]
            h = np.array([sx, sy, sz])
            quad = []
            for ua, ub in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[ax], p[a], p[b] = sgn * h[ax], ua * h[a], ub * h[b]
                quad.append(p)
                T.append(((p[a] + h[a]) / tile, (p[b] + h[b]) / tile))
            n0 = len(V)
            V += quad
            # outward winding: (a x b) points along +ax for a cyclic (ax, a, b); flip otherwise
            cyc = (a - ax) % 3 == 1
            f1, f2 = (n0, n0 + 1, n0 + 2), (n0, n0 + 2, n0 + 3)
            if (sgn > 0) != cyc:
                f1, f2 = f1[::-1], f2[::-1]
            F += [f1, f2]
    fmt = lambda a: " ".join("%.4g" % x for x in np.ravel(a))  # noqa: E731
    return '    <mesh name="%s" vertex="%s" texcoord="%s" face="%s"/>\n' % (
        name, fmt(V), fmt(T), " ".join("%d" % i for f in F for i in f))


def _boxes_to_meshes(xml, textured):
    """Every box geom with a textured material -> seen as the same box as a mesh with texture coordinates (a 2d
    texture on a primitive streaks: MuJoCo maps it by the geom's x-y). The box itself stays where it is, with
    its name and index, hidden (group 3): contacts, and so flights, are exactly the plain course's. The meshes
    (no collision) go at the end of the world, just before the rover."""
    meshes, visuals = [], []

    def sub(mo):
        name, mat, size, pos = mo.groups()
        if mat not in textured:
            return mo.group(0)
        sx, sy, sz = map(float, size.split())
        meshes.append(_box_mesh("rbox_" + name, sx, sy, sz, textured[mat]))
        visuals.append('    <geom name="%s_vis" type="mesh" mesh="rbox_%s" material="%s_uv" pos="%s" contype="0" '
                       'conaffinity="0"/>\n' % (name, name, mat, pos))
        return mo.group(0)[:-2] + ' group="3" rgba="0 0 0 0"/>'
    xml = _BOX.sub(sub, xml)
    if visuals:
        k = xml.index('    <body name="rover"')
        xml = xml[:k] + "".join(visuals) + xml[k:]
    return xml, meshes


# ----------------------------------------------------------------------------------------------- clutter
def _path_points(course):
    """The rover's path as (N, 2) points (looped courses: their loop; corridors: rover_pose over the run)."""
    if getattr(course, "looped", False):
        return np.asarray(course.pts)[::4]
    import courses
    ts = np.arange(0.0, (course.end_x + 10 - courses.START_X) / courses.ROVER_SPEED, 0.25)
    return np.array([course.rover_pose(t)[:2] for t in ts])


def _region(course):
    """(bounds (x0, x1, y0, y1), ok(x, y, r) -> bool) where clutter of radius r may stand."""
    if hasattr(course, "clutter_region"):              # city.py
        return course.clutter_region()
    P = _path_points(course)
    if getattr(course, "looped", False):             # town.py
        import town
        boxes = course.boxes()
        trees = course.trees()

        def ok(x, y, r):
            if np.hypot(P[:, 0] - x, P[:, 1] - y).min() < 2.5 + r:
                return False
            for _, x0, x1, y0, y1, z0, z1, _m in boxes:
                if x0 - r - 0.4 < x < x1 + r + 0.4 and y0 - r - 0.4 < y < y1 + r + 0.4:
                    return False
            return all(np.hypot(x - tx, y - ty) > tr + r + 0.8 for _, tx, ty, tr, _cz, cr in trees)
        return (-town.WALL_X + 0.5, town.WALL_X - 0.5, -town.WALL_Y + 0.5, town.WALL_Y - 0.5), ok
    import courses                                      # a corridor (courses.Course)
    zones = []
    for kind, x, _side in course.stations:
        # the station plus most of the approach where the drone swings out toward a gap (decoy: 8 m before, the
        # rover leaves the centre line 5 m before; pocket: 8 m before, 5 m after)
        zones.append({"beam": (x - 4.0, x + 3.0), "pocket": (x - 8.0, x + courses.POCKET_D + 5.0),
                      "decoy": (x - 8.0, x + 9.0)}[kind])

    # dead ends nobody has to fly through: the pocket without the rover's hatches, and the decoy's blind recess
    dead = []
    for kind, x, side in course.stations:
        if kind == "pocket":           # the rover drives through the pocket on side -side; this one is on +side
            ys = sorted((side * (courses.LANE_W / 2 + 0.2), side * (courses.HALF_W - 0.1)))
            dead.append((x + 0.2, x + courses.POCKET_D - 0.4, ys[0], ys[1]))
        elif kind == "decoy":
            ys = sorted((-side * (courses.HALF_W - 0.4), -side * (courses.HALF_W - 0.4 - courses.GAP_W)))
            dead.append((x + 0.2, x + 7.0, ys[0], ys[1]))

    def ok(x, y, r):
        if np.hypot(P[:, 0] - x, P[:, 1] - y).min() < 2.5 + r:
            return False
        for a0, a1, b0, b1 in dead:
            if a0 + r + 0.3 <= x <= a1 - r - 0.3 and b0 + r + 0.3 <= y <= b1 - r - 0.3:
                return True
        if abs(y) < 2.8 + r or abs(y) > courses.HALF_W - 0.15 - r:     # off the centre line, inside the walls
            return False
        return not any(a - r <= x <= b + r for a, b in zones)
    return (4.0, course.end_x + 8.0, -courses.HALF_W, courses.HALF_W), ok


def _clutter(course, n, rng, neg):
    """Up to n scanned objects placed where _region allows, 0.6 m apart: (asset lines, geom lines)."""
    names = list(OBJECTS) + (list(HARD_NEGATIVE_OBJECTS) if neg else [])
    (x0, x1, y0, y1), ok = _region(course)
    placed, assets, geoms, mesh_done = [], [], [], set()
    tries = 0
    while len(placed) < n and tries < 400 * n:
        tries += 1
        name = names[int(rng.integers(len(names)))]
        try:
            obj, tex, meta = scanned_object(name)
        except Exception:
            names.remove(name)
            if not names:
                break
            continue
        if meta["red_fraction"] > RED_MAX and name not in HARD_NEGATIVE_OBJECTS:
            continue
        lo, hi = np.array(meta["bbox"][0]), np.array(meta["bbox"][1])
        size = OBJECTS.get(name) or HARD_NEGATIVE_OBJECTS[name]
        s = size * rng.uniform(0.85, 1.15) / float((hi - lo).max())
        r = float(np.hypot(*(hi - lo)[:2])) * s / 2
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if not ok(x, y, r) or any(np.hypot(x - px, y - py) < r + pr + 0.6 for px, py, pr in placed):
            continue
        k = len(placed)
        placed.append((x, y, r))
        mname = "gso_%d" % k
        assets.append('    <texture name="%s_t" type="2d" file="%s"/>\n'
                      '    <material name="keep_%s" texture="%s_t" specular="0.2"/>\n'
                      '    <mesh name="%s" file="%s" scale="%.4f %.4f %.4f"/>\n'
                      % (mname, tex, mname, mname, mname, decimated(obj), s, s, s))
        # the mesh is re-centred by MuJoCo on compile and the geom frame moved to match, so pos is where the
        # object's own origin (centre of its base) lands: on the floor
        geoms.append('    <geom name="clutter%d" type="mesh" mesh="%s" material="keep_%s" pos="%.3f %.3f 0" '
                     'euler="0 0 %.1f"/>\n' % (k, mname, mname, x, y, rng.uniform(0, 360)))
    course.clutter = placed                 # (x, y, radius) of each object, for maps and checks
    return assets, geoms


# ------------------------------------------------------------------------------------------ frames, checks
def onboard(name, seed=0, times=(5.0, 30.0, 55.0, 80.0), back=4.0, size=(512, 384)):
    """Onboard camera frames (the RGB frame Laya sees: flight.Eye's lens) from `back` m behind the rover along its
    path, at rover time `times`. Returns [(rgb, rover mask, seconds to render)], the mask from a segmentation
    render at the same size."""
    import time
    import mujoco, courses, flight
    c = courses.make(name, seed)
    m = mujoco.MjModel.from_xml_path(c.write(HERE))
    d = mujoco.MjData(m)
    eye = flight.Eye(m, rgb_size=size)
    seg = mujoco.Renderer(m, size[1], size[0])
    seg.enable_segmentation_rendering()
    rover = np.nonzero(m.geom_bodyid == m.body("rover").id)[0]
    out = []
    for t in times:
        c.drive(m, d, t)
        r = c.rover_pose(t)
        if getattr(c, "looped", False):
            p, _ = c._at(c.rover_u(t) - back)
        else:
            import courses as cs
            p = (r[0] - back, c.rover_pose(max(t - back / cs.ROVER_SPEED, 0.0))[1])
        yaw = float(np.arctan2(r[1] - p[1], r[0] - p[0]))
        d.qpos[:7] = [p[0], p[1], 1.6, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        t0 = time.time()
        eye.look(d, d.qpos[:3], yaw, t)
        dt = time.time() - t0
        seg.update_scene(d, eye.cam)
        mask = np.isin(seg.render()[:, :, 0], rover)
        out.append((eye.last_rgb.copy(), mask, dt))
    return out


def red_check(frames):
    """Per frame: rover-coloured pixels on the rover and off it (anything off it could be taken for the rover),
    and all red/orange/brown pixels off it."""
    rows = []
    for rgb, mask, _ in frames:
        rl = rover_like(rgb)
        rows.append({"rover_px": int(mask.sum()), "rover_like_on_rover": int((rl & mask).sum()),
                     "rover_like_elsewhere": int((rl & ~mask).sum()),
                     "red_brown_elsewhere_pct": round(100 * red_fraction(np.where(mask[..., None], 0, rgb)), 2)})
    return rows


def docs(names=("mixed@real", "town@real", "city", "city@real"), seed=0, out_dir=None):
    """docs/realism-<name>.png: four onboard frames of each course (2x2, half size), plus the maps
    docs/course-<name>.png (courses.render). Prints the red check."""
    from PIL import Image
    import courses
    out_dir = out_dir or os.path.join(HERE, "docs")
    for name in names:
        tag_ = name.replace("@", "-")
        times = (8.0, 45.0, 90.0, 140.0) if name.startswith("city") else \
            (5.0, 30.0, 55.0, 80.0) if name.startswith("town") else (4.0, 14.0, 27.0, 44.0)
        fr = onboard(name, seed, times)
        tiles = [Image.fromarray(f[0]).resize((384, 288), Image.LANCZOS) for f in fr]
        grid = Image.new("RGB", (768, 576))
        for k, im in enumerate(tiles):
            grid.paste(im, ((k % 2) * 384, (k // 2) * 288))
        grid.quantize(192, method=Image.Quantize.MEDIANCUT).save(os.path.join(out_dir, "realism-%s.png" % tag_),
                                                                 optimize=True)
        if name != "city@real":
            im = Image.open(io.BytesIO(courses.render(name, seed, HERE))).convert("RGB")
            if im.size[0] > 900:
                im = im.resize((900, round(im.size[1] * 900 / im.size[0])), Image.LANCZOS)
            im.quantize(128, method=Image.Quantize.MEDIANCUT).save(os.path.join(out_dir, "course-%s.png" % tag_),
                                                                   optimize=True)
        print(name, "render ms", [round(1000 * f[2]) for f in fr], red_check(fr))


# ------------------------------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        ok, bad = fetch()
        print("fetched %d, failed %d" % (len(ok), len(bad)))
        for b in bad:
            print("  ", *b)
    elif len(sys.argv) > 1 and sys.argv[1] == "docs":
        docs(*([sys.argv[2].split(",")] if len(sys.argv) > 2 else []))
    else:
        print(__doc__)
