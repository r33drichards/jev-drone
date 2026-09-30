"""A real city block for the rover to lap: Vesterbro, Copenhagen, from OpenStreetMap.

The rover drives a smooth loop along the street centre lines round one block: Vesterbrogade (north-west,
a wide avenue), Oehlenschlaegersgade (east) and Kaalundsgade (south, bending north-west back to
Vesterbrogade), at ROVER_SPEED (1.1 m/s), one lap = LAP_M (~200 m, ~3 min). Every building within
~110 m of the block is extruded from its OSM footprint to its height (building:levels x 3.2 m, 16 m when
untagged), so the rover drops out of sight behind the corner buildings at every turn. Street trees
(OSM natural=tree) stand where OSM has them, unless within 3 m of the loop.

Geometry and data (realism_assets/city_vesterbro.json, built by `python city.py build`):
  * data (c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright): the
    footprints, heights, streets and trees within the bbox saved in the JSON, simplified (Douglas-Peucker
    0.3 m) and projected to metres about the loop's centre (x east, y north)
  * each building is a visual mesh (walls with window-bay texture coordinates, a flat roof; no collision)
    plus a convex decomposition of its footprint as invisible (group 3) collision prisms
  * sidewalks: a 0.12 m kerb slab round every building, cut back to leave >= 2.3 m of street either side
    of the loop (below the aircraft's ignore band, so flight.Eye never counts them as obstacles)
  * the floor is the single plane "floor", tiled asphalt
  * by default the city always wears realism textures and a Poly Haven sky (seed = course seed); a suffix
    adds more, e.g. "city@c12" or "city@real" (clutter), "city@real+neg" (hard negatives)

Success is town.py's LapTracker (a full lap followed, still tracking at the end). The drone starts
START_BEHIND (4.5 m) behind the rover along the loop, nose on it. Fly SECONDS (lap time + 40 s).

    MUJOCO_GL=osmesa python run.py --course city --backend const:oracle --fast --seconds 225 --seeds 0
    python city.py check          # rover path clearance to buildings/trees, curvature, lap length
    python city.py render         # docs/course-city.png
"""
import json, os, sys
import numpy as np
import town

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "realism_assets", "city_vesterbro.json")
ROVER_SPEED = 1.1
START_BEHIND = town.START_BEHIND
LEVEL_H = 3.2
DEFAULT_LEVELS = 5
PATH_CLEAR_MIN = 2.5          # the loop keeps at least this far from every wall (checked by `check`)
TREE_CLEAR = 3.0              # trees closer than this to the loop are left out
KERB_H, KERB_W, KERB_STREET = 0.12, 2.2, 2.3
BAY_W = 2.6                   # facade texture: one window bay wide, one storey tall
N_FACADES = 4

# the source: the block Vesterbrogade / Oehlenschlaegersgade / Kaalundsgade, Copenhagen
CENTRE_LATLON = (55.67221, 12.54944)
HALF_DEG = (0.00125, 0.00225)          # bbox half-size (lat, lon): ~140 m each way


# ----------------------------------------------------------------------------------------- build (OSM)
ROADS = {"residential", "tertiary", "secondary", "primary", "unclassified", "living_street", "pedestrian",
         "service", "tertiary_link", "secondary_link"}
ROAD_W = {"primary": 14.0, "secondary": 13.0, "tertiary": 11.0, "residential": 8.0, "unclassified": 7.0,
          "living_street": 6.0, "pedestrian": 6.0, "service": 4.0, "tertiary_link": 7.0, "secondary_link": 7.0}


def _area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _dp(P, eps):
    """Douglas-Peucker on an open polyline (N, 2)."""
    if len(P) < 3:
        return P
    a, b = P[0], P[-1]
    ab = b - a
    L = np.hypot(*ab)
    d = (np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L if L > 1e-9
         else np.hypot(*(P - a).T))
    k = int(np.argmax(d))
    if d[k] <= eps:
        return np.vstack([a, b])
    return np.vstack([_dp(P[:k + 1], eps)[:-1], _dp(P[k:], eps)])


def _simplify_ring(P, eps=0.3):
    """Simplify a closed ring (no repeated end point); counter-clockwise; no near-duplicate points."""
    k = int(np.argmax(np.hypot(*(P - P[0]).T)))       # split at the far point so both halves are real chains
    Q = np.vstack([_dp(np.vstack([P[:k + 1]]), eps)[:-1], _dp(np.vstack([P[k:], P[:1]]), eps)[:-1]])
    keep = [Q[0]]
    for q in Q[1:]:
        if np.hypot(*(q - keep[-1])) > 0.2:
            keep.append(q)
    Q = np.array(keep)
    if len(Q) > 3 and np.hypot(*(Q[0] - Q[-1])) < 0.2:
        Q = Q[:-1]
    return Q if _area(Q) > 0 else Q[::-1]


def _faces(nodes, roads):
    """Faces of the planar street graph (lists of node ids), by always taking the next edge clockwise."""
    adj = {}
    for nd, _ in roads:
        for a, b in zip(nd, nd[1:]):
            if a != b:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
    ang = {v: sorted(adj[v], key=lambda w: np.arctan2(nodes[w][1] - nodes[v][1], nodes[w][0] - nodes[v][0]))
           for v in adj}
    used, out = set(), []
    for u in adj:
        for v in adj[u]:
            if (u, v) in used:
                continue
            cyc, a, b = [], u, v
            while (a, b) not in used and len(cyc) < 5000:
                used.add((a, b))
                cyc.append(a)
                L = ang[b]
                a, b = b, L[(L.index(a) - 1) % len(L)]
            out.append(cyc)
    return out


def build(out=DATA):
    """Download the bbox from the OSM API and write the city JSON (ODbL: keep the attribution)."""
    import xml.etree.ElementTree as ET
    import realism
    lat0, lon0 = CENTRE_LATLON
    bbox = (lon0 - HALF_DEG[1], lat0 - HALF_DEG[0], lon0 + HALF_DEG[1], lat0 + HALF_DEG[0])
    url = "https://api.openstreetmap.org/api/0.6/map?bbox=%.6f,%.6f,%.6f,%.6f" % bbox
    raw = realism._download(url, os.path.join(realism.CACHE, "osm", "vesterbro.osm"))
    root = ET.parse(raw).getroot()
    k = np.pi / 180 * 6371000.0
    to_xy = lambda lat, lon: ((lon - lon0) * k * np.cos(np.radians(lat0)), (lat - lat0) * k)  # noqa: E731
    nodes, trees = {}, []
    for n in root.iter("node"):
        nodes[n.get("id")] = to_xy(float(n.get("lat")), float(n.get("lon")))
        tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
        if tags.get("natural") == "tree":
            trees.append(nodes[n.get("id")])
    blds, roads = [], []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        nd = [x.get("ref") for x in w.findall("nd")]
        if any(n not in nodes for n in nd):
            continue
        if "building" in tags and len(nd) > 3 and nd[0] == nd[-1]:
            lv = tags.get("building:levels")
            h = tags.get("height")
            try:
                height = float(h.split()[0]) if h else LEVEL_H * float(lv) + 0.8 if lv else None
            except ValueError:
                height = None
            blds.append((np.array([nodes[n] for n in nd[:-1]]), height, tags.get("building")))
        elif tags.get("highway") in ROADS and tags.get("area") != "yes":
            roads.append((nd, tags))
    # the loop: the street-graph face that contains the block centre
    best = None
    for cyc in _faces(nodes, roads):
        P = np.array([nodes[n] for n in cyc])
        if _area(P) > 0 and _inside((0.0, 0.0), P) and (best is None or _area(P) < _area(best)):
            best = P
    if best is None:
        raise RuntimeError("no street loop round the block centre")
    c = best.mean(axis=0)
    shift = lambda P: (np.asarray(P) - c)  # noqa: E731
    R = 115.0
    B = []
    for P, h, kind in blds:
        Q = _simplify_ring(shift(P))
        if len(Q) >= 3 and np.hypot(*Q.mean(0)) < R and abs(_area(Q)) > 4.0:
            B.append({"xy": np.round(Q, 2).tolist(), "h": round(h, 1) if h else None, "kind": kind})
    RD = []
    for nd, tags in roads:
        P = shift([nodes[n] for n in nd])
        if np.hypot(*P.T).min() < R + 20:
            RD.append({"xy": np.round(_dp(P, 0.3), 2).tolist(), "highway": tags["highway"],
                       "name": tags.get("name"), "width": ROAD_W[tags["highway"]]})
    T = [np.round(p, 2).tolist() for p in shift(trees) if np.hypot(*p) < R] if trees else []
    data = {"name": "Vesterbro, Copenhagen: Vesterbrogade / Oehlenschlaegersgade / Kaalundsgade",
            "attribution": "(c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright",
            "source": url, "centre_latlon": [lat0, lon0], "origin_offset_m": np.round(c, 2).tolist(),
            "loop": np.round(shift(best), 2).tolist(), "buildings": B, "roads": RD, "trees": T}
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote %s: %d buildings, %d roads, %d trees, loop %d nodes" % (out, len(B), len(RD), len(T), len(best)))
    return data


# ------------------------------------------------------------------------------------------- geometry
def _inside(p, P):
    x, y = p
    c = False
    n = len(P)
    for i in range(n):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            c = not c
    return c


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _triangulate(P):
    """Ear clipping of a simple counter-clockwise polygon -> list of index triples."""
    idx = list(range(len(P)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            a, b, c = P[i0], P[i1], P[i2]
            if _cross(a, b, c) <= 1e-9:
                continue
            if any(_cross(a, b, P[j]) >= 0 and _cross(b, c, P[j]) >= 0 and _cross(c, a, P[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:                        # degenerate: fan the rest
            tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, len(idx) - 1)]
            return tris
    tris.append(tuple(idx))
    return tris


def _convex(poly, P):
    n = len(poly)
    return all(_cross(P[poly[i]], P[poly[(i + 1) % n]], P[poly[(i + 2) % n]]) >= -1e-9 for i in range(n))


def _convex_pieces(P):
    """Hertel-Mehlhorn: triangulate, then merge neighbours while the union stays convex -> index lists."""
    polys = [list(t) for t in _triangulate(P)]
    merged = True
    while merged:
        merged = False
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                a, b = polys[i], polys[j]
                for ia in range(len(a)):
                    u, v = a[ia], a[(ia + 1) % len(a)]
                    if v in b and b[(b.index(v) + 1) % len(b)] == u:     # shared edge u-v (opposite winding)
                        jb = b.index(u)
                        rest = [b[(jb + k) % len(b)] for k in range(1, len(b) - 1)]
                        cand = a[:ia + 1] + rest + a[ia + 1:]
                        if _convex(cand, P):
                            polys[i] = cand
                            polys.pop(j)
                            merged = True
                            break
                if merged:
                    break
            if merged:
                break
    return polys


def _seg_dist(Q, A, B):
    """Distance from each point of Q (n, 2) to the nearest of segments A->B (m, 2)."""
    d = B - A
    L = np.maximum((d ** 2).sum(1), 1e-12)
    out = np.full(len(Q), np.inf)
    for s in range(0, len(Q), 2000):
        q = Q[s:s + 2000]
        t = np.clip(((q[:, None, :] - A[None]) * d[None]).sum(2) / L[None], 0, 1)
        c = A[None] + t[..., None] * d[None]
        out[s:s + 2000] = np.sqrt(((q[:, None, :] - c) ** 2).sum(2)).min(1)
    return out


def _smooth_loop(P, window_m=14.0, step=0.05):
    """A closed polyline -> dense, smooth points every `step` m: resample at 0.25 m, circular moving average
    over window_m (twice), resample. Rounds each street corner on a radius of a few metres."""
    def resample(Q, ds):
        Q2 = np.vstack([Q, Q[:1]])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q2, axis=0).T))])
        u = np.arange(0.0, s[-1], ds)
        return np.column_stack([np.interp(u, s, Q2[:, 0]), np.interp(u, s, Q2[:, 1])])
    Q = resample(np.asarray(P, float), 0.25)
    w = max(3, int(window_m / 0.25) | 1)
    ker = np.ones(w) / w
    for _ in range(2):
        Q = np.column_stack([np.convolve(np.concatenate([Q[-w:, i], Q[:, i], Q[:w, i]]), ker, "same")[w:-w]
                             for i in range(2)])
    pts = resample(Q, step)
    seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
    return pts, float(seg.sum())


def load(path=DATA):
    if not os.path.exists(path):
        build(path)
    return json.load(open(path))


_CACHE = {}


def geometry():
    """(data, loop points, lap length, buildings [(P ccw, height, convex pieces)], trees kept) -- cached."""
    if "g" not in _CACHE:
        data = load()
        pts, lap = _smooth_loop(np.array(data["loop"]))
        blds = []
        for b in data["buildings"]:
            P = np.array(b["xy"], float)
            if _area(P) < 0:
                P = P[::-1]
            h = b["h"] or (LEVEL_H * DEFAULT_LEVELS + 0.8)
            blds.append((P, float(h), _convex_pieces(P)))
        trees = []
        for x, y in data["trees"]:
            if np.hypot(pts[:, 0] - x, pts[:, 1] - y).min() >= TREE_CLEAR and \
                    not any(_inside((x, y), P) for P, _, _ in blds):
                trees.append((x, y))
        _CACHE["g"] = (data, pts, lap, blds, trees)
    return _CACHE["g"]


# ---------------------------------------------------------------------------------------------- course
_HEAD = """<mujoco model="{name}">
  <include file="mujoco_menagerie/skydio_x2/x2.xml"/>
  <statistic extent="20" center="0 0 2"/>
  <option timestep="0.002" density="1.2" viscosity="1.8e-5"/>
  <visual>
    <global fovy="50" offwidth="1400" offheight="1000"/>
    <headlight diffuse=".7 .7 .7" ambient=".4 .4 .4"/>
    <map znear="0.0025" zfar="3.0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".55 .7 .9" rgb2=".15 .2 .3" width="512" height="512"/>
    <material name="street" rgba=".30 .31 .33 1"/>
    <material name="uv_pavement" rgba=".55 .55 .53 1"/>
    <material name="uv_facade0" rgba=".80 .78 .70 1"/>
    <material name="uv_facade1" rgba=".62 .66 .72 1"/>
    <material name="uv_facade2" rgba=".85 .85 .82 1"/>
    <material name="uv_facade3" rgba=".55 .60 .52 1"/>
    <material name="uv_roof" rgba=".30 .30 .32 1"/>
    <material name="bark" rgba=".35 .32 .28 1"/>
    <material name="leaf" rgba=".22 .42 .20 1"/>
    <material name="rover" rgba=".95 .15 .10 1"/>
"""
_WORLD = """  </asset>
  <worldbody>
    <light pos="0 0 30" dir="0 0 -1" directional="true" diffuse=".6 .6 .6"/>
    <geom name="floor" type="plane" size="160 160 .05" material="street"/>
"""
_TAIL = """    <body name="rover" mocap="true" pos="0 0 .2">
      <geom name="rover_geom" type="box" material="rover" size=".35 .25 .2"/>
      <geom name="rover_mast" type="box" material="rover" size=".06 .06 .35" pos="0 0 .5"/>
    </body>
  </worldbody>
</mujoco>
"""
BASE_APPEARANCE = {"textures": "polyhaven", "sky": "polyhaven", "sun": True}


def _fmt(a):
    return " ".join("%.3f" % x for x in np.ravel(a))


def _prism_mesh(name, P, z0, z1, uv_walls=True, roof=True, bottom=False, tile=None):
    """Inline MJCF mesh of polygon P (ccw) extruded from z0 to z1: walls with per-face vertices and texture
    coordinates (u along the perimeter in bays, v in storeys, or in metres/`tile`), a triangulated roof."""
    V, T, F = [], [], []
    n = len(P)
    s = 0.0
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        L = float(np.hypot(*(b - a)))
        k = len(V)
        V += [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
        if tile:
            T += [(s / tile, z0 / tile), ((s + L) / tile, z0 / tile), ((s + L) / tile, z1 / tile), (s / tile, z1 / tile)]
        else:
            T += [(s / BAY_W, z0 / LEVEL_H), ((s + L) / BAY_W, z0 / LEVEL_H), ((s + L) / BAY_W, z1 / LEVEL_H),
                  (s / BAY_W, z1 / LEVEL_H)]
        F += [(k, k + 1, k + 2), (k, k + 2, k + 3)]          # ccw polygon: outward walls
        s += L
    rt = tile or 3.0
    tris = _triangulate(P)
    if roof:
        k = len(V)
        V += [(p[0], p[1], z1) for p in P]
        T += [(p[0] / rt, p[1] / rt) for p in P]
        F += [(k + a, k + b, k + c) for a, b, c in tris]
    if bottom:
        k = len(V)
        V += [(p[0], p[1], z0) for p in P]
        T += [(p[0] / rt, p[1] / rt) for p in P]
        F += [(k + a, k + c, k + b) for a, b, c in tris]
    return '    <mesh name="%s" vertex="%s" texcoord="%s" face="%s"/>\n' % (
        name, _fmt(V), _fmt(T), " ".join("%d" % i for f in F for i in f))


def _roof_mesh(name, P, h, tile=3.0):
    """A flat roof: polygon P (ccw) at height h, a thin slab (MuJoCo needs a mesh with volume)."""
    V, T, F = [], [], []
    tris = _triangulate(P)
    n = len(P)
    for z in (h, h - 0.05):
        V += [(p[0], p[1], z) for p in P]
        T += [(p[0] / tile, p[1] / tile) for p in P]
    F += [(a, b, c) for a, b, c in tris] + [(n + a, n + c, n + b) for a, b, c in tris]
    return '    <mesh name="%s" vertex="%s" texcoord="%s" face="%s"/>\n' % (
        name, _fmt(V), _fmt(T), " ".join("%d" % i for f in F for i in f))


def _offset_convex(Q, d):
    """A convex ccw polygon pushed out by d (mitred)."""
    n = len(Q)
    out = []
    for i in range(n):
        p0, p1, p2 = Q[i - 1], Q[i], Q[(i + 1) % n]
        e1 = (p1 - p0) / max(np.hypot(*(p1 - p0)), 1e-9)
        e2 = (p2 - p1) / max(np.hypot(*(p2 - p1)), 1e-9)
        n1, n2 = np.array([e1[1], -e1[0]]), np.array([e2[1], -e2[0]])     # outward normals of a ccw polygon
        m = n1 + n2
        m = m / max(np.hypot(*m), 1e-9)
        if float(m @ n1) > 0.75:                      # gentle corner: mitre
            out.append(p1 + m * d / float(m @ n1))
        else:                                         # sharp corner: bevel (a mitre would spike out)
            out += [p1 + n1 * d, p1 + n2 * d]
    return np.array(out)


class CityCourse(town.TownCourse):
    """courses.Course's interface (via town.TownCourse: looped, LapTracker) for the real city block."""
    looped = True
    first_barrier_x = 1e9
    end_x = 1e9

    def __init__(self, name="city", seed=0):
        self.name, self.seed = name, seed
        data, pts, lap, self.blds, self.tree_xy = geometry()
        rng = np.random.default_rng(5000 + seed)
        self.direction = int(rng.choice([1, -1]))
        self.u0 = float(rng.uniform(0.0, lap))
        self.pts = pts if self.direction > 0 else pts[::-1].copy()
        s = np.hypot(*np.diff(np.vstack([self.pts, self.pts[:1]]), axis=0).T)
        self.u = np.concatenate([[0.0], np.cumsum(s)[:-1]])
        self.lap_m = lap
        self.speed = ROVER_SPEED
        self.data = data
        self.appearance = None
        self.base_appearance = dict(BASE_APPEARANCE)
        self.seconds = round(lap / ROVER_SPEED + 40.0)

    # --- geometry ---------------------------------------------------------------------------------
    def boxes(self):
        return []                       # (town.TownCourse's obstacle list; the city's are meshes)

    def trees(self):
        return [("tree%d" % i, x, y, 0.22, 4.6, 1.9) for i, (x, y) in enumerate(self.tree_xy)]

    def xml(self):
        g, assets = [_HEAD.format(name=self.name)], []
        body = [_WORLD]
        rng = np.random.default_rng(11)                    # which facade material each building wears: fixed
        P_loop = self.pts[::10]
        for i, (P, h, pieces) in enumerate(self.blds):
            assets.append(_prism_mesh("bld%d" % i, P, 0.0, h, roof=False))
            assets.append(_roof_mesh("bld%dr" % i, P, h))
            body.append('    <geom name="bld%d" type="mesh" mesh="bld%d" material="uv_facade%d" contype="0" '
                        'conaffinity="0"/>\n' % (i, i, int(rng.integers(N_FACADES))))
            body.append('    <geom name="bld%dr" type="mesh" mesh="bld%dr" material="uv_roof" contype="0" '
                        'conaffinity="0"/>\n' % (i, i))
            for j, piece in enumerate(pieces):
                Q = P[piece]
                V = np.vstack([np.column_stack([Q, np.zeros(len(Q))]), np.column_stack([Q, np.full(len(Q), h)])])
                assets.append('    <mesh name="bld%dc%d" vertex="%s"/>\n' % (i, j, _fmt(V)))
                body.append('    <geom name="bld%dc%d" type="mesh" mesh="bld%dc%d" group="3" rgba="0 0 0 0"/>\n'
                            % (i, j, i, j))
                # the kerb slab: this piece pushed out, but never within KERB_STREET of the loop
                dmin = float(np.hypot(*(P_loop[:, None, :] - Q[None]).transpose(2, 0, 1)).min())
                w = min(KERB_W, dmin - KERB_STREET)
                if w > 0.3:
                    K = _offset_convex(Q, w)
                    assets.append(_prism_mesh("kerb%dp%d" % (i, j), K, 0.0, KERB_H, tile=3.0))
                    body.append('    <geom name="kerb%dp%d" type="mesh" mesh="kerb%dp%d" material="uv_pavement" '
                                'contype="0" conaffinity="0"/>\n' % (i, j, i, j))
        for name, x, y, r, cz, cr in self.trees():
            body.append('    <geom name="%s" type="cylinder" material="bark" size="%.2f %.2f" pos="%.2f %.2f %.2f"/>\n'
                        % (name, r, cz / 2, x, y, cz / 2))
            body.append('    <geom name="%sC" type="sphere" material="leaf" size="%.2f" pos="%.2f %.2f %.2f"/>\n'
                        % (name, cr, x, y, cz))
        g += assets + body + [_TAIL]
        import realism                  # the city always wears real textures and sky (BASE_APPEARANCE)
        if self.appearance is None:
            self.appearance = realism.parse(dict(self.base_appearance), self.seed)
        return realism.apply("".join(g), self)

    def clearance(self):
        """The rover centre's smallest horizontal distance to any building wall or tree trunk over the loop."""
        best = (1e9, None)
        for i, (P, h, _) in enumerate(self.blds):
            dmin = float(_seg_dist(self.pts[::2], P, np.roll(P, -1, axis=0)).min())
            if dmin < best[0]:
                best = (dmin, "bld%d" % i)
        for name, x, y, r, cz, cr in self.trees():
            dmin = float(np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y).min() - r)
            if dmin < best[0]:
                best = (dmin, name)
        return best

    def clutter_region(self):
        """realism.py: where clutter may stand -- on the kerbs and pavements: >= 2.6 m + r from the loop, not in a
        building (0.3 m margin), not on a tree, and within 8 m of the loop (its pavements and the junction mouths)."""
        P = self.pts[::4]
        polys = [p for p, _, _ in self.blds]

        def ok(x, y, r):
            dmin = np.hypot(P[:, 0] - x, P[:, 1] - y).min()
            if dmin < 2.6 + r or dmin > 8.0:          # on the loop's own pavements, where the camera sees it
                return False
            if any(_inside((x, y), Q) for Q in polys):
                return False
            if any(_seg_dist(np.array([[x, y]]), Q, np.roll(Q, -1, axis=0))[0] < r + 0.3 for Q in polys):
                return False
            return all(np.hypot(x - tx, y - ty) > r + 0.6 for tx, ty in self.tree_xy)
        lo, hi = P.min(0) - 12.0, P.max(0) + 12.0
        return (lo[0], hi[0], lo[1], hi[1]), ok

    # --- map --------------------------------------------------------------------------------------
    @property
    def MAP_SPAN(self):
        lo, hi = self.pts.min(0) - 30.0, self.pts.max(0) + 30.0
        return (float(hi[0] - lo[0]), float(hi[1] - lo[1]))

    def map_centre(self):
        lo, hi = self.pts.min(0) - 30.0, self.pts.max(0) + 30.0
        return (lo + hi) / 2

    def map_px(self, x, y, w, h):
        sx, sy = self.MAP_SPAN
        cx, cy = self.map_centre()
        scale = h / (2 * max(sy / 2, (sx / 2) * h / w))
        return (w / 2 + (x - cx) * scale, h / 2 - (y - cy) * scale)


def make(name="city", seed=0):
    if "@" in name:
        import realism
        return realism.make(name, seed)
    if name != "city":
        raise KeyError("unknown city course %r (only 'city')" % name)
    return CityCourse(name, seed)


def render(name="city", seed=0, directory=".", width=1100, height=900, path_dots=True):
    """Top-down view of the city block with the rover's loop dotted on, as PNG bytes."""
    import io, mujoco
    from PIL import Image, ImageDraw
    c = make(name, seed)
    m = mujoco.MjModel.from_xml_path(c.write(directory))
    m.vis.map.zfar = 400.0
    d = mujoco.MjData(m)
    d.qpos[:3] = [0.0, 0.0, -5.0]
    c.drive(m, d, 0.0)
    mujoco.mj_forward(m, d)
    m.vis.headlight.diffuse[:] = [.35, .35, .35]
    m.vis.headlight.ambient[:] = [.45, .45, .45]
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    sx, sy = c.MAP_SPAN
    fovy = 12.0
    m.vis.global_.fovy = fovy
    half_h = max(sy / 2, (sx / 2) * height / width)
    cx, cy = c.map_centre()
    cam.lookat[:] = [cx, cy, 0]
    cam.elevation, cam.azimuth = -90.0, 90.0
    cam.distance = half_h / np.tan(np.deg2rad(fovy) / 2)
    r.update_scene(d, cam)
    img = Image.fromarray(r.render())
    dr = ImageDraw.Draw(img)
    W, H = img.size
    px = lambda x, y: c.map_px(x, y, W, H)  # noqa: E731
    if path_dots:
        for k in range(0, len(c.pts), 8):
            u, v = px(*c.pts[k])
            dr.ellipse([u - 1.5, v - 1.5, u + 1.5, v + 1.5], fill=(235, 40, 30))
        for uu in np.arange(0.0, c.lap_m, 10.0):
            p, hh = c._at(c.u0 + uu)
            u, v = px(*p)
            a = np.array([np.cos(hh), -np.sin(hh)]) * 7
            nrm = np.array([-a[1], a[0]]) * 0.6
            dr.polygon([(u + a[0], v + a[1]), (u - a[0] + nrm[0], v - a[1] + nrm[1]),
                        (u - a[0] - nrm[0], v - a[1] - nrm[1])], fill=(255, 230, 60))
        u, v = px(*c.rover_pose(0.0)[:2])
        dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(255, 255, 255), width=2)
    for x, y, rr in getattr(c, "clutter", []):
        u, v = px(x, y)
        dr.rectangle([u - 3, v - 3, u + 3, v + 3], outline=(80, 200, 255), width=1)
    dr.text((8, 6), "%s seed %d  |  lap %.0f m  |  rover %.2f m/s  |  (c) OpenStreetMap contributors, ODbL"
            % (name, seed, c.lap_m, c.speed), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def check():
    c = CityCourse("city", 0)
    clr, what = c.clearance()
    pts = c.pts
    # curvature radius from three points 2 m apart
    k = 40
    a, b, cc = pts, np.roll(pts, -k, 0), np.roll(pts, -2 * k, 0)
    ab, bc, ca = np.hypot(*(b - a).T), np.hypot(*(cc - b).T), np.hypot(*(a - cc).T)
    area2 = np.abs((b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0]))
    R = ab * bc * ca / np.maximum(2 * area2, 1e-9)
    print("lap %.1f m (%.0f s at %.2f m/s), fly %d s; %d buildings, %d trees kept; min clearance %.2f m (%s); "
          "min turn radius %.1f m" % (c.lap_m, c.lap_m / c.speed, c.speed, c.seconds, len(c.blds), len(c.tree_xy),
                                      clr, what, float(R.min())))
    return clr


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        build()
    elif cmd == "check":
        check()
    elif cmd == "render":
        open(os.path.join(HERE, "docs", "course-city.png"), "wb").write(render("city", 0, HERE))
