"""A "Nuketown"-style map: two houses facing each other across a street, and a rover driving a figure-8.

Layout (x east, y north, metres; the map is 58 m x 40 m inside a 5 m perimeter wall):

    y=+20 +--------------------------------------------------------------+
          |                          bus  car                            |
          |   +--------fence---------  ||      ---------fence--------+   |
          |   | tree   . . . . .       ||        . . . . .   tree    |   |
          |   |      .  +-------+ .           . +-------+  .         |   |
          |   |back  .  | YELLOW|   .       .   |  BLUE |  .   back  |   |
          |   |fence .  | house |door  X   door| house |  .  fence  |   |
          |   |      .  +-------+   .      .    +-------+  .         |   |
          |   |        . . . . . .            . . . . . .            |   |
          |   +shed----fence---------  crates  ---------fence----shed+   |
          |                                truck                         |
    y=-20 +--------------------------------------------------------------+
        x=-29        west lot        street |x| < 6        east lot     x=+29

  street   x in [-6, 6], the whole height of the map. A bus is parked at its north end, a truck at its
           south end (both > 3 m tall), with a burnt-out car and a stack of crates beside them.
  houses   BH (blue, east) and YH (yellow, west): 6 m deep (x in [9, 15]) x 8 m wide (y in [-4, 4]),
           6.2 m walls plus a roof slab to 7.2 m -- two storeys, never climbable. Front doors face the street.
  yards    each lot is fenced on three sides: side fences at y = +-10.5 from the street edge to the back
           fence at |x| = 22, all 2.6 m tall (above CLIMB_ALT, so nothing here is climbed; the oracle is
           always hold_course). A shed and a tree stand in the back corners of each yard, a planter box
           (1.2 m) and a porch in front. Outside the lots: alleys to the perimeter wall.

The rover drives a figure-8 at ROVER_SPEED (1.15 m/s, as on courses.py): around the east house clockwise
and the west house counter-clockwise, on tangent lines and arcs of circles centred on the house corners --
R_FRONT = 3.5 m at the street corners, R_BACK = 1.8 m behind the house -- crossing the street diagonally
(~45 deg) at the map centre between the bus and the truck. It passes behind each house through the backyard.
A drone within ~5 m of it keeps it in view round those back corners (code pursuit, which trails ~4 m, does);
one that falls further behind loses it at every back corner (arc radius r, sight is lost once the aircraft is
a m before the corner and the rover b m past it with a*b > r^2). One lap is LAP_M ~ 100 m (~87 s); the
rover's path keeps 1.8 m from the house's back corners and >= 2.8 m from everything else (TownCourse.clearance). The rover mocap faces its heading.

A seed varies: the direction of travel, the start point on the loop, and the vehicles' positions (+-1 m).

Success on a looped course (run.episode fills these in via LapTracker.report):
  lap_progress_m    how far the aircraft has come along the rover's loop: its position projected on the loop,
                    searched only a little behind and at most 12 m ahead of the previous projection (so the
                    crossing never jumps it half a lap), updated only while it is within 5 m of the loop and
                    never decreasing. Cutting a lobe or wandering off freezes it until the aircraft rejoins
                    the loop near where it left.
  laps_followed     lap_progress_m / LAP_M.
  within_8m_pct     % of sim steps with the aircraft within 8 m (3D) of the rover.
  lap_done_at_s     first time lap_progress_m >= LAP_M with the rover within 8 m at that moment.
  finished_at_s     lap_done_at_s, kept only if the aircraft is still tracking at the end: not lost at the end
                    (run.py's lost_at_end: the last unseen stretch <= 2 s), within 8 m of the rover for >= 70%
                    of the last 10 s, and mean standoff over the flight < MEAN_STANDOFF_MAX (8 m). Otherwise None.
  crossed_barrier   always False and max_x_m meaningless (there is no barrier / end line); end_x_m is None.
  path              None: pathmetrics measures weaving along +x, which a loop does not have.

Usage: courses.make / courses.render hand any name starting with "town" to this module, so everything that
takes a course name takes "town" (run.py --course town, flightgif, and every modal_laya.py config):

    MUJOCO_GL=osmesa python run.py --course town --backend const:oracle --fast --seconds 120 --seeds 0
    modal run modal_laya.py::baseline --configs code-pursuit-oracle,laya-pursuit-v2 --courses town \
        --seeds 0,1,2,3,4,5 --seconds 120 --budget 300 --model /ckpt/smolvlm/drone-rover-v2/last
    modal run modal_laya.py::gifs --config laya-pursuit-v2 --courses town --seeds 0,1 --seconds 120 \
        --budget 300 --model /ckpt/smolvlm/drone-rover-v2/last
    modal run modal_laya.py::courses_png --names town        # docs/course-town.png (or town.render locally)

Fly it 120 s (SECONDS): a lap takes ~87 s. modal_laya's printouts still show max_x / crossed, which mean
nothing here; read finished_at_s, laps_followed, within_8m_pct from episodes.jsonl.
"""
import os
import numpy as np

ROVER_SPEED = 1.15
R_FRONT = 3.5            # rover path radius round the house's street corners (centred on the corner)
R_BACK = 1.8             # ... and round its back corners: tight, so a drone > ~5 m behind loses sight at them
HOUSE = (9.0, 15.0, -4.0, 4.0)      # east house footprint (x0, x1, y0, y1); the west one is mirrored in x
HOUSE_H, ROOF_H = 6.2, 7.2
FENCE_H = 2.6
FENCE_Y = 10.5           # side fences at |y| = FENCE_Y, from the street edge ...
FENCE_X0, FENCE_X1 = 6.0, 22.0      # ... to the back fence at |x| = FENCE_X1
WALL_X, WALL_Y, WALL_H = 29.0, 20.0, 5.0
START_BEHIND = 4.5       # the aircraft starts this far behind the rover along the loop
MEAN_STANDOFF_MAX = 8.0
SECONDS = 120.0

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
    <texture name="grid" type="2d" builtin="checker" rgb1=".22 .24 .26" rgb2=".28 .30 .33" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="18 18" reflectance="0"/>
    <material name="wall"   rgba=".50 .50 .52 1"/>
    <material name="blue"   rgba=".35 .55 .80 1"/>
    <material name="yellow" rgba=".90 .78 .35 1"/>
    <material name="roofE"  rgba=".20 .33 .55 1"/>
    <material name="roofW"  rgba=".65 .50 .18 1"/>
    <material name="roof"   rgba=".35 .22 .18 1"/>
    <material name="window" rgba=".12 .16 .22 1"/>
    <material name="door"   rgba=".45 .25 .12 1"/>
    <material name="fence"  rgba=".85 .85 .80 1"/>
    <material name="bus"    rgba=".95 .75 .10 1"/>
    <material name="truck"  rgba=".30 .45 .30 1"/>
    <material name="cab"    rgba=".75 .75 .78 1"/>
    <material name="burnt"  rgba=".18 .15 .13 1"/>
    <material name="crate"  rgba=".55 .40 .22 1"/>
    <material name="shed"   rgba=".55 .35 .30 1"/>
    <material name="bark"   rgba=".35 .25 .15 1"/>
    <material name="leaf"   rgba=".20 .45 .20 1"/>
    <material name="rover"  rgba=".95 .15 .10 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 14" dir="0 0 -1" directional="true" diffuse=".6 .6 .6"/>
    <geom name="floor" type="plane" size="90 90 .05" material="grid"/>
"""
_TAIL = """    <body name="rover" mocap="true" pos="0 0 .2">
      <geom name="rover_geom" type="box" material="rover" size=".35 .25 .2"/>
      <geom name="rover_mast" type="box" material="rover" size=".06 .06 .35" pos="0 0 .5"/>
    </body>
  </worldbody>
</mujoco>
"""


def _geom_box(name, x0, x1, y0, y1, z0, z1, mat):
    x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    return ('    <geom name="%s" type="box" material="%s" size="%.3f %.3f %.3f" pos="%.3f %.3f %.3f"/>\n'
            % (name, mat, (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


# --------------------------------------------------------------------------------------------- path
def _tangent(c1, s1, r1, c2, s2, r2):
    """The tangent segment leaving circle c1 (turn s1: +1 ccw, -1 cw; radius r1) toward circle c2 (s2, r2):
    (heading, point on c1, point on c2). On a circle turning s the centre is s*r to the left of the heading."""
    D = np.subtract(c2, c1)
    d, phi = float(np.hypot(*D)), float(np.arctan2(D[1], D[0]))
    th = phi - np.arcsin((s2 * r2 - s1 * r1) / d)
    h = np.array([np.cos(th), np.sin(th)])
    nl = np.array([-h[1], h[0]])
    return th, np.asarray(c1) - s1 * r1 * nl, np.asarray(c2) - s2 * r2 * nl


def figure8(step=0.05):
    """The loop as dense points (N, 2) every `step` m in the base direction, and its length. Base
    direction: across the centre heading north-east, clockwise round the east house, across heading
    north-west, counter-clockwise round the west house. Each house corner is rounded on a circle centred
    on it: R_FRONT at the street corners, R_BACK behind the house."""
    x0, x1, y0, y1 = HOUSE
    circles = [((x0, y1), -1, R_FRONT), ((x1, y1), -1, R_BACK), ((x1, y0), -1, R_BACK), ((x0, y0), -1, R_FRONT),
               ((-x0, y1), 1, R_FRONT), ((-x1, y1), 1, R_BACK), ((-x1, y0), 1, R_BACK), ((-x0, y0), 1, R_FRONT)]
    n = len(circles)
    tans = [_tangent(*circles[k], *circles[(k + 1) % n]) for k in range(n)]
    pts = []
    for k in range(n):
        # the tangent line leaving circle k, then the arc on circle k+1 up to the next tangent
        th_in, a, b = tans[k]
        seg = float(np.hypot(*(b - a)))
        for u in np.arange(0.0, seg, step):
            pts.append(a + (b - a) * u / seg)
        c, s, r = circles[(k + 1) % n]
        th_out = tans[(k + 1) % n][0]
        turn = s * ((s * (th_out - th_in)) % (2 * np.pi))       # signed, in the circle's direction
        # the angle of the point on the circle, seen from its centre: heading - s*90 deg
        for u in np.arange(0.0, abs(turn) * r, step):
            ang = th_in - s * np.pi / 2 + s * u / r
            pts.append(np.asarray(c) + r * np.array([np.cos(ang), np.sin(ang)]))
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    return pts, float(seg.sum())


_PTS, LAP_M = figure8()


class TownCourse:
    """courses.Course's interface for the town. `looped`: run.episode measures laps, not x."""
    looped = True
    first_barrier_x = 1e9            # never crossed: no barrier
    end_x = 1e9                      # never reached: finished_at_s comes from LapTracker

    def __init__(self, name="town", seed=0):
        self.name, self.seed = name, seed
        rng = np.random.default_rng(3000 + seed)
        self.direction = int(rng.choice([1, -1]))
        self.u0 = float(rng.uniform(0.0, LAP_M))           # the rover's start, in travel-order arc length
        self.bus_dy = float(rng.uniform(-1.0, 1.0))
        self.truck_dy = float(rng.uniform(-1.0, 1.0))
        self.car_dx = float(rng.uniform(-0.8, 0.8))
        # the loop in travel order: dense points every 0.05 m, u = arc length from the base start
        self.pts = _PTS if self.direction > 0 else _PTS[::-1].copy()
        s = np.linalg.norm(np.diff(np.vstack([self.pts, self.pts[:1]]), axis=0), axis=1)
        self.u = np.concatenate([[0.0], np.cumsum(s)[:-1]])
        self.lap_m = LAP_M
        self.speed = ROVER_SPEED

    # --- geometry -------------------------------------------------------------------------------
    def boxes(self):
        """Every obstacle as (name, x0, x1, y0, y1, z0, z1, material); the tree canopies are spheres
        (trees(), below), counted here by their trunks."""
        b = []
        x0, x1, hy0, hy1 = HOUSE
        for side, tag, mat in ((1, "E", "blue"), (-1, "W", "yellow")):
            X = lambda v: side * v  # noqa: E731
            b.append(("house" + tag, X(x0), X(x1), hy0, hy1, 0.0, HOUSE_H, mat))
            b.append(("roof" + tag, X(x0 - 0.3), X(x1 + 0.3), hy0 - 0.3, hy1 + 0.3, HOUSE_H, ROOF_H, "roof" + tag))
            # facade details, 5 cm proud of the walls: door and windows toward the street, windows elsewhere
            f = x0 - 0.05
            b.append(("door" + tag, X(f), X(x0), -0.6, 0.6, 0.0, 2.2, "door"))
            for k, (ya, yb) in enumerate(((-3.2, -1.6), (1.6, 3.2))):
                b.append(("winF%d%s" % (k, tag), X(f), X(x0), ya, yb, 1.0, 2.2, "window"))
                b.append(("winU%d%s" % (k, tag), X(f), X(x0), ya, yb, 3.8, 5.0, "window"))
                b.append(("winB%d%s" % (k, tag), X(x1), X(x1 + 0.05), ya, yb, 3.8, 5.0, "window"))
            for k, (xa, xb) in enumerate(((x0 + 1.0, x0 + 2.4), (x1 - 2.4, x1 - 1.0))):
                b.append(("winN%d%s" % (k, tag), X(xa), X(xb), hy1, hy1 + 0.05, 3.8, 5.0, "window"))
                b.append(("winS%d%s" % (k, tag), X(xa), X(xb), hy0 - 0.05, hy0, 3.8, 5.0, "window"))
            # porch roof on posts, in front of the door (tall: 2.8 m)
            b.append(("porch" + tag, X(x0 - 1.4), X(x0), -1.3, 1.3, 2.6, 2.9, "roof"))
            for k, py in enumerate((-1.2, 1.2)):
                b.append(("post%d%s" % (k, tag), X(x0 - 1.4), X(x0 - 1.2), py - 0.1, py + 0.1, 0.0, 2.6, "fence"))
            # fences: two sides and the back
            for k, fy in enumerate((FENCE_Y, -FENCE_Y)):
                b.append(("fenceS%d%s" % (k, tag), X(FENCE_X0), X(FENCE_X1), fy - 0.06, fy + 0.06, 0.0, FENCE_H, "fence"))
            b.append(("fenceB" + tag, X(FENCE_X1 - 0.06), X(FENCE_X1 + 0.06), -FENCE_Y, FENCE_Y, 0.0, FENCE_H, "fence"))
            # garden: a shed in one back corner, a planter by the front, the tree trunk is in trees()
            b.append(("shed" + tag, X(19.2), X(21.8), -10.3, -8.8, 0.0, 2.7, "shed"))
            b.append(("planter" + tag, X(10.0), X(12.5), 4.2, 4.7, 0.0, 1.2, "crate"))
        # the street: a bus at the north end, a truck at the south end, a burnt car and crates
        by = self.bus_dy
        b.append(("bus", -4.4, -1.9, 9.5 + by, 18.5 + by, 0.3, 3.3, "bus"))
        ty = self.truck_dy
        b.append(("truckBox", 1.3, 3.8, -17.5 + ty, -11.5 + ty, 0.4, 3.6, "truck"))
        b.append(("truckCab", 1.4, 3.7, -11.4 + ty, -9.6 + ty, 0.0, 2.8, "cab"))
        cx = self.car_dx
        b.append(("car", 1.4 + cx, 3.3 + cx, 12.0, 16.4, 0.0, 1.5, "burnt"))
        b.append(("crate0", -4.6, -3.2, -15.0, -13.6, 0.0, 1.4, "crate"))
        b.append(("crate1", -4.4, -3.4, -15.0, -14.0, 1.4, 2.4, "crate"))
        # perimeter wall
        W = 0.4
        b.append(("wallN", -WALL_X - W, WALL_X + W, WALL_Y, WALL_Y + W, 0.0, WALL_H, "wall"))
        b.append(("wallS", -WALL_X - W, WALL_X + W, -WALL_Y - W, -WALL_Y, 0.0, WALL_H, "wall"))
        b.append(("wallE", WALL_X, WALL_X + W, -WALL_Y, WALL_Y, 0.0, WALL_H, "wall"))
        b.append(("wallW", -WALL_X - W, -WALL_X, -WALL_Y, WALL_Y, 0.0, WALL_H, "wall"))
        return b

    @staticmethod
    def trees():
        """(name, x, y, trunk radius, canopy centre z, canopy radius): one per backyard."""
        return [("tree" + tag, side * 20.3, 8.6, 0.25, 3.6, 1.4) for side, tag in ((1, "E"), (-1, "W"))]

    def xml(self):
        g = [_HEAD.format(name=self.name)]
        for b in self.boxes():
            g.append(_geom_box(*b))
        for name, x, y, r, cz, cr in self.trees():
            g.append('    <geom name="%s" type="cylinder" material="bark" size="%.2f %.2f" pos="%.2f %.2f %.2f"/>\n'
                     % (name, r, cz / 2, x, y, cz / 2))
            g.append('    <geom name="%sC" type="sphere" material="leaf" size="%.2f" pos="%.2f %.2f %.2f"/>\n'
                     % (name, cr, x, y, cz))
        g.append(_TAIL)
        if getattr(self, "appearance", None):     # opt-in real-world look (realism.py); off by default
            import realism
            return realism.apply("".join(g), self)
        return "".join(g)

    def write(self, directory):
        path = os.path.join(directory, ".course_%s_%d.xml" % (self.name, self.seed))
        with open(path, "w") as f:
            f.write(self.xml())
        return path

    # --- motion ---------------------------------------------------------------------------------
    def _at(self, u):
        """Point and heading on the loop at travel-order arc length u (wrapped)."""
        u = float(u) % self.lap_m
        k = int(np.searchsorted(self.u, u, side="right") - 1)
        k1 = (k + 1) % len(self.pts)
        du = (self.u[k1] if k1 else self.lap_m) - self.u[k]
        a = (u - self.u[k]) / max(du, 1e-9)
        p = self.pts[k] + a * (self.pts[k1] - self.pts[k])
        h = self.pts[k1] - self.pts[k]
        return p, float(np.arctan2(h[1], h[0]))

    def rover_u(self, t):
        return self.u0 + self.speed * t

    def rover_pose(self, t):
        p, _ = self._at(self.rover_u(t))
        return np.array([p[0], p[1], 0.2])

    def drive(self, m, d, t):
        p, h = self._at(self.rover_u(t))
        mid = m.body("rover").mocapid[0]
        d.mocap_pos[mid] = [p[0], p[1], 0.2]
        d.mocap_quat[mid] = [np.cos(h / 2), 0.0, 0.0, np.sin(h / 2)]

    def start_pose(self, rng, alt):
        """qpos[:7] for the aircraft: START_BEHIND m behind the rover along the loop (+-0.3 m), at `alt`,
        nose on the rover."""
        p, _ = self._at(self.u0 - START_BEHIND + rng.uniform(-0.3, 0.3))
        r = self.rover_pose(0.0)
        yaw = float(np.arctan2(r[1] - p[1], r[0] - p[0]))
        return np.array([p[0], p[1], alt, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    def lap_tracker(self, start_xy):
        return LapTracker(self, start_xy)

    # --- the right answer -----------------------------------------------------------------------
    def oracle(self, pos):
        """Nothing here is climbable (every fence is 2.6 m, above CLIMB_ALT) and there are no traps: the
        right answer is always to leave it to the pursuit and the reactive layer."""
        return "hold_course"

    def altitude_target(self, pos):
        """Cruise everywhere (altitude.py): nothing here needs flying over."""
        import altitude
        return altitude.CRUISE

    # --- checks ---------------------------------------------------------------------------------
    def clearance(self):
        """The rover centre's smallest horizontal distance to any obstacle over the whole loop, and to what."""
        best = (1e9, None)
        P = self.pts
        for name, x0, x1, y0, y1, z0, z1, _ in self.boxes():
            if z0 > 1.0:              # overhead (porch roof, bus body over its wheels is z0=0.3: kept)
                continue
            dx = np.maximum(np.maximum(x0 - P[:, 0], P[:, 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - P[:, 1], P[:, 1] - y1), 0.0)
            dmin = float(np.hypot(dx, dy).min())
            if dmin < best[0]:
                best = (dmin, name)
        for name, x, y, r, cz, cr in self.trees():
            dmin = float(np.hypot(P[:, 0] - x, P[:, 1] - y).min() - r)
            if dmin < best[0]:
                best = (dmin, name)
        return best

    # --- map ------------------------------------------------------------------------------------
    MAP_SPAN = (2 * WALL_X + 3.0, 2 * WALL_Y + 3.0)

    def map_px(self, x, y, w, h):
        """World (x, y) -> pixel on render()'s image scaled to w x h: +x right, +y up, centred."""
        sx, sy = self.MAP_SPAN
        scale = h / (2 * max(sy / 2, (sx / 2) * h / w))
        return (w / 2 + x * scale, h / 2 - y * scale)


class LapTracker:
    """Progress of the aircraft along the rover's loop, and the looped-course success (module docstring)."""
    BACK_M, AHEAD_M, NEAR_M, TRACK_M = 2.0, 12.0, 5.0, 8.0

    def __init__(self, course, start_xy):
        self.c = course
        P, U = course.pts, course.u
        k = int(np.argmin(np.linalg.norm(P - np.asarray(start_xy)[:2], axis=1)))
        # the aircraft starts behind the rover: pick the unwrapped u just behind the rover's start
        u = U[k]
        while u > course.u0:
            u -= course.lap_m
        while u < course.u0 - course.lap_m / 2:
            u += course.lap_m
        self.u_start = self.u_best = float(u)
        self.lap_done_at = None
        self.n = self.n_within = 0
        self.tail = []               # (t, within 8 m) over the last 10 s

    def update(self, t, pos, standoff):
        """Call at a fixed rate (run.episode: every 50 sim steps, 10 Hz) with the aircraft's position and
        its distance to the rover."""
        c = self.c
        n = len(c.pts)
        step = c.u[1] - c.u[0]
        k0 = int(np.floor((self.u_best - self.BACK_M) / step))
        ks = np.arange(k0, k0 + int((self.BACK_M + self.AHEAD_M) / step) + 1)
        P = c.pts[ks % n]
        dist = np.linalg.norm(P - np.asarray(pos)[:2], axis=1)
        j = int(np.argmin(dist))
        if dist[j] < self.NEAR_M:
            self.u_best = max(self.u_best, float(ks[j] * step))
        within = bool(standoff < self.TRACK_M)
        self.n += 1
        self.n_within += within
        self.tail.append((t, within))
        self.tail = [(tt, w) for tt, w in self.tail if t - tt <= 10.0]
        if self.lap_done_at is None and self.u_best - self.u_start >= c.lap_m and within:
            self.lap_done_at = t

    def report(self, out, standoffs):
        """Replace the x-course fields of run.episode's result with the looped-course ones."""
        prog = float(self.u_best - self.u_start)     # plain Python types only: the result is unpickled
                                                    # where numpy may be missing (the Modal client)
        tail = float(np.mean([w for _, w in self.tail])) if self.tail else 0.0
        mean_so = float(np.mean(standoffs)) if len(standoffs) else 1e9
        tracking = bool((not out.get("lost_at_end")) and tail >= 0.7 and mean_so < MEAN_STANDOFF_MAX)
        out.update(lap_m=round(self.c.lap_m, 1), lap_progress_m=round(prog, 1),
                   laps_followed=round(prog / self.c.lap_m, 2),
                   rover_laps=round(self.c.speed * len(standoffs) * 0.002 / self.c.lap_m, 2),
                   within_8m_pct=round(100.0 * float(np.mean(np.asarray(standoffs) < self.TRACK_M)), 1),
                   last10s_within_8m_pct=round(100 * tail, 1), tracking_at_end=bool(tracking),
                   lap_done_at_s=None if self.lap_done_at is None else round(float(self.lap_done_at), 1),
                   finished_at_s=(round(float(self.lap_done_at), 1) if (self.lap_done_at is not None and tracking)
                                  else None),
                   end_x_m=None, crossed_barrier=False, path=None,
                   direction=self.c.direction, start_u_m=round(self.c.u0, 1))
        return out


def make(name="town", seed=0):
    if "@" in name:                      # "town@real": realism.py
        import realism
        return realism.make(name, seed)
    if name != "town":
        raise KeyError("unknown town course %r (only 'town')" % name)
    return TownCourse(name, seed)


def render(name="town", seed=0, directory=".", width=1160, height=800, path_dots=True):
    """Top-down view of the town with the rover's figure-8 dotted on, as PNG bytes."""
    import io, mujoco
    from PIL import Image, ImageDraw
    c = make(name, seed)
    m = mujoco.MjModel.from_xml_path(c.write(directory))
    m.vis.map.zfar = 80.0
    d = mujoco.MjData(m)
    d.qpos[:3] = [0.0, 0.0, -5.0]            # the aircraft out of sight, under the floor
    c.drive(m, d, 0.0)
    mujoco.mj_forward(m, d)
    m.vis.headlight.diffuse[:] = [.25, .25, .25]     # straight down, the flight lighting washes the roofs out
    m.vis.headlight.ambient[:] = [.35, .35, .35]
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    sx, sy = c.MAP_SPAN
    # an orthographic-looking view: a narrow lens from far above; fit whichever span binds
    fovy = 12.0
    m.vis.global_.fovy = fovy
    half_h = max(sy / 2, (sx / 2) * height / width)
    cam.lookat[:] = [0, 0, 0]
    cam.elevation, cam.azimuth = -90.0, 90.0
    cam.distance = half_h / np.tan(np.deg2rad(fovy) / 2)
    r.update_scene(d, cam)
    img = Image.fromarray(r.render())
    dr = ImageDraw.Draw(img)
    # the lens sees the floor at distance D; points above it look bigger, so the map is exact on the floor
    W, H = img.size
    scale = H / (2 * half_h)
    px = lambda x, y: (W / 2 + x * scale, H / 2 - y * scale)  # noqa: E731
    for sx_ in (-6.0, 6.0):                  # the street's kerbs (the floor is one plane: drawn on, not modelled)
        for yy in np.arange(-WALL_Y, WALL_Y, 1.0):
            dr.line([px(sx_, yy), px(sx_, yy + 0.5)], fill=(200, 200, 200), width=1)
    if path_dots:
        for k in range(0, len(c.pts), 8):
            u, v = px(*c.pts[k])
            dr.ellipse([u - 1.5, v - 1.5, u + 1.5, v + 1.5], fill=(235, 40, 30))
        # direction arrows every 10 m, and the rover's start
        for uu in np.arange(0.0, c.lap_m, 10.0):
            p, h = c._at(c.u0 + uu)
            u, v = px(*p)
            a = np.array([np.cos(h), -np.sin(h)]) * 7
            nrm = np.array([-a[1], a[0]]) * 0.6
            dr.polygon([(u + a[0], v + a[1]), (u - a[0] + nrm[0], v - a[1] + nrm[1]),
                        (u - a[0] - nrm[0], v - a[1] - nrm[1])], fill=(255, 230, 60))
        u, v = px(*c.rover_pose(0.0)[:2])
        dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(255, 255, 255), width=2)
    dr.text((8, 6), "town seed %d  |  lap %.0f m  |  rover %.2f m/s" % (seed, c.lap_m, c.speed), fill=(255, 255, 255))
    for txt, (x, y) in (("blue house", (12, 0)), ("yellow house", (-12, 0)), ("bus", (-3.2, 14 + c.bus_dy)),
                        ("truck", (2.5, -14 + c.truck_dy))):
        u, v = px(x, y)
        dr.text((u - 3 * len(txt), v - 6), txt, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
