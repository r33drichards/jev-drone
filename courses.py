"""Courses beyond world.xml, built so that no constant answer flies them.

On the classic course the only full-width obstacles are two low beams, where
climbing is right, and the floor is open for 90 m, so a climb is never punished.
Answering "climb" to everything and letting run.Guidance veto it clears that course
(results/laya/README.md). These courses add stations where the climb check PASSES
and climbing is still wrong:

  pocket   Two dead ends side by side with a 4.5 m lane between them. Each has a
           low front wall (top 2.1 m, so Guidance's climb check passes), a low
           divider along the lane and a tall wall 12 m back. The rover drives
           straight through one of them by floor hatches too low for the
           aircraft, so a trailing drone meets that wall head-on and then loses
           sight of the rover. Climb and the aircraft sinks inside after its climb
           hold, facing a wall it cannot climb; the right answer is the gap toward
           the lane, then back onto the rover past the far wall. Only
           free_ahead_above_m (~15 m, the far wall, against 45 m over a real beam)
           tells the two apart in the JSON; a camera frame shows it plainly.
  beam     The classic low beam across the whole corridor: the right answer is
           climb, so a model that never climbs fails too.
  decoy    A tall wall with the real gap on one side and a blind recess on the
           other that reads as MORE open space. Greedy "go to the wider side"
           picks the recess; the rover went through the gap.

Tall walls line the corridor (|y| = 6), so nothing can be flown around. Which side
each gap is on is drawn from the seed, so "always gap_left" fails too.

Every course has an oracle: the right maneuver for any drone position, from the
layout itself. `const:oracle` in run.py flies it, which shows the course can be
flown by run.Guidance when the tactical answers are right.
"""
import os
import numpy as np

HALF_W = 7.0         # corridor half-width
GAP_W = 4.5          # every real gap, against the corridor wall; 2.8 m could not be threaded past the 2.2 m reflex
TALL = 5.0           # above CLIMB_ALT (3.0) plus the camera's "above" band, so it can never be climbed
LOW_TOP = 2.1        # same top as the classic beams: the climb check passes over it
ROVER_SPEED = 1.15
START_X = 6.0

_HEAD = """<mujoco model="{name}">
  <include file="mujoco_menagerie/skydio_x2/x2.xml"/>
  <statistic extent="20" center="30 0 2"/>
  <option timestep="0.002" density="1.2" viscosity="1.8e-5"/>
  <visual>
    <global fovy="50" offwidth="1400" offheight="900"/>
    <headlight diffuse=".7 .7 .7" ambient=".4 .4 .4"/>
    <map znear="0.0025" zfar="3.0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".5 .65 .85" rgb2=".1 .15 .25" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".22 .24 .26" rgb2=".28 .30 .33" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="18 18" reflectance="0"/>
    <material name="wall"   rgba=".45 .47 .52 1"/>
    <material name="low"    rgba=".95 .60 .10 1"/>
    <material name="rover"  rgba=".95 .15 .10 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" directional="true" diffuse=".6 .6 .6"/>
    <geom name="floor" type="plane" size="90 90 .05" material="grid"/>
"""
_TAIL = """    <body name="rover" mocap="true" pos="6 0 .2">
      <geom name="rover_geom" type="box" material="rover" size=".35 .25 .2"/>
      <geom name="rover_mast" type="box" material="rover" size=".06 .06 .35" pos="0 0 .5"/>
    </body>
  </worldbody>
</mujoco>
"""


def _box(name, x0, x1, y0, y1, z1, mat="wall"):
    """An axis-aligned box from the floor to z1."""
    return ('    <geom name="%s" type="box" material="%s" size="%.3f %.3f %.3f" pos="%.3f %.3f %.3f"/>\n'
            % (name, mat, (x1 - x0) / 2, abs(y1 - y0) / 2, z1 / 2, (x0 + x1) / 2, (y0 + y1) / 2, z1 / 2))


def _lintel(name, x0, x1, y0, y1, z0, z1, mat="wall"):
    """A box from z0 to z1: the wall above a hatch."""
    return ('    <geom name="%s" type="box" material="%s" size="%.3f %.3f %.3f" pos="%.3f %.3f %.3f"/>\n'
            % (name, mat, (x1 - x0) / 2, abs(y1 - y0) / 2, (z1 - z0) / 2, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def _smooth(a, b, u):
    u = min(max(u, 0.0), 1.0)
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * u))


class Course:
    """A straight corridor of stations. `stations` is a list of (kind, x, side)."""

    def __init__(self, name, stations, end_x, seed=0):
        self.name, self.stations, self.end_x, self.seed = name, stations, end_x, seed
        self.first_barrier_x = min(x for _, x, _ in stations)
        # rover lateral waypoints (x, y): centre, except swing out through each gap
        self.way = [(0.0, 0.0)]
        for kind, x, side in stations:
            if kind == "pocket":
                # the rover drives through the pocket, in and out by hatches too low for the aircraft,
                # so a trailing drone meets the front wall head-on (the climb check passes) and
                # then loses sight of the rover behind it; the aircraft's way round is the gap
                yc = -side * HATCH_C
                self.way += [(x - 9.0, 0.0), (x - 4.0, yc), (x + POCKET_D + 2.0, yc), (x + POCKET_D + 7.0, 0.0)]
            elif kind == "decoy":
                y = side * (HALF_W - GAP_W / 2)
                # through the gap, inward behind the wall, past the baffle's inner end
                self.way += [(x - 5.0, 0.0), (x - 0.5, y), (x + 1.0, y), (x + 4.0, 0.0)]
        self.way.append((1e9, 0.0))

    # --- geometry -----------------------------------------------------------------------
    def xml(self):
        g = [_HEAD.format(name=self.name)]
        L = self.end_x + 10
        g.append(_box("wallL", -2, L, HALF_W, HALF_W + 0.4, TALL))
        g.append(_box("wallR", -2, L, -HALF_W - 0.4, -HALF_W, TALL))
        for i, (kind, x, side) in enumerate(self.stations):
            s = side
            if kind == "beam":
                g.append('    <geom name="beam%d" type="capsule" material="low" size=".50 %.2f" euler="90 0 0" '
                         'pos="%.2f 0 1.60"/>\n' % (i, HALF_W, x))
            elif kind == "pocket":
                # a pocket either side of a centre lane; the rover's pocket (-s) has the hatches
                for p in (1, -1):
                    inner, outer = p * LANE_W / 2, p * HALF_W
                    hatch = p == -s
                    h0, h1 = p * (HATCH_C - HATCH_W / 2), p * (HATCH_C + HATCH_W / 2)
                    for name, x0, x1, top, mat in (("low", x - 0.2, x + 0.2, LOW_TOP, "low"),
                                                   ("back", x + POCKET_D - 0.4, x + POCKET_D, TALL, "wall")):
                        tag = "%s%d%s" % (name, i, "L" if p > 0 else "R")
                        if hatch:     # two boxes either side of a floor hatch, and a lintel over it
                            g.append(_box(tag + "a", x0, x1, inner, h0, top, mat))
                            g.append(_box(tag + "b", x0, x1, h1, outer, top, mat))
                            g.append(_lintel(tag + "c", x0, x1, h0, h1, HATCH_H, top, mat))
                        else:
                            g.append(_box(tag, x0, x1, inner, outer, top, mat))
                    # the divider is LOW too: a tall one shows in the camera's "above" band from the
                    # approach and the climb check (rightly) refuses. The only tall wall is the far end,
                    # deep enough that the check passes; after the climb hold the aircraft sinks inside.
                    g.append(_box("div%d%s" % (i, "L" if p > 0 else "R"), x, x + POCKET_D, inner, inner + 0.2 * p,
                                  LOW_TOP, "low"))
            elif kind == "decoy":
                edge = s * (HALF_W - GAP_W)      # real gap: edge .. s*HALF_W, then a dogleg
                n0, n1 = -s * (HALF_W - 0.4), -s * (HALF_W - 0.4 - GAP_W)   # the recess, as wide, wrong side
                g.append(_box("dwall%d" % i, x - 0.2, x + 0.2, n1, edge, TALL))
                g.append(_box("dstub%d" % i, x - 0.2, x + 0.2, -s * HALF_W, n0, TALL))
                # a baffle 4.5 m behind the real gap: seen from the approach, the gap reads only
                # ~4.5 m deep, the recess 7 m; the route turns inward behind the wall and runs past the baffle
                g.append(_box("baffle%d" % i, x + 4.5, x + 4.9, edge, s * HALF_W, TALL))
                # the recess reads 7 m deep, more open than the real gap, and is a dead end
                g.append(_box("nback%d" % i, x + 7.0, x + 7.4, -s * HALF_W, n1, TALL))
                g.append(_box("nside%d" % i, x, x + 7.4, n1, n1 + 0.2 * s, TALL))
        g.append(_TAIL)
        return "".join(g)

    def write(self, directory):
        path = os.path.join(directory, ".course_%s_%d.xml" % (self.name, self.seed))
        with open(path, "w") as f:
            f.write(self.xml())
        return path

    # --- motion -------------------------------------------------------------------------
    def rover_pose(self, t):
        x = START_X + ROVER_SPEED * t
        for (x0, y0), (x1, y1) in zip(self.way, self.way[1:]):
            if x0 <= x < x1:
                return np.array([x, _smooth(y0, y1, (x - x0) / max(x1 - x0, 1e-9)), 0.2])
        return np.array([x, 0.0, 0.2])

    def drive(self, m, d, t):
        d.mocap_pos[m.body("rover").mocapid[0]] = self.rover_pose(t)

    # --- the right answer ---------------------------------------------------------------
    def oracle(self, pos):
        """The answer that flies this course with run.Guidance as it is: climb at a beam, and at a
        pocket or decoy leave it to the pursuit and reactive layers (hold_course). Commanding
        gap_left/right there does worse: Guidance's gap slide has no notion of where to stop, so
        it carries the aircraft across the lane into a wall (checked with course_check.py)."""
        x = float(pos[0])
        for kind, sx, side in self.stations:
            if kind == "beam" and sx - 5.0 <= x <= sx + 0.6:
                return "climb"
        return "hold_course"


POCKET_D = 12.0
LANE_W = 4.5         # the centre lane between the two pockets: the way round
HATCH_C = 4.6        # hatch centre, mid-pocket
HATCH_W = 1.4        # the rover is 0.7 m wide
HATCH_H = 1.1        # the rover stands 1.05 m; the aircraft never flies below MIN_ALT = 1.15 m

# name -> list of (kind, x); a side of +1/-1 per gap station is drawn from the seed
LAYOUTS = {
    # two traps between two beams: climb is right twice and fatal twice
    "pockets": ([("beam", 16.0), ("pocket", 26.0), ("beam", 50.0), ("pocket", 60.0)], 80.0),
    # a trap, a beam, a decoy, a trap: needs climb once and the correct side three times
    "mixed": ([("pocket", 16.0), ("beam", 38.0), ("decoy", 48.0), ("pocket", 64.0)], 84.0),
    # no climbing anywhere: every full-width obstacle is a trap or a wall
    "no-climb": ([("pocket", 16.0), ("decoy", 36.0), ("pocket", 50.0)], 70.0),
}
# The rover reaches the end of the longest course at about t = 68 s; fly them for 90 s.
SECONDS = 90.0


def render(name, seed=0, directory=".", width=1400, height=320):
    """Top-down view of a course with the rover's path dotted on, as PNG bytes."""
    import io, mujoco
    from PIL import Image, ImageDraw
    c = make(name, seed)
    m = mujoco.MjModel.from_xml_path(c.write(directory))
    m.vis.map.zfar = 50.0                 # a fraction of stat.extent: see world.xml
    d = mujoco.MjData(m)
    d.qpos[:3] = [1.5, 0, 1.6]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    span = c.end_x + 4.0
    fovy = 30.0
    m.vis.global_.fovy = fovy
    cam.lookat[:] = [span / 2, 0, 0]
    cam.elevation, cam.azimuth = -90.0, 90.0
    cam.distance = (span / 2) / (np.tan(np.deg2rad(fovy) / 2) * width / height)
    r.update_scene(d, cam)
    img = Image.fromarray(r.render())
    # world (x, y) -> pixel: x runs left to right, +y up
    px = lambda x, y: (width / 2 + (x - span / 2) / span * width, height / 2 - y / span * width)  # noqa: E731
    dr = ImageDraw.Draw(img)
    for t in np.arange(0, (c.end_x - START_X) / ROVER_SPEED, 0.4):
        x, y, _ = c.rover_pose(t)
        u, v = px(x, y)
        dr.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(235, 40, 30))
    for kind, x, side in c.stations:
        u, _ = px(x, 0)
        dr.text((u - 12, 4), kind, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def make(name, seed=0):
    stations, end_x = LAYOUTS[name]
    rng = np.random.default_rng(1000 + seed)
    out = [(k, x, 0 if k == "beam" else int(rng.choice([-1, 1]))) for k, x in stations]
    return Course(name, out, end_x, seed)
