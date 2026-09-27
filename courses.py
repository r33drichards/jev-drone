"""Courses beyond world.xml, built so that no constant answer flies them.

On the classic course the only full-width obstacles are two low beams, where
climbing is right, and the floor is open for 90 m, so a climb is never punished.
Answering "climb" to everything and letting run.Guidance veto it clears that course
(results/laya/README.md). These courses add stations where the climb check PASSES
and climbing is still wrong:

  pocket   A low wall (top 2.1 m, so Guidance's climb check passes) is the front of
           a dead end: tall walls behind it on three sides. The rover drives
           through a gap beside it. Climb and you are boxed in above the pocket
           floor with no way forward; the right answer is gap_<side>.
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

HALF_W = 6.0         # corridor half-width
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
                y = side * (HALF_W - 1.4)
                self.way += [(x - 5.0, 0.0), (x - 1.0, y), (x + POCKET_D + 1.0, y), (x + POCKET_D + 5.0, 0.0)]
            elif kind == "decoy":
                y = side * (HALF_W - 1.4)
                # through the gap, inward behind the wall, past the baffle's inner end, back to centre
                self.way += [(x - 5.0, 0.0), (x - 0.5, y), (x + 1.0, y), (x + 4.0, side * 1.2),
                             (x + 6.0, side * 1.2), (x + 9.0, 0.0)]
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
                edge = s * (HALF_W - 2.8)        # the gap runs from here to the corridor wall
                far = -s * HALF_W
                g.append(_box("low%d" % i, x - 0.2, x + 0.2, far, edge, LOW_TOP, "low"))
                g.append(_box("div%d" % i, x, x + POCKET_D, edge - 0.2 * s, edge, TALL))
                g.append(_box("back%d" % i, x + POCKET_D - 0.4, x + POCKET_D, far, edge, TALL))
            elif kind == "decoy":
                edge = s * (HALF_W - 2.8)        # real gap: edge .. s*HALF_W, then a dogleg
                n0, n1 = -s * (HALF_W - 0.4), -s * (HALF_W - 3.4)   # the recess, on the wrong side
                g.append(_box("dwall%d" % i, x - 0.2, x + 0.2, n1, edge, TALL))
                g.append(_box("dstub%d" % i, x - 0.2, x + 0.2, -s * HALF_W, n0, TALL))
                # a baffle 4.5 m behind the real gap: seen from the approach, the gap reads only
                # ~4.5 m deep; the route turns inward behind the wall and runs past the baffle
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
        x = float(pos[0])
        for kind, sx, side in self.stations:
            if kind == "beam" and sx - 5.0 <= x <= sx + 0.6:
                return "climb"
            if kind in ("pocket", "decoy") and sx - 7.0 <= x <= sx + 1.0:
                return "gap_left" if side > 0 else "gap_right"
        return "hold_course"


POCKET_D = 8.0

# name -> list of (kind, x); a side of +1/-1 per gap station is drawn from the seed
LAYOUTS = {
    # two traps between two beams: climb is right twice and fatal twice
    "pockets": ([("beam", 16.0), ("pocket", 26.0), ("beam", 42.0), ("pocket", 52.0)], 66.0),
    # a trap, a decoy, and a beam: needs climb, needs the correct side, twice
    "mixed": ([("pocket", 16.0), ("beam", 32.0), ("decoy", 40.0), ("pocket", 52.0)], 66.0),
    # no climbing anywhere: every full-width obstacle is a trap or a wall
    "no-climb": ([("pocket", 16.0), ("decoy", 30.0), ("pocket", 42.0)], 56.0),
}


def make(name, seed=0):
    stations, end_x = LAYOUTS[name]
    rng = np.random.default_rng(1000 + seed)
    out = [(k, x, 0 if k == "beam" else int(rng.choice([-1, 1]))) for k, x in stations]
    return Course(name, out, end_x, seed)
