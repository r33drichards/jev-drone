"""A lookahead teacher: fly by trying every candidate command in the simulator and keeping the best.
(Each rollout commits to the candidate for COMMIT_S and then brakes to a hover: an option is only as good as the
safe stop it leaves -- holding a command for the whole horizon left no safe option in 16% of decisions.)

Hand-written rules (the course oracles, avoid.label_cmd) cap what Laya can learn: imitation reaches at most the
rule it imitates. This teacher has no rule. Every DECIDE_S it saves the whole simulator (MuJoCo state + the
flight controller's internals), rolls each candidate command forward HORIZON_S with the real physics and the
rover driving as it really will, scores what happened (score()), restores the state, and flies the best
command until the next decision -- a one-step model-predictive search, the shallow end of the AlphaZero idea
(github.com/ericjang/autogo: search produces the targets, the network learns them). It uses privileged
information Laya never has (where the rover is and will be); that is the point of a teacher.

A command is (forward speed m/s, side slide m/s + = left, heading offset deg from the rover's direction), all in
the drone's frame, the same kind of thing Laya would output. Altitude follows the course's altitude_target.

    import teacher
    r = teacher.fly("town-x4", seed=0, seconds=60)      # the ceiling check: the teacher flies itself

fly() also returns, per decision, every candidate's score (the soft target a student would train on).
"""
import copy, time
import numpy as np
import mujoco

SPEEDS = [-1.0, 0.5, 2.5, 4.5, 6.5]
SLIDES = [-1.8, 0.0, 1.8]
HEADINGS = [-35.0, 0.0, 35.0]
CANDIDATES = [(v, s, h) for v in SPEEDS for s in SLIDES for h in HEADINGS]
DECIDE_S = 0.25              # a decision four times a second
HORIZON_S = 2.0              # each rollout's length (1.2 s missed traps: stopping from 4 m/s takes ~1 s and 2 m):
COMMIT_S = 0.5               # the candidate for this long, then a brake to hover (can it still stop safely?)
BRAKE = (0.0, 0.0, 0.0)
DT = 0.002                   # the physics step (the course XML's timestep)
CTRL_EVERY = 10              # the command is turned into a velocity setpoint at 50 Hz, as run.Guidance does
YAW_RATE = np.deg2rad(120.0)
STANDOFF = 4.0
HALF_FOV = np.deg2rad(62.0)  # probe.TAN_H's horizontal half-field
NOSE = 0.25                  # flight.Eye.NOSE_OFFSET_M
CELL_M = 0.5                 # navigation grid resolution
CLEAR_M = 0.5                # a cell is blocked if geometry is within this of it at flying height (8 rays)
FLY_Z = 1.6                  # the navigation map's height (run.CRUISE_ALT); beams are marked passable
LEAD_S = 1.0                 # score the path distance to where the rover will be this long after the rollout


class NavGrid:
    """Where the drone can fly at FLY_Z, as a CELL_M grid over the scene's geometry (a cell is blocked when a ray
    from its centre in any of 8 directions meets geometry within CLEAR_M), beams passable (the altitude control
    flies over them). field(xy) -> path distances from every cell to xy (8-connected Dijkstra); dist(field, xy)
    looks one up. Path distance rewards the detour round a pocket that a straight-line distance punishes."""

    def __init__(self, w):
        m, d = w.m, w.d
        mujoco.mj_forward(m, d)
        lo, hi = np.array([1e9, 1e9]), np.array([-1e9, -1e9])
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or m.geom_bodyid[g] in (w.x2, w.rover_body):
                continue
            p, rb = d.geom_xpos[g][:2], m.geom_rbound[g]
            lo, hi = np.minimum(lo, p - rb), np.maximum(hi, p + rb)
        self.lo = lo - 2.0
        n = np.ceil((hi + 2.0 - self.lo) / CELL_M).astype(int)
        self.nx, self.ny = int(n[0]), int(n[1])
        blocked = np.zeros((self.nx, self.ny), dtype=bool)
        gid = np.array([-1], dtype=np.int32)
        dirs = [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.arange(8) * np.pi / 4]
        beams = [(sx - 0.8, sx + 0.8) for kind, sx, _ in getattr(w.c, "stations", []) if kind == "beam"]
        for i in range(self.nx):
            x = self.lo[0] + (i + 0.5) * CELL_M
            if any(a <= x <= b for a, b in beams):
                continue
            for j in range(self.ny):
                p = np.array([x, self.lo[1] + (j + 0.5) * CELL_M, FLY_Z])
                for v in dirs:
                    r = mujoco.mj_ray(m, d, p, v, None, 1, w.x2, gid)
                    if 0 <= r < CLEAR_M and m.geom_bodyid[gid[0]] != w.rover_body:
                        blocked[i, j] = True
                        break
        self.blocked = blocked

    def cell(self, xy):
        ij = np.floor((np.asarray(xy[:2]) - self.lo) / CELL_M).astype(int)
        return int(np.clip(ij[0], 0, self.nx - 1)), int(np.clip(ij[1], 0, self.ny - 1))

    def field(self, xy):
        import heapq
        dist = np.full((self.nx, self.ny), np.inf)
        s = self.cell(xy)
        dist[s] = 0.0
        h = [(0.0, s)]
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while h:
            dd, (i, j) = heapq.heappop(h)
            if dd > dist[i, j]:
                continue
            for di, dj, c in steps:
                a, b = i + di, j + dj
                if 0 <= a < self.nx and 0 <= b < self.ny and not self.blocked[a, b]:
                    nd = dd + c * CELL_M
                    if nd < dist[a, b]:
                        dist[a, b] = nd
                        heapq.heappush(h, (nd, (a, b)))
        return dist

    def dist(self, field, xy):
        v = field[self.cell(xy)]
        return float(v) if np.isfinite(v) else 60.0


def _yaw(q):
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class World:
    """One flight's model, data, controller and course, with save / restore for rollouts."""

    def __init__(self, course, seed):
        import courses, flight
        self.c = courses.make(course, seed)
        self.m = mujoco.MjModel.from_xml_path(self.c.write("."))
        self.d = mujoco.MjData(self.m)
        self.scratch = mujoco.MjData(self.m)
        self.pilot = flight.Pilot(self.m)
        x2 = self.m.body("x2").id
        self.x2_geoms = set(np.nonzero(self.m.geom_bodyid == x2)[0].tolist())
        self.x2 = x2
        self.rover_body = self.m.body("rover").id
        self.alt_fn = getattr(self.c, "altitude_target", None)
        self.yaw_cmd = None

    def pilot_state(self):
        p = self.pilot
        return (p.v_cmd.copy(), None if p.b3_prev is None else p.b3_prev.copy(), p.recovering, self.yaw_cmd)

    def set_pilot_state(self, s):
        p = self.pilot
        p.v_cmd, p.b3_prev, p.recovering, self.yaw_cmd = s[0].copy(), None if s[1] is None else s[1].copy(), s[2], s[3]

    def setpoint(self, cmd, t):
        """Velocity (world) and yaw setpoints for a command at time t (privileged: aims at the true rover)."""
        d = self.d
        pos, yaw = d.qpos[:3], _yaw(d.qpos[3:7])
        rv = self.c.rover_pose(t)
        want = float(np.arctan2(rv[1] - pos[1], rv[0] - pos[0])) + np.deg2rad(cmd[2])
        prev = yaw if self.yaw_cmd is None else self.yaw_cmd
        prev = yaw + float(np.clip(_wrap(prev - yaw), -0.6, 0.6))
        step = YAW_RATE * DT * CTRL_EVERY
        self.yaw_cmd = prev + float(np.clip(_wrap(want - prev), -step, step))
        alt = float(self.alt_fn(pos)) if self.alt_fn is not None else 1.6
        vz = float(np.clip(1.6 * (alt - pos[2]), -2.0, 2.0))
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], vz]), self.yaw_cmd

    def step(self, cmd, t, i):
        """One physics step under `cmd` (setpoint refreshed every CTRL_EVERY steps). -> hit geom or None."""
        if i % CTRL_EVERY == 0 or getattr(self, "_sp", None) is None:
            self._sp = self.setpoint(cmd, t)
        self.c.drive(self.m, self.d, t)
        self.d.ctrl[:] = self.pilot(self.d, self._sp[0], self._sp[1], DT)
        mujoco.mj_step(self.m, self.d)
        for k in range(self.d.ncon):
            g1, g2 = self.d.contact[k].geom1, self.d.contact[k].geom2
            if (g1 in self.x2_geoms) != (g2 in self.x2_geoms):
                return g2 if g1 in self.x2_geoms else g1
        return None

    def sees_rover(self, t):
        """Rover within the camera's horizontal field and in line of sight (a ray from the camera)."""
        d = self.d
        pos, yaw = d.qpos[:3], _yaw(d.qpos[3:7])
        cam = pos + NOSE * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        rv = np.asarray(self.c.rover_pose(t), dtype=float) + np.array([0.0, 0.0, 0.35])
        v = rv - cam
        dist = float(np.linalg.norm(v))
        if abs(_wrap(np.arctan2(v[1], v[0]) - yaw)) > HALF_FOV or dist > 30.0:
            return False, dist
        gid = np.array([-1], dtype=np.int32)
        r = mujoco.mj_ray(self.m, d, cam, v / dist, None, 1, self.x2, gid)
        ok = r < 0 or r >= dist - 0.3 or (gid[0] >= 0 and self.m.geom_bodyid[gid[0]] == self.rover_body)
        return bool(ok), dist


def score(hit_t, seen, dist_end, seen_end):
    """A rollout's value: a collision dominates (earlier is worse); then the rover in view at the end and over
    the rollout, and the PATH distance (NavGrid, round walls) to where the rover will be LEAD_S after the rollout
    near STANDOFF."""
    if hit_t is not None:
        return -1000.0 + 100.0 * hit_t
    return 20.0 * seen_end + 10.0 * seen - 2.0 * abs(dist_end - STANDOFF)


def rollout(w, cmd, t0):
    """Roll `cmd` forward HORIZON_S from the current state (which the caller restores). -> score."""
    n = int(HORIZON_S / DT)
    k = int(COMMIT_S / DT)
    seen_n = checks = 0
    for i in range(n):
        t = t0 + i * DT
        c = cmd if i < k else (BRAKE[0], BRAKE[1], cmd[2])
        if i == k:
            w._sp = None                                    # refresh the setpoint at the switch
        if w.step(c, t, i if i < k else i - k) is not None:
            return score(i * DT, 0, 0, False)
        if i % 100 == 99:                                   # 5 visibility checks a second
            s, _ = w.sees_rover(t)
            seen_n += s
            checks += 1
    s_end, dist = w.sees_rover(t0 + n * DT)
    if getattr(w, "nav_field", None) is not None:
        dist = w.nav.dist(w.nav_field, w.d.qpos[:2])
    return score(None, seen_n / max(checks, 1), dist, s_end)


def decide(w, t):
    """Try every candidate from the current state; restore it. -> (best command, scores)."""
    if getattr(w, "nav", None) is not None:
        w.nav_field = w.nav.field(w.c.rover_pose(t + HORIZON_S + LEAD_S))
    mujoco.mj_copyData(w.scratch, w.m, w.d)
    ps = w.pilot_state()
    scores = []
    for cmd in CANDIDATES:
        scores.append(rollout(w, cmd, t))
        mujoco.mj_copyData(w.d, w.m, w.scratch)
        w.set_pilot_state(ps)
    return CANDIDATES[int(np.argmax(scores))], scores


def fly(course, seed=0, seconds=60.0, alt=1.6, path=True, record=False):
    """The teacher flies the course itself (the ceiling check). -> an episode-like result dict. `path`: score
    the path distance on a NavGrid (else the straight-line distance, the first version). `record`: also render
    the onboard camera and lidar at every decision (flight.Eye) and return out["samples"]: per decision the JPEG
    frame, the command context (command.CMD_KEYS) and the soft targets (command.soft_targets)."""
    w = World(course, seed)
    eye = None
    if record:
        import flight
        eye = flight.Eye(w.m, rgb_size=(512, 384))
    samples, prev = [], (0.0, 0.0)
    rng = np.random.default_rng(seed)
    if hasattr(w.c, "start_pose"):
        w.d.qpos[:7] = w.c.start_pose(rng, alt)
    else:
        w.d.qpos[:3] = [1.5 + rng.uniform(-.3, .3), rng.uniform(-.5, .5), alt]
        w.d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(w.m, w.d)
    t_nav = time.time()
    w.nav = NavGrid(w) if path else None
    t_nav = time.time() - t_nav
    lap = w.c.lap_tracker(w.d.qpos[:2].copy()) if getattr(w.c, "looped", False) else None
    end_x = getattr(w.c, "end_x", None)
    n = int(seconds / DT)
    per = int(DECIDE_S / DT)
    cmd, hits, hit_objs, seen, checks, grounded = (0.5, 0.0, 0.0), 0, set(), 0, 0, 0
    finished_at = crashed_at = None
    decisions, standoffs = [], []
    wall0 = time.time()
    for i in range(n):
        t = i * DT
        if i % per == 0:
            if eye is not None:
                pos0, yaw0 = w.d.qpos[:3].copy(), _yaw(w.d.qpos[3:7])
                scene = eye.look(w.d, pos0, yaw0, t)
            cmd, sc = decide(w, t)
            if eye is not None:
                samples.append(_sample(w, eye, scene, pos0, yaw0, t, sc, prev, course, seed))
                prev = (cmd[0], samples[-1]["turn_chosen"])
            decisions.append({"t": round(t, 2), "cmd": cmd, "best": round(max(sc), 1),
                              "n_safe": int(sum(s > -500 for s in sc))})
        hit = w.step(cmd, t, i % per)
        if hit is not None and hit not in hit_objs:
            hit_objs.add(hit)
            hits += 1
        pos = w.d.qpos[:3]
        rv = w.c.rover_pose(t)
        standoffs.append(float(np.linalg.norm(pos - rv)))
        if i % 100 == 0:
            s, _ = w.sees_rover(t)
            seen += s
            checks += 1
        if i % 50 == 0 and lap is not None:
            lap.update(t, pos.copy(), standoffs[-1])
        if end_x is not None and not getattr(w.c, "looped", False) and finished_at is None and pos[0] >= end_x:
            finished_at = t
        if pos[2] < 0.35:
            grounded += 1
            if grounded > 750:
                crashed_at = t
                break
        else:
            grounded = max(0, grounded - 2)
    out = {"course": course, "seed": seed, "controller": "lookahead-teacher", "flew_s": round(len(standoffs) * DT, 1),
           "finished_at_s": None if finished_at is None else round(finished_at, 1),
           "crashed_at_s": crashed_at, "collisions": hits,
           "target_visible_pct": round(100 * seen / max(checks, 1), 1),
           "mean_standoff_m": round(float(np.mean(standoffs)), 2), "max_x_m": None,
           "decisions": len(decisions), "no_safe_option_pct": round(100 * float(np.mean([d["n_safe"] == 0 for d in decisions])), 1),
           "wall_s": round(time.time() - wall0, 1), "candidates": len(CANDIDATES),
           "horizon_s": HORIZON_S, "decide_s": DECIDE_S, "path_score": bool(path),
           "nav_grid_s": round(t_nav, 1)}
    if lap is not None:
        lap.report(out, standoffs)
    out["ok"] = bool(out.get("finished_at_s") is not None or out.get("lap_done_at_s"))
    if record:
        out["samples"] = samples
    return out


def _sample(w, eye, scene, pos, yaw, t, scores, prev, course, seed):
    """One decision as a training sample for the student (command.py)."""
    import io, command, avoid
    from PIL import Image
    rv = w.c.rover_pose(t)
    to_rover = float(np.arctan2(rv[1] - pos[1], rv[0] - pos[0]))
    turns = [float(np.rad2deg(_wrap(to_rover + np.deg2rad(c[2]) - yaw))) for c in CANDIDATES]
    sp, rel = avoid.travel(w.d.qvel[:3], yaw)
    ctx = {"altitude_m": round(float(pos[2]), 2), "speed_mps": round(sp, 2), "travel_deg": round(rel, 0),
           "prev_speed": round(float(prev[0]), 1), "prev_turn": round(float(prev[1]), 0), "lidar": avoid.sensor(scene)}
    buf = io.BytesIO()
    Image.fromarray(eye.last_rgb).save(buf, "JPEG", quality=90)
    best = int(np.argmax(scores))
    return {"course": course, "seed": seed, "t": round(t, 2), "jpeg": buf.getvalue(), "context": ctx,
            "targets": command.soft_targets(CANDIDATES, scores, turns), "best": CANDIDATES[best],
            "turn_chosen": round(turns[best], 1), "best_score": round(float(max(scores)), 1),
            "visible": bool(scene["target"]["visible"]), "bearing_deg": scene["target"]["bearing_deg"]}
