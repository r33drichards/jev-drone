"""Collision avoidance as Laya's decision, from the drone's 360-degree lidar.

The code avoided collisions from the forward camera's depth image (flight.Eye: five sector ranges and the clear
distance along the flight path; slide away under 4 m, reflex under 2.2 m) -- which only covers the camera's
~+-62 degrees, so at speed the drone clipped walls side-on at corners. The drone now also carries a scanning
lidar (flight.Lidar: 360 horizontal beams all round at 1 degree, 20 m, 2% noise, 10 scans a second, like a small
spinning unit). Its summary --
the nearest return in 24 sectors of 15 degrees round the drone -- and the drone's speed and direction of travel
go into Laya's context, and Laya answers `avoid` -- keep course,
dodge left, dodge right or brake -- trained on every frame against a label from the full scan: the drone's
body swept along its direction of travel out to the stopping distance (label_scan()). run.Guidance then flies the answer (avoid="laya"); the code's own slide and speed cap are off, and
only an emergency reflex at EMERGENCY_M remains as a backstop.

The question rides in the same predict as altitude.question() (altitude.LayaAltitude with avoid=True), so it
costs no extra GPU call; both read CONTEXT_KEYS.
"""
import numpy as np

BRAKE_ACC = 4.0              # m/s^2 the drone can shed forward speed at
REACTION_S = 0.25            # one camera frame plus an answer's latency, roughly
MARGIN_M = 1.0               # keep this much air after stopping
HALF_WIDTH_M = 0.75          # flight.Lidar.HALF_WIDTH_M: the drone's half-width plus margin
DODGE_GAP_M = 1.5            # lateral clearance a dodge needs (the drone shifting a half-width and more)
SLOW_MPS = 1.0               # below this, "brake" is no longer the answer
EMERGENCY_M = 1.0            # the code's last-resort reflex distance when Laya avoids (was 2.2)
OPTIONS = ["keep_course", "dodge_left", "dodge_right", "brake"]
_DESC = [
    "Nothing along the drone's direction of travel is close enough to matter at this speed: keep following.",
    "Something along the drone's direction of travel is closer than it can stop from, and there is room on the "
    "left: slide left around it.",
    "Something along the drone's direction of travel is closer than it can stop from, and there is room on the "
    "right: slide right around it.",
    "The way ahead is blocked closer than the drone can stop from and neither side has room: brake hard.",
]


def sensor(scene):
    """The lidar summary Laya reads (flight.Lidar.summary, put in the scene by flight.Eye.look)."""
    return dict(scene["lidar"])


def forward_speed(vel, yaw):
    """The drone's speed along its nose (m/s) from its world velocity."""
    return float(vel[0] * np.cos(yaw) + vel[1] * np.sin(yaw))


def travel(vel, yaw):
    """(speed m/s, direction of travel relative to the nose, deg, + = left) from the world velocity."""
    vx, vy = float(vel[0]), float(vel[1])
    sp = float(np.hypot(vx, vy))
    rel = float(np.rad2deg((np.arctan2(vy, vx) - yaw + np.pi) % (2 * np.pi) - np.pi)) if sp > 0.05 else 0.0
    return sp, rel


def stop_distance(v):
    v = max(float(v), 0.0)
    return v * v / (2 * BRAKE_ACC) + REACTION_S * v + MARGIN_M


SPEED_LEVELS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.5, 7.0]        # safe_speed score levels (m/s)
SLIDE_LEVELS = [-2.6, -1.3, -0.5, 0.0, 0.5, 1.3, 2.6]      # slide score levels (m/s, + = left)
REACT_M = 4.0               # returns closer than this within REACT_CONE of the nose make the drone ease away
REACT_CONE = np.deg2rad(60.0)   # alongside (to 90 deg) slowed the drone below the rover's speed in narrow streets
SPEED_CAP = 7.0             # run.FWD_CAP_FAST
# (a cone round the travel direction in the stopping check, like the camera-depth braking limit, was tried and
# did worse: town-x4 1/12 vs 4/12)


def _frame(lidar, vel, yaw):
    sp, rel = travel(vel, yaw)
    th = np.deg2rad(rel) if sp > 0.3 else 0.0
    ang_nose = lidar.ang + (lidar.yaw - yaw)                # beams relative to the nose, now (turned since)
    r = lidar.last
    hit = r < lidar.RANGE_M - 1e-6
    return sp, th, ang_nose, r, hit


def label_cmd(lidar, vel, yaw):
    """The trained avoidance command from the true 360-degree scan and the drone's world velocity:
    (safe_speed m/s, slide m/s + = left). safe_speed: the fastest the drone can go and still stop, braking at
    BRAKE_ACC, short of the nearest return in its body's strip (HALF_WIDTH_M either side) along its direction of
    travel, keeping 1.5 m; eased further (x (1 - 0.7 urgency)) when something is within REACT_M and REACT_CONE
    of the nose. slide: away from that nearest return, toward the roomier side, 2.6 m/s x urgency x the room on
    that side / 4 m -- the code's own graded avoidance (run.Guidance's reactive layer and braking limit), but
    from the lidar all round instead of the camera's forward depth."""
    sp, th, ang_nose, r, hit = _frame(lidar, vel, yaw)
    a_t = ang_nose - th
    along, lat = r * np.cos(a_t), r * np.sin(a_t)
    strip = hit & (along > 0) & (np.abs(lat) < HALF_WIDTH_M)
    free = float(along[strip].min()) if strip.any() else lidar.RANGE_M
    a_n = (ang_nose + np.pi) % (2 * np.pi) - np.pi           # -pi..pi, 0 = nose
    cap = float(np.clip(np.sqrt(2 * BRAKE_ACC * max(0.0, free - 1.5)), 0.25, SPEED_CAP))
    front = hit & (np.abs(a_n) <= REACT_CONE)                 # ahead (a fence alongside is not a threat)
    wr = float(r[front].min()) if front.any() else lidar.RANGE_M
    slide = 0.0
    if wr < REACT_M:
        urgency = (REACT_M - wr) / REACT_M
        left = float(r[hit & (a_n > np.pi / 8) & (a_n < 5 * np.pi / 8)].min()) if (hit & (a_n > np.pi / 8) & (a_n < 5 * np.pi / 8)).any() else lidar.RANGE_M
        right = float(r[hit & (a_n < -np.pi / 8) & (a_n > -5 * np.pi / 8)].min()) if (hit & (a_n < -np.pi / 8) & (a_n > -5 * np.pi / 8)).any() else lidar.RANGE_M
        side = 1.0 if left > right else -1.0
        slide = side * 2.6 * urgency * min(1.0, max(left, right) / 4.0)
        cap = min(cap, SPEED_CAP * (1.0 - 0.7 * urgency))
    return round(cap, 2), round(float(slide), 2)


def label_scan(lidar, vel, yaw):
    """The command as a coarse word, for reports only (label_cmd is what trains and flies)."""
    cap, slide = label_cmd(lidar, vel, yaw)
    if abs(slide) >= 1.0:
        return "dodge_left" if slide > 0 else "dodge_right"
    return "brake" if cap < 1.0 else "keep_course"


def read(probs, levels, sharpen=1.0):
    p = np.asarray(probs, dtype=float) ** float(sharpen)
    return float(np.dot(p / max(p.sum(), 1e-12), levels))


def question():
    """Two score questions answered from the frame + the lidar context: safe_speed (SPEED_LEVELS) and slide
    (SLIDE_LEVELS), read as probability-weighted levels (read())."""
    import probe
    ctx = probe.questions()["visible"]["instructions"].split("Is the red rover")[0]
    lid = ("The context has the drone's 360-degree lidar (metres to the nearest return in 24 sectors of 15 "
           "degrees, from dead ahead counter-clockwise: 6 is left, 12 behind, 18 right), its speed and its "
           "direction of travel relative to the nose. ")
    return {
        "safe_speed": {"type": "score", "instructions": ctx + lid + "How fast can the drone safely fly right now "
                       "and still stop short of anything along its direction of travel?",
                       "criteria": ["about %g m/s" % v for v in SPEED_LEVELS]},
        "slide": {"type": "score", "instructions": ctx + lid + "How fast should the drone slide sideways right now "
                  "to keep clear of nearby obstacles (positive = to its left)?",
                  "criteria": ["%s about %g m/s" % ("right" if v < 0 else ("left" if v > 0 else "no slide,"),
                                                    abs(v)) if v else "no slide" for v in SLIDE_LEVELS]},
    }
