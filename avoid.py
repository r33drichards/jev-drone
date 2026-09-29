"""Collision avoidance as Laya's decision, from the drone's depth sensor.

The drone's forward camera also gives a depth image (flight.Eye). The code used to act on it directly: it
reduces each depth frame to five sector ranges (far-left .. far-right) and the clear distance along the flight
path, then slides away from walls closer than 4 m, caps speed to what it can stop from, and takes over below
2.2 m (the reflex). Here that sensor summary and the drone's own forward speed go into Laya's context instead,
and Laya answers `avoid` -- keep course, dodge left, dodge right or brake -- trained on every frame against a
label computed from the true geometry (label()). run.Guidance then flies the answer (avoid="laya"); the code's
own slide and speed cap are off, and only an emergency reflex at EMERGENCY_M remains as a backstop.

The question rides in the same predict as altitude.question() (altitude.LayaAltitude with avoid=True), so it
costs no extra GPU call; both read CONTEXT_KEYS.
"""
import numpy as np

SECTORS = ("far_left", "left", "center", "right", "far_right")
SENSOR_CAP_M = 20.0          # ranges beyond this read as 20 (the sensor summary's scale)
BRAKE_ACC = 4.0              # m/s^2 the drone can shed forward speed at
REACTION_S = 0.25            # one camera frame plus an answer's latency, roughly
MARGIN_M = 1.0               # keep this much air after stopping
DODGE_ROOM_M = 2.5           # a side counts as open when this far clear
EMERGENCY_M = 1.0            # the code's last-resort reflex distance when Laya avoids (was 2.2)
OPTIONS = ["keep_course", "dodge_left", "dodge_right", "brake"]
_DESC = [
    "Nothing in the flight path is close enough to matter at this speed: keep following the rover.",
    "Something in the flight path is closer than the drone can stop from at this speed, and there is room on "
    "the left: slide left around it.",
    "Something in the flight path is closer than the drone can stop from at this speed, and there is room on "
    "the right: slide right around it.",
    "The flight path is blocked closer than the drone can stop from and neither side has room: brake hard.",
]


def sensor(scene):
    """The depth sensor summary Laya reads: the five sector ranges and the clear path ahead (m, 0.1 m steps)."""
    sec = scene["sector_range_m"]
    out = {k: round(float(min(sec[k], SENSOR_CAP_M)), 1) for k in SECTORS}
    out["path_ahead"] = round(float(min(scene["path_ahead_m"], SENSOR_CAP_M)), 1)
    return out


def forward_speed(vel, yaw):
    """The drone's speed along its nose (m/s) from its world velocity."""
    return float(vel[0] * np.cos(yaw) + vel[1] * np.sin(yaw))


def stop_distance(v):
    v = max(float(v), 0.0)
    return v * v / (2 * BRAKE_ACC) + REACTION_S * v + MARGIN_M


def label(sens, speed_mps):
    """The trained answer from the true sensor reading and speed: keep course while the path is clear for the
    stopping distance (plus 1 m), else dodge toward the side with room, else brake."""
    path = sens["path_ahead"]
    if path > stop_distance(speed_mps) + 1.0:
        return "keep_course"
    left = min(sens["far_left"], sens["left"])
    right = min(sens["far_right"], sens["right"])
    room = max(left, right)
    if room >= max(DODGE_ROOM_M, path + 0.5):
        return "dodge_left" if left >= right else "dodge_right"
    return "brake"


def question():
    import probe
    ctx = probe.questions()["visible"]["instructions"].split("Is the red rover")[0]
    return {"avoid": {"type": "choice",
                      "instructions": ctx + "The context has the depth sensor (metres to the nearest obstacle in "
                      "five sectors from far left to far right, and the clear distance along the flight path) and "
                      "the drone's forward speed. What should the drone do right now to avoid a collision?",
                      "criteria": dict(zip(OPTIONS, _DESC))}}
