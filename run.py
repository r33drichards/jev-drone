"""Follow-and-avoid episode: Skydio X2 chases a ground rover through a pillar field.

Layering:
  500 Hz  geometric controller           (flight.Pilot)
   50 Hz  guidance + hard safety reflex  (this file -- always owns safety)
   15 Hz  onboard camera -> scene        (flight.Eye)
  ~3 Hz   Jev tactical judgment          (tactics.Tactician, advisory only)
"""
import os, sys, json, time, argparse
import numpy as np
import mujoco
import flight
from tactics import THRESHOLDS as THRESH, decision_needed

STANDOFF = 3.5
MIN_ALT = 1.15           # never command a descent below this; the floor is invisible to a
                         # downward-blind camera, so altitude has to be protected in code
CRUISE_ALT = 1.6
CHASE_FOVY = 50.0        # cinematic lens for the third-person render
CLIMB_ALT = 3.0          # beams top out at 2.1; this clears them with margin
CLIMB_HOLD_STEPS = 90    # ~1.8 s at 50 Hz guidance
GUIDE_DT = 0.02          # Guidance runs every 10 physics steps of 2 ms
REFLEX_M = 2.2           # code-owned: below this, Jev's opinion is irrelevant
TRACE = int(os.environ.get("TRACE") or 0)   # 1: every judgment change; 2: also the state twice a second


ROVER_SPEED = 1.15
SEARCH_SPEED = 2.4
FWD_CAP_FAST = 7.0       # forward-speed cap when speed_scale > 1
BRAKE_ACC, BRAKE_MARGIN_M = 4.0, 1.5    # speed_scale > 1: v <= sqrt(2 a (path_ahead - margin))
TRAIL_LOOKAHEAD_M = 3.0   # tune aim="trail": steer at the rover's trail this far ahead
EMERGENCY_REFLEX_M = 1.0  # avoid="laya": the code's last-resort reflex distance (avoid.EMERGENCY_M)
TARGET_MAX_SPEED = 1.6   # the rover cannot move faster than this; clamp the estimate       # while searching, still out-pace the rover


def rover_pose(t):
    return np.array([6.0 + ROVER_SPEED * t, 2.0 * np.sin(0.26 * t), 0.2])


def _qz(theta):
    """Quaternion for Rz(theta) applied to a capsule already lying along y."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    h = np.sqrt(0.5)
    # Rz(theta) (x) Rx(90deg), in w x y z
    return np.array([c * h, c * h, s * h, s * h])


def drive_course(m, d, t):
    """Move every mocap obstacle. Kept slow: mocap bodies teleport rather than
    move, so a fast sweep can materialise inside the drone and fling it."""
    d.mocap_pos[m.body("rover").mocapid[0]] = rover_pose(t)
    for i, name in enumerate(("arm0", "arm1")):
        mid = m.body(name).mocapid[0]
        d.mocap_quat[mid] = _qz(0.45 * t + i * 1.9)
    # the gap tracks the rover, so the leader always fits; the drone trails by
    # several seconds and therefore meets the gap somewhere else entirely
    c = 2.0 * np.sin(0.26 * t)
    for name, off in (("gateL", 31.6), ("gateR", -31.6)):
        mid = m.body(name).mocapid[0]
        p = d.mocap_pos[mid].copy()
        p[1] = c + off
        d.mocap_pos[mid] = p


# reappear-guided reacquisition (Guidance reacq=): act once the pursuit source has not seen the rover for
# `after_s`, on an answer at most `max_age_s` old; left / right = `turn_deg` off the answer frame's nose (the
# v3 label splits at 20 deg, and left spans 20-90); behind = turn around toward the side it was last seen on;
# at most `max_step_rad` of turn per camera frame; forward speed capped at `cap` (laya_pursuit.RangeSpeed's
# lost_cap) x clip(cos(heading error), `min_speed_frac`, 1).
#   `occluded_max`: act only when the answer's P(occluded) is at most this: the rover is off to the side or
#       behind, not hidden behind something in view. Turning toward a rover behind a pocket wall faces the wall
#       (the same failure as the extrapolated search); with the true labels on CPU, acting regardless lost 9 of
#       20 code-pursuit flights that finish without it, gated it lost none. 1 acts regardless.
#   `ahead`: what an "ahead" answer does: "none" (nothing: the code's lost-target behaviour), "answer" (hold
#       the answer frame's heading at the capped speed) or "live" (hold the live heading at the capped speed).
#   `clear_m`: act on left / right only while the camera sees at least this much room on that side (the more
#       open of its two sectors); 0 = regardless.
REACQ_DEFAULTS = {"after_s": 1.0, "max_age_s": 1.5, "turn_deg": 45.0, "max_step_rad": 0.8, "cap": 2.4,
                  "min_speed_frac": 0.3, "ahead": "none", "clear_m": 0.0, "occluded_max": 0.5}


class Guidance:
    """Turns a scene summary (+ an optional Jev judgment) into a velocity command.

    Rule the whole thing obeys: never translate faster sideways than the camera can
    see. The aircraft has a 91 deg forward view, so evasion is done by YAWING toward
    open space and flying into it, not by sliding blindly.
    """

    def __init__(self, eye, search_lead_s=5.0, search_on_hold=False, speed_law=None, tune=None, reacq=None,
                 speed_scale=1.0, avoid_mode=False, cmd_mode=False):
        self.eye = eye
        # cmd_mode: Laya's command (command.py: speed, slide, turn from its frame + lidar context) flies the
        # aircraft; pursuit, reactive avoidance and the braking limit are off, the reflex only under 1 m
        self.cmd_mode = bool(cmd_mode)
        self.n_cmd_steps = 0
        # avoid_mode: Laya's avoid answer (avoid.py) flies collision avoidance; the code's reactive slide and
        # braking limit are off and the reflex only fires under avoid.EMERGENCY_M
        self.avoid_mode = bool(avoid_mode)
        self.n_emergency = 0
        self.avoid_steps = {}
        self.trail = []                  # tune["aim"] == "trail": (t, world xy) of the rover's fixes, 6 s
        # a faster rover (town-x<k>): cruise / search speeds and the rover-speed clamp scale with it, the
        # forward cap with it up to FWD_CAP_FAST (the airframe's 28 deg tilt limit)
        k = float(speed_scale)
        self.k = k
        self.fwd_cap = min(3.6 * k, FWD_CAP_FAST) if k > 1 else 3.6
        self.search_speed = min(SEARCH_SPEED * k, FWD_CAP_FAST - 0.5) if k > 1 else SEARCH_SPEED
        self.target_max_speed = TARGET_MAX_SPEED * k
        # pursuit / avoidance tuning against S-shaped paths (pathmetrics.py); empty = the behaviour below
        # unchanged. Keys: "slide" ("off" | "hyst" | "center"), "slide_hyst_m", "center_gain",
        # "yaw_tau_s" (low-pass on the pursuit heading), "yaw_db_deg" (soft deadband on the bearing),
        # "yaw_rate_dps" (cap on how fast the yaw command may move, every branch: pursuit, search, reflex),
        # "aim" ("track": steer at the smoothed world fix, not the raw bearing), "aim_tau_s", "aim_lead_s"
        self.tune = dict(tune or {})
        self._aim_hdg = self._aim_w = self._slide_side = None
        self._aim_t = self._aim_wt = 0.0
        # forward speed from a model-supplied range (laya_pursuit.RangeSpeed); None: the code's law below
        self.speed_law = speed_law
        # how far (s) the lost-target search may carry the last fix forward along its velocity
        self.search_lead_s = float(search_lead_s)
        # a fresh hold_course judgment normally pre-empts the baseline search, so with the
        # const:oracle tactics a lost rover is never searched for; True lets it fall through
        self.search_on_hold = bool(search_on_hold)
        self.last_bearing = 0.0
        self.yaw_sp = None
        self.sweep = 0.0
        self.lost_for = 0.0
        self.climb_hold = 0
        self._yaw_cmd = None            # tune["yaw_rate_dps"]: the rate-limited yaw command
        self.commit = None
        self.commit_left = 0
        self.search_yaw = None
        # world-frame estimate of the target, built from our own pose + the camera
        # bearing/range. No ground truth; this is what the aircraft could work out.
        self.tgt_w = None
        self.tgt_v = np.zeros(2)
        self.tgt_t = 0.0
        self.tgt_hist = []
        # diagnostics only (never steer): world-fix updates, and the fix's age at each search heading
        self.n_fix = 0
        self.n_search_nofix = 0
        self.search_ages = []
        self.n_search_steps = 0         # guidance steps flown on a search heading (either branch)
        # reappear-guided reacquisition (episode reacquire=...): None = off, the behaviour above unchanged;
        # else REACQ_DEFAULTS overridden by the dict given. Diagnostics: steps flown on it, per side.
        self.reacq = None if reacq is None else dict(REACQ_DEFAULTS, **reacq)
        self.n_reacq_steps = 0
        self.reacq_side_steps = {}

    def _search_heading(self, yaw, t, pos):
        """Where the target probably is now: last fix, carried forward by the
        velocity we observed while we could still see it."""
        if self.tgt_w is None or pos is None:
            self.n_search_nofix += 1
            return self.search_yaw if self.search_yaw is not None else yaw
        self.search_ages.append(t - self.tgt_t)
        lead = float(np.clip(t - self.tgt_t, 0.0, self.search_lead_s))   # do not extrapolate forever
        aim = self.tgt_w + self.tgt_v * lead
        d = aim - pos[:2]
        if np.linalg.norm(d) < 0.5:
            return yaw
        return float(np.arctan2(d[1], d[0]))

    def _tuned_yaw(self, tgt, yaw, t, pos, fresh):
        """The pursuit heading (relative to the live yaw) with self.tune applied."""
        tn = self.tune
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
        if not tgt["visible"]:
            self._aim_hdg = self._aim_w = None
            return 0.0
        b = float(self.last_bearing)
        if tn.get("aim") == "track" and pos is not None and self.tgt_w is not None and t - self.tgt_t < 0.3:
            if fresh:
                if self._aim_w is None:
                    self._aim_w = self.tgt_w.copy()
                elif self.tgt_t > self._aim_wt:
                    a = 1.0 - np.exp(-(self.tgt_t - self._aim_wt) / max(tn.get("aim_tau_s", 0.6), 1e-3))
                    self._aim_w = self._aim_w + a * (self.tgt_w - self._aim_w)
                self._aim_wt = self.tgt_t
            aim = self._aim_w + self.tgt_v * float(tn.get("aim_lead_s", 0.0))
            dv = aim - pos[:2]
            if np.linalg.norm(dv) > 0.5:
                b = float(wrap(np.arctan2(dv[1], dv[0]) - yaw))
        db = np.deg2rad(float(tn.get("yaw_db_deg", 0.0)))
        if db > 0:
            b = float(np.sign(b) * max(0.0, abs(b) - db))
        tau = float(tn.get("yaw_tau_s", 0.0))
        if tau > 0:
            if fresh or self._aim_hdg is None:
                h = yaw + b
                if self._aim_hdg is None:
                    self._aim_hdg = h
                else:
                    self._aim_hdg += (1.0 - np.exp(-(t - self._aim_t) / tau)) * wrap(h - self._aim_hdg)
                self._aim_t = t
            b = float(wrap(self._aim_hdg - yaw))
        return float(np.clip(b, -0.6, 0.6))

    def _tuned_slide(self, left_room, right_room, urgency):
        """The reactive layer's strafe with self.tune["slide"]: "off" (none), "hyst" (keep the side
        already chosen until the other is roomier by slide_hyst_m), "center" (proportional to the
        room difference, so it fades to nothing mid-lane instead of flipping full-strength), "lane"
        (centre only when walled in on both sides, else unchanged)."""
        mode = self.tune["slide"]
        room = max(left_room, right_room)
        if mode == "off":
            return 0.0
        if mode == "hyst":
            h = float(self.tune.get("slide_hyst_m", 0.75))
            if self._slide_side is None:
                self._slide_side = 1.0 if left_room > right_room else -1.0
            elif self._slide_side > 0 and right_room > left_room + h:
                self._slide_side = -1.0
            elif self._slide_side < 0 and left_room > right_room + h:
                self._slide_side = 1.0
            side_room = left_room if self._slide_side > 0 else right_room
            return self._slide_side * 2.6 * urgency * min(1.0, side_room / 4.0)
        if mode == "center" or (mode == "lane" and room < float(self.tune.get("lane_m", 6.0))):
            # "lane": the centring law only when walled in on BOTH sides (the roomier side under lane_m);
            # an obstacle on one side still gets the full-strength strafe away from it
            k = float(self.tune.get("center_gain", 1.0))
            return float(np.clip(k * (left_room - right_room), -1.0, 1.0)) * 2.6 * urgency * min(1.0, room / 4.0)
        if mode == "lane":
            return (1.0 if left_room > right_room else -1.0) * 2.6 * urgency * min(1.0, room / 4.0)
        raise ValueError("tune['slide'] must be off, hyst, center or lane, got %r" % mode)

    def _reappear_heading(self, ans, tgt, yaw, t, sec=None):
        """The world heading to fly while the rover is lost, from the latest reappear answer: None when the
        rover is in view, lost for less than reacq["after_s"], or the answer is stale (older than max_age_s)
        or from before this loss began. The side is relative to the nose AT THE ANSWER'S FRAME, so the heading
        is fixed in the world and a new answer (every ~0.3 s) corrects it: left / right = that nose +- turn_deg,
        behind = turn around, toward the side the rover was last seen on, ahead = per reacq["ahead"]. None too
        when the answer says the rover is occluded (P > occluded_max) or, with clear_m, that side is walled off.
        -> (world heading, side) or None."""
        rq = self.reacq
        if rq is None or ans is None or tgt["visible"] or self.lost_for <= rq["after_s"]:
            return None
        if t - ans["t"] > rq["max_age_s"] or ans["t"] < t - self.lost_for - 1e-6:
            return None
        y0, side = float(ans["yaw"]), ans["side"]
        if ans.get("occluded") is not None and ans["occluded"] > rq["occluded_max"]:
            return None                                   # hidden behind something in view: leave it to the code
        if side in ("left", "right") and rq["clear_m"] > 0 and sec is not None:
            names = ("far_left", "left") if side == "left" else ("far_right", "right")
            if max(sec[n] for n in names) < rq["clear_m"]:
                return None                               # that side is walled off: leave it to the code
        if side == "left":
            return y0 + np.deg2rad(rq["turn_deg"]), side
        if side == "right":
            return y0 - np.deg2rad(rq["turn_deg"]), side
        if side == "behind":
            lb = (ans.get("context") or {}).get("last_seen_bearing_deg")
            s = -1.0 if (lb is not None and lb < 0) else 1.0
            return y0 + s * (np.pi - 0.05), side          # just short of pi, so the turn's direction is s
        if rq["ahead"] == "none":
            return None
        return (y0 if rq["ahead"] == "answer" else yaw), side    # ahead: hold a heading

    def _open_side(self, sec, left):
        """Bearing of the more open sector on the requested side."""
        names = ("far_left", "left") if left else ("far_right", "right")
        best = max(names, key=lambda n: sec[n])
        return self.eye.sector_bearing(best)

    def __call__(self, scene, judg, yaw, z, use_jev, fresh=False, t=0.0, pos=None, fix=None, vel=None,
                 reappear=None, alt_sp=None, avoid_ans=None, cmd=None):
        """`fix`: the target as another perception sees it (laya_pursuit.Locator), in the shape of
        scene["target"]; it replaces the camera's for pursuit only. Its range_m is the code's, and
        None when only the model sees the rover: then hold the not-visible speed -- except with a
        `speed_law`, where range_m is the model's and the law (given fix["t_est"], the estimate's frame
        time, and `vel`, our world velocity) sets forward speed instead of the line below."""
        if self.yaw_sp is None:
            self.yaw_sp = yaw
        sec = scene["sector_range_m"]
        tgt = scene["target"] if fix is None else fix

        if tgt["visible"]:
            self.last_bearing = np.deg2rad(tgt["bearing_deg"])
            rng = tgt["range_m"] if tgt["range_m"] is not None else STANDOFF + 1.5
            self.lost_for = 0.0
            self.search_yaw = None
            if fresh and pos is not None and tgt["range_m"] is not None:
                b = self.last_bearing
                c, s = np.cos(yaw), np.sin(yaw)
                off = np.array([rng * np.cos(b), rng * np.sin(b)])
                w = pos[:2] + np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
                self.tgt_w, self.tgt_t = w, t
                self.n_fix += 1
                # Differentiate over a ~1s baseline, not over one camera frame:
                # a 0.07s interval turns pixel noise into tens of m/s.
                self.tgt_hist.append((t, w))
                self.tgt_hist = [(ti, wi) for ti, wi in self.tgt_hist if t - ti <= 1.2]
                if self.tune.get("aim") == "trail":
                    self.trail.append((t, np.array(w, dtype=float)))
                    self.trail = [(ti, wi) for ti, wi in self.trail if t - ti <= 6.0]
                if len(self.tgt_hist) >= 2:
                    (t0, w0), (t1, w1) = self.tgt_hist[0], self.tgt_hist[-1]
                    if t1 - t0 >= 0.4:
                        v = (w1 - w0) / (t1 - t0)
                        sp = np.linalg.norm(v)
                        if sp > self.target_max_speed:     # cannot be faster than the rover
                            v = v / sp * self.target_max_speed
                        self.tgt_v = 0.6 * self.tgt_v + 0.4 * v
        else:
            rng = STANDOFF + 1.5
            self.lost_for = tgt["unseen_for_s"] or 0.0
            if self.search_yaw is None:
                self.search_yaw = self.yaw_sp

        # Every heading correction below is expressed RELATIVE to the live yaw and
        # re-derived each camera frame, so nothing can accumulate into a spin.
        yaw_rel = float(np.clip(self.last_bearing, -0.6, 0.6)) if tgt["visible"] else 0.0
        if self.tune:
            yaw_rel = self._tuned_yaw(tgt, yaw, t, pos, fresh)
        absolute_yaw = None
        fwd = float(np.clip(1.15 * (rng - STANDOFF) + 1.35 * self.k, 0.0, self.fwd_cap))
        if self.speed_law is not None:
            fwd = self.speed_law(t, bool(tgt["visible"]), tgt["range_m"], tgt.get("t_est"), tgt["bearing_deg"],
                                 yaw, vel)
        if self.tune.get("aim") == "trail" and tgt["visible"] and pos is not None and self.trail:
            # follow the rover's own trail instead of flying straight at it (which cuts corners into houses):
            # drop the crumbs we have reached, then steer at the oldest one still TRAIL_LOOKAHEAD_M away
            self.trail = [(ti, wi) for ti, wi in self.trail if np.linalg.norm(wi - pos[:2]) > 1.5]
            for ti, wi in self.trail:
                dv = wi - pos[:2]
                if np.linalg.norm(dv) >= TRAIL_LOOKAHEAD_M:
                    b = float((np.arctan2(dv[1], dv[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
                    yaw_rel = float(np.clip(b, -0.6, 0.6))
                    break
        if self.k > 1 and tgt["visible"]:
            # slow for sharp turns at speed: full speed within 20 deg of the nose, half at 70 deg and beyond
            fwd *= float(np.clip(1.0 - (abs(self.last_bearing) - 0.35) / 1.75, 0.5, 1.0))

        # altitude: the Altimeter's setpoint when one flies it (altitude.py; `climb` is then ignored), else
        # cruise, or CLIMB_ALT for the climb hold
        alt_ext = alt_sp
        self.climb_hold = max(0, self.climb_hold - 1)
        alt_sp = alt_ext if alt_ext is not None else (CLIMB_ALT if self.climb_hold else CRUISE_ALT)

        # --- reactive layer: turn away from the worst threat and slow down --------
        left_room = min(sec["far_left"], sec["left"])
        right_room = min(sec["far_right"], sec["right"])
        wr = min(sec.values())
        turn_bias, slide = 0.0, 0.0
        if wr < 4.0 and not self.avoid_mode:
            urgency = (4.0 - wr) / 4.0
            side = 1.0 if left_room > right_room else -1.0
            room = max(left_room, right_room)
            # Strafe, keeping the nose on the target -- but only as fast as the side we
            # are strafing into is actually observed to be clear.
            slide = side * 2.6 * urgency * min(1.0, room / 4.0)
            if self.tune.get("slide"):
                slide = self._tuned_slide(left_room, right_room, urgency)
            fwd *= 1.0 - 0.7 * urgency

        # --- reappear-guided reacquisition (off unless self.reacq): where the checkpoint says a lost rover
        # will come back into view replaces holding the heading and the extrapolated search, below
        reacq_h = self._reappear_heading(reappear, tgt, yaw, t, sec) if self.reacq is not None else None

        # --- Jev's tactical commitment (advisory) --------------------------------
        acted = False
        hold_search = False
        tactical = (use_jev and judg["source"] in ("jev", "laya") and judg["age_s"] < THRESH["stale_after_s"]
                    and (decision_needed(scene) or self.climb_hold))
        if tactical:
            self.commit_left = max(0, self.commit_left - 1)
            if self.commit_left == 0 or judg["risk"] >= THRESH["override_risk"]:
                if judg["maneuver"] != self.commit:
                    self.commit, self.commit_left = judg["maneuver"], THRESH["commit_steps"]
            mv = self.commit
            if not decision_needed(scene):
                mv = "hold_course"                 # the way is clear; stop maneuvering
                self.commit, self.commit_left = None, 0
            if judg["target_truly_lost"] >= THRESH["really_lost"] and mv != "climb":
                mv = "reacquire"
            acted = True

            if mv in ("gap_left", "gap_right"):
                left = mv == "gap_left"
                room = left_room if left else right_room
                slide = (1.0 if left else -1.0) * 2.6 * min(1.0, room / 4.0)
                if not tgt["visible"]:             # nothing to point at, so face the gap
                    yaw_rel = self._open_side(sec, left)
                fwd = max(fwd, 0.7)
            # only commit to going over it if there is demonstrably clear air up there
            elif (mv == "climb" and alt_ext is None and scene["sectors_blocked"] >= 4
                  and scene["free_ahead_above_m"] > 2.2 * scene["free_ahead_level_m"]):
                self.climb_hold = THRESH["climb_steps"]
                alt_sp = CLIMB_ALT
                fwd, slide, turn_bias = min(fwd, 0.5), 0.0, 0.0
            elif mv == "brake":
                fwd, slide = fwd * 0.15, slide * 0.3
            elif mv == "reacquire" and reacq_h is None:
                if fresh:
                    self.sweep += 0.35
                    self.yaw_sp = self._search_heading(yaw, t, pos) + 0.45 * np.sin(self.sweep)
                absolute_yaw = self.yaw_sp
                # must out-run the rover, or a lost target can never be regained
                fwd, slide, turn_bias = self.search_speed, 0.0, 0.0
                self.n_search_steps += 1
            else:
                acted = False                      # hold_course changes nothing
                hold_search = self.search_on_hold and mv == "hold_course" and self.lost_for > 1.2
            if judg["risk"] > THRESH["risk_slow_down"]:
                fwd *= 0.45
        if reacq_h is None and (hold_search or (not tactical and self.lost_for > 1.2)):
            # baseline search: never keep flying a bearing we can no longer see
            if fresh:
                self.sweep += 0.3
                self.yaw_sp = self._search_heading(yaw, t, pos) + 0.35 * np.sin(self.sweep)
            absolute_yaw = self.yaw_sp
            fwd, slide = self.search_speed, 0.0
            self.n_search_steps += 1

        if reacq_h is not None and (not tactical or mv in ("hold_course", "reacquire")):
            # turn toward the predicted side, at most max_step_rad past the live yaw per camera frame (re-derived
            # each frame, like the pursuit turn), at a moderate speed that drops while the nose is far off
            # the heading (a turn-around at search speed flies away from the rover); the reactive slide stays
            err = float((reacq_h[0] - yaw + np.pi) % (2 * np.pi) - np.pi)
            absolute_yaw = yaw + float(np.clip(err, -self.reacq["max_step_rad"], self.reacq["max_step_rad"]))
            fwd = min(fwd, self.reacq["cap"]) * float(np.clip(np.cos(err), self.reacq["min_speed_frac"], 1.0))
            acted = acted or (tactical and mv == "reacquire")
            self.n_reacq_steps += 1
            self.reacq_side_steps[reacq_h[1]] = self.reacq_side_steps.get(reacq_h[1], 0) + 1

        # --- braking limit for a faster rover (speed_scale > 1): never faster than we can stop from in the
        # clear path ahead (BRAKE_ACC, BRAKE_MARGIN_M); off at normal speed, where the reflex below suffices
        if self.k > 1 and not self.avoid_mode and not self.cmd_mode:
            fwd = min(fwd, max(0.25, float(np.sqrt(2 * BRAKE_ACC * max(0.0, scene["path_ahead_m"] - BRAKE_MARGIN_M)))))
        if self.cmd_mode and cmd is not None and t - cmd["t"] < 0.6:
            # the student flies its own command: forward speed, side slide and a heading set relative to the nose
            # it had when the frame was taken
            fwd, slide = float(cmd["speed"]), float(cmd["slide"])
            absolute_yaw = float(cmd["yaw"]) + np.deg2rad(float(cmd["turn"]))
            self.n_cmd_steps += 1
        elif self.cmd_mode:                  # no fresh command: hover in place (never the code's pursuit)
            fwd, slide, absolute_yaw = 0.0, 0.0, yaw
        if self.avoid_mode and avoid_ans is not None and t - avoid_ans[1] < 0.6:
            # Laya's graded collision avoidance (avoid.py): a safe forward speed and a sideways slide, from its
            # frame + the lidar + its speed and direction of travel
            cap, sl = avoid_ans[0]
            fwd = min(fwd, float(cap))
            slide = float(sl)
            w = "dodge" if abs(sl) >= 1.0 else ("brake" if cap < 1.0 else "clear")
            self.avoid_steps[w] = self.avoid_steps.get(w, 0) + 1

        # --- hard reflex: code overrides everything, Jev included ------------------
        # Reflex on what is in the path, not on what is merely alongside.
        near, nb = scene["path_ahead_m"], np.deg2rad(scene["nearest_bearing_deg"])
        reflex = near < (getattr(self, "emergency_m", EMERGENCY_REFLEX_M) if (self.avoid_mode or self.cmd_mode)
                         else REFLEX_M)
        if reflex and (self.avoid_mode or self.cmd_mode):
            self.n_emergency += 1
        if reflex:
            side = 1.0 if left_room > right_room else -1.0
            slide = side * 2.6 * min(1.0, max(left_room, right_room) / 3.0)
            fwd = min(fwd, 0.25) if near > 1.3 else -0.8

        if fresh:
            self.yaw_sp = absolute_yaw if absolute_yaw is not None else yaw + yaw_rel + turn_bias
        yaw_cmd = self.yaw_sp
        rate = float(self.tune.get("yaw_rate_dps", 0.0))
        if rate > 0 and yaw_cmd is not None:
            # yaw-rate cap: ramp the command toward the setpoint at most `rate` deg/s (a step of up to 0.6 rad per
            # camera frame is a ~9 rad/s demand: the yaw torque saturates the motors and flight.Pilot's airmode
            # turns that into lift). Kept within 0.6 rad of the live yaw so it never winds up.
            wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
            prev = yaw if self._yaw_cmd is None else yaw + float(np.clip(wrap(self._yaw_cmd - yaw), -0.6, 0.6))
            step = np.deg2rad(rate) * GUIDE_DT
            yaw_cmd = prev + float(np.clip(wrap(yaw_cmd - prev), -step, step))
            self._yaw_cmd = yaw_cmd

        # never slide sideways faster than the forward view can clear
        lat = float(np.clip(slide, -2.6, 2.6))

        vz = 1.6 * (max(alt_sp, MIN_ALT) - z)
        if z < MIN_ALT:
            vz = max(vz, 0.8)

        v_body = np.array([fwd, lat, np.clip(vz, -2.0, 2.0)])
        c, s = np.cos(yaw), np.sin(yaw)
        v_world = np.array([c * v_body[0] - s * v_body[1], s * v_body[0] + c * v_body[1], v_body[2]])
        return v_world, yaw_cmd, acted, reflex


def episode(seed=0, seconds=35.0, use_jev=True, video=None, hz=None, budget=None, realtime=True,
            backend="jev", laya_model=None, laya_image=False, lockstep=False, course="classic", record=None,
            pursuit="code", pursuit_model=None, pursuit_threshold=0.5, pursuit_noise_deg=0.0,
            pursuit_delay_s=0.0, pursuit_lockstep=False, search_lead_s=5.0, search_on_hold=False,
            pursuit_questions="v1", pursuit_sharpen=None, pursuit_gain=None, pursuit_range=None,
            speed_law="auto", speed_params=None, guide_tune=None, reacquire=None, reacquire_model=None,
            reacquire_hz=3.0, reacquire_params=None, reacquire_wrong_p=0.0, reacquire_delay_s=0.0,
            tactics_kw=None, appearance=None, altitude=None, altitude_model=None, altitude_hz=3.0,
            altitude_wrong_p=0.0, altitude_wrong=1.0, altitude_sharpen=1.0, record_rgb=False, yaw_desat=False,
            timing="wall", record_every=100, speed_scale=None, avoid=None, avoid_hz=6.0, emergency_m=None,
            policy=None, cmd_model=None, cmd_hz=10.0):
    """`record`: a list to append a snapshot to every 0.2 s of sim time (pose, obstacles,
    judgment, and the camera frame the model saw), for rendering after the flight
    (flightgif.py). Cheap, so the flight stays real time.

    `pursuit`: where the pursuit HEADING comes from. "code" (the segmentation bearing),
    "laya-strips" / "laya-frame" (Laya on the RGB frame, laya_pursuit.py), or "sim" (the true
    bearing + `pursuit_noise_deg` noise, `pursuit_delay_s` late: a model-free stand-in). Range, and
    so forward speed, stays the code's in those modes. "laya-pursuit" is laya-frame with Laya's
    range too (its range answer, same predict), so Laya sets heading AND forward speed; "sim-pursuit"
    is its model-free stand-in, the sim locator supplying both (range corrupted by `pursuit_range`,
    a dict of laya_pursuit.SimBackend's range_* arguments).

    `pursuit_questions` ("v1" | "v2"), `pursuit_sharpen`, `pursuit_gain`: laya_pursuit.FrameBackend's
    question set and steer read-out (None: the set's defaults).

    `speed_law`: how a MODEL's range becomes forward speed. "auto" (default): "robust" for
    laya-pursuit and sim-pursuit, where the range is the model's; code pursuit and the heading-only
    modes always use the code's law. "robust": laya_pursuit.RangeSpeed(**`speed_params`); "code": the
    code's law on the model's range, as laya-pursuit flew before.

    `search_lead_s`: the longest the lost-target search carries the last world fix forward along
    the rover's observed velocity (Guidance._search_heading). `search_on_hold`: search for a
    rover lost > 1.2 s even while the tactical answer is hold_course (by default a fresh
    hold_course pre-empts the search, so with const:oracle tactics it never runs).

    `backend="laya-v3"`: the drone-rover-v3 checkpoint (`laya_model`) answers the tactical maneuver from the
    frame + v3 context (tactics.LayaV3Backend; risk / target_truly_lost fixed at the oracle control's values).
    `tactics_kw`: extra LayaV3Backend arguments, e.g. {"climb_p": 0.12} (climb when P(climb) >= it; None = argmax).

    `reacquire`: None (default: off, nothing below changes), "laya" (the v3 checkpoint's reappear answers;
    `reacquire_model`, default pursuit_model or laya_model) or "sim" (the v3 label from the simulator, wrong
    with probability `reacquire_wrong_p`, `reacquire_delay_s` late). Once the pursuit source (the locator, else
    the code's segmentation) has not seen the rover for reacquire_params["after_s"] (1 s), the frame + v3
    context goes to laya_pursuit.Reacquirer at most `reacquire_hz` times a second, and Guidance turns toward
    the predicted side instead of holding course or the extrapolated search (run.REACQ_DEFAULTS,
    Guidance._reappear_heading).

    `altitude`: None (default: cruise, and CLIMB_ALT on a climb), "sim" (the course's altitude_target, with
    `altitude_wrong_p` answers of `altitude_wrong` m instead) or "laya" (`altitude_model`, default
    pursuit_model or laya_model, read with `altitude_sharpen`): altitude.Altimeter asks how far to move up or
    down at most `altitude_hz` times a second and flies the setpoint; a `climb` maneuver is then ignored.

    `timing`: "wall" (default): Laya's workers run in threads beside a sim paced to the wall clock, so when the
    box cannot keep real time the world slows and Laya gets extra time. "virtual": latency-faithful, every
    Laya call (pursuit locator, reacquirer, altimeter) is timed on one laya_pursuit.GpuClock: the sim stands
    still while the model computes, and each answer lands at sim time start + its measured GPU latency,
    queued behind the other questions' calls. The sim then runs as fast as it can (realtime off); what Laya
    answers, how often and how late is what this GPU gives in real time. (laya-v3 tactics are not timed.)

    "wallclock": true real time. Every checkpoint runs in one separate server process (laya_server), which
    owns the GPU and answers the questions one at a time; the sim runs paced to the wall clock and nothing
    waits for Laya. Valid only while the sim holds 1x: the result reports max_behind_s and behind_pct
    (share of steps more than 50 ms behind the wall clock); the sim alone runs ~4.5x real time.

    `avoid`: None (default: the code avoids collisions from the depth sensor), "laya" (the altitude checkpoint
    also answers avoid.question() from its frame + the depth sensor summary + forward speed, in the same
    predict, at `avoid_hz`, and Guidance flies that answer; the code keeps only an emergency reflex under
    EMERGENCY_REFLEX_M) or "sim" (the same with the answer from the true geometry, avoid.label). Needs altitude.

    `policy`: None (default: pursuit + reactive layers) or "laya-cmd": `cmd_model` (default pursuit_model or
    laya_model) answers command.question() -- forward speed, side slide, turn -- from each frame + the lidar
    context at up to `cmd_hz`, and Guidance flies it (command.py; trained on the lookahead teacher, teacher.py).

    `speed_scale`: scale the pursuit's speeds for a rover k times faster (default: the course's own factor,
    e.g. 4 on any "<course>-x4"; Guidance speed_scale, and RangeSpeed cruise / caps / rover speed).

    `appearance`: realism.py's real-world look (textures, sky, scanned clutter) for a courses.py course, as a
    spec ("real", "tex+sky+c20", ...) or dict; None (default) leaves the scene as it is. A course-name suffix
    does the same without this argument ("mixed@real"); when both are given this one replaces the suffix."""
    if timing not in ("wall", "virtual", "wallclock"):
        raise ValueError("timing must be wall, virtual or wallclock, got %r" % (timing,))
    if timing == "wallclock":
        import laya_server
        laya_server.start()          # before any backend is built: tactics.shared_laya routes to it
        laya_server.reclaim()        # a reused container: the last flight's workers held pipes
        laya_server.reset_stats()
        realtime = True
    gpu = None
    if timing == "virtual":
        import laya_pursuit
        gpu = laya_pursuit.GpuClock()
        realtime = False             # nothing waits on the wall clock any more
    rng = np.random.default_rng(seed)
    lap_course = None                # a looped course (town.py): laps, not an end line
    if course == "classic":
        if appearance is not None:
            raise ValueError("appearance needs a courses.py course, not classic (world.xml)")
        m = mujoco.MjModel.from_xml_path("world.xml")
        drive, rover_at, barrier_x, end_x, oracle = drive_course, rover_pose, 19.0, 77.0, None
    else:
        import courses
        c = courses.make(course, seed)
        if appearance is not None:           # realism.py (a "@" suffix on the course name needs none of this)
            import realism
            realism.set_appearance(c, appearance, seed)
            c.name = "%s@%s" % (c.name.split("@")[0], realism.tag(appearance))    # its own .course_*.xml
        m = mujoco.MjModel.from_xml_path(c.write(os.path.dirname(os.path.abspath(__file__))))
        drive, rover_at, barrier_x, end_x, oracle = c.drive, c.rover_pose, c.first_barrier_x, c.end_x, c.oracle
        lap_course = c if getattr(c, "looped", False) else None
        look = getattr(c, "appearance", None)       # (c is reused below)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    x2 = m.body("x2").id
    x2_geoms = set(np.nonzero(m.geom_bodyid == x2)[0].tolist())

    if lap_course is None:
        d.qpos[:3] = [1.5 + rng.uniform(-.3, .3), rng.uniform(-.5, .5), CRUISE_ALT]
        d.qpos[3:7] = [1, 0, 0, 0]
    else:                            # behind the rover on its loop, nose on it
        d.qpos[:7] = lap_course.start_pose(rng, CRUISE_ALT)
    lap = None if lap_course is None else lap_course.lap_tracker(d.qpos[:2].copy())
    mujoco.mj_forward(m, d)

    pilot = flight.Pilot(m)
    pilot.yaw_desat = bool(yaw_desat)     # False: the old airmode mixing (yaw demand can add lift)
    if reacquire not in (None, "sim", "laya"):
        raise ValueError("reacquire must be None, sim or laya, got %r" % (reacquire,))
    eye = flight.Eye(m, rgb_size=(512, 384) if ((use_jev and (laya_image or backend == "laya-v3"))
                                                or pursuit.startswith("laya") or reacquire == "laya"
                                                or altitude == "laya" or record_rgb or policy) else None)
    model_range = pursuit in ("laya-pursuit", "sim-pursuit")      # the pursuit's range is a model's
    if speed_law not in ("auto", "robust", "code"):
        raise ValueError("speed_law must be auto, robust or code, got %r" % speed_law)
    if speed_scale is None:
        speed_scale = 1.0
        if course != "classic":
            import courses
            speed_scale = courses.split_speed(course)[1]
    k = float(speed_scale)
    law = None
    if model_range and speed_law in ("auto", "robust"):
        import laya_pursuit
        sp = dict(speed_params or {})
        if k > 1:
            sp = dict(dict(cruise=1.35 * k, fwd_max=min(3.6 * k, FWD_CAP_FAST), lost_cap=min(2.4 * k, FWD_CAP_FAST - 0.5),
                           rover_speed=1.15 * k, target_max_speed=1.6 * k), **sp)
        law = laya_pursuit.RangeSpeed(**sp)
    rq = None if reacquire is None else dict(reacquire_params or {})
    if rq is not None and k > 1:
        rq.setdefault("cap", min(REACQ_DEFAULTS["cap"] * k, FWD_CAP_FAST - 0.5))
    if avoid not in (None, "laya", "sim"):
        raise ValueError("avoid must be None, laya or sim, got %r" % (avoid,))
    if avoid and not altitude:
        raise ValueError("avoid rides on the altitude stream: set altitude too")
    guide = Guidance(eye, search_lead_s, search_on_hold, speed_law=law, tune=guide_tune, reacq=rq, speed_scale=k,
                     avoid_mode=avoid is not None, cmd_mode=policy == "laya-cmd")
    cmds = cmd_now = None
    prev_cmd = (0.0, 0.0)
    if policy is not None:
        if policy != "laya-cmd":
            raise ValueError("policy must be None or laya-cmd, got %r" % (policy,))
        import command as cmdmod
        cmds = cmdmod.CommandStream(cmdmod.LayaCommand(cmd_model or pursuit_model or laya_model), hz=cmd_hz,
                                    gpu=gpu)
    if emergency_m is not None:
        guide.emergency_m = float(emergency_m)
    loc = None
    if pursuit != "code":
        import laya_pursuit
        loc = laya_pursuit.Locator(
            laya_pursuit.make_locator_backend(pursuit, pursuit_model, pursuit_threshold, pursuit_noise_deg,
                                              pursuit_delay_s, seed, questions=pursuit_questions,
                                              sharpen=pursuit_sharpen, gain=pursuit_gain, **(pursuit_range or {})),
            truth=lambda: laya_pursuit.true_fix(m, d, scene), lockstep=pursuit_lockstep,
            gpu=gpu if pursuit.startswith("laya") else None)
    tac = None
    if use_jev:
        from tactics import Tactician, DEFAULT, make_backend
        be = make_backend(backend, model=laya_model if backend in ("laya", "laya-v3") else None,  # "const:<maneuver>" is a control
                          **({"use_image": laya_image} if backend == "laya" else {}),
                          **(dict(tactics_kw or {}) if backend == "laya-v3" else {}),
                          **(dict({"seed": seed}, **(tactics_kw or {})) if backend.startswith("const:") else {}))
        tac = Tactician(backend=be, lockstep=lockstep,
                        **{k: v for k, v in (("hz", hz), ("budget", budget)) if v})
        if backend == "const:oracle":
            if oracle is None:
                raise ValueError("const:oracle needs a course from courses.py")
            be.bind(lambda: oracle(d.qpos[:3]))
        judg = dict(DEFAULT)
    else:
        judg = {"maneuver": "hold_course", "risk": 0.0, "confidence": 0.0,
                "target_truly_lost": 0.0, "source": "off", "age_s": 0.0, "probabilities": {}}
    # the v3 context (laya_pursuit.LastSeen) for the v3 tactics and for reacquisition, from the pursuit source
    wants_ctx = bool(tac) and getattr(tac.backend, "wants_context", False)
    seen = reacq = rp = None
    if wants_ctx or reacquire or altitude == "laya":
        import laya_pursuit
        seen = laya_pursuit.LastSeen()
    if reacquire:
        reacq = laya_pursuit.Reacquirer(
            laya_pursuit.make_reappear_backend(reacquire, reacquire_model or pursuit_model or laya_model,
                                               reacquire_wrong_p, reacquire_delay_s, seed),
            hz=reacquire_hz,
            truth=lambda: laya_pursuit.reappear_truth(m, d, rover_at, t, bool(scene["target"]["visible"])),
            gpu=gpu if reacquire == "laya" else None)
    reacq_after = guide.reacq["after_s"] if reacquire else None
    altim = alt_now = None
    if altitude is not None:
        import altitude as altmod
        target_fn = getattr(c, "altitude_target", None) if course != "classic" else None
        if target_fn is None:
            raise ValueError("altitude needs a courses.py / town.py course (its altitude_target)")
        altim = altmod.Altimeter(
            altmod.make_backend(altitude, target_fn=lambda: target_fn(d.qpos[:3]),
                                model=altitude_model or pursuit_model or laya_model, wrong_p=altitude_wrong_p,
                                wrong=altitude_wrong, sharpen=altitude_sharpen, seed=seed, avoid=avoid is not None),
            hz=avoid_hz if avoid else altitude_hz, truth=lambda: target_fn(d.qpos[:3]), start=CRUISE_ALT,
            gpu=gpu if altitude == "laya" else None)
        alt_track = []

    writer = cam = big = None
    if video:
        import imageio.v2 as imageio
        from hud import Hud
        big = mujoco.Renderer(m, 880, 1180)
        hud = Hud(1180, 880)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation = 2.9, -10.0
        writer = imageio.get_writer(video, fps=30, quality=8, macro_block_size=1)
        tape = []

    v_des, yaw_cmd = np.zeros(3), 0.0
    scene, fresh = None, False
    vis, frames, standoffs, hits, hit_steps, grounded = 0, 0, [], 0, set(), 0
    track = []                      # (t, x, y, yaw) at 10 Hz, for pathmetrics
    crashed_at, max_x, crossed, finished_at = None, -99.0, False, None
    jev_steps = reflex_steps = 0
    fix, fix_steps, guide_steps = None, 0, 0
    # reacquisition, on the visibility the pursuit steers on (the locator's, else the code's),
    # sampled every guidance step; stretches before the first sighting are not losses
    seen_once, unseen_since, gaps = False, None, []
    lost_steps = lost_search = lost_reflex = n_search_prev = 0
    n = int(seconds / dt)

    if timing == "wallclock":
        laya_server.reset_stats()    # the flight's calls only, not the checkpoint loads and warm-ups
    wall0 = time.time()
    max_behind, n_behind = 0.0, 0
    for i in range(n):
        t = i * dt
        if realtime:
            # Sim time must track wall-clock, or an API in the loop is being judged
            # against a world running 10x too fast to be a fair test.
            lag = t - (time.time() - wall0)
            if lag > 0.0005:
                time.sleep(lag)
            elif lag < -0.05:        # the sim is behind the wall clock (the world slows for everyone)
                n_behind += 1
                max_behind = max(max_behind, -lag)
        drive(m, d, t)

        pos = d.qpos[:3].copy()
        quat = d.qpos[3:7]
        yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                               1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))

        if i % 33 == 0:                                  # ~15 Hz perception
            scene = eye.look(d, pos, yaw, t)
            fresh = True
            frames += 1
            vis += scene["target"]["visible"]
            if seen is not None and not loc:
                seen.update(yaw, scene["target"])
            if tac and decision_needed(scene):
                t_off = time.time()
                if wants_ctx:
                    tac.offer(scene, t, eye.last_rgb, context=seen.context(yaw, fix if loc else scene["target"]))
                else:
                    tac.offer(scene, t, eye.last_rgb)
                if lockstep and realtime:
                    wall0 += time.time() - t_off   # the world waited for the model
            if loc:
                t_off = time.time()
                loc.offer(eye.last_rgb, t, yaw, scene["target"]["bearing_deg"])
                if loc.lockstep and realtime:
                    wall0 += time.time() - t_off
            if reacq is not None:
                src = fix if loc else scene["target"]
                if src is not None and not src["visible"] and (src["unseen_for_s"] or 0.0) > reacq_after:
                    t_off = time.time()
                    reacq.offer(eye.last_rgb, t, yaw, seen.context(yaw, src))
                    if reacq.lockstep and realtime:
                        wall0 += time.time() - t_off
            if cmds is not None:
                import avoid as avmod
                sp, rel = avmod.travel(d.qvel[:3], yaw)
                t_off = time.time()
                cmds.offer(eye.last_rgb, t, yaw, {"altitude_m": round(float(pos[2]), 2), "speed_mps": round(sp, 2),
                                                  "travel_deg": round(rel, 0), "prev_speed": round(prev_cmd[0], 1),
                                                  "prev_turn": round(prev_cmd[1], 0), "lidar": avmod.sensor(scene)})
                if cmds.lockstep and realtime:
                    wall0 += time.time() - t_off
            if altim is not None:
                actx = {"altitude_m": round(float(pos[2]), 2)}
                if seen is not None:
                    actx.update(seen.context(yaw, fix if loc else scene["target"]))
                if avoid:
                    import avoid as avmod
                    sp, rel = avmod.travel(d.qvel[:3], yaw)
                    actx.update(speed_mps=round(sp, 2), travel_deg=round(rel, 0), lidar=avmod.sensor(scene))
                    actx[altmod.TRUTH_KEY] = avmod.label_cmd(eye.lidar, d.qvel[:3], yaw)
                t_off = time.time()
                altim.offer(eye.last_rgb, t, actx)
                if altim.lockstep and realtime:
                    wall0 += time.time() - t_off

        if i % 10 == 0 and scene:                        # 50 Hz guidance
            if tac:
                prev = judg.get("maneuver"), judg.get("source")
                judg = tac.read(t)
                if TRACE and (judg.get("maneuver"), judg.get("source")) != prev:
                    print("  t=%5.1f %-11s p=%.2f risk=%.2f lost=%.2f | pos=(%.1f,%.1f,%.1f) blk=%d/5 near=%.2f tall=%s vis=%s"
                          % (t, judg["maneuver"], (judg.get("probabilities") or {}).get(judg["maneuver"], 0),
                             judg["risk"], judg["target_truly_lost"], pos[0], pos[1], pos[2],
                             scene["sectors_blocked"], scene["nearest_obstacle_m"],
                             scene["free_ahead_above_m"], scene["target"]["visible"]),
                          flush=True)
            if loc:
                # the model's heading; the range (so forward speed) is the model's in laya-pursuit and
                # sim-pursuit, else the code's (None where only the model sees the rover)
                est = loc.read(t, yaw)
                fix = {"visible": est["visible"], "bearing_deg": est["bearing_deg"],
                       "range_m": est["range_m"] if model_range else scene["target"]["range_m"],
                       "unseen_for_s": est["unseen_for_s"]}
                if law is not None:
                    fix["t_est"] = est["t_est"]
                fix_steps += est["visible"]
                guide_steps += 1
            if seen is not None and loc:
                seen.update(yaw, fix)
            if reacq is not None:
                rp = reacq.read(t)
            if altim is not None:
                alt_now = altim.read(t)
                if i % 500 == 0:
                    alt_track.append((round(t, 1), round(float(pos[0]), 1), round(float(pos[2]), 2), round(alt_now, 2)))
            if cmds is not None:
                cmd_now = cmds.read(t)
                if cmd_now is not None:
                    prev_cmd = (cmd_now["speed"], cmd_now["turn"])
            v_des, yaw_cmd, acted, reflex = guide(scene, judg, yaw, pos[2], use_jev, fresh, t, pos, fix,
                                                  d.qvel[:3].copy() if law is not None else None,
                                                  reappear=rp, alt_sp=alt_now,
                                                  avoid_ans=altim.avoid if (avoid and altim is not None) else None,
                                                  cmd=cmd_now)
            if TRACE >= 2 and i % 250 == 0:
                sec = scene["sector_range_m"]
                print("    t=%5.1f pos=(%5.1f,%5.1f,%4.1f) yaw=%4.0f mv=%-11s commit=%-11s reflex=%d v=(%4.1f,%4.1f,%4.1f)"
                      " sec=[%s] path=%.1f above=%.1f level=%.1f tgt=%s"
                      % (t, *pos, np.rad2deg(yaw), judg.get("maneuver"), guide.commit, reflex, *v_des,
                         " ".join("%4.1f" % v for v in sec.values()), scene["path_ahead_m"],
                         scene["free_ahead_above_m"], scene["free_ahead_level_m"],
                         "%+.0f@%.1f" % (scene["target"]["bearing_deg"], scene["target"]["range_m"])
                         if scene["target"]["visible"] else "lost %.1fs" % (scene["target"]["unseen_for_s"] or 0)),
                      flush=True)
            fresh = False
            jev_steps += acted
            reflex_steps += reflex
            pvis = bool((fix if loc else scene["target"])["visible"])
            if pvis:
                if unseen_since is not None:
                    gaps.append((t - unseen_since, True))
                seen_once, unseen_since = True, None
            elif seen_once and unseen_since is None:
                unseen_since = t
            if not pvis and guide.lost_for > 1.2:        # lost long enough that a search is due
                lost_steps += 1
                lost_search += guide.n_search_steps > n_search_prev
                lost_reflex += reflex
            n_search_prev = guide.n_search_steps

        d.ctrl[:] = pilot(d, v_des, yaw_cmd, dt)
        mujoco.mj_step(m, d)

        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in x2_geoms) != (g2 in x2_geoms):
                obj = g2 if g1 in x2_geoms else g1
                if obj not in hit_steps:
                    hit_steps.add(obj); hits += 1

        standoffs.append(float(np.linalg.norm(pos - rover_at(t))))
        if i % 50 == 0:
            track.append((t, pos[0], pos[1], yaw))
            if lap is not None:
                lap.update(t, pos, standoffs[-1])
        max_x = max(max_x, float(pos[0]))
        if pos[0] > barrier_x + 0.8:      # past the first barrier (beam0 at x=19 on the classic course)
            crossed = True
        if finished_at is None and pos[0] >= end_x:
            finished_at = t
        if pos[2] < 0.35:
            grounded += 1
            if grounded > 750:          # 1.5 s on the deck: it is down and not coming back
                crashed_at = t
                break
        else:
            grounded = max(0, grounded - 2)

        if record is not None and i % record_every == 0 and scene:    # 5 snapshots per sim second by default
            rgb = eye.last_rgb
            record.append({"t": t, "qpos": d.qpos.copy(), "mocap_pos": d.mocap_pos.copy(),
                           "mocap_quat": d.mocap_quat.copy(), "yaw": yaw,
                           "judg": {k: judg.get(k) for k in ("maneuver", "confidence", "risk",
                                                             "target_truly_lost", "source", "age_s")},
                           "reflex": bool(scene["path_ahead_m"] < REFLEX_M), "climbing": bool(guide.climb_hold),
                           "target_visible": bool(scene["target"]["visible"]), "hits": hits,
                           "rgb": None if rgb is None else rgb[::2, ::2].copy(),
                           # the pursuit input as Guidance used it this step (`fix` from loc.read; None when
                           # pursuit is the code's), next to the truth and the code's segmentation bearing
                           "loc": None if fix is None else {
                               "visible": bool(fix["visible"]), "bearing_deg": fix["bearing_deg"],
                               "unseen_for_s": fix["unseen_for_s"], "age_s": est.get("age_s"),
                               "range_m": fix["range_m"] if model_range else None,
                               "p_visible": est.get("p_visible")},
                           "true_bearing_deg": (lambda r: float(np.rad2deg(np.arctan2(
                               -np.sin(yaw) * r[0] + np.cos(yaw) * r[1], np.cos(yaw) * r[0] + np.sin(yaw) * r[1]))))(
                               d.mocap_pos[m.body("rover").mocapid[0]] - pos
                               - flight.Eye.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0.0])),
                           "code_bearing_deg": scene["target"]["bearing_deg"],
                           **({"reappear": None if rp is None else {k: rp.get(k) for k in (
                               "side", "t", "true_side", "occluded", "eta_s")}} if reacq is not None else {}),
                           "code_range_m": scene["target"]["range_m"],
                           # the altitude operator (altitude.py): altitude, setpoint, the latest answer and
                           # the course's target (what the answer should have moved toward)
                           **({"alt": {"z": float(pos[2]), "sp": alt_now,
                                       "dz": altim.dzs[-1] if altim.dzs else None,
                                       "target": float(target_fn(d.qpos[:3]))}} if altim is not None else {}),
                           "guide": {"lost_for": float(getattr(guide, "lost_for", 0.0) or 0.0),
                                     "yaw_sp": None if guide.yaw_sp is None else float(guide.yaw_sp),
                                     # which lost-target branch Guidance could take this step, mirroring its
                                     # conditions: "tactical" (a live judgment owns it; only "reacquire" there
                                     # searches), "baseline" (the lost_for > 1.2 search), or None
                                     "branch": ("tactical" if (use_jev and judg.get("source") in ("jev", "laya")
                                                               and (judg.get("age_s") or 0) < THRESH["stale_after_s"]
                                                               and (decision_needed(scene) or guide.climb_hold))
                                                else "baseline" if guide.lost_for > 1.2 else None),
                                     "tgt_w": None if getattr(guide, "tgt_w", None) is None
                                     else [float(v) for v in guide.tgt_w]}})

        if writer and i % 17 == 0 and scene:              # 30 fps video
            cam.lookat[:] = pos
            cam.azimuth = np.rad2deg(yaw)        # sit behind the aircraft, looking where it looks
            m.vis.global_.fovy = CHASE_FOVY
            big.update_scene(d, cam)
            eye._aim(pos, yaw); eye.depth.update_scene(d, eye.cam)
            depth = np.clip(eye.depth.render(), 0, 25.0)
            tel = {"t": t, "standoff": standoffs[-1], "speed": float(np.linalg.norm(d.qvel[:3])),
                   "hits": hits, "model": tac.model if tac else "disabled",
                   "mode": "JEV ENGAGED" if use_jev else "ABLATION: NO JEV",
                   "calls": tac.calls if tac else 0, "skipped": tac.skipped if tac else 0,
                   "tokens": tac.tokens if tac else 0, "hz": (1.0 / tac.min_dt) if tac else 0,
                   "lat": f"{np.median(tac.latency):.2f}s" if (tac and tac.latency) else "--",
                   "tan_h": eye.tan_h, "climbing": bool(guide.climb_hold)}
            writer.append_data(hud.draw(big.render(), depth, scene, judg, tel))
            tape.append({"qpos": d.qpos.copy().tolist(), "mocap": d.mocap_pos.copy().tolist(),
                         "depth": depth.astype(np.float16), "scene": scene,
                         "judg": judg, "tel": {k: v for k, v in tel.items() if k != "tan_h"}})

    if writer:
        writer.close()
        np.save(video + ".tape.npy", np.array(tape, dtype=object), allow_pickle=True)
    out = {"seed": seed, "jev": use_jev, "collisions": hits,
           "target_visible_pct": round(100 * vis / max(frames, 1), 1),
           "mean_standoff_m": round(float(np.mean(standoffs)), 2),
           "max_standoff_m": round(float(np.max(standoffs)), 2),
           "final_gap_m": round(standoffs[-1], 2),
           "distance_flown_m": round(float(np.linalg.norm(d.qpos[:3] - np.array([1.5, 0, CRUISE_ALT]))), 1),
           "steps_jev_acted_pct": round(100 * jev_steps / (n / 10), 1),
           "steps_reflex_pct": round(100 * reflex_steps / (n / 10), 1),
           "max_x_m": round(max_x, 1), "crossed_barrier": crossed, "course": course, "end_x_m": end_x,
           "finished_at_s": None if finished_at is None else round(finished_at, 1), "crashed_at_s": crashed_at, "flew_s": round(len(standoffs) * dt, 1),
           # sim seconds per wall second; below 1 means the box could not keep up, which hands the
           # decision model extra time (lockstep pauses are excluded, since wall0 absorbs them)
           "realtime_factor": round(len(standoffs) * dt / max(time.time() - wall0, 1e-9), 2)}
    if tac:
        out["jev"] = tac.stats()
        tac.close()
    out["pursuit"] = pursuit
    if course != "classic" and look:
        out["appearance"] = dict(look)
    # S-path (weaving) metrics, pathmetrics.py
    import pathmetrics
    tr = np.array(track)
    rv = np.array([rover_at(tt)[:2] for tt in np.arange(0.0, len(standoffs) * dt + 0.05, 0.05)])
    out["path"] = pathmetrics.summary(tr[:, 0], tr[:, 1:3], tr[:, 3], rv, x1=end_x)
    if guide_tune:
        out["guide_tune"] = dict(guide_tune)
    if model_range:
        out["speed_law"] = "robust" if law is not None else "code"
        if law is not None:
            out["speed_params"] = law.p
    # loss = the pursuit source's view of the rover gone for > 2 s after a first sighting; one still
    # open when the flight ends counts as a loss (and toward the longest), not toward reacquire time
    if unseen_since is not None:
        gaps.append((len(standoffs) * dt - unseen_since, False))
    losses = [g for g in gaps if g[0] > 2.0]
    back = [g for g, ok in losses if ok]
    ages = guide.search_ages
    out.update(search_lead_s=search_lead_s, search_on_hold=search_on_hold, loss_events=len(losses),
               longest_unseen_s=round(max([g for g, _ in gaps], default=0.0), 1),
               mean_reacquire_s=round(float(np.mean(back)), 1) if back else None,
               lost_at_end=bool(losses and not losses[-1][1]),
               world_fix_updates=guide.n_fix, search_steps_no_fix=guide.n_search_nofix,
               search_fix_age_p50_s=round(float(np.median(ages)), 1) if ages else None,
               search_fix_age_over_5s_pct=round(100 * float(np.mean(np.array(ages) > 5.0)), 1) if ages else None,
               # of the steps lost > 1.2 s: how many flew a search heading (where the lead can matter;
               # a fresh hold_course judgment pre-empts the search) and how many the reflex held
               lost_s=round(lost_steps * 10 * dt, 1),
               lost_searching_pct=round(100 * lost_search / lost_steps, 1) if lost_steps else None,
               lost_reflex_pct=round(100 * lost_reflex / lost_steps, 1) if lost_steps else None)
    if reacq is not None:
        rs = reacq.stats()
        reacq.close()
        out.update(reacquire=reacquire, reacquire_steps=guide.n_reacq_steps,
                   reacquire_side_steps=dict(guide.reacq_side_steps), reacquire_params=dict(guide.reacq),
                   reacquire_side_acc=rs["side_acc"], reacquirer=rs)
    if loc:
        st = loc.stats()
        loc.close()
        # used = a fresh, visible estimate set the pursuit heading on that guidance step
        out.update(pursuit_used_pct=round(100 * fix_steps / max(guide_steps, 1), 1),
                   pursuit_bearing_mae_deg=st["bearing_mae_deg"], pursuit_median_latency_s=st["median_latency_s"],
                   pursuit_p90_latency_s=st["p90_latency_s"], locator=st)
        if model_range:
            out["pursuit_range_mae_m"] = st["range_mae_m"]
    out["timing"] = timing
    if cmds is not None:
        out.update(policy=policy, commander=cmds.stats(), cmd_steps_pct=round(100 * guide.n_cmd_steps / max(n / 10, 1), 1),
                   emergency_reflex_steps=guide.n_emergency)
        cmds.close()
    if avoid:
        out.update(avoid=avoid, emergency_reflex_steps=guide.n_emergency,
                   emergency_reflex_pct=round(100 * guide.n_emergency / max(n / 10, 1), 2),
                   avoid_steps=dict(guide.avoid_steps))
    if realtime:
        out.update(max_behind_s=round(max_behind, 3), behind_pct=round(100 * n_behind / max(n, 1), 2))
    if timing == "wallclock":
        out["laya_server"] = laya_server.stats(len(standoffs) * dt)
        if not out["laya_server"]["calls"]:
            out["invalid"] = "no Laya call reached the server"
    if gpu is not None:
        out["gpu_clock"] = gpu.stats(len(standoffs) * dt)
    if altim is not None:
        out.update(altitude=altitude, altimeter=altim.stats(), altitude_track=alt_track)
        altim.close()
    if lap is not None:              # looped course: laps followed; finished = a lap, still tracking at the end
        lap.report(out, standoffs)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--seconds", type=float, default=35.0)
    p.add_argument("--no-jev", action="store_true")
    p.add_argument("--video", default=None)
    p.add_argument("--hz", type=float, default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--fast", action="store_true", help="run faster than real time (unfair to Jev)")
    p.add_argument("--backend", default="jev",
                   help="jev, laya, laya-v3 (the v3 checkpoint's maneuver from frame + context), or const:<maneuver>")
    p.add_argument("--laya-model", default=None, help="Hub id or local path (default: tactics.LAYA_MODEL)")
    p.add_argument("--laya-image", action="store_true", help="also give Laya the onboard camera frame")
    p.add_argument("--lockstep", action="store_true", help="pause the sim while the model decides (no latency)")
    p.add_argument("--out", default=None, help="append one JSON line per episode to this file")
    p.add_argument("--course", default="classic", help="classic (world.xml) or a layout in courses.py")
    p.add_argument("--appearance", default=None,
                   help="realism.py look for a courses.py course, e.g. real, tex+sky+c20, real+neg (or JSON); "
                        "same as a course suffix like mixed@real")
    p.add_argument("--pursuit", default="code",
                   choices=["code", "laya-strips", "laya-frame", "laya-pursuit", "sim", "sim-pursuit"],
                   help="source of the pursuit heading (range/speed stay the code's, except laya-pursuit and "
                        "sim-pursuit, where the locator's range sets forward speed)")
    p.add_argument("--pursuit-questions", default="v1", choices=["v1", "v2"],
                   help="laya-frame/laya-pursuit: v1 steer+speed, or v2 steer7+range8 (probe.questions_v2)")
    p.add_argument("--pursuit-sharpen", type=float, default=None, help="steer read-out power (default: per question set)")
    p.add_argument("--pursuit-gain", type=float, default=None, help="steer read-out gain (default: per question set)")
    p.add_argument("--range-noise", type=float, default=0.0, help="sim-pursuit: range noise std (m)")
    p.add_argument("--range-bias", type=float, default=1.0, help="sim-pursuit: range multiplier")
    p.add_argument("--range-offset", type=float, default=0.0, help="sim-pursuit: range offset (m), after the multiplier")
    p.add_argument("--range-tau", type=float, default=0.0, help="sim-pursuit: range noise correlation time (s; 0 white)")
    p.add_argument("--range-levels", default=None, help="sim-pursuit: quantize range to v1, v2, or comma centres (m)")
    p.add_argument("--range-soft", type=float, default=0.0,
                   help="sim-pursuit: quantization kernel width (m); 0 = nearest level")
    p.add_argument("--speed-law", default="auto", choices=["auto", "robust", "code"],
                   help="forward speed from a model's range: auto = robust for laya-pursuit/sim-pursuit")
    p.add_argument("--speed-param", action="append", default=[], metavar="K=V",
                   help="laya_pursuit.RangeSpeed parameter, repeatable (e.g. tau_s=0.8 gain=0.6 predict=0)")
    p.add_argument("--pursuit-model", default=None, help="Laya checkpoint for laya-* pursuit (default: tactics.LAYA_MODEL)")
    p.add_argument("--pursuit-threshold", type=float, default=0.5, help="P(visible) needed to steer on an estimate")
    p.add_argument("--pursuit-noise", type=float, default=0.0, help="sim pursuit: bearing noise std (deg)")
    p.add_argument("--pursuit-delay", type=float, default=0.0, help="sim pursuit: answer latency (sim s)")
    p.add_argument("--pursuit-lockstep", action="store_true", help="pause the sim while the locator looks")
    p.add_argument("--search-lead", type=float, default=5.0,
                   help="longest (s) the lost-target search extrapolates the last fix along its velocity")
    p.add_argument("--search-on-hold", action="store_true",
                   help="search for a lost rover even while the tactical answer is hold_course")
    p.add_argument("--reacquire", default=None, choices=["sim", "laya"],
                   help="turn toward where a lost rover will reappear: the v3 checkpoint's answer (laya, "
                        "--reacquire-model / --pursuit-model / --laya-model) or the simulator's label (sim)")
    p.add_argument("--reacquire-model", default=None, help="v3 checkpoint for --reacquire laya")
    p.add_argument("--reacquire-hz", type=float, default=3.0, help="most reappear questions per sim second")
    p.add_argument("--reacquire-wrong", type=float, default=0.0, help="sim reacquire: P(wrong side)")
    p.add_argument("--reacquire-delay", type=float, default=0.0, help="sim reacquire: answer latency (sim s)")
    p.add_argument("--reacquire-param", action="append", default=[], metavar="K=V",
                   help="run.REACQ_DEFAULTS override, repeatable (after_s, max_age_s, turn_deg, max_step_rad, cap, "
                        "min_speed_frac)")
    a = p.parse_args()
    rkw = {k: v for k, v, dflt in (("range_noise_m", a.range_noise, 0.0), ("range_bias", a.range_bias, 1.0),
                                   ("range_offset_m", a.range_offset, 0.0),
                                   ("range_tau_s", a.range_tau, 0.0), ("range_levels", a.range_levels, None),
                                   ("range_soft_m", a.range_soft, 0.0)) if v != dflt}
    sp = {k: (bool(float(v)) if k == "predict" else float(v)) for k, v in (kv.split("=", 1) for kv in a.speed_param)}
    for s in a.seeds:
        r = episode(s, a.seconds, not a.no_jev, a.video, a.hz, a.budget, realtime=not a.fast,
                    backend=a.backend, laya_model=a.laya_model, laya_image=a.laya_image, lockstep=a.lockstep, course=a.course,
                    pursuit=a.pursuit, pursuit_model=a.pursuit_model, pursuit_threshold=a.pursuit_threshold,
                    pursuit_noise_deg=a.pursuit_noise, pursuit_delay_s=a.pursuit_delay, pursuit_lockstep=a.pursuit_lockstep,
                    search_lead_s=a.search_lead, search_on_hold=a.search_on_hold,
                    pursuit_questions=a.pursuit_questions, pursuit_sharpen=a.pursuit_sharpen, pursuit_gain=a.pursuit_gain,
                    pursuit_range=rkw or None, speed_law=a.speed_law, speed_params=sp or None,
                    reacquire=a.reacquire, reacquire_model=a.reacquire_model, reacquire_hz=a.reacquire_hz,
                    reacquire_wrong_p=a.reacquire_wrong, reacquire_delay_s=a.reacquire_delay,
                    reacquire_params={k: (v if k == "ahead" else float(v))
                                      for k, v in (kv.split("=", 1) for kv in a.reacquire_param)} or None,
                    appearance=a.appearance)
        print(json.dumps(r))
        sys.stdout.flush()
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
