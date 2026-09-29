"""Laya-steered pursuit: the rover's bearing from the onboard colour frame, not the segmentation mask.

run.py's pursuit points the nose at scene["target"]["bearing_deg"], which comes from the simulator's
segmentation render -- a sensor no real aircraft has. Here a (fine-tuned) Laya Vision checkpoint looks
at the RGB frame and says where the rover is; Guidance steers on that instead. Only the HEADING moves
over: range, and so forward speed, still come from the code's scene, because Laya's speed answer is not
trained well enough yet to hold a standoff on (probe.py scores it).

Two ways to ask, both with the questions probe.py scores and rover_data.py trains on:

  strips   probe.strips() cuts the rover band into 5 vertical crops; one probe.STRIP_Q noul per crop.
           Visible when the best crop's P(yes) clears `threshold`; bearing from the P-weighted crop
           centre. 5 predicts per estimate (laya's predict scores one state at a time).
  frame    one predict on the whole frame: probe.questions() "visible" and "steer"; bearing from the
           steer score's expected level, through rover_data.STEER_CENTRES (+27 ... -27 deg).

`sim` is a test double that needs no model: the true bearing from the simulator, plus optional Gaussian
noise and a delay, so the whole path (worker, staleness, Guidance) runs on a CPU. It can corrupt the range
the same way (noise, a multiplicative bias, correlated noise, quantization to a score's levels), for
`sim-pursuit`, where it sets forward speed as Laya does in laya-pursuit.

`RangeSpeed` is the forward-speed law for a model-supplied range (laya-pursuit, sim-pursuit): a filtered,
own-motion-predicted range and a gentler, clipped speed command. Code pursuit keeps run.Guidance's law.

Runs in a worker thread like tactics.Tactician: offer() never blocks the 500 Hz loop (unless lockstep),
read() returns the latest estimate and its age. Bearing is + to the aircraft's left, as in flight.py.

v3 (bottom of the file): `v3_state` / `LastSeen` build the drone-rover-v3 state (frame + context JSON);
`Reacquirer` asks where a lost rover will reappear (`LayaReappear`: the checkpoint; `SimReappear`: the v3
label from the simulator, for CPU runs) for run.Guidance's reappear-guided reacquisition. Every Laya user in
the process shares one loaded checkpoint (tactics.shared_laya).
"""
import threading, queue, time
import numpy as np
import probe
from rover_data import STEER_CENTRES

# range (m) at each probe.SPEED level: the middle of each level's band (<2.5, 2.5-4, 4-7, >7). No
# fitting needed: 0.81 m mean error on the held-out no-climb frames, a forward-speed command
# error of 0.52 m/s against 1.00 for a constant guess (results/laya-steer/README.md)
SPEED_RANGE_M = [2.0, 3.25, 5.5, 9.0]
# the read-out per question set: (steer question, its level centres, range question, its centres, default
# steer sharpen, default steer gain, default range sharpen). Fitted on held-out frames: v1 steer (2, 1.16),
# FrameBackend; v2 (drone-rover-v2/last) steer sharpen 2, gain 1 (4.4 deg MAE, 3.8 within +-34) and range
# sharpen 2 (0.43 m MAE, bias -0.10 m). v1's range is read raw, as it always was.
QUESTIONS = {
    "v1": ("steer", STEER_CENTRES, "speed", SPEED_RANGE_M, 2.0, 1.16, 1.0),
    "v2": ("steer7", probe.STEER7_CENTRES, "range8", probe.RANGE8_CENTRES, 2.0, 1.0, 2.0),
}
FRESH_S = 0.5            # an estimate older than this (frame time -> now) no longer steers
N_STRIPS = 5


def true_fix(m, d, scene):
    """Ground truth for the sim locator and the error metric -- never steers a laya mode.
    Same geometry as probe.py / rover_data.py: camera on the nose, bearing = atan2(left, fwd)."""
    import flight
    q = d.qpos[3:7]
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    cam = d.qpos[:3] + flight.Eye.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0.0])
    r = d.mocap_pos[m.body("rover").mocapid[0]] - cam
    fwd, left = np.cos(yaw) * r[0] + np.sin(yaw) * r[1], -np.sin(yaw) * r[0] + np.cos(yaw) * r[1]
    return bool(scene["target"]["visible"]), float(np.rad2deg(np.arctan2(left, fwd))), float(np.hypot(fwd, left))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def level_centres(levels):
    """None, "v1" (SPEED_RANGE_M), "v2" (probe.RANGE8_CENTRES), or a list / comma string of centres (m)."""
    if levels is None or levels == "":
        return None
    if isinstance(levels, str):
        if levels in QUESTIONS:
            return list(QUESTIONS[levels][3])
        levels = levels.split(",")
    return [float(v) for v in levels]


def quantize_range(r, centres, soft_m=0.0):
    """What a score question over range levels reads back for a rover at `r` m. soft_m 0: the nearest
    level's centre (a hard, confident answer). soft_m > 0: the expected value over the levels with
    P(level) ~ exp(-(r - centre)^2 / 2 soft_m^2) -- a calibrated score's spread, which also compresses
    answers toward the middle at the ends of the scale, as the v1 speed read-out did."""
    c = np.asarray(centres, dtype=float)
    if soft_m <= 0:
        return float(c[int(np.argmin(np.abs(c - r)))])
    w = np.exp(-0.5 * ((r - c) / soft_m) ** 2)
    if w.sum() < 1e-12:                   # far outside the scale: all weight on the nearest end
        return float(c[int(np.argmin(np.abs(c - r)))])
    return float((w * c).sum() / w.sum())


class SimBackend:
    """Test double: the truth, plus noise. `delay_s` is applied in SIM time by the Locator, which also
    takes no new frame until the last answer is out, so the answer rate is 1/delay -- the shape of a
    real model (strips on an L4: 5 x ~40 ms ~ 0.2 s, ~5 Hz). Answered inline (`instant`), so a sim
    flight is deterministic: no thread-timing race on when the answer lands.

    Range (used only by sim-pursuit, where it sets forward speed): true range x `range_bias` +
    `range_offset_m`, plus Gaussian noise of std `range_noise_m` -- white, or AR(1) with correlation time `range_tau_s` (a model
    misjudging one scene misjudges the next frame too; steps are taken per answer, at the answer
    interval max(delay, one camera frame)) -- then, with `range_levels`, read back through
    quantize_range(). All off by default, and drawn from their own random stream, so the bearing
    noise (and every existing sim flight) is unchanged."""
    instant = True
    FRAME_S = 33 * 0.002                   # run.episode's camera interval

    def __init__(self, noise_deg=0.0, delay_s=0.0, seed=0, range_noise_m=0.0, range_bias=1.0,
                 range_levels=None, range_soft_m=0.0, range_tau_s=0.0, range_offset_m=0.0):
        self.noise_deg, self.delay_s = float(noise_deg), float(delay_s)
        self.rng = np.random.default_rng(seed)
        self.model = "sim(noise=%g,delay=%g)" % (noise_deg, delay_s)
        self.range_noise_m, self.range_bias = float(range_noise_m), float(range_bias)
        self.range_levels, self.range_soft_m = level_centres(range_levels), float(range_soft_m)
        self.range_tau_s, self.range_offset_m = float(range_tau_s), float(range_offset_m)
        self.corrupt_range = bool(self.range_noise_m or self.range_bias != 1.0 or self.range_offset_m
                                  or self.range_levels)
        if self.corrupt_range:
            self.rng_r = np.random.default_rng([seed, 7919])
            self._n = 0.0                                  # AR(1) state, in units of std
            dt = max(self.delay_s, self.FRAME_S)
            self._rho = float(np.exp(-dt / self.range_tau_s)) if self.range_tau_s > 0 else 0.0
            self.model += "+range(noise=%g,bias=%g,offset=%g,tau=%g%s)" % (
                self.range_noise_m, self.range_bias, self.range_offset_m, self.range_tau_s,
                ",levels=%s,soft=%g" % (self.range_levels, self.range_soft_m) if self.range_levels else "")

    def _range(self, r):
        if not self.corrupt_range:
            return r
        self._n = self._rho * self._n + np.sqrt(1 - self._rho ** 2) * float(self.rng_r.normal())
        r = max(0.3, r * self.range_bias + self.range_offset_m + self.range_noise_m * self._n)
        return quantize_range(r, self.range_levels, self.range_soft_m) if self.range_levels else r

    def locate(self, frame, truth):
        vis, b, rng = truth
        if not vis:
            return False, None, 0.0, None
        return True, b + self.noise_deg * float(self.rng.normal()), 1.0, self._range(rng)


class _Laya:
    def __init__(self, model=None, threshold=0.5, device=None, revision=None):
        from tactics import shared_laya
        # one loaded checkpoint per process: the v3 tactics and the reacquisition asker on the same
        # checkpoint reuse it (tactics.shared_laya; same load_vlm call and budgets as before)
        self.agent = shared_laya(model, device, revision)
        self.threshold = threshold
        self.delay_s = 0.0                # real latency is wall-clock, measured by the Locator
        self.warmup_s = None

    def warm_up(self):
        """Two throwaway predicts before the flight starts. The first call on a GPU pays for CUDA
        setup and kernel selection (seconds); made inside a real-time flight it left the aircraft
        with no heading for its opening seconds, flying 'target lost' straight past the rover."""
        import time
        blank = np.zeros((384, 512, 3), dtype=np.uint8)
        t0 = time.time()
        self.locate(blank)
        self.warmup_s = round(time.time() - t0, 2)
        self.locate(blank)

    @staticmethod
    def _img(frame):
        from PIL import Image
        return frame if isinstance(frame, Image.Image) else Image.fromarray(frame)


class StripsBackend(_Laya):
    """One probe.STRIP_Q noul per vertical crop of the rover band. `sharpen` > 1 raises the strip
    probabilities to that power before the weighted centre, so a clear winner is not pulled
    toward the middle by the others' leftover P(yes)."""

    def __init__(self, model=None, threshold=0.5, sharpen=1.0, **kw):
        super().__init__(model, threshold, **kw)
        self.sharpen = sharpen
        self.model = "laya-strips:" + (model or "default")
        self.centres = (np.arange(N_STRIPS) + 0.5) / N_STRIPS
        self.warm_up()

    def locate(self, frame, truth=None):
        q = {"rover": probe.STRIP_Q}
        ps = np.array([float(self.agent.predict({"image": c}, q)["answers"]["rover"]["noul"])
                       for c in probe.strips(self._img(frame), N_STRIPS)])
        pmax = float(ps.max())
        if pmax < self.threshold:
            return False, None, pmax
        w = ps ** self.sharpen
        x = float((w * self.centres).sum() / max(w.sum(), 1e-9))
        return True, probe.x_to_bearing(x), pmax


class FrameBackend(_Laya):
    """One predict on the whole frame: visible (noul) and steer (a score over levels).

    `questions`: "v1" asks probe.questions() "steer" (5 levels, rover_data.STEER_CENTRES, +-27 deg)
    and, with `speed`, "speed" (4 bands, read as SPEED_RANGE_M); "v2" asks probe.questions_v2()
    "steer7" (7 levels, probe.STEER7_CENTRES, +-60 deg) and "range8" (8 levels, probe.RANGE8_CENTRES,
    2-8.5 m). Bearing and range are each the score's expected level centre, after raising the level
    probabilities to `sharpen` (steer) / `range_sharpen` (range) and renormalising.

    The v1 steer score's expected level compresses toward the centre: the calibrated temperature
    flattens the levels, and the outer centres sit at +-27 deg. Read raw, a rover 12-25 deg off
    the nose came out ~8 deg too central, so pursuit under-turned and lost it. `sharpen` raises the
    level probabilities to that power and renormalises; `gain` rescales the result. The v1 defaults
    (2, 1.16) were fitted on held-out mixed frames within +-34 deg (pursuit's turn clip) and
    scored on the unseen no-climb layout: 4.5 deg mean error there, against 6.3 read raw
    (results/probe/README.md). v2's (sharpen 2, gain 1; range sharpen 2) were fitted the same way on the
    drone-rover-v2 checkpoint (QUESTIONS)."""

    def __init__(self, model=None, threshold=0.5, sharpen=None, gain=None, speed=False, questions="v1",
                 range_sharpen=None, **kw):
        """`speed`: also ask the range question in the same predict and return a range (m), so Laya sets
        forward speed too (pursuit="laya-pursuit"). `sharpen` / `gain` / `range_sharpen` None: the question
        set's default."""
        super().__init__(model, threshold, **kw)
        if questions not in QUESTIONS:
            raise ValueError("questions must be one of %s, got %r" % (sorted(QUESTIONS), questions))
        steer_q, self.steer_c, range_q, self.range_c, sh, g, rsh = QUESTIONS[questions]
        self.sharpen = sh if sharpen is None else float(sharpen)
        self.gain = g if gain is None else float(gain)
        self.range_sharpen = rsh if range_sharpen is None else float(range_sharpen)
        self.speed, self.questions = speed, questions
        self.model = ("laya-pursuit:" if speed else "laya-frame:") + (model or "default") + (
            "" if questions == "v1" else ":" + questions)
        qs = probe.questions() if questions == "v1" else probe.questions_v2()
        self.steer_q, self.range_q = steer_q, range_q
        self.qs = {"visible": qs["visible"], steer_q: qs[steer_q]}
        if speed:
            self.qs[range_q] = qs[range_q]
        self.warm_up()

    def locate(self, frame, truth=None):
        a = self.agent.predict({"image": self._img(frame)}, self.qs)["answers"]
        pv = float(a["visible"]["noul"])
        if pv < self.threshold:
            return False, None, pv, None
        p = np.array([float(a[self.steer_q]["probabilities"][str(i)]) for i in range(len(self.steer_c))]) ** self.sharpen
        rng = None
        if self.speed:
            ps = np.array([float(a[self.range_q]["probabilities"][str(i)]) for i in range(len(self.range_c))])
            ps = ps ** self.range_sharpen
            rng = float(np.dot(ps / max(ps.sum(), 1e-9), self.range_c))
        return True, self.gain * float((p / p.sum() * np.array(self.steer_c)).sum()), pv, rng


def make_locator_backend(mode, model=None, threshold=0.5, noise_deg=0.0, delay_s=0.0, seed=0, questions="v1",
                         sharpen=None, gain=None, **range_kw):
    """`range_kw`: SimBackend's range corruption (range_noise_m, range_bias, range_levels, range_soft_m,
    range_tau_s), for the sim modes. `questions` / `sharpen` / `gain`: FrameBackend's read-out."""
    if mode in ("sim", "sim-pursuit"):
        return SimBackend(noise_deg, delay_s, seed, **range_kw)
    if mode == "laya-strips":
        return StripsBackend(model, threshold)
    if mode == "laya-frame":
        return FrameBackend(model, threshold, sharpen, gain, questions=questions)
    if mode == "laya-pursuit":
        return FrameBackend(model, threshold, sharpen, gain, speed=True, questions=questions,
                            range_sharpen=range_kw.get("range_sharpen"))
    raise ValueError("pursuit locator must be sim, sim-pursuit, laya-strips, laya-frame or laya-pursuit, got %r" % mode)


class RangeSpeed:
    """Forward speed from a MODEL's range to the rover (laya-pursuit, sim-pursuit); code pursuit keeps
    run.Guidance's law, fwd = clip(1.15 (range - 3.5) + 1.35, 0, 3.6), which is fine on the segmentation
    range (~0.1 m error) and fails on a model's: every range error becomes 1.15x as much speed error,
    every estimate straight away, and a low reading brakes the aircraft to a stop while the rover drives
    on. Four changes, each a parameter:

    1. Filter the range, knowing when each estimate was taken (they arrive at ~10-15 Hz, irregularly, and
       stop while the rover is out of view): first-order, time constant `tau_s`, so a burst of estimates
       counts for its duration, not its number.
    2. Predict between estimates from our own motion (`predict`): range changes at (rover speed along the
       line of sight - our own velocity along it). Our velocity is known; the rover's along-LOS speed `u`
       is taken as `rover_speed` (the courses drive it at a constant 1.15 m/s). The filter then has no lag
       when WE change speed, which is most of the range change, so `tau_s` can be long enough to average
       noise. Each estimate is also carried forward over its own age before it is fused. `rate_tau_s` > 0
       also corrects `u` from the filter's innovations (time constant rate_tau_s, clamped to
       [0, `target_max_speed`]); off by default: in the sim it wandered 0.4-1.6 m/s through occlusions and
       re-acquisitions and did not fly better (76 vs 74 of 100 finished).
    3. A gentler gain (`gain` m/s per m, vs 1.15), with a `deadband_m` either side of the standoff, and
       `cruise` (m/s) at the standoff.
    4. Asymmetric limits: slower than `fwd_floor` only once the filtered range is inside `close_m` (a low
       reading alone cannot stop us and let the rover drive away: falling behind is what lost the v1
       flights), at most `fwd_max`; while the rover is out of view, coast on the predicted range for up
       to `coast_s` at no more than `lost_cap` (we may have overrun it), then the code's lost-target speed
       (1.15 x 1.5 + 1.35 = 3.07 m/s) capped at `lost_cap`. After `reset_s` without an estimate, the next
       one re-initialises the filter instead of being averaged into a stale prediction."""

    DEFAULTS = dict(standoff=3.5, tau_s=0.8, predict=True, rate_tau_s=0.0, gain=0.6, deadband_m=0.3,
                    cruise=1.35, fwd_floor=0.9, close_m=2.6, fwd_max=3.6, lost_cap=2.4, coast_s=1.5,
                    reset_s=3.0, rover_speed=1.15, target_max_speed=1.6)

    def __init__(self, **params):
        bad = set(params) - set(self.DEFAULTS)
        if bad:
            raise ValueError("unknown RangeSpeed parameters: %s" % sorted(bad))
        self.p = dict(self.DEFAULTS, **params)
        for k, v in self.p.items():
            setattr(self, k, v)
        self.r = None               # filtered range (m), predicted to the last call's time
        self.u = self.rover_speed   # rover speed along the line of sight (m/s)
        self.los = None             # world angle of the line of sight at the last estimate (rad)
        self.t = None               # time of the last call
        self.t_meas = None          # frame time of the last estimate fused
        self.n_meas = 0

    def __call__(self, t, visible, range_m, t_est, bearing_deg, yaw, vel):
        """-> forward speed (m/s). `range_m` / `t_est`: the latest estimate and its frame time (None when
        not visible); `bearing_deg`: its bearing at the live `yaw`; `vel`: our world velocity (>= 2 axes)."""
        v_own = 0.0
        if vel is not None and self.los is not None:
            v_own = float(vel[0] * np.cos(self.los) + vel[1] * np.sin(self.los))
        if self.r is not None and self.t is not None and self.predict:
            self.r = max(0.5, self.r + (self.u - v_own) * (t - self.t))
        self.t = t
        if visible and range_m is not None and bearing_deg is not None:
            self.los = float(yaw + np.deg2rad(bearing_deg))
            if t_est is None:
                t_est = t
            if self.r is None or self.t_meas is None or t_est - self.t_meas > self.reset_s:
                self.r, self.u = float(range_m), self.rover_speed
                self.t_meas = t_est
                self.n_meas += 1
            elif t_est > self.t_meas:
                z = float(range_m) + ((self.u - v_own) * (t - t_est) if self.predict else 0.0)
                dtm = t_est - self.t_meas
                e = z - self.r
                self.r += (1.0 - np.exp(-dtm / self.tau_s)) * e
                if self.predict and self.rate_tau_s > 0:
                    self.u = float(np.clip(self.u + dtm / self.rate_tau_s * e / self.tau_s,
                                           0.0, self.target_max_speed))
                self.t_meas = t_est
                self.n_meas += 1
        if self.r is None:                      # never seen: the code's not-visible speed
            return min(1.15 * 1.5 + 1.35, self.lost_cap)
        lost_for = t - self.t_meas
        if not visible and lost_for > self.coast_s:
            return min(1.15 * 1.5 + 1.35, self.lost_cap)
        e = self.r - self.standoff
        e = np.sign(e) * max(abs(e) - self.deadband_m, 0.0)
        fwd = self.cruise + self.gain * e
        lo = 0.0 if self.r < self.close_m else self.fwd_floor
        hi = self.fwd_max if visible else min(self.fwd_max, self.lost_cap)
        return float(np.clip(fwd, lo, hi))


class GpuClock:
    """Latency-faithful timing for every Laya call in a flight (run.episode timing="virtual"). The sim stands
    still while the model computes (the workers run lockstep, and that pause is not sim time), each call's
    wall-clock latency L is measured, and its answer is released at sim time start + L, where start is the
    later of the frame's time and the moment the GPU is free: one GPU, calls from every question queued on it
    in order. So the world never waits for Laya and Laya never gets extra time: what it can answer, and how
    late, is what this GPU would give in real time, however slowly the sim itself runs.

    run(t, fn) -> (result, release sim time)."""

    def __init__(self):
        self.free_at = float("-inf")
        self.busy_s = self.wait_s = 0.0
        self.calls = 0
        self._lock = threading.Lock()
        self.by_stream = {}

    def run(self, t, fn, stream="?"):
        t0 = time.time()
        res = fn()
        lat = time.time() - t0
        with self._lock:
            start = max(t, self.free_at)
            end = start + lat
            self.free_at = end
            self.busy_s += lat
            self.wait_s += start - t
            self.calls += 1
            st = self.by_stream.setdefault(stream, [0, 0.0, 0.0])
            st[0] += 1
            st[1] += lat
            st[2] += end - t
        return res, end

    def stats(self, flew_s):
        return {"calls": self.calls, "gpu_busy_pct": round(100 * self.busy_s / max(flew_s, 1e-9), 1),
                "mean_queue_wait_s": round(self.wait_s / max(self.calls, 1), 3),
                "by_stream": {k: {"calls": n, "per_s": round(n / max(flew_s, 1e-9), 2),
                                  "mean_compute_s": round(c / max(n, 1), 3), "mean_answer_age_s": round(a / max(n, 1), 3)}
                              for k, (n, c, a) in self.by_stream.items()}}


class Locator:
    """Where is the rover? Asked every camera frame, answered when the backend is free.

    A frame offered while the worker is busy replaces the one waiting (latest wins), so the answer
    is never older than one inference. `lockstep` makes offer() wait for the answer: the sim stands
    still while the model looks, which measures perception quality with latency taken out.

    `truth()` -> (visible, bearing_deg) is ground truth: the sim backend answers from it, and every
    estimate is scored against it (taken at the frame's time), so a flight reports its own
    perception error."""

    def __init__(self, backend, truth=None, lockstep=False, fresh_s=FRESH_S, gpu=None):
        self.backend = backend
        self.model = backend.model
        self.truth = truth
        self.gpu = gpu                      # GpuClock: latency-faithful virtual time (lockstep, released at start + L)
        self.lockstep = lockstep or getattr(backend, "instant", False) or gpu is not None
        self.fresh_s = fresh_s
        self.delay_s = getattr(backend, "delay_s", 0.0)
        self._busy_until = float("-inf")    # sim time; only a delayed (sim) backend uses it
        self._q = queue.Queue(maxsize=1)
        self._pending = []                  # (ready_at sim time, estimate), in frame order
        self._latest = None
        self._last_seen = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.n_offer = self.n_busy = self.n_dropped = self.errors = 0
        self.last_error = None
        self.latency = []
        self.err, self.ref_err, self.vis_ok, self.range_err = [], [], [], []
        self.false_visible = self.missed_visible = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def offer(self, frame, now, yaw, ref_bearing=None):
        """Non-blocking (unless lockstep). `ref_bearing`: the code's own (segmentation) bearing for
        this frame, scored against the same truth for comparison; never used to steer."""
        self.n_offer += 1
        truth = self.truth() if self.truth else None
        if truth and truth[0] and ref_bearing is not None:  # truth: (visible, bearing_deg, range_m)
            self.ref_err.append(abs(ref_bearing - truth[1]))
        if now < self._busy_until:
            self.n_busy += 1
            return
        if self.delay_s:
            self._busy_until = now + self.delay_s
        item = (frame, now, yaw, truth)
        self._done.clear()
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.n_dropped += 1
            except queue.Empty:
                pass
            self._q.put_nowait(item)
        if self.lockstep:
            self._done.wait()

    def read(self, now, yaw=None):
        """The latest estimate, re-expressed at the live yaw: it was measured off the nose as it
        pointed when the frame was taken, and the aircraft has turned since (the code knows its own
        yaw). Not visible when stale."""
        with self._lock:
            while self._pending and self._pending[0][0] <= now:
                e = self._pending.pop(0)[1]
                self._latest = e
                if e["visible"]:
                    self._last_seen = e["t"]
            e = self._latest
        if e is None:
            return {"visible": False, "bearing_deg": None, "range_m": None, "age_s": None, "fresh": False,
                    "unseen_for_s": None, "t_est": None}
        age = now - e["t"]
        fresh = age < self.fresh_s
        b = e["bearing_deg"]
        if b is not None and yaw is not None:
            b -= float(np.rad2deg(_wrap(yaw - e["yaw"])))
        vis = bool(fresh and e["visible"])
        unseen = 0.0 if vis else (None if self._last_seen is None else round(now - self._last_seen, 2))
        return {"visible": vis, "bearing_deg": b if vis else None, "range_m": e.get("range_m") if vis else None,
                "age_s": round(age, 3), "fresh": fresh, "unseen_for_s": unseen, "p_visible": e["p_visible"],
                "t_est": e["t"]}

    def _worker(self):
        while not self._stop.is_set():
            try:
                frame, t, yaw, truth = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                if self.gpu is not None:
                    res, ready = self.gpu.run(t, lambda: self.backend.locate(frame, truth), "steer")
                    self._busy_until = ready        # this stream's next frame waits for its answer
                else:
                    res, ready = self.backend.locate(frame, truth), t + self.delay_s
                vis, b, p = res[:3]
                rng = res[3] if len(res) > 3 else None     # backends without a range estimate return 3
                self.latency.append(time.time() - t0)
                if truth is not None:
                    self.vis_ok.append(vis == truth[0])
                    self.false_visible += vis and not truth[0]
                    self.missed_visible += truth[0] and not vis
                    if vis and truth[0]:
                        self.err.append(abs(b - truth[1]))
                        if rng is not None:
                            self.range_err.append(abs(rng - truth[2]))
                with self._lock:
                    self._pending.append((ready,
                                          {"visible": vis, "bearing_deg": b, "range_m": rng, "p_visible": p,
                                           "t": t, "yaw": yaw}))
            except Exception as ex:                  # degrade to "not seen", never crash the flight
                self.errors += 1
                self.last_error = f"{type(ex).__name__}: {ex}"[:160]
            finally:
                self._done.set()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def stats(self):
        lat = sorted(self.latency)
        r = lambda x, k=3: None if x is None else round(float(x), k)  # noqa: E731
        return {"locator": self.model, "lockstep": self.lockstep, "sim_delay_s": self.delay_s,
                "offers": self.n_offer, "busy": self.n_busy, "dropped_stale": self.n_dropped,
                "estimates": len(lat), "errors": self.errors, "last_error": self.last_error,
                "median_latency_s": r(lat[len(lat) // 2]) if lat else None,
                "p90_latency_s": r(lat[int(len(lat) * 0.9)]) if lat else None,
                "bearing_mae_deg": r(np.mean(self.err), 2) if self.err else None, "bearing_n": len(self.err),
                "code_bearing_mae_deg": r(np.mean(self.ref_err), 2) if self.ref_err else None,
                "visible_acc": r(np.mean(self.vis_ok)) if self.vis_ok else None,
                "false_visible": int(self.false_visible), "missed_visible": int(self.missed_visible),
                "warmup_s": getattr(self.backend, "warmup_s", None),
                "range_mae_m": r(np.mean(self.range_err), 2) if self.range_err else None}


# --- v3: the context JSON, and reacquisition of a lost rover ----------------------------------------------
# The drone-rover-v3 checkpoint answers occluded / reappear / reappear_eta / maneuver (probe.questions_v3)
# from the onboard frame plus a small context: how long the rover has been out of sight and where it was
# last seen, relative to the current nose (probe.V3_CONTEXT_KEYS). The state is exactly the training
# records' (rover_data.records_v3 / evaluate_v3): {"image": frame, "context": json.dumps(context)}, with the
# context's keys in V3_CONTEXT_KEYS order, the bearing rounded to 0.1 deg and the range to 0.01 m.

def v3_state(frame, context=None, keys=None):
    """The v3 predict state: the frame plus `context` (a dict over `keys`, default probe.V3_CONTEXT_KEYS) as
    JSON text, in that key order (the training records' state_text order)."""
    import json
    st = {}
    if frame is not None:
        from PIL import Image
        st["image"] = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    ctx = context or {}
    st["context"] = json.dumps({k: ctx.get(k) for k in (keys or probe.V3_CONTEXT_KEYS)})
    return st


class LastSeen:
    """The v3 context as the flight knows it, from whatever the pursuit steers on (the code's segmentation, or
    the locator): unseen_for_s is that source's own, and the last sighting's bearing is re-expressed at the live
    yaw (the aircraft knows how far it has turned since), as rover_data.collect_flight_v3 labels it."""

    def __init__(self):
        self.seen = None                    # (bearing deg at the yaw of that moment, range m or None, yaw rad)

    def update(self, yaw, tgt):
        if tgt is not None and tgt["visible"] and tgt["bearing_deg"] is not None:
            self.seen = (float(tgt["bearing_deg"]), tgt.get("range_m"), float(yaw))

    def context(self, yaw, tgt):
        unseen = None if tgt is None else (0.0 if tgt["visible"] else tgt.get("unseen_for_s"))
        if self.seen is None:
            return {"unseen_for_s": unseen, "last_seen_bearing_deg": None, "last_seen_range_m": None}
        b, r, y0 = self.seen
        b = float((b - np.rad2deg(yaw - y0) + 180.0) % 360.0 - 180.0)
        return {"unseen_for_s": unseen, "last_seen_bearing_deg": round(b, 1),
                "last_seen_range_m": None if r is None else round(float(r), 2)}


REAPPEAR_SIDES = ("left", "ahead", "right", "behind")     # probe.REAPPEAR's keys


def reappear_side(b):
    """probe.REAPPEAR answer for a bearing (deg, + left) off the nose: |b| > 90 behind (wins), |b| <= 20
    ahead, else left / right. The same rule as rover_data.reappear_answer, the v3 label."""
    if abs(b) > 90.0:
        return "behind"
    if abs(b) <= 20.0:
        return "ahead"
    return "left" if b > 0 else "right"


def reappear_truth(m, d, rover_at, t, visible, horizon_s=None):
    """What the v3 labels say for this moment (rover_data.collect_flight_v3): reappear = the side of
    rover_at(t + horizon) from the camera relative to the current nose; occluded = not visible although the
    rover is within the horizontal half-FOV (atan(probe.TAN_H)) and 20 m. -> (side, bearing deg, occluded)."""
    import flight
    horizon_s = probe.REAPPEAR_HORIZON_S if horizon_s is None else horizon_s
    q = d.qpos[3:7]
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    cam = d.qpos[:3] + flight.Eye.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def rel(p):
        r = np.asarray(p, dtype=float) - cam
        f, l = np.cos(yaw) * r[0] + np.sin(yaw) * r[1], -np.sin(yaw) * r[0] + np.cos(yaw) * r[1]
        return float(np.rad2deg(np.arctan2(l, f))), float(np.hypot(f, l))

    b3, _ = rel(rover_at(t + horizon_s))
    bn, rn = rel(d.mocap_pos[m.body("rover").mocapid[0]])
    occl = (not visible) and abs(bn) <= float(np.rad2deg(np.arctan(probe.TAN_H))) and rn < 20.0
    return reappear_side(b3), b3, bool(occl)


# expected reappear time (s) at each probe.REAPPEAR_ETA level (rover_data.ETA_CENTRES, the soft-target centres)
ETA_CENTRES = [1.0, 3.0, 7.0, 13.0]


class SimReappear:
    """Test double for the checkpoint's reacquisition answers: the v3 label itself (reappear_truth), so the
    reacquisition logic can be flown on a CPU. `wrong_p`: probability of a wrong side instead (uniform over
    the other three), from its own seeded stream; `delay_s`: answer latency in sim time. No ETA (None)."""
    instant = True

    def __init__(self, wrong_p=0.0, delay_s=0.0, seed=0):
        self.wrong_p, self.delay_s = float(wrong_p), float(delay_s)
        self.rng = np.random.default_rng([seed, 4099])
        self.model = "sim-reappear(wrong=%g,delay=%g)" % (wrong_p, delay_s)

    def answer(self, frame, context, truth):
        side, _, occl = truth
        if self.wrong_p and self.rng.random() < self.wrong_p:
            others = [s for s in REAPPEAR_SIDES if s != side]
            side = others[int(self.rng.integers(len(others)))]
        return {"side": side, "p_side": 1.0, "occluded": float(occl), "eta_s": None}


class LayaReappear:
    """The v3 checkpoint's own answers: occluded (noul), reappear (choice), reappear_eta (score; read as the
    expected ETA_CENTRES value) in one predict on the frame + v3 context (v3_state). Shares the loaded agent
    (tactics.shared_laya) and warms up before the flight, as the locator does."""
    instant = False
    delay_s = 0.0

    def __init__(self, model=None, device=None, revision=None):
        from tactics import shared_laya
        self.agent = shared_laya(model, device, revision)
        q3 = probe.questions_v3()
        self.qs = {k: q3[k] for k in ("occluded", "reappear", "reappear_eta")}
        self.model = "laya-reappear:" + (model or "default")
        blank = np.zeros((384, 512, 3), dtype=np.uint8)
        ctx = {"unseen_for_s": 2.0, "last_seen_bearing_deg": 30.0, "last_seen_range_m": 3.5}
        t0 = time.time()
        self.answer(blank, ctx, None)
        self.warmup_s = round(time.time() - t0, 2)
        self.answer(blank, ctx, None)

    def answer(self, frame, context, truth=None):
        a = self.agent.predict(v3_state(frame, context), self.qs)["answers"]
        probs = {k: float(v) for k, v in a["reappear"]["probabilities"].items()}
        side = a["reappear"]["choice"]
        pe = np.array([float(a["reappear_eta"]["probabilities"][str(i)]) for i in range(len(ETA_CENTRES))])
        return {"side": side, "p_side": probs.get(side), "probabilities": probs,
                "occluded": float(a["occluded"]["noul"]),
                "eta_s": round(float(np.dot(pe / max(pe.sum(), 1e-9), ETA_CENTRES)), 2)}


def make_reappear_backend(mode, model=None, wrong_p=0.0, delay_s=0.0, seed=0):
    if mode == "sim":
        return SimReappear(wrong_p, delay_s, seed)
    if mode == "laya":
        return LayaReappear(model)
    raise ValueError("reacquire must be sim or laya, got %r" % mode)


class Reacquirer:
    """Asks where a lost rover will reappear: offered every camera frame while the pursuit source has not seen
    the rover for a while (run.episode decides when), answered at most `hz` times a (sim) second. Same shape as
    Locator: a worker thread for a model (latest frame wins while it is busy), inline and deterministic for the
    sim backend. read() returns the latest answer with its frame time and the yaw the frame was taken at (its
    side is relative to that nose), or None. `truth()` -> reappear_truth tuple: the sim answers from it, and
    every answer is scored against it."""

    def __init__(self, backend, hz=3.0, truth=None, gpu=None):
        self.backend = backend
        self.model = backend.model
        self.min_dt = 1.0 / float(hz)
        self.truth = truth
        self.gpu = gpu                      # GpuClock (see Locator)
        self._busy_until = float("-inf")
        self.lockstep = getattr(backend, "instant", False) or gpu is not None
        self.delay_s = getattr(backend, "delay_s", 0.0)
        self._last_sent = float("-inf")
        self._q = queue.Queue(maxsize=1)
        self._pending = []
        self._latest = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.n_offer = self.n_rate = self.n_dropped = self.errors = self.n_answers = 0
        self.last_error = None
        self.latency, self.correct = [], []
        self.sides = {s: 0 for s in REAPPEAR_SIDES}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def offer(self, frame, now, yaw, context):
        self.n_offer += 1
        if now - self._last_sent < self.min_dt or now < self._busy_until:
            self.n_rate += 1
            return
        self._last_sent = now
        item = (frame, now, yaw, context, self.truth() if self.truth else None)
        self._done.clear()
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.n_dropped += 1
            except queue.Empty:
                pass
            self._q.put_nowait(item)
        if self.lockstep:
            self._done.wait()

    def read(self, now):
        with self._lock:
            while self._pending and self._pending[0][0] <= now:
                self._latest = self._pending.pop(0)[1]
            return None if self._latest is None else dict(self._latest)

    def _worker(self):
        while not self._stop.is_set():
            try:
                frame, t, yaw, ctx, truth = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                if self.gpu is not None:
                    a, ready = self.gpu.run(t, lambda: self.backend.answer(frame, ctx, truth), "reacquire")
                    self._busy_until = ready
                else:
                    a, ready = self.backend.answer(frame, ctx, truth), t + self.delay_s
                self.latency.append(time.time() - t0)
                self.n_answers += 1
                self.sides[a["side"]] = self.sides.get(a["side"], 0) + 1
                if truth is not None:
                    self.correct.append(a["side"] == truth[0])
                with self._lock:
                    self._pending.append((ready, dict(a, t=t, yaw=yaw, context=ctx,
                                                                 true_side=None if truth is None else truth[0])))
            except Exception as ex:                  # degrade to "no answer", never crash the flight
                self.errors += 1
                self.last_error = f"{type(ex).__name__}: {ex}"[:160]
            finally:
                self._done.set()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def stats(self):
        lat = sorted(self.latency)
        return {"reacquirer": self.model, "offers": self.n_offer, "rate_limited": self.n_rate,
                "dropped_stale": self.n_dropped, "answers": self.n_answers, "errors": self.errors,
                "last_error": self.last_error, "sides": self.sides,
                "side_acc": round(float(np.mean(self.correct)), 3) if self.correct else None,
                "median_latency_s": round(lat[len(lat) // 2], 3) if lat else None,
                "warmup_s": getattr(self.backend, "warmup_s", None)}
