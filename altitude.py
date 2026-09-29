"""Altitude as a continuous operator: ascend / descend a bit, instead of the one-shot `climb` maneuver.

`climb` (tactics.MANEUVERS) is all or nothing: one answer sends the aircraft to CLIMB_ALT and holds it
there for climb_steps (~3 s). Over a pocket's low front wall that is fatal (results/laya-steer/README.md,
v3.2: one P(climb) call over the threshold stalled half the mixed flights in the first pocket). Here the
checkpoint answers how far to move up or down from where it is now (a score over ALT_LEVELS, read as the
probability-weighted level), about three times a second, and each answer moves the altitude setpoint by at
most MAX_STEP_M. A wrong answer costs half a metre that the next answer takes back; clearing a 2.1 m beam
from cruise takes several ascends in a row, so the "votes" are built into the operator.

    run.episode(..., altitude="sim")                    # the course's own answer (courses.Course.altitude_target)
    run.episode(..., altitude="sim", altitude_wrong_p=0.1)   # with 10% false "ascend 1 m" answers
    run.episode(..., altitude="laya", altitude_model=...)    # a checkpoint trained on question()

With altitude on, a `climb` maneuver from the tactics is ignored (flown as hold_course): altitude is this
module's alone.
"""
import queue, threading, time
import numpy as np

ALT_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]      # metres to move from the current altitude
MAX_STEP_M = 0.5        # one answer moves the setpoint at most this far
ALT_MIN = 1.3           # setpoint floor (run.MIN_ALT = 1.15 is the hard floor under it)
ALT_MAX = 3.2           # setpoint ceiling: over a 2.1 m beam with margin, under every 5 m wall
CRUISE = 1.6            # run.CRUISE_ALT
OVER_BEAM = 2.9         # the course target over a beam (top 2.1 m; run.CLIMB_ALT is 3.0)
CONTEXT_KEYS = ("altitude_m", "unseen_for_s", "last_seen_bearing_deg", "last_seen_range_m")
# with the avoid question (avoid.py) the context also carries the drone's forward speed and its depth sensor
CONTROL_KEYS = CONTEXT_KEYS + ("speed_mps", "sensor")

_OPTIONS = [
    ("descend_1m", "Well above where it should be: nothing needs clearing here any more, and flying high "
                   "loses the rover under the camera. Drop about a metre."),
    ("descend_half", "A little high for this stretch: drop about half a metre."),
    ("hold_altitude", "At the right height for what is ahead: keep it."),
    ("ascend_half", "Something low and full-width ahead must be flown over, and the aircraft is close to "
                    "high enough: rise about half a metre."),
    ("ascend_1m", "A low bar or beam spans the whole corridor ahead and the aircraft is well below the "
                  "height to clear it, with open air above it: rise about a metre. Not for a wall with a "
                  "dead end or a tall wall behind it."),
]


def question():
    """The score question (probe.questions_v3's format and preamble): drone_rover_alt records use it, and
    LayaAltitude asks it. Criteria are the option descriptions, in ALT_LEVELS order."""
    import probe
    ctx = probe.questions()["visible"]["instructions"].split("Is the red rover")[0]
    return {"altitude": {
        "type": "score",
        "instructions": ctx + "How far should the aircraft move up or down from its current altitude "
                        "(altitude_m) to keep following the rover over or past what is ahead?",
        "criteria": [d for _, d in _OPTIONS],
    }}


def label(target_z, z):
    """The trained answer: the move toward the course's target altitude, clipped to +-1 m (a float in metres;
    rover_data turns it into soft level targets)."""
    return float(np.clip(target_z - z, ALT_LEVELS[0], ALT_LEVELS[-1]))


def read(probs, sharpen=1.0):
    """Probability-weighted level, in metres. `probs`: level probabilities in ALT_LEVELS order."""
    p = np.asarray(probs, dtype=float) ** float(sharpen)
    return float(np.dot(p / max(p.sum(), 1e-12), ALT_LEVELS))


DEADBAND_M = 0.2        # an answer this close to zero keeps the setpoint


def next_setpoint(sp, z, dz):
    """Where one answer moves the setpoint: toward z + dz (the answer is relative to the altitude the frame
    was taken at), at most MAX_STEP_M, inside [ALT_MIN, ALT_MAX] -- and only in the answer's direction. A
    descend answer never raises the setpoint: re-anchoring on z alone let an aircraft pushed upward by
    something else (flight.Pilot's airmode lift in hard yaw) drag the setpoint up with it while every answer
    said descend (v3.3 GIF flights, no-climb seed 1: setpoint 1.6 -> 3.2 m in 1.4 s of descend answers).
    |dz| < DEADBAND_M keeps the setpoint (no re-anchoring jitter at hold)."""
    if abs(dz) < DEADBAND_M:
        return float(sp)
    want = z + dz
    if dz > 0:
        new = max(sp, min(want, sp + MAX_STEP_M))
    else:
        new = min(sp, max(want, sp - MAX_STEP_M))
    return float(np.clip(new, ALT_MIN, ALT_MAX))


class SimAltitude:
    """Test double: the course's own answer (label(target, z)). With probability `wrong_p` it answers
    `wrong` instead (default +1 m, the false-climb error; "random": any level, for data flights that visit
    off-target altitudes), from its own seeded stream."""
    instant = True
    delay_s = 0.0

    def __init__(self, target_fn, wrong_p=0.0, wrong=1.0, seed=0, avoid=False):
        self.avoid = bool(avoid)             # also answer avoid.question() from the true sensor (avoid.label)
        self.target_fn, self.wrong_p = target_fn, float(wrong_p)
        self.wrong = wrong if wrong == "random" else float(wrong)
        self.rng = np.random.default_rng([seed, 7919])
        self.model = "sim-altitude(wrong=%g@%s)" % (wrong_p, wrong)

    def answer(self, frame, context):
        out = {}
        if self.avoid and context.get("sensor"):
            import avoid
            out["avoid"] = avoid.label(context["sensor"], context.get("speed_mps") or 0.0)
        if self.wrong_p and self.rng.random() < self.wrong_p:
            if self.wrong == "random":             # data flights: wander off the target altitude
                return dict(out, dz=float(self.rng.choice(ALT_LEVELS)))
            return dict(out, dz=self.wrong)
        return dict(out, dz=label(self.target_fn(), context["altitude_m"]))


class LayaAltitude:
    """A checkpoint's answer to question() on the frame + context (the v3 state format plus altitude_m).
    Shares the loaded agent (tactics.shared_laya) and warms up before the flight."""
    instant = False
    delay_s = 0.0

    def __init__(self, model=None, sharpen=1.0, device=None, revision=None, avoid=False):
        from tactics import shared_laya
        self.agent = shared_laya(model, device, revision)
        self.qs = question()
        self.avoid = bool(avoid)             # also ask avoid.question() in the same predict (CONTROL_KEYS context)
        self.keys = CONTROL_KEYS if self.avoid else CONTEXT_KEYS
        if self.avoid:
            import avoid as _av
            self.qs.update(_av.question())
        self.sharpen = float(sharpen)
        self.model = "laya-altitude:" + (model or "default")
        blank = np.zeros((384, 512, 3), dtype=np.uint8)
        ctx = {"altitude_m": CRUISE, "unseen_for_s": 0.0, "last_seen_bearing_deg": 0.0, "last_seen_range_m": 3.5,
               "speed_mps": 0.0, "sensor": {k: 20.0 for k in ("far_left", "left", "center", "right", "far_right",
                                                             "path_ahead")}}
        t0 = time.time()
        self.answer(blank, ctx)
        self.warmup_s = round(time.time() - t0, 2)
        self.answer(blank, ctx)

    def answer(self, frame, context):
        import laya_pursuit
        st = laya_pursuit.v3_state(frame, context, keys=self.keys)
        ans = self.agent.predict(st, self.qs)["answers"]
        a = ans["altitude"]
        pr = [float(a["probabilities"][str(i)]) for i in range(len(ALT_LEVELS))]
        out = {"dz": round(read(pr, self.sharpen), 3), "probabilities": [round(p, 3) for p in pr]}
        if self.avoid:
            out["avoid"] = ans["avoid"]["choice"]
            out["avoid_probs"] = {k: round(float(v), 3) for k, v in ans["avoid"]["probabilities"].items()}
        return out


def make_backend(mode, target_fn=None, model=None, wrong_p=0.0, wrong=1.0, sharpen=1.0, seed=0, avoid=False):
    if mode == "sim":
        return SimAltitude(target_fn, wrong_p, wrong, seed, avoid=avoid)
    if mode == "laya":
        return LayaAltitude(model, sharpen, avoid=avoid)
    raise ValueError("altitude must be sim or laya, got %r" % (mode,))


class Altimeter:
    """Offered every camera frame, answered at most `hz` times a (sim) second; the Reacquirer's shape (a worker
    thread for a model, latest frame wins while it is busy; inline for the sim). read() returns the setpoint
    after every answer that has arrived by then (each moved by next_setpoint from the altitude its frame was
    taken at). `truth()` -> the course's target altitude, to score the answers."""

    def __init__(self, backend, hz=3.0, truth=None, start=CRUISE, gpu=None):
        self.backend = backend
        self.model = backend.model
        self.min_dt = 1.0 / float(hz)
        self.truth = truth
        self.gpu = gpu                      # laya_pursuit.GpuClock: latency-faithful virtual time
        self._busy_until = float("-inf")
        self.lockstep = getattr(backend, "instant", False) or gpu is not None
        self.sp = float(start)
        self._last_sent = float("-inf")
        self._q = queue.Queue(maxsize=1)
        self._pending = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.n_offer = self.n_rate = self.n_dropped = self.errors = self.n_answers = 0
        self.last_error = None
        self.latency, self.abs_err, self.dzs = [], [], []
        self.avoid = None                    # the latest avoid answer that has arrived: (answer, frame time)
        self.avoid_counts, self.avoid_ok = {}, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def offer(self, frame, now, context):
        self.n_offer += 1
        if now - self._last_sent < self.min_dt or now < self._busy_until:
            self.n_rate += 1
            return
        self._last_sent = now
        item = (frame, now, dict(context), self.truth() if self.truth else None)
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
                _, z, dz, av, tf = self._pending.pop(0)
                self.sp = next_setpoint(self.sp, z, dz)
                if av is not None:
                    self.avoid = (av, tf)
            return self.sp

    def _worker(self):
        while not self._stop.is_set():
            try:
                frame, t, ctx, truth = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                if self.gpu is not None:
                    a, ready = self.gpu.run(t, lambda: self.backend.answer(frame, ctx), "altitude")
                    self._busy_until = ready
                else:
                    a, ready = self.backend.answer(frame, ctx), t + self.backend.delay_s
                self.latency.append(time.time() - t0)
                self.n_answers += 1
                self.dzs.append(a["dz"])
                if truth is not None:
                    self.abs_err.append(abs(a["dz"] - label(truth, ctx["altitude_m"])))
                if "avoid" in a:
                    self.avoid_counts[a["avoid"]] = self.avoid_counts.get(a["avoid"], 0) + 1
                    if ctx.get("sensor"):
                        import avoid as _av
                        self.avoid_ok.append(a["avoid"] == _av.label(ctx["sensor"], ctx.get("speed_mps") or 0.0))
                with self._lock:
                    self._pending.append((ready, ctx["altitude_m"], a["dz"], a.get("avoid"), t))
            except Exception as e:           # a failed call leaves the setpoint where it was
                self.errors += 1
                self.last_error = repr(e)[:200]
            finally:
                self._done.set()

    def stats(self):
        dz = np.array(self.dzs) if self.dzs else np.zeros(0)
        return {"model": self.model, "answers": self.n_answers, "offers": self.n_offer, "rate_limited": self.n_rate,
                "dropped": self.n_dropped, "errors": self.errors, "last_error": self.last_error,
                "median_latency_s": round(float(np.median(self.latency)), 3) if self.latency else None,
                "dz_mae_m": round(float(np.mean(self.abs_err)), 3) if self.abs_err else None,
                "ascend_pct": round(100 * float(np.mean(dz > 0.25)), 1) if dz.size else None,
                "descend_pct": round(100 * float(np.mean(dz < -0.25)), 1) if dz.size else None,
                **({"avoid_counts": dict(self.avoid_counts),
                    "avoid_acc": round(float(np.mean(self.avoid_ok)), 3) if self.avoid_ok else None}
                   if self.avoid_counts else {})}

    def close(self):
        self._stop.set()
