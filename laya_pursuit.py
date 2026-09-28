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
noise and a delay, so the whole path (worker, staleness, Guidance) runs on a CPU.

Runs in a worker thread like tactics.Tactician: offer() never blocks the 500 Hz loop (unless lockstep),
read() returns the latest estimate and its age. Bearing is + to the aircraft's left, as in flight.py.
"""
import threading, queue, time
import numpy as np
import probe
from rover_data import STEER_CENTRES

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
    return bool(scene["target"]["visible"]), float(np.rad2deg(np.arctan2(left, fwd)))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SimBackend:
    """Test double: the truth, plus noise. `delay_s` is applied in SIM time by the Locator, which also
    takes no new frame until the last answer is out, so the answer rate is 1/delay -- the shape of a
    real model (strips on an L4: 5 x ~40 ms ~ 0.2 s, ~5 Hz). Answered inline (`instant`), so a sim
    flight is deterministic: no thread-timing race on when the answer lands."""
    instant = True

    def __init__(self, noise_deg=0.0, delay_s=0.0, seed=0):
        self.noise_deg, self.delay_s = float(noise_deg), float(delay_s)
        self.rng = np.random.default_rng(seed)
        self.model = "sim(noise=%g,delay=%g)" % (noise_deg, delay_s)

    def locate(self, frame, truth):
        vis, b = truth
        if not vis:
            return False, None, 0.0
        return True, b + self.noise_deg * float(self.rng.normal()), 1.0


class _Laya:
    def __init__(self, model=None, threshold=0.5, device=None, revision=None):
        import laya
        from tactics import LAYA_MODEL, LAYA_BUDGETS
        self.agent = laya.load_vlm(model or LAYA_MODEL, device=device, revision=revision, **LAYA_BUDGETS)
        self.threshold = threshold
        self.delay_s = 0.0                # real latency is wall-clock, measured by the Locator

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
    """One predict on the whole frame: visible (noul) and steer (score over 5 levels)."""

    def __init__(self, model=None, threshold=0.5, **kw):
        super().__init__(model, threshold, **kw)
        self.model = "laya-frame:" + (model or "default")
        qs = probe.questions()
        self.qs = {"visible": qs["visible"], "steer": qs["steer"]}

    def locate(self, frame, truth=None):
        a = self.agent.predict({"image": self._img(frame)}, self.qs)["answers"]
        pv = float(a["visible"]["noul"])
        if pv < self.threshold:
            return False, None, pv
        # expected level (0 = hard left) -> bearing; the centres are evenly spaced, so this is linear
        return True, float(np.interp(float(a["steer"]["score"]), range(len(STEER_CENTRES)), STEER_CENTRES)), pv


def make_locator_backend(mode, model=None, threshold=0.5, noise_deg=0.0, delay_s=0.0, seed=0):
    if mode == "sim":
        return SimBackend(noise_deg, delay_s, seed)
    if mode == "laya-strips":
        return StripsBackend(model, threshold)
    if mode == "laya-frame":
        return FrameBackend(model, threshold)
    raise ValueError("pursuit locator must be sim, laya-strips or laya-frame, got %r" % mode)


class Locator:
    """Where is the rover? Asked every camera frame, answered when the backend is free.

    A frame offered while the worker is busy replaces the one waiting (latest wins), so the answer
    is never older than one inference. `lockstep` makes offer() wait for the answer: the sim stands
    still while the model looks, which measures perception quality with latency taken out.

    `truth()` -> (visible, bearing_deg) is ground truth: the sim backend answers from it, and every
    estimate is scored against it (taken at the frame's time), so a flight reports its own
    perception error."""

    def __init__(self, backend, truth=None, lockstep=False, fresh_s=FRESH_S):
        self.backend = backend
        self.model = backend.model
        self.truth = truth
        self.lockstep = lockstep or getattr(backend, "instant", False)
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
        self.err, self.ref_err, self.vis_ok = [], [], []
        self.false_visible = self.missed_visible = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def offer(self, frame, now, yaw, ref_bearing=None):
        """Non-blocking (unless lockstep). `ref_bearing`: the code's own (segmentation) bearing for
        this frame, scored against the same truth for comparison; never used to steer."""
        self.n_offer += 1
        truth = self.truth() if self.truth else None
        if truth and truth[0] and ref_bearing is not None:
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
            return {"visible": False, "bearing_deg": None, "age_s": None, "fresh": False, "unseen_for_s": None}
        age = now - e["t"]
        fresh = age < self.fresh_s
        b = e["bearing_deg"]
        if b is not None and yaw is not None:
            b -= float(np.rad2deg(_wrap(yaw - e["yaw"])))
        vis = bool(fresh and e["visible"])
        unseen = 0.0 if vis else (None if self._last_seen is None else round(now - self._last_seen, 2))
        return {"visible": vis, "bearing_deg": b if vis else None, "age_s": round(age, 3), "fresh": fresh,
                "unseen_for_s": unseen, "p_visible": e["p_visible"]}

    def _worker(self):
        while not self._stop.is_set():
            try:
                frame, t, yaw, truth = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                vis, b, p = self.backend.locate(frame, truth)
                self.latency.append(time.time() - t0)
                if truth is not None:
                    self.vis_ok.append(vis == truth[0])
                    self.false_visible += vis and not truth[0]
                    self.missed_visible += truth[0] and not vis
                    if vis and truth[0]:
                        self.err.append(abs(b - truth[1]))
                with self._lock:
                    self._pending.append((t + self.delay_s,
                                          {"visible": vis, "bearing_deg": b, "p_visible": p, "t": t, "yaw": yaw}))
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
                "false_visible": int(self.false_visible), "missed_visible": int(self.missed_visible)}
