"""Laya flies by command: the lookahead teacher's choice (teacher.py) as three questions in the drone's frame.

The teacher picks among (forward speed, side slide, heading offset from the rover). Laya cannot know the rover's
true direction, so the student's command is expressed relative to its own nose: speed, slide and turn (how far
the heading setpoint should be from the current nose). Labels are soft: the teacher's candidate scores become a
distribution (softmax at TEMP; a collision's -1000 is ~0), and each question's target is that distribution
spread over its levels (rover_data.soft_target per candidate, weighted) -- AlphaZero's visit-count targets
(github.com/ericjang/autogo), not just the argmax. run.episode(policy="laya-cmd") flies the answers.
"""
import numpy as np

SPEED_LEVELS = [-1.0, 0.5, 2.5, 4.5, 6.5]            # teacher.SPEEDS
SLIDE_LEVELS = [-1.8, 0.0, 1.8]                      # teacher.SLIDES
TURN_LEVELS = [-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0]   # deg from the nose, + = left
TEMP = 5.0
CMD_KEYS = ("altitude_m", "speed_mps", "travel_deg", "prev_speed", "prev_turn", "lidar")
# without the previous command: in the teacher's data it predicts the next choice nearly as well as the student
# does, and a student that leans on it copies its own lagging commands in flight (autoresearch X1/X6)
CMD_KEYS_NOPREV = tuple(k for k in CMD_KEYS if not k.startswith("prev_"))


def question():
    import probe
    ctx = probe.questions()["visible"]["instructions"].split("Is the red rover")[0]
    lid = ("The context has the drone's altitude, its speed and direction of travel, the command it flew last "
           "(speed and turn), and its 360-degree lidar (metres to the nearest return in 24 sectors of 15 degrees, "
           "from dead ahead counter-clockwise: 6 is left, 12 behind, 18 right). ")
    return {
        "cmd_speed": {"type": "score", "instructions": ctx + lid + "How fast should the drone fly forward for the "
                      "next half second to keep following the rover without hitting anything?",
                      "criteria": ["back away about 1 m/s" if v < 0 else "about %g m/s" % v for v in SPEED_LEVELS]},
        "cmd_slide": {"type": "score", "instructions": ctx + lid + "Should the drone slide sideways for the next half "
                      "second, and which way?", "criteria": ["slide right", "no slide", "slide left"]},
        "cmd_turn": {"type": "score", "instructions": ctx + lid + "How far should the drone turn its heading from "
                     "where it points now (positive = to its left)?",
                     "criteria": ["turn right about %d degrees" % -v if v < 0 else
                                  ("keep heading" if v == 0 else "turn left about %d degrees" % v) for v in TURN_LEVELS]},
    }


def soft_targets(cands, scores, turns, temp=None):
    """Per-question soft targets from the teacher's candidate scores. `cands`: (speed, slide, offset) per
    candidate; `turns`: each candidate's heading setpoint relative to the nose (deg) at the decision. `temp`:
    the softmax temperature over scores (default TEMP)."""
    from rover_data import soft_target
    s = np.asarray(scores, dtype=float)
    p = np.exp((s - s.max()) / (TEMP if temp is None else temp))
    p = p / p.sum()
    out = {}
    for name, levels, val in (("cmd_speed", SPEED_LEVELS, [c[0] for c in cands]),
                              ("cmd_slide", SLIDE_LEVELS, [c[1] for c in cands]),
                              ("cmd_turn", TURN_LEVELS, turns)):
        t = np.zeros(len(levels))
        for pi, v in zip(p, val):
            t += pi * np.asarray(soft_target(float(v), levels))
        out[name] = [round(float(x), 4) for x in t / max(t.sum(), 1e-12)]
    return out


def read(probs, levels):
    p = np.asarray(probs, dtype=float)
    return float(np.dot(p / max(p.sum(), 1e-12), levels))


class LayaCommand:
    """A checkpoint's answers to question() on the frame + CMD_KEYS context, in one predict."""
    instant = False
    delay_s = 0.0

    def __init__(self, model=None, device=None, revision=None, keys=None):
        import time
        from tactics import shared_laya
        self.agent = shared_laya(model, device, revision)
        self.keys = tuple(keys or CMD_KEYS)
        self.qs = question()
        self.model = "laya-cmd:" + (model or "default")
        blank = np.zeros((384, 512, 3), dtype=np.uint8)
        ctx = {"altitude_m": 1.6, "speed_mps": 0.0, "travel_deg": 0.0, "prev_speed": 0.0, "prev_turn": 0.0,
               "lidar": {"sectors": [20.0] * 24}}
        t0 = time.time()
        self.answer(blank, ctx)
        self.warmup_s = round(time.time() - t0, 2)
        self.answer(blank, ctx)

    def probs(self, frame, context):
        """Each question's probability per level (the policy RL samples from)."""
        import laya_pursuit
        a = self.agent.predict(laya_pursuit.v3_state(frame, context, keys=self.keys), self.qs)["answers"]
        return {q: [float(a[q]["probabilities"][str(i)]) for i in range(len(levels))]
                for q, levels in (("cmd_speed", SPEED_LEVELS), ("cmd_slide", SLIDE_LEVELS), ("cmd_turn", TURN_LEVELS))}

    def answer(self, frame, context):
        import laya_pursuit
        a = self.agent.predict(laya_pursuit.v3_state(frame, context, keys=self.keys), self.qs)["answers"]
        get = lambda q, levels: read([float(a[q]["probabilities"][str(i)]) for i in range(len(levels))], levels)  # noqa: E731
        return {"speed": round(get("cmd_speed", SPEED_LEVELS), 2), "slide": round(get("cmd_slide", SLIDE_LEVELS), 2),
                "turn": round(get("cmd_turn", TURN_LEVELS), 1)}


class CommandStream:
    """Offered every camera frame, answered at most `hz` times a (sim) second (the Altimeter's shape: a worker
    thread, latest frame wins; laya_pursuit.GpuClock for timing="virtual"). read(now) -> the latest answer that
    has arrived, with the yaw its frame was taken at and its frame time, or None."""

    def __init__(self, backend, hz=10.0, gpu=None):
        import queue, threading
        self.backend, self.model = backend, backend.model
        self.min_dt = 1.0 / float(hz)
        self.gpu = gpu
        self.lockstep = gpu is not None
        self._busy_until = self._last_sent = float("-inf")
        self._q = queue.Queue(maxsize=1)
        self._pending, self._latest = [], None
        self._lock, self._done = threading.Lock(), threading.Event()
        self.n_offer = self.n_answers = self.errors = 0
        self.last_error = None
        self.latency = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def offer(self, frame, now, yaw, context):
        import queue
        self.n_offer += 1
        if now - self._last_sent < self.min_dt or now < self._busy_until:
            return
        self._last_sent = now
        self._done.clear()
        item = (frame, now, yaw, dict(context))
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
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
        import queue, time
        while not self._stop.is_set():
            try:
                frame, t, yaw, ctx = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                if self.gpu is not None:
                    a, ready = self.gpu.run(t, lambda: self.backend.answer(frame, ctx), "command")
                    self._busy_until = ready
                else:
                    a, ready = self.backend.answer(frame, ctx), t
                self.latency.append(time.time() - t0)
                self.n_answers += 1
                with self._lock:
                    self._pending.append((ready, dict(a, yaw=yaw, t=t)))
            except Exception as e:
                self.errors += 1
                self.last_error = repr(e)[:200]
            finally:
                self._done.set()

    def stats(self):
        lat = sorted(self.latency)
        return {"model": self.model, "answers": self.n_answers, "offers": self.n_offer, "errors": self.errors,
                "last_error": self.last_error, "median_latency_s": round(lat[len(lat) // 2], 3) if lat else None}

    def close(self):
        self._stop.set()
