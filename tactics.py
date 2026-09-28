"""Jev tactical layer: the drone's programmable common sense.

Everything a human should review lives at the top of this file: the questions,
the option rubrics, and the thresholds. Nothing else in the project hard-codes a
number that changes how the aircraft reacts to a judgment.

Runs off the control loop in a worker thread. The flight code never blocks on it
and never *needs* it -- it reads whatever judgment is currently cached, and the
reflex layer in run.py owns safety regardless of what comes back.

Two backends answer the same questions: Jev over the TypeSafe API (JSON only), or
Laya Vision (https://github.com/r33drichards/laya-vision) run locally, which can
also see the onboard camera frame.
"""
import os, threading, queue, time
import numpy as np

MODEL = "jev-latest"
LAYA_MODEL = "thaitea/laya-vision"
# Laya's per-option token cap defaults to 48; these budgets keep every rubric below whole.
LAYA_BUDGETS = {"option_max_len": 256, "head_max_len": 1024, "max_len": 3072}

# --- how the flight code reacts to a judgment -------------------------------
THRESHOLDS = {
    "stale_after_s": 1.5,     # ignore a judgment older than this; the world moved on
    "risk_slow_down": 1.45,   # score above which we bleed speed regardless of maneuver
    "call_hz": 3.0,           # upper bound on how often we ask
    "call_budget": 160,      # hard cap per episode, so a bug cannot run up a bill
    "consult_within_m": 4.0,  # only ask when something is actually in the way...
    "consult_lost_s": 1.0,    # ...or the target has been missing this long
    "climb_steps": 150,       # ~3s: long enough to rise AND cross, not just bob up
    "commit_steps": 55,       # ~1.4s at 50Hz: commit to a maneuver instead of chattering
    "override_risk": 1.7,     # ...unless things get this dangerous, then re-decide now
    "really_lost": 0.5,       # Noul above which we stop trusting the remembered bearing
}

# The drone's own capabilities. Without this the model cannot know that "climb"
# is physically available, or what counts as a small height.
AIRCRAFT = {
    "type": "quadrotor, camera-only, no map and no GPS",
    "cruise_altitude_m": 1.6,
    "can_climb_to_m": 3.0,
    "climb_takes_about_s": 1.5,
    "top_speed_mps": 3.6,
    "note": "All distances are from a forward camera. 25 m means nothing was detected.",
}

MISSION = ("Follow the ground rover and keep it in view. Do not hit anything. "
           "Losing the rover briefly is acceptable; hitting an obstacle is not.")

MANEUVERS = {
    "hold_course": (
        "Nothing meaningfully blocks the pursuit line: the path ahead is clear for "
        "several metres. Keep flying straight at the target."),
    "gap_left": (
        "Something blocks the way ahead, and the left sectors show clearly more free "
        "space than the right. Steer around it to the left."),
    "gap_right": (
        "Something blocks the way ahead, and the right sectors show clearly more free "
        "space than the left. Steer around it to the right."),
    "climb": (
        "The obstruction ahead is LOW: free_ahead_above_m is much larger than "
        "free_ahead_level_m, so there is clear air over the top of it. This is the right "
        "answer when every sector is blocked, because that means there is no gap to "
        "steer through, but the thing is short enough to simply fly over."),
    "brake": (
        "Close to something on several sides and no option is clearly better. Bleed off "
        "speed and hold until the picture improves."),
    "reacquire": (
        "The target has been out of sight long enough that the remembered bearing is "
        "stale. Stop chasing it and sweep to find the target again."),
}

QUESTIONS = {
    "maneuver": {
        "type": "choice",
        "instructions": {
            "role": "You are the tactical decision layer of an autonomous quadrotor.",
            "mission": MISSION,
            "ask": "Which single maneuver should the drone commit to right now?",
        },
        "criteria": MANEUVERS,
    },
    "risk": {
        "type": "score",
        "instructions": "How dangerous is the drone's immediate situation?",
        "criteria": ["clear and open", "tight but manageable", "about to hit something"],
    },
    "target_truly_lost": {
        "type": "noul",
        "instructions": (
            "Has the drone genuinely lost the rover? Judge from unseen_for_s: a fraction "
            "of a second behind a pillar is a normal occlusion, several seconds of nothing "
            "in an open scene means the chase line is stale."),
        "criteria": {"true": "Give up the remembered bearing and sweep to search.",
                     "false": "Keep flying the last known bearing; it should reappear."},
    },
}


def decision_needed(scene):
    """Code decides WHEN there is a judgment worth paying for. On an empty corridor
    with the target in view there is nothing to decide, so we do not ask."""
    return (scene["nearest_obstacle_m"] < THRESHOLDS["consult_within_m"]
            or scene["sectors_blocked"] >= 1
            or (scene["target"]["unseen_for_s"] or 0.0) >= THRESHOLDS["consult_lost_s"])


def build_state(scene):
    """Everything the model is allowed to know, and nothing it cannot observe."""
    return {"mission": MISSION, "aircraft": AIRCRAFT, "observed": scene}


class JevBackend:
    """Jev over the TypeSafe API. Takes JSON only; an image is ignored."""
    sees_images = False

    def __init__(self, model=MODEL):
        from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
        key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not key:
            raise RuntimeError("set TYPESAFE_API_KEY (see .env.example)")
        self.client = TypeSafeClient(api_key=key)
        self.model = model
        kinds = {"choice": Choice, "score": Score, "noul": Noul}
        self.questions = {k: kinds[q["type"]](instructions=q["instructions"], criteria=q["criteria"])
                          for k, q in QUESTIONS.items()}

    def ask(self, state, image=None):
        r = self.client.system_one(state=state, model=self.model, questions=self.questions)
        a = r.answers
        return {
            "maneuver": a["maneuver"].choice,
            "confidence": round(a["maneuver"].confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in a["maneuver"].probabilities.items()},
            "risk": round(a["risk"].score, 2),
            "target_truly_lost": round(a["target_truly_lost"].noul, 3),
            "source": "jev",
        }, r.usage.input_tokens + r.usage.output_tokens

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class LayaBackend:
    """Laya Vision, loaded in-process. Same questions; with `use_image` the onboard
    camera frame goes in beside the JSON state."""

    def __init__(self, model=LAYA_MODEL, use_image=False, device=None, revision=None, **budgets):
        import laya
        self.agent = laya.load_vlm(model, device=device, revision=revision, **dict(LAYA_BUDGETS, **budgets))
        self.model = model + (" +image" if use_image else "")
        self.sees_images = use_image
        self.truncated = 0

    def ask(self, state, image=None):
        if self.sees_images and image is not None:
            from PIL import Image
            state = dict(state, image=Image.fromarray(image) if not isinstance(image, Image.Image) else image)
        r = self.agent.predict(state, QUESTIONS)
        a = r["answers"]
        if any("truncated" in v for v in a.values()):
            self.truncated += 1
        return {
            "maneuver": a["maneuver"]["choice"],
            "confidence": round(float(a["maneuver"]["confidence"]), 3),
            "probabilities": {k: round(float(v), 3) for k, v in a["maneuver"]["probabilities"].items()},
            "risk": round(float(a["risk"]["score"]), 2),
            "target_truly_lost": round(float(a["target_truly_lost"]["noul"]), 3),
            "source": "laya",
        }, r["usage"]["input_tokens"]

    def close(self):
        pass


_SHARED = {}
_SHARED_LOCK = threading.Lock()


class _LockedAgent:
    """One loaded Laya checkpoint, shared by every user in the process (the pursuit locator, the v3
    tactics, the reacquisition asker), with predict() serialised: they run in separate worker threads,
    and one model on one GPU answers one state at a time anyway. Loading a checkpoint twice costs a
    second copy of the weights in GPU memory and another load of several seconds."""

    def __init__(self, agent, name):
        self.agent, self.name = agent, name
        self.lock = threading.Lock()
        self.users = 0

    def predict(self, state, questions, **kw):
        with self.lock:
            return self.agent.predict(state, questions, **kw)

    def __getattr__(self, k):
        return getattr(self.agent, k)


def shared_laya(model=None, device=None, revision=None):
    """laya.load_vlm(model, **LAYA_BUDGETS), loaded once per (model, device, revision) per process."""
    key = (model or LAYA_MODEL, device, revision)
    with _SHARED_LOCK:
        a = _SHARED.get(key)
        if a is None:
            import laya
            a = _SHARED[key] = _LockedAgent(laya.load_vlm(key[0], device=device, revision=revision, **LAYA_BUDGETS),
                                            key[0])
        a.users += 1
        return a


# What the v3 tactical backend reports for the two questions the drone-rover-v3 checkpoint is not
# trained on. ConstBackend's values (the oracle control's): below override_risk (1.7), risk_slow_down
# (1.45) and really_lost (0.5), so neither ever re-decides a commitment, bleeds speed or forces the
# reacquire search. v3 tactics then differ from const:oracle ONLY in the maneuver, and what happens to a
# lost rover is left to the pursuit / reacquisition layer (run.py reacquire=...), not to a made-up noul.
V3_RISK = 0.93
V3_LOST = 0.45
# drone-rover-v3/v3.1's `maneuver` almost never answers climb by argmax (climb recall 0-5% on held-out frames)
# but ranks climb frames well (AUC 0.91-0.94), so climb is decided by P(climb) >= V3_CLIMB_P. 0.12 is the best
# F1 for v3.1/best on the held-out rover-test-v3 frames (results/probe/rover-test-v3/..v3.1_best/preds-v3.jsonl:
# recall 0.54, precision 0.62; 0.10: 0.62 / 0.47; 0.15: 0.38 / 0.75). On the behind-heavy rover-test-v3b set
# precision is far lower (0.06 at 0.12). A false climb is vetoed by Guidance's climb check unless there is
# measured clear air over a full-width obstacle -- which a pocket's low front wall also passes, so a false climb
# there still costs; a missed climb at a beam stops the flight.
V3_CLIMB_P = 0.12


class LayaV3Backend:
    """The drone-rover-v3 checkpoint's own tactical question: probe.questions_v3()["maneuver"] (options =
    MANEUVERS, trained on the course oracle's answer) from the onboard frame plus the v3 context JSON
    (probe.V3_CONTEXT_KEYS: unseen_for_s, last_seen_bearing_deg, last_seen_range_m), in the state format
    laya_pursuit.v3_state builds (the one the v3 training records use). With `with_scene` the Tactician's
    scene JSON goes in as well (not the trained format; off by default).

    The checkpoint answers only `maneuver`; risk and target_truly_lost are fixed at V3_RISK / V3_LOST (see
    there). The loaded agent is tactics.shared_laya's, so a laya pursuit locator on the same checkpoint
    reuses it. Warmed up at construction, like the locator, so the flight's first call is not the slow one.

    `climb_p`: answer climb whenever P(climb) >= climb_p, else the argmax (V3_CLIMB_P; None = argmax only).
    `climb_votes`: only once P(climb) >= climb_p on that many calls in a row. One call is not enough: a climb
    commits Guidance for commit_steps, and over a pocket's low front wall that flies the aircraft into the
    pocket. drone-rover-v3.2a at 0.1543 crossed the threshold at least once in 14 of 21 pocket visits of the
    rover-test-tac probe flights, and flights stalled in the first pocket (8/16 on mixed with code pursuit);
    3 in a row at 0.12, at the flight's ~1.5 calls/s, fires in 1-2 of 21 pocket visits and 16 of 17 beams."""
    sees_images = True
    wants_context = True

    def __init__(self, model=None, device=None, revision=None, with_scene=False, climb_p=V3_CLIMB_P,
                 climb_votes=1, **_ignored):
        import probe
        self.agent = shared_laya(model, device, revision)
        self.model = "laya-v3:" + (model or "default") + ("+scene" if with_scene else "")
        self.with_scene = bool(with_scene)
        self.climb_p = None if climb_p is None else float(climb_p)
        if self.climb_p is not None:
            self.model += ":climb_p=%g" % self.climb_p
        self.climb_votes = max(1, int(climb_votes))
        if self.climb_votes > 1:
            self.model += ":votes=%d" % self.climb_votes
        self.over = 0                       # consecutive calls with P(climb) >= climb_p
        self.n_climb_threshold = 0          # calls where the threshold, not the argmax, chose climb
        self.qs = {"maneuver": probe.questions_v3()["maneuver"]}
        self.truncated = 0
        self.warmup_s = None
        self.warm_up()
        self.over = 0

    def warm_up(self):
        import numpy as np
        blank = np.zeros((384, 512, 3), dtype=np.uint8)
        ctx = {"unseen_for_s": 0.0, "last_seen_bearing_deg": 0.0, "last_seen_range_m": 3.5}
        t0 = time.time()
        self.ask({"observed": {}}, blank, ctx)
        self.warmup_s = round(time.time() - t0, 2)
        self.ask({"observed": {}}, blank, ctx)

    def ask(self, state, image=None, context=None):
        import laya_pursuit
        st = laya_pursuit.v3_state(image, context)
        if self.with_scene:
            st["scene"] = state.get("observed", state)
        r = self.agent.predict(st, self.qs)
        a = r["answers"]["maneuver"]
        if "truncated" in a:
            self.truncated += 1
        probs = {k: float(v) for k, v in a["probabilities"].items()}
        mv = a["choice"]
        if self.climb_p is not None:
            self.over = self.over + 1 if probs.get("climb", 0.0) >= self.climb_p else 0
            if mv == "climb" and self.over < self.climb_votes:
                mv = max((k for k in probs if k != "climb"), key=probs.get)
            elif mv != "climb" and self.over >= self.climb_votes:
                mv = "climb"
                self.n_climb_threshold += 1
        return {
            "maneuver": mv,
            "confidence": round(float(probs.get(mv, a["confidence"])), 3),
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "risk": V3_RISK,
            "target_truly_lost": V3_LOST,
            "source": "laya",
        }, r.get("usage", {}).get("input_tokens", 0)

    def close(self):
        pass


class ConstBackend:
    """A control, not a model: the same answer every time. If a model flies no better
    than this, its judgments are not what is flying the course."""
    sees_images = False

    def __init__(self, maneuver="climb", risk=0.93, lost=0.45, wrong_p=0.0, wrong="climb", seed=0):
        self.model = "const:" + maneuver + ("(wrong=%g@%s)" % (wrong_p, wrong) if wrong_p else "")
        # `wrong_p`: answer `wrong` instead with this probability (a control for altitude.SimAltitude's
        # wrong_p: the same rate of false climbs, into the one-shot climb instead of a half-metre step)
        self.wrong_p, self.wrong = float(wrong_p), wrong
        self.rng = np.random.default_rng([seed, 6007])
        self.answer = {"maneuver": maneuver, "confidence": 0.0, "probabilities": {m: float(m == maneuver) for m in MANEUVERS},
                       "risk": risk, "target_truly_lost": lost, "source": "laya"}

    def bind(self, answer_fn):
        """`const:oracle`: answer from the course layout (courses.Course.oracle) instead."""
        self.answer_fn = answer_fn

    def ask(self, state, image=None):
        fn = getattr(self, "answer_fn", None)
        if fn is None:
            return dict(self.answer), 0
        mv = fn()
        if self.wrong_p and self.rng.random() < self.wrong_p:
            mv = self.wrong
        return dict(self.answer, maneuver=mv, probabilities={m: float(m == mv) for m in MANEUVERS}), 0

    def close(self):
        pass


def make_backend(name="jev", **kw):
    if name.startswith("const:"):
        return ConstBackend(name.split(":", 1)[1], **{k: v for k, v in kw.items() if k in ("wrong_p", "wrong", "seed")})
    if name == "jev":
        return JevBackend(**({"model": kw["model"]} if kw.get("model") else {}))
    if name == "laya":
        return LayaBackend(**{k: v for k, v in kw.items() if v is not None})
    if name == "laya-v3":
        return LayaV3Backend(**{k: v for k, v in kw.items() if k != "use_image" and (v is not None or k == "climb_p")})
    raise ValueError("backend must be jev, laya, laya-v3 or const:<maneuver>, got %r" % name)


DEFAULT = {"maneuver": "hold_course", "risk": 0.0, "confidence": 0.0,
           "target_truly_lost": 0.0, "source": "default", "age_s": 0.0,
           "probabilities": {}}


class Tactician:
    """Asks the backend for a judgment at most `hz` times a second, and only when the
    scene has actually changed enough to be worth a call.

    `lockstep` makes offer() wait for the answer: the sim stands still while the
    model thinks, which measures judgment quality with latency taken out."""

    def __init__(self, hz=THRESHOLDS["call_hz"], budget=THRESHOLDS["call_budget"], backend=None, lockstep=False):
        self.backend = backend or JevBackend()
        self.min_dt = 1.0 / hz
        self.budget = budget
        self.model = self.backend.model
        self.lockstep = lockstep
        self._done = threading.Event()
        self.calls = 0
        self.attempts = 0
        self.skipped = 0
        self.errors = 0
        self.n_offer = 0
        self.n_ratelimited = 0
        self.n_full = 0
        self.last_error = None
        self.tokens = 0
        self.latency = []
        self._q = queue.Queue(maxsize=1)
        self._latest = dict(DEFAULT)
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._last_sent = float("-inf")
        self._last_key = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @staticmethod
    def _key(scene):
        """Coarse fingerprint: only re-ask when the situation is materially new."""
        t = scene["target"]
        return (
            scene["sectors_blocked"],
            min(int(scene["free_ahead_above_m"] / 3.0), 8),
            tuple(min(int(v / 1.5), 6) for v in scene["sector_range_m"].values()),
            min(int(scene["nearest_obstacle_m"] / 1.5), 6),
            t["visible"],
            None if t["bearing_deg"] is None else int(t["bearing_deg"] / 15),
            (t["unseen_for_s"] or 0) > 2.0,
        )

    def offer(self, scene, now, image=None, context=None):
        """Non-blocking (unless lockstep). Hand the latest scene over if it's worth a call. `context`: the
        v3 context dict (laya_pursuit.LastSeen.context), passed on only to a backend that `wants_context`."""
        self.n_offer += 1
        if self.attempts >= self.budget or now - self._last_sent < self.min_dt:
            self.n_ratelimited += 1
            return
        key = self._key(scene)
        if key == self._last_key:
            self.skipped += 1
            return
        try:
            self._done.clear()
            self._q.put_nowait((scene, now, image, context))
            self._last_sent, self._last_key = now, key
        except queue.Full:
            self.n_full += 1
            return
        if self.lockstep:
            self._done.wait()

    def read(self, now):
        with self._lock:
            out = dict(self._latest)
        out["age_s"] = round(now - self._stamp, 2)
        return out

    def _worker(self):
        while not self._stop.is_set():
            try:
                scene, now, image, context = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            self.attempts += 1  # counts against the budget whether or not the call succeeds
            try:
                if getattr(self.backend, "wants_context", False):
                    judgment, tokens = self.backend.ask(build_state(scene), image, context)
                else:
                    judgment, tokens = self.backend.ask(build_state(scene), image)
                self.tokens += tokens
                self.calls += 1
                self.latency.append(time.time() - t0)
                with self._lock:
                    self._latest, self._stamp = judgment, now
            except Exception as e:                    # degrade, never crash the flight
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"[:160]
                # A failed call replaces the cached judgment with an error placeholder, so the
                # next offer() for this same scene must not be skipped as "unchanged" - otherwise
                # a static scene never gets re-asked and never recovers a real judgment.
                self._last_key = None
                with self._lock:
                    self._latest = dict(DEFAULT, source=f"error:{type(e).__name__}")
            finally:
                self._done.set()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.backend.close()

    def stats(self):
        lat = sorted(self.latency)
        return {"backend": self.model, "lockstep": self.lockstep,
                "truncated_calls": getattr(self.backend, "truncated", None), "calls": self.calls, "attempts": self.attempts, "skipped_unchanged": self.skipped, "errors": self.errors,
                "last_error": self.last_error, "tokens": self.tokens,
                "offers": self.n_offer, "rate_limited": self.n_ratelimited, "queue_full": self.n_full,
                "median_latency_s": round(lat[len(lat) // 2], 3) if lat else None,
                "p90_latency_s": round(lat[int(len(lat) * 0.9)], 3) if lat else None,
                **({"warmup_s": self.backend.warmup_s} if hasattr(self.backend, "warmup_s") else {}),
                **({"climb_by_threshold": self.backend.n_climb_threshold}
                   if hasattr(self.backend, "n_climb_threshold") else {})}
