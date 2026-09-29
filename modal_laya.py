"""Run the course with Laya Vision as the tactical layer, on Modal GPUs.

Laya runs in-process, so it needs a GPU to answer in time for a live control loop
(about 40 ms per call on an L4; seconds on a CPU). MuJoCo renders through OSMesa
(software), so nothing depends on the GPU driver's EGL.

The laya package comes from a local checkout of laya-vision, LAYA_DIR (default
../laya-vision), so an uncommitted change there is what runs.

    modal run modal_laya.py::scenes                         # the 7 hand-built scenes, text only
    modal run modal_laya.py::baseline --seeds 0,1,2          # every flight configuration below, in parallel
    modal run modal_laya.py::baseline --configs laya-image --seeds 1 --seconds 65
    modal run modal_laya.py::baseline --configs code-pursuit-oracle,laya-steer-strips,laya-steer-frame \
        --courses mixed --seconds 90 --model /ckpt/smolvlm/<run>/best     # Laya steers the pursuit

Results land in results/laya/<timestamp>/ as JSON lines, one per episode.
"""
import json, os, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
LAYA_DIR = os.environ.get("LAYA_DIR", os.path.join(HERE, "..", "laya-vision"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libosmesa6", "libgl1", "libegl1", "libglvnd0")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors",
                 "huggingface_hub", "num2words", "numpy", "pillow", "mujoco>=3.13", "imageio", "imageio-ffmpeg")
    .run_commands(
        "git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git"
        " /opt/mujoco_menagerie && cd /opt/mujoco_menagerie && git sparse-checkout set skydio_x2")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "HF_HOME": "/cache/hf",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONFAULTHANDLER": "1"})
    .add_local_dir(HERE, "/root/jev", ignore=["site", ".git", ".venv", "mujoco_menagerie", "assets", "results",
                                             "*.mp4", "*.npy", "__pycache__"])
    .add_local_dir(os.path.join(LAYA_DIR, "laya"), "/root/jev/laya", ignore=["__pycache__"])
)
app = modal.App("jev-drone-laya", image=image)
hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")   # fine-tuned runs: pass --model /ckpt/smolvlm/<run>/best
ROVER_SET = "drone_rover"
# at most this many GPU containers per flight function at once: a 72-flight sweep otherwise asks
# for one GPU per flight (~40 at once); capped, it queues and takes a few times longer
GPU_MAX = int(os.environ.get("JEV_GPU_MAX", "10"))          # /data/vqa/drone_rover on laya-datasets, for laya-vision's finetune_long

# drone-rover-v3.2a pursuit read-out (readout.py on its rover-test preds-v2.jsonl)
_V32_PURSUIT = {"pursuit": "laya-pursuit", "pursuit_questions": "v2", "pursuit_sharpen": 1.5,
                "pursuit_range": {"range_sharpen": 1.5}}
_V32_VOTE = {"climb_p": 0.12, "climb_votes": 3}

# name -> (use the model?, backend, laya sees the camera frame, lockstep[, pursuit kwargs for run.episode])
CONFIGS = {
    "no-model": (False, "laya", False, False),   # the ablation: greedy heuristic, never consults a model
    "laya-text": (True, "laya", False, False),   # the same JSON Jev gets
    "laya-image": (True, "laya", True, False),   # JSON + the onboard camera frame
    "laya-text-lockstep": (True, "laya", False, True),   # sim waits for every answer: judgment without latency
    # controls: the same answer every call, with laya's typical risk/lost values (~0.93 / ~0.45)
    "always-climb": (True, "const:climb", False, False),
    "always-hold": (True, "const:hold_course", False, False),
    "always-gap-left": (True, "const:gap_left", False, False),
    "always-gap-right": (True, "const:gap_right", False, False),
    "oracle": (True, "const:oracle", False, False),        # the right answer from the layout: is it flyable?
    # Laya steers the pursuit from the RGB frame (laya_pursuit.py); tactics are the oracle, so only the
    # pursuit heading differs from code-pursuit-oracle. --model is the pursuit checkpoint. Courses from
    # courses.py only (the oracle needs a layout): --courses mixed,pockets,no-climb
    "code-pursuit-oracle": (True, "const:oracle", False, False, {"pursuit": "code"}),   # the baseline
    "laya-steer-strips": (True, "const:oracle", False, False, {"pursuit": "laya-strips"}),
    "laya-steer-frame": (True, "const:oracle", False, False, {"pursuit": "laya-frame"}),
    # Laya sets heading AND forward speed (its speed answer as a range), from one predict per frame
    "laya-pursuit": (True, "const:oracle", False, False, {"pursuit": "laya-pursuit"}),
    "laya-steer-strips-lockstep": (True, "const:oracle", False, False, {"pursuit": "laya-strips", "pursuit_lockstep": True}),
    # model-free stand-ins (CPU): the true bearing, then with a strips-like 0.2 s latency and ~6 deg noise
    "sim-steer": (True, "const:oracle", False, False, {"pursuit": "sim"}),
    "sim-steer-noisy": (True, "const:oracle", False, False,
                        {"pursuit": "sim", "pursuit_noise_deg": 6.0, "pursuit_delay_s": 0.2}),
    # the lost-target search carries the last fix forward up to search_lead_s (default 5 s); the
    # pockets hide the rover ~10 s, so let it extrapolate 12 s, or without a practical bound (60 s).
    # -hold: search_on_hold, so the oracle's hold_course no longer pre-empts that search (without it
    # the search only runs once the judgment goes stale). e.g. code-pursuit-hold, laya-steer-frame-hold-lead12
    **{"%s%s%s" % (name, "-hold" if hold else "", "-lead%d" % lead if lead != 5 else ""):
       (True, "const:oracle", False, False,
        dict(pk, search_lead_s=float(lead), **({"search_on_hold": True} if hold else {})))
       for name, pk in (("code-pursuit", {"pursuit": "code"}), ("laya-steer-frame", {"pursuit": "laya-frame"}),
                        ("sim-steer-noisy", {"pursuit": "sim", "pursuit_noise_deg": 6.0, "pursuit_delay_s": 0.2}))
       for hold, leads in ((False, (12, 60)), (True, (5, 12, 60))) for lead in leads},
    # Laya's range sets forward speed through laya_pursuit.RangeSpeed (filtered, own-motion predicted,
    # gentler gain, asymmetric limits: run.episode speed_law="auto"), as laya-pursuit above now does too.
    # -oldlaw: the code's law straight on Laya's range, as laya-pursuit flew before (2/12 finished).
    # -v2: the drone-rover-v2 questions (probe.questions_v2: steer7 over +-60 deg, range8 over 2-8.5 m),
    # read with laya_pursuit.QUESTIONS["v2"] (steer sharpen 2, gain 1; range sharpen 2, fitted on the
    # drone-rover-v2 probe); pass the v2 checkpoint with --model.
    "laya-pursuit-oldlaw": (True, "const:oracle", False, False, {"pursuit": "laya-pursuit", "speed_law": "code"}),
    "laya-pursuit-v2": (True, "const:oracle", False, False, {"pursuit": "laya-pursuit", "pursuit_questions": "v2"}),
    "laya-steer-frame-v2": (True, "const:oracle", False, False, {"pursuit": "laya-frame", "pursuit_questions": "v2"}),
    # model-free stand-ins for laya-pursuit (CPU): the sim locator supplies bearing AND range, with about the
    # fine-tuned model's steering (4 deg, 0.1 s) and v2's range error (0.45 m noise, -0.1 m bias; probe of
    # drone-rover-v2); -noisy: 1 m range noise (v1 flew with 1.0-1.7 m error)
    **{"sim-pursuit%s%s" % (noisy, suffix): (True, "const:oracle", False, False,
                                             dict({"pursuit": "sim-pursuit", "pursuit_noise_deg": 4.0,
                                                   "pursuit_delay_s": 0.1, "pursuit_range": rk}, **extra))
       for noisy, rk in (("", {"range_noise_m": 0.45, "range_offset_m": -0.1}), ("-noisy", {"range_noise_m": 1.0}))
       for suffix, extra in (("", {}), ("-oldlaw", {"speed_law": "code"}))},
    # drone-rover-v3 checkpoints (pass one with --model, e.g. /ckpt/smolvlm/drone-rover-v3.1/best; one loaded
    # copy serves pursuit, tactics and reacquisition): tactics = tactics.LayaV3Backend, the checkpoint's
    # `maneuver` from the frame + v3 context, climb whenever P(climb) >= tactics.V3_CLIMB_P (0.12; argmax
    # almost never says climb), risk and target_truly_lost fixed at the oracle control's values; reacquire =
    # while the pursuit source has lost the rover > 1 s, ask the checkpoint where it will reappear (<= 3 Hz)
    # and turn that way unless it says occluded (run.REACQ_DEFAULTS, laya_pursuit.Reacquirer). Pursuit asks
    # the v2 perception questions, which v3 keeps. -argmax: climb only when it is the argmax.
    **{name: (True, backend, backend == "laya-v3", False, dict(pk, **({"reacquire": "laya"} if rq else {})))
       for name, backend, pk, rq in (
           ("laya-full-v3", "laya-v3", {"pursuit": "laya-pursuit", "pursuit_questions": "v2"}, True),
           ("laya-pursuit-v3-reacq", "const:oracle", {"pursuit": "laya-pursuit", "pursuit_questions": "v2"}, True),
           ("laya-pursuit-v3", "const:oracle", {"pursuit": "laya-pursuit", "pursuit_questions": "v2"}, False),
           # drone-rover-v3.1's score temperature is 0.74 (v2's: 2.10), so its level probabilities are
           # sharp: the v2 read-out (power 2) snapped 60% of steer estimates onto a level centre and the
           # heading moved in 15-deg jumps (in-flight bearing error 9.7 deg vs 5.6 for v2). Softer powers
           # (steer 0.6, range 0.75) were best on held-out frames: 5.2-5.5 deg, 0.43 m, ~25% snapped.
           ("laya-full-v3-soft", "laya-v3", {"pursuit": "laya-pursuit", "pursuit_questions": "v2",
                                             "pursuit_sharpen": 0.6, "pursuit_range": {"range_sharpen": 0.75}}, True),
           ("laya-pursuit-v3-reacq-soft", "const:oracle", {"pursuit": "laya-pursuit", "pursuit_questions": "v2",
                                             "pursuit_sharpen": 0.6, "pursuit_range": {"range_sharpen": 0.75}}, True),
           ("laya-pursuit-v3-soft", "const:oracle", {"pursuit": "laya-pursuit", "pursuit_questions": "v2",
                                             "pursuit_sharpen": 0.6, "pursuit_range": {"range_sharpen": 0.75}}, False),
           ("laya-tactics-v3", "laya-v3", {"pursuit": "code"}, False),
           ("laya-tactics-v3-argmax", "laya-v3", {"pursuit": "code", "tactics_kw": {"climb_p": None}}, False))},
    # CPU controls: the reacquisition logic on the simulator's own v3 label (laya_pursuit.SimReappear), with
    # code pursuit and with the sim stand-in for laya-pursuit; compare with code-pursuit-oracle / sim-pursuit
    # hybrid: drone-rover-v2 flies the pursuit (v3.1 steers worse in flight: 13/24 vs 21/24 on seeds 0-11),
    # the --model checkpoint (v3.1) answers the reacquisition questions; oracle tactics (v3.1's climb answer
    # cannot separate beams from pocket walls: no P(climb) threshold gives both recall and precision)
    "hybrid-v2pursuit-reacq": (True, "const:oracle", False, False,
                               {"pursuit": "laya-pursuit", "pursuit_questions": "v2", "reacquire": "laya",
                                "pursuit_model": "/ckpt/smolvlm/drone-rover-v2/last"}),
    # drone-rover-v3.2a (pass --model /ckpt/smolvlm/drone-rover-v3.2a/best), one checkpoint for everything:
    # read-out fitted by readout.py (steer and range power 1.5), climb when P(climb) >= 0.1543, the
    # rover-test-tac probe's best threshold (beam recall 78%, false climbs at pocket walls 4.8% of frames;
    # v3.1 managed 29% precision at its best threshold). -oracle: oracle tactics; tactics-: code pursuit.
    **{name: (True, backend, backend == "laya-v3", False, pk)
       for name, backend, pk in (
           ("laya-full-v3.2", "laya-v3", dict(_V32_PURSUIT, reacquire="laya", tactics_kw={"climb_p": 0.1543})),
           ("laya-pursuit-v3.2-reacq", "const:oracle", dict(_V32_PURSUIT, reacquire="laya")),
           ("laya-tactics-v3.2", "laya-v3", {"pursuit": "code", "tactics_kw": {"climb_p": 0.1543}}),
           # -vote: climb only after 3 calls in a row at P(climb) >= 0.12 (tactics.LayaV3Backend climb_votes);
           # one call over 0.1543 flew the aircraft into the first pocket in half the mixed flights
           ("laya-full-v3.2-vote", "laya-v3", dict(_V32_PURSUIT, reacquire="laya", tactics_kw=_V32_VOTE)),
           ("laya-tactics-v3.2-vote", "laya-v3", {"pursuit": "code", "tactics_kw": _V32_VOTE}))},
    # v3.2 hybrids: drone-rover-v2/last flies the pursuit (v3.2a's own pursuit + reacquisition finished 30/48:
    # most failures lost the rover at the first pocket and never found it), --model (v3.2a) answers
    # reacquisition, and tactics (climb_votes) in -vote / oracle tactics in -oracle
    **{"hybrid-v2pursuit-v3.2" + suffix: (True, backend, backend == "laya-v3", False,
                                          dict({"pursuit": "laya-pursuit", "pursuit_questions": "v2", "reacquire": "laya",
                                                "pursuit_model": "/ckpt/smolvlm/drone-rover-v2/last"}, **extra))
       for suffix, backend, extra in (("-oracle", "const:oracle", {}), ("-vote", "laya-v3", {"tactics_kw": _V32_VOTE}))},
    # altitude as a continuous operator (altitude.py): the course's own ascend / descend answers, and the
    # same rate of false "+1 m" answers against false one-shot climbs (tactics.ConstBackend wrong_p)
    "code-pursuit-alt": (True, "const:oracle", False, False, {"pursuit": "code", "altitude": "sim"}),
    **{"code-pursuit-alt-wrong%02d" % int(100 * p): (True, "const:oracle", False, False,
                                                     {"pursuit": "code", "altitude": "sim", "altitude_wrong_p": p})
       for p in (0.05, 0.1, 0.2)},
    **{"code-pursuit-oracle-wrong%02d" % int(100 * p): (True, "const:oracle", False, False,
                                                        {"pursuit": "code", "tactics_kw": {"wrong_p": p}})
       for p in (0.05, 0.1, 0.2)},
    # the altitude operator answered by a checkpoint trained on drone_rover_alt (--model): code pursuit to
    # isolate it; and all Laya: v2 pursuit, --model reacquires and flies altitude. Tactics hold_course (the
    # oracle's only other answer is climb, which the altitude operator replaces)
    # v3.3 alone, every decision: its own pursuit (v3.2a's fitted read-out), reacquisition and altitude
    "laya-full-v3.3": (True, "const:hold_course", False, False, dict(_V32_PURSUIT, reacquire="laya", altitude="laya")),
    # the same with yaw-first desaturation in the mixer (flight.Pilot.yaw_desat, off by default since
    # results/laya/20260929-015224: no balloons, but 24/48 against 39/48 without it)
    "hybrid-v2pursuit-alt-desat": (True, "const:hold_course", False, False,
                                   {"pursuit": "laya-pursuit", "pursuit_questions": "v2", "reacquire": "laya",
                                    "altitude": "laya", "pursuit_model": "/ckpt/smolvlm/drone-rover-v2/last",
                                    "yaw_desat": True}),
    "laya-alt": (True, "const:hold_course", False, False, {"pursuit": "code", "altitude": "laya"}),
    "hybrid-v2pursuit-alt": (True, "const:hold_course", False, False,
                             {"pursuit": "laya-pursuit", "pursuit_questions": "v2", "reacquire": "laya",
                              "altitude": "laya", "pursuit_model": "/ckpt/smolvlm/drone-rover-v2/last"}),
    "code-pursuit-simreacq": (True, "const:oracle", False, False, {"pursuit": "code", "reacquire": "sim"}),
    "sim-pursuit-simreacq": (True, "const:oracle", False, False,
                             {"pursuit": "sim-pursuit", "pursuit_noise_deg": 4.0, "pursuit_delay_s": 0.1,
                              "pursuit_range": {"range_noise_m": 0.45, "range_offset_m": -0.1}, "reacquire": "sim"}),
}
# latency-faithful timing (run.episode timing="virtual", laya_pursuit.GpuClock): the same flights, with every Laya
# answer landing at its measured GPU latency in sim time, queued on one GPU, and the world never slowed for it
for _n in ("hybrid-v2pursuit-alt", "hybrid-v2pursuit-alt-desat", "laya-full-v3.3", "laya-alt"):
    CONFIGS[_n + "-rt"] = CONFIGS[_n][:4] + (dict(CONFIGS[_n][4], timing="virtual"),)
# true wall-clock real time (run.episode timing="wallclock", laya_server): the -rt configs with Laya in its own
# process and the sim paced to the wall clock
for _n in ("laya-full-v3.3-rt", "laya-alt-rt"):
    CONFIGS[_n.replace("-rt", "-wc")] = CONFIGS[_n][:4] + (dict(CONFIGS[_n][4], timing="wallclock"),)
# heading smoothing against the airmode balloons (run.Guidance tune): a low-pass on the pursuit heading, a cap on
# the yaw command's rate, and both
for _s, _t in (("tau", {"yaw_tau_s": 0.3}), ("rate", {"yaw_rate_dps": 120.0}),
               ("smooth", {"yaw_tau_s": 0.3, "yaw_rate_dps": 120.0})):
    CONFIGS["laya-full-v3.3-rt-" + _s] = CONFIGS["laya-full-v3.3-rt"][:4] + (
        dict(CONFIGS["laya-full-v3.3-rt"][4], guide_tune=_t),)
CONFIGS["laya-full-v3.3-wc-rate"] = CONFIGS["laya-full-v3.3-rt-rate"][:4] + (
    dict(CONFIGS["laya-full-v3.3-rt-rate"][4], timing="wallclock"),)


def _config(name):
    """(use_model, backend, img, lockstep, pursuit kwargs); the older 4-tuples fly code pursuit."""
    c = CONFIGS[name]
    return tuple(c[:4]) + (dict(c[4]) if len(c) > 4 else {},)


def _needs_gpu(name):
    use_model, backend, _, _, pk = _config(name)
    return ((use_model and not backend.startswith("const:")) or pk.get("pursuit", "code").startswith("laya")
            or pk.get("reacquire") == "laya" or pk.get("altitude") == "laya")


def _enter():
    os.chdir("/root/jev")
    if not os.path.exists("mujoco_menagerie"):
        os.symlink("/opt/mujoco_menagerie", "mujoco_menagerie")
        os.symlink("mujoco_menagerie/skydio_x2/assets", "assets")
    import sys
    sys.path.insert(0, "/root/jev")
    try:  # triton's bundled LLVM must load before OSMesa's, or the process segfaults
        import torch._dynamo  # noqa: F401
    except ImportError:
        pass


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=30 * 60, volumes={"/cache/hf": hf_vol})
def scenes_remote(model: str = "", variants=(("full budgets, no image", []),)):
    _enter()
    import io, contextlib, scenes
    text, rows = "", []
    for name, flags in variants:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            scenes.main(["--backend", "laya", "--out", "/tmp/scenes.json"] + flags + (["--model", model] if model else []))
        text += "=== %s\n%s\n" % (name, buf.getvalue())
        rows.append(dict(variant=name, **json.load(open("/tmp/scenes.json"))))
    return text, rows


@app.function(gpu=["L4", "A10G"], cpu=8, memory=16384, timeout=60 * 60, max_containers=GPU_MAX,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def fly(config: str, seed: int, seconds: float, model: str = "", course: str = "classic", budget: int = 0):
    if _config(config)[4].get("timing") == "wallclock":     # render on the GPU: the CPU is the sim's
        os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    _enter()
    import run
    use_model, backend, img, lockstep, pk = _config(config)
    t0 = time.time()
    pk = dict(pk)
    # a config may name its own pursuit checkpoint ("pursuit_model"); --model then serves tactics and
    # reacquisition (hybrid: v2 flies the pursuit, v3.1 answers where a lost rover will reappear)
    pm = pk.pop("pursuit_model", None)
    extra = {"reacquire_model": model or None} if pm and pk.get("reacquire") == "laya" else {}
    if pm and pk.get("altitude") == "laya":
        extra["altitude_model"] = model or None
    r = run.episode(seed, seconds, use_jev=use_model, backend=backend, laya_model=model or None,
                    laya_image=img, lockstep=lockstep, course=course, budget=budget or None,
                    pursuit_model=pm or model or None, **extra, **pk)
    r.update(config=config, wall_s=round(time.time() - t0, 1))
    try:
        import torch
        r["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return r


@app.function(cpu=4, memory=8192, timeout=60 * 60)
def fly_cpu(config: str, seed: int, seconds: float, model: str = "", course: str = "classic", budget: int = 0):
    """The controls (no-model, const:*, sim pursuit) never load a model, so they need no GPU."""
    return fly.local(config, seed, seconds, model, course, budget)


@app.function(cpu=2, memory=4096, timeout=10 * 60)
def render_course(course: str, seed: int):
    """Top-down view of a course, with the rover's path, as PNG bytes (courses.render)."""
    _enter()
    import courses
    return courses.render(course, seed, "/root/jev")


@app.function(gpu=["L4", "A10G"], cpu=8, memory=16384, timeout=60 * 60, max_containers=GPU_MAX,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def fly_gif(config: str, seed: int, seconds: float, course: str, budget: int = 0, model: str = "",
            realtime: bool = False):
    """fly(), recording the flight, then render it as a GIF (flightgif.py) after the flight ends.
    `realtime`: a 1x GIF (flightgif realtime), half size. Returns (episode result, GIF bytes, the snapshots
    without images as JSON text)."""
    if _config(config)[4].get("timing") == "wallclock":     # as fly(): render on the GPU
        os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    _enter()
    import run, flightgif
    import numpy as np
    use_model, backend, img, lockstep, pk = _config(config)
    rec = []
    pk = dict(pk)
    pm = pk.pop("pursuit_model", None)            # as fly(): --model then serves reacquire and altitude
    extra = {"reacquire_model": model or None} if pm and pk.get("reacquire") == "laya" else {}
    if pm and pk.get("altitude") == "laya":
        extra["altitude_model"] = model or None
    r = run.episode(seed, seconds, use_jev=use_model, backend=backend, laya_model=model or None,
                    laya_image=img, lockstep=lockstep, course=course, budget=budget or None, record=rec,
                    pursuit_model=pm or model or None, record_every=50 if realtime else 100, **extra, **pk)
    r.update(config=config)
    outcome = "finished" if r["finished_at_s"] is not None else "stopped at x=%.0f m" % r["max_x_m"]
    title = "%s  |  %s seed %d  |  %s" % (config, course, seed, outcome)
    if pk.get("timing") == "wallclock":
        title = "%s seed %d  |  %s  |  wall clock rt %.2f  |  1x" % (course, seed, outcome, r["realtime_factor"])
    # the snapshots minus the heavy parts (images, full state), as JSON text for loss analysis
    rid = None
    track = []
    for s in rec:
        if rid is None:
            import mujoco, courses
            cm = courses.make(course, seed)
            rid = mujoco.MjModel.from_xml_path(cm.write("/root/jev")).body("rover").mocapid[0]
        track.append({"t": round(s["t"], 2), "pos": [round(float(v), 2) for v in s["qpos"][:3]],
                      "yaw_deg": round(float(np.rad2deg(s["yaw"])), 1),
                      "rover": [round(float(v), 2) for v in s["mocap_pos"][rid][:2]],
                      **{k: s.get(k) for k in ("target_visible", "loc", "true_bearing_deg", "code_bearing_deg",
                                               "code_range_m", "guide", "reflex", "climbing", "hits",
                                               "alt", "reappear")},
                      "maneuver": s["judg"].get("maneuver")})
    label = "tactics (oracle)" if backend == "const:oracle" else "tactics (%s)" % (
        backend if use_model else "none")
    if backend == "const:hold_course" and pk.get("altitude"):
        label = None                 # no tactical layer: altitude.py replaces the only other answer (climb)
    short = lambda p: (p or "").replace("/ckpt/smolvlm/drone-rover-", "Laya ").replace("/best", "").replace("/last", "")  # noqa: E731
    who = {"pursuit": short(pm or model) if pk.get("pursuit", "code") != "code" else "code",
           "reacquire": short(model) if pk.get("reacquire") == "laya" else None,
           "altitude": short(model) if pk.get("altitude") == "laya" else pk.get("altitude")}
    gkw = {}
    if realtime:                 # a 1x MP4 at 10 frames per flight second (snapshots every 0.1 s)
        gkw = dict(realtime=True, fmt="mp4", snap_dt=0.1, scale=0.75 if course.startswith("city") else 1.0)
    return (r, flightgif.make_gif(rec, course, seed, title, "/root/jev", tactics_label=label, who=who, **gkw),
            json.dumps(track, default=float))


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=60 * 60,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def probe_remote(frames: dict, rows: list, model: str = "", n_permutations: int = 1, mode: str = "full",
                 n_strips: int = 5):
    """probe.py: ask Laya where the rover is in each recorded frame, from the whole frame ("full") or one
    yes/no per vertical strip ("strips")."""
    _enter()
    import laya, probe
    agent = laya.load_vlm(model or "thaitea/laya-vision", option_max_len=256, head_max_len=1024, max_len=3072)
    if mode == "strips":
        preds = probe.evaluate_strips(agent, frames, rows, n_strips)
        summary = probe.score_strips(preds, n_strips)
    elif mode == "v2":
        preds = probe.evaluate_v2(agent, frames, rows)
        summary = {"sharpen%g" % k: probe.score_v2(preds, k) for k in (1, 2, 3)}
    elif mode == "alt":
        import rover_data
        preds = rover_data.evaluate_alt(agent, frames, rows)
        summary = {"sharpen%g" % k: rover_data.score_alt(preds, k) for k in (0.5, 1.0, 1.5, 2.0)}
    elif mode == "v3":
        import rover_data
        preds = rover_data.evaluate_v3(agent, frames, rows)
        summary = rover_data.score_v3(preds)
    else:
        preds = probe.evaluate(agent, frames, rows, n_permutations)
        summary = probe.score(preds)
    # as JSON text: numpy scalars would not unpickle where the Modal client has no numpy
    return json.dumps(preds, default=float), json.dumps(summary, default=float)


@app.function(cpu=2, memory=4096, timeout=30 * 60, volumes={"/data": data_vol})
def collect_job(course: str, seed: int, split: str, seconds: float = 60.0):
    """One oracle flight -> its frames' images on the datasets volume, and their records (rover_data.py)."""
    _enter()
    import rover_data
    frames = rover_data.collect_flight(course, seed, seconds)
    recs = rover_data.records(frames, "/data/vqa/%s/images" % ROVER_SET)
    data_vol.commit()
    return json.dumps([dict(r, split=split) for r in recs])


@app.function(cpu=1, memory=4096, timeout=10 * 60, volumes={"/data": data_vol})
def finalize_rover_set(recs_json: list, meta: dict):
    import collections
    base = "/data/vqa/%s" % ROVER_SET
    data_vol.reload()
    by = collections.defaultdict(list)
    for chunk in recs_json:
        for r in json.loads(chunk):
            by[r.pop("split")].append(r)
    for split, rs in by.items():
        with open(os.path.join(base, split + ".jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    counts = {k: len(v) for k, v in by.items()}
    json.dump(dict(meta, counts=counts), open(os.path.join(base, "meta.json"), "w"), indent=1)
    open(os.path.join(base, "_READY"), "w").close()      # last: the loaders skip a set without it
    data_vol.commit()
    return counts


@app.function(cpu=2, memory=8192, timeout=20 * 60, volumes={"/data": data_vol})
def build_rover_set_v2_remote():
    """/data/vqa/drone_rover_v2 (create-only) from drone_rover's records: steer7 and range8 with soft
    targets (rover_data.v2_records); images stay in drone_rover/ and are referenced relatively."""
    _enter()
    import rover_data
    base, v1 = "/data/vqa/%s_v2" % ROVER_SET, "/data/vqa/%s" % ROVER_SET
    data_vol.reload()
    if os.path.exists(os.path.join(base, "_READY")):
        raise SystemExit("%s already exists; refusing to overwrite" % base)
    os.makedirs(base, exist_ok=True)
    counts = {}
    for split in ("train", "val"):
        recs = [json.loads(l) for l in open(os.path.join(v1, split + ".jsonl"))]
        out = rover_data.v2_records(recs)
        with open(os.path.join(base, split + ".jsonl"), "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        counts[split] = len(out)
    meta = json.load(open(os.path.join(v1, "meta.json")))
    json.dump(dict(meta, source="rover_data.v2_records(drone_rover)", counts=counts),
              open(os.path.join(base, "meta.json"), "w"), indent=1)
    open(os.path.join(base, "_READY"), "w").close()
    data_vol.commit()
    return counts


# --- drone_rover_v3: reacquisition (occluded / reappear / reappear_eta) and tactics (maneuver) --------------
ROVER_SET_V3 = ROVER_SET + "_v3"


@app.function(cpu=2, memory=4096, timeout=40 * 60, volumes={"/data": data_vol})
def collect_v3_job(course: str, seed: int, split: str, seconds: float = 0.0):
    """One flight -> its v3 frames' images on the datasets volume, and their records (rover_data.records_v3)."""
    _enter()
    import rover_data
    frames = rover_data.collect_flight_v3(course, seed, seconds or None)
    recs = rover_data.records_v3(frames, "/data/vqa/%s/images" % ROVER_SET_V3)
    data_vol.commit()
    return json.dumps([dict(r, split=split) for r in recs], default=float)


@app.function(cpu=1, memory=4096, timeout=10 * 60, volumes={"/data": data_vol})
def finalize_rover_set_v3(recs_json: list, meta: dict):
    """Balance maneuver per split (rover_data.balance_maneuver), write <split>.jsonl, meta.json, _READY last."""
    _enter()
    import collections, rover_data
    base = "/data/vqa/%s" % ROVER_SET_V3
    data_vol.reload()
    if os.path.exists(os.path.join(base, "_READY")):
        raise SystemExit("%s already exists; refusing to overwrite" % base)
    by = collections.defaultdict(list)
    for chunk in recs_json:
        for r in json.loads(chunk):
            by[r.pop("split")].append(r)
    counts, labels = {}, {}
    for split, rs in sorted(by.items()):
        rs = rover_data.balance_maneuver(rs, seed=0 if split == "train" else 1)
        with open(os.path.join(base, split + ".jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        counts[split] = len(rs)
        c = collections.Counter("%s=%s" % (r["id"].rsplit("-", 1)[1], r["label"]) for r in rs)
        labels[split] = dict(sorted(c.items()))
    json.dump(dict(meta, counts=counts, labels=labels), open(os.path.join(base, "meta.json"), "w"), indent=1)
    open(os.path.join(base, "_READY"), "w").close()      # last: the loaders skip a set without it
    data_vol.commit()
    return counts, labels


@app.function(cpu=1, timeout=120, volumes={"/data": data_vol})
def rover_set_v3_exists():
    return os.path.exists("/data/vqa/%s/_READY" % ROVER_SET_V3)


@app.function(cpu=2, memory=4096, timeout=40 * 60)
def test_frames_v3_job(course: str, seed: int, seconds: float = 0.0):
    """Held-out v3 frames in probe.py's format: every snapshot with all its v3 truth (rover_data.v3_frame_labels)."""
    _enter()
    import rover_data
    fr = rover_data.collect_flight_v3(course, seed, seconds or None)
    rows, blobs = [], {}
    for k, f in enumerate(fr):
        name = "v3-%s-%d-%04d.jpg" % (course, seed, k)
        blobs[name] = f["jpeg"]
        rows.append(dict(frame=name, **rover_data.v3_frame_labels(f)))
    return json.dumps(rows, default=float), blobs


# --- drone_rover_v3b: more "rover lost / behind" examples in v3's format (rover_data.collect_flight_v3b) --------
ROVER_SET_V3B = ROVER_SET + "_v3b"


@app.function(cpu=2, memory=4096, timeout=50 * 60, volumes={"/data": data_vol})
def collect_v3b_job(course: str, seed: int, kind: str, split: str):
    """One flight -> its v3b frames' images on the datasets volume, and their records (rover_data.records_v3)."""
    _enter()
    import rover_data
    frames = rover_data.collect_flight_v3b(course, seed, kind)
    recs = rover_data.records_v3(frames, "/data/vqa/%s/images" % ROVER_SET_V3B)
    data_vol.commit()
    return json.dumps([dict(r, split=split) for r in recs], default=float)


@app.function(cpu=1, memory=4096, timeout=10 * 60, volumes={"/data": data_vol})
def finalize_rover_set_v3b(recs_json: list, meta: dict):
    """Balance maneuver (rover_data.balance_maneuver) and cap reappear=ahead (balance_reappear) per split,
    write <split>.jsonl, meta.json, _READY last."""
    _enter()
    import collections, rover_data
    base = "/data/vqa/%s" % ROVER_SET_V3B
    data_vol.reload()
    if os.path.exists(os.path.join(base, "_READY")):
        raise SystemExit("%s already exists; refusing to overwrite" % base)
    by = collections.defaultdict(list)
    for chunk in recs_json:
        for r in json.loads(chunk):
            by[r.pop("split")].append(r)
    counts, labels, named = {}, {}, {}
    for split, rs in sorted(by.items()):
        rs = rover_data.balance_maneuver(rs, seed=0 if split == "train" else 1)
        rs = rover_data.balance_reappear(rs, seed=0 if split == "train" else 1)
        with open(os.path.join(base, split + ".jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        counts[split] = len(rs)
        c = collections.Counter("%s=%s" % (r["id"].rsplit("-", 1)[1], r["label"]) for r in rs)
        labels[split] = dict(sorted(c.items()))
        named[split] = rover_data.label_counts(rs)
    json.dump(dict(meta, counts=counts, labels=labels, label_names=named), open(os.path.join(base, "meta.json"), "w"),
              indent=1)
    open(os.path.join(base, "_READY"), "w").close()      # last: the loaders skip a set without it
    data_vol.commit()
    return counts, named


@app.function(cpu=1, timeout=120, volumes={"/data": data_vol})
def rover_set_v3b_exists():
    return os.path.exists("/data/vqa/%s/_READY" % ROVER_SET_V3B)


@app.function(cpu=2, memory=4096, timeout=50 * 60)
def test_frames_v3b_job(course: str, seed: int, kind: str):
    """Held-out v3b frames in probe.py's format (rover_data.v3b_test_row), with their JPEG bytes."""
    _enter()
    import rover_data
    rows, blobs = [], {}
    for f in rover_data.collect_flight_v3b(course, seed, kind):
        name = f["stem"] + ".jpg"
        blobs[name] = f["jpeg"]
        rows.append(rover_data.v3b_test_row(f, name))
    return json.dumps(rows, default=float), blobs


# --- v3.2: drone_rover_tac (maneuver: beams vs pocket walls) and drone_rover_town (town perception + reacquisition)
ROVER_SET_TAC = ROVER_SET + "_tac"
ROVER_SET_TOWN = ROVER_SET + "_town"
ROVER_SET_ALT = ROVER_SET + "_alt"       # altitude.py's ascend / descend operator
ROVER_SET_ONP = ROVER_SET + "_onpolicy"  # frames from the checkpoint's own real-time flights (DAgger)


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=30 * 60,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def sim_speed(gl: str = "osmesa", course: str = "mixed", seconds: float = 30.0, seed: int = 0):
    """How fast the sim alone runs (code pursuit, oracle tactics, the camera rendered as a Laya flight
    renders it) with MuJoCo on `gl` ("osmesa": software on the CPU; "egl": the GPU): sim s per wall s."""
    os.environ["MUJOCO_GL"] = gl
    os.environ["PYOPENGL_PLATFORM"] = gl
    _enter()
    import run
    t0 = time.time()
    r = run.episode(seed, seconds, use_jev=True, backend="const:oracle", course=course, realtime=False,
                    record_rgb=True)
    wall = time.time() - t0
    return json.dumps({"gl": gl, "sim_s": seconds, "wall_s": round(wall, 2), "speed_x": round(seconds / wall, 2),
                       "finished": r["finished_at_s"], "vis": r["target_visible_pct"]})


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=30 * 60,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def sim_profile(gl: str = "egl", seconds: float = 15.0):
    """cProfile of the sim alone (as sim_speed): the top functions by own time and by cumulative time."""
    os.environ["MUJOCO_GL"] = gl
    os.environ["PYOPENGL_PLATFORM"] = gl
    _enter()
    import cProfile, pstats, io, run
    pr = cProfile.Profile()
    pr.enable()
    run.episode(0, seconds, use_jev=True, backend="const:oracle", course="mixed", realtime=False, record_rgb=True)
    pr.disable()
    out = io.StringIO()
    st = pstats.Stats(pr, stream=out)
    st.sort_stats("tottime").print_stats(22)
    st.sort_stats("cumulative").print_stats(28)
    return out.getvalue()


@app.local_entrypoint()
def sim_profile_check(gl: str = "egl", seconds: float = 15.0):
    print(sim_profile.remote(gl, seconds))


@app.local_entrypoint()
def sim_speed_check(gls: str = "osmesa,egl", seconds: float = 30.0):
    for gl in gls.split(","):
        try:
            print(sim_speed.remote(gl, "mixed", seconds))
            print(sim_speed.remote(gl, "mixed", 3 * seconds))
        except Exception as e:
            print(gl, "FAILED", repr(e)[:400])


@app.function(cpu=1, timeout=120, volumes={"/data": data_vol})
def rover_set_ready(name: str):
    return os.path.exists("/data/vqa/%s/_READY" % name)


@app.function(cpu=2, memory=4096, timeout=50 * 60, volumes={"/data": data_vol})
def collect_tac_job(course: str, seed: int, split: str):
    """One oracle flight -> its tac frames' images on the datasets volume, and their maneuver records
    (rover_data.collect_flight_tac / records_tac)."""
    _enter()
    import rover_data
    frames = rover_data.collect_flight_tac(course, seed)
    recs = rover_data.records_tac(frames, "/data/vqa/%s/images" % ROVER_SET_TAC)
    data_vol.commit()
    return json.dumps([dict(r, split=split) for r in recs], default=float)


@app.function(cpu=2, memory=8192, timeout=60 * 60, volumes={"/data": data_vol})
def collect_town_job(seed: int, kind: str, split: str, seconds: float = 0.0):
    """One town flight (code pursuit or a rover_data.TOWN_FAILURE stand-in) -> its views' images and records
    (rover_data.collect_flight_town / records_town)."""
    _enter()
    import rover_data
    frames = rover_data.collect_flight_town(seed, kind, seconds or None)
    recs = rover_data.records_town(frames, "/data/vqa/%s/images" % ROVER_SET_TOWN)
    data_vol.commit()
    return json.dumps([dict(r, split=split) for r in recs], default=float)


@app.function(cpu=2, memory=4096, timeout=50 * 60, volumes={"/data": data_vol})
def collect_alt_job(course: str, seed: int, split: str, wander_p: float = 0.3):
    """One oracle flight with wandering altitude answers -> its frames' images and `altitude` records
    (rover_data.collect_flight_alt / records_alt)."""
    _enter()
    import rover_data
    frames = rover_data.collect_flight_alt(course, seed, wander_p)
    recs = rover_data.records_alt(frames, "/data/vqa/%s/images" % ROVER_SET_ALT)
    data_vol.commit()
    return json.dumps([dict(r, split=split) for r in recs], default=float)


@app.function(cpu=2, memory=4096, timeout=50 * 60)
def test_frames_alt_job(course: str, seed: int, wander_p: float = 0.3):
    """Held-out altitude frames in probe.py's format (the records' truth fields + state_text)."""
    _enter()
    import rover_data, altitude
    rows, blobs = [], {}
    for k, f in enumerate(rover_data.collect_flight_alt(course, seed, wander_p)):
        name = "alt-%s-%d-%04d.jpg" % (course, seed, k)
        blobs[name] = f["jpeg"]
        rows.append({"frame": name, "state_text": json.dumps({q: f["context"].get(q) for q in altitude.CONTEXT_KEYS}),
                     **{q: f[q] for q in ("course", "seed", "t", "altitude_m", "target_alt_m", "dz_m",
                                          "station_kind", "pos", "visible")}})
    return json.dumps(rows, default=float), blobs


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=60 * 60, max_containers=GPU_MAX,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only(), "/data": data_vol})
def collect_onpolicy_job(course: str, seed: int, split: str, config: str, model: str):
    """One flight with `config` flying on `model` (latency-faithful timing) -> its frames' images and records
    (rover_data.collect_flight_onpolicy / records_onpolicy), plus a one-line flight summary."""
    _enter()
    import rover_data
    _, _, _, _, pk = _config(config)
    pk = {k: v for k, v in pk.items() if k not in ("pursuit_model", "timing")}
    frames, r = rover_data.collect_flight_onpolicy(course, seed, model, pk)
    recs = rover_data.records_onpolicy(frames, "/data/vqa/%s/images" % ROVER_SET_ONP)
    data_vol.commit()
    summary = {"course": course, "seed": seed, "finished_at_s": r["finished_at_s"],
               "target_visible_pct": r["target_visible_pct"], "frames": len(frames)}
    return json.dumps([dict(rr, split=split) for rr in recs] + [dict(summary, split="_flight")], default=float)


@app.function(cpu=1, memory=8192, timeout=15 * 60, volumes={"/data": data_vol})
def finalize_rover_set_v32(name: str, recs_json: list, meta: dict):
    """Balance per split (tac: rover_data.balance_tac; town: balance_reappear), write <split>.jsonl, meta.json,
    _READY last. Create-only."""
    _enter()
    import collections, rover_data
    base = "/data/vqa/%s" % name
    data_vol.reload()
    if os.path.exists(os.path.join(base, "_READY")):
        raise SystemExit("%s already exists; refusing to overwrite" % base)
    by = collections.defaultdict(list)
    flights = []
    for chunk in recs_json:
        for r in json.loads(chunk):
            sp = r.pop("split")
            (flights if sp == "_flight" else by[sp]).append(r)
    counts, report = {}, {}
    for split, rs in sorted(by.items()):
        seed = 0 if split == "train" else 1
        if name == ROVER_SET_TAC:
            pre = rover_data.tac_counts(rs)
            rs, before, after = rover_data.balance_tac(rs, seed=seed)
            report[split] = {"before_balance": pre, "groups_before": before, "groups_after": after,
                             "after_balance": rover_data.tac_counts(rs),
                             "by_course": {c: rover_data.tac_counts([r for r in rs if r["course"] == c])["groups"]
                                           for c in sorted({r["course"] for r in rs})}}
        elif name == ROVER_SET_ONP:
            rs, report[split] = rover_data.balance_onpolicy(rs, seed=seed)
        elif name == ROVER_SET_ALT:
            rs, before, after = rover_data.balance_alt(rs, seed=seed)
            report[split] = {"groups_before": before, "groups_after": after,
                             "by_course": {c: sum(r["course"] == c for r in rs) for c in sorted({r["course"] for r in rs})}}
        else:
            rs = rover_data.balance_reappear(rs, seed=seed)
            report[split] = rover_data.town_counts(rs)
        with open(os.path.join(base, split + ".jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        counts[split] = len(rs)
    json.dump(dict(meta, counts=counts, report=report, **({"flights": flights} if flights else {})),
              open(os.path.join(base, "meta.json"), "w"), indent=1)
    open(os.path.join(base, "_READY"), "w").close()      # last: the loaders skip a set without it
    data_vol.commit()
    return counts, report


@app.function(cpu=2, memory=4096, timeout=50 * 60)
def test_frames_tac_job(course: str, seed: int):
    """Held-out tac frames in probe.py's format (rover_data.tac_frame_labels: v3 truth + station_kind)."""
    _enter()
    import rover_data
    rows, blobs = [], {}
    for k, f in enumerate(rover_data.collect_flight_tac(course, seed)):
        name = "tac-%s-%d-%04d.jpg" % (course, seed, k)
        blobs[name] = f["jpeg"]
        rows.append(dict(frame=name, **rover_data.tac_frame_labels(f)))
    return json.dumps(rows, default=float), blobs


@app.function(cpu=2, memory=8192, timeout=60 * 60)
def test_frames_town_job(seed: int, kind: str):
    """Held-out town views in probe.py's format (rover_data.town_frame_labels: v2 + v3 truth, view, scenery_px)."""
    _enter()
    import rover_data
    rows, blobs = [], {}
    for f in rover_data.collect_flight_town(seed, kind):
        name = f["stem"] + ".jpg"
        blobs[name] = f["jpeg"]
        rows.append(dict(frame=name, **rover_data.town_frame_labels(f)))
    return json.dumps(rows, default=float), blobs


@app.function(cpu=1, timeout=120, volumes={"/data": data_vol})
def rover_set_exists():
    return os.path.exists("/data/vqa/%s/_READY" % ROVER_SET)


@app.function(cpu=2, memory=4096, timeout=30 * 60)
def test_frames_job(course: str, seed: int, seconds: float = 60.0):
    """Held-out frames in probe.py's format (labels + JPEG bytes), for scoring a checkpoint."""
    _enter()
    import rover_data
    fr = rover_data.collect_flight(course, seed, seconds)
    rows = []
    blobs = {}
    for k, f in enumerate(fr):
        name = "%s-%d-%04d.jpg" % (course, seed, k)
        blobs[name] = f["jpeg"]
        rows.append({"frame": name, **{x: f[x] for x in ("course", "seed", "t", "view", "yaw_offset_deg", "visible",
                                                           "pixels", "bearing_deg", "range_m")}})
    return json.dumps(rows, default=float), blobs      # JSON: numpy scalars would not unpickle locally


def _outdir():
    d = os.path.join(HERE, "results", "laya", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def _get(fc):
    try:
        return fc.get()
    except Exception as e:
        return e


@app.local_entrypoint()
def courses_png(names: str = "pockets,mixed,no-climb", seed: int = 0):
    d = os.path.join(HERE, "docs")
    for n in names.split(","):
        path = os.path.join(d, "course-%s.png" % n)
        open(path, "wb").write(render_course.remote(n, seed))
        print("wrote", path)


@app.local_entrypoint()
def gifs(config: str = "laya-image", courses: str = "pockets,mixed", seeds: str = "0,1,2,3,4,5",
         seconds: float = 90.0, budget: int = 240, model: str = "", realtime: bool = False, out_dir: str = ""):
    """Fly and record every (course, seed); write docs/gifs/<config>-<course>-seed<k>-<outcome>.gif
    (outcome: finished, lowvis = finished with the rover in view < 30%, stuck), and each flight's
    snapshots without images to results/laya/<timestamp>/track-*.json."""
    d = out_dir or os.path.join(HERE, "docs", "gifs")
    os.makedirs(d, exist_ok=True)
    jobs = [(k, int(s)) for k in courses.split(",") for s in seeds.split(",")]
    # --seconds 0: each course's own length (tactics 110 s, town 120 s, city 225 s, else 90 s), with the call
    # budget scaled to it (240 per 90 s)
    secs = {k: seconds or {"tactics": 110.0, "town": 120.0, "city": 225.0}.get(k.split("@")[0], 90.0) for k, _ in jobs}
    calls = [fly_gif.spawn(config, s, secs[k], k, budget if seconds else int(240 * secs[k] / 90), model, realtime)
             for k, s in jobs]
    out = _outdir()
    for (k, s), fc in zip(jobs, calls):
        res = _get(fc)
        if isinstance(res, Exception):
            print("FAILED", k, s, repr(res)[:300])
            continue
        r, gif, track = res
        tag = ("stuck" if r["finished_at_s"] is None
               else "lowvis" if r["target_visible_pct"] < 30 else "finished")
        path = os.path.join(d, "%s-%s-seed%d-%s.%s" % (config, k, s, tag, "mp4" if realtime else "gif"))
        open(path, "wb").write(gif)
        open(os.path.join(out, "track-%s-%s-seed%d.json" % (config, k, s)), "w").write(track)
        with open(os.path.join(out, "episodes.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        print("%-8s seed=%d %-8s fin=%s max_x=%5.1f vis=%4.1f%% rt=%.2f behind=%s/%s%% laya_calls=%s mae=%s lat=%s gif=%dKB"
              % (k, s, tag, r["finished_at_s"], r["max_x_m"], r["target_visible_pct"], r["realtime_factor"],
                 r.get("max_behind_s"), r.get("behind_pct"), (r.get("laya_server") or {}).get("calls"),
                 r.get("pursuit_bearing_mae_deg"), r.get("pursuit_median_latency_s"), len(gif) // 1024), flush=True)


@app.local_entrypoint()
def probe(frames: str = "/tmp/probe", model: str = "", n_permutations: int = 1, mode: str = "full",
          n_strips: int = 5):
    """Score zero-shot Laya on frames from `python probe.py collect --out <frames>`."""
    rows = [json.loads(l) for l in open(os.path.join(frames, "labels.jsonl"))]
    blobs = {r["frame"]: open(os.path.join(frames, "frames", r["frame"]), "rb").read() for r in rows}
    preds, summary = (json.loads(x) for x in probe_remote.remote(blobs, rows, model, n_permutations, mode, n_strips))
    d = os.path.join(HERE, "results", "probe", os.path.basename(os.path.normpath(frames)),
                     (model.strip("/").replace("/ckpt/smolvlm/", "").replace("/", "_") if model else "zero-shot"))
    os.makedirs(d, exist_ok=True)
    tag = {"strips": "strips%d" % n_strips, "v2": "v2", "v3": "v3", "alt": "alt"}.get(mode, "perm%d" % n_permutations)
    with open(os.path.join(d, "preds-%s.jsonl" % tag), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    json.dump(summary, open(os.path.join(d, "summary-%s.json" % tag), "w"), indent=1)
    print(json.dumps(summary, indent=1))


@app.local_entrypoint()
def build_rover_set(train_seeds: str = "0,1,2,3,4,5,6,7", val_seeds: str = "8,9", courses: str = "classic,pockets,mixed",
                    seconds: float = 60.0):
    """Write /data/vqa/drone_rover (create-only) from oracle flights, one Modal CPU job per flight."""
    if rover_set_exists.remote():
        raise SystemExit("/data/vqa/%s already exists; refusing to overwrite" % ROVER_SET)
    jobs = [(c, int(s), sp) for sp, seeds in (("train", train_seeds), ("val", val_seeds))
            for c in courses.split(",") for s in seeds.split(",")]
    calls = [collect_job.spawn(c, s, sp, seconds) for c, s, sp in jobs]
    out = []
    for (c, s, sp), fc in zip(jobs, calls):
        out.append(fc.get())
        print("collected", c, s, sp, flush=True)
    counts = finalize_rover_set.remote(out, {"source": "jev-drone rover_data.py", "courses": courses,
                                             "train_seeds": train_seeds, "val_seeds": val_seeds, "seconds": seconds})
    print("wrote /data/vqa/%s:" % ROVER_SET, counts)


@app.local_entrypoint()
def build_rover_set_v2():
    print("wrote /data/vqa/%s_v2:" % ROVER_SET, build_rover_set_v2_remote.remote())


@app.local_entrypoint()
def build_rover_set_v3(train_seeds: str = "0,1,2,3,4,5,6,7", val_seeds: str = "8,9",
                       courses: str = "pockets,mixed,classic"):
    """Write /data/vqa/drone_rover_v3 (create-only) from new flights, one Modal CPU job per flight: oracle
    tactics on pockets / mixed, const:climb on classic (rover_data.collect_flight_v3). no-climb and seeds >= 20
    are held out for evaluation (rover_test_frames_v3)."""
    if rover_set_v3_exists.remote():
        raise SystemExit("/data/vqa/%s already exists; refusing to overwrite" % ROVER_SET_V3)
    jobs = [(c, int(s), sp) for sp, seeds in (("train", train_seeds), ("val", val_seeds))
            for c in courses.split(",") for s in seeds.split(",")]
    assert not any(c == "no-climb" or s >= 20 for c, s, _ in jobs), "no-climb and seeds >= 20 are held out"
    calls = [collect_v3_job.spawn(c, s, sp) for c, s, sp in jobs]
    out = []
    for (c, s, sp), fc in zip(jobs, calls):
        out.append(fc.get())
        print("collected", c, s, sp, len(json.loads(out[-1])), "records", flush=True)
    counts, labels = finalize_rover_set_v3.remote(out, {"source": "jev-drone rover_data.collect_flight_v3 / records_v3",
                                                        "courses": courses, "train_seeds": train_seeds,
                                                        "val_seeds": val_seeds})
    print("wrote /data/vqa/%s:" % ROVER_SET_V3, counts)
    print(json.dumps(labels, indent=1))


@app.local_entrypoint()
def rover_test_frames_v3(courses: str = "no-climb,no-climb,mixed,mixed", seeds: str = "0,1,20,21",
                         out: str = "/tmp/rover-test-v3"):
    """Held-out v3 frames (the unseen no-climb layout, unseen mixed seeds) in probe.py's format, with state_text
    and every v3 truth field: score with rover_data.evaluate_v3 / score_v3."""
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    jobs = list(zip(courses.split(","), [int(s) for s in seeds.split(",")]))
    rows = []
    for rs, blobs in test_frames_v3_job.starmap([(c, s) for c, s in jobs]):
        for name, b in blobs.items():
            open(os.path.join(out, "frames", name), "wb").write(b)
        rows += json.loads(rs)
    with open(os.path.join(out, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "frames,", sum(not r["visible"] for r in rows), "with the rover out of sight ->", out)


@app.local_entrypoint()
def build_rover_set_v3b(train_seeds: str = "10,11,12,13,14,15,16,17", val_seeds: str = "18,19",
                        rot_courses: str = "pockets,mixed,classic", fail_courses: str = "pockets,mixed",
                        fail_kinds: str = "sim6,sim10,simr"):
    """Write /data/vqa/drone_rover_v3b (create-only), one Modal CPU job per flight: rotated views of oracle
    flights on rot_courses, and frames around losses on failure flights (rover_data.V3B_FAILURE) on fail_courses
    (rover_data.collect_flight_v3b). Seeds 0-9 are drone_rover_v3's; no-climb and seeds >= 20 are held out
    (rover_test_frames_v3b)."""
    if rover_set_v3b_exists.remote():
        raise SystemExit("/data/vqa/%s already exists; refusing to overwrite" % ROVER_SET_V3B)
    jobs = [(c, int(s), k, sp) for sp, seeds in (("train", train_seeds), ("val", val_seeds)) for s in seeds.split(",")
            for c, k in [(c, "rotated") for c in rot_courses.split(",")]
            + [(c, k) for c in fail_courses.split(",") for k in fail_kinds.split(",")]]
    assert not any(c == "no-climb" or s >= 20 for c, s, _, _ in jobs), "no-climb and seeds >= 20 are held out"
    calls = [collect_v3b_job.spawn(*j) for j in jobs]
    out = []
    for j, fc in zip(jobs, calls):
        r = _get(fc)
        if isinstance(r, Exception):
            print("FAILED", j, repr(r)[:300], flush=True)
            continue
        out.append(r)
        print("collected", *j, len(json.loads(r)), "records", flush=True)
    if len(out) < len(jobs):
        raise SystemExit("%d of %d flights failed; not finalizing (images stay; rerun after a fix)"
                         % (len(jobs) - len(out), len(jobs)))
    counts, named = finalize_rover_set_v3b.remote(out, {
        "source": "jev-drone rover_data.collect_flight_v3b / records_v3", "rot_courses": rot_courses,
        "fail_courses": fail_courses, "fail_kinds": fail_kinds, "train_seeds": train_seeds, "val_seeds": val_seeds})
    print("wrote /data/vqa/%s:" % ROVER_SET_V3B, counts)
    print(json.dumps(named, indent=1))


@app.local_entrypoint()
def rover_test_frames_v3b(courses: str = "no-climb,no-climb,no-climb,mixed,mixed,mixed", seeds: str = "0,1,2,20,21,22",
                          kinds: str = "rotated,sim6,sim10,simr", out: str = "/tmp/rover-test-v3b"):
    """Held-out v3b frames (rotated views + failure-flight losses on the unseen no-climb layout and unseen mixed
    seeds) in probe.py's format with state_text and every v3 truth field (view, yaw_offset_deg, source too;
    maneuver is None on rotated views): score with rover_data.evaluate_v3 / score_v3."""
    import collections
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    jobs = [(c, int(s), k) for c, s in zip(courses.split(","), seeds.split(",")) for k in kinds.split(",")]
    rows = []
    for rs, blobs in test_frames_v3b_job.starmap(jobs):
        for name, b in blobs.items():
            open(os.path.join(out, "frames", name), "wb").write(b)
        rows += json.loads(rs)
    with open(os.path.join(out, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lost = [r for r in rows if not r["visible"]]
    print(len(rows), "frames,", len(lost), "with the rover out of sight ->", out)
    for src in sorted({r["source"] for r in rows}):
        rr = [r for r in lost if r["source"] == src]
        print("  %-8s lost=%4d reappear=%s occluded=%s eta=%s" % (
            src, len(rr), dict(collections.Counter(r["reappear"] for r in rr)),
            dict(collections.Counter(r["occluded"] for r in rr)), dict(collections.Counter(r["eta_level"] for r in rr))))
    print("  all lost reappear:", dict(collections.Counter(r["reappear"] for r in lost)))


def _collect_all(fn, jobs):
    """Spawn fn(*job) for every job; stop before finalizing if any flight failed."""
    calls = [fn.spawn(*j) for j in jobs]
    out = []
    for j, fc in zip(jobs, calls):
        r = _get(fc)
        if isinstance(r, Exception):
            print("FAILED", j, repr(r)[:300], flush=True)
            continue
        out.append(r)
        print("collected", *j, len(json.loads(r)), "records", flush=True)
    if len(out) < len(jobs):
        raise SystemExit("%d of %d flights failed; not finalizing (images stay; rerun after a fix)"
                         % (len(jobs) - len(out), len(jobs)))
    return out


# seeds the v3.2 sets never train on: evaluation flights use 0-23 on mixed / no-climb; the tac test set uses
# mixed 24-27, no-climb 24-25, tactics 0-3; the town test set and town evaluation use town seeds 0-9
TAC_HELD_OUT = {"mixed": range(0, 28), "no-climb": range(0, 50), "tactics": range(0, 4), "town": range(0, 10)}


@app.local_entrypoint()
def build_rover_set_alt(train_seeds: str = "30,31,32,33,34,35,36,37,38,39,40,41", val_seeds: str = "46,47",
                        courses: str = "pockets,mixed,tactics", town_train: str = "10,11,12,13", town_val: str = "20",
                        wander: str = "0.2,0.35"):
    """Write /data/vqa/drone_rover_alt (create-only): altitude-operator flights (rover_data.collect_flight_alt)
    at each wander probability, balanced hold <= ascend + descend per split."""
    if rover_set_ready.remote(ROVER_SET_ALT):
        raise SystemExit("/data/vqa/%s already exists; refusing to overwrite" % ROVER_SET_ALT)
    ws = [float(w) for w in wander.split(",")]
    jobs = [(c, int(s), sp, w) for sp, seeds in (("train", train_seeds), ("val", val_seeds))
            for c in courses.split(",") for s in seeds.split(",") for w in ws]
    jobs += [("town", int(s), sp, ws[0]) for sp, seeds in (("train", town_train), ("val", town_val))
             for s in seeds.split(",") if s]
    assert not any(s in TAC_HELD_OUT.get(c, ()) for c, s, _, _ in jobs), "held-out seed in the training jobs"
    out = _collect_all(collect_alt_job, jobs)
    counts, report = finalize_rover_set_v32.remote(ROVER_SET_ALT, out, {
        "source": "jev-drone rover_data.collect_flight_alt / records_alt / balance_alt", "courses": courses,
        "train_seeds": train_seeds, "val_seeds": val_seeds, "town_train": town_train, "town_val": town_val,
        "wander": wander})
    print("wrote /data/vqa/%s:" % ROVER_SET_ALT, counts)
    print(json.dumps(report, indent=1))


@app.local_entrypoint()
def build_rover_set_onpolicy(config: str = "laya-full-v3.3", model: str = "/ckpt/smolvlm/drone-rover-v3.3/best",
                             train: str = "pockets:30-41,mixed:30-41,tactics:10-17,no-climb:50-57,town:10-15",
                             val: str = "pockets:46-47,mixed:46-47,tactics:20-21,town:20"):
    """Write /data/vqa/drone_rover_onpolicy (create-only): `config` flies on `model` with latency-faithful timing
    and every frame it saw is labelled from the sim (rover_data.collect_flight_onpolicy). `train` / `val`:
    course:first-last seed ranges, all outside the evaluation seeds (TAC_HELD_OUT)."""
    if rover_set_ready.remote(ROVER_SET_ONP):
        raise SystemExit("/data/vqa/%s already exists; refusing to overwrite" % ROVER_SET_ONP)

    def parse(spec, split):
        out = []
        for part in spec.split(","):
            c, rng = part.split(":")
            a, b = (rng.split("-") + [rng])[:2]
            out += [(c, s, split, config, model) for s in range(int(a), int(b) + 1)]
        return out
    jobs = parse(train, "train") + parse(val, "val")
    assert not any(s in TAC_HELD_OUT.get(c, ()) for c, s, *_ in jobs), "held-out seed in the jobs"
    out = _collect_all(collect_onpolicy_job, jobs)
    counts, report = finalize_rover_set_v32.remote(ROVER_SET_ONP, out, {
        "source": "jev-drone rover_data.collect_flight_onpolicy / records_onpolicy / balance_onpolicy",
        "config": config, "model": model, "train": train, "val": val, "timing": "virtual"})
    print("wrote /data/vqa/%s:" % ROVER_SET_ONP, counts)
    print(json.dumps(report, indent=1))


@app.local_entrypoint()
def rover_test_frames_alt(courses: str = "mixed,mixed,mixed,no-climb,no-climb,tactics,tactics",
                          seeds: str = "24,25,26,24,25,0,1", out: str = "/tmp/rover-test-alt"):
    """Held-out altitude frames: score with probe --mode alt (rover_data.evaluate_alt / score_alt)."""
    import collections
    jobs = list(zip(courses.split(","), [int(s) for s in seeds.split(",")]))
    rows = _write_test(out, test_frames_alt_job.starmap(jobs))
    print(len(rows), "frames ->", out, dict(collections.Counter(
        ("ascend" if r["dz_m"] > 0.25 else "descend" if r["dz_m"] < -0.25 else "hold") for r in rows)))


@app.local_entrypoint()
def build_rover_set_tac(train_seeds: str = "30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45",
                        val_seeds: str = "46,47,48,49", courses: str = "classic,pockets,mixed,tactics"):
    """Write /data/vqa/drone_rover_tac (create-only): maneuver-only oracle flights (rover_data.collect_flight_tac),
    dense near beams and pocket walls, balanced climb : pocket-hold : other-hold ~ 1:1:1 per split."""
    if rover_set_ready.remote(ROVER_SET_TAC):
        raise SystemExit("/data/vqa/%s already exists; refusing to overwrite" % ROVER_SET_TAC)
    jobs = [(c, int(s), sp) for sp, seeds in (("train", train_seeds), ("val", val_seeds))
            for c in courses.split(",") for s in seeds.split(",")]
    assert not any(s in TAC_HELD_OUT.get(c, ()) for c, s, _ in jobs), "held-out seed in the training jobs"
    out = _collect_all(collect_tac_job, jobs)
    counts, report = finalize_rover_set_v32.remote(ROVER_SET_TAC, out, {
        "source": "jev-drone rover_data.collect_flight_tac / records_tac / balance_tac", "courses": courses,
        "train_seeds": train_seeds, "val_seeds": val_seeds})
    print("wrote /data/vqa/%s:" % ROVER_SET_TAC, counts)
    print(json.dumps(report, indent=1))


@app.local_entrypoint()
def build_rover_set_town(train_seeds: str = "10,11,12,13,14,15,16,17,18,19", val_seeds: str = "20,21",
                         train_sim: str = "10:sim8,11:sim8,12:sim8,13:sim8,14:sim10,15:sim10",
                         val_sim: str = "20:sim8"):
    """Write /data/vqa/drone_rover_town (create-only): code-pursuit town flights on the seeds, plus the
    `seed:kind` failure flights (rover_data.TOWN_FAILURE stand-ins, which lose the rover)
    (rover_data.collect_flight_town / records_town). Town seeds 0-9 are held out for evaluation."""
    if rover_set_ready.remote(ROVER_SET_TOWN):
        raise SystemExit("/data/vqa/%s already exists; refusing to overwrite" % ROVER_SET_TOWN)
    jobs = []
    for sp, seeds, sims in (("train", train_seeds, train_sim), ("val", val_seeds, val_sim)):
        jobs += [(int(s), "code", sp) for s in seeds.split(",") if s]
        jobs += [(int(x.split(":")[0]), x.split(":")[1], sp) for x in sims.split(",") if x]
    assert not any(s in TAC_HELD_OUT["town"] for s, _, _ in jobs), "town seeds 0-9 are held out"
    out = _collect_all(collect_town_job, jobs)
    counts, report = finalize_rover_set_v32.remote(ROVER_SET_TOWN, out, {
        "source": "jev-drone rover_data.collect_flight_town / records_town", "train_seeds": train_seeds,
        "val_seeds": val_seeds, "train_sim": train_sim, "val_sim": val_sim})
    print("wrote /data/vqa/%s:" % ROVER_SET_TOWN, counts)
    print(json.dumps(report, indent=1))


def _write_test(out, results):
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    rows = []
    for rs, blobs in results:
        for name, b in blobs.items():
            open(os.path.join(out, "frames", name), "wb").write(b)
        rows += json.loads(rs)
    with open(os.path.join(out, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


@app.local_entrypoint()
def rover_test_frames_tac(courses: str = "mixed,mixed,mixed,mixed,no-climb,no-climb,tactics,tactics,tactics,tactics",
                          seeds: str = "24,25,26,27,24,25,0,1,2,3", out: str = "/tmp/rover-test-tac"):
    """Held-out tac frames (probe.py format, state_text, v3 truth + station_kind / station_dx_m): score with
    rover_data.evaluate_v3 / score_v3 (its maneuver_tac block is score_tac)."""
    import collections
    jobs = list(zip(courses.split(","), [int(s) for s in seeds.split(",")]))
    rows = _write_test(out, test_frames_tac_job.starmap(jobs))
    print(len(rows), "frames ->", out)
    for c in sorted({r["course"] for r in rows}):
        rr = [r for r in rows if r["course"] == c]
        print("  %-9s %s" % (c, dict(collections.Counter("%s/%s" % (r["station_kind"], r["maneuver"]) for r in rr))))


@app.local_entrypoint()
def rover_test_frames_town(seeds: str = "0,1,2,3", sim: str = "0:sim8,1:sim8", out: str = "/tmp/rover-test-town"):
    """Held-out town views (probe.py format, state_text, v2 perception + v3 reacquisition truth, view, scenery_px):
    score perception with probe.evaluate_v2 / score_v2 and reacquisition with rover_data.evaluate_v3 / score_v3."""
    import collections
    jobs = [(int(s), "code") for s in seeds.split(",") if s] + [(int(x.split(":")[0]), x.split(":")[1])
                                                                for x in sim.split(",") if x]
    rows = _write_test(out, test_frames_town_job.starmap(jobs))
    lost = [r for r in rows if not r["visible"]]
    print(len(rows), "views,", len(lost), "with the rover out of sight ->", out)
    print("  views:", dict(collections.Counter(r["view"] for r in rows)))
    print("  hard negatives (red scenery, no rover):",
          sum(1 for r in lost if r["scenery_px"] >= 20))
    print("  lost reappear:", dict(collections.Counter(r["reappear"] for r in lost if r["reacq"])))
    print("  lost occluded:", dict(collections.Counter(r["occluded"] for r in lost if r["reacq"])))


@app.local_entrypoint()
def rover_test_frames(courses: str = "no-climb,no-climb,mixed,mixed", seeds: str = "0,1,20,21",
                      out: str = "/tmp/rover-test", seconds: float = 60.0):
    """Held-out frames (a layout never trained on, and unseen seeds) in probe.py's format."""
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    jobs = list(zip(courses.split(","), [int(s) for s in seeds.split(",")]))
    rows = []
    for rs, blobs in test_frames_job.starmap([(c, s, seconds) for c, s in jobs]):
        rs = json.loads(rs)
        for name, b in blobs.items():
            open(os.path.join(out, "frames", name), "wb").write(b)
        rows += rs
    with open(os.path.join(out, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "frames,", sum(r["visible"] for r in rows), "with the rover in view ->", out)


@app.local_entrypoint()
def scenes(model: str = "", all_variants: bool = True):
    variants = [("full budgets, no image", []), ("full budgets, grey image", ["--blank-image"]),
                ("48-token options, no image", ["--short"]),
                ("48-token options, grey image", ["--short", "--blank-image"])]
    text, rows = scenes_remote.remote(model, variants if all_variants else variants[:1])
    print(text)
    d = _outdir()
    json.dump(rows, open(os.path.join(d, "scenes.json"), "w"), indent=1)
    print("wrote", d)


@app.local_entrypoint()
def baseline(configs: str = "no-model,laya-text,laya-image,laya-text-lockstep", seeds: str = "0,1,2",
             seconds: float = 65.0, model: str = "", courses: str = "classic", budget: int = 0):
    """`budget` caps model calls per flight; 0 keeps tactics.THRESHOLDS["call_budget"] (160, sized
    for the 65 s classic flight). Scale it with --seconds, or a long flight runs out mid-course."""
    jobs = [(c, int(s), k) for k in courses.split(",") for c in configs.split(",") for s in seeds.split(",")]
    d = _outdir()
    path = os.path.join(d, "episodes.jsonl")
    calls = [(fly if _needs_gpu(c) else fly_cpu).spawn(c, s, seconds, model, k, budget) for c, s, k in jobs]
    for r in (_get(fc) for fc in calls):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:300])
            continue
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        st = r["jev"] if isinstance(r["jev"], dict) else {}
        print("%-9s %-26s seed=%d fin=%-5s max_x=%5.1f crossed=%-5s vis=%4.1f%% reflex=%4.1f%% hits=%d rt=%.2f calls=%s med=%s%s"
              % (r.get("course", "classic"), r["config"], r["seed"], r.get("finished_at_s"), r["max_x_m"], r["crossed_barrier"], r["target_visible_pct"],
                 r["steps_reflex_pct"], r["collisions"], r["realtime_factor"], st.get("calls"),
                 st.get("median_latency_s"),
                 "" if r.get("pursuit", "code") == "code" else " | steer=%s used=%s%% mae=%s lat=%s/%s"
                 % (r["pursuit"], r.get("pursuit_used_pct"), r.get("pursuit_bearing_mae_deg"),
                    r.get("pursuit_median_latency_s"), r.get("pursuit_p90_latency_s"))), flush=True)
    print("wrote", path)


# ---- real drone footage test set (realdata.py): evaluation only, never training ----
# UAV123 / VisDrone-SOT are research / non-commercial datasets. The CPU jobs read the source zips over
# HTTP range requests and write only the subsampled, resized frames to laya-datasets (/data/realtest/);
# nothing is downloaded to the local disk. results/realtest/README.md has the terms and the numbers.
REALTEST_SET = "real-v1"
real_image = (modal.Image.debian_slim(python_version="3.12").pip_install("requests", "numpy", "pillow")
              .add_local_file(os.path.join(HERE, "realdata.py"), "/root/realdata.py")
              .add_local_file(os.path.join(HERE, "probe.py"), "/root/probe.py"))


def _real():
    import sys
    sys.path.insert(0, "/root")
    import realdata
    return realdata


@app.function(image=real_image, cpu=2, memory=8192, timeout=30 * 60)
def realtest_fetch(name: str):
    """The layout of one realdata.SOURCES zip, read remotely (only its central directory is fetched)."""
    return json.dumps(_real().inspect(name))


@app.function(image=real_image, cpu=1, memory=4096, timeout=20 * 60)
def realtest_uav123_jobs():
    return _real().uav123_jobs()


@app.function(image=real_image, cpu=2, memory=8192, timeout=90 * 60, volumes={"/data": data_vol})
def realtest_convert(dataset: str, jobs: list, set_name: str = REALTEST_SET):
    """Convert some sequences onto /data/realtest/<set>/frames; return their label rows (JSON)."""
    rd = _real()
    out = "/data/realtest/%s" % set_name
    rows = (rd.convert_uav123([tuple(j) for j in jobs], out) if dataset == "uav123"
            else rd.convert_visdrone([tuple(j) for j in jobs], out))
    data_vol.commit()
    return json.dumps(rows)


@app.function(image=real_image, cpu=1, memory=4096, timeout=10 * 60, volumes={"/data": data_vol})
def realtest_finalize(rows_json: str, meta: dict, set_name: str = REALTEST_SET):
    data_vol.reload()
    d = "/data/realtest/%s" % set_name
    rows = json.loads(rows_json)
    with open(os.path.join(d, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    data_vol.commit()
    return len(rows), len(os.listdir(os.path.join(d, "frames")))


@app.local_entrypoint()
def realtest_inspect(names: str = "uav123_10fps,visdrone_sot"):
    """Print each source zip's layout (read remotely)."""
    ns = names.split(",")
    for n, fc in zip(ns, [realtest_fetch.spawn(n) for n in ns]):
        r = _get(fc)
        print("=====", n)
        print(r if isinstance(r, Exception) else json.dumps(json.loads(r), indent=0)[:12000], flush=True)


@app.local_entrypoint()
def realtest_build(set_name: str = REALTEST_SET, chunk: int = 6):
    """Build /data/realtest/<set>/ (labels.jsonl, frames/, meta.json) from UAV123 car / person / truck /
    bike and the labelled VisDrone-SOT val + test-dev sequences, in parallel CPU jobs."""
    import realdata
    uav = realtest_uav123_jobs.remote()
    vd = [(sp, s) for sp, d in realdata.VISDRONE_CLASS.items() for s in sorted(d)]
    tasks = [("uav123", uav[i:i + chunk]) for i in range(0, len(uav), chunk)]
    tasks += [("visdrone", vd[i:i + chunk]) for i in range(0, len(vd), chunk)]
    print(len(uav), "UAV123 sub-sequences,", len(vd), "VisDrone sequences,", len(tasks), "jobs", flush=True)
    rows = []
    for (ds, js), fc in zip(tasks, [realtest_convert.spawn(ds, js, set_name) for ds, js in tasks]):
        r = _get(fc)
        if isinstance(r, Exception):
            print("FAILED", ds, js, repr(r)[:400], flush=True)
            continue
        rows += json.loads(r)
        print(ds, [j[0] if ds == "uav123" else j[1] for j in js], "->", len(rows), "rows", flush=True)
    meta = {"hfov_deg_assumed": realdata.HFOV_DEG, "long_side": realdata.LONG_SIDE, "crop_h": realdata.CROP_H,
            "uav123_every": 6, "visdrone_every": 30, "cap_per_seq": 24, "uav123_jobs": uav, "visdrone": vd,
            "sources": {k: v["url"] for k, v in realdata.SOURCES.items()}}
    print("rows, frames on the volume:", realtest_finalize.remote(json.dumps(rows), meta, set_name))


@app.function(gpu="L4", cpu=4, memory=16384, timeout=4 * 60 * 60,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only(), "/data": data_vol.read_only()})
def realtest_probe(model: str = "", set_name: str = REALTEST_SET, limit: int = 0):
    """Ask one checkpoint realdata.questions_real (as trained, and naming the real target) on every frame
    of /data/realtest/<set>; returns (preds, summary) as JSON text."""
    _enter()
    import laya, realdata
    d = "/data/realtest/%s" % set_name
    rows = [json.loads(l) for l in open(os.path.join(d, "labels.jsonl"))]
    if limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]
    agent = laya.load_vlm(model or "thaitea/laya-vision", option_max_len=256, head_max_len=1024, max_len=3072)
    t0 = time.time()
    preds = realdata.evaluate_real(agent, os.path.join(d, "frames"), rows)
    summary = realdata.score_groups(preds)
    summary["_meta"] = {"model": model or "thaitea/laya-vision", "set": set_name, "rows": len(rows),
                        "seconds": round(time.time() - t0, 1)}
    return json.dumps(preds, default=float), json.dumps(summary, default=float)


def _run_tag(model):
    return model.strip("/").replace("/ckpt/smolvlm/", "").replace("/", "_") if model else "zero-shot"


@app.local_entrypoint()
def realtest_score(models: str = ",/ckpt/smolvlm/drone-rover-v2/last,/ckpt/smolvlm/drone-rover-v3.1/best",
                   set_name: str = REALTEST_SET, limit: int = 0, preds_dir: str = ""):
    """One L4 job per checkpoint ('' = zero-shot thaitea/laya-vision). Summaries ->
    results/realtest/<run>/summary.json. Per-frame predictions carry the datasets' boxes, so they go
    to --preds-dir (default: not saved), never into the repo."""
    ms = models.split(",")
    calls = [realtest_probe.spawn(m, set_name, limit) for m in ms]
    for m, fc in zip(ms, calls):
        r = _get(fc)
        if isinstance(r, Exception):
            print("FAILED", m or "zero-shot", repr(r)[:600], flush=True)
            continue
        preds, summary = r
        d = os.path.join(HERE, "results", "realtest", _run_tag(m) + ("-limit%d" % limit if limit else ""))
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "summary.json"), "w").write(json.dumps(json.loads(summary), indent=1))
        if preds_dir:
            os.makedirs(preds_dir, exist_ok=True)
            open(os.path.join(preds_dir, _run_tag(m) + ".json"), "w").write(preds)
        a = json.loads(summary)
        for w in ("rover", "target"):
            x = a[w]["all"]
            print("%-28s %-6s auc=%.3f where=%.3f(maj %.3f) side=%.3f rho_steer=%.3f rho_range=%.3f" % (
                _run_tag(m), w, x["visible_auc"] or 0, x["where_acc"], x["where_majority_baseline"],
                x["steer_side_acc"] or 0, x["steer_spearman_vs_offset"] or 0, x["range_spearman_vs_box"] or 0),
                flush=True)


@app.function(image=real_image, cpu=1, memory=4096, timeout=10 * 60, volumes={"/data": data_vol.read_only()})
def realtest_draw(items_json: str, set_name: str = REALTEST_SET):
    rd = _real()
    return [rd.draw_sample("/data/realtest/%s/frames" % set_name, row, ans) for row, ans in json.loads(items_json)]


@app.local_entrypoint()
def realtest_samples(preds_dir: str, frames: str, wording: str = "target", set_name: str = REALTEST_SET,
                     out: str = "results/realtest/samples"):
    """Draw the listed frames (comma-separated names) with each checkpoint's answers (from
    realtest_score --preds-dir) to small PNGs."""
    runs = [("zero-shot", "zero-shot"), ("v2 last", "ckpt_smolvlm_drone-rover-v2_last"), ("v3.1 best", "ckpt_smolvlm_drone-rover-v3.1_best")]
    preds = {tag: {p["frame"]: p for p in json.load(open(os.path.join(preds_dir, tag + ".json")))
                   if p["wording"] == wording} for _, tag in runs}
    names = frames.split(",")
    items = [(preds[runs[0][1]][n], [(lab, preds[tag][n]) for lab, tag in runs]) for n in names]
    os.makedirs(os.path.join(HERE, out), exist_ok=True)
    for n, png in zip(names, realtest_draw.remote(json.dumps(items), set_name)):
        path = os.path.join(HERE, out, n.rsplit(".", 1)[0] + ".png")
        open(path, "wb").write(png)
        print("wrote", path, len(png) // 1024, "KB")
