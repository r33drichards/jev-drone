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
    .apt_install("git", "libosmesa6", "libgl1")
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
    "code-pursuit-simreacq": (True, "const:oracle", False, False, {"pursuit": "code", "reacquire": "sim"}),
    "sim-pursuit-simreacq": (True, "const:oracle", False, False,
                             {"pursuit": "sim-pursuit", "pursuit_noise_deg": 4.0, "pursuit_delay_s": 0.1,
                              "pursuit_range": {"range_noise_m": 0.45, "range_offset_m": -0.1}, "reacquire": "sim"}),
}


def _config(name):
    """(use_model, backend, img, lockstep, pursuit kwargs); the older 4-tuples fly code pursuit."""
    c = CONFIGS[name]
    return tuple(c[:4]) + (dict(c[4]) if len(c) > 4 else {},)


def _needs_gpu(name):
    use_model, backend, _, _, pk = _config(name)
    return ((use_model and not backend.startswith("const:")) or pk.get("pursuit", "code").startswith("laya")
            or pk.get("reacquire") == "laya")


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


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=60 * 60, max_containers=GPU_MAX,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def fly(config: str, seed: int, seconds: float, model: str = "", course: str = "classic", budget: int = 0):
    _enter()
    import run
    use_model, backend, img, lockstep, pk = _config(config)
    t0 = time.time()
    pk = dict(pk)
    # a config may name its own pursuit checkpoint ("pursuit_model"); --model then serves tactics and
    # reacquisition (hybrid: v2 flies the pursuit, v3.1 answers where a lost rover will reappear)
    pm = pk.pop("pursuit_model", None)
    extra = {"reacquire_model": model or None} if pm and pk.get("reacquire") == "laya" else {}
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


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=60 * 60, max_containers=GPU_MAX,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def fly_gif(config: str, seed: int, seconds: float, course: str, budget: int = 0, model: str = ""):
    """fly(), recording the flight, then render it as a GIF (flightgif.py) after the flight ends.
    Returns (episode result, GIF bytes, the snapshots without images as JSON text)."""
    _enter()
    import run, flightgif
    import numpy as np
    use_model, backend, img, lockstep, pk = _config(config)
    rec = []
    r = run.episode(seed, seconds, use_jev=use_model, backend=backend, laya_model=model or None,
                    laya_image=img, lockstep=lockstep, course=course, budget=budget or None, record=rec,
                    pursuit_model=model or None, **pk)
    r.update(config=config)
    outcome = "finished" if r["finished_at_s"] is not None else "stopped at x=%.0f m" % r["max_x_m"]
    title = "%s  |  %s seed %d  |  %s" % (config, course, seed, outcome)
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
                                               "code_range_m", "guide", "reflex", "climbing", "hits")},
                      "maneuver": s["judg"].get("maneuver")})
    label = "tactics (oracle)" if backend == "const:oracle" else "tactics (%s)" % (
        backend if use_model else "none")
    return (r, flightgif.make_gif(rec, course, seed, title, "/root/jev", tactics_label=label),
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
         seconds: float = 90.0, budget: int = 240, model: str = ""):
    """Fly and record every (course, seed); write docs/gifs/<config>-<course>-seed<k>-<outcome>.gif
    (outcome: finished, lowvis = finished with the rover in view < 30%, stuck), and each flight's
    snapshots without images to results/laya/<timestamp>/track-*.json."""
    d = os.path.join(HERE, "docs", "gifs")
    os.makedirs(d, exist_ok=True)
    jobs = [(k, int(s)) for k in courses.split(",") for s in seeds.split(",")]
    calls = [fly_gif.spawn(config, s, seconds, k, budget, model) for k, s in jobs]
    out = _outdir()
    for (k, s), fc in zip(jobs, calls):
        res = _get(fc)
        if isinstance(res, Exception):
            print("FAILED", k, s, repr(res)[:300])
            continue
        r, gif, track = res
        tag = ("stuck" if r["finished_at_s"] is None
               else "lowvis" if r["target_visible_pct"] < 30 else "finished")
        path = os.path.join(d, "%s-%s-seed%d-%s.gif" % (config, k, s, tag))
        open(path, "wb").write(gif)
        open(os.path.join(out, "track-%s-%s-seed%d.json" % (config, k, s)), "w").write(track)
        with open(os.path.join(out, "episodes.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        print("%-8s seed=%d %-8s fin=%s max_x=%5.1f vis=%4.1f%% rt=%.2f mae=%s lat=%s gif=%dKB"
              % (k, s, tag, r["finished_at_s"], r["max_x_m"], r["target_visible_pct"], r["realtime_factor"],
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
    tag = {"strips": "strips%d" % n_strips, "v2": "v2", "v3": "v3"}.get(mode, "perm%d" % n_permutations)
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
