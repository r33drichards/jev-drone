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
ROVER_SET = "drone_rover"          # /data/vqa/drone_rover on laya-datasets, for laya-vision's finetune_long

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
    "laya-steer-strips-lockstep": (True, "const:oracle", False, False, {"pursuit": "laya-strips", "pursuit_lockstep": True}),
    # model-free stand-ins (CPU): the true bearing, then with a strips-like 0.2 s latency and ~6 deg noise
    "sim-steer": (True, "const:oracle", False, False, {"pursuit": "sim"}),
    "sim-steer-noisy": (True, "const:oracle", False, False,
                        {"pursuit": "sim", "pursuit_noise_deg": 6.0, "pursuit_delay_s": 0.2}),
}


def _config(name):
    """(use_model, backend, img, lockstep, pursuit kwargs); the older 4-tuples fly code pursuit."""
    c = CONFIGS[name]
    return tuple(c[:4]) + (dict(c[4]) if len(c) > 4 else {},)


def _needs_gpu(name):
    use_model, backend, _, _, pk = _config(name)
    return (use_model and not backend.startswith("const:")) or pk.get("pursuit", "code").startswith("laya")


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


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=60 * 60,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def fly(config: str, seed: int, seconds: float, model: str = "", course: str = "classic", budget: int = 0):
    _enter()
    import run
    use_model, backend, img, lockstep, pk = _config(config)
    t0 = time.time()
    r = run.episode(seed, seconds, use_jev=use_model, backend=backend, laya_model=model or None,
                    laya_image=img, lockstep=lockstep, course=course, budget=budget or None,
                    pursuit_model=model or None, **pk)
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


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=60 * 60,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def fly_gif(config: str, seed: int, seconds: float, course: str, budget: int = 0, model: str = ""):
    """fly(), recording the flight, then render it as a GIF (flightgif.py) after the flight ends."""
    _enter()
    import run, flightgif
    use_model, backend, img, lockstep, pk = _config(config)
    rec = []
    r = run.episode(seed, seconds, use_jev=use_model, backend=backend, laya_model=model or None,
                    laya_image=img, lockstep=lockstep, course=course, budget=budget or None, record=rec,
                    pursuit_model=model or None, **pk)
    r.update(config=config)
    outcome = "finished" if r["finished_at_s"] is not None else "stopped at x=%.0f m" % r["max_x_m"]
    title = "%s  |  %s seed %d  |  %s" % (config, course, seed, outcome)
    return r, flightgif.make_gif(rec, course, seed, title, "/root/jev")


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
    """Fly and record every (course, seed); write docs/gifs/<config>-<course>-<seed>-<outcome>.gif."""
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
        r, gif = res
        tag = "finished" if r["finished_at_s"] is not None else "stuck"
        path = os.path.join(d, "%s-%s-seed%d-%s.gif" % (config, k, s, tag))
        open(path, "wb").write(gif)
        with open(os.path.join(out, "episodes.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        print("%-8s seed=%d %-8s max_x=%5.1f vis=%4.1f%% rt=%.2f gif=%dKB"
              % (k, s, tag, r["max_x_m"], r["target_visible_pct"], r["realtime_factor"], len(gif) // 1024), flush=True)


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
    tag = "strips%d" % n_strips if mode == "strips" else "perm%d" % n_permutations
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
