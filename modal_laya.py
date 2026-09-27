"""Run the course with Laya Vision as the tactical layer, on Modal GPUs.

Laya runs in-process, so it needs a GPU to answer in time for a live control loop
(about 40 ms per call on an L4; seconds on a CPU). MuJoCo renders through OSMesa
(software), so nothing depends on the GPU driver's EGL.

The laya package comes from a local checkout of laya-vision, LAYA_DIR (default
../laya-vision), so an uncommitted change there is what runs.

    modal run modal_laya.py::scenes                         # the 7 hand-built scenes, text only
    modal run modal_laya.py::baseline --seeds 0,1,2          # every flight configuration below, in parallel
    modal run modal_laya.py::baseline --configs laya-image --seeds 1 --seconds 65

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

# name -> (use the model?, backend, laya sees the camera frame, lockstep)
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
}


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


@app.function(gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=60 * 60, volumes={"/cache/hf": hf_vol})
def fly(config: str, seed: int, seconds: float, model: str = "", course: str = "classic"):
    _enter()
    import run
    use_model, backend, img, lockstep = CONFIGS[config]
    t0 = time.time()
    r = run.episode(seed, seconds, use_jev=use_model, backend=backend, laya_model=model or None,
                    laya_image=img, lockstep=lockstep, course=course)
    r.update(config=config, wall_s=round(time.time() - t0, 1))
    try:
        import torch
        r["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return r


@app.function(cpu=4, memory=8192, timeout=60 * 60)
def fly_cpu(config: str, seed: int, seconds: float, model: str = "", course: str = "classic"):
    """The controls (no-model, const:*) never load a model, so they need no GPU."""
    return fly.local(config, seed, seconds, model, course)


@app.function(cpu=2, memory=4096, timeout=10 * 60)
def render_course(course: str, seed: int):
    """Top-down and chase-height views of a course, as PNG bytes."""
    _enter()
    import io, mujoco, numpy as np, courses
    from PIL import Image
    c = courses.make(course, seed)
    m = mujoco.MjModel.from_xml_path(c.write("/root/jev"))
    d = mujoco.MjData(m)
    d.qpos[:3] = [1.5, 0, 1.6]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 360, 1400)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [c.end_x / 2, 0, 0]; cam.distance, cam.elevation, cam.azimuth = c.end_x * 0.62, -89.9, 90.0
    m.vis.global_.fovy = 50
    r.update_scene(d, cam)
    top = r.render()
    # rover path, drawn as dots in the top view
    img = Image.fromarray(top)
    buf = io.BytesIO(); img.save(buf, "PNG")
    return buf.getvalue()


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
             seconds: float = 65.0, model: str = "", courses: str = "classic"):
    jobs = [(c, int(s), k) for k in courses.split(",") for c in configs.split(",") for s in seeds.split(",")]
    d = _outdir()
    path = os.path.join(d, "episodes.jsonl")
    gpu = lambda c: CONFIGS[c][0] and not CONFIGS[c][1].startswith("const:")  # noqa: E731
    calls = [(fly if gpu(c) else fly_cpu).spawn(c, s, seconds, model, k) for c, s, k in jobs]
    for r in (_get(fc) for fc in calls):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:300])
            continue
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        st = r["jev"] if isinstance(r["jev"], dict) else {}
        print("%-9s %-20s seed=%d fin=%-5s max_x=%5.1f crossed=%-5s vis=%4.1f%% reflex=%4.1f%% hits=%d rt=%.2f calls=%s med=%s"
              % (r.get("course", "classic"), r["config"], r["seed"], r.get("finished_at_s"), r["max_x_m"], r["crossed_barrier"], r["target_visible_pct"],
                 r["steps_reflex_pct"], r["collisions"], r["realtime_factor"], st.get("calls"),
                 st.get("median_latency_s")), flush=True)
    print("wrote", path)
