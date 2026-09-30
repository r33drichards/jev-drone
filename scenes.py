"""Hand-built scenes: does the tactical model make the right call, before any flying?

The seven situations from the README, written in the same schema flight.Eye
produces. Each has the maneuver a sensible pilot would pick (and, where it is the
point of the scene, whether the target should count as lost). Free, fast, and
backend-agnostic, so it is the first thing to run on a new decision model:

    python scenes.py --backend laya
    python scenes.py --backend jev
"""
import argparse, json, time
from tactics import build_state, make_backend

FAR = 45.0   # flight.Eye.MAX_RANGE: nothing detected


def scene(sectors, above, level, below=FAR, target=None):
    names = ["far_left", "left", "center", "right", "far_right"]
    sec = dict(zip(names, sectors))
    nearest = min(sectors)
    t = target or {"visible": True, "bearing_deg": 0.0, "range_m": 6.0, "pixels": 40, "unseen_for_s": 0.0}
    return {"sector_range_m": sec,
            "sectors_blocked": sum(v < 3.0 for v in sectors),
            "path_ahead_m": min(sec["left"], sec["center"], sec["right"]),
            "free_ahead_above_m": above, "free_ahead_level_m": level, "free_ahead_below_m": below,
            "room_above_m": None, "room_below_m": None, "room_left_m": None, "room_right_m": None,
            "nearest_obstacle_m": nearest,
            "nearest_bearing_deg": [30.0, 15.0, 0.0, -15.0, -30.0][sectors.index(nearest)],
            "target": t}


def lost(s):
    return {"visible": False, "bearing_deg": None, "range_m": None, "pixels": 0, "unseen_for_s": s}


# name, scene, expected maneuver(s), expected target_truly_lost (None = not the point), Jev's published answer
CASES = [
    ("low barrier, all 5 blocked", scene([2.4] * 5, above=FAR, level=2.4, target=lost(0.4)),
     {"climb"}, None, "climb p=0.93 risk=1.42"),
    ("tall pillar ahead, right wide open", scene([2.6, 2.2, 2.0, 12.0, FAR], above=2.0, level=2.0),
     {"gap_right"}, None, "gap_right p=0.72 risk=1.53"),
    ("tall pillar ahead, left wide open", scene([FAR, 12.0, 2.0, 2.2, 2.6], above=2.0, level=2.0),
     {"gap_left"}, None, "gap_left p=0.40 risk=1.58"),
    ("target gone 7 s, wide open", scene([FAR] * 5, above=FAR, level=FAR, target=lost(7.0)),
     {"reacquire"}, True, "reacquire p=0.96 lost=0.85"),
    ("boxed in, close on all sides, tall", scene([1.1, 0.9, 0.8, 0.9, 1.1], above=0.8, level=0.8),
     {"brake"}, None, "brake p=0.84 risk=1.89"),
    ("all clear, target dead ahead", scene([FAR] * 5, above=FAR, level=FAR),
     {"hold_course"}, None, "hold_course p=0.82 risk=0.46"),
    ("brief occlusion 0.6 s, path clear", scene([FAR, FAR, 9.0, FAR, FAR], above=FAR, level=9.0, target=lost(0.6)),
     {"hold_course"}, False, "muddy p=0.24, lost=0.12"),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["jev", "laya"], default="laya")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None, help="write the rows as JSON here")
    p.add_argument("--short", action="store_true", help="laya: the checkpoint's own budgets (options cut to 48 tokens)")
    p.add_argument("--blank-image", action="store_true", help="laya: send a plain grey frame beside the JSON")
    a = p.parse_args(argv)
    kw = {}
    if a.backend == "laya":
        kw["use_image"] = a.blank_image
        if a.short:
            kw.update(option_max_len=48, head_max_len=256, max_len=1024)
    be = make_backend(a.backend, model=a.model, **kw)
    img = None
    if a.blank_image:
        import numpy as np
        img = np.full((384, 512, 3), 128, dtype=np.uint8)
    be.ask(build_state(CASES[-1][1]), img)      # warm-up: weights, kernels, caches
    rows, right = [], 0
    for name, sc, want, want_lost, jev in CASES:
        t0 = time.time()
        j, _ = be.ask(build_state(sc), img)
        dt = time.time() - t0
        ok = j["maneuver"] in want and (want_lost is None or (j["target_truly_lost"] > 0.5) == want_lost)
        right += ok
        rows.append(dict(case=name, ok=ok, latency_s=round(dt, 3), jev=jev, **j))
        top = sorted(j["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        print("%-4s %-36s -> %-11s p=%.2f risk=%.2f lost=%.2f  [%s]  (%.0f ms)   jev: %s"
              % ("ok" if ok else "MISS", name, j["maneuver"], j["probabilities"].get(j["maneuver"], 0),
                 j["risk"], j["target_truly_lost"], ", ".join("%s %.2f" % kv for kv in top), dt * 1e3, jev),
              flush=True)
    label = be.model + (" short-budget" if a.short else "")
    print("\n%s: %d / %d correct%s" % (label, right, len(CASES),
          "" if not getattr(be, "truncated", 0) else "  (%d calls had inputs truncated)" % be.truncated))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"backend": label, "args": vars(a), "correct": right, "n": len(CASES), "rows": rows}, f, indent=1)
    be.close()


if __name__ == "__main__":
    main()
