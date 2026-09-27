"""Probe: can Laya see where the rover is, zero-shot, from the onboard camera frame?

Before handing pursuit to Laya, check the perception it would rest on. `collect` flies
the oracle (the flight itself does not matter, only the views) and saves the onboard
frame every 0.5 s with the ground truth from the simulator: the rover's bearing off
the nose, its range, and whether it is in view. `evaluate` asks Laya about each frame
and scores the answers against that truth.

    MUJOCO_GL=osmesa python probe.py collect --out /tmp/probe
    modal run modal_laya.py::probe --frames /tmp/probe        # evaluate on a GPU

Bearing is + to the aircraft's left, as everywhere in flight.py.
"""
import argparse, io, json, os
import numpy as np

# steer levels, from the rover's bearing (degrees, + left). Pursuit clips its yaw command at
# +-0.6 rad (34 deg); the camera sees +-55 deg horizontally.
STEER = ["hard left: the rover is far to the left of the image",
         "left: the rover is left of centre",
         "straight: the rover is near the centre of the image",
         "right: the rover is right of centre",
         "hard right: the rover is far to the right of the image"]
STEER_EDGES = [20.0, 7.0, -7.0, -20.0]           # > 20: hard left ... < -20: hard right
# speed levels, from range (m); pursuit holds 3.5 m behind the rover
SPEED = ["back off: the rover is very close, nearer than about 2.5 m",
         "slow: the rover is close, about 2.5 to 4 m away",
         "cruise: the rover is at a moderate distance, about 4 to 7 m",
         "fast: the rover is far away, more than about 7 m"]
SPEED_EDGES = [2.5, 4.0, 7.0]


def steer_level(b):
    return int(sum(b < e for e in STEER_EDGES))


def speed_level(r):
    return int(sum(r > e for e in SPEED_EDGES))


def questions():
    ctx = ("Onboard forward camera of a quadrotor following a small red ground rover "
           "(a red box with a thin red mast). ")
    return {
        "visible": {"type": "noul", "instructions": ctx + "Is the red rover visible anywhere in the image?"},
        "where": {"type": "choice", "instructions": ctx + "Where is the red rover in the image?",
                  "criteria": {"left": "in the left third of the image", "centre": "in the middle third",
                               "right": "in the right third of the image",
                               "not visible": "the red rover is not in the image"}},
        "steer": {"type": "score", "instructions": ctx + "Which way should the quadrotor turn to point at the rover?",
                  "criteria": STEER},
        "speed": {"type": "score", "instructions": ctx + "How far away is the red rover?", "criteria": SPEED},
    }


def collect(out, courses=("classic", "mixed"), seeds=(0, 1, 2), seconds=60.0, every_s=0.5):
    import mujoco, run, flight
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    rows = []
    for course in courses:
        for seed in seeds:
            snaps = []
            orig_look = flight.Eye.look

            def look(self, data, pos, yaw, t, _snaps=snaps):
                sc = orig_look(self, data, pos, yaw, t)
                if not _snaps or t - _snaps[-1]["t"] >= every_s - 1e-6:
                    rover = data.mocap_pos[self.m.body("rover").mocapid[0]].copy()
                    cam = np.asarray(pos) + self.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0])
                    d = rover - cam
                    fwd, left = (np.cos(yaw) * d[0] + np.sin(yaw) * d[1],
                                 -np.sin(yaw) * d[0] + np.cos(yaw) * d[1])
                    _snaps.append({"t": t, "rgb": self.last_rgb.copy(), "visible": bool(sc["target"]["visible"]),
                                   "pixels": int(sc["target"]["pixels"]),
                                   "bearing_deg": float(np.rad2deg(np.arctan2(left, fwd))),
                                   "range_m": float(np.hypot(fwd, left))})
                return sc

            flight.Eye.look = look
            try:
                backend = "const:oracle" if course != "classic" else "const:climb"
                run.episode(seed, seconds, use_jev=True, backend=backend, laya_image=True, course=course,
                            realtime=False)
            finally:
                flight.Eye.look = orig_look
            from PIL import Image
            for k, s in enumerate(snaps):
                name = "%s-%d-%03d.jpg" % (course, seed, k)
                Image.fromarray(s.pop("rgb")).save(os.path.join(out, "frames", name), quality=90)
                rows.append(dict(s, frame=name, course=course, seed=seed))
            print(course, seed, len(snaps), "frames", flush=True)
    with open(os.path.join(out, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def evaluate(agent, frames, rows, n_permutations=1):
    """`frames`: name -> JPEG bytes. Returns per-frame predictions."""
    from PIL import Image
    qs = questions()
    out = []
    for r in rows:
        img = Image.open(io.BytesIO(frames[r["frame"]])).convert("RGB")
        a = agent.predict({"image": img}, qs, n_permutations=n_permutations)["answers"]
        out.append(dict(r, p_visible=float(a["visible"]["noul"]), where=a["where"]["choice"],
                        where_probs={k: float(v) for k, v in a["where"]["probabilities"].items()},
                        steer=float(a["steer"]["score"]), speed=float(a["speed"]["score"]),
                        steer_probs=[float(a["steer"]["probabilities"][str(i)]) for i in range(len(STEER))],
                        speed_probs=[float(a["speed"]["probabilities"][str(i)]) for i in range(len(SPEED))]))
    return out


def _auc(pos, neg):
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else None


def score(preds):
    vis = [p for p in preds if p["visible"]]
    hid = [p for p in preds if not p["visible"]]
    third = lambda b: "left" if b > 18.3 else ("right" if b < -18.3 else "centre")  # noqa: E731  image thirds
    where_truth = [third(p["bearing_deg"]) if p["visible"] else "not visible" for p in preds]
    where_acc = float(np.mean([p["where"] == w for p, w in zip(preds, where_truth)]))
    where_prior = max(np.mean([w == k for w in where_truth]) for k in set(where_truth))
    st_true = [steer_level(p["bearing_deg"]) for p in vis]
    st_pred = [int(np.argmax(p["steer_probs"])) for p in vis]
    sp_true = [speed_level(p["range_m"]) for p in vis]
    sp_pred = [int(np.argmax(p["speed_probs"])) for p in vis]
    side = [p for p in vis if abs(p["bearing_deg"]) > 7]
    # steer's expected level: below 2 means turn left; the rover left (+ bearing) should give < 2
    side_acc = float(np.mean([(p["steer"] < 2) == (p["bearing_deg"] > 0) for p in side])) if side else None
    maj = lambda xs: max(np.mean([x == k for x in xs]) for k in set(xs)) if xs else None  # noqa: E731
    return {
        "frames": len(preds), "visible_frames": len(vis),
        "visible_auc": _auc([p["p_visible"] for p in vis], [p["p_visible"] for p in hid]),
        "where_acc": where_acc, "where_majority_baseline": float(where_prior),
        "where_answer_counts": {k: sum(p["where"] == k for p in preds) for k in ("left", "centre", "right", "not visible")},
        "steer_acc": float(np.mean(np.array(st_true) == np.array(st_pred))) if vis else None,
        "steer_majority_baseline": maj(st_true),
        "steer_spearman_vs_bearing": _spearman([-p["bearing_deg"] for p in vis], [p["steer"] for p in vis]),
        "steer_side_acc": side_acc, "steer_side_n": len(side),
        "speed_acc": float(np.mean(np.array(sp_true) == np.array(sp_pred))) if vis else None,
        "speed_majority_baseline": maj(sp_true),
        "speed_spearman_vs_range": _spearman([p["range_m"] for p in vis], [p["speed"] for p in vis]),
        "mean_steer_probs": np.mean([p["steer_probs"] for p in vis], axis=0).round(3).tolist() if vis else None,
        "mean_speed_probs": np.mean([p["speed_probs"] for p in vis], axis=0).round(3).tolist() if vis else None,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["collect"])
    p.add_argument("--out", default="/tmp/probe")
    p.add_argument("--seconds", type=float, default=60.0)
    a = p.parse_args()
    collect(a.out, seconds=a.seconds)
