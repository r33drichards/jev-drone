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


# the onboard camera: 110 deg vertical fov, 4:3, so tan(horizontal half-fov) = tan(55 deg) * 4/3 (~62 deg)
TAN_H = float(np.tan(np.deg2rad(55.0)) * 96 / 72)


def bearing_to_x(b):
    """Horizontal image position of a bearing, 0 = left edge, 1 = right edge (+ bearing is left)."""
    return 0.5 - 0.5 * np.tan(np.deg2rad(b)) / TAN_H


def x_to_bearing(x):
    return float(np.rad2deg(np.arctan((0.5 - x) * 2 * TAN_H)))


# v2 questions (drone-rover-v2): wider steering and a finer range, both trained with soft targets
# spread between the two levels either side of the true value (rover_data.soft_target). v1 topped
# out at +-27 deg and read large bearings far too small; its 4 speed bands gave ~1-1.7 m range error
# in flight, too coarse to set forward speed.
STEER7_CENTRES = [60.0, 35.0, 15.0, 0.0, -15.0, -35.0, -60.0]      # deg, + left; level 0 = hardest left
STEER7 = ["hard left: the rover is at the far left edge of the image, about 60 degrees off",
          "left: the rover is well left of centre, about 35 degrees",
          "slightly left: the rover is a little left of centre, about 15 degrees",
          "straight: the rover is at the centre of the image",
          "slightly right: the rover is a little right of centre, about 15 degrees",
          "right: the rover is well right of centre, about 35 degrees",
          "hard right: the rover is at the far right edge of the image, about 60 degrees off"]
RANGE8_CENTRES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.75, 6.0, 8.5]           # m
RANGE8 = ["about 2 metres away, very close", "about 2.5 metres away", "about 3 metres away",
          "about 3.5 metres away", "about 4 metres away", "about 4.75 metres away",
          "about 6 metres away", "about 8.5 metres or more away, far"]


def questions_v2():
    qs = questions()
    ctx = qs["visible"]["instructions"].split("Is the red rover")[0]
    return {"visible": qs["visible"], "where": qs["where"],
            "steer7": {"type": "score", "instructions": ctx + "How far off the nose is the rover, and which way?",
                       "criteria": STEER7},
            "range8": {"type": "score", "instructions": ctx + "How far away is the red rover?", "criteria": RANGE8}}


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


def collect(out, courses=("classic", "mixed"), seeds=(0, 1, 2), seconds=60.0, every_s=0.5, yaw_jitter_deg=0.0):
    """`yaw_jitter_deg`: render each saved view with the camera turned by a random offset up to this
    much, so the rover lands anywhere across the frame. Pursuit keeps it centred otherwise: with no
    jitter, 98% of visible rovers fall within +-12 deg, which cannot test left from right."""
    import mujoco, run, flight
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    rows = []
    rng = np.random.default_rng(7)
    for course in courses:
        for seed in seeds:
            snaps = []
            orig_look = flight.Eye.look

            def look(self, data, pos, yaw, t, _snaps=snaps):
                sc = orig_look(self, data, pos, yaw, t)
                if not _snaps or t - _snaps[-1]["t"] >= every_s - 1e-6:
                    visible, pixels, rgb = bool(sc["target"]["visible"]), int(sc["target"]["pixels"]), self.last_rgb
                    if yaw_jitter_deg:
                        yaw = yaw + np.deg2rad(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
                        self._aim(pos, yaw)
                        self.rgb.update_scene(data, self.cam)
                        rgb = self.rgb.render()
                        self.seg.update_scene(data, self.cam)
                        pixels = int(np.isin(self.seg.render()[:, :, 0], self.target_ids).sum())
                        visible = pixels >= 3
                    rover = data.mocap_pos[self.m.body("rover").mocapid[0]].copy()
                    cam = np.asarray(pos) + self.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0])
                    d = rover - cam
                    fwd, left = (np.cos(yaw) * d[0] + np.sin(yaw) * d[1],
                                 -np.sin(yaw) * d[0] + np.cos(yaw) * d[1])
                    _snaps.append({"t": t, "rgb": rgb.copy(), "visible": visible, "pixels": pixels,
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


STRIP_Q = {"type": "noul", "instructions":
           "Crop of a quadrotor's forward camera. Is there a small red ground rover (a red box with a "
           "thin red mast) in this image?"}
STRIP_BAND = (0.30, 0.90)   # the rows the rover can occupy: it is on the floor, 2-8 m ahead, camera 12 deg down


def strips(img, n=5, band=STRIP_BAND, size=512):
    """Cut the band of rows the rover can be in into n vertical strips, each upscaled so its long side is
    `size` px: the rover goes from ~25 px in a 512 px frame to ~5x that, more than one image token wide."""
    w, h = img.size
    y0, y1 = int(band[0] * h), int(band[1] * h)
    out = []
    for k in range(n):
        x0, x1 = int(k * w / n), int((k + 1) * w / n)
        c = img.crop((x0, y0, x1, y1))
        f = size / max(c.size)
        out.append(c.resize((max(1, int(c.size[0] * f)), max(1, int(c.size[1] * f)))))
    return out


def evaluate_strips(agent, frames, rows, n=5):
    """Ask one noul per strip: is the rover in this crop? (the detector-regions branch's 'map' shape)."""
    from PIL import Image
    q = {"rover": STRIP_Q}
    out = []
    for r in rows:
        img = Image.open(io.BytesIO(frames[r["frame"]])).convert("RGB")
        ps = [float(agent.predict({"image": c}, q)["answers"]["rover"]["noul"]) for c in strips(img, n)]
        out.append(dict(r, strip_probs=ps))
    return out


def score_strips(preds, n=5):
    vis = [p for p in preds if p["visible"]]
    hid = [p for p in preds if not p["visible"]]
    centres = (np.arange(n) + 0.5) / n
    true_strip = [min(n - 1, int(bearing_to_x(p["bearing_deg"]) * n)) for p in vis]
    pred_strip = [int(np.argmax(p["strip_probs"])) for p in vis]

    def est_x(p):  # probability-weighted strip centre
        w = np.asarray(p["strip_probs"])
        return float((w * centres).sum() / max(w.sum(), 1e-9))

    side = [p for p in vis if abs(p["bearing_deg"]) > 7]
    maj = max(np.mean([t == k for t in true_strip]) for k in set(true_strip))
    # per-crop detection: the rover's strip against every strip without it (other strips of the same
    # frame, and all strips of frames where it is out of view); independent of where rovers tend to be
    pos = [p["strip_probs"][t] for p, t in zip(vis, true_strip)]
    neg = [q for p, t in zip(vis, true_strip) for k, q in enumerate(p["strip_probs"]) if k != t]
    neg += [q for p in hid for q in p["strip_probs"]]
    return {
        "frames": len(preds), "visible_frames": len(vis), "strips": n,
        "crop_detection_auc": _auc(pos, neg),
        "crop_p_yes_rover_mean": float(np.mean(pos)) if pos else None,
        "crop_p_yes_empty_mean": float(np.mean(neg)) if neg else None,
        "visible_auc_max_strip": _auc([max(p["strip_probs"]) for p in vis], [max(p["strip_probs"]) for p in hid]),
        "strip_acc_argmax": float(np.mean(np.array(true_strip) == np.array(pred_strip))),
        "strip_majority_baseline": float(maj),
        "strip_within_one": float(np.mean(np.abs(np.array(true_strip) - np.array(pred_strip)) <= 1)),
        "side_acc_argmax": float(np.mean([(int(np.argmax(p["strip_probs"])) < n // 2) == (p["bearing_deg"] > 0)
                                          for p in side if int(np.argmax(p["strip_probs"])) != n // 2]))
        if side else None,
        "side_acc_weighted": float(np.mean([(est_x(p) < 0.5) == (p["bearing_deg"] > 0) for p in side])) if side else None,
        "side_n": len(side),
        "spearman_x": _spearman([bearing_to_x(p["bearing_deg"]) for p in vis], [est_x(p) for p in vis]),
        "bearing_mae_deg_weighted": float(np.mean([abs(x_to_bearing(est_x(p)) - p["bearing_deg"]) for p in vis])),
        "bearing_mae_deg_always_straight": float(np.mean([abs(p["bearing_deg"]) for p in vis])),
        "true_strip_counts": [int(sum(t == k for t in true_strip)) for k in range(n)],
        "pred_strip_counts": [int(sum(t == k for t in pred_strip)) for k in range(n)],
        "mean_strip_prob_visible": np.mean([p["strip_probs"] for p in vis], axis=0).round(3).tolist(),
        "mean_strip_prob_hidden": np.mean([p["strip_probs"] for p in hid], axis=0).round(3).tolist() if hid else None,
    }


def evaluate_v2(agent, frames, rows):
    """probe.questions_v2() on each whole frame: visible, where, steer7, range8."""
    from PIL import Image
    qs = questions_v2()
    out = []
    for r in rows:
        img = Image.open(io.BytesIO(frames[r["frame"]])).convert("RGB")
        a = agent.predict({"image": img}, qs)["answers"]
        out.append(dict(r, p_visible=float(a["visible"]["noul"]), where=a["where"]["choice"],
                        steer7_probs=[float(a["steer7"]["probabilities"][str(i)]) for i in range(len(STEER7))],
                        range8_probs=[float(a["range8"]["probabilities"][str(i)]) for i in range(len(RANGE8))]))
    return out


def score_v2(preds, sharpen=1.0):
    """Bearing (deg) and range (m) read as the probability-weighted level centre (probabilities
    raised to `sharpen` first), scored on frames with the rover in view; also within +-34 deg,
    the span pursuit's turn clip uses."""
    vis = [p for p in preds if p["visible"]]
    hid = [p for p in preds if not p["visible"]]

    def ev(ps, centres):
        q = np.asarray(ps) ** sharpen
        return float((q / q.sum() * np.asarray(centres)).sum())

    b = np.array([p["bearing_deg"] for p in vis])
    eb = np.array([ev(p["steer7_probs"], STEER7_CENTRES) for p in vis])
    rg = np.array([p["range_m"] for p in vis])
    er = np.array([ev(p["range8_probs"], RANGE8_CENTRES) for p in vis])
    inr = np.abs(b) <= 34
    side = np.abs(b) > 7
    cmd = lambda x: np.clip(1.15 * (x - 3.5) + 1.35, 0, 3.6)  # noqa: E731  run.Guidance's speed law
    third = lambda x: "left" if bearing_to_x(x) < 1 / 3 else ("right" if bearing_to_x(x) > 2 / 3 else "centre")  # noqa: E731
    wt = [third(p["bearing_deg"]) if p["visible"] else "not visible" for p in preds]
    return {"frames": len(preds), "visible_frames": len(vis), "sharpen": sharpen,
            "visible_auc": _auc([p["p_visible"] for p in vis], [p["p_visible"] for p in hid]),
            "where_acc": float(np.mean([p["where"] == w for p, w in zip(preds, wt)])),
            "bearing_mae_deg": float(np.mean(np.abs(eb - b))),
            "bearing_mae_deg_within34": float(np.mean(np.abs(eb - b)[inr])),
            "bearing_mae_deg_always_straight": float(np.mean(np.abs(b))),
            "bearing_side_acc": float(np.mean(np.sign(eb[side]) == np.sign(b[side]))),
            "bearing_spearman": _spearman(b, eb),
            "bearing_mean_bias_by_band": {"%d-%d" % (lo, hi): float(np.mean((eb - b)[m] * np.sign(b[m])))
                                          for lo, hi in ((0, 10), (10, 25), (25, 45), (45, 90))
                                          for m in [(np.abs(b) >= lo) & (np.abs(b) < hi)] if m.any()},
            "range_mae_m": float(np.mean(np.abs(er - rg))),
            "range_mae_m_constant_median": float(np.mean(np.abs(np.median(rg) - rg))),
            "range_bias_m": float(np.mean(er - rg)),
            "range_spearman": _spearman(rg, er),
            "speed_cmd_mae_mps": float(np.mean(np.abs(cmd(er) - cmd(rg))))}


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
    third = lambda b: "left" if bearing_to_x(b) < 1 / 3 else ("right" if bearing_to_x(b) > 2 / 3 else "centre")  # noqa: E731
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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--yaw-jitter", type=float, default=0.0, help="turn each saved view by up to this many degrees")
    a = p.parse_args()
    collect(a.out, seeds=a.seeds, seconds=a.seconds, yaw_jitter_deg=a.yaw_jitter)
