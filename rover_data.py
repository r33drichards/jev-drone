"""Training data for Laya to see the rover: labelled onboard frames, from the simulator.

Every example's label comes from the simulator, not from a person or a model:

  strip crops   the frame's rover band (probe.STRIP_BAND) cut into 5 vertical strips; one noul
                per strip, probe.STRIP_Q, true when the rover's segmentation pixels fall in it
  whole frame   probe.questions(): visible (noul), where (choice), steer (score, soft target
                spread between the two levels either side of the true bearing), speed (score)

The questions are exactly the ones probe.py asks, so a fine-tuned checkpoint is scored by the
same probe on held-out frames. Frames come from oracle flights: each snapshot keeps the real
view and `jitter_views` views with the camera turned by a random +-jitter_deg, since pursuit
otherwise keeps the rover dead centre.
"""
import io, json, os
import numpy as np
import probe

N_STRIPS = 5
STEER_CENTRES = [27.0, 13.5, 0.0, -13.5, -27.0]     # bearing (deg, + left) at each steer level


def steer_target(b):
    """A soft target over the 5 steer levels: linear between the two level centres either side of b."""
    b = float(np.clip(b, STEER_CENTRES[-1], STEER_CENTRES[0]))
    t = [0.0] * 5
    for k in range(4):
        hi, lo = STEER_CENTRES[k], STEER_CENTRES[k + 1]
        if lo <= b <= hi:
            w = (b - lo) / (hi - lo)
            t[k], t[k + 1] = w, 1.0 - w
            return t
    t[2] = 1.0
    return t


def soft_target(v, centres):
    """A soft target over score levels whose centres are monotone (either direction): linear between the
    two centres either side of v, all weight on the end level outside the range."""
    c = list(centres)
    lo_first = c[0] < c[-1]
    t = [0.0] * len(c)
    if (v <= c[0]) == lo_first:
        t[0] = 1.0
        return t
    if (v >= c[-1]) == lo_first:
        t[-1] = 1.0
        return t
    for k in range(len(c) - 1):
        a, b = c[k], c[k + 1]
        if min(a, b) <= v <= max(a, b):
            w = (v - b) / (a - b)
            t[k], t[k + 1] = w, 1.0 - w
            return t
    return t


def v2_records(v1_recs, image_prefix="../drone_rover/"):
    """drone_rover_v2 from drone_rover's records: strips, visible and where unchanged; steer -> steer7
    and speed -> range8 (probe.questions_v2), soft targets from the stored true bearing and range.
    Images are the v1 set's, referenced relative to the v2 directory."""
    qs = probe.questions_v2()
    out = []
    for r in v1_recs:
        r = dict(r, image=image_prefix + r["image"])
        kind = r["id"].rsplit("-", 1)[-1]
        if kind == "steer":
            t = soft_target(r["bearing_deg"], probe.STEER7_CENTRES)
            r.update(id=r["id"][:-5] + "steer7", question=qs["steer7"], label=int(np.argmax(t)), target=t)
        elif kind == "speed":
            t = soft_target(r["range_m"], probe.RANGE8_CENTRES)
            r.update(id=r["id"][:-5] + "range8", question=qs["range8"], label=int(np.argmax(t)), target=t)
        out.append(r)
    return out


def collect_flight(course, seed, seconds=60.0, every_s=0.5, jitter_views=2, jitter_deg=45.0):
    """Fly the oracle and return frames: {"jpeg", "strips": [jpeg]*5, labels}. The colour frame is rendered
    only for the views kept (not 15 times a second), with the eye's own segmentation for the labels."""
    import mujoco, run, flight
    from PIL import Image
    rng = np.random.default_rng(10_000 + 97 * seed + len(course))
    frames = []
    state = {"r": None, "last": -1e9}
    orig_look = flight.Eye.look

    def view(self, data, pos, yaw):
        if state["r"] is None:
            state["r"] = mujoco.Renderer(self.m, 384, 512)
        self._aim(pos, yaw)
        state["r"].update_scene(data, self.cam)
        rgb = state["r"].render()
        self.seg.update_scene(data, self.cam)
        seg = np.isin(self.seg.render()[:, :, 0], self.target_ids)          # 72 x 96
        return rgb, seg

    def look(self, data, pos, yaw, t):
        sc = orig_look(self, data, pos, yaw, t)
        if t - state["last"] < every_s - 1e-6:
            return sc
        state["last"] = t
        rover = data.mocap_pos[self.m.body("rover").mocapid[0]].copy()
        offsets = [0.0] + list(rng.uniform(-jitter_deg, jitter_deg, jitter_views))
        for k, off in enumerate(offsets):
            vy = yaw + np.deg2rad(off)
            rgb, seg = view(self, data, pos, vy)
            cam = np.asarray(pos) + self.NOSE_OFFSET_M * np.array([np.cos(vy), np.sin(vy), 0])
            d = rover - cam
            fwd, left = np.cos(vy) * d[0] + np.sin(vy) * d[1], -np.sin(vy) * d[0] + np.cos(vy) * d[1]
            h, w = seg.shape
            y0, y1 = int(probe.STRIP_BAND[0] * h), int(probe.STRIP_BAND[1] * h)
            strip_px = [int(seg[y0:y1, int(i * w / N_STRIPS):int((i + 1) * w / N_STRIPS)].sum())
                        for i in range(N_STRIPS)]
            img = Image.fromarray(rgb)
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
            crops = []
            for c in _strips_native(img):
                b2 = io.BytesIO(); c.save(b2, "JPEG", quality=90); crops.append(b2.getvalue())
            frames.append({"course": course, "seed": seed, "t": round(t, 2), "view": k, "yaw_offset_deg": round(off, 1),
                           "jpeg": buf.getvalue(), "strips": crops, "strip_pixels": strip_px,
                           "pixels": int(seg.sum()), "visible": bool(seg.sum() >= 3),
                           "bearing_deg": float(np.rad2deg(np.arctan2(left, fwd))), "range_m": float(np.hypot(fwd, left))})
        return sc

    flight.Eye.look = look
    try:
        backend = "const:climb" if course == "classic" else "const:oracle"
        run.episode(seed, seconds, use_jev=True, backend=backend, course=course, realtime=False)
    finally:
        flight.Eye.look = orig_look
    return frames


def _strips_native(img):
    """probe.strips without the upscale: the processor squashes every image to 512x512 anyway."""
    w, h = img.size
    y0, y1 = int(probe.STRIP_BAND[0] * h), int(probe.STRIP_BAND[1] * h)
    return [img.crop((int(k * w / N_STRIPS), y0, int((k + 1) * w / N_STRIPS), y1)) for k in range(N_STRIPS)]


def records(frames, image_dir, rel_prefix="images"):
    """Write the frames' images under image_dir and return the jsonl records for them."""
    os.makedirs(image_dir, exist_ok=True)
    qs = probe.questions()
    recs = []
    for f in frames:
        stem = "%s-%d-%06.2f-%d" % (f["course"], f["seed"], f["t"], f["view"])
        open(os.path.join(image_dir, stem + ".jpg"), "wb").write(f["jpeg"])
        full = "%s/%s.jpg" % (rel_prefix, stem)
        meta = {k: f[k] for k in ("course", "seed", "t", "view", "yaw_offset_deg", "visible", "bearing_deg", "range_m")}
        for i, c in enumerate(f["strips"]):
            open(os.path.join(image_dir, "%s-s%d.jpg" % (stem, i)), "wb").write(c)
            recs.append({"id": "%s-s%d" % (stem, i), "image": "%s/%s-s%d.jpg" % (rel_prefix, stem, i),
                         "question": probe.STRIP_Q, "label": int(f["strip_pixels"][i] >= 1), **meta})
        recs.append({"id": stem + "-visible", "image": full, "question": qs["visible"], "label": int(f["visible"]), **meta})
        if f["visible"]:
            x = probe.bearing_to_x(f["bearing_deg"])
            where = 0 if x < 1 / 3 else (2 if x > 2 / 3 else 1)
        else:
            where = 3
        recs.append({"id": stem + "-where", "image": full, "question": qs["where"], "label": where, **meta})
        if f["visible"]:
            recs.append({"id": stem + "-steer", "image": full, "question": qs["steer"],
                         "label": probe.steer_level(f["bearing_deg"]), "target": steer_target(f["bearing_deg"]), **meta})
            recs.append({"id": stem + "-speed", "image": full, "question": qs["speed"],
                         "label": probe.speed_level(f["range_m"]), **meta})
    return recs


# ---------------------------------------------------------------------------------------------------------
# v3: reacquisition (object permanence) and tactics, probe.questions_v3()
# ---------------------------------------------------------------------------------------------------------
# Each record is the natural onboard frame (512x384, true heading: reappear and maneuver are relative to it)
# plus `state_text`, JSON of probe.V3_CONTEXT_KEYS as the flight knows them from its own segmentation.
#
#   maneuver      every snapshot: the course oracle at the drone position (classic_oracle on world.xml);
#                 hold_course is subsampled to <= 4x climb per split when the set is written
#   occluded      rover not visible (seg pixels < 3): true iff its true bearing from the camera is within the
#                 horizontal half-FOV (atan(probe.TAN_H), ~62 deg) and its horizontal range < 20 m
#   reappear      bearing of rover_pose(t + REAPPEAR_HORIZON_S) from the camera, relative to the current
#                 nose: |b| > 90 behind (wins), else |b| <= 20 ahead, b > 20 left, b < -20 right
#   reappear_eta  first s in 0, 0.25, ..., 15 s at which the segment from the CURRENT camera position to the
#                 rover at rover_pose(t + s) (centre, or mast top) is clear of geometry (mj_ray, the drone and
#                 the rover's current body excluded); level from REAPPEAR_ETA_EDGES, never clear -> last level
# When the rover IS visible, 1 snapshot in `visible_every` also gets occluded=false and reappear.
ETA_CENTRES = [1.0, 3.0, 7.0, 13.0]   # s; midpoints fall exactly on REAPPEAR_ETA_EDGES (2, 5, 10), so the
                                      # soft target's argmax is always the edge level
CLASSIC_BEAMS_X = (19.0, 44.0)        # world.xml beam0 / beam1
OCCLUDED_MAX_RANGE_M = 20.0
MAST_TOP_DZ = 0.8                     # rover centre z 0.2 -> mast top ~1.05 m
V3_COURSES = {"pockets": "const:oracle", "mixed": "const:oracle", "classic": "const:climb"}
V3_SECONDS = {"pockets": 90.0, "mixed": 90.0, "no-climb": 90.0, "classic": 70.0}


def classic_oracle(pos):
    """world.xml has no courses.Course: climb from 5 m before each low beam to 0.6 m past it."""
    x = float(pos[0])
    return "climb" if any(bx - 5.0 <= x <= bx + 0.6 for bx in CLASSIC_BEAMS_X) else "hold_course"


def reappear_answer(b):
    """probe.REAPPEAR key for a bearing (deg, + left) relative to the nose; behind takes precedence."""
    if abs(b) > 90.0:
        return "behind"
    if abs(b) <= 20.0:
        return "ahead"
    return "left" if b > 0 else "right"


def eta_level(eta_s):
    """REAPPEAR_ETA level; None (not clear within the search horizon) is the last level."""
    return len(probe.REAPPEAR_ETA_EDGES) if eta_s is None else int(sum(eta_s > e for e in probe.REAPPEAR_ETA_EDGES))


def _rel(cam, yaw, p):
    """(bearing deg + left, horizontal range m) of point p from the camera at heading yaw."""
    d = np.asarray(p, dtype=float) - cam
    fwd, left = np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]
    return float(np.rad2deg(np.arctan2(left, fwd))), float(np.hypot(fwd, left))


def _wrap_deg(a):
    return float((a + 180.0) % 360.0 - 180.0)


def collect_flight_v3(course, seed, seconds=None, every_s=0.5, lost_every_s=0.25, eta_max_s=15.0, eta_step_s=0.25):
    """Fly (oracle tactics; const:climb on classic) and return v3 snapshots, every `every_s` and every
    `lost_every_s` while the rover is out of the eye's segmentation. Each: the natural frame as JPEG, the
    flight's context (probe.V3_CONTEXT_KEYS) and every v3 label (see the block comment above)."""
    import mujoco, run, flight, courses
    from PIL import Image
    seconds = seconds or V3_SECONDS.get(course, 90.0)
    if course == "classic":
        rover_at, oracle = run.rover_pose, classic_oracle
    else:
        c = courses.make(course, seed)
        rover_at, oracle = c.rover_pose, c.oracle
    half_fov = float(np.rad2deg(np.arctan(probe.TAN_H)))
    frames = []
    st = {"r": None, "last": -1e9, "seen": None, "d2": None}
    orig_look = flight.Eye.look

    def clear(m, d2, a, b, x2):
        x = mujoco.mj_ray(m, d2, a, b - a, None, 1, x2, np.array([-1], dtype=np.int32))
        return x < 0 or x >= 1.0 - 1e-6

    def look(self, data, pos, yaw, t):
        sc = orig_look(self, data, pos, yaw, t)
        tg = sc["target"]
        vis = bool(tg["visible"])
        if vis:
            st["seen"] = (t, float(tg["bearing_deg"]), float(tg["range_m"]), yaw)
        if t - st["last"] < (every_s if vis else lost_every_s) - 1e-6:
            return sc
        st["last"] = t
        m = self.m
        if st["r"] is None:
            st["r"] = mujoco.Renderer(m, 384, 512)
            st["d2"] = mujoco.MjData(m)
            st["x2"] = m.body("x2").id
            st["rmid"] = m.body("rover").mocapid[0]
        self._aim(pos, yaw)
        st["r"].update_scene(data, self.cam)
        buf = io.BytesIO()
        Image.fromarray(st["r"].render()).save(buf, "JPEG", quality=90)
        # a private copy of the world for the rays, the rover's body parked out of the way
        d2 = st["d2"]
        d2.qpos[:], d2.mocap_pos[:], d2.mocap_quat[:] = data.qpos, data.mocap_pos, data.mocap_quat
        d2.mocap_pos[st["rmid"]] = [0.0, 0.0, -50.0]
        mujoco.mj_kinematics(m, d2)
        cam = np.asarray(pos, dtype=float) + self.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        rover = data.mocap_pos[st["rmid"]].copy()
        b_now, r_now = _rel(cam, yaw, rover)
        b3, r3 = _rel(cam, yaw, rover_at(t + probe.REAPPEAR_HORIZON_S))
        eta = None
        for s in np.arange(0.0, eta_max_s + 1e-9, eta_step_s):
            p = np.asarray(rover_at(t + s), dtype=float)
            if clear(m, d2, cam, p, st["x2"]) or clear(m, d2, cam, p + [0, 0, MAST_TOP_DZ], st["x2"]):
                eta = float(s)
                break
        seen = st["seen"]
        ctx = {"unseen_for_s": tg["unseen_for_s"],
               "last_seen_bearing_deg": None if seen is None else round(_wrap_deg(seen[1] - np.rad2deg(yaw - seen[3])), 1),
               "last_seen_range_m": None if seen is None else round(seen[2], 2)}
        in_fov = abs(b_now) <= half_fov and r_now < OCCLUDED_MAX_RANGE_M
        frames.append({"course": course, "seed": seed, "t": round(t, 2), "jpeg": buf.getvalue(),
                       "visible": vis, "pixels": int(tg["pixels"]), "bearing_deg": b_now, "range_m": r_now,
                       "pos": [round(float(v), 2) for v in pos], "yaw_deg": round(float(np.rad2deg(yaw)), 1),
                       "context": ctx, "maneuver": oracle(pos), "in_fov": bool(in_fov),
                       "occluded": bool(in_fov and not vis),
                       "reappear": reappear_answer(b3), "reappear_bearing_deg": b3, "reappear_range_m": r3,
                       "eta_s": eta, "los_now": eta == 0.0})
        return sc

    flight.Eye.look = look
    try:
        run.episode(seed, seconds, use_jev=True, backend=V3_COURSES.get(course, "const:oracle"), course=course,
                    realtime=False)
    finally:
        flight.Eye.look = orig_look
    return frames


def v3_frame_labels(f):
    """The truth fields a v3 record / probe row carries besides its image."""
    return {"course": f["course"], "seed": f["seed"], "t": f["t"], "visible": f["visible"], "pixels": f["pixels"],
            "bearing_deg": round(f["bearing_deg"], 2), "range_m": round(f["range_m"], 2),
            "state_text": json.dumps(f["context"]), "maneuver": f["maneuver"], "occluded": f["occluded"],
            "reappear": f["reappear"], "reappear_bearing_deg": round(f["reappear_bearing_deg"], 2),
            "eta_s": f["eta_s"], "eta_level": eta_level(f["eta_s"])}


def records_v3(frames, image_dir, rel_prefix="images", visible_every=4):
    """Write the frames' images under image_dir and return their v3 records (probe.questions_v3). Every
    record carries its truth fields (v3_frame_labels) too; `maneuver` records are balanced later
    (balance_maneuver), across a whole split."""
    from tactics import MANEUVERS
    os.makedirs(image_dir, exist_ok=True)
    qs = probe.questions_v3()
    man, rea = list(MANEUVERS), list(probe.REAPPEAR)
    recs, n_vis = [], 0
    for f in frames:
        stem = "v3-%s-%d-%06.2f" % (f["course"], f["seed"], f["t"])
        open(os.path.join(image_dir, stem + ".jpg"), "wb").write(f["jpeg"])
        base = dict(image="%s/%s.jpg" % (rel_prefix, stem), **v3_frame_labels(f))
        recs.append(dict(base, id=stem + "-maneuver", question=qs["maneuver"], label=man.index(f["maneuver"])))
        extra = not f["visible"]
        if f["visible"]:
            extra = n_vis % visible_every == 0
            n_vis += 1
        if extra:
            recs.append(dict(base, id=stem + "-occluded", question=qs["occluded"], label=int(f["occluded"])))
            recs.append(dict(base, id=stem + "-reappear", question=qs["reappear"], label=rea.index(f["reappear"])))
        if not f["visible"]:
            eta = probe.REAPPEAR_ETA_EDGES[-1] + 5.0 if f["eta_s"] is None else f["eta_s"]
            recs.append(dict(base, id=stem + "-reappear_eta", question=qs["reappear_eta"],
                             label=eta_level(f["eta_s"]), target=soft_target(eta, ETA_CENTRES)))
    return recs


def balance_maneuver(recs, ratio=4.0, seed=0):
    """Keep every non-maneuver record and every maneuver record except hold_course, which is subsampled
    (seeded) to at most `ratio` x the climb records."""
    man = [r for r in recs if r["id"].endswith("-maneuver")]
    climb = sum(r["maneuver"] == "climb" for r in man)
    hold = [i for i, r in enumerate(recs) if r["id"].endswith("-maneuver") and r["maneuver"] == "hold_course"]
    cap = int(round(ratio * climb))
    drop = set()
    if len(hold) > cap:
        rng = np.random.default_rng(seed)
        drop = set(rng.choice(hold, len(hold) - cap, replace=False).tolist())
    return [r for i, r in enumerate(recs) if i not in drop]


def evaluate_v3(agent, frames, rows):
    """probe.questions_v3()'s v3 questions on each frame + its state_text context. `frames`: name -> JPEG."""
    from PIL import Image
    from tactics import MANEUVERS
    q3 = probe.questions_v3()
    qs = {k: q3[k] for k in ("occluded", "reappear", "reappear_eta", "maneuver")}
    out = []
    for r in rows:
        img = Image.open(io.BytesIO(frames[r["frame"]])).convert("RGB")
        a = agent.predict({"image": img, "context": r["state_text"]}, qs)["answers"]
        out.append(dict(r, p_occluded=float(a["occluded"]["noul"]),
                        reappear_probs={k: float(a["reappear"]["probabilities"][k]) for k in probe.REAPPEAR},
                        reappear_eta_probs=[float(a["reappear_eta"]["probabilities"][str(i)])
                                            for i in range(len(probe.REAPPEAR_ETA))],
                        maneuver_probs={k: float(a["maneuver"]["probabilities"][k]) for k in MANEUVERS}))
    return out


def score_v3(preds):
    """Score evaluate_v3 predictions against the v3 truth fields.

    Each pred: the row's truth (visible, occluded, reappear, eta_s / eta_level, maneuver) plus p_occluded,
    reappear_probs {key: p}, reappear_eta_probs [p per level], maneuver_probs {key: p}.
      occluded      AUC of p_occluded, true vs false, over lost frames (and over all frames, visible = false)
      reappear      argmax accuracy vs the majority answer, confusion (truth -> predicted), on lost frames and
                    on visible ones separately
      reappear_eta  on lost frames: argmax level accuracy vs majority, within-one, Spearman of the expected
                    time (probability-weighted ETA_CENTRES) against the true ETA (None -> 20 s)
      maneuver      argmax accuracy vs majority, climb recall / precision, confusion, on every frame
    """
    lost = [p for p in preds if not p["visible"]]
    vis = [p for p in preds if p["visible"]]
    maj = lambda xs: float(max(np.mean([x == k for x in xs]) for k in set(xs))) if xs else None  # noqa: E731

    def argmax(d):
        return max(d, key=d.get)

    def conf(ps, truth_key, prob_key, keys):
        return {t: {k: int(sum(p[truth_key] == t and argmax(p[prob_key]) == k for p in ps)) for k in keys}
                for t in keys if any(p[truth_key] == t for p in ps)}

    def reappear(ps):
        if not ps:
            return None
        tr = [p["reappear"] for p in ps]
        return {"n": len(ps), "acc": float(np.mean([argmax(p["reappear_probs"]) == t for p, t in zip(ps, tr)])),
                "majority_baseline": maj(tr), "truth_counts": {k: tr.count(k) for k in probe.REAPPEAR},
                "confusion": conf(ps, "reappear", "reappear_probs", list(probe.REAPPEAR))}

    out = {"frames": len(preds), "lost_frames": len(lost),
           "occluded_auc_lost": probe._auc([p["p_occluded"] for p in lost if p["occluded"]],
                                           [p["p_occluded"] for p in lost if not p["occluded"]]),
           "occluded_auc_all": probe._auc([p["p_occluded"] for p in preds if p["occluded"]],
                                          [p["p_occluded"] for p in preds if not p["occluded"]]),
           "occluded_true_lost": int(sum(p["occluded"] for p in lost)),
           "occluded_acc_lost": float(np.mean([(p["p_occluded"] > 0.5) == p["occluded"] for p in lost])) if lost else None,
           "occluded_majority_lost": maj([p["occluded"] for p in lost]),
           "reappear_lost": reappear(lost), "reappear_visible": reappear(vis)}
    if lost:
        lv = np.array([p["eta_level"] for p in lost])
        pl = np.array([int(np.argmax(p["reappear_eta_probs"])) for p in lost])
        ev = np.array([float(np.dot(np.asarray(p["reappear_eta_probs"]) / sum(p["reappear_eta_probs"]), ETA_CENTRES))
                       for p in lost])
        te = np.array([20.0 if p["eta_s"] is None else p["eta_s"] for p in lost])
        out["reappear_eta"] = {"n": len(lost), "acc": float(np.mean(lv == pl)), "majority_baseline": maj(lv.tolist()),
                               "within_one": float(np.mean(np.abs(lv - pl) <= 1)),
                               "spearman_expected_vs_true_s": probe._spearman(te, ev),
                               "mae_s_capped15": float(np.mean(np.abs(np.minimum(te, 15) - ev))),
                               "truth_counts": [int((lv == k).sum()) for k in range(len(probe.REAPPEAR_ETA))],
                               "pred_counts": [int((pl == k).sum()) for k in range(len(probe.REAPPEAR_ETA))]}
    if preds:
        keys = list(preds[0]["maneuver_probs"])
        tr = [p["maneuver"] for p in preds]
        pr = [argmax(p["maneuver_probs"]) for p in preds]
        tp = sum(t == "climb" and q == "climb" for t, q in zip(tr, pr))
        out["maneuver"] = {"n": len(preds), "acc": float(np.mean([t == q for t, q in zip(tr, pr)])),
                           "majority_baseline": maj(tr),
                           "climb_recall": tp / max(1, tr.count("climb")) if "climb" in tr else None,
                           "climb_precision": tp / max(1, pr.count("climb")) if "climb" in pr else None,
                           "climb_auc": probe._auc([p["maneuver_probs"]["climb"] for p in preds if p["maneuver"] == "climb"],
                                                   [p["maneuver_probs"]["climb"] for p in preds if p["maneuver"] != "climb"]),
                           "confusion": conf(preds, "maneuver", "maneuver_probs", keys)}
    return out
