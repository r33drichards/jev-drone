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
V3_SECONDS = {"pockets": 90.0, "mixed": 90.0, "no-climb": 90.0, "classic": 70.0, "tactics": 110.0, "town": 120.0}


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


def collect_flight_v3(course, seed, seconds=None, every_s=0.5, lost_every_s=0.25, eta_max_s=15.0, eta_step_s=0.25,
                      every_at=None):
    """Fly (oracle tactics; const:climb on classic) and return v3 snapshots, every `every_s` and every
    `lost_every_s` while the rover is out of the eye's segmentation. Each: the natural frame as JPEG, the
    flight's context (probe.V3_CONTEXT_KEYS) and every v3 label (see the block comment above).
    `every_at`: a function of the drone position -> snapshot interval (s), replacing every_s / lost_every_s
    (drone_rover_tac: dense near beams and pocket walls, sparse elsewhere)."""
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
        gap = every_at(pos) if every_at is not None else (every_s if vis else lost_every_s)
        if t - st["last"] < gap - 1e-6:
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
    (balance_maneuver), across a whole split. A frame with a "stem" key (v3b) is written under that name, and
    one whose maneuver is None (a v3b rotated view) gets no maneuver record."""
    from tactics import MANEUVERS
    os.makedirs(image_dir, exist_ok=True)
    qs = probe.questions_v3()
    man, rea = list(MANEUVERS), list(probe.REAPPEAR)
    recs, n_vis = [], 0
    for f in frames:
        stem = f.get("stem") or "v3-%s-%d-%06.2f" % (f["course"], f["seed"], f["t"])
        open(os.path.join(image_dir, stem + ".jpg"), "wb").write(f["jpeg"])
        base = dict(image="%s/%s.jpg" % (rel_prefix, stem), **v3_frame_labels(f))
        if f["maneuver"] is not None:
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
    preds = [p for p in preds if p.get("maneuver") is not None]   # v3b rotated views have no maneuver truth
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
        if any("station_kind" in p for p in preds):
            out["maneuver_tac"] = score_tac(preds)
    return out


# --- drone_rover_v3b: more "rover lost / behind" examples, same records and labels as v3 ---------------------
# Two sources, both labelled exactly as collect_flight_v3 labels a frame (occluded, reappear, reappear_eta from
# the rendered view's own camera position and heading; context from the eye's segmentation sightings):
#   rotated views   oracle flights (V3_COURSES), a snapshot every 0.5 s rendered at `n_views` headings yaw + d,
#                   the aircraft as if turned in place to that heading (camera at pos + nose offset along it). d
#                   is drawn so the rover's position REAPPEAR_HORIZON_S ahead lands in a class drawn from
#                   V3B_VIEW_CLASS_P (behind / left / right well represented). Context: rendered view shows the
#                   rover -> unseen 0, last seen = that view's segmentation bearing / range; else a lag u ~
#                   U(V3B_LAG_S) is drawn and the latest real (natural-view) sighting at or before t - u is
#                   re-expressed at the rotated heading (bearing_then + yaw_then - yaw_view, as collect_flight_v3
#                   and laya_pursuit.LastSeen re-express it at the live yaw), unseen_for_s = t - t_sighting.
#                   No maneuver (the tactical answer belongs to the real heading).
#   failure flights run.episode's model-free pursuit stand-ins (V3B_FAILURE) with oracle tactics, which lose the
#                   rover the way Laya does; natural heading, every 0.25 s from 1 s before the eye loses the rover
#                   until the eye sees it again or 10 s pass (losses shorter than 0.25 s skipped), all four v3
#                   questions (maneuver = the course oracle at that position).
V3B_VIEW_CLASS_P = {"behind": 0.35, "left": 0.25, "right": 0.25, "ahead": 0.15}
V3B_LAG_S = (0.07, 4.0)
V3B_MAX_UNSEEN_S = 15.0          # rotated views skip snapshots where the natural view lost the rover longer ago
V3B_FAILURE = {
    "sim6": dict(pursuit="sim", pursuit_noise_deg=6.0, pursuit_delay_s=0.2),
    "sim10": dict(pursuit="sim", pursuit_noise_deg=10.0, pursuit_delay_s=0.4),
    "simr": dict(pursuit="sim-pursuit", pursuit_noise_deg=6.0, pursuit_delay_s=0.2,
                 pursuit_range={"range_noise_m": 1.5, "range_tau_s": 1.0}),
}


def _view_bearing(rng):
    """A bearing (deg, + left) for the rover's future position in the rotated view, class ~ V3B_VIEW_CLASS_P."""
    ks = list(V3B_VIEW_CLASS_P)
    c = ks[int(rng.choice(len(ks), p=[V3B_VIEW_CLASS_P[k] for k in ks]))]
    if c == "ahead":
        return float(rng.uniform(-20.0, 20.0))
    if c == "left":
        return float(rng.uniform(20.0, 90.0))
    if c == "right":
        return float(rng.uniform(-90.0, -20.0))
    return float(rng.uniform(90.0, 180.0) * rng.choice([-1.0, 1.0]))


class _V3Labeller:
    """Renders and labels a view of a recorded sim state (qpos, mocap_pos, mocap_quat) at any heading, as
    collect_flight_v3 does live: RGB JPEG, the eye's segmentation target test (>= 3 pixels), the v3 labels."""

    def __init__(self, m, rover_at, oracle, eta_max_s=15.0, eta_step_s=0.25):
        import mujoco, flight
        self.mj, self.m, self.rover_at, self.oracle = mujoco, m, rover_at, oracle
        self.eye = flight.Eye(m)
        self.rgb = mujoco.Renderer(m, 384, 512)
        self.d3, self.d2 = mujoco.MjData(m), mujoco.MjData(m)
        self.x2 = m.body("x2").id
        self.rmid = m.body("rover").mocapid[0]
        self.etas = np.arange(0.0, eta_max_s + 1e-9, eta_step_s)
        self.half_fov = float(np.rad2deg(np.arctan(probe.TAN_H)))
        self.count_ids = None                               # geom ids whose seg pixels view() also counts

    def set_state(self, st):
        mj = self.mj
        for d in (self.d3, self.d2):
            d.qpos[:], d.mocap_pos[:], d.mocap_quat[:] = st["qpos"], st["mocap_pos"], st["mocap_quat"]
        mj.mj_forward(self.m, self.d3)                      # lights and cameras too, for the render
        self.d2.mocap_pos[self.rmid] = [0.0, 0.0, -50.0]    # the rays' copy: rover body parked out of the way
        mj.mj_kinematics(self.m, self.d2)

    def _clear(self, a, b):
        x = self.mj.mj_ray(self.m, self.d2, a, b - a, None, 1, self.x2, np.array([-1], dtype=np.int32))
        return x < 0 or x >= 1.0 - 1e-6

    def view(self, pos, yaw, t):
        """-> dict: jpeg, seg_visible, seg_pixels, seg_bearing_deg, seg_range_m, and the v3 truth fields."""
        from PIL import Image
        eye = self.eye
        eye._aim(pos, yaw)
        eye.seg.update_scene(self.d3, eye.cam)
        seg = eye.seg.render()[:, :, 0]
        mask = np.isin(seg, eye.target_ids)
        px = int(mask.sum())
        count_px = None if self.count_ids is None else int(np.isin(seg, self.count_ids).sum())
        sb = sr = None
        if px >= 3:
            eye.depth.update_scene(self.d3, eye.cam)
            z = np.clip(eye.depth.render(), 0.0, eye.MAX_RANGE)
            sb = float(eye._bearing(float(np.nonzero(mask)[1].mean())))
            sr = round(float(np.median(z[mask])), 2)
        self.rgb.update_scene(self.d3, eye.cam)
        buf = io.BytesIO()
        Image.fromarray(self.rgb.render()).save(buf, "JPEG", quality=90)
        cam = np.asarray(pos, dtype=float) + eye.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        b_now, r_now = _rel(cam, yaw, self.d3.mocap_pos[self.rmid].copy())
        b3, r3 = _rel(cam, yaw, self.rover_at(t + probe.REAPPEAR_HORIZON_S))
        eta = None
        for s in self.etas:
            p = np.asarray(self.rover_at(t + s), dtype=float)
            if self._clear(cam, p) or self._clear(cam, p + [0, 0, MAST_TOP_DZ]):
                eta = float(s)
                break
        in_fov = abs(b_now) <= self.half_fov and r_now < OCCLUDED_MAX_RANGE_M
        return {"jpeg": buf.getvalue(), "seg_visible": px >= 3, "seg_pixels": px, "seg_bearing_deg": sb,
                "seg_range_m": sr, "bearing_deg": b_now, "range_m": r_now, "in_fov": bool(in_fov),
                "reappear": reappear_answer(b3), "reappear_bearing_deg": b3, "reappear_range_m": r3,
                "eta_s": eta, "los_now": eta == 0.0, "count_px": count_px}


def _fly_record(course, seed, every_s, backend, seconds=None, **episode_kw):
    """Fly run.episode, recording at the eye's frames: every sighting (t, seg bearing, seg range, yaw), the
    loss intervals of the eye's segmentation, and a snapshot (sim state, pose, the eye's target, the v3
    context as collect_flight_v3 builds it) every `every_s`. -> (model, rover_at, oracle, snaps, sightings, losses)"""
    import run, flight, courses
    seconds = seconds or V3_SECONDS.get(course, 90.0)
    if course == "classic":
        rover_at, oracle = run.rover_pose, classic_oracle
    else:
        c = courses.make(course, seed)
        rover_at, oracle = c.rover_pose, c.oracle
    snaps, sightings, losses = [], [], []
    st = {"last": -1e9, "m": None, "vis": None}
    orig_look = flight.Eye.look

    def look(self, data, pos, yaw, t):
        sc = orig_look(self, data, pos, yaw, t)
        tg = sc["target"]
        vis = bool(tg["visible"])
        st["m"] = self.m
        if vis:
            sightings.append((t, float(tg["bearing_deg"]), float(tg["range_m"]), float(yaw)))
            if losses and losses[-1][1] is None:
                losses[-1][1] = t
        elif st["vis"] and sightings:
            losses.append([t, None])
        st["vis"] = vis
        if t - st["last"] >= every_s - 1e-6:
            st["last"] = t
            seen = sightings[-1] if sightings else None
            ctx = {"unseen_for_s": tg["unseen_for_s"],
                   "last_seen_bearing_deg": None if seen is None else round(_wrap_deg(seen[1] - np.rad2deg(yaw - seen[3])), 1),
                   "last_seen_range_m": None if seen is None else round(seen[2], 2)}
            snaps.append({"t": round(t, 2), "t_raw": t, "pos": np.asarray(pos, dtype=float).copy(), "yaw": float(yaw),
                          "state": {"qpos": data.qpos.copy(), "mocap_pos": data.mocap_pos.copy(),
                                    "mocap_quat": data.mocap_quat.copy()},
                          "visible": vis, "pixels": int(tg["pixels"]), "context": ctx, "n_sightings": len(sightings)})
        return sc

    flight.Eye.look = look
    try:
        run.episode(seed, seconds, use_jev=True, backend=backend, course=course, realtime=False, **episode_kw)
    finally:
        flight.Eye.look = orig_look
    return st["m"], rover_at, oracle, snaps, sightings, losses


def _frame(course, seed, sn, lab, stem, **extra):
    """A v3 frame dict (collect_flight_v3's keys) from a snapshot and a labeller view."""
    return dict({"course": course, "seed": seed, "t": sn["t"], "jpeg": lab["jpeg"], "stem": stem,
                 "bearing_deg": lab["bearing_deg"], "range_m": lab["range_m"],
                 "pos": [round(float(v), 2) for v in sn["pos"]], "in_fov": lab["in_fov"],
                 "reappear": lab["reappear"], "reappear_bearing_deg": lab["reappear_bearing_deg"],
                 "reappear_range_m": lab["reappear_range_m"], "eta_s": lab["eta_s"], "los_now": lab["los_now"]}, **extra)


def collect_flight_v3b(course, seed, kind="rotated", n_views=2, every_s=None):
    """v3b frames from one flight (see the block comment above). kind: "rotated" (oracle flight, rotated views)
    or a V3B_FAILURE key (failure flight, natural views around losses). Frames have collect_flight_v3's keys
    plus "stem", "view" ("rotated" / "natural"), "yaw_offset_deg", "source"."""
    import zlib
    rng = np.random.default_rng([seed, zlib.crc32(("%s/%s" % (course, kind)).encode())])
    if kind == "rotated":
        m, rover_at, oracle, snaps, sightings, _ = _fly_record(course, seed, every_s or 0.5,
                                                               V3_COURSES.get(course, "const:oracle"))
    else:
        m, rover_at, oracle, snaps, sightings, losses = _fly_record(course, seed, every_s or 0.25, "const:oracle",
                                                                    **V3B_FAILURE[kind])
    lab = _V3Labeller(m, rover_at, oracle)
    sight_t = np.array([s[0] for s in sightings]) if sightings else np.zeros(0)
    frames = []
    if kind == "rotated":
        for sn in snaps:
            if (sn["context"]["unseen_for_s"] or 0.0) > V3B_MAX_UNSEEN_S:
                continue            # an aircraft left stranded far behind: not V3B_MAX_UNSEEN_S s more of it
            lab.set_state(sn["state"])
            pos, yaw, t = sn["pos"], sn["yaw"], sn["t_raw"]
            cam = pos + lab.eye.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0.0])
            b3_nat, _ = _rel(cam, yaw, rover_at(t + probe.REAPPEAR_HORIZON_S))
            for k in range(n_views):
                off = _wrap_deg(b3_nat - _view_bearing(rng))
                yv = yaw + np.deg2rad(off)
                v = lab.view(pos, yv, t)
                if v["seg_visible"]:
                    ctx = {"unseen_for_s": 0.0, "last_seen_bearing_deg": round(v["seg_bearing_deg"], 1),
                           "last_seen_range_m": v["seg_range_m"]}
                else:
                    u = float(rng.uniform(*V3B_LAG_S))
                    i = int(np.searchsorted(sight_t[:sn["n_sightings"]], t - u, side="right")) - 1
                    if i < 0:
                        ctx = {"unseen_for_s": None, "last_seen_bearing_deg": None, "last_seen_range_m": None}
                    else:
                        ts, bs, rs, ys = sightings[i]
                        ctx = {"unseen_for_s": round(t - ts, 2),
                               "last_seen_bearing_deg": round(_wrap_deg(bs - np.rad2deg(yv - ys)), 1),
                               "last_seen_range_m": round(rs, 2)}
                frames.append(_frame(course, seed, sn, v, "v3b-rot-%s-%d-%06.2f-%d" % (course, seed, sn["t"], k),
                                     visible=v["seg_visible"], pixels=v["seg_pixels"],
                                     yaw_deg=round(float(np.rad2deg(yv)), 1), context=ctx, maneuver=None,
                                     occluded=bool(v["in_fov"] and not v["seg_visible"]),
                                     view="rotated", yaw_offset_deg=round(off, 1), source="rotated"))
        return frames
    windows = [(a - 1.0, min(b if b is not None else 1e9, a + 10.0)) for a, b in losses
               if (b if b is not None else 1e9) - a >= 0.25]
    for sn in snaps:
        t = sn["t_raw"]
        if not any(lo - 1e-6 <= t <= hi + 1e-6 for lo, hi in windows):
            continue
        lab.set_state(sn["state"])
        v = lab.view(sn["pos"], sn["yaw"], t)
        frames.append(_frame(course, seed, sn, v, "v3b-%s-%s-%d-%06.2f" % (kind, course, seed, sn["t"]),
                             visible=sn["visible"], pixels=sn["pixels"], yaw_deg=round(float(np.rad2deg(sn["yaw"])), 1),
                             context=sn["context"], maneuver=oracle(sn["pos"]),
                             occluded=bool(v["in_fov"] and not sn["visible"]), view="natural", yaw_offset_deg=0.0,
                             source=kind, seg_agrees=bool(v["seg_visible"] == sn["visible"])))
    return frames


def v3b_test_row(f, name):
    """A probe-format row (rover_test_frames_v3's, plus view / yaw_offset_deg / source)."""
    return dict(frame=name, **v3_frame_labels(f), view=f["view"], yaw_offset_deg=f["yaw_offset_deg"],
                source=f["source"])


def label_counts(recs):
    """{question: {label name: n}} over v3 records."""
    import collections
    from tactics import MANEUVERS
    names = {"reappear": list(probe.REAPPEAR), "maneuver": list(MANEUVERS), "occluded": ["false", "true"],
             "reappear_eta": [str(i) for i in range(len(probe.REAPPEAR_ETA))],
             "visible": ["false", "true"], "where": ["left", "centre", "right", "not visible"],
             "steer7": [str(i) for i in range(len(probe.STEER7))], "range8": [str(i) for i in range(len(probe.RANGE8))]}
    out = collections.defaultdict(collections.Counter)
    for r in recs:
        q = r["id"].rsplit("-", 1)[1]
        out[q][names[q][r["label"]]] += 1
    return {q: dict(sorted(c.items())) for q, c in sorted(out.items())}


def balance_reappear(recs, max_share=None, seed=0):
    """Subsample reappear records of an over-represented answer (seeded) to at most max_share of the reappear
    records, e.g. {"ahead": 0.4}; every other record is kept."""
    max_share = {"ahead": 0.4} if max_share is None else max_share
    rea = list(probe.REAPPEAR)
    idx = {k: [i for i, r in enumerate(recs) if r["id"].endswith("-reappear") and rea[r["label"]] == k] for k in rea}
    n = sum(len(v) for v in idx.values())
    rng, drop = np.random.default_rng(seed), set()
    for k, share in max_share.items():
        rest = n - len(idx[k])
        cap = int(share / (1.0 - share) * rest)
        if len(idx[k]) > cap:
            drop |= set(rng.choice(idx[k], len(idx[k]) - cap, replace=False).tolist())
    return [r for i, r in enumerate(recs) if i not in drop]


# ---------------------------------------------------------------------------------------------------------
# drone_rover_tac: `maneuver` at beams (climb) against pocket front walls (hold_course), probe.questions_v3()
# ---------------------------------------------------------------------------------------------------------
# v3.1 could not tell a real beam from a pocket's low front wall (results/laya-steer/README.md): P(climb) rarely
# passed 0.2 and it climbed into pockets. This set is maneuver-only, oracle flights (collect_flight_v3 with the
# natural heading, labels exactly v3's: maneuver = the course oracle, state_text = the v3 context), sampled every
# TAC_DENSE_S while the drone is within TAC_BEFORE_M before a beam or a pocket's front wall (TAC_AFTER_M past it),
# every TAC_SPARSE_S elsewhere. Each frame is tagged with the station it is approaching:
#   station_kind   the first station (in x order) with -TAC_AFTER_M <= station_x - drone_x <= TAC_BEFORE_M
#                  (beam / pocket / decoy), else "none"; station_dx_m = station_x - drone_x (m, + ahead; for
#                  "none" the next station ahead, None past the last)
# Beam frames 5-10 m out are hold_course (the oracle climbs from 5 m before to 0.6 m past), so the "beam" tag
# carries both answers; every "pocket" frame is hold_course: the hard negatives.
# balance_tac keeps climb : pocket-hold : other-hold ~ 1 : 1 : 1 per split.
TAC_BEFORE_M, TAC_AFTER_M = 10.0, 1.0
TAC_DENSE_S, TAC_SPARSE_S = 0.25, 1.0
TAC_DENSE_KINDS = ("beam", "pocket")


def tac_stations(course, seed):
    """[(kind, x)] in x order: world.xml's two beams on classic, else the courses.Course stations."""
    if course == "classic":
        return [("beam", float(x)) for x in CLASSIC_BEAMS_X]
    import courses
    return sorted([(k, float(x)) for k, x, _ in courses.make(course, seed).stations], key=lambda s: s[1])


def station_at(stations, x, before=TAC_BEFORE_M, after=TAC_AFTER_M):
    """(station_kind, station_dx_m) for a drone at x (see the block comment above)."""
    for kind, sx in stations:
        dx = sx - float(x)
        if -after <= dx <= before:
            return kind, round(dx, 2)
    ahead = [sx - float(x) for _, sx in stations if sx - float(x) > before]
    return "none", (round(min(ahead), 2) if ahead else None)


def collect_flight_tac(course, seed, seconds=None):
    """Oracle flight (const:climb on classic), natural heading, v3 frames sampled densely near beams / pocket
    walls, each with station_kind and station_dx_m."""
    st = tac_stations(course, seed)

    def every_at(pos):
        kind, _ = station_at(st, pos[0])
        return TAC_DENSE_S if kind in TAC_DENSE_KINDS else TAC_SPARSE_S

    frames = collect_flight_v3(course, seed, seconds, every_at=every_at)
    for f in frames:
        f["station_kind"], f["station_dx_m"] = station_at(st, f["pos"][0])
    return frames


def tac_frame_labels(f):
    return dict(v3_frame_labels(f), station_kind=f["station_kind"], station_dx_m=f["station_dx_m"],
                pos=f["pos"], yaw_deg=f["yaw_deg"])


def tac_group(r):
    """climb / pocket_hold / other_hold."""
    if r["maneuver"] == "climb":
        return "climb"
    return "pocket_hold" if r["station_kind"] == "pocket" else "other_hold"


def records_tac(frames, image_dir, rel_prefix="images"):
    """Write the frames' images and return one `maneuver` record per frame (probe.questions_v3()["maneuver"],
    the frame + state_text), with the tac truth fields."""
    from tactics import MANEUVERS
    os.makedirs(image_dir, exist_ok=True)
    q = probe.questions_v3()["maneuver"]
    man = list(MANEUVERS)
    recs = []
    for f in frames:
        stem = "tac-%s-%d-%06.2f" % (f["course"], f["seed"], f["t"])
        open(os.path.join(image_dir, stem + ".jpg"), "wb").write(f["jpeg"])
        recs.append(dict(tac_frame_labels(f), id=stem + "-maneuver", image="%s/%s.jpg" % (rel_prefix, stem),
                         question=q, label=man.index(f["maneuver"])))
    return recs


def balance_tac(recs, ratio=(1.0, 1.0, 1.0), seed=0):
    """Subsample (seeded) so climb : pocket_hold : other_hold ~ ratio, never dropping a climb record unless
    pocket_hold runs short (then the climbs are capped to it). -> (records, counts before, counts after)."""
    groups = {"climb": [], "pocket_hold": [], "other_hold": []}
    for i, r in enumerate(recs):
        groups[tac_group(r)].append(i)
    before = {k: len(v) for k, v in groups.items()}
    unit = min(before["climb"] / ratio[0], before["pocket_hold"] / ratio[1])
    if unit <= 0:
        return list(recs), before, dict(before)
    rng, keep = np.random.default_rng(seed), set()
    for (k, idx), w in zip(groups.items(), ratio):
        cap = int(round(unit * w))
        keep |= set(idx) if len(idx) <= cap else set(rng.choice(idx, cap, replace=False).tolist())
    out = [r for i, r in enumerate(recs) if i in keep]
    after = {k: sum(tac_group(r) == k for r in out) for k in groups}
    return out, before, after


def tac_counts(recs):
    """{station_kind: {maneuver: n}} and {group: n}."""
    import collections
    by = collections.defaultdict(collections.Counter)
    for r in recs:
        by[r["station_kind"]][r["maneuver"]] += 1
    return {"by_station": {k: dict(v) for k, v in sorted(by.items())},
            "groups": dict(collections.Counter(tac_group(r) for r in recs))}


def score_tac(preds, thresholds=None):
    """maneuver per station_kind, from evaluate_v3 predictions on rows with station_kind (tac_frame_labels).

      by_station      per kind: n, climb truth count, argmax-climb rate, mean P(climb) for truth climb / hold
      climb_recall_beams        argmax climb recall on beam frames whose truth is climb
      false_climb_pocket        argmax climb rate on pocket frames (all hold_course: the hard negatives)
      beam_vs_pocket_auc        P(climb): truth-climb beam frames against pocket frames
      best_threshold            climb iff P(climb) >= thr, thr chosen by F1 over every frame; its recall,
                                precision, recall at beams and false-climb rate at pocket fronts
      at_thresholds             the same numbers at a few fixed thresholds (0.12 is tactics.V3_CLIMB_P)
    """
    preds = [p for p in preds if p.get("maneuver") is not None and p.get("station_kind") is not None]
    if not preds:
        return None
    pc = np.array([p["maneuver_probs"]["climb"] for p in preds])
    am = np.array([max(p["maneuver_probs"], key=p["maneuver_probs"].get) == "climb" for p in preds])
    truth = np.array([p["maneuver"] == "climb" for p in preds])
    kind = np.array([p["station_kind"] for p in preds])
    beam_c, pocket = (kind == "beam") & truth, kind == "pocket"

    def at(pred):
        tp = int((pred & truth).sum())
        rec = tp / max(1, int(truth.sum()))
        prec = tp / int(pred.sum()) if pred.sum() else None
        f1 = 2 * rec * prec / (rec + prec) if prec else 0.0
        return {"recall": rec, "precision": prec, "f1": f1, "n_pred_climb": int(pred.sum()),
                "climb_recall_beams": float(pred[beam_c].mean()) if beam_c.any() else None,
                "false_climb_pocket": float(pred[pocket].mean()) if pocket.any() else None,
                "false_climb_other": float(pred[~truth & ~pocket].mean()) if (~truth & ~pocket).any() else None}

    by = {}
    for k in sorted(set(kind.tolist())):
        m = kind == k
        by[k] = {"n": int(m.sum()), "truth_climb": int((m & truth).sum()), "argmax_climb_rate": float(am[m].mean()),
                 "mean_p_climb_truth_climb": float(pc[m & truth].mean()) if (m & truth).any() else None,
                 "mean_p_climb_truth_hold": float(pc[m & ~truth].mean()) if (m & ~truth).any() else None}
    cands = np.unique(np.concatenate([pc, [1.01]]))
    fits = [(at(pc >= c)["f1"], c) for c in cands]
    f1, thr = max(fits)
    out = {"n": len(preds), "truth_climb": int(truth.sum()), "by_station": by,
           "argmax": at(am), "climb_recall_beams": at(am)["climb_recall_beams"],
           "false_climb_pocket": at(am)["false_climb_pocket"],
           "beam_vs_pocket_auc": probe._auc(pc[beam_c].tolist(), pc[pocket].tolist()),
           "climb_auc": probe._auc(pc[truth].tolist(), pc[~truth].tolist()),
           "best_threshold": dict(at(pc >= thr), threshold=float(thr)),
           "at_thresholds": {"%g" % c: at(pc >= c) for c in (thresholds or (0.12, 0.2, 0.3, 0.5))}}
    return out


# ---------------------------------------------------------------------------------------------------------
# drone_rover_town: perception (v2) and reacquisition (v3) on the town map (town.py)
# ---------------------------------------------------------------------------------------------------------
# Flights: const:oracle tactics (always hold_course on the town), code pursuit ("code") or one of the model-free
# stand-ins that lose the rover (TOWN_FAILURE keys), TOWN_SECONDS each. The sim state is recorded
# every 0.25 s (_fly_record); a snapshot is kept every TOWN_EVERY_S, and every one while the eye has lost the
# rover. Each kept snapshot is rendered (_V3Labeller: every label from the rendered view's own camera) as
#   natural   the true heading: perception (visible, where, steer7 / range8 with soft targets, exactly as
#             v2_records builds them) and, when the rover is out of view or for 1 in TOWN_VISIBLE_EVERY visible
#             snapshots, occluded / reappear (+ reappear_eta when out of view) with the flight's v3 context
#   jitter    `jitter_views` views turned by U(+-TOWN_JITTER_DEG): perception only
#   rotated   with probability rot_p, one v3b-style turned-in-place view whose rover-in-3-s bearing is drawn from
#             V3B_VIEW_CLASS_P ("behind" well represented), context re-expressed at that heading as v3b does:
#             visible, and occluded / reappear / reappear_eta under the same rule as natural views
#   scenery   with probability scenery_p, a view aimed (+-15 deg) at a random rover-coloured scenery geom
#             (TOWN_RED_MATERIALS: shed, porch roof, planter / crates, doors, the ochre west roof) 3-20 m away:
#             perception only. With the rover out of the segmentation these are the hard negatives for visible.
# Every view records scenery_px, the rover-coloured scenery pixels in the eye's 96x72 segmentation.
TOWN_EVERY_S, TOWN_LOST_EVERY_S = 0.5, 0.25
TOWN_JITTER_DEG = 45.0
TOWN_VISIBLE_EVERY = 4
TOWN_RED_MATERIALS = ("shed", "roof", "crate", "door", "roofW")
TOWN_SCENERY_MIN_PX = 20        # a view "shows red scenery" at this many segmentation pixels (of 6912)
TOWN_MAX_UNSEEN_S = 15.0        # snapshots whose natural view lost the rover longer ago are skipped (a stranded
                                # aircraft far behind: sim10 in the town loses the rover early and never recovers)
# failure stand-ins for the town: V3B_FAILURE's, plus sim8, between sim6 (never loses the rover on the town:
# 99.9% in view, seeds 10-12) and sim10 (2-3% in view, stranded or crashed by ~15 s): 22-45% in view, seeds 10-11
TOWN_FAILURE = dict(V3B_FAILURE, sim8=dict(pursuit="sim", pursuit_noise_deg=8.0, pursuit_delay_s=0.3))


def _town_scenery(m):
    """(geom ids, [(name, x, y)]) of the rover-coloured scenery."""
    mats = {m.material(i).name: i for i in range(m.nmat)}
    want = [mats[k] for k in TOWN_RED_MATERIALS if k in mats]
    ids = np.nonzero(np.isin(m.geom_matid, want))[0]
    return ids, [(m.geom(int(i)).name, float(m.geom_pos[i][0]), float(m.geom_pos[i][1])) for i in ids]


def _where(visible, b):
    if not visible:
        return 3
    x = probe.bearing_to_x(b)
    return 0 if x < 1 / 3 else (2 if x > 2 / 3 else 1)


def collect_flight_town(seed, kind="code", seconds=None, jitter_views=2, rot_p=0.5, scenery_p=0.5,
                        course="town"):
    """drone_rover_town frames from one flight (see the block comment above). kind: "code" or a TOWN_FAILURE
    key. Frames carry v3_frame_labels' keys plus stem, view, yaw_offset_deg, source, scenery_px, and
    perception / reacq flags (which questions records_town asks of the view)."""
    import zlib
    rng = np.random.default_rng([seed, zlib.crc32(("town/%s" % kind).encode())])
    kw = {} if kind == "code" else TOWN_FAILURE[kind]
    m, rover_at, oracle, snaps, sightings, losses = _fly_record(course, seed, TOWN_LOST_EVERY_S, "const:oracle",
                                                                seconds=seconds, **kw)
    lab = _V3Labeller(m, rover_at, oracle)
    sc_ids, sc_geoms = _town_scenery(m)
    lab.count_ids = sc_ids
    sight_t = np.array([s[0] for s in sightings]) if sightings else np.zeros(0)
    frames, last, n_vis = [], -1e9, {"natural": 0, "rotated": 0}

    def reacq_flag(view, vis):
        if not vis:
            return True
        n_vis[view] += 1
        return (n_vis[view] - 1) % TOWN_VISIBLE_EVERY == 0

    def add(sn, v, view, off, yv, ctx, maneuver, perception, reacq_ok):
        vis = v["seg_visible"]
        stem = "town-%s-%d-%06.2f-%s%s" % (kind, seed, sn["t"], view, "" if view == "natural" else "%+.0f" % off)
        f = _frame(course, seed, sn, v, stem, visible=vis, pixels=v["seg_pixels"],
                   yaw_deg=round(float(np.rad2deg(yv)), 1), context=ctx, maneuver=maneuver,
                   occluded=bool(v["in_fov"] and not vis), view=view, yaw_offset_deg=round(float(off), 1),
                   source=kind, scenery_px=int(v["count_px"]), perception=perception,
                   reacq=bool(reacq_ok and reacq_flag(view, vis)))
        frames.append(f)
        return f

    for sn in snaps:
        t = sn["t_raw"]
        if sn["visible"] and t - last < TOWN_EVERY_S - 1e-6:
            continue
        if (sn["context"]["unseen_for_s"] or 0.0) > TOWN_MAX_UNSEEN_S:
            continue
        last = t
        lab.set_state(sn["state"])
        pos, yaw = sn["pos"], sn["yaw"]
        v = lab.view(pos, yaw, t)
        f = add(sn, v, "natural", 0.0, yaw, sn["context"], oracle(pos), True, True)
        f["seg_agrees"] = bool(v["seg_visible"] == sn["visible"])
        for _ in range(jitter_views):
            off = float(rng.uniform(-TOWN_JITTER_DEG, TOWN_JITTER_DEG))
            yv = yaw + np.deg2rad(off)
            add(sn, lab.view(pos, yv, t), "jitter", off, yv, sn["context"], None, True, False)
        cam = pos + lab.eye.NOSE_OFFSET_M * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        if rng.random() < rot_p:
            b3_nat, _ = _rel(cam, yaw, rover_at(t + probe.REAPPEAR_HORIZON_S))
            off = _wrap_deg(b3_nat - _view_bearing(rng))
            yv = yaw + np.deg2rad(off)
            rv = lab.view(pos, yv, t)
            if rv["seg_visible"]:
                ctx = {"unseen_for_s": 0.0, "last_seen_bearing_deg": round(rv["seg_bearing_deg"], 1),
                       "last_seen_range_m": rv["seg_range_m"]}
            else:
                u = float(rng.uniform(*V3B_LAG_S))
                i = int(np.searchsorted(sight_t[:sn["n_sightings"]], t - u, side="right")) - 1
                if i < 0:
                    ctx = {"unseen_for_s": None, "last_seen_bearing_deg": None, "last_seen_range_m": None}
                else:
                    ts, bs, rs, ys = sightings[i]
                    ctx = {"unseen_for_s": round(t - ts, 2),
                           "last_seen_bearing_deg": round(_wrap_deg(bs - np.rad2deg(yv - ys)), 1),
                           "last_seen_range_m": round(rs, 2)}
            add(sn, rv, "rotated", off, yv, ctx, None, False, True)
        if rng.random() < scenery_p:
            near = [(n, x, y) for n, x, y in sc_geoms if 3.0 <= np.hypot(x - cam[0], y - cam[1]) <= 20.0]
            if near:
                n, x, y = near[int(rng.integers(len(near)))]
                yv = float(np.arctan2(y - cam[1], x - cam[0])) + np.deg2rad(rng.uniform(-15.0, 15.0))
                off = _wrap_deg(np.rad2deg(yv - yaw))
                sv = lab.view(pos, yv, t)
                if sv["count_px"] >= TOWN_SCENERY_MIN_PX:
                    add(sn, sv, "scenery", off, yv, sn["context"], None, True, False)["scenery_target"] = n
    return frames


def town_frame_labels(f):
    """The truth fields of a town record / probe row: v3's plus view, yaw_offset_deg, source, scenery_px."""
    return dict(v3_frame_labels(f), view=f["view"], yaw_offset_deg=f["yaw_offset_deg"], source=f["source"],
                scenery_px=f["scenery_px"], perception=f["perception"], reacq=f["reacq"])


def records_town(frames, image_dir, rel_prefix="images"):
    """Write the frames' images and return their records: probe.questions_v2's visible / where / steer7 /
    range8 (soft targets, as v2_records) on perception views, probe.questions_v3's occluded / reappear /
    reappear_eta (as records_v3) on reacq views."""
    os.makedirs(image_dir, exist_ok=True)
    q2, q3 = probe.questions_v2(), probe.questions_v3()
    rea = list(probe.REAPPEAR)
    recs = []
    for f in frames:
        stem = f["stem"]
        open(os.path.join(image_dir, stem + ".jpg"), "wb").write(f["jpeg"])
        base = dict(image="%s/%s.jpg" % (rel_prefix, stem), **town_frame_labels(f))
        vis = f["visible"]
        if f["perception"] or f["view"] == "rotated":
            recs.append(dict(base, id=stem + "-visible", question=q2["visible"], label=int(vis)))
        if f["perception"]:
            recs.append(dict(base, id=stem + "-where", question=q2["where"], label=_where(vis, f["bearing_deg"])))
            if vis:
                t = soft_target(f["bearing_deg"], probe.STEER7_CENTRES)
                recs.append(dict(base, id=stem + "-steer7", question=q2["steer7"], label=int(np.argmax(t)), target=t))
                t = soft_target(f["range_m"], probe.RANGE8_CENTRES)
                recs.append(dict(base, id=stem + "-range8", question=q2["range8"], label=int(np.argmax(t)), target=t))
        if f["reacq"]:
            recs.append(dict(base, id=stem + "-occluded", question=q3["occluded"], label=int(f["occluded"])))
            recs.append(dict(base, id=stem + "-reappear", question=q3["reappear"], label=rea.index(f["reappear"])))
            if not vis:
                eta = probe.REAPPEAR_ETA_EDGES[-1] + 5.0 if f["eta_s"] is None else f["eta_s"]
                recs.append(dict(base, id=stem + "-reappear_eta", question=q3["reappear_eta"],
                                 label=eta_level(f["eta_s"]), target=soft_target(eta, ETA_CENTRES)))
    return recs


def town_counts(recs):
    """label_counts per view type, and the visible hard-negative count (red scenery in view, no rover)."""
    import collections
    out = {"labels": label_counts(recs), "by_view": {}}
    for v in sorted({r["view"] for r in recs}):
        out["by_view"][v] = label_counts([r for r in recs if r["view"] == v])
    vis = [r for r in recs if r["id"].endswith("-visible")]
    out["visible_hard_negatives"] = sum(1 for r in vis if not r["visible"] and r["scenery_px"] >= TOWN_SCENERY_MIN_PX)
    out["visible_with_scenery_and_rover"] = sum(1 for r in vis if r["visible"] and r["scenery_px"] >= TOWN_SCENERY_MIN_PX)
    out["frames_by_view"] = dict(collections.Counter(r["view"] for r in vis))
    return out
