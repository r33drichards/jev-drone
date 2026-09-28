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
