"""Real drone footage as a probe test set: does the simulator-trained perception transfer?

Evaluation only. UAV123 and VisDrone-SOT are research / non-commercial datasets; nothing here
trains on them, and only code and score summaries go into git. The footage is downloaded and
converted inside Modal CPU jobs straight onto the laya-datasets volume (modal_laya.py
realtest_* functions), never onto the local disk:

    modal run modal_laya.py::realtest_build         # download + convert -> /data/realtest/<set>/
    modal run modal_laya.py::realtest_score --model /ckpt/smolvlm/drone-rover-v2/last

Each converted set is /data/realtest/<set>/labels.jsonl + frames/*.jpg (long side 512 px), with
one row per frame in the probe.py shape (visible, bearing_deg, range proxy) plus the box itself,
so results can be read in pixel terms that do not depend on the assumed field of view.
"""
import io, json, math, os, re, zipfile, zlib
try:
    import numpy as np
except ImportError:      # the Modal client's own Python: only the tables below are used there
    np = None

# sources (see results/realtest/README.md for the terms). Both are read as remote zips with HTTP range
# requests (RemoteFile), so only the central directory and the members actually used are transferred:
# no whole-archive download, on the Modal worker or anywhere else.
SOURCES = {
    # UAV123 (Mueller, Smith, Ghanem, ECCV 2016), the official 10 fps release (4.7 GB zip: the same
    # sequences and boxes as the 30 fps one, every 3rd frame), from the Google Drive link on
    # https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset (no form or login).
    "uav123_10fps": {"url": "https://drive.usercontent.google.com/download?id=0B6sQMCU1i4NbZmFlQmJBVDlLRDg"
                            "&export=download&confirm=t&resourcekey=0--jsSKS1oGFidNhgMF75cSQ"},
    # VisDrone2019-SOT (Zhu et al., TPAMI 2021). The official Google Drive links in
    # https://github.com/VisDrone/VisDrone-Dataset answered "Quota exceeded" for every part (val,
    # test-dev, train 1/2) when this was built; the Baidu mirror needs an account. This is a
    # third-party re-upload of the full release on the Hugging Face Hub (72.7 GB zip, public, not gated).
    "visdrone_sot": {"url": "https://huggingface.co/datasets/huseyincavus/visdrone2019-sot/resolve/main/"
                            "visdrone2019-sot.zip"},
    "visdrone_sot_val_gdrive": {"url": "https://drive.usercontent.google.com/download?"
                                       "id=18SNAOlCJtApnG2m45ud-1e_OtGYill0D&export=download&confirm=t"},
    "visdrone_sot_testdev_gdrive": {"url": "https://drive.usercontent.google.com/download?"
                                           "id=1xCiHjU4JlR9QsYtiHYy2UUd3m6NthoBC&export=download&confirm=t"},
}

OUT_DIR = "/data/realtest"


class RemoteFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, for zipfile.ZipFile(RemoteFile(url))."""

    def __init__(self, url, block=4 << 20):
        import requests
        self.url, self.s, self.pos, self.block = url, requests.Session(), 0, block
        self.cache = {}
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=120, allow_redirects=True)
        if r.status_code != 206:
            raise RuntimeError("%s: no range support (HTTP %d, %s): %r" % (
                url, r.status_code, r.headers.get("content-type"), r.raw.read(300)))
        self.final = r.url
        self.size = int(r.headers["content-range"].split("/")[1])
        r.close()
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, a, b):
        import time
        for k in range(6):
            try:
                r = self.s.get(self.final, headers={"Range": "bytes=%d-%d" % (a, b)}, timeout=300)
                if r.status_code in (401, 403):   # a signed redirect expired: resolve it again
                    self.final = self.s.get(self.url, headers={"Range": "bytes=0-0"}, stream=True,
                                            timeout=120).url
                    raise RuntimeError("HTTP %d" % r.status_code)
                if r.status_code != 206:
                    raise RuntimeError("HTTP %d" % r.status_code)
                self.bytes_read += len(r.content)
                return r.content
            except Exception:
                if k == 5:
                    raise
                time.sleep(2 * (k + 1))

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > self.block:            # big reads (a whole member) go straight through
            out = self._get(self.pos, self.pos + n - 1)
        else:                         # small ones (headers, the central directory) through a block cache
            out = b""
            while len(out) < n:
                bi = (self.pos + len(out)) // self.block
                if bi not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    a = bi * self.block
                    self.cache[bi] = self._get(a, min(self.size, a + self.block) - 1)
                blk = self.cache[bi]
                o = self.pos + len(out) - bi * self.block
                out += blk[o:o + n - len(out)]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def open_zip(name):
    return zipfile.ZipFile(RemoteFile(SOURCES[name]["url"]))


# VisDrone-SOT does not name the target's class: labelled here by eye from each sequence's middle frame
# (val + test-dev, 46 sequences). Left out: riders (bicycle / motorbike), a tricycle cart, animals.
VISDRONE_CLASS = {
    "val": {"uav0000024_00000_s": "person", "uav0000053_00264_s": "person", "uav0000054_00000_s": "person",
            "uav0000086_00870_s": "person", "uav0000115_00606_s": "car", "uav0000245_00001_s": "car",
            "uav0000317_02945_s": "car"},
    "test-dev": {"uav0000021_00000_s": "person", "uav0000074_01656_s": "person", "uav0000074_04320_s": "person",
                 "uav0000074_04992_s": "person", "uav0000074_05712_s": "person", "uav0000074_06312_s": "person",
                 "uav0000074_11915_s": "person", "uav0000079_02568_s": "person", "uav0000088_00000_s": "person",
                 "uav0000116_00503_s": "car", "uav0000155_01201_s": "car", "uav0000164_00000_s": "car",
                 "uav0000180_00050_s": "car", "uav0000184_00625_s": "car", "uav0000207_00675_s": "car",
                 "uav0000208_00000_s": "car", "uav0000241_00001_s": "car", "uav0000242_02327_s": "car",
                 "uav0000242_05160_s": "truck", "uav0000294_00000_s": "car", "uav0000294_00069_s": "person",
                 "uav0000294_01449_s": "car", "uav0000324_00069_s": "truck", "uav0000340_01356_s": "car",
                 "uav0000353_00001_s": "car", "uav0000367_02761_s": "car", "uav0000367_04137_s": "car",
                 "uav0000368_03312_s": "truck", "uav0000368_03612_s": "car"},
}
UAV123_SEQ = re.compile(r"^(car|person|truck|bike)\d+(_\d+)?$")   # _s sequences are rendered (UE4): excluded
NOUN = {"car": "car", "truck": "truck", "person": "person", "bike": "cyclist"}

# Bearing needs a field of view and these cameras' are not published. ASSUMED: 70 deg horizontal for
# every full frame (a typical small-drone camera; the sim camera is ~124 deg). A crop's field of view
# follows from its width. Rank correlations and side accuracy do not depend on this guess; the
# pixel-offset metrics (dx = box centre x / width - 0.5) are reported alongside.
HFOV_DEG = 70.0
LONG_SIDE = 512
CROP_H = 0.6          # jitter crops: 4:3 windows this fraction of the frame height


def _box_row(box, W, H, tan_half):
    """Truth fields for a box (x, y, w, h in the frame's pixels), or None = target absent."""
    if box is None:
        return {"visible": False, "box": None, "x_c": None, "y_c": None, "dx": None, "box_frac": None,
                "bearing_deg": None, "range_proxy": None, "where_truth": "not visible"}
    x, y, w, h = box
    x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return _box_row(None, W, H, tan_half)
    xc, yc = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
    frac = math.sqrt((x1 - x0) * (y1 - y0) / (W * H))       # box side as a fraction of the frame's
    return {"visible": True, "box": [round(v, 1) for v in (x0, y0, x1 - x0, y1 - y0)],
            "x_c": round(xc, 4), "y_c": round(yc, 4), "dx": round(xc - 0.5, 4), "box_frac": round(frac, 5),
            "bearing_deg": round(math.degrees(math.atan((0.5 - xc) * 2 * tan_half)), 2),
            "range_proxy": round(1.0 / frac, 3),            # only its rank means anything
            "where_truth": "left" if xc < 1 / 3 else ("right" if xc > 2 / 3 else "centre")}


def _save(img, box, path):
    """Resize to LONG_SIDE, save JPEG; return the box scaled to the saved image."""
    W, H = img.size
    f = LONG_SIDE / max(W, H)
    im = img.resize((max(1, round(W * f)), max(1, round(H * f))), resample=3)
    im.save(path, quality=90)
    return im.size, (None if box is None else [v * f for v in box])


def _views(rng, img, box, k):
    """The full frame, plus (on some frames) a 4:3 jitter crop that puts the target at a random
    horizontal position, and a crop beside the target that leaves it out (a negative from the same
    scene; it may hold other objects of the same class). -> [(view, image, box in that image)]"""
    W, H = img.size
    out = [("full", img, box)]
    if box is None:
        return out
    x, y, w, h = box
    ch = CROP_H * H
    cw = min(W, ch * 4 / 3)
    if k % 2 == 0 and w < 0.7 * cw and h < 0.7 * ch:
        u, v = rng.uniform(0.08, 0.92), rng.uniform(0.3, 0.75)
        x0 = min(max(0.0, x + w / 2 - u * cw), W - cw)
        y0 = min(max(0.0, y + h / 2 - v * ch), H - ch)
        c = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        out.append(("crop", c, [x - int(x0), y - int(y0), w, h]))
    if k % 4 == 1:
        m = 0.1 * max(w, h)
        left, right = x - m, W - (x + w + m)
        side = max(left, right)
        nw = min(cw, side)
        if nw >= 0.25 * W:
            nh = nw * 3 / 4
            x0 = (left - nw) * rng.uniform(0, 1) if left >= right else x + w + m + (right - nw) * rng.uniform(0, 1)
            y0 = rng.uniform(0, H - nh)
            out.append(("crop-neg", img.crop((int(x0), int(y0), int(x0 + nw), int(y0 + nh))), None))
    return out


def _pick(n, every, cap, absent):
    """Frame indices: every `every`-th, thinned evenly to `cap`; plus absent (NaN) frames, every
    other one up to cap // 2, so the set has natural negatives (only rank-free metrics use them)."""
    idx = list(range(0, n, every))
    if len(idx) > cap:
        idx = [idx[int(i * len(idx) / cap)] for i in range(cap)]
    neg = [i for i in absent if i not in set(idx)][::2][:cap // 2]
    return sorted(set(idx) | set(neg))


def uav123_jobs():
    """[(seq, dir, start, end)] for the real car / person / truck / bike sequences in configSeqs.m."""
    z = open_zip("uav123_10fps")
    cfg = z.read("UAV123_10fps/configSeqs.m").decode("latin1")
    jobs = []
    for m in re.finditer(r"struct\('name','([^']+)','path','[^']*?\\(\w+)\\','startFrame',(\d+),'endFrame',(\d+)", cfg):
        name, d, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if UAV123_SEQ.match(name) and (name, d, a, b) not in jobs:
            jobs.append((name, d, a, b))
    return jobs


def convert_uav123(jobs, out, every=6, cap=24, seed=0):
    """Frames of UAV123 sub-sequences -> out/frames/*.jpg; returns their rows. NaN box = the target is
    fully occluded or out of view (UAV123 ReadMe) -> visible False."""
    from PIL import Image
    z = open_zip("uav123_10fps")
    root = "UAV123_10fps/"
    tan_half = math.tan(math.radians(HFOV_DEG / 2))
    rows = []
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    for name, d, a, b in jobs:
        rng = np.random.default_rng(zlib.crc32(b"%d/%s" % (seed, name.encode())))
        lines = z.read(root + "anno/UAV123_10fps/%s.txt" % name).decode().splitlines()
        boxes = [None if "nan" in l.lower() else [float(v) for v in l.split(",")] for l in lines]
        cls = UAV123_SEQ.match(name).group(1)
        for k, i in enumerate(_pick(len(boxes), every, cap, [i for i, bx in enumerate(boxes) if bx is None])):
            img = Image.open(io.BytesIO(z.read(root + "data_seq/UAV123_10fps/%s/%06d.jpg" % (d, a + i)))).convert("RGB")
            rows += _emit(out, "uav123", name, cls, a + i, img, boxes[i], rng, k, tan_half,
                          reason=None if boxes[i] is not None else "occluded or out of view (NaN box)")
    return rows


def convert_visdrone(seqs, out, every=30, cap=24, seed=0):
    """seqs: [(split, seq)]. VisDrone-SOT has a box on every frame (occlusion / out-of-view are only
    sequence-level attributes), so every frame here counts as target-in-view."""
    from PIL import Image
    z = open_zip("visdrone_sot")
    names = z.namelist()
    tan_half = math.tan(math.radians(HFOV_DEG / 2))
    rows = []
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    for split, seq in seqs:
        rng = np.random.default_rng(zlib.crc32(b"%d/%s" % (seed, seq.encode())))
        root = "VisDrone2019-SOT-%s/VisDrone2019-SOT-%s/" % (split, split)
        lines = z.read(root + "annotations/%s.txt" % seq).decode().splitlines()
        frames = sorted(n for n in names if n.startswith(root + "sequences/%s/" % seq) and n.endswith(".jpg"))
        boxes = [[float(v) for v in l.split(",")] for l in lines]
        boxes = [bx if bx[2] > 0 and bx[3] > 0 else None for bx in boxes]
        for k, i in enumerate(_pick(min(len(boxes), len(frames)), every, cap, [])):
            img = Image.open(io.BytesIO(z.read(frames[i]))).convert("RGB")
            rows += _emit(out, "visdrone", seq, VISDRONE_CLASS[split][seq], i + 1, img, boxes[i], rng, k, tan_half,
                          reason=None, split=split)
    return rows


def _emit(out, dataset, seq, cls, frame_no, img, box, rng, k, tan_half, reason=None, split=None):
    rows = []
    W0, H0 = img.size
    for view, im, bx in _views(rng, img, box, k):
        fname = "%s-%s-%06d-%s.jpg" % (dataset, seq, frame_no, view)
        (w, h), sbx = _save(im, bx, os.path.join(out, "frames", fname))
        th = tan_half * (im.size[0] / W0)          # a crop sees a proportionally narrower field
        r = {"frame": fname, "dataset": dataset, "seq": seq, "cls": cls, "source_frame": frame_no,
             "view": view, "src_size": [W0, H0], "size": [w, h], "hfov_deg": round(2 * math.degrees(math.atan(th)), 2)}
        if split:
            r["split"] = split
        r.update(_box_row(sbx, w, h, th))
        if not r["visible"]:
            r["absent_reason"] = reason or ("crop beside the target" if view == "crop-neg" else "box outside the crop")
        rows.append(r)
    return rows


def inspect(name, n=40):
    """A summary of a source zip's layout: member count, top-level dirs, sample names, a few annotation lines."""
    z = open_zip(name)
    names = z.namelist()
    dirs = sorted({"/".join(x.split("/")[:3]) for x in names})
    txt = [x for x in names if x.endswith(".txt")]
    sample = {t: z.read(t).decode("utf8", "replace").splitlines()[:3] for t in txt[:6]}
    return {"name": name, "bytes": z.fp.size if hasattr(z.fp, "size") else None, "members": len(names), "dirs": dirs[:400],
            "txt": txt[:600], "sample_names": names[:n], "sample_txt": sample}


# ---- questions: probe.questions_v2() as trained, and the same with the real target named ----
ROVER_CTX = ("Onboard forward camera of a quadrotor following a small red ground rover "
             "(a red box with a thin red mast). ")


def questions_real(cls=None):
    """probe.questions_v2() (visible, where, steer7, range8). cls=None: exactly as trained (the rover).
    cls='car' etc.: the instructions name the tracked object instead ("the car being followed"); the
    options (criteria) stay byte-identical, including their wording about the rover."""
    import copy, probe
    qs = probe.questions_v2()
    if cls is None:
        return qs
    t = "the %s being followed" % NOUN[cls]
    qs = copy.deepcopy(qs)
    for q in qs.values():
        a = q["instructions"]
        assert a.startswith(ROVER_CTX), a
        rest = a[len(ROVER_CTX):].replace("the red rover", t).replace("the rover", t)
        assert t in rest, rest
        q["instructions"] = "Onboard forward camera of a quadrotor following a %s on the ground. %s" % (NOUN[cls], rest)
    return qs


WORDINGS = ("rover", "target")


def evaluate_real(agent, frames_dir, rows, wordings=WORDINGS):
    """Ask questions_real on each frame, once per wording. Returns one pred per (row, wording)."""
    import probe
    from PIL import Image
    cache = {}
    out = []
    for r in rows:
        img = Image.open(os.path.join(frames_dir, r["frame"])).convert("RGB")
        for w in wordings:
            key = None if w == "rover" else r["cls"]
            if key not in cache:
                cache[key] = questions_real(key)
            a = agent.predict({"image": img}, cache[key])["answers"]
            out.append(dict(r, wording=w, p_visible=float(a["visible"]["noul"]), where=a["where"]["choice"],
                            steer7_probs=[float(a["steer7"]["probabilities"][str(i)]) for i in range(len(probe.STEER7))],
                            range8_probs=[float(a["range8"]["probabilities"][str(i)]) for i in range(len(probe.RANGE8))]))
    return out


SIDE_DX = 0.1      # side accuracy on frames with the box centre more than 10% of the width off centre


def _ev(ps, centres):
    q = np.asarray(ps, dtype=float)
    return float((q / q.sum() * np.asarray(centres)).sum())


def score_real(preds):
    """Metrics for one group of preds (one wording). Baselines alongside each."""
    import probe
    vis = [p for p in preds if p["visible"]]
    hid = [p for p in preds if not p["visible"]]
    nat = [p for p in hid if p["view"] == "full"]
    cneg = [p for p in hid if p["view"] != "full"]
    auc = lambda ps, ns: probe._auc([p["p_visible"] for p in ps], [p["p_visible"] for p in ns])  # noqa: E731
    wt = [p["where_truth"] for p in preds]
    maj = lambda xs: float(max(np.mean([x == k for x in xs]) for k in set(xs))) if xs else None  # noqa: E731
    out = {"frames": len(preds), "visible_frames": len(vis), "absent_natural": len(nat), "absent_crop": len(cneg),
           "visible_auc": auc(vis, hid), "visible_auc_natural_negatives": auc(vis, nat),
           "visible_auc_crop_negatives": auc(vis, cneg),
           "p_visible_mean_visible": float(np.mean([p["p_visible"] for p in vis])) if vis else None,
           "p_visible_mean_absent": float(np.mean([p["p_visible"] for p in hid])) if hid else None,
           "where_acc": float(np.mean([p["where"] == w for p, w in zip(preds, wt)])) if preds else None,
           "where_majority_baseline": maj(wt),
           "where_acc_visible": float(np.mean([p["where"] == p["where_truth"] for p in vis])) if vis else None,
           "where_majority_baseline_visible": maj([p["where_truth"] for p in vis]),
           "where_answer_counts": {k: int(sum(p["where"] == k for p in preds)) for k in ("left", "centre", "right", "not visible")},
           "where_truth_counts": {k: int(sum(w == k for w in wt)) for k in ("left", "centre", "right", "not visible")}}
    if len(vis) > 2:
        dx = np.array([p["dx"] for p in vis])
        b = np.array([p["bearing_deg"] for p in vis])
        bs = np.array([probe.x_to_bearing(p["x_c"]) for p in vis])
        eb = np.array([_ev(p["steer7_probs"], probe.STEER7_CENTRES) for p in vis])
        side = np.abs(dx) > SIDE_DX
        er = np.array([_ev(p["range8_probs"], probe.RANGE8_CENTRES) for p in vis])
        rp = np.array([p["range_proxy"] for p in vis])
        seq_rho = []
        for s in sorted({p["seq"] for p in vis}):
            m = np.array([p["seq"] == s for p in vis])
            if m.sum() >= 8 and len(set(rp[m])) > 2 and er[m].std() > 0:
                seq_rho.append(probe._spearman(rp[m], er[m]))
        out.update({
            "steer_side_acc": float(np.mean(np.sign(eb[side]) == np.sign(b[side]))) if side.any() else None,
            "steer_side_n": int(side.sum()), "steer_side_baseline": 0.5,
            "steer_side_left_frac": float(np.mean(b[side] > 0)) if side.any() else None,
            # + bearing = left = negative dx, so this is the rank correlation with -dx: FOV-free
            "steer_spearman_vs_offset": probe._spearman(-dx, eb),
            "steer_bearing_mae_deg": float(np.mean(np.abs(eb - b))),
            "steer_bearing_mae_deg_always_straight": float(np.mean(np.abs(b))),
            # the same against the angle the sim camera (~124 deg wide) would give that pixel position:
            # what a model that learned "image position -> degrees" in the sim should answer
            "steer_bearing_mae_deg_simcam": float(np.mean(np.abs(eb - bs))),
            "steer_bearing_mae_deg_simcam_always_straight": float(np.mean(np.abs(bs))),
            "steer_expected_mean_deg": float(eb.mean()), "steer_expected_std_deg": float(eb.std()),
            "steer_level_probs_mean": np.mean([p["steer7_probs"] for p in vis], axis=0).round(3).tolist(),
            # bigger box = nearer: the proxy is 1 / box side, so + means "reads a bigger box as nearer"
            "range_spearman_vs_box": probe._spearman(rp, er),
            "range_spearman_within_seq_mean": float(np.mean(seq_rho)) if seq_rho else None,
            "range_within_seq_n": len(seq_rho),
            "range_expected_mean_m": float(er.mean()), "range_expected_std_m": float(er.std()),
        })
    return out


def score_groups(preds):
    """score_real overall and by dataset, class, and view, for each wording."""
    out = {}
    for w in sorted({p["wording"] for p in preds}):
        ps = [p for p in preds if p["wording"] == w]
        g = {"all": score_real(ps), "all_full_frames": score_real([p for p in ps if p["view"] == "full"])}
        for key in ("dataset", "cls", "view"):
            for v in sorted({p[key] for p in ps}):
                g["%s=%s" % (key, v)] = score_real([p for p in ps if p[key] == v])
        for d in sorted({p["dataset"] for p in ps}):
            for c in sorted({p["cls"] for p in ps if p["dataset"] == d}):
                g["dataset=%s,cls=%s" % (d, c)] = score_real([p for p in ps if p["dataset"] == d and p["cls"] == c])
        out[w] = g
    return out


def draw_sample(frames_dir, row, answers, width=384):
    """A small PNG of one frame: its box (green) and each checkpoint's answers underneath, with the
    steer read-out as a tick on the top edge at that checkpoint's bearing (sim-camera mapping).
    answers: [(label, pred)] with pred from evaluate_real."""
    import probe
    from PIL import Image, ImageDraw
    img = Image.open(os.path.join(frames_dir, row["frame"])).convert("RGB")
    f = width / img.size[0]
    img = img.resize((width, round(img.size[1] * f)), resample=3)
    line = 12
    out = Image.new("RGB", (width, img.size[1] + line * (len(answers) + 1) + 4), (20, 20, 20))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    if row["box"]:
        x, y, w, h = [v * f for v in row["box"]]
        d.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
    colours = [(255, 210, 0), (0, 200, 255), (255, 90, 200), (255, 255, 255)]
    truth = "truth: %s, %s" % (row["cls"], row["where_truth"] if row["visible"] else "not visible")
    d.text((3, img.size[1] + 2), truth, fill=(0, 255, 0))
    for k, (label, p) in enumerate(answers):
        eb = _ev(p["steer7_probs"], probe.STEER7_CENTRES)
        er = _ev(p["range8_probs"], probe.RANGE8_CENTRES)
        c = colours[k % len(colours)]
        xb = probe.bearing_to_x(eb) * width
        d.polygon([(xb - 5, 0), (xb + 5, 0), (xb, 9)], fill=c)
        d.text((3, img.size[1] + 2 + line * (k + 1)),
               "%s: P(vis) %.2f, %s, steer %+.0f deg, range %.1f m" % (label, p["p_visible"], p["where"], eb, er), fill=c)
    b = io.BytesIO()
    out.save(b, "PNG", optimize=True)
    return b.getvalue()
