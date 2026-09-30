"""S-path (weaving) metrics for a flight, from the aircraft's xy track and the rover's.

Every course runs along +x, so "lateral" is y. The track is resampled by ARC LENGTH (0.25 m) before
anything is measured, so hovering, braking and search loops do not count as distance, and a weave
is measured per metre flown, not per second.

  weave_per_10m        heading swings per 10 m flown: the heading of the ground track changes
                       direction after turning at least SWING_DEG one way (a zig-zag count with
                       hysteresis, so jitter does not count). A straight run is 0; one full S is 2.
  rover_weave_per_10m  the same count on the rover's own track, over the x the aircraft covered.
  yaw_swings_per_10m   the same zig-zag count on the NOSE yaw (sampled in time), per 10 m flown:
                       heading-command wobble that the ground track may not show.
  lat_hf_rms_m         RMS lateral (y) offset of the track from itself smoothed with a Gaussian of
                       HP_SIGMA_M of arc (a high-pass: swings with wavelength well under ~4*sigma
                       pass, the slow course-following swings do not).
  rover_lat_hf_rms_m   the same high-pass on the rover's y AT THE AIRCRAFT'S x (what a drone flying
                       exactly the rover's line would score).
  extra_lat_hf_rms_m   the high-pass of (aircraft y - rover y at the same x): lateral weaving that
                       is NOT the rover's own line. The part of lat_hf that follows the rover is
                       what remains.
  extra_frac           extra_lat_hf^2 / lat_hf^2, clipped to [0, 1]: the share of the high-pass
                       lateral variance that is the aircraft's own (1 = none of it is the rover's).
  path_ratio           arc length / x progress, from the first time the track passes x0 to the
                       first time it reaches x1 (default: the start and end of the course);
                       rover_path_ratio is the rover's over the same x. A straight corridor is 1.
"""
import numpy as np

STEP_M = 0.25
SWING_DEG = 8.0
YAW_SWING_DEG = 5.0
HP_SIGMA_M = 2.5


def _resample(xy, step=STEP_M):
    """Points every `step` m of arc along a polyline (duplicates dropped)."""
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 2 * step:
        return xy[:1], s[-1]
    q = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])]), s[-1]


def _gauss(y, sigma_n):
    """Gaussian smoothing with edge padding; sigma in samples."""
    if sigma_n <= 0 or len(y) < 3:
        return np.asarray(y, float).copy()
    r = int(np.ceil(3 * sigma_n))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_n) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, r, mode="edge"), k, mode="valid")


def zigzag_count(a, thresh):
    """Direction reversals of a (continuous, unwrapped) signal, each leg >= thresh."""
    a = np.asarray(a, float)
    if len(a) < 3:
        return 0
    n, direction, ext = 0, 0, a[0]
    for v in a[1:]:
        if direction == 0:
            if abs(v - ext) >= thresh:
                direction, ext = (1 if v > ext else -1), v
            continue
        if direction > 0:
            if v > ext:
                ext = v
            elif ext - v >= thresh:
                n, direction, ext = n + 1, -1, v
        else:
            if v < ext:
                ext = v
            elif v - ext >= thresh:
                n, direction, ext = n + 1, 1, v
    return n


def _heading_swings(xy):
    """Zig-zag count of the ground-track heading, and the arc length it was measured over."""
    p, L = _resample(xy)
    if len(p) < 8:
        return 0, L
    p = np.column_stack([_gauss(p[:, 0], 1.0), _gauss(p[:, 1], 1.0)])   # 0.25 m jitter off
    h = np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0])))
    return zigzag_count(np.rad2deg(h), SWING_DEG), L


def _rover_y_at(rover_xy, x):
    r = np.asarray(rover_xy, float)
    order = np.argsort(r[:, 0])
    return np.interp(x, r[order, 0], r[order, 1])


def _path_ratio(xy, x0, x1):
    xy = np.asarray(xy, float)
    i0 = np.argmax(xy[:, 0] >= x0) if np.any(xy[:, 0] >= x0) else None
    i1 = np.argmax(xy[:, 0] >= x1) if np.any(xy[:, 0] >= x1) else None
    if i0 is None:
        return None
    i1 = len(xy) - 1 if i1 is None else i1
    if i1 <= i0 or xy[i1, 0] - xy[i0, 0] < 2.0:
        return None
    L = float(np.sum(np.linalg.norm(np.diff(xy[i0:i1 + 1], axis=0), axis=1)))
    return L / float(xy[i1, 0] - xy[i0, 0])


def summary(t, xy, yaw, rover_xy, x0=None, x1=None):
    """`t`, `xy` (N,2), `yaw` (N,): the aircraft, sampled together (e.g. 10 Hz). `rover_xy` (M,2): the
    rover's track, densely enough to interpolate y over x. `x0`, `x1`: the stretch for path_ratio
    (default: where the rover's track starts, and the furthest the aircraft got)."""
    xy = np.asarray(xy, float)
    rover_xy = np.asarray(rover_xy, float)
    x0 = float(rover_xy[:, 0].min()) if x0 is None else float(x0)
    x1 = float(xy[:, 0].max()) if x1 is None else float(x1)
    # measure from where the rover's line exists (the aircraft starts behind it)
    k0 = int(np.argmax(xy[:, 0] >= x0)) if np.any(xy[:, 0] >= x0) else len(xy)
    k1 = int(np.argmax(xy[:, 0] >= x1)) + 1 if np.any(xy[:, 0] >= x1) else len(xy)
    d = xy[k0:k1]
    out = {"weave_per_10m": None, "rover_weave_per_10m": None, "yaw_swings_per_10m": None,
           "lat_hf_rms_m": None, "rover_lat_hf_rms_m": None, "extra_lat_hf_rms_m": None, "extra_frac": None,
           "path_ratio": None, "rover_path_ratio": None, "path_m": 0.0}
    if len(d) < 10:
        return out
    n, L = _heading_swings(d)
    out["path_m"] = round(float(L), 1)
    if L < 5.0:
        return out
    out["weave_per_10m"] = round(float(10.0 * n / L), 2)
    # the rover over the same x
    lo, hi = d[:, 0].min(), d[:, 0].max()
    r = rover_xy[(rover_xy[:, 0] >= lo) & (rover_xy[:, 0] <= hi)]
    if len(r) >= 10:
        nr, Lr = _heading_swings(r)
        out["rover_weave_per_10m"] = round(float(10.0 * nr / max(Lr, 1e-9)), 2)
    ny = zigzag_count(np.rad2deg(np.unwrap(np.asarray(yaw, float)[k0:k1])), YAW_SWING_DEG)
    out["yaw_swings_per_10m"] = round(float(10.0 * ny / L), 2)
    # lateral high-pass along the arc
    p, _ = _resample(d)
    sig = HP_SIGMA_M / STEP_M
    yd = p[:, 1]
    yr = _rover_y_at(rover_xy, p[:, 0])
    hd = yd - _gauss(yd, sig)
    hr = yr - _gauss(yr, sig)
    e = yd - yr
    he = e - _gauss(e, sig)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
    out["lat_hf_rms_m"] = round(rms(hd), 3)
    out["rover_lat_hf_rms_m"] = round(rms(hr), 3)
    out["extra_lat_hf_rms_m"] = round(rms(he), 3)
    out["extra_frac"] = round(float(np.clip(rms(he) ** 2 / max(rms(hd) ** 2, 1e-12), 0.0, 1.0)), 2)
    pr = _path_ratio(xy, x0, x1)
    out["path_ratio"] = None if pr is None else round(pr, 3)
    rpr = _path_ratio(rover_xy[np.argsort(rover_xy[:, 0])], x0, x1)
    out["rover_path_ratio"] = None if rpr is None else round(rpr, 3)
    return out
