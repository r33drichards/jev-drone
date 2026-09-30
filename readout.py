"""Fit a checkpoint's steer / range read-out from its probe predictions.

Every fine-tune re-fits the calibration temperatures, which changes how sharp the score levels'
probabilities are: drone-rover-v2's score temperature was 2.10, v3.1's 0.74. The flight read-out
raises the level probabilities to a power ("sharpen") before taking the probability-weighted level
centre, and a power tuned for one checkpoint can wreck another (v2's power 2 snapped 60% of v3.1's
steer estimates onto level centres, so the heading moved in 15 deg jumps). So fit it per checkpoint:

    python readout.py results/probe/rover-test/ckpt_smolvlm_<run>/preds-v2.jsonl

It fits on the mixed-course frames and reports the held-out no-climb frames, for steer (error within
+-34 deg, pursuit's turn clip, plus a penalty for snapping onto level centres) and for range, and
writes readout.json beside the predictions: {"steer_sharpen", "range_sharpen", ...}. Pass them to a
flight as pursuit_sharpen and pursuit_range={"range_sharpen": ...}.
"""
import json, os, sys
import numpy as np
import probe

POWERS = [0.35, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0]


def _ev(ps, centres, k):
    q = np.asarray(ps, dtype=float) ** k
    return float((q / q.sum() * np.asarray(centres)).sum())


def _steer_cost(rows, k, snap_w=2.0):
    """Mean error within +-34 deg plus snap_w deg per unit of the fraction snapped onto a level
    centre (within 2 deg): a snapped read-out turns the heading into steps."""
    C = np.asarray(probe.STEER7_CENTRES)
    b = np.array([r["bearing_deg"] for r in rows])
    e = np.array([_ev(r["steer7_probs"], C, k) for r in rows])
    m = np.abs(b) <= 34
    snap = float(np.mean(np.min(np.abs(e[:, None] - C[None, :]), axis=1) < 2.0))
    mae = float(np.mean(np.abs(e - b)[m])) if m.any() else float("nan")
    return mae + snap_w * snap, mae, snap


def _range_mae(rows, k):
    C = probe.RANGE8_CENTRES
    return float(np.mean([abs(_ev(r["range8_probs"], C, k) - r["range_m"]) for r in rows]))


def fit(preds):
    vis = [p for p in preds if p.get("visible") and p.get("steer7_probs")]
    fit_rows = [p for p in vis if p.get("course") != "no-climb"]
    test_rows = [p for p in vis if p.get("course") == "no-climb"] or vis
    sk = min(POWERS, key=lambda k: _steer_cost(fit_rows, k)[0])
    rk = min(POWERS, key=lambda k: _range_mae(fit_rows, k))
    out = {"steer_sharpen": sk, "range_sharpen": rk, "fit_frames": len(fit_rows), "test_frames": len(test_rows)}
    for name, k in (("fitted", (sk, rk)), ("v2_default", (2.0, 2.0)), ("raw", (1.0, 1.0))):
        _, mae, snap = _steer_cost(test_rows, k[0])
        out[name] = {"steer_mae_within34_deg": round(mae, 2), "steer_snapped_frac": round(snap, 3),
                     "range_mae_m": round(_range_mae(test_rows, k[1]), 3)}
    return out


def main(path):
    preds = [json.loads(l) for l in open(path)]
    out = fit(preds)
    dst = os.path.join(os.path.dirname(path), "readout.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1])
