# Probe: can zero-shot Laya see where the rover is?

The check before handing pursuit to Laya. `probe.py collect` flew the oracle on the classic and
`mixed` courses (seeds 0-2, 60 s each) and saved the onboard frame every 0.5 s, 684 frames in all,
with the ground truth from the simulator: the rover's bearing off the nose, its range, and whether
it is in view (segmentation). `modal run modal_laya.py::probe` then asked `thaitea/laya-vision`
four questions about each frame, zero-shot, with the full option descriptions.

![sample frames with their labels](../../docs/probe-frames.png)

The frames are fair: the rover is in view in 562 of them, at a median of about 590 pixels in the
512×384 frame, a distinct red box with a mast.

| question | zero-shot Laya | baseline |
|---|---|---|
| `visible` (noul): is the rover in the image? | AUC 0.70 | 0.50 |
| `where` (choice): left / centre / right third, or not visible | 17% correct | 78% (always the commonest) |
| `steer` (score, 5 levels): which way to turn | 31% correct, rank corr. with bearing −0.04 | 57% (always "straight") |
| `steer`, side only (rover more than 7° off the nose) | 48% | 50% |
| `speed` (score, 4 levels): how far away | 28% correct, rank corr. with range 0.07 | 65% (always "slow") |

Details in `summary-perm1.json`; per-frame answers in `preds-perm1.jsonl`.

- **`where` answers "not visible" for 603 of 684 frames**, never "centre", although the rover is in
  view in 82% of them and near the centre in most.
- **`steer` and `speed` are flat.** The mean probabilities over visible frames are
  [0.19, 0.20, 0.22, 0.20, 0.20] and [0.28, 0.27, 0.23, 0.22], almost uniform, and they do not
  move with the true bearing or range.
- **The only signal is `visible`, AUC 0.70:** something about the red box registers, but not
  where it is.

**Verdict:** zero-shot Laya cannot fly pursuit. Its steering would be a coin flip. That matches
the tactical results: the checkpoint reads these synthetic frames poorly. The next step is a
fine-tune, and the labels are free. Every frame here already carries its true bearing and range,
and `collect` makes more at about 110 frames per simulated minute. The model to beat is the code
pursuit law, whose inputs are exactly these two numbers.

## Strips: one yes/no per crop (after the detector-regions branch)

laya-vision's `claude/youthful-newton-mjsunc` branch found three things about this checkpoint.
It sees an image as one 512 px tile, 64 image tokens, so small objects nearly vanish. "Where is
X?" as a choice does not work. And a yes/no question on a crop that zooms in on the target
does. Here the rover is ~25 px wide in a 512 px frame, a third of one token's patch.

So `--mode strips` cuts the band of rows the rover can be in (30-90% of the height) into 5
vertical strips, upscales each to 512 px, and asks one yes/no per strip: "Is there a small red
ground rover in this image?" The strip with the highest yes-probability gives the bearing.

The oracle frames above cannot test left from right: pursuit keeps 98% of visible rovers
within ±12°, inside the middle strip. So `probe.py collect --yaw-jitter 45` renders each saved
view with the camera turned by a random ±45°, which spreads the rover across the frame
(`probe-jitter/`, 456 frames, 374 with the rover in view). Both probes were run on those:

| on the jittered frames | whole frame (`where`, `steer`) | 5 strips, yes/no each | baseline |
|---|---|---|---|
| correct side, rover >7° off the nose (312 frames) | 53% (`steer`) | **75%** (weighted), **85%** (top strip, on the 216 frames it is not the middle one) | 50% |
| rank correlation with true bearing | 0.05 | **0.53** | 0 |
| correct strip (of 5) | - | 57%, 89% within one strip | 42% (always the commonest) |
| bearing error, mean | - | 20° (top strip), 18° (sharpened weights) | 25° (always straight) |
| rover in view, AUC | 0.71 | 0.66 (max over strips) | 0.50 |
| crop contains the rover, AUC | - | 0.79 (0.87 on the unjittered frames) | 0.50 |

- **Strips make it see the rover.** Side accuracy goes from a coin flip to 75-85%, and the
  estimate tracks the true bearing (rank correlation 0.53, up from 0.05).
- **It gets the direction, not the angle.** A mean error of 18-20° against 25° for always
  straight is too coarse to hold the rover centred, where pursuit works within ±12°.
- **The probabilities are low and flat.** P(yes) averages 0.17 on the rover's strip against 0.13
  on empty ones, so only the ranking carries information; any threshold has to be fitted.
- **It costs 5 image encodes per decision** instead of one.

**Verdict:** the branch's zoom idea works here. Cropping turns a model that cannot see the rover
into one that finds its side most of the time, zero-shot. That is enough for "which way is it"
decisions, such as reacquiring a lost rover, but not for fine pursuit. For pursuit, the next step
is still a fine-tune, now with crops as the input. The labels come for free, and the strip layout
gives the model a target several tokens wide.

Reproduce:

```bash
MUJOCO_GL=osmesa python probe.py collect --out /tmp/probe
modal run modal_laya.py::probe --frames /tmp/probe
modal run modal_laya.py::probe --frames /tmp/probe --mode strips
MUJOCO_GL=osmesa python probe.py collect --out /tmp/probe-jitter --seeds 0 1 --yaw-jitter 45
modal run modal_laya.py::probe --frames /tmp/probe-jitter            # and --mode strips
```
