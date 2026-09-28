# laya-steer: Laya supplies the pursuit heading

A fine-tuned Laya reads the onboard camera frame and says where the rover is. Its bearing
replaces the code's in pursuit (`run.py --pursuit`, `laya_pursuit.py`). Range, and so speed,
avoidance, the tactical layer (`const:oracle`) and the reflex all stay as code.

## The checkpoint: drone-rover-v1

laya-vision `finetune_long` from `autoresearch/full/long-sep24-b64/best` (= `thaitea/laya-vision`)
on `drone_rover`: 69k simulator-labelled examples from `rover_data.py` (classic, pockets and mixed,
seeds 0-7), mixed 2:1 with the 19 Cauldron sets. 0.6 epochs, best at step 5296, 30 min on an
A100 (about $2). Checkpoint `/ckpt/smolvlm/drone-rover-v1/best` on laya-checkpoints.

| val (1000 each) | acc | ECE |
|---|---|---|
| drone_rover | 95.2% | 0.020 |
| cauldron_vqav2 / aokvqa / ai2d | 77.1% / 74.5% / 77.7% | 0.19 / 0.22 / 0.17 |

At its first eval (step 2648) the three Cauldron sets were within about a point of these
(77.0 / 73.0 / 76.2). They were not scored for the base checkpoint on the same 1000-question
subsets, so whether the drone data cost any general ability is not measured here.

## Probe on held-out frames

1368 frames from the unseen no-climb layout and mixed seeds 20-21, half with the camera turned
by up to ±45°. Details in `results/probe/rover-test/`.

| whole frame, one predict | zero-shot | drone-rover-v1 |
|---|---|---|
| rover visible, AUC | 0.73 | 0.993 |
| where (left / centre / right / not visible) | 36% | 94% |
| steer: correct side | 51% | 98% |
| steer vs bearing, rank correlation | 0.03 | 0.97 |
| speed vs range, rank correlation | −0.03 | 0.81 |

**Read-out fix.** The steer score's expected level compresses toward the centre: a rover 12-25°
off the nose read about 8° too central. `FrameBackend` now sharpens the level probabilities
(power 2) and applies a gain of 1.16. Both were fitted on the mixed frames within ±34° and scored
on no-climb: 4.5° mean error, against 6.3° read raw and 12.9° for always straight.

## Flights (`results/laya/20260928-055058`)

90 s, oracle tactics, runs finished:

| | mixed, seeds 0-2 | mixed, 20-22 (held out) | no-climb, 0-2 and 20-22 (unseen layout) | total |
|---|---|---|---|---|
| code pursuit | 3/3 | 3/3 | 6/6 | **12/12** |
| laya-steer-frame | 2/3 | 2/3 | 4/6 | **8/12** |

- **In flight the heading error is 3.2-5.8° at 70-100 ms latency** (0.18 s on one slower host).
  Code segmentation is about 1°.
- **The runs split into two groups.** In 8 flights the rover stays in view 64-76% of the time,
  the same as code pursuit. In the other 10 it is in view only 12-19%: Laya loses it early and
  seldom gets it back. Some of those still finish because the fallback search stumbles back
  onto it. The gap is reacquisition after a loss, not steering accuracy.
- **Earlier failures that are now fixed.** The first GPU predict takes 2.7-4.0 s. Made inside a
  real-time flight, it left the aircraft with no heading for its opening seconds, so it
  overshot the rover (backends now warm up first). The strips method (5 predicts, 0.33-0.44 s)
  is too slow for pursuit on mixed: 0/3, stuck at the first pocket.
- `realtime_factor` was 0.57-1.00 in the Laya flights (the frame is rendered in software), so
  latency is if anything flattering.

**Next:** find where the losses happen, with recorded GIFs of the 12%-visibility runs. The first
pocket, where the rover drives through a hatch out of sight, is the likely place. That is the
reacquisition case the object-permanence proposal (occluded / reappear questions) is aimed at.

## v2: wider steering, finer range, Laya sets speed (`results/laya/20260928-093007` + `-093623`)

**Checkpoint:** `drone-rover-v2`, continued from v1 on `drone_rover_v2`. That is the same frames
relabelled with `steer7` (7 levels, ±60°) and `range8` (8 levels, 2-8.5 m), both with soft
targets. 18 min on an A100. We fly `last/` (step 4238, drone val 93.2%), not the trainer's
`best/` (step 2119, 90.7%, picked on the four-set mean). On held-out frames, sharpened power 2:

| | v1 | v2 last |
|---|---|---|
| bearing error, all / within ±34° | ~6° (corrected) / 4.5° | 4.4° / 3.8° |
| under-read at 25-45° / 45°+ | -19° past 25° (raw) | -3.6° / -8.1° |
| range error | 0.81 m | 0.43 m (bias -0.10 m) |
| forward-speed command error | 0.52 m/s | 0.30 m/s |

**Speed law:** `RangeSpeed` (laya_pursuit.py) is used whenever the range comes from a model. It
low-passes the range over time and predicts it between estimates from the drone's own velocity.
It uses gain 0.6 with a 0.3 m deadband at the 3.5 m standoff, a 0.9 m/s floor unless close, and
a 2.4 m/s cap while the rover is out of view. In the sim stand-in (bearing 4° + 0.1 s) it
finished 202/260 flights against 176/260 for the code law, and held 15-19/20 up to ~1 m of
range error.

**Flights** (90 s, oracle tactics, budget 240, runs finished):

| | mixed | no-climb | total | v1 |
|---|---|---|---|---|
| code pursuit | 6/6 | 6/6 | 12/12 | 12/12 |
| Laya heading, code speed (`laya-steer-frame-v2`) | 4/6 | 3/6 | 7/12 | 9/12 |
| **Laya heading and speed (`laya-pursuit-v2`)** | **5/6** | **3/6** | **8/12** | **2/12** |

- **Letting Laya set speed is now viable:** 8/12 against 2/12 with v1's 4 speed bands and the
  code's speed law.
- **Heading-only flights did not improve** (7/12 against 9/12). At 12 flights per cell a
  difference of 2 is within the run-to-run variance seen before: real-time flights are not
  deterministic, and the same config has swung by 2-3 between runs.
- **Failures are still early rover losses that are never recovered,** mostly at the pockets
  (rover in view ~12-30% in failed runs). Code pursuit is still 12/12, so reacquisition, not
  perception accuracy, is the remaining gap.
- The run was interrupted by a container restart after 32 of 36 flights. The 4 missing
  (no-climb laya-pursuit-v2, seeds 2, 20, 21, 22) were re-flown separately in `-093623`.
