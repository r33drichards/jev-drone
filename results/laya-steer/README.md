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
(77.0 / 73.0 / 76.2), so the drone data did not cost general ability.

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
