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

## v3 / v3.1: reacquisition and tactics (`results/laya/20260928-183953`, `-191230`, `-192706`)

**Checkpoints.** `drone-rover-v3` added reacquisition questions (`occluded`, `reappear`,
`reappear_eta`) and the tactical `maneuver` question (`drone_rover_v3`: new oracle flights with
the frame plus a small JSON context). `drone-rover-v3.1` added `drone_rover_v3b`: rotated views
and sim failure flights, which raised the "behind" training examples from 303 to 3,716. Each
fine-tune took about 20 min on an A100.

**Held-out probes** (`results/probe/rover-test-v3*`), v3.1 on the behind-heavy set, lost-rover
frames only:

| | v3 | v3.1 | baseline |
|---|---|---|---|
| reappear side | 50% | 70% | 36% |
| "behind" recognised | 33% | 73% | |
| occluded, AUC | 0.80 | 0.95 | 0.50 |
| reappear time, level | 55% | 82% | 52% |

**Flights** (mixed + no-climb, oracle tactics unless noted, runs finished):

| | seeds | finished |
|---|---|---|
| code pursuit | 0-23 | 45/48 (94%) |
| Laya v2 pursuit | 0-23 | 39/48 (81%) |
| v3.1 pursuit (v2 read-out) | 0-23 | 22/48 |
| v3.1 pursuit (refitted read-out) | 0-11 | 13/24 (v2: 21/24 on the same seeds) |
| v3.1 full control (tactics + pursuit + reacquisition) | 0-11 | 6/24 |
| **v2 pursuit + v3.1 reacquisition** | 0-23 | **43/48 (90%)** |

- **Reacquisition works.** With v2 flying the pursuit and v3.1 answering where a lost rover will
  reappear, runs still lost at the end fell from 32 to 17 of 48, and no-climb went from 19/24 to
  23/24. When it acted, the side was right 78% of the time (median per flight). The turn only
  fires when v3.1 says the rover is NOT hidden behind something in view; turning regardless lost
  flights in CPU tests.
- **Re-fit the read-out for every checkpoint.** v3.1's calibrated score temperature is 0.74
  (v2: 2.10). The v2 read-out (power 2) snapped 60% of its steer estimates onto level centres,
  so the heading moved in 15° jumps (in-flight error 9.7° against 5.6°). Power 0.6 (steer) and
  0.75 (range) fix the snapping.
- **v3.1 is still a worse pursuit pilot than v2 in flight,** although it matches it on held-out
  frames. The extra training traded away some in-flight steering.
- **v3.1's tactics are not usable.** P(climb) rarely exceeds 0.2. At 0.12 the climb answer
  catches 54-66% of real climbs, but its precision is only 29% on pocket-heavy frames: it climbs
  into pockets (13-37 threshold climbs per flight, many ending at the first pocket's far wall).
  No threshold gives both recall and precision. It needs more and harder tactical data: beam
  approaches, plus pocket front walls labelled hold.

## v3.2: tactics and town data (`results/laya/20260928-223613`, `-230207`, `-230719`, `-233520`)

drone-rover-v3.2a = v3.1 plus two new sets: drone_rover_tac (the `tactics` course, where beams and
pocket front walls alternate, so climb and hold are seen side by side) and drone_rover_town (Nuketown
figure-8 laps). Its read-out was fitted with readout.py (steer and range power 1.5). v3.2b (a
different data mix) tied on steering and was worse on reappear (67.2% vs 72.6%), so v3.2a flew.

**Probe (held-out frames):** steer 3.8° within ±34°, range 0.40 m, reappear 72.6% (v3.1 70.2%).
Beam vs pocket P(climb) AUC 0.95, where v3.1 could not separate them. Best single threshold 0.1543:
beam recall 78%, false climbs on 4.8% of pocket frames. Town: visible AUC 0.999, bearing 3.5°.

**Flights, 24 seeds on each of mixed and no-climb** (90 s, budget 240). Bars: code pursuit 45/48,
v2 + v3.1 hybrid 43/48.

| config | pursuit | reacquire | tactics | mixed | no-climb | total |
|---|---|---|---|---|---|---|
| laya-full-v3.2 | v3.2a | v3.2a | v3.2a, P ≥ 0.1543 | 7 | 7 | 14/48 |
| laya-pursuit-v3.2-reacq | v3.2a | v3.2a | oracle | 17 | 13 | 30/48 |
| laya-tactics-v3.2 | code | – | v3.2a, P ≥ 0.1543 | 10 | 7 | 17/48 |
| laya-tactics-v3.2-vote | code | – | v3.2a, 3 × P ≥ 0.12 | 16 | 22 | 38/48 |
| hybrid-v2pursuit-v3.2-oracle | v2 | v3.2a | oracle | 22 | 18 | 40/48 |
| **hybrid-v2pursuit-v3.2-vote** | v2 | v3.2a | v3.2a, 3 × P ≥ 0.12 | 20 | 21 | **41/48** |

**Town, 6 laps** (120 s, budget 300; code pursuit 6/6, v2 pursuit 0/6):
laya-full-v3.2 4/6, laya-pursuit-v3.2-reacq 4/6, hybrid-v2pursuit-v3.2-vote 0/6.

- **A single threshold call is fatal.** One P(climb) over the threshold commits Guidance to a climb,
  and over a pocket's low front wall that flies the aircraft into the pocket (stalls at x ≈ 26.5, the
  first pocket's far wall). In the tac probe flights 0.1543 was crossed at least once in 14 of 21
  pocket visits. tactics.LayaV3Backend's `climb_votes=3` at 0.12 fires in 1-2 of 21 pocket visits and
  16 of 17 beam approaches (probe frames replayed at the flight's ~1.5 calls/s). In flight this took
  Laya tactics from 17/48 to 38/48; no-climb went from 7 to 22/24. The remaining mixed failures stall
  at x ≈ 54 (decoy) and x ≈ 74 (second pocket's far wall), so some false climbs survive the votes.
- **The first all-Laya flight near the bars:** hybrid-v2pursuit-v3.2-vote. Laya makes every decision:
  v2 steers and sets speed, v3.2a reacquires and decides climbs. It finished 41/48, against 45/48 for
  code and 43/48 for the v3.1 hybrid, which still had oracle tactics.
- **v3.2a's own pursuit is still weaker in flight than v2** on the corridors (30/48 against 40/48 with
  the same reacquisition). Most failures lose the rover at the first pocket and never find it again.
  In the town the reverse holds: v2 pursuit 0/6 (red scenery, under-turning), v3.2a 4/6. No single
  checkpoint steers well everywhere yet.

## Altitude as a continuous operator (`altitude.py`; `results/laya/20260928-234837`)

The one-shot `climb` answer sends the aircraft to 3.0 m and holds it there for ~3 s, so one wrong call over a
pocket's low front wall traps it. altitude.py replaces it with a score question: how far to move up or down
from the current altitude (−1, −0.5, 0, +0.5, +1 m), asked ~3 times a second. Each answer moves the setpoint
at most 0.5 m, toward (altitude at the frame + answer). Clearing a 2.1 m beam from 1.6 m cruise takes
several ascends in a row, and a wrong one is taken back by the next answer. The course's own answer is
courses.Course.altitude_target: 2.9 m from 5 m before a beam until 1.2 m past it, cruise elsewhere.

**Same error rate, both operators** (CPU, code pursuit, 12 seeds on each of mixed, no-climb and pockets).
The oracle answers are replaced, with the given probability per call, by a false "go up": `climb` for the
one-shot operator (tactics.ConstBackend wrong_p), "+1 m" for the altitude operator (altitude.SimAltitude).

| false "up" answers | one-shot climb | ascend / descend |
|---|---|---|
| 0 | 35/36 | 34/36 |
| 5% | 16/36 | 34/36 |
| 10% | 7/36 | 35/36 |
| 20% | 1/36 | 26/36 |

v3.2a's single-threshold climb made false climbs at about that rate (4.8% of pocket frames at 0.1543).
The altitude operator tolerates 10% of them with no loss.

### drone-rover-v3.3: Laya flies altitude (`results/laya/20260929-002941`)

**Data:** drone_rover_alt (`modal_laya.py::build_rover_set_alt`, rover_data.collect_flight_alt). Oracle
flights on pockets, mixed, tactics and town whose altitude answers are randomly replaced 20% or 35% of the
time, so the aircraft spends time too high and too low. Each frame is labelled with the move toward the
course target, with soft targets over the five levels. Hold is subsampled to at most ascend + descend:
7,958 train records (1,428 ascend, 2,551 descend, 3,979 hold) and 1,290 val. Seeds are disjoint from every
evaluation seed.

**Fine-tune:** from v3.2a/best, the v3.2 mix plus drone_rover_alt=15, 24.6 min on an A100. Val level accuracy
on drone_rover_alt 0.893; tac 0.997, town 0.941, v2 0.927, all held.

**Probe** (1,632 held-out frames: mixed 24-26, no-climb 24-25, tactics 0-1; read-out power 1):
mean error 0.086 m (always-hold: 0.24 m), Spearman 0.91. A false ascend at a pocket wall while low: 0.1% of
frames (v3.2a's climb answer: 4.8%). Ascend before a beam: 63% of frames, several per approach. Descend
when high: 92%.

**Flights, 24 seeds on each of mixed and no-climb** (90 s, budget 240):

| config | pursuit | reacquire | altitude | mixed | no-climb | total |
|---|---|---|---|---|---|---|
| code pursuit (bar), oracle climb | code | – | oracle climb | – | – | 45/48 |
| laya-tactics-v3.2-vote (before) | code | – | v3.2a climb, 3 votes | 16 | 22 | 38/48 |
| **laya-alt** | code | – | **v3.3 ascend/descend** | **24** | 21 | **45/48** |
| hybrid-v2pursuit-v3.2-vote (before) | v2 | v3.2a | v3.2a climb, 3 votes | 20 | 21 | 41/48 |
| **hybrid-v2pursuit-alt** (all Laya) | v2 | v3.3 | v3.3 ascend/descend | 21 | 21 | **42/48** |

- **Laya flying altitude matches the code with oracle climbs** (45/48), and finished all 24 mixed flights.
  Every mixed failure of the all-Laya hybrid got past the beam (max x 51-79). Those losses are pursuit and
  reacquisition, not altitude.
- The three laya-alt no-climb failures (seeds 0, 4, 9) held cruise (1.54-1.68 m) through the first pocket
  and lost the rover there. Code pursuit with the oracle finishes these seeds. The only visible difference is a
  ±0.07 m setpoint jitter, from re-anchoring each near-zero answer to the measured altitude. A deadband
  (keep the setpoint when |dz| < 0.2 m) would remove it; not yet tried.

## Real-time evaluation: latency-faithful timing (`results/laya/20260929-022552` + town)

Until now the sim was paced to the wall clock with Laya's workers in threads beside it. When the box could not
keep up (rt 0.6-0.9 on recent runs), the world slowed down and Laya got extra time, and Laya's calls also
competed with the physics and rendering for the CPU (0.1-0.25 s per call). `run.episode(timing="virtual")`
fixes both: the sim stands still while Laya computes, and each answer lands at sim time start + its measured
GPU latency. Every question's calls queue on one GPU (laya_pursuit.GpuClock). What Laya can answer, how often
and how late is what this GPU gives in real time; the sim's own speed no longer matters. yaw_desat is off.

Measured (v3.3 full control): 0.064 s per call, GPU 91% busy, steering 10.9 answers/s arriving 0.08 s after
their frame, altitude 2.5/s, reacquisition 2.5/s while lost.

| config (-rt = virtual timing) | steering | reacquire | altitude | mixed | no-climb | total | town |
|---|---|---|---|---|---|---|---|
| code pursuit, oracle climbs (bar) | code | – | oracle | – | – | 45/48 | 6/6 |
| **laya-alt-rt** | code | – | v3.3 | 24 | 23 | **47/48** | – |
| **laya-full-v3.3-rt** | v3.3 | v3.3 | v3.3 | 22 | 22 | **44/48** | **6/6** |
| hybrid-v2pursuit-alt-rt | v2 | v3.3 | v3.3 | 15 | 21 | 36/48 | 3/6 |

- **A single checkpoint in full control matches the code** on the corridors (44 vs 45/48) and in the town
  (6/6, rover in view 99.9%, no reflex, no collisions). Laya flying altitude on code steering beats the code
  with oracle climbs (47/48).
- **The earlier wall-clock numbers understated Laya.** Contention made its answers 2-3x slower than the GPU
  allows; with that removed (and yaw_desat off), laya-full-v3.3 went from 24/48 to 44/48.
- **Ballooning remains** (airmode lift under hard yaw): 31 of 48 v3.3 flights went above 4 m (median peak
  6.4 m, max 13.7 m). Most still finish. A heading low-pass or yaw-rate cap, rather than the mixer change, is
  the next thing to try.
- v2 steering is now clearly worse than v3.3 (36/48, town 3/6): it answers 7.6 times a second against 10.9,
  since two checkpoints share the GPU, and it was never trained on the town or the latest data.

## Lookahead teacher: the 4x ceiling check (`teacher.py`, `results/teacher/`)

Every hand-written rule caps what Laya can learn by imitating it. teacher.py has no rule: four times a second it
saves the simulator and the flight controller, tries 45 candidate commands (forward speed -1..6.5 m/s x side
slide x heading offset from the rover) for 0.5 s each followed by a brake to hover over a 2 s lookahead, with the
rover driving as it really will, scores what happened (a collision dominates, then rover in view, then distance
near 4 m), restores the state and flies the best. It uses privileged information (where the rover is and will be),
in the spirit of AlphaZero (github.com/ericjang/autogo): search makes the targets, a network learns them.

Ceiling check: the teacher flies every 4x course itself, 12 seeds each (CPU, ~30x real time).

| 4x course | teacher | best hand-written controller, perfect perception |
|---|---|---|
| town-x4 | **12/12**, rover in view 100% (median), 0 collisions | 8/12 |
| city-x4 | **12/12**, 99.7%, 0 collisions | 12/12 |
| pockets-x4 | **11/12**, 79% | 1-3/12 |
| tactics-x4 | **10/12**, 80% | 0/4 |
| mixed-x4 | 5/12, 22% | 4/12 |
| no-climb-x4 | 4/12, 18% | 2/4 |

Zero collisions and zero crashes in all 72 flights. The 4x rover is flyable; the hand-written pursuit and avoidance
were the limit. The teacher's own failures (mixed, no-climb) lose the rover, not the aircraft: its 2 s lookahead
and "rover in view" score cannot plan the detour round a pocket or past the decoy. A longer horizon or a score for
staying close to where the rover is going is the next teacher improvement. Two design points mattered: holding a
command for the whole horizon left no safe option in 16% of decisions (commit 0.5 s then brake instead), and a
1.2 s horizon steered into fences it could no longer avoid (2 s: none).
