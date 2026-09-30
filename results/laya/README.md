# Laya Vision as the tactical layer: baseline

Zero-shot [`thaitea/laya-vision`](https://huggingface.co/thaitea/laya-vision) (laya-vision
`79c85a6`) answering the same three questions as Jev, with every option description kept whole
(`option_max_len=256`, `head_max_len=1024`). Run on Modal L4/A10G with `modal_laya.py`, 65 s
flights, seeds 0-2, paced to real time. The Jev column in the top-level README is not rerun here:
this environment has no TypeSafe key.

## Hand-built scenes (`scenes.py`): 1 / 7 in every variant

`20260927-063106/scenes.json`. The model gives almost the same answer to every scene: one option
at p≈0.20-0.23 and the rest just below. What changes that answer is the budget, not the scene.
With full descriptions it is always `climb`. With options cut to 48 tokens it is always
`gap_right`. `risk` stays at 0.91-0.97 and `target_truly_lost` at 0.43-0.46 whatever the
situation. A grey image beside the JSON changes nothing.

| variant | correct | always answers |
|---|---|---|
| full budgets, no image | 1/7 | climb |
| full budgets, grey image | 1/7 | climb |
| 48-token options, no image | 1/7 | gap_right |
| 48-token options, grey image | 1/7 | gap_right |

For comparison, the README reports 6/7 for Jev on the same situations.

## Flights (`episodes.jsonl` in `20260927-063342` and `20260927-063619`)

| config | past beam0 | whole course | max_x per seed (m) | target in view | time in reflex | collisions (3 runs) | median latency |
|---|---|---|---|---|---|---|---|
| `no-model` (heuristic only) | 1/3 | 1/3 | 77.2 / 18.4 / 18.2 | 34% | 58% | 4 | - |
| `laya-text` | 3/3 | 3/3 | 77.2 / 77.2 / 77.2 | 95% | 8% | 3 | 71 ms |
| `laya-image` (JSON + onboard frame) | 3/3 | 3/3 | 77.5 / 77.3 / 77.3 | 89% | 8% | 4 | 115 ms |
| `laya-text-lockstep` (sim waits) | 3/3 | 3/3 | 77.2 / 77.2 / 77.2 | 95% | 8% | 3 | 59 ms |
| `always-climb` (control, no model) | 3/3 | 3/3 | 77.2 / 77.2 / 77.2 | 95% | 8% | 3 | - |
| `always-hold` (control, no model) | 1/3 | 0/3 | 18.3 / 18.0 / 36.5 | 21% | 56% | 1 | - |

**Laya is not what flies the course.** `laya-text` and `always-climb` agree on every metric of
every seed, and so does `laya-text-lockstep`. The model answers `climb` whatever it sees.
`run.Guidance` climbs only when at least four sectors are blocked and there is measured clear air
above. So a constant "climb" plus the code's veto clears the beam, and every other judgment
falls through to the heuristic. `laya-image` differs a little, so the frame does move the answer,
but it is no better (89% visibility, one more collision). Its runs also fell to 0.77-0.83×
real time, because OSMesa renders the 512×384 frame in software.

Two things about the harness itself:

- The no-model heuristic cleared the whole course on seed 0. The README says the baseline stops at
  station 2 every time, but that figure predates later controller fixes. The "structurally
  incapable" claim now holds for seeds 1 and 2 only.
- Always climbing when code allows it is a strong policy on this course. Beating the controls
  takes the maneuvers `always-climb` cannot express, such as gap choice, brake and reacquire.

## Courses that do not reward always-climb (`courses.py`)

The classic course never punishes a climb: its only full-width obstacles are beams, where
climbing is right. `courses.py` adds three layouts ([maps](../../docs/), seed 0 shown there;
gap sides are drawn from the seed):

- **pocket**: two dead ends side by side with a lane between them. Each has a low front wall (it
  passes the climb check), a low divider and a tall wall 12 m back. The rover drives through one
  pocket by floor hatches too low for the aircraft. A drone that climbs in sinks after its climb
  hold and faces a wall it cannot climb.
- **beam**: as on the classic course, so a model that never climbs fails too.
- **decoy**: the real gap looks shallow (a baffle behind it), and a dead-end recess on the other
  side looks deeper.

|  | pockets | mixed | no-climb |
|---|---|---|---|
| layout | beam, pocket, beam, pocket | pocket, beam, decoy, pocket | pocket, decoy, pocket |
| **oracle** (climb at beams, else hold) | **3/3** | **3/3** | **3/3** |
| always climb | 1/3 | 0/3 | 0/3 |
| always hold | 0/3 | 0/3 | 3/3 |
| always gap_left / gap_right | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| always brake / reacquire | 0/3 / 0/3 | 0/3 / 1/3 | 0/3 / 0/3 |
| no model (heuristic) | 0/3 | 0/3 | 0/3 |
| **laya-text** | 1/3 | 1/3 | 1/3 |
| **laya-image** | 1/3 | 1/3 | 0/3 |

Runs finished out of 3 (seeds 0-2, 90 s flights). Controls come from `course_check.py`
(`courses-controls.jsonl`, faster than real time). Laya runs were on Modal, paced to real time,
with a call budget of 240 (`20260927-072032`); the default 160 ran out on some 90 s flights.

- **`mixed` is the course to use.** The oracle finishes every run, and no constant answer or the
  heuristic finishes any. `no-climb` tests one thing, whether the model knows not to climb:
  always-hold passes it by construction.
- **The oracle answers `hold_course` at pockets and decoys, not `gap_*`.** Guidance's gap slide
  has no notion of where to stop. Commanded through a pocket, it carries the aircraft across the
  lane into a wall (0/3 in `course_check.py`). What works is climbing at beams and otherwise
  leaving it to the pursuit and reactive layers. That is a finding about the guidance: in this
  stack the tactical layer's value is mostly in knowing when to climb and when not to.
- **Laya is no longer identical to always-climb here.** Its answers vary with the scene enough
  to change the flight, but it finishes 1 run in 3 at best, against the oracle's 3/3. Real-time
  runs are not deterministic: an earlier run of the same configs, with the 160 budget, got 2/3
  on `no-climb` for laya-text.
- **Only `free_ahead_above_m` separates a pocket from a beam in the JSON** (about 15 m, the far
  wall, against 45 m over a real beam). The camera frame shows the difference plainly, which is
  what the image variant is for. At zero shot it does not help.

## Reproduce

```bash
modal run modal_laya.py::scenes
modal run modal_laya.py::baseline --seeds 0,1,2 --seconds 65
modal run modal_laya.py::baseline --configs always-climb,always-hold --seeds 0,1,2
MUJOCO_GL=osmesa python course_check.py --seconds 90             # controls on the new courses
modal run modal_laya.py::baseline --configs laya-text,laya-image --courses pockets,mixed,no-climb \
    --seconds 90 --budget 240
```
