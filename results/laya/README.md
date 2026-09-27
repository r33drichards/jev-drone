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

## Reproduce

```bash
modal run modal_laya.py::scenes
modal run modal_laya.py::baseline --seeds 0,1,2 --seconds 65
modal run modal_laya.py::baseline --configs always-climb,always-hold --seeds 0,1,2
```
