# Jev on the same harness

`jev-latest` over the TypeSafe API, flown from a Claude Code cloud container on 2026-09-27:
real-time pacing (every run at realtime_factor 1.00), MuJoCo on OSMesa, seeds 0-2. Classic course
65 s with the default 160-call budget. The new courses 90 s with a 240-call budget, the same as
the Laya runs. No call errored.

## Hand-built scenes: 5 / 7 (`scenes.json`)

It misses "tall pillar, left wide open" (climb p=0.60) and "boxed in" (climb p=0.77, brake 0.16).
The README reports 6/7 from an earlier model version. Median latency from this container is
about 0.19 s per call, against 0.11 s in the README.

## Flights (`episodes.jsonl`): runs finished out of 3

|  | classic | pockets | mixed | no-climb |
|---|---|---|---|---|
| oracle (from the layout) | - | 3/3 | 3/3 | 3/3 |
| always climb | 3/3 | 1/3 | 0/3 | 0/3 |
| no model (heuristic) | 1/3 | 0/3 | 0/3 | 0/3 |
| laya-text (zero shot) | 3/3 | 1/3 | 1/3 | 1/3 |
| **jev** | **2/3** | **0/3** | **0/3** | **0/3** |

The control and Laya rows come from `results/laya/README.md`.

- **Classic:** Jev finishes 2/3 (seed 2 reaches x=54.6) with 69-94% of frames on the target,
  close to the README's single run (whole course, 82%).
- **Pockets:** Jev stops at x≈36 on every seed. That is inside the first pocket, just short of
  its far wall (x=38), where always-climb gets trapped too (36.3). Jev climbs into the dead end.
  On a beam and a pocket the JSON differs only in `free_ahead_above_m` (about 15 m, the far wall,
  against 45 m for open air), and nothing in the question says a wall behind the obstacle matters.
- **Mixed / no-climb:** 0/3, stopping at the first pocket or the decoy.

So on these courses Jev does no better than Laya zero-shot or the constant answers. The state
cannot say "dead end", and the `climb` rubric asks only whether there is clear air over the
obstacle. That is the README's own lesson, "the state has to contain the answer", applied to a
new situation. Adding the far-wall range, or "is there a way on past it", to the scene and to the
rubric is the obvious first change, for either model.
