# autoresearch: why Laya cannot fly the 4x rover, by Analysis of Competing Hypotheses

An autonomous research loop on one question, run with Heuer's
[Analysis of Competing Hypotheses](https://en.wikipedia.org/wiki/Analysis_of_competing_hypotheses) (ACH) instead of
a single metric to push. The state is `ach.json` (hypotheses, evidence, the matrix, the experiment queue);
`python autoresearch/ach.py show` renders it and ranks the hypotheses; `runs/<id>.md` records each experiment.

## The question

The lookahead teacher (`teacher.py`, privileged: knows where the rover is and will be) follows the 4x rover on all
six 4x courses (70/72; 36/36 even when it holds its heading between decisions as the student must). Laya flying
by command (`policy="laya-cmd"`, v3.5 soft targets, v3.6 sharp targets, rl1-rl4 outcome-reward RL) finishes none
(0/30, 0/31), keeps the rover in view 10-20% of the time and collides 2-3 times a flight. **What is the cause, and
what fixes it?** Success = a Laya checkpoint that, flying itself (`laya-cmd-wc`, wall clock), finishes a majority
of 4x flights, measured on seeds it never trained on.

## ACH, as this loop runs it

1. **Hypotheses.** Keep the full set of plausible causes in `ach.json`, mutually distinguishable. Add one whenever
   evidence fits none well. Never delete: a rejected hypothesis stays, with the evidence that rejected it.
2. **Evidence.** Every result (an experiment, an earlier run in `results/`, a code fact) is an evidence row with its
   source (commit, result path) and a credibility (high / medium / low).
3. **Diagnostics.** Rate each evidence row against EVERY hypothesis, one row at a time: `CC` very consistent,
   `C` consistent, `N` neutral / not applicable, `I` inconsistent, `II` very inconsistent. Evidence consistent with
   all hypotheses is not diagnostic and does not move the ranking; say so and do not over-weight it.
4. **Refinement.** Choose the next experiment for its **diagnosticity**: the result that would be inconsistent with
   some leading hypotheses and consistent with others, cheapest first. Write the prediction of each hypothesis
   BEFORE running (`predictions` in the experiment entry). That is what makes the result evidence rather than a
   story told afterwards.
5. **Inconsistency.** Rank hypotheses by weighted inconsistency (`ach.py`: `I` = 1, `II` = 2, times credibility
   1 / 0.6 / 0.3), lowest = most likely. Try to disprove the leader, not confirm it.
6. **Sensitivity.** `ach.py show` also reports, for the top hypotheses, which single evidence rows would flip the
   ranking if wrong. A conclusion that rests on one low-credibility row gets that row re-tested before acting on it.
7. **Conclusions and milestones.** When one hypothesis leads clearly and is robust, implement the fix it implies as
   an **intervention experiment** (its predicted outcome written first); its result is evidence like any other.
   Keep `conclusions` in `ach.json` current: the ranking, why the others are rejected, and the milestones that
   would change the conclusion.

## Rules

- GPU fan-out stays at 10 or fewer (the Modal functions cap it with `GPU_MAX`).
- Real footage (UAV123, VisDrone) is for evaluation only; never train on it.
- Every experiment gets a commit (code) and its results committed under `results/` and `autoresearch/runs/`; push
  to `claude/busy-keller-knhbhh` (PR r33drichards/jev-drone#1). No model identifiers in commits.
- Cheapest diagnostic first: CPU teacher-world checks and code audits before GPU flights, flights before training.
- Evaluation seeds stay disjoint from training seeds (teacher data 20-61, RL 100-102; evaluate on 0-11).
- Stop and ask the human before anything that costs more than ~2 GPU-hours in one go, deletes data, or changes
  the success criterion.

## The loop

LOOP FOREVER (self-paced):

1. `python autoresearch/ach.py show`: the ranking, sensitivity and queue.
2. If an experiment is running, check it; when done, write `runs/<id>.md` (setup, predictions, result), add the
   evidence row(s) with ratings for every hypothesis, commit, push.
3. Otherwise pick the most diagnostic affordable experiment from the queue (or add one), record its predictions,
   commit, launch it in the background.
4. Update `conclusions`. Report to the human only on a real change: a hypothesis rejected, a new leader, an
   intervention result, or a blocker.
