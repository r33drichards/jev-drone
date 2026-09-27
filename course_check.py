"""Fly the control policies on the courses, in parallel, and print who finishes.

A course is useful only if the oracle finishes it and no constant answer does:

    MUJOCO_GL=osmesa python course_check.py --courses pockets,mixed --seeds 0 1 2

Runs faster than real time (the controls answer instantly, so pacing changes nothing).
"""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

POLICIES = {"no-model": ["--no-jev"], "oracle": ["--backend", "const:oracle"],
            "climb": ["--backend", "const:climb"], "hold": ["--backend", "const:hold_course"],
            "gap_left": ["--backend", "const:gap_left"], "gap_right": ["--backend", "const:gap_right"],
            "brake": ["--backend", "const:brake"], "reacquire": ["--backend", "const:reacquire"]}


def fly(course, policy, seed, seconds):
    cmd = [sys.executable, "run.py", "--fast", "--seconds", str(seconds), "--seeds", str(seed),
           "--course", course] + POLICIES[policy]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()
    return course, policy, seed, (json.loads(out[-1]) if out else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--courses", default="pockets,mixed,no-climb")
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seconds", type=float, default=65.0)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    jobs = [(c, pol, s, a.seconds) for c in a.courses.split(",") for pol in a.policies.split(",") for s in a.seeds]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: fly(*j), jobs))
    rows = {}
    for c, pol, s, r in res:
        rows.setdefault((c, pol), []).append(r)
    print("%-9s %-10s %-8s %s" % ("course", "policy", "finished", "max_x per seed (m) / collisions"))
    for (c, pol), rs in rows.items():
        fin = sum(1 for r in rs if r and r["finished_at_s"] is not None)
        print("%-9s %-10s %d/%d      %s" % (c, pol, fin, len(rs), "  ".join(
            "%5.1f/%d" % (r["max_x_m"], r["collisions"]) if r else "ERR" for r in rs)), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            for c, pol, s, r in res:
                f.write(json.dumps(dict(course=c, policy=pol, seed=s, result=r)) + "\n")


if __name__ == "__main__":
    main()
