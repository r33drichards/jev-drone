"""Render and rank the ACH matrix in autoresearch/ach.json (program.md).

    python autoresearch/ach.py show     # matrix, ranking by weighted inconsistency, sensitivity, experiment queue
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT = {"CC": 0.0, "C": 0.0, "N": 0.0, "I": 1.0, "II": 2.0}
CRED = {"high": 1.0, "medium": 0.6, "low": 0.3}


def load(path=os.path.join(HERE, "ach.json")):
    return json.load(open(path))


def scores(ach, skip=()):
    """Weighted inconsistency per hypothesis (lower = more likely), leaving out evidence ids in `skip`."""
    s = {h: 0.0 for h in ach["hypotheses"]}
    for e in ach["evidence"]:
        if e["id"] in skip:
            continue
        for h in s:
            s[h] += WEIGHT[e["ratings"].get(h, "N")] * CRED[e.get("credibility", "medium")]
    return s


def diagnostic(e, hyps):
    """An evidence row is diagnostic when its ratings differ across hypotheses."""
    return len({WEIGHT[e["ratings"].get(h, "N")] for h in hyps}) > 1


def sensitivity(ach):
    """For the ranking's top two: the single evidence rows whose removal changes which one leads, or brings a
    third hypothesis to within the leader's score."""
    def leaders(s):
        return sorted(h for h in s if s[h] <= min(s.values()) + 1e-9)

    top = leaders(scores(ach))
    out = []
    for e in ach["evidence"]:
        alt = leaders(scores(ach, skip={e["id"]}))
        if alt != top:
            out.append("%s: without it the lead is %s instead of %s" % (e["id"], "=".join(alt), "=".join(top)))
    return out


def show(ach):
    hyps = list(ach["hypotheses"])
    print("Q:", ach["question"], "\n")
    print("%-5s %-5s %s" % ("id", "cred", " ".join("%-3s" % h for h in hyps)) + "  diag")
    for e in ach["evidence"]:
        print("%-5s %-5s %s" % (e["id"], e.get("credibility", "medium")[:4],
                                " ".join("%-3s" % e["ratings"].get(h, "N") for h in hyps))
              + "  " + ("yes" if diagnostic(e, hyps) else "no"))
    s = scores(ach)
    print("\nranking (weighted inconsistency, lower = more likely):")
    for h in sorted(s, key=s.get):
        print("  %-3s %5.1f  %s" % (h, s[h], ach["hypotheses"][h]))
    sens = sensitivity(ach)
    print("\nsensitivity:", "; ".join(sens) if sens else "no single evidence row changes the leader")
    print("\nexperiments:")
    for x in ach["experiments"]:
        print("  %-4s %-9s %-28s %s" % (x["id"], x["status"], x.get("cost", "")[:28], x["what"][:90]))
    c = ach.get("conclusions", {})
    if c.get("summary"):
        print("\nconclusions:", c["summary"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show(load())
    else:
        print(__doc__)
