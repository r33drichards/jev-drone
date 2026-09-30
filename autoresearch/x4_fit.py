"""autoresearch X4: can the teacher's choice be predicted from what a student could observe, or only from what
the teacher alone knows? Reads runs/X4-data.jsonl (teacher_features), fits per feature set a softmax regression and
a k-nearest-neighbour classifier, leave-one-seed-out, and prints held-out top-1 for the turn, speed and slide level.

    python autoresearch/x4_fit.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import command  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = {"turn": command.TURN_LEVELS, "speed": command.SPEED_LEVELS, "slide": command.SLIDE_LEVELS}


def obs_vec(o):
    b = np.deg2rad(o["bearing"]) if o["visible"] else 0.0
    v = float(o["visible"])
    tr = np.deg2rad(o["travel"])
    return [v, v * np.sin(b), v * np.cos(b), v * min(o["range"] or 0.0, 30.0) / 30.0, o["speed"] / 7.0,
            np.sin(tr), np.cos(tr), o["alt"] / 2.0] + [min(x, 20.0) / 20.0 for x in o["lidar"]]


def priv_vec(p):
    return [x / 20.0 for k in ("now", "fut1", "fut2", "fut3") for x in p[k]] + [x / 7.0 for x in p["vel"]]


def rows(data):
    X = {"obs": [], "obs+memory": [], "obs+priv": []}
    Y = {k: [] for k in LEVELS}
    G = []
    for f in data:
        d = f["decisions"]
        for i, dd in enumerate(d):
            ft = dd["feat"]
            o = obs_vec(ft["obs"])
            hist = []
            for j in (i - 1, i - 2):          # the two previous decisions' observations (what a memory would hold)
                hist += obs_vec(d[j]["feat"]["obs"])[:4] if j >= 0 else [0.0] * 4
            X["obs"].append(o)
            X["obs+memory"].append(o + hist)
            X["obs+priv"].append(o + priv_vec(ft["priv"]))
            for k, lv in LEVELS.items():
                Y[k].append(int(np.argmin([abs(ft["choice"][k] - x) for x in lv])))
            G.append(f["seed"])
    return {k: np.asarray(v, dtype=float) for k, v in X.items()}, {k: np.asarray(v) for k, v in Y.items()}, np.asarray(G)


def softmax_fit(X, y, k, l2=1e-3, steps=600, lr=0.5):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    W = np.zeros((Xb.shape[1], k))
    Y = np.eye(k)[y]
    for _ in range(steps):
        z = Xb @ W
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        W -= lr * (Xb.T @ (p - Y) / len(X) + l2 * W)
    return lambda Xt: np.argmax(np.hstack([Xt, np.ones((len(Xt), 1))]) @ W, 1)


def knn_fit(X, y, kk=15):
    def pred(Xt):
        out = []
        for i in range(0, len(Xt), 512):
            d = ((Xt[i:i + 512, None, :] - X[None, :, :]) ** 2).sum(-1)
            nn = np.argsort(d, 1)[:, :kk]
            out += [np.bincount(y[r], minlength=y.max() + 1).argmax() for r in nn]
        return np.asarray(out)
    return pred


def cv(X, y, G, fit):
    hit = []
    for g in np.unique(G):
        tr, te = G != g, G == g
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        f = fit((X[tr] - mu) / sd, y[tr])
        hit += list(f((X[te] - mu) / sd) == y[te])
    return float(np.mean(hit))


def main():
    data = [json.loads(line) for line in open(os.path.join(HERE, "runs", "X4-data.jsonl"))]
    X, Y, G = rows(data)
    res = {"decisions": int(len(G)), "flights": len(data)}
    print("%d decisions from %d teacher flights; held-out by seed" % (len(G), len(data)))
    print("%-11s %-6s %8s %8s %8s" % ("features", "target", "majority", "softmax", "knn"))
    for fs, Xf in X.items():
        for k, y in Y.items():
            maj = float(np.bincount(y).max() / len(y))
            a = cv(Xf, y, G, lambda Xt, yt: softmax_fit(Xt, yt, len(LEVELS[k])))
            b = cv(Xf, y, G, knn_fit)
            res.setdefault(fs, {})[k] = {"majority": round(maj, 3), "softmax": round(a, 3), "knn": round(b, 3)}
            print("%-11s %-6s %8.3f %8.3f %8.3f" % (fs, k, maj, a, b))
    json.dump(res, open(os.path.join(HERE, "runs", "X4-fit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
