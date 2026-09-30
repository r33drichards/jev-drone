"""Laya in its own process, for true wall-clock real-time flights (run.episode timing="wallclock").

With the checkpoints loaded in threads beside the sim (timing="wall"), each predict holds the interpreter
and the CPU while the physics, the controller and the camera wait, so the sim fell behind the wall clock
(rt 0.6-0.9) and slowed the world around Laya. The sim alone runs ~4.5x real time (modal_laya.py::sim_speed),
so the fix is separation: this module starts one server process that owns the GPU and every checkpoint, and
hands the flight a RemoteAgent per checkpoint whose predict() sends the state over a pipe and blocks only
the calling worker thread (waiting on a pipe releases the interpreter). The sim loop keeps running at 1x.

The server answers one request at a time, in arrival order across all pipes: one GPU, the pursuit, reacquire
and altitude questions queued on it as they would be on a drone's computer. Each reply carries the server's
own compute time.

    import laya_server
    laya_server.start()                  # before any backend is built (tactics.shared_laya routes here)
    ...
    laya_server.stats(); laya_server.stop()
"""
import multiprocessing as mp
import threading
import time

N_PIPES = 8           # one per worker thread (locator, reacquirer, altimeter, tactician, warm-ups): plenty

_server = None        # (process, [parent conns])
_pool = []            # free parent conns
_pool_lock = threading.Lock()
_local = threading.local()
_gen = 0              # server generation: a thread's cached pipe from an earlier server is stale
_stats = {"calls": 0, "compute_s": 0.0, "roundtrip_s": 0.0}
_stats_lock = threading.Lock()


SERVER_THREADS = 2    # CPU threads for the server's torch / tokenizer work: the rest are the sim's


def _serve(conns, budgets):
    """The server process: load checkpoints on first use, answer requests one at a time."""
    import os
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(SERVER_THREADS)
    from multiprocessing.connection import wait
    import torch
    torch.set_num_threads(SERVER_THREADS)
    import laya
    agents = {}
    live = list(conns)
    while live:
        for c in wait(live):
            try:
                msg = c.recv()
            except EOFError:
                live.remove(c)
                continue
            if msg[0] == "stop":
                return
            _, name, device, revision, state, questions, kw = msg
            try:
                t0 = time.time()
                key = (name, device, revision)
                if key not in agents:
                    agents[key] = laya.load_vlm(name, device=device, revision=revision, **budgets)
                out = agents[key].predict(state, questions, **kw)
                c.send(("ok", out, time.time() - t0))
            except Exception as e:                      # the flight degrades that answer; never dies
                c.send(("err", "%s: %s" % (type(e).__name__, e), 0.0))


def start():
    """Start the server process (idempotent)."""
    global _server, _gen
    if _server is not None:
        return
    _gen += 1
    from tactics import LAYA_BUDGETS
    ctx = mp.get_context("spawn")        # a fresh interpreter: no forked CUDA state, no shared GIL
    pairs = [ctx.Pipe() for _ in range(N_PIPES)]
    p = ctx.Process(target=_serve, args=([b for _, b in pairs], dict(LAYA_BUDGETS)), daemon=True)
    p.start()
    _server = (p, [a for a, _ in pairs])
    _pool[:] = [a for a, _ in pairs]


def reclaim():
    """Hand every pipe back to the pool and invalidate the pipes cached by earlier threads. Call at the start of
    each flight: a container reused for the next flight keeps the server (and its loaded checkpoints), but the
    last flight's worker threads, now closed, never returned their pipes (with N_PIPES pipes and ~4 workers per
    flight, the second flight ran out and every Laya call failed)."""
    global _gen
    if _server is None:
        return
    with _pool_lock:
        _gen += 1
        _pool[:] = list(_server[1])


def active():
    return _server is not None


def _conn():
    c = getattr(_local, "conn", None)
    if getattr(_local, "gen", None) != _gen:
        c = None
    if c is None:
        with _pool_lock:
            if not _pool:
                raise RuntimeError("laya_server: more worker threads than pipes (N_PIPES=%d)" % N_PIPES)
            c = _local.conn = _pool.pop()
            _local.gen = _gen
    return c


class RemoteAgent:
    """tactics._LockedAgent's interface for a checkpoint served by the server process."""

    def __init__(self, name, device=None, revision=None):
        self.name, self.device, self.revision = name, device, revision
        self.users = 0

    def predict(self, state, questions, **kw):
        c = _conn()
        t0 = time.time()
        c.send(("predict", self.name, self.device, self.revision, state, questions, kw))
        status, out, compute = c.recv()
        with _stats_lock:
            _stats["calls"] += 1
            _stats["compute_s"] += compute
            _stats["roundtrip_s"] += time.time() - t0
        if status != "ok":
            raise RuntimeError(out)
        return out


def stats(flew_s=None):
    with _stats_lock:
        n = max(_stats["calls"], 1)
        out = {"calls": _stats["calls"], "mean_compute_s": round(_stats["compute_s"] / n, 4),
               "mean_roundtrip_s": round(_stats["roundtrip_s"] / n, 4)}
        if flew_s:
            out["gpu_busy_pct"] = round(100 * _stats["compute_s"] / flew_s, 1)
        return out


def reset_stats():
    with _stats_lock:
        _stats.update(calls=0, compute_s=0.0, roundtrip_s=0.0)


def stop():
    """Stop the server process and release every pipe."""
    global _server
    if _server is None:
        return
    p, conns = _server
    try:
        conns[0].send(("stop",))
    except Exception:
        pass
    p.join(timeout=5)
    if p.is_alive():
        p.kill()
    _server = None
    _pool.clear()
