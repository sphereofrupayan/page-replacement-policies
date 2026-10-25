"""Synthetic reference-string generators. All take (rng, N, P, F) and return a list of page ids."""
import numpy as np


def scan(rng, N, P, F, span=4):
    """Sequential runs (64-256 pages) starting at random addresses in a span*P page range."""
    out = []
    while len(out) < N:
        run = int(rng.integers(64, 257))
        start = int(rng.integers(0, span * P - run))
        out.extend(range(start, start + run))
    return out[:N]


def loop(rng, N, P, F, size_factor=1.5, noise=0.05):
    L = int(size_factor * F)
    base = int(rng.integers(0, P - L))
    out = []
    i = 0
    for _ in range(N):
        if rng.random() < noise:
            out.append(int(rng.integers(0, P)))
        else:
            out.append(base + i % L)
            i += 1
    return out


def temporal(rng, N, P, F, depth_frac=0.3, p_new=0.04):
    """LRU-stack model: re-reference depth ~ geometric(mean depth_frac*F)."""
    stack = [int(x) for x in rng.permutation(P)]
    p = 1.0 / (depth_frac * F)
    depths = rng.geometric(p, size=N) - 1
    cold = rng.random(N) < p_new
    rnd = rng.integers(0, P, size=N)
    out = []
    for i in range(N):
        if cold[i]:
            pg = int(rnd[i])
            stack.remove(pg)
        else:
            pg = stack.pop(min(int(depths[i]), P - 1))
        stack.insert(0, pg)
        out.append(pg)
    return out


def zipf(rng, N, P, F, alpha=1.0):
    ranks = np.arange(1, P + 1, dtype=float)
    pr = ranks ** (-alpha)
    pr /= pr.sum()
    perm = rng.permutation(P)
    return [int(x) for x in perm[rng.choice(P, size=N, p=pr)]]


def uniform(rng, N, P, F):
    return [int(x) for x in rng.integers(0, P, size=N)]


def mixed(rng, N, P, F):
    order = [temporal, loop, zipf, scan, temporal, loop]
    seg = N // len(order)
    out = []
    for g in order:
        out.extend(g(rng, seg, P, F))
    return out


WORKLOADS = {"Scan": scan, "Loop": loop, "Temporal": temporal, "Zipf": zipf,
             "Uniform": uniform, "Mixed": mixed}
