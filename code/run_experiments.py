"""Principal experiment: 6 workloads x 3 operating points x 10 seeds x 6 policies -> raw CSV."""
import csv, json, time, numpy as np
from policies import fifo, lru, clock, lfu, opt, adaptive, DEFAULT_PARAMS
from workloads import WORKLOADS

POINTS = {"S": dict(N=20000, F=64, P=800), "M": dict(N=60000, F=128, P=1600), "L": dict(N=150000, F=256, P=3200)}
SEEDS = list(range(10))
POLICIES = ["FIFO", "LRU", "Clock", "LFU", "OPT", "Adaptive"]
FN = {"FIFO": fifo, "LRU": lru, "Clock": clock, "LFU": lfu}

json.dump(dict(points=POINTS, seeds=SEEDS, workloads=list(WORKLOADS), adaptive_params=DEFAULT_PARAMS,
               tuning_seeds=[100, 101, 102, 103]), open("../config/experiment_config.json", "w"), indent=2)

rows, timeline = [], []
t_start = time.time()
for pt, cfg in POINTS.items():
    N, F, P = cfg["N"], cfg["F"], cfg["P"]
    for wname, gen in WORKLOADS.items():
        for seed in SEEDS:
            tr = gen(np.random.default_rng(seed), N, P, F)
            for pol in POLICIES:
                rec = [] if (pol == "Adaptive" and wname == "Mixed" and seed == 0 and pt == "M") else None
                t0 = time.perf_counter()
                if pol == "OPT":
                    faults = opt(tr, F)
                elif pol == "Adaptive":
                    faults = adaptive(tr, F, record=rec)
                else:
                    faults = FN[pol](tr, F)
                dt = time.perf_counter() - t0
                rows.append(dict(point=pt, N=N, frames=F, pages=P, workload=wname, seed=seed, policy=pol,
                                 faults=faults, hit_ratio=round(1 - faults / N, 6), time_s=round(dt, 5)))
                if rec is not None:
                    timeline = rec
        print(pt, wname, "done", round(time.time() - t_start), "s", flush=True)

with open("../data/raw_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open("../data/adaptive_timeline_mixed_M_seed0.csv", "w", newline="") as f:
    tl = [r for r in timeline if "t" in r]
    w = csv.DictWriter(f, fieldnames=list(tl[0].keys())); w.writeheader(); w.writerows(tl)
print("total switches (Mixed, M, seed0):", [r for r in timeline if "switches" in r])
print("finished in", round(time.time() - t_start), "s")
