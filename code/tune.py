"""Parameter tuning of the adaptive selector on SEPARATE tuning seeds (100-103), operating point M."""
import itertools, json, numpy as np, pandas as pd
from policies import adaptive
from workloads import WORKLOADS
F, P, N = 128, 1600, 60000
SEEDS = [100, 101, 102, 103]
traces = {(w, s): g(np.random.default_rng(s), N, P, F) for w, g in WORKLOADS.items() for s in SEEDS}
rows = []
for wm, rt, ut, ag in itertools.product([4, 8, 16], [0.45, 0.55, 0.65], [0.10, 0.25], ["none", "halve", "reset"]):
    prm = dict(w_mult=wm, rec_thr=rt, reuse_thr=ut, age=ag)
    per = {w: np.mean([1 - adaptive(traces[(w, s)], F, prm) / N for s in SEEDS]) for w in WORKLOADS}
    rows.append(dict(**prm, **per, macro=np.mean(list(per.values()))))
df = pd.DataFrame(rows).sort_values("macro", ascending=False)
df.to_csv("../data/tuning_sweep.csv", index=False)
print(df.head(10).round(4).to_string(index=False))
print(df.tail(3).round(4).to_string(index=False))
