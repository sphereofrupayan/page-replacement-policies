"""Validation on controlled reference strings + randomised cross-check against naive reference code."""
import csv, numpy as np
from policies import fifo, lru, clock, lfu, opt, adaptive
import reference_impl as R

cases = {
 "Textbook (Silberschatz) F=3": ([7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1], 3, dict(FIFO=15, LRU=12, OPT=9)),
 "Belady anomaly F=3": ([1,2,3,4,1,2,5,1,2,3,4,5], 3, dict(FIFO=9)),
 "Belady anomaly F=4": ([1,2,3,4,1,2,5,1,2,3,4,5], 4, dict(FIFO=10)),
 "Cyclic 1..4 x5, F=3": ([1,2,3,4]*5, 3, dict(FIFO=20, LRU=20)),
}
rows = []
for name, (tr, F, expected) in cases.items():
    got = dict(FIFO=fifo(tr,F), LRU=lru(tr,F), OPT=opt(tr,F), Clock=clock(tr,F), LFU=lfu(tr,F))
    ref = dict(FIFO=R.ref_fifo(tr,F), LRU=R.ref_lru(tr,F), OPT=R.ref_opt(tr,F), Clock=R.ref_clock(tr,F), LFU=R.ref_lfu(tr,F))
    for k in got:
        exp = expected.get(k, "")
        rows.append(dict(case=name, policy=k, simulator=got[k], reference_code=ref[k], textbook=exp,
                         ok=(got[k]==ref[k]) and (exp=="" or got[k]==exp)))
# randomised agreement
rng = np.random.default_rng(2026)
agree = {k: 0 for k in ["FIFO","LRU","OPT","Clock","LFU"]}
TRIALS = 500
for _ in range(TRIALS):
    P = int(rng.integers(4, 15)); n = int(rng.integers(30, 120)); F = int(rng.integers(2, max(3, P-1)))
    tr = [int(x) for x in rng.integers(0, P, n)]
    pairs = dict(FIFO=(fifo(tr,F),R.ref_fifo(tr,F)), LRU=(lru(tr,F),R.ref_lru(tr,F)), OPT=(opt(tr,F),R.ref_opt(tr,F)),
                 Clock=(clock(tr,F),R.ref_clock(tr,F)), LFU=(lfu(tr,F),R.ref_lfu(tr,F)))
    for k,(a,b) in pairs.items(): agree[k] += (a==b)
# sanity: OPT <= every online policy; adaptive never below OPT
viol = 0
for _ in range(200):
    P = int(rng.integers(6, 30)); n = int(rng.integers(100, 400)); F = int(rng.integers(2, P-1))
    tr = [int(x) for x in rng.integers(0, P, n)]
    o = opt(tr,F)
    viol += any(f(tr,F) < o for f in (fifo,lru,clock,lfu)) or adaptive(tr,F,dict(w_mult=4)) < o
for k,v in agree.items():
    rows.append(dict(case=f"{TRIALS} random traces vs naive reference", policy=k, simulator=v, reference_code=TRIALS, textbook="", ok=(v==TRIALS)))
rows.append(dict(case="OPT lower-bound check (200 traces)", policy="all", simulator=viol, reference_code=0, textbook="", ok=(viol==0)))
with open("../data/validation.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
print("ALL OK:", all(r["ok"] for r in rows))
