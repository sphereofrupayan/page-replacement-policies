"""Collect table values for the Word report (reads the CSVs produced by the experiment)."""
import json, struct, pandas as pd
D = "../data/"
s = pd.read_csv(D + "summary_by_cell.csv"); m = pd.read_csv(D + "summary_macro.csv"); t = pd.read_csv(D + "paired_tests.csv")
WL = ["Scan", "Loop", "Temporal", "Zipf", "Uniform", "Mixed"]; ORDER = ["FIFO", "LRU", "Clock", "LFU", "Adaptive", "OPT"]
fixed = ["FIFO", "LRU", "Clock", "LFU"]
sm = s[s.point == "M"]
t4 = []
for w in WL:
    r = sm[sm.workload == w].set_index("policy")
    best = r.loc[fixed, "hit_mean"].idxmax()
    t4.append([w] + [dict(v=f"{r.loc[p,'hit_mean']:.3f}", sd=f"{r.loc[p,'hit_sd']:.3f}", best=(p == best)) for p in ORDER])
t5 = []
for p in ORDER:
    row = [p]
    for pt in "SML":
        row.append(f"{m[(m.point==pt)&(m.policy==p)].macro_hit.iloc[0]:.3f}")
    mm = m[(m.point == "M") & (m.policy == p)].iloc[0]
    row.append("-" if p == "OPT" else f"{mm.max_regret:.3f} ({mm.worst_workload})")
    row.append(f"{mm.mean_gap_to_opt:.3f}")
    for pt in "SML":
        row.append(f"{m[(m.point==pt)&(m.policy==p)].macro_time.iloc[0]*1000:.1f}")
    t5.append(row)
regret = {pt: {p: float(m[(m.point == pt) & (m.policy == p)].max_regret.iloc[0]) for p in ORDER} for pt in "SML"}
def png_size(path):
    with open(path, "rb") as f:
        b = f.read(24)
    return struct.unpack(">II", b[16:24])
imgs = {k: png_size(f"../figures/{k}.png") for k in ["fig1_workflow", "fig2_adaptive_flow", "fig3_patterns", "fig4_hit_ratio_M", "fig5_scaling", "fig6_timeline"]}
mixed = {pt: float(t[(t.point == pt) & (t.workload == "Mixed") & (t.vs == "LRU")].mean_diff.iloc[0]) for pt in "SML"}
json.dump(dict(t4=t4, t5=t5, regret=regret, imgs=imgs, mixed_gain=mixed), open(D + "doc_data.json", "w"), indent=1)
print(json.dumps(dict(t5=t5, mixed=mixed, regret=regret)))
