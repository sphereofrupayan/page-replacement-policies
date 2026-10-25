"""Statistical summaries + all figures for the report."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy import stats

from policies import adaptive
from workloads import WORKLOADS

FIG = "../figures/"
DATA = "../data/"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
COL = {"FIFO": "#8c8c8c", "LRU": "#1f77b4", "Clock": "#17becf", "LFU": "#d62728", "OPT": "#2ca02c", "Adaptive": "#ff7f0e"}
ORDER = ["FIFO", "LRU", "Clock", "LFU", "Adaptive", "OPT"]
WL = ["Scan", "Loop", "Temporal", "Zipf", "Uniform", "Mixed"]

df = pd.read_csv(DATA + "raw_results.csv")

# ------------------------------------------------------------------ summaries
g = df.groupby(["point", "workload", "policy"])
summ = g.agg(hit_mean=("hit_ratio", "mean"), hit_sd=("hit_ratio", "std"),
             faults_mean=("faults", "mean"), time_mean=("time_s", "mean"), time_sd=("time_s", "std"),
             n=("seed", "count")).reset_index()
summ["hit_ci95"] = stats.t.ppf(0.975, summ["n"] - 1) * summ["hit_sd"] / np.sqrt(summ["n"])
summ["time_ci95"] = stats.t.ppf(0.975, summ["n"] - 1) * summ["time_sd"] / np.sqrt(summ["n"])
summ.to_csv(DATA + "summary_by_cell.csv", index=False)

fixed = ["FIFO", "LRU", "Clock", "LFU"]
rows = []
for pt in ["S", "M", "L"]:
    s = summ[summ.point == pt]
    for pol in ORDER:
        sp = s[s.policy == pol].set_index("workload")
        best_fixed = s[s.policy.isin(fixed)].groupby("workload").hit_mean.max()
        regret = (best_fixed - sp.hit_mean).clip(lower=0)
        opt_gap = s[s.policy == "OPT"].set_index("workload").hit_mean - sp.hit_mean
        rows.append(dict(point=pt, policy=pol, macro_hit=sp.hit_mean.mean(),
                         max_regret=regret.max() if pol != "OPT" else 0.0,
                         worst_workload=regret.idxmax() if pol != "OPT" else "-",
                         mean_gap_to_opt=opt_gap.mean(), macro_time=sp.time_mean.mean(),
                         mixed_hit=sp.hit_mean["Mixed"], mixed_time=sp.time_mean["Mixed"]))
macro = pd.DataFrame(rows)
macro.to_csv(DATA + "summary_macro.csv", index=False)

# paired tests, adaptive vs LRU / best fixed (by mean) per (point, workload)
trows = []
for pt in ["S", "M", "L"]:
    for w in WL:
        d = df[(df.point == pt) & (df.workload == w)].pivot(index="seed", columns="policy", values="hit_ratio")
        bf = d[fixed].mean().idxmax()
        for other in ["LRU", bf]:
            diff = d["Adaptive"] - d[other]
            if np.allclose(diff, 0):
                p_w, p_t = 1.0, 1.0
            else:
                try:
                    p_w = stats.wilcoxon(d["Adaptive"], d[other]).pvalue
                except ValueError:
                    p_w = 1.0
                p_t = stats.ttest_rel(d["Adaptive"], d[other]).pvalue
            trows.append(dict(point=pt, workload=w, vs=other, best_fixed=bf, mean_diff=diff.mean(),
                              wilcoxon_p=p_w, paired_t_p=p_t))
tests = pd.DataFrame(trows)
tests.to_csv(DATA + "paired_tests.csv", index=False)

# friedman test across policies on Mixed, M
d = df[(df.point == "M") & (df.workload == "Mixed")].pivot(index="seed", columns="policy", values="hit_ratio")
fried = stats.friedmanchisquare(*[d[p] for p in ["FIFO", "LRU", "Clock", "LFU", "Adaptive"]])
json.dump(dict(friedman_stat=float(fried.statistic), friedman_p=float(fried.pvalue)), open(DATA + "friedman.json", "w"))

# ------------------------------------------------------------------ Fig 1: workflow
def box(ax, x, y, w, h, text, fc="#e8f1fb", ec="#1f4e79", fs=5.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))

fig, ax = plt.subplots(figsize=(7.2, 2.0))
ax.set_xlim(0, 16); ax.set_ylim(0, 4.4); ax.axis("off")
box(ax, 0.1, 1.5, 2.7, 1.2, "Workload generators\n(6 patterns, seeded)", "#fdf0e0", "#a65e00")
box(ax, 3.4, 1.5, 2.7, 1.2, "Reference strings\nN = 20k / 60k / 150k", "#fdf0e0", "#a65e00")
box(ax, 6.9, 3.15, 3.9, 0.95, "Fixed policies: FIFO, LRU,\nClock, LFU", "#e8f1fb")
box(ax, 6.9, 1.85, 3.9, 0.85, "Belady OPT\n(offline lower bound)", "#e6f4e6", "#2a7a2a")
box(ax, 6.9, 0.3, 3.9, 1.2, "Adaptive engine\nmonitor -> classify -> switch", "#ffe9d6", "#a65e00", bold=True)
box(ax, 11.5, 1.5, 2.2, 1.2, "Metrics\nfaults, hit ratio,\nexec. time", "#f0e8f7", "#5b2c83")
box(ax, 14.1, 1.5, 1.8, 1.2, "CSV, stats\nand plots\n(10 seeds)", "#f0e8f7", "#5b2c83")
arrow(ax, 2.8, 2.1, 3.4, 2.1)
for yy in (3.6, 2.27, 0.9):
    arrow(ax, 6.1, 2.1, 6.9, yy)
    arrow(ax, 10.8, yy, 11.5, 2.1)
arrow(ax, 13.7, 2.1, 14.1, 2.1)
ax.text(8.0, 4.3, "Same machine, interpreter and trace for every policy", ha="center", fontsize=6.6, style="italic", color="#444")
fig.savefig(FIG + "fig1_workflow.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 2: adaptive flow
def diamond(ax, cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx - w, cy), (cx, cy + h), (cx + w, cy), (cx, cy - h)], closed=True, fc="#fff7d6", ec="#8a6d00", lw=1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=6.2)

fig, ax = plt.subplots(figsize=(7.2, 2.5))
ax.set_xlim(0, 16); ax.set_ylim(0, 5); ax.axis("off")
box(ax, 0.1, 2.1, 2.5, 1.0, "Incoming page\nreferences", "#fdf0e0", "#a65e00")
box(ax, 3.1, 2.1, 2.5, 1.0, "Collect window\nW = 4 x frames", "#e8f1fb")
box(ax, 6.1, 2.1, 2.5, 1.0, "Compute features\nseq, reuse, rec, cov", "#e8f1fb")
arrow(ax, 2.6, 2.6, 3.1, 2.6); arrow(ax, 5.6, 2.6, 6.1, 2.6)
diamond(ax, 10.3, 3.9, 1.5, 0.8, "seq >= 0.45 and\nreuse < 0.10 ?")
diamond(ax, 10.3, 2.35, 1.5, 0.7, "rec >= 0.55 ?")
arrow(ax, 8.6, 2.8, 9.2, 3.6)
arrow(ax, 10.3, 3.1, 10.3, 3.05)
box(ax, 12.3, 3.6, 1.4, 0.6, "FIFO", "#eeeeee", "#555", bold=True)
box(ax, 12.3, 2.05, 1.4, 0.6, "LRU", "#dcebf9", "#1f77b4", bold=True)
box(ax, 9.5, 0.7, 1.6, 0.6, "LFU", "#fbdcdc", "#d62728", bold=True)
arrow(ax, 11.8, 3.9, 12.3, 3.9); arrow(ax, 11.8, 2.35, 12.3, 2.35); arrow(ax, 10.3, 1.65, 10.3, 1.3)
ax.text(11.9, 4.0, "yes", fontsize=6.2); ax.text(11.9, 2.45, "yes", fontsize=6.2)
ax.text(10.4, 3.03, "no", fontsize=6.2); ax.text(10.4, 1.4, "no", fontsize=6.2)
box(ax, 14.1, 0.7, 1.8, 3.5, "Switch at\nwindow end;\nresident pages\nand metadata\nare kept", "#e6f4e6", "#2a7a2a", fs=5.4)
arrow(ax, 13.7, 3.9, 14.1, 3.9); arrow(ax, 13.7, 2.35, 14.1, 2.35); arrow(ax, 11.1, 1.0, 14.1, 1.0)
ax.plot([15.0, 15.0], [0.7, 0.2], color="#333", lw=1.1); ax.plot([15.0, 4.35], [0.2, 0.2], color="#333", lw=1.1)
arrow(ax, 4.35, 0.2, 4.35, 2.1)
ax.text(0.1, 4.7, "Window size, thresholds and counter-aging chosen on separate tuning seeds (100-103).",
        fontsize=6.4, style="italic", color="#444")
fig.savefig(FIG + "fig2_adaptive_flow.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 3: patterns
fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.0), sharey=False)
cfg = dict(N=20000, F=64, P=800)
for ax, w in zip(axes.ravel(), WL):
    tr = WORKLOADS[w](np.random.default_rng(0), cfg["N"], cfg["P"], cfg["F"])
    seg = tr[:3000] if w != "Mixed" else tr[::7]
    ax.plot(range(len(seg)), seg, color="#1f4e79", marker=".", markersize=0.8, linestyle="none")
    ax.set_title(w if w != "Mixed" else "Mixed (6 phases, sampled)", fontsize=8)
    ax.set_xlabel("reference index", fontsize=7); ax.set_ylabel("page", fontsize=7)
    ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(FIG + "fig3_patterns.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 4: hit ratio bars (M)
fig, ax = plt.subplots(figsize=(7.2, 2.5))
s = summ[summ.point == "M"]
width = 0.13
for i, pol in enumerate(ORDER):
    sp = s[s.policy == pol].set_index("workload").loc[WL]
    ax.bar(np.arange(len(WL)) + (i - 2.5) * width, sp.hit_mean, width, yerr=sp.hit_ci95, capsize=1.5,
           color=COL[pol], label=pol, edgecolor="white", linewidth=0.3, error_kw=dict(lw=0.6))
ax.set_xticks(np.arange(len(WL))); ax.set_xticklabels(WL)
ax.set_ylabel("hit ratio (mean ± 95% CI)"); ax.set_ylim(0, 1.0)
ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.savefig(FIG + "fig4_hit_ratio_M.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 5: scaling
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
pts = ["S", "M", "L"]; Ns = [20000, 60000, 150000]
for pol in ORDER:
    v = [summ[(summ.point == p) & (summ.workload == "Mixed") & (summ.policy == pol)].hit_mean.iloc[0] for p in pts]
    axes[0].plot(Ns, v, marker="o", ms=3, color=COL[pol], label=pol, lw=1.2)
    v2 = [macro[(macro.point == p) & (macro.policy == pol)].macro_hit.iloc[0] for p in pts]
    axes[1].plot(Ns, v2, marker="o", ms=3, color=COL[pol], lw=1.2)
    v3 = [macro[(macro.point == p) & (macro.policy == pol)].macro_time.iloc[0] for p in pts]
    axes[2].plot(Ns, v3, marker="o", ms=3, color=COL[pol], lw=1.2)
axes[0].set_title("Mixed workload: hit ratio", fontsize=8); axes[1].set_title("Mean over 6 workloads: hit ratio", fontsize=8)
axes[2].set_title("Mean execution time (s)", fontsize=8)
for a in axes:
    a.set_xlabel("references N", fontsize=7); a.set_xscale("log"); a.tick_params(labelsize=6)
    a.set_xticks(Ns); a.set_xticklabels(["20k", "60k", "150k"]); a.minorticks_off()
axes[2].set_yscale("log")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=6, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
fig.savefig(FIG + "fig5_scaling.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 6: timeline on Mixed (M, seed 0)
tl = pd.read_csv(DATA + "adaptive_timeline_mixed_M_seed0.csv")
cfgM = dict(N=60000, F=128, P=1600)
tr = WORKLOADS["Mixed"](np.random.default_rng(0), cfgM["N"], cfgM["P"], cfgM["F"])
F, Wn = cfgM["F"], 4 * cfgM["F"]

def flags_lru(trace, F):
    from collections import OrderedDict
    od, out = OrderedDict(), []
    for p in trace:
        if p in od:
            od.move_to_end(p); out.append(0)
        else:
            if len(od) >= F: od.popitem(last=False)
            od[p] = None; out.append(1)
    return out

def flags_lfu(trace, F):
    import heapq
    freq, tl_, heap, res, out, t = {}, {}, [], set(), [], 0
    for p in trace:
        t += 1
        if p in res:
            freq[p] += 1; tl_[p] = t; heapq.heappush(heap, (freq[p], t, p)); out.append(0); continue
        if len(res) >= F:
            while True:
                f, tt, q = heapq.heappop(heap)
                if q in res and freq[q] == f and tl_[q] == tt: break
            res.discard(q); del freq[q], tl_[q]
        res.add(p); freq[p] = 1; tl_[p] = t; heapq.heappush(heap, (1, t, p)); out.append(1)
    return out

def per_window(flags, W):
    a = np.asarray(flags)[: (len(flags) // W) * W].reshape(-1, W)
    return 1 - a.mean(axis=1)

wl, wf = per_window(flags_lru(tr, F), Wn), per_window(flags_lfu(tr, F), Wn)
x = np.arange(1, len(tl) + 1) * Wn
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 2.7), sharex=True, gridspec_kw=dict(height_ratios=[0.35, 1]))
pol_y = {"FIFO": 0, "LRU": 1, "LFU": 2}
a1.step(x, [pol_y[p] for p in tl.policy], where="post", color=COL["Adaptive"], lw=1.4)
a1.set_yticks([0, 1, 2]); a1.set_yticklabels(["FIFO", "LRU", "LFU"], fontsize=6.5); a1.set_ylabel("active", fontsize=7)
names = ["Temporal", "Loop", "Zipf", "Scan", "Temporal", "Loop"]
for i, nm in enumerate(names):
    for a in (a1, a2):
        a.axvline(i * 10000, color="#bbb", lw=0.6, ls="--")
    a2.text(i * 10000 + 5000, 1.03, nm, ha="center", fontsize=6.8, color="#444")
a2.plot(x, wl[: len(x)], color=COL["LRU"], lw=1, label="LRU")
a2.plot(x, wf[: len(x)], color=COL["LFU"], lw=1, label="LFU")
a2.plot(x, tl.hit.values, color=COL["Adaptive"], lw=1.6, label="Adaptive")
a2.set_ylim(0, 1.12); a2.set_ylabel("window hit ratio", fontsize=7); a2.set_xlabel("references (phase boundaries dashed)", fontsize=7)
a2.legend(ncol=3, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32))
fig.tight_layout(h_pad=0.3)
fig.savefig(FIG + "fig6_timeline.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ console report
pd.set_option("display.width", 200)
print(macro.round(4).to_string(index=False))
print(tests[tests.point == "M"].round(4).to_string(index=False))
print("friedman", fried)
for pt in ["S", "M", "L"]:
    print(pt)
    print(summ[summ.point == pt].pivot(index="workload", columns="policy", values="hit_mean")[ORDER].round(3))
