"""Deliberately naive, list-based reference implementations used ONLY for cross-validation."""
def ref_fifo(trace, F):
    mem, faults = [], 0
    for p in trace:
        if p not in mem:
            faults += 1
            if len(mem) == F: mem.pop(0)
            mem.append(p)
    return faults

def ref_lru(trace, F):
    mem, faults = [], 0
    for p in trace:
        if p in mem: mem.remove(p)
        else:
            faults += 1
            if len(mem) == F: mem.pop(0)
        mem.append(p)
    return faults

def ref_opt(trace, F):
    mem, faults = [], 0
    for i, p in enumerate(trace):
        if p in mem: continue
        faults += 1
        if len(mem) == F:
            def nxt(q):
                rest = trace[i+1:]
                return rest.index(q) if q in rest else 10**9
            mem.remove(max(mem, key=nxt))
        mem.append(p)
    return faults

def ref_clock(trace, F):
    mem, bit, hand, faults = [], [], 0, 0
    for p in trace:
        if p in mem: bit[mem.index(p)] = 1; continue
        faults += 1
        if len(mem) < F: mem.append(p); bit.append(1); continue
        while True:
            if bit[hand] == 0:
                mem[hand] = p; bit[hand] = 1; hand = (hand + 1) % F; break
            bit[hand] = 0; hand = (hand + 1) % F
    return faults

def ref_lfu(trace, F):
    mem, cnt, last, faults = [], {}, {}, 0
    for t, p in enumerate(trace):
        if p in mem: cnt[p] += 1; last[p] = t; continue
        faults += 1
        if len(mem) == F:
            v = min(mem, key=lambda q: (cnt[q], last[q]))
            mem.remove(v); del cnt[v], last[v]
        mem.append(p); cnt[p] = 1; last[p] = t
    return faults
