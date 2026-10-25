"""Page-replacement simulators: FIFO, LRU, OPT (Belady), Clock, LFU and the adaptive selector.
Every policy exposes the same signature: simulate(trace, frames) -> number of page faults."""
import heapq
from collections import OrderedDict, deque
import numpy as np


def next_use_array(trace):
    """nxt[i] = index of the next reference to trace[i], or len(trace) if none."""
    n = len(trace)
    nxt = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        p = trace[i]
        nxt[i] = last.get(p, n)
        last[p] = i
    return nxt


def fifo(trace, F):
    q, res, faults = deque(), set(), 0
    for p in trace:
        if p in res:
            continue
        faults += 1
        if len(res) >= F:
            res.discard(q.popleft())
        q.append(p)
        res.add(p)
    return faults


def lru(trace, F):
    od, faults = OrderedDict(), 0
    for p in trace:
        if p in od:
            od.move_to_end(p)
        else:
            faults += 1
            if len(od) >= F:
                od.popitem(last=False)
            od[p] = None
    return faults


def clock(trace, F):
    """Second-chance / Clock. A page loaded on a fault gets reference bit 1."""
    slots, ref, where = [None] * F, [0] * F, {}
    hand = used = faults = 0
    for p in trace:
        s = where.get(p)
        if s is not None:
            ref[s] = 1
            continue
        faults += 1
        if used < F:
            s = used
            used += 1
        else:
            while ref[hand]:
                ref[hand] = 0
                hand = (hand + 1) % F
            s = hand
            del where[slots[s]]
            hand = (hand + 1) % F
        slots[s] = p
        where[p] = s
        ref[s] = 1
    return faults


def lfu(trace, F):
    """In-cache LFU: counter is dropped on eviction; ties broken by least-recent use."""
    freq, tl, heap, res = {}, {}, [], set()
    faults = t = 0
    for p in trace:
        t += 1
        if p in res:
            freq[p] += 1
            tl[p] = t
            heapq.heappush(heap, (freq[p], t, p))
            continue
        faults += 1
        if len(res) >= F:
            while True:
                f, tt, q = heapq.heappop(heap)
                if q in res and freq[q] == f and tl[q] == tt:
                    break
            res.discard(q)
            del freq[q], tl[q]
        res.add(p)
        freq[p] = 1
        tl[p] = t
        heapq.heappush(heap, (1, t, p))
    return faults


def opt(trace, F, nxt=None):
    """Belady's optimal (offline): evict the resident page used farthest in the future."""
    if nxt is None:
        nxt = next_use_array(trace)
    nu, heap, res, faults = {}, [], set(), 0
    for i, p in enumerate(trace):
        if p in res:
            nu[p] = nxt[i]
            heapq.heappush(heap, (-nxt[i], p))
            continue
        faults += 1
        if len(res) >= F:
            while True:
                negn, q = heapq.heappop(heap)
                if q in res and nu[q] == -negn:
                    break
            res.discard(q)
        res.add(p)
        nu[p] = nxt[i]
        heapq.heappush(heap, (-nxt[i], p))
    return faults


# ----------------------------------------------------------------------------------------
# Adaptive policy
# ----------------------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    w_mult=4,        # monitoring window W = w_mult * frames
    seq_thr=0.45,    # fraction of +1 strides that marks a sequential window
    reuse_thr=0.10,  # a sequential window with less reuse than this is pure streaming
    rec_thr=0.55,    # recency potential (lower bound of LRU hit ratio) above which recency wins
    age="none",      # counter treatment when switching into LFU: none | halve | reset
)


def window_features(w, F):
    """Locality descriptors of one window (numpy int array)."""
    W = len(w)
    seq = float(np.mean(np.diff(w) == 1))
    order = np.argsort(w, kind="stable")
    sw = w[order]
    same = sw[1:] == sw[:-1]
    gaps = (order[1:] - order[:-1])[same]
    rec = float(np.count_nonzero(gaps <= F)) / W          # certain-LRU-hit fraction
    reuse = float(np.count_nonzero(same)) / W              # any reuse inside window
    _, cnt = np.unique(w, return_counts=True)
    cnt = np.sort(cnt)[::-1]
    cov = float(cnt[:F].sum()) / W                         # top-F frequency coverage
    return dict(seq=seq, rec=rec, reuse=reuse, cov=cov)


def choose_policy(f, prm):
    if f["seq"] >= prm["seq_thr"] and f["reuse"] < prm["reuse_thr"]:
        return "FIFO"          # streaming: no reuse to exploit, cheapest policy
    if f["rec"] >= prm["rec_thr"]:
        return "LRU"           # strong short-range temporal locality
    return "LFU"               # weak recency but repeated pages: frequency/loop behaviour


def adaptive(trace, F, params=None, record=None):
    prm = dict(DEFAULT_PARAMS)
    if params:
        prm.update(params)
    W = int(prm["w_mult"] * F)
    od = OrderedDict()               # exact LRU order (maintained for every policy)
    fq = deque()                     # FIFO order with stamps
    tin, freq, tl, heap = {}, {}, {}, []
    slots, ref, where = [None] * F, [0] * F, {}
    hand = used = faults = t = 0
    cur = "LRU"
    win, wfaults, switches = [], 0, 0
    for p in trace:
        t += 1
        win.append(p)
        if p in od:                                   # ---- hit
            od.move_to_end(p)
            ref[where[p]] = 1
            freq[p] += 1
            tl[p] = t
            heapq.heappush(heap, (freq[p], t, p))
        else:                                         # ---- fault
            faults += 1
            wfaults += 1
            if used < F:
                s = used
                used += 1
            else:
                if cur == "LRU":
                    v = next(iter(od))
                elif cur == "FIFO":
                    while True:
                        stamp, v = fq.popleft()
                        if v in od and tin[v] == stamp:
                            break
                elif cur == "LFU":
                    while True:
                        f, tt, v = heapq.heappop(heap)
                        if v in od and freq[v] == f and tl[v] == tt:
                            break
                else:  # Clock
                    while ref[hand]:
                        ref[hand] = 0
                        hand = (hand + 1) % F
                    v = slots[hand]
                    hand = (hand + 1) % F
                s = where.pop(v)
                del od[v], tin[v], freq[v], tl[v]
            slots[s] = p
            where[p] = s
            ref[s] = 1
            od[p] = None
            tin[p] = t
            fq.append((t, p))
            freq[p] = 1
            tl[p] = t
            heapq.heappush(heap, (1, t, p))
        if len(win) == W:                             # ---- monitoring window closes
            feats = window_features(np.asarray(win), F)
            new = choose_policy(feats, prm)
            if record is not None:
                record.append(dict(t=t, policy=cur, hit=1 - wfaults / W, **feats))
            if new != cur:
                switches += 1
                if new == "LFU" and prm["age"] != "none":
                    for q in freq:
                        freq[q] = 1 if prm["age"] == "reset" else max(1, freq[q] >> 1)
                    heap = [(freq[q], tl[q], q) for q in freq]
                    heapq.heapify(heap)
                cur = new
            win, wfaults = [], 0
    if record is not None:
        record.append(dict(switches=switches))
    return faults


FIXED = {"FIFO": fifo, "LRU": lru, "Clock": clock, "LFU": lfu}
