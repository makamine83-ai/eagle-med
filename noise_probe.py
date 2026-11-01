"""Estimate score noise: spread of risk estimates for fixed candidates in a stable network."""
import sys, numpy as np
from eaglesim.core import load_table
from eaglesim.scenarios import Harness
table = load_table(sys.argv[1] if len(sys.argv) > 1 else None)
vals = []
for seed in (1, 2, 3):
    h = Harness("S1", "EAGLE", seed, table)
    g = h.sim.groups[0]
    rows = []
    def snap():
        for i in (5, 9, 14):
            inst = g.inst[i]
            if h.sim.now > 60 and inst.score_ready:
                rows.append((i, *inst.rho[[0, 1, 3, 7, 12]]))
        if h.sim.now < 178:
            h.sim.at(h.sim.now + 1.0, snap)
    h.sim.at(61.0, snap)
    h.run()
    a = np.array(rows)
    for i in (5, 9, 14):
        b = a[a[:, 0] == i][:, 1:]
        vals.append(np.abs(b - np.median(b, axis=0)).ravel())
v = np.concatenate(vals)
print("abs dev rho: mean %.4f p95 %.4f p99 %.4f" % (v.mean(), np.percentile(v, 95), np.percentile(v, 99)))
