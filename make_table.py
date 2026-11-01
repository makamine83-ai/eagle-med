"""Aggregates the ns-3 calibration CSV files into the link table used by the event simulator.

Quantile vectors are averaged across seeds per link class, access category and load; loss comes from
the pooled sent and received counts. Delays cannot decrease with quantile, so each vector stays sorted.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict


def main(folder, out):
    rows = defaultdict(list)
    for f in glob.glob(os.path.join(folder, "c_*.csv")):
        for r in csv.DictReader(open(f)):
            rows[(r["cls"], int(r["ac"]), float(r["load"]))].append(r)
    table = defaultdict(dict)
    summary = []
    for (cls, ac, load), rs in sorted(rows.items()):
        sent = sum(int(r["sent"]) for r in rs)
        recv = sum(int(r["recv"]) for r in rs)
        qs = [[float(r[f"q{i}"]) for r in rs if float(r[f"q{i}"]) >= 0] for i in range(101)]
        q = [sum(v) / len(v) if v else 0.0 for v in qs]
        table[f"{cls}|{ac}"][str(load)] = {"loss": max(0.0, 1 - recv / max(1, sent)), "q": q}
        summary.append((cls, ac, load, len(rs), sent, recv, 1 - recv / max(1, sent), q[50] * 1e3, q[95] * 1e3, q[99] * 1e3))
    json.dump(table, open(out, "w"))
    with open(out.replace(".json", "_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("cls ac load seeds sent recv loss p50_ms p95_ms p99_ms".split())
        w.writerows(summary)
    for s in summary:
        print(*[round(x, 4) if isinstance(x, float) else x for x in s])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
