"""Full EAGLE-Med experimental campaign.

    python run_campaign.py --table link_table.json --out results --procs 4 [--seeds 50] [--part main,scale,sens,supp,micro]

Seeds are fixed in advance (1001, 1002, ...) and every variant runs on every seed, so results are
paired by construction. No run is dropped: failed runs are recorded with an error field.
"""
import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import platform
import time
import traceback

import numpy as np

import eaglesim.scenarios as sc
from eaglesim.arms import ARMS, register
from eaglesim.core import load_table
from eaglesim.params import ABLATIONS, BASELINES
from eaglesim.scenarios import Harness

TABLE = None


def init(path):
    global TABLE
    TABLE = load_table(path)
    register()


def job(j):
    tag, scn, var, seed, n, ov, *extra = j
    arm, bg = extra if extra else ("plain", None)
    try:
        saved = sc.BG_LOAD
        if bg:
            sc.BG_LOAD = bg
        r = ARMS[arm](scn, var, seed, TABLE, n=n, overrides=ov).run()
        sc.BG_LOAD = saved
        r["tag"] = tag
        r["error"] = ""
        return r
    except Exception:
        return {"tag": tag, "scn": scn, "variant": var, "seed": seed, "n": n, "error": traceback.format_exc()[-400:]}


def plan(args):
    seeds = [1000 + i for i in range(1, args.seeds + 1)]
    parts = args.part.split(",")
    jobs = []
    if "main" in parts:
        for scn in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S2N"):
            for v in ABLATIONS + BASELINES + ["Raft+AP"]:
                jobs += [("main", scn, v, s, 20, None) for s in seeds]
        jobs += [("main", "S3", "Raft-100", s, 20, None) for s in seeds]
    if "scale" in parts:
        for n, v in itertools.product((10, 20, 40, 80), ["EAGLE"] + BASELINES):
            jobs += [("scale", "S8", v, s, n, None) for s in seeds[:args.scale_seeds]]
    if "sens" in parts:
        grid = {"eps": (0.05, 0.20), "gamma_c": (0.85, 0.96), "neg": (2.0, 5.0), "K": (4, 5), "Tdw": (5.0, 20.0), "omega": (0.0, 1.0), "Trt": (2.0, 10.0)}
        for scn in ("S2", "S3", "S5", "S6"):
            for k, vals in grid.items():
                for x in vals:
                    jobs += [(f"sens:{k}={x}", scn, "EAGLE", s, 20, {k: x}) for s in seeds[:args.sens_seeds]]
    if "supp" in parts:
        for v in ("EAGLE", "Raft"):
            jobs += [("supp:falselabel", "S1", v, s, 20, None, "falselabel", None) for s in seeds]
        for arm in ("rate1.3", "rate0.7", "lognormal"):
            for v in ("EAGLE", "Raft", "TrustRaft"):
                jobs += [(f"supp:{arm}", "S2", v, s, 20, None, arm, None) for s in seeds]
        for bg in (50.0, 90.0):
            for v in ("EAGLE", "Raft", "Raft+AP"):
                jobs += [(f"supp:bg{bg:.0f}", "S3", v, s, 20, None, "plain", bg) for s in seeds]
        for scn in ("S2", "S3", "S7"):
            jobs += [("supp:swaptop", scn, "Bully", s, 20, None, "swaptop", None) for s in seeds]
        for scn in ("S2", "S3", "S6"):
            jobs += [("supp:kappa0", scn, "EAGLE", s, 20, None, "kappa0", None) for s in seeds]
        for scn in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
            jobs += [("supp:seeded", scn, "ResRaft-seeded", s, 20, None, "plain", None) for s in seeds]
    return jobs


def micro(path):
    """Wall-clock cost of one evaluation round on this host, per group size."""
    rows = []
    table = load_table(path)
    for n in (10, 20, 40, 80, 160):
        h = Harness("S1", "EAGLE", 77, table, n=n)
        g = h.sim.groups[0]
        for m in g.mem:
            g.inst[m].t_eval = 0.0
            g.inst[m].claim[:] = 50.0
        inst = g.inst[3]
        ts = []
        for k in range(300):
            h.sim.now = 1.0 + k
            t = time.perf_counter()
            inst.evaluate()
            ts.append(time.perf_counter() - t)
        rows.append({"n": n, "median_us": 1e6 * float(np.median(ts)), "p95_us": 1e6 * float(np.percentile(ts, 95))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--scale-seeds", type=int, default=20, dest="scale_seeds")
    ap.add_argument("--sens-seeds", type=int, default=20, dest="sens_seeds")
    ap.add_argument("--part", default="main,scale,sens,supp,micro")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = plan(args)
    print("jobs", len(jobs), flush=True)
    t0 = time.time()
    fields = None
    with open(os.path.join(args.out, "raw_runs.csv"), "w", newline="", encoding="utf-8") as f, mp.Pool(args.procs, init, (args.table,)) as pool:
        w = None
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            if r.get("error"):
                print("ERROR", r, flush=True)
            if w is None and not r.get("error"):
                fields = list(r.keys())
                fields = [k for k in fields if k not in ("tag", "error")] + ["tag", "error"]
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
            if w is not None:
                w.writerow(r)
            if k % 200 == 0:
                f.flush()
                print(k, len(jobs), round(time.time() - t0), flush=True)
    if "micro" in args.part:
        json.dump(micro(args.table), open(os.path.join(args.out, "micro.json"), "w"), indent=1)
    meta = {"jobs": len(jobs), "seconds": time.time() - t0, "python": platform.python_version(), "numpy": np.__version__,
            "host": platform.platform(), "cpus": mp.cpu_count(),
            "table_sha256": hashlib.sha256(open(args.table, "rb").read()).hexdigest() if args.table else "synthetic"}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(meta, flush=True)


if __name__ == "__main__":
    main()
