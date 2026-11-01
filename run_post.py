"""Control arms added after campaign v3 was read, on the same evaluation seeds. Writes post_runs_a.csv, _b.csv or _c.csv and leaves the
frozen campaign untouched.

    python run_post.py --table data/link_table.json --out results --procs 4 --seeds 50 --part a
"""
import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import traceback

from eaglesim import scenarios as sc
from eaglesim.arms import register
from eaglesim.arms_post import ARMS_POST, register_post
from eaglesim.core import load_table
from eaglesim.scenarios import Harness

TABLE = None
ARMS = dict(ARMS_POST, plain=Harness)


def init(path):
    global TABLE
    TABLE = load_table(path)
    register()
    register_post()


def job(j):
    tag, scn, var, seed, arm, bg, *ov = j
    try:
        saved = sc.BG_LOAD
        sc.BG_LOAD = bg
        r = ARMS[arm](scn, var, seed, TABLE, overrides=ov[0] if ov else None).run()
        sc.BG_LOAD = saved
        r["tag"] = tag
        r["error"] = ""
        return r
    except Exception:
        return {"tag": tag, "scn": scn, "variant": var, "seed": seed, "n": 20, "error": traceback.format_exc()[-400:]}


def plan(seeds, part="a"):
    """Part b was added after part a had been launched: it removes the trust stage under the same high loads."""
    if part == "b":
        return [(f"post:noT{bg:.0f}", "S3", "EAGLE-noT", s, "plain", bg) for bg in (50.0, 70.0, 90.0) for s in seeds]
    if part == "c":
        # part c separates the causes of the high-load failure, after an independent check of the first explanation
        jobs = []
        for bg in (70.0, 90.0):
            b = f"{bg:.0f}"
            jobs += [(f"post:eta0{b}", "S3", "EAGLE", s, "plain", bg, {"eta": 0.0}) for s in seeds]
            jobs += [(f"post:nosil{b}", "S3", "EAGLE-nosil", s, "plain", bg) for s in seeds]
            jobs += [(f"post:hb{b}", "S3", "EAGLE", s, "holdhb", bg) for s in seeds]
            jobs += [(f"post:eta0hb{b}", "S3", "EAGLE", s, "holdhb", bg, {"eta": 0.0}) for s in seeds]
            jobs += [(f"post:noThb{b}", "S3", "EAGLE-noT", s, "holdhb", bg) for s in seeds]
            jobs += [(f"post:eta0nosil{b}", "S3", "EAGLE-nosil", s, "plain", bg, {"eta": 0.0}) for s in seeds]
            jobs += [(f"post:eta0nosilhb{b}", "S3", "EAGLE-nosil", s, "holdhb", bg, {"eta": 0.0}) for s in seeds]
            jobs += [(f"post:r100{b}", "S3", "Raft-100", s, "plain", bg) for s in seeds]
        for bg in (50.0, 70.0, 90.0):
            jobs += [(f"post:held{bg:.0f}", "S3", "EAGLE", s, "heldtrace", bg) for s in seeds]
        return jobs
    jobs = []
    for scn in ("S2", "S3", "S6"):
        jobs += [("post:kappa0", scn, "EAGLE", s, "kappa0live", 20.0) for s in seeds]
    for bg in (20.0, 50.0, 70.0, 90.0):
        jobs += [(f"post:floor{bg:.0f}", "S3", "EAGLE-floor", s, "plain", bg) for s in seeds]
    jobs += [("post:bg70", "S3", v, s, "plain", 70.0) for v in ("EAGLE", "Raft") for s in seeds]
    jobs += [("post:calm90", "S1", v, s, "plain", 90.0) for v in ("EAGLE", "EAGLE-floor", "Raft") for s in seeds]
    jobs += [("post:calm70", "S1", v, s, "plain", 70.0) for v in ("EAGLE", "EAGLE-floor", "Raft") for s in seeds]
    jobs += [("post:floor20", scn, "EAGLE-floor", s, "plain", 20.0) for scn in ("S2", "S5", "S6") for s in seeds]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--part", default="a")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = plan([1000 + i for i in range(1, a.seeds + 1)], a.part)
    with mp.Pool(a.procs, init, (a.table,)) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2), 1):
            rows.append(r)
            if i % 100 == 0:
                print(i, len(jobs), flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, f"post_runs_{a.part}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["tag"], x["scn"], x["variant"], x["seed"])):
            w.writerow(r)
    print("jobs", len(jobs), "errors", sum(1 for r in rows if r.get("error")), "table", hashlib.sha256(open(a.table, "rb").read()).hexdigest()[:16], flush=True)


if __name__ == "__main__":
    main()
