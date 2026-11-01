"""Turns the campaign statistics into results_digest.json, the only source of numbers for the manuscript.

    python make_digest.py results ../manuscript/results_digest.json closed_loop_folder
"""
import csv
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

LOADS = (20, 50, 90)
SUPP_METRICS = ("hit_ratio", "hit_ratio_em", "reelect_s", "recovery_s", "leader_changes", "avail", "avail_window", "kbytes_per_node_s", "alert_vac_mean_s", "alert_mean_s", "alert_p99_s", "mal_lead_s", "mal_wins", "mal_incumbent", "elec_msgs", "held_peak")


def load_summary(path):
    out = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[f'{r["scn"]}|{r["n"]}'][r["variant"]][r["metric"]] = {k: float(r[k]) for k in ("count", "mean", "sd", "median", "q25", "q75", "ci95_low", "ci95_high")}
    return out


def load_tests(path):
    out = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[f'{r["scn"]}|{r["n"]}'][r["variant"]][r["metric"]] = {"pairs": int(r["pairs"]), "diff": float(r["median_diff"]), "p": float(r["p"]),
                                                                 "p_holm": float(r["p_holm"]), "r": float(r["rank_biserial"])}
    return out


def run_audit(rows):
    errors = [r for r in rows if r.get("error")]
    skipped = [r for r in rows if r.get("skipped") == "True"]
    per = defaultdict(int)
    for r in skipped:
        per[f'{r["scn"]}/{r["variant"]}/{r["tag"]}'] += 1
    tags = defaultdict(int)
    for r in rows:
        tags[r["tag"].split(":")[0]] += 1
    return {"runs": len(rows), "errors": len(errors), "skipped": len(skipped), "skipped_by": dict(per), "by_tag": dict(tags)}


def closed_loop(folder):
    """ns-3 closed-loop summary per load and mode, with seed-level values for the comparison with the simulator."""
    out = {}
    for load in LOADS:
        for mode in (0, 1):
            rec, lat, near_seed, near, lost, total, members = [], [], [], [], 0, 0, set()
            for f in sorted(glob.glob(os.path.join(folder, f"l{load}_m{mode}_s*.csv"))):
                rows = list(csv.DictReader(open(f)))
                rec.append(float(rows[0]["failover_s"]))
                members.add(int(rows[0]["members"]))
                crash = float(rows[0]["crash_s"])
                ns = []
                for r in rows:
                    birth, x = float(r["birth"]), float(r["latency"])
                    total += 1
                    if x < 0 or x != x:
                        lost += 1
                        continue
                    lat.append(x)
                    if crash <= birth <= crash + 1.0:
                        ns.append(x)
                near += ns
                if ns:
                    near_seed.append(st.mean(ns))
            if not rec:
                continue
            lat.sort()
            out[f"{load}|{mode}"] = {"seeds": len(rec), "members": sorted(members), "failover_mean": st.mean(rec), "failover_sd": st.stdev(rec), "failover_median": st.median(rec), "failover": rec,
                                     "failed": sum(x < 0 for x in rec), "alert_mean": st.mean(lat), "alert_p50": lat[len(lat) // 2], "alert_p95": lat[int(0.95 * len(lat))], "alert_p99": lat[int(0.99 * len(lat))],
                                     "lost": lost, "alerts": total, "near_mean": st.mean(near) if near else None, "near_n": len(near), "near_seed": near_seed}
    return out


def column(rows, scn, var, metric, n="20", tag="main"):
    out = {}
    for r in rows:
        if r["scn"] == scn and r["variant"] == var and r["n"] == n and r["tag"] == tag and r.get(metric) not in ("", "nan", None):
            out[int(r["seed"])] = float(r[metric])
    return out


def paired(a, b):
    """Unadjusted Wilcoxon on the common seeds of two columns, with the means of both."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    nz = [x for x in d if x]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
    return {"seeds": len(seeds), "changed": len(nz), "a": st.mean(a[s] for s in seeds), "b": st.mean(b[s] for s in seeds), "p": p}


def compare(des, ns3):
    t = stats.ttest_ind(des, ns3, equal_var=False)
    ci = t.confidence_interval(0.95)
    return {"des": st.mean(des), "ns3": st.mean(ns3), "des_sd": st.stdev(des), "ns3_sd": st.stdev(ns3), "low": float(ci.low), "high": float(ci.high), "rel": (st.mean(des) - st.mean(ns3)) / st.mean(ns3), "p": float(t.pvalue)}


def derived(rows, cl):
    """Quantities the tables do not hold: skewed distributions, control arms and the simulator against ns-3."""
    out = {"attack": {}, "hit_spread": {}, "xval": {}, "supp": defaultdict(lambda: defaultdict(dict)), "cmp": {}}
    for v in ("EAGLE", "Raft", "Raft+AP", "ResRaft", "TrustRaft", "Bully", "Ring", "EAGLE-noT", "EAGLE-noS", "EAGLE-noA", "EAGLE-noU", "EAGLE-noW"):
        x = column(rows, "S6", v, "mal_lead_s")
        w = column(rows, "S6", v, "mal_wins")
        inc = column(rows, "S6", v, "mal_incumbent")
        led = sum(a > 0 for a in x.values())
        ci = stats.binomtest(led, len(x)).proportion_ci(0.95, "exact")
        won = sum(a > 0 for a in w.values())
        wci = stats.binomtest(won, len(w)).proportion_ci(0.95, "exact")
        out["attack"][v] = {"led": led, "seeds": len(x), "low": ci.low, "high": ci.high, "won": won, "won_low": wci.low, "won_high": wci.high, "incumbent": sum(a > 0 for a in inc.values()),
                            "median": st.median(x.values()), "max": max(x.values()), "mean": st.mean(x.values()),
                            "kept": sorted(round(x[s], 2) for s in x if x[s] > 0)[:8]}
    for scn, m in (("S2", "hit_ratio_em"), ("S3", "hit_ratio_em"), ("S7", "hit_ratio_em")):
        out["hit_spread"][scn] = {}
        for v in ("EAGLE", "Raft", "ResRaft", "TrustRaft", "EAGLE-noT", "EAGLE-noU", "EAGLE-noW", "EAGLE-noS"):
            x = sorted(column(rows, scn, v, m).values())
            out["hit_spread"][scn][v] = {"q10": float(np.quantile(x, 0.1)), "median": st.median(x), "q90": float(np.quantile(x, 0.9)), "min": x[0], "below_50": sum(a < 0.5 for a in x)}
    # noU seeds that never changed leader: what ranking before the surge gives without any re-election
    still = [s for s, x in column(rows, "S2", "EAGLE-noU", "leader_changes").items() if x == 0]
    a, b = column(rows, "S2", "EAGLE-noU", "hit_ratio_em"), column(rows, "S2", "Raft", "hit_ratio_em")
    out["noU_static"] = {"seeds": len(still), "noU": st.mean(a[s] for s in still), "Raft": st.mean(b[s] for s in still)}
    tags = sorted({r["tag"] for r in rows if r["tag"].startswith("supp:")})
    for tag in tags:
        for key in sorted({(r["scn"], r["variant"]) for r in rows if r["tag"] == tag}):
            scn, v = key
            for m in SUPP_METRICS:
                x = column(rows, scn, v, m, tag=tag)
                if x:
                    out["supp"][tag][f"{scn}|{v}"][m] = {"mean": st.mean(x.values()), "median": st.median(x.values()), "n": len(x)}
    # arm against the same protocol in the main scenario, or against another protocol inside the arm
    c = out["cmp"]
    for m in ("hit_ratio", "leader_changes", "avail", "kbytes_per_node_s"):
        for v in ("EAGLE", "Raft"):
            c[f"falselabel|{v}|{m}"] = paired(column(rows, "S1", v, m, tag="supp:falselabel"), column(rows, "S1", v, m))
    for arm in ("rate1.3", "rate0.7", "lognormal"):
        for v in ("Raft", "TrustRaft"):
            c[f"{arm}|EAGLE_vs_{v}"] = paired(column(rows, "S2", "EAGLE", "hit_ratio_em", tag=f"supp:{arm}"), column(rows, "S2", v, "hit_ratio_em", tag=f"supp:{arm}"))
        c[f"{arm}|EAGLE_vs_main"] = paired(column(rows, "S2", "EAGLE", "hit_ratio_em", tag=f"supp:{arm}"), column(rows, "S2", "EAGLE", "hit_ratio_em"))
    for scn, m in (("S2", "hit_ratio_em"), ("S3", "hit_ratio_em"), ("S3", "reelect_s"), ("S6", "reelect_s"), ("S6", "hit_ratio"), ("S6", "mal_lead_s")):
        c[f"kappa0|{scn}|{m}"] = paired(column(rows, scn, "EAGLE", m, tag="supp:kappa0"), column(rows, scn, "EAGLE", m))
    for scn, m in (("S2", "hit_ratio_em"), ("S3", "hit_ratio_em"), ("S7", "hit_ratio_em")):
        c[f"swaptop|{scn}"] = paired(column(rows, scn, "Bully", m, tag="supp:swaptop"), column(rows, scn, "EAGLE", m))
    for scn, m in (("S1", "hit_ratio"), ("S2", "hit_ratio_em"), ("S3", "hit_ratio_em"), ("S4", "hit_ratio"), ("S5", "hit_ratio"), ("S6", "hit_ratio"), ("S7", "hit_ratio_em")):
        c[f"seeded|{scn}"] = paired(column(rows, scn, "EAGLE", m), column(rows, scn, "ResRaft-seeded", m, tag="supp:seeded"))
    for bg in ("bg50", "bg90"):
        for m in ("alert_vac_mean_s", "hit_ratio_em", "reelect_s"):
            for v in ("Raft", "Raft+AP"):
                c[f"{bg}|{m}|EAGLE_vs_{v}"] = paired(column(rows, "S3", "EAGLE", m, tag=f"supp:{bg}"), column(rows, "S3", v, m, tag=f"supp:{bg}"))
    if cl:
        for mode, v in ((0, "Raft"), (1, "EAGLE")):
            for load in LOADS:
                k = f"{load}|{mode}"
                if k not in cl:
                    continue
                tag = "main" if load == 20 else f"supp:bg{load}"
                des = list(column(rows, "S3", v, "reelect_s", tag=tag).values())
                vac = list(column(rows, "S3", v, "alert_vac_mean_s", tag=tag).values())
                failed = [x for x in cl[k]["failover"] if x >= 0]
                out["xval"][f"{v}|{load}"] = {"failover": compare(des, failed), "failed": len(cl[k]["failover"]) - len(failed)}
                if len(cl[k]["near_seed"]) > 1:
                    out["xval"][f"{v}|{load}"]["vacancy"] = compare(vac, cl[k]["near_seed"])
    return out


def post_arms(rows, main):
    """Arms added after campaign v3 was read: means per arm, and comparisons with the campaign on the same seeds."""
    all_rows = main + rows
    out = {"arm": defaultdict(dict), "cmp": {}, "runs": len(rows), "errors": sum(1 for r in rows if r.get("error")), "skipped": defaultdict(int)}
    for r in rows:
        if r.get("skipped") == "True":
            out["skipped"][f'{r["tag"]}|{r["scn"]}|{r["variant"]}'] += 1
    for tag, scn, v in sorted({(r["tag"], r["scn"], r["variant"]) for r in rows}):
        for m in SUPP_METRICS:
            x = column(rows, scn, v, m, tag=tag)
            if x:
                out["arm"][f"{tag}|{scn}|{v}"][m] = {"mean": st.mean(x.values()), "median": st.median(x.values()), "n": len(x)}
    c = out["cmp"]
    # the same protocol and scenario with kappa at zero, and with the suspension floor at 20 Mb/s
    for scn, m in (("S2", "hit_ratio_em"), ("S2", "elec_msgs"), ("S3", "hit_ratio_em"), ("S3", "reelect_s"), ("S3", "elec_msgs"), ("S6", "reelect_s"), ("S6", "hit_ratio"), ("S6", "mal_lead_s"), ("S6", "mal_wins")):
        c[f"kappa0|{scn}|{m}"] = paired(column(rows, scn, "EAGLE", m, tag="post:kappa0"), column(main, scn, "EAGLE", m))
    for scn, m in (("S2", "hit_ratio_em"), ("S5", "hit_ratio"), ("S5", "leader_changes"), ("S5", "avail"), ("S6", "hit_ratio"), ("S6", "mal_wins")):
        c[f"floor20|{scn}|{m}"] = paired(column(rows, scn, "EAGLE-floor", m, tag="post:floor20"), column(main, scn, "EAGLE", m))
    # each high-load arm against EAGLE-Med and Raft at the same load
    tags = {20: "main", 50: "supp:bg50", 70: "post:bg70", 90: "supp:bg90"}
    for bg, tag in tags.items():
        for m in ("avail", "hit_ratio_em", "leader_changes"):
            for arm, v in ((f"post:floor{bg}", "EAGLE-floor"), (f"post:noT{bg}", "EAGLE-noT")):
                a = column(all_rows, "S3", v, m, tag=arm)
                if a:
                    c[f"S3|{bg}|{v}|{m}|EAGLE"] = paired(a, column(all_rows, "S3", "EAGLE", m, tag=tag))
                    c[f"S3|{bg}|{v}|{m}|Raft"] = paired(a, column(all_rows, "S3", "Raft", m, tag=tag))
            c[f"S3|{bg}|EAGLE|{m}|Raft"] = paired(column(all_rows, "S3", "EAGLE", m, tag=tag), column(all_rows, "S3", "Raft", m, tag=tag))
    # arms that separate the causes of the high-load failure
    for bg in (70, 90):
        for arm, v in (("eta0", "EAGLE"), ("nosil", "EAGLE-nosil"), ("hb", "EAGLE"), ("eta0hb", "EAGLE"), ("noThb", "EAGLE-noT"), ("eta0nosil", "EAGLE-nosil"), ("eta0nosilhb", "EAGLE-nosil"), ("r100", "Raft-100")):
            for m in ("avail", "hit_ratio_em"):
                a = column(rows, "S3", v, m, tag=f"post:{arm}{bg}")
                if a:
                    c[f"{arm}|{bg}|{m}|Raft"] = paired(a, column(all_rows, "S3", "Raft", m, tag=tags[bg]))
                    c[f"{arm}|{bg}|{m}|EAGLE"] = paired(a, column(all_rows, "S3", "EAGLE", m, tag=tags[bg]))
    for bg in (70, 90):
        for m in ("avail", "hit_ratio"):
            for v in ("EAGLE", "EAGLE-floor"):
                c[f"S1|{bg}|{v}|{m}|Raft"] = paired(column(rows, "S1", v, m, tag=f"post:calm{bg}"), column(rows, "S1", "Raft", m, tag=f"post:calm{bg}"))
    return out


def main(results, out, cl=None):
    rows = list(csv.DictReader(open(os.path.join(results, "raw_runs.csv"), encoding="utf-8")))
    closed = closed_loop(cl) if cl else {}
    post = []
    for f in sorted(glob.glob(os.path.join(results, "post_runs_*.csv"))):
        post += list(csv.DictReader(open(f, encoding="utf-8")))
    d = {"summary": load_summary(os.path.join(results, "summary.csv")), "tests": load_tests(os.path.join(results, "tests.csv")),
         "derived": derived(rows, closed), "closed_loop": closed, "post": post_arms(post, rows) if post else None,
         "sensitivity": list(csv.DictReader(open(os.path.join(results, "sensitivity.csv"), encoding="utf-8"))),
         "audit": run_audit(rows), "meta": json.load(open(os.path.join(results, "meta.json"))),
         "micro": json.load(open(os.path.join(results, "micro.json")))}
    json.dump(d, open(out, "w", encoding="utf-8"), indent=1)
    print(d["audit"], d["meta"]["seconds"])


if __name__ == "__main__":
    main(*sys.argv[1:])
