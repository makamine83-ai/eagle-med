"""Statistics for the EAGLE-Med campaign.

Reads the raw per-run CSV and writes:
  summary.csv   mean, SD, median, quartiles, 95 % t interval per scenario, variant and metric
  tests.csv     paired Wilcoxon signed-rank tests of EAGLE against each other variant on the same seeds,
                Holm-corrected within scenario and metric, with the matched rank-biserial effect size
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

METRICS = ["first_election_s", "reelect_s", "recovery_s", "alert_mean_s", "alert_p95_s", "alert_p99_s", "alert_max_s", "alert_vac_mean_s",
           "alert_delivery", "avail", "avail_window", "dual_leader_s", "leader_changes", "tenure_s", "elec_msgs",
           "msgs_per_node_s", "kbytes_per_node_s", "energy_mW_per_gw", "elec_success", "hit_ratio", "hit_ratio_em", "req_lat_p50", "mal_lead_s", "mal_incumbent", "mal_wins", "wall_s"]
SENS_METRICS = ["reelect_s", "alert_mean_s", "avail", "leader_changes", "hit_ratio_em", "mal_lead_s"]


def load(path, tags=("main", "scale")):
    # sensitivity runs reuse the scenario, variant and seed of the main runs, so they are kept out by tag
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    data = defaultdict(dict)  # (scn, n, variant) -> seed -> row
    for r in rows:
        if r["tag"] in tags:
            data[(r["scn"], int(r["n"]), r["variant"])][int(r["seed"])] = r
    return data


def holm(fam, col):
    """Appends the Holm-adjusted p value of column col to each row of one family."""
    order = sorted(range(len(fam)), key=lambda i: fam[i][col])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, fam[i][col] * (len(fam) - rank)))
        fam[i].append(running)


def val(r, m):
    try:
        v = float(r.get(m, "nan"))
    except ValueError:
        v = float("nan")
    return v


def main(path, outdir, ref="EAGLE"):
    data = load(path)
    with open(f"{outdir}/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow("scn n variant metric count mean sd median q25 q75 ci95_low ci95_high".split())
        for key in sorted(data):
            for m in METRICS:
                x = np.array([val(r, m) for r in data[key].values()])
                x = x[~np.isnan(x)]
                if len(x) == 0:
                    continue
                mean, sd = x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0
                h = stats.t.ppf(0.975, len(x) - 1) * sd / math.sqrt(len(x)) if len(x) > 1 else 0.0
                w.writerow([*key, m, len(x), f"{mean:.6g}", f"{sd:.6g}", f"{np.median(x):.6g}", f"{np.quantile(x, 0.25):.6g}", f"{np.quantile(x, 0.75):.6g}", f"{mean - h:.6g}", f"{mean + h:.6g}"])
    tests = []
    keys = {(k[0], k[1]) for k in data}
    for scn, n in sorted(keys):
        vs = [k[2] for k in data if k[0] == scn and k[1] == n and k[2] != ref]
        if (scn, n, ref) not in data:
            continue
        for m in METRICS:
            fam = []
            for v in vs:
                a, b = data[(scn, n, ref)], data[(scn, n, v)]
                seeds = sorted(set(a) & set(b))
                x = np.array([val(a[s], m) for s in seeds])
                y = np.array([val(b[s], m) for s in seeds])
                ok = ~(np.isnan(x) | np.isnan(y))
                x, y = x[ok], y[ok]
                d = x - y
                nz = d[d != 0]
                if len(nz) < 6:
                    continue
                p = stats.wilcoxon(nz).pvalue
                r = stats.rankdata(np.abs(nz))
                rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
                fam.append([scn, n, v, m, len(nz), float(np.median(d)), p, rb])
            holm(fam, 6)
            tests += fam
    with open(f"{outdir}/tests.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow("scn n variant metric pairs median_diff p p_holm rank_biserial".split())
        for t in tests:
            w.writerow([t[0], t[1], t[2], t[3], t[4], f"{t[5]:.6g}", f"{t[6]:.4g}", f"{t[8]:.4g}", f"{t[7]:.3f}"])
    sensitivity(path, outdir, data, ref)
    print("wrote", outdir)


def sensitivity(path, outdir, main_data, ref):
    """Each one-at-a-time setting against the frozen setting on the same seeds, Holm-corrected per scenario and metric."""
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["tag"].startswith("sens:")]
    grid = defaultdict(dict)  # (scn, tag) -> seed -> row
    for r in rows:
        grid[(r["scn"], r["tag"])][int(r["seed"])] = r
    out, untested = [], []
    for scn in sorted({k[0] for k in grid}):
        base = main_data[(scn, 20, ref)]
        for m in SENS_METRICS:
            fam = []
            for tag in sorted(t for s, t in grid if s == scn):
                runs = grid[(scn, tag)]
                seeds = sorted(set(runs) & set(base))
                x = np.array([val(runs[s], m) for s in seeds])
                y = np.array([val(base[s], m) for s in seeds])
                ok = ~(np.isnan(x) | np.isnan(y))
                x, y = x[ok], y[ok]
                if len(x) == 0:
                    continue
                nz = (x - y)[x != y]
                if len(nz) < 6:
                    untested.append([scn, tag[5:], m, len(x), float(x.mean()), float(y.mean())])  # too few changed pairs for a test
                    continue
                fam.append([scn, tag[5:], m, len(x), float(x.mean()), float(y.mean()), stats.wilcoxon(nz).pvalue])
            holm(fam, 6)
            out += fam
    with open(f"{outdir}/sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow("scn setting metric pairs mean mean_frozen p p_holm".split())
        for t in out:
            w.writerow([t[0], t[1], t[2], t[3], f"{t[4]:.6g}", f"{t[5]:.6g}", f"{t[6]:.4g}", f"{t[7]:.4g}"])
        for t in untested:
            w.writerow([t[0], t[1], t[2], t[3], f"{t[4]:.6g}", f"{t[5]:.6g}", "nan", "nan"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
