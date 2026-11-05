# EAGLE-Med simulation package

Reproduces every number, table and chart in the manuscript "EAGLE-Med: Emergency-Aware and Graded
Leader Election for Trustworthy Medical Edge Computing Networks" bechir aziza, University of Batna 2

All data here are synthetic simulation outputs. No patient or clinical data are used, and nothing here is a clinical
validation.

## Layout

| Path | Content |
|---|---|
| `ns3/hospital_channel.cc` | ns-3.41 packet-level calibration: four 802.11ac wards on a 1 Gb/s backbone |
| `ns3/closedloop.cc` | ns-3.41 closed-loop cross-validation: election and alerts as real UDP traffic |
| `make_table.py` | Aggregates calibration CSV files into `data/link_table.json` |
| `eaglesim/core.py` | Event kernel, nodes, trace-driven link model, metrics |
| `eaglesim/params.py` | Protocol profile, parameters, variants |
| `eaglesim/protocols.py` | EAGLE-Med, Raft, ResRaft, TrustRaft, Bully, Ring |
| `eaglesim/scenarios.py` | Deployment, workload, alert paths, fault injection, Scenarios S1 to S8 and S2N |
| `eaglesim/arms.py` | Control arms, each changing one assumption of a main scenario |
| `eaglesim/arms_post.py` | Arms added after campaign v3 was read, as subclasses, so the frozen sources stay unchanged |
| `run_post.py` | Runs those arms on the same evaluation seeds |
| `run_campaign.py` | Full campaign; seeds are fixed in advance |
| `analyze.py` | Mean, SD, median, 95 % CI, paired Wilcoxon tests with Holm correction, rank-biserial effect size |
| `noise_probe.py` | Measures the risk-score noise that sets the stability margin |
| `test_invariants.py` | Checks one winner per term, no dual leadership, alert independence and dwell on running code |
| `CAMPAIGN_HASHES.txt` | SHA-256 prefixes of the sources and link table used by the reported campaign |
| `POST_HASHES.txt` | The same for the arms added after it |
| `data/` | ns-3 link table and its summary |
| `results/` | Raw per-run CSV, statistics and run metadata |
| `results/results_digest.json` | Every number the manuscript quotes, built by `make_digest.py` |
| `calibration_ns3/cal/` | Per-run output of the ns-3 calibration grid |
| `closed_loop_ns3/cl/` | Per-run output of the closed-loop ns-3 grid |
| `results_v2/` | Outputs of the superseded campaign v2, kept for the audit trail |

## Requirements

- Python 3.12 or later with numpy and scipy. No other package is needed for the simulator.
- ns-3.41 for the two C++ programs. On Ubuntu 24.04:

```bash
apt-get install ns3 libns3-dev libgsl-dev libsqlite3-dev libxml2-dev
```

## Reproduction

1. Compile and run the calibration grid, loads 0, 5, 20, 50, 90 and 140 Mb/s per ward, seeds 1 to 5:

```bash
g++ -std=c++20 -O2 -w ns3/hospital_channel.cc -o hc $(pkg-config --cflags --libs ns3-wifi ns3-internet ns3-csma ns3-applications ns3-mobility ns3-propagation ns3-network ns3-core ns3-bridge ns3-spectrum ns3-traffic-control ns3-stats)
```

```bash
./hc --load=20 --seed=1 --time=45 --out=calibration_ns3/cal/c_20_1.csv
```

2. Build the link table:

```bash
python make_table.py calibration_ns3/cal data/link_table.json
```

3. Run the campaign. It used four CPU cores on a Kaggle session:

```bash
python run_campaign.py --table data/link_table.json --out results --procs 4 --seeds 50
```

4. Compute the statistics:

```bash
python analyze.py results/raw_runs.csv results
```

5. Run the closed-loop ns-3 grid, modes 0 and 1, loads 20, 50 and 90 Mb/s, seeds 1 to 20, with the crash time
   `20 + ((seed * 37) % 100) / 100` s:

```bash
g++ -std=c++20 -O2 -w ns3/closedloop.cc -o cl_bin $(pkg-config --cflags --libs ns3-wifi ns3-internet ns3-csma ns3-applications ns3-mobility ns3-propagation ns3-network ns3-core ns3-bridge ns3-spectrum ns3-traffic-control ns3-stats)
```

```bash
./cl_bin --mode=1 --seed=1 --load=20 --time=46 --crash=20.37 --out=closed_loop_ns3/cl/l20_m1_s1.csv
```

6. Run the arms added after the campaign, parts a, b and c:

```bash
python run_post.py --table data/link_table.json --out results --procs 4 --seeds 50 --part a
```

```bash
python run_post.py --table data/link_table.json --out results --procs 4 --seeds 50 --part b
```

```bash
python run_post.py --table data/link_table.json --out results --procs 4 --seeds 50 --part c
```

7. Build the number source of the manuscript, which also reads the closed-loop files and the post arms, then
   compare it with `results/results_digest.json`:

```bash
python make_digest.py results results_digest.json closed_loop_ns3/cl
```

Steps 2, 4 and 7 run in minutes on a laptop from the files in this repository. Steps 1, 3, 5 and 6 repeat the
simulations. Step 3 took 3.7 hours on four CPU cores.

## Control arms

`eaglesim/arms.py` defines arms that change one assumption of a main scenario. They run in the same campaign, carry
tags `supp:*` in `raw_runs.csv`, stay out of the main statistics, and their p values are unadjusted. They were
designed after the audit described below, so they are checks and not preregistered tests.

`eaglesim/arms_post.py` and `run_post.py` hold the arms added after campaign v3 was read. Their rows carry tags
`post:*` and live in `results/post_runs_a.csv`, `_b.csv` and `_c.csv`, so `raw_runs.csv` stays as the campaign
wrote it. They exist for two reasons found on reading the campaign:

- The `supp:kappa0` arm had no effect. It replaced the profile on the harness while each node kept its own reference,
  so its rows equal the main runs. `post:kappa0` replaces it on the nodes. The `supp:kappa0` rows are kept and are not
  reported as a test of anything.
- During an emergency at 90 Mb/s per ward EAGLE-Med loses its leader (`supp:bg90`). The first explanation, selective
  silence, was refuted by an independent check and by `post:nosil*`. The probe wait `rtt - minrtt` includes channel
  queueing, so measured capacity falls under load, honest claims look inflated and suspensions reach all 20 nodes
  (`post:held*`, `post:eta0*`), and the 0.18 s emergency lease expires (`post:hb*`, `post:r100*`). `post:noT*` removes
  the trust stage, `post:noThb*` also holds the heartbeat at 0.2 s, and `post:eta0hb*`, `post:eta0nosil*` and
  `post:eta0nosilhb*` combine the rest. `post:floor*` stops new suspensions below `n - Q + 1` unsuspended nodes, and
  `post:bg70` and `post:calm*` place the limit and test it without an emergency. These arms are diagnostics, not part
  of the reported protocol. Part c was added after parts a and b, and the versions of the post sources used by each part
  are recorded in `POST_HASHES.txt`.

Part a was launched from a copy of `run_post.py` that wrote `post_runs.csv`, renamed here to `post_runs_a.csv`. Its job
list is unchanged.

## Notes on interpretation

- Development used seeds 1 to 12. Evaluation seeds are 1001 to 1050, and no evaluation run is dropped.
- All variants see the same scenario and workload draws for a given seed, so comparisons are paired.
  Network, protocol and service streams are not shared, since they depend on protocol behaviour.
- Simulated energy is an airtime model with ns-3 default radio currents, not a measurement.
- ns-3.41 detail that cost time: `SetIpTos` must be called after `Connect`, because `Connect` and `Bind`
  overwrite the socket TOS with the value carried in the address, and `SendTo` reads it from the address.
  Without this, every frame silently uses AC_BE.
- Nothing here is a clinical validation. Urgency levels and anchors are network operating points.

## Change log before the reported campaign

Three campaigns were run. Only the last, `eagle-campaign-v3`, is reported, and its sources match
`CAMPAIGN_HASHES.txt`. Nothing in the frozen sources changed after it was read.

- `eagle-campaign` was started and then abandoned before any of its output was read.
- `eagle-campaign-v2` was completed and read. Its outputs are kept in `results_v2/` with their hashes. An
  independent audit of the manuscript against the code and raw runs then found the defects listed under
  "Corrections after campaign v2". The evaluation seeds are the same in v2 and v3, so they are not unseen data.
  No parameter or anchor changed between v2 and v3.

### Corrections after campaign v2

- Proposition 2 had a gap. A preemptive winner waited for `L`, while the old leader may run until `L + T_hb`,
  and a timeout winner with a voter that still heard the leader did not wait at all. A winner now waits for
  step-down or `K T_hb + eps_L` whenever one of its voters still heard the leader.
- Voters applied the recency rule at the pre-vote but not at the vote, so a heartbeat arriving between the two, or
  a node skipping the pre-vote, could turn a follower against a live leader. The rule now holds at the vote.
- ResRaft compared candidates with the best score of all nodes, silent ones included, so a cut or partitioned top
  node blocked votes. The reference is now the best node heard within 1.5 `T_g`.
- The S6 attacker's timer equalled `K T_hb`, so voters still counted the last heartbeat as recent and refused it
  for a reason unrelated to its behaviour. It now fires 5 ms later. Runs also record whether the attacker was
  already leader when compromised and how many elections it won afterwards.
- S5 leader cuts that found no serving leader were skipped. They now retry, like crashes.
- `analyze.py` gave a p value of 1 to sensitivity comparisons with too few changed pairs. They are now marked as
  untested, and the summary adds quartiles.
- `ns3/closedloop.cc` used the election timer as heartbeat recency, kept 18 members instead of 20, crashed at a
  fixed time and ended before alert lifetimes expired. All four are fixed, and it now runs at 20, 50 and 90 Mb/s.

### Corrections before campaign v2

The defects below had been found by a strengthened invariant test and an independent review of the code against
the manuscript. All checks used development seeds only.

Protocol:

- A leader never recorded its own election time, so its hand-off request ignored dwell and probation.
- A preemptive attempt refused by a quorum was repeated every second. It now waits `T_rt`, chosen on
  development seeds and included in the sensitivity grid.
- A claim that no probe had measured counted as misbehaviour. It now counts at half weight and adds no
  anomaly evidence.
- Observers that had not reported a target counted as zero in the pooled median. The median now runs over
  the observers that reported it, which matters at large `n`.
- Voters relaxed eligibility after failed campaigns and stayed relaxed. They now drop only the score
  filter, and only during the vacancy.
- A vote could be cleared within its term. It is now kept for the whole term.
- Delayed digests could revive a cleared emergency. Urgency labels now carry a version.
- Gossip did not restart after recovery.
- The heartbeat reception time was overwritten by the election timer, which produced false evidence of
  selective silence and made voters refuse pre-votes after their own campaigns.
- The stability margin is now the noise guard plus the transition cost, so that Proposition 3 holds.
- The code used `eps = 0.15` while the manuscript said 0.10. The measured p99 is 0.077, so it is 0.10.
- Suspensions can no longer be relaxed, and late answers count as neither good nor bad evidence.

Baselines:

- The Bully and Ring detector re-armed from the current time instead of the last heartbeat, which delayed
  their detection.
- Bully and Ring skipped the state synchronisation time that quorum protocols pay.
- ResRaft never refreshed its timers from its scores, and dead nodes set its vote threshold.

Evaluation:

- The link model picked the calibration row above the load, so almost every message used the 50 Mb/s row.
  It now interpolates between rows.
- Messages, energy and leader changes included the warm-up, and control traffic included the workload.
- Recovery stopped on answers to requests issued before the crash.
- Mean tenure left out the tenure still running at the end of a run.
- Election success used different start points per protocol. It now uses leaderless intervals for all.
- Alert relays were chosen with knowledge of which nodes were alive.
- A fault scheduled during a vacancy was skipped. It now hits the next leader.
- The sensitivity arm `K = 2` was invalid, because the lease was then shorter than a heartbeat period.
- `analyze.py` let sensitivity runs overwrite main runs. It now separates them by tag.
- The ablation `noS` now also removes probation and the retry interval, and `noT` no longer pays for
  evidence bytes it does not use.

## Licence

Code is under the MIT licence (`LICENSE`). Data and analysis outputs are under CC BY 4.0 (`LICENSE-DATA.md`). ns-3 is a
separate GPL-2.0 project and is not included.
