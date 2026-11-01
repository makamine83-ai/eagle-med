"""Scenario harness: deployment, coordination workload, alert paths, fault injection, run summary."""
import statistics
import time

from .core import AC_VO, SIZE, Node, Sim
from .params import EagleParams, VARIANTS
from .protocols import CLASSES, Group

# deployment constants (see Table 3 of the paper)
SERVER_CAP = (320.0, 260.0)
GW_CAP = (30.0, 110.0)
BG_LOAD = 20.0            # Mb/s per ward
TX_I, RX_I, VOLT = 0.380, 0.313, 3.0   # ns-3 WifiRadioEnergyModel default currents and 3 V supply
PHY_RATE, FRAME_OH = 120e6, 180e-6     # effective rate and per-frame overhead for small frames

SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S2N"]
T_END = {"S1": 180.0, "S2": 180.0, "S3": 180.0, "S4": 180.0, "S5": 240.0, "S6": 240.0, "S7": 180.0, "S8": 120.0, "S2N": 180.0}
T0 = 60.0  # warm-up during which evidence accumulates; no metric window starts earlier
VACANCY_LIMIT = 3.0  # an election succeeds if a leader serves within this time of the vacancy, s


def build(sim, n, wards, gw_cap=GW_CAP):
    """Two wired servers and n-2 wireless gateways shared among the wards, plus two alert sinks."""
    rng = sim.rng_scn
    sim.wards = wards
    nodes = []
    for i in range(2):
        nd = Node(i, "S", i % wards, SERVER_CAP[i], rng.uniform(0.25, 0.40), 1.0, True)
        nd.home = i
        nodes.append(nd)
    for i in range(2, n):
        w = (i - 2) % wards
        mains = rng.random() < 0.6
        nodes.append(Node(i, "G", w, rng.uniform(*gw_cap), rng.uniform(0.10, 0.35), 1.0 if mains else rng.uniform(0.5, 1.0), mains))
    sinks = []
    for i in range(2):
        # clinical stations: outside the election, one on each side of any ward partition
        nodes.append(Node(n + i, "S", 0 if i == 0 else wards - 1, 1000.0, 0.0, 1.0, True))
        sinks.append(n + i)
    sim.nodes = nodes
    return sinks


class Harness:
    FAULT_TRIES = 100

    def __init__(self, scn, variant, seed, table, n=20, wards=4, overrides=None, fast_hb=None):
        self.scn, self.variant, self.seed = scn, variant, seed
        self.n, self.wards = n, wards
        self.sim = Sim(seed, table, BG_LOAD)
        self.sim.deliver_hook = self.on_plain
        self.sim.m.t0 = T0
        spec = dict(VARIANTS[variant])
        family = spec.pop("family")
        self.ap = spec.pop("alert_plane", family == "eagle")
        fast_hb = spec.pop("fast_hb", fast_hb)
        p = EagleParams(**spec)
        for k, v in (overrides or {}).items():
            setattr(p, k, v)
        if family == "eagle":
            self.ap = p.use_alert_plane
        self.p = p
        # negative control: every gateway can carry the emergency load on its own
        self.sinks = build(self.sim, n, wards, (150.0, 250.0) if scn == "S2N" else GW_CAP)
        self.t0, self.t_end = T0, T_END[scn]
        if scn == "S8":
            self.t_end = 120.0
        self.cls = CLASSES[family]
        if fast_hb:
            self.cls = type(self.cls.__name__ + "Fast", (self.cls,), {"fixed_hb": fast_hb})
        prio = self.priorities()
        if scn == "S7":
            self.make_domains(prio)
        else:
            self.sim.groups = [None]
            self.sim.groups[0] = Group(self.sim, 0, range(n), self.cls, p, prio)
        self.alert_seq = 0
        self.alert_done = {}
        self.req_seq = 0
        self.pend_req = {}
        self.u = {g.gid: 0 for g in self.sim.groups}

    def priorities(self):
        # static priority follows nominal capacity
        order = sorted(range(self.n), key=lambda i: self.sim.nodes[i].cap)
        return {m: r for r, m in enumerate(order)}

    def make_domains(self, prio):
        """Scenario 7: one election group per ward; the two servers are candidates in all of them."""
        gs = []
        for w in range(self.wards):
            mem = [0, 1] + [i for i in range(2, self.n) if self.sim.nodes[i].ward == w]
            self.sim.groups.append(None)
            gs.append(Group(self.sim, w, mem, self.cls, self.p, {m: prio[m] for m in mem}))
            self.sim.groups[w] = gs[-1]

    # coordination workload ------------------------------------------------------------------
    def rate_now(self, g):
        return self.p.profile.at("lam", self.u[g.gid] / 2.0)

    def arrivals(self, g):
        sim = self.sim
        if sim.now > self.t_end:
            return
        self.issue(g)
        sim.at(sim.now + sim.rng_load.expovariate(self.rate_now(g)), self.arrivals, g)

    def issue(self, g):
        sim = self.sim
        rng = sim.rng_load
        cand = g.mem
        r = cand[int(rng.random() * len(cand))]
        psi = self.u[g.gid] / 2.0
        if psi > 0 and rng.random() < 0.7 and self.hot is not None:
            hot = [m for m in g.mem if sim.nodes[m].ward == self.hot]
            if hot:
                r = hot[int(rng.random() * len(hot))]
        D = self.p.profile.at("D", psi)
        inst = g.inst[r]
        in_win = sim.now >= self.t0
        if in_win:
            sim.m.req_total += 1
            sim.m.req_em_total += psi > 0
        if not sim.nodes[r].up:
            return
        l = inst.leader
        if l < 0:
            if in_win:
                sim.m.req_lost += 1
            return
        self.req_seq += 1
        rid = self.req_seq
        self.pend_req[rid] = [sim.now, D, r, l, g, in_win, psi > 0, False]
        if l == r:
            self.at_leader(r, l, rid, g)
        else:
            sim.send(r, l, "REQ", (rid,), None, 0, None)
        sim.at(sim.now + D, self.req_deadline, rid)
        sim.at(sim.now + self.p.Tresp, self.req_timeout, rid)

    def at_leader(self, src, l, rid, g):
        sim = self.sim
        nd = sim.nodes[l]
        inst = g.inst[l]
        if not inst.is_serving():
            return
        if nd.mal and sim.rng_svc.random() < nd.drop_q:
            return
        start = max(sim.now, nd.busy)
        nd.busy = start + sim.rng_svc.expovariate(max(1e-3, nd.mu()))
        if src == l:
            sim.at(nd.busy, self.req_done, rid)
        else:
            sim.at(nd.busy, sim.send, l, src, "RESP", (rid,), None, 0, None)

    def req_done(self, rid):
        rec = self.pend_req.pop(rid, None)
        if rec is None:
            return
        t0, D, r, l, g, win, em, missed = rec
        sim = self.sim
        if missed:
            return  # a late answer is neither good nor bad evidence: lateness shows through measured capacity
        g.inst[r].req_outcome(l, True)
        sim.m.note_req_ok(g.gid, t0)
        if win:
            sim.m.req_lat.append(sim.now - t0)
            sim.m.req_hit += 1
            sim.m.req_em_hit += em

    def req_deadline(self, rid):
        rec = self.pend_req.get(rid)
        if rec is not None:
            rec[7] = True
            if rec[5]:
                self.sim.m.req_lost += 1

    def req_timeout(self, rid):
        rec = self.pend_req.pop(rid, None)
        if rec is not None and self.sim.nodes[rec[2]].up:
            rec[4].inst[rec[2]].req_outcome(rec[3], False)

    # alert paths ----------------------------------------------------------------------------
    def emit_alert(self, g, src):
        sim = self.sim
        if not sim.nodes[src].up:
            return
        self.alert_seq += 1
        aid = self.alert_seq
        sim.m.alerts[aid] = [sim.now, None, sim.now >= self.t0]
        self.alert_try(aid, src, g, 0)

    def alert_try(self, aid, src, g, att):
        sim = self.sim
        if aid in self.alert_done or not sim.nodes[src].up:
            return
        if sim.now - sim.m.alerts[aid][0] > self.p.alert_ttl:
            return
        if self.ap:
            for k in self.sinks:
                sim.send(src, k, "ALERT", (aid, src, "dir"), None, AC_VO, None)
            peers = [m for m in g.mem if sim.nodes[m].ward != sim.nodes[src].ward]
            if peers:
                rly = peers[int(sim.rng_prot.random() * len(peers))]
                sim.send(src, rly, "ALERT", (aid, src, "rly"), None, AC_VO, None)
        else:
            l = g.inst[src].leader
            if l == src:
                self.leader_fwd(aid, src, g)
            elif l >= 0:
                sim.send(src, l, "ALERT", (aid, src, "ldr"), None, AC_VO, None)
        sim.at(sim.now + self.p.Tr_alert, self.alert_try, aid, src, g, att + 1)

    def leader_fwd(self, aid, src, g):
        sim = self.sim
        l = g.inst[src].leader
        if l < 0:
            return
        nd = sim.nodes[l]
        if not g.inst[l].is_serving():
            return
        if nd.mal and sim.rng_svc.random() < nd.drop_q:
            return
        for k in self.sinks:
            sim.send(l, k, "ALERT", (aid, src, "dir"), None, AC_VO, None)

    def on_plain(self, s, d, kind, msg):
        sim = self.sim
        if kind == "REQ":
            g = self.pend_req.get(msg[0])
            if g is not None:
                self.at_leader(s, d, msg[0], g[4])
        elif kind == "RESP":
            self.req_done(msg[0])
        elif kind == "ALERT":
            aid, src, how = msg
            if d in self.sinks:
                rec = sim.m.alerts[aid]
                if rec[1] is None:
                    rec[1] = sim.now
                sim.send(d, src, "AACK", (aid,), None, AC_VO, None)
            elif how == "rly":
                for k in self.sinks:
                    sim.send(d, k, "ALERT", (aid, src, "dir"), None, AC_VO, None)
            elif how == "ldr":
                g = self.group_of(src)
                if g.inst[d].is_serving() and not (sim.nodes[d].mal and sim.rng_svc.random() < sim.nodes[d].drop_q):
                    for k in self.sinks:
                        sim.send(d, k, "ALERT", (aid, src, "dir"), None, AC_VO, None)
        elif kind == "AACK":
            self.alert_done[msg[0]] = True

    def group_of(self, m):
        for g in self.sim.groups:
            if m in g.ix:
                return g
        return self.sim.groups[0]

    def alerts_for(self, g, rate, t0, t1, ward):
        """Poisson alert stream from gateways of a ward over [t0, t1)."""
        sim = self.sim
        t = t0
        gws = [m for m in g.mem if sim.nodes[m].kind == "G" and (ward is None or sim.nodes[m].ward == ward)]
        rng = sim.rng_scn
        while gws:
            t += rng.expovariate(rate)
            if t >= t1:
                break
            sim.at(t, self.emit_alert, g, gws[int(rng.random() * len(gws))])

    # fault injection ------------------------------------------------------------------------
    def set_u(self, g, u, ward=None):
        sim = self.sim
        self.u[g.gid] = u
        if ward is not None:
            self.hot = ward
        for m in g.mem:
            if sim.nodes[m].up:
                inst = g.inst[m]
                sim.at(sim.now + 0.002 + 0.008 * sim.rng_scn.random(), inst.set_urgency, u)

    def surge(self, node, load):
        self.sim.nodes[node].load = load

    def crash_leader(self, g, down_for, tries=0):
        sim = self.sim
        cur = g.serving()
        if not cur:
            # a vacancy at the scheduled instant: the fault hits the next leader, so no variant escapes it
            if tries < self.FAULT_TRIES:
                sim.at(sim.now + 0.05, self.crash_leader, g, down_for, tries + 1)
            else:
                self.skipped = True
            return
        l = cur[0]
        self.crash_node(l, down_for)

    def crash_node(self, l, down_for):
        sim = self.sim
        nd = sim.nodes[l]
        for g in sim.groups:
            if l in g.ix:
                if g.inst[l].is_serving():
                    sim.m.note_leader_lost(g.gid, l)
                g.inst[l].crash()
        nd.up = False
        if down_for:
            sim.at(sim.now + down_for, self.recover_node, l)

    def recover_node(self, l):
        sim = self.sim
        sim.nodes[l].up = True
        sim.nodes[l].busy = sim.now
        for g in sim.groups:
            if l in g.ix:
                g.inst[l].recover()

    def degrade(self, g, t0, ramp, floor, tries=0):
        """Scenario 4: the leader's residual capacity declines linearly during the ramp."""
        sim = self.sim
        cur = g.serving()
        if not cur:
            if tries < self.FAULT_TRIES:
                sim.at(sim.now + 0.05, self.degrade, g, sim.now + 0.05, ramp, floor, tries + 1)
            else:
                self.skipped = True
            return
        l = cur[0]
        nd = sim.nodes[l]
        start, batt0 = nd.load, nd.battery
        steps = 30
        for k in range(1, steps + 1):
            frac = k / steps
            sim.at(t0 + ramp * frac, self.surge, l, start + (floor - start) * frac)
            if not nd.mains:
                sim.at(t0 + ramp * frac, setattr, nd, "battery", max(0.05, batt0 * (1 - 0.8 * frac)))

    def ge_burst(self, node, t_end, loss, jit, mean_bad, mean_good):
        """Gilbert-Elliott state process on the radio of one gateway."""
        sim = self.sim
        rng = sim.rng_scn
        link = sim.link
        def to_bad():
            if sim.now >= t_end:
                link.bad.pop(node, None)
                return
            link.bad[node] = (loss, jit)
            sim.at(sim.now + rng.expovariate(1.0 / mean_bad), to_good)
        def to_good():
            link.bad.pop(node, None)
            sim.at(sim.now + rng.expovariate(1.0 / mean_good), to_bad)
        sim.at(sim.now + rng.expovariate(1.0 / mean_good), to_bad)

    def cut_leader(self, g, dur, tries=0):
        sim = self.sim
        cur = g.serving()
        if not cur:
            if tries < self.FAULT_TRIES:
                sim.at(sim.now + 0.05, self.cut_leader, g, dur, tries + 1)
            else:
                self.skipped = True
            return
        sim.cut[cur[0]] = sim.now + dur

    def start_partition(self, wards_a):
        sim = self.sim
        side = {}
        for nd in sim.nodes:
            side[nd.id] = 1 if nd.ward in wards_a else 0
        sim.partition = side

    def end_partition(self):
        self.sim.partition = None

    def compromise(self, node, drop_q, claim_x):
        """A compromised node drops traffic when leading, inflates claims and always campaigns first."""
        nd = self.sim.nodes[node]
        nd.mal, nd.drop_q, nd.claim_x = True, drop_q, claim_x
        for g in self.sim.groups:
            if node in g.ix:
                inst = g.inst[node]
                if hasattr(inst, "Tmin_to"):
                    # just past the point where voters still count a heartbeat as recent, or the attacker's
                    # campaign would be refused for a reason that has nothing to do with its behaviour
                    inst.timeout = lambda i=inst: i.Tmin_to() + 0.005
                    inst.to_cur = inst.timeout()

    def bad_mouth(self, liars, victims, praise, end):
        """Colluders overwrite their reports with false evidence that peers receive through gossip."""
        sim = self.sim
        g = sim.groups[0]

        def tick():
            if sim.now >= end:
                return
            for m in liars:
                inst = g.inst[m]
                if not sim.nodes[m].up or not hasattr(inst, "Rg") or inst.Rg.ndim != 2:
                    continue
                for v in victims:
                    x = g.ix[v]
                    inst.Rg[:, x] = 0.0
                    inst.Bd[:, x] = 30.0
                x = g.ix[praise]
                inst.Rg[:, x] = 30.0
                inst.Bd[:, x] = 0.0
            sim.at(sim.now + 1.0, tick)
        sim.at(sim.now, tick)

    # scenario scripts -----------------------------------------------------------------------
    def schedule(self):
        sim, s = self.sim, self.scn
        self.hot = None
        self.skipped = False
        groups = sim.groups
        g0 = groups[0]
        rng = sim.rng_scn
        for g in groups:
            sim.at(0.05 + 0.01 * g.gid, self.arrivals, g)
            self.alerts_for(g, 0.05, 1.0, self.t_end, None)
        if s == "S1":
            pass
        elif s in ("S2", "S3", "S2N"):
            te = 90.0 + rng.uniform(0, 5)
            w = 0
            self.window = (te, te + 60.0)
            sim.at(te, self.set_u, g0, 2, w)
            if s != "S2N":
                sim.at(te, self.surge, 0, 0.90)        # the ward's home server absorbs emergency analytics
                sim.at(te + 60.0, self.surge, 0, 0.30)
            sim.at(te + 60.0, self.set_u, g0, 0)
            self.alerts_for(g0, 2.0, te, te + 60.0, w)
            if s == "S3":
                sim.at(te + 12.0, self.crash_leader, g0, 60.0)
        elif s == "S4":
            te = 90.0
            self.window = (te, self.t_end)
            sim.at(te, self.degrade, g0, te, 30.0, 0.97)
            self.alerts_for(g0, 0.5, te, self.t_end, None)
        elif s == "S5":
            t1 = T0
            self.window = (t1, self.t_end)
            for m in g0.mem:
                if sim.nodes[m].kind == "G":
                    sim.at(t1, self.ge_burst, m, self.t_end, 0.5, 3.0, 0.5, 6.0)
            for k in range(4):
                sim.at(t1 + 20.0 + 40.0 * k + rng.uniform(0, 10), self.cut_leader, g0, rng.uniform(1.0, 3.0))
            tp = 170.0
            sim.at(tp, self.start_partition, {0})
            sim.at(tp + 15.0, self.end_partition)
            self.alerts_for(g0, 0.5, t1, self.t_end, None)
        elif s == "S6":
            self.window = (T0, self.t_end)
            attacker = 1      # second server: the runner-up under every capacity-aware policy
            top = max(range(self.n), key=lambda i: self.sim.nodes[i].cap)
            sim.at(60.0, self.compromise, attacker, 0.6, 5.0)
            gw = [m for m in g0.mem if sim.nodes[m].kind == "G"]
            liars = gw[: max(2, self.n // 10)]
            sim.at(60.0, self.bad_mouth, liars, [0], attacker, self.t_end)
            sim.at(90.0, self.crash_leader, g0, 30.0)
            self.alerts_for(g0, 0.5, T0, self.t_end, None)
        elif s == "S7":
            te = 90.0
            self.window = (te, te + 60.0)
            for k in range(min(3, len(groups))):
                g = groups[k]
                sim.at(te + 2.0 * k, self.set_u, g, 2, k)
                self.alerts_for(g, 2.0, te + 2.0 * k, te + 60.0, k)
                sim.at(te + 60.0, self.set_u, g, 0)
            sim.at(te, self.surge, 0, 0.75)            # both servers absorb analytics for the wards in crisis
            sim.at(te, self.surge, 1, 0.60)
            sim.at(te + 60.0, self.surge, 0, 0.30)
            sim.at(te + 60.0, self.surge, 1, 0.30)
        elif s == "S8":
            te = 80.0
            self.window = (te, self.t_end)
            sim.at(te, self.set_u, g0, 2, 0)
            sim.at(te + 8.0, self.crash_leader, g0, 0)
            self.alerts_for(g0, 2.0, te, self.t_end, 0)

    def snap_radio(self):
        self.radio0 = {nd.id: (nd.tx_msgs, nd.tx_bytes, nd.rx_msgs, nd.rx_bytes) for nd in self.sim.nodes}

    def run(self):
        wall = time.perf_counter()
        sim = self.sim
        for g in sim.groups:
            g.start()
        self.schedule()
        sim.at(self.t0, self.snap_radio)
        sim.run(self.t_end)
        wall = time.perf_counter() - wall
        return self.summary(wall)

    def summary(self, wall):
        sim, m = self.sim, self.sim.m
        groups = [g.gid for g in sim.groups]
        t0, t1 = self.t0, self.t_end
        win = getattr(self, "window", (t0, t1))
        avail, dual = m.availability(t0, t1, groups)
        avail_win, _ = m.availability(win[0], win[1], groups)
        lat = [a[1] - a[0] for a in m.alerts.values() if a[2] and a[1] is not None]
        # alerts raised while the leader had just failed: the case the alert plane exists for
        vac = [a[1] - a[0] for a in m.alerts.values() if a[1] is not None and any(0 <= a[0] - c[0] <= 1.0 for c in m.crash_t)]
        n_alert = sum(1 for a in m.alerts.values() if a[2])
        lat.sort()
        pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else float("nan")
        nodes = sim.nodes[: self.n]
        tot_e = 0.0
        for nd in nodes:
            if nd.kind == "G":
                tm, tb, rm, rb = self.radio0[nd.id]
                ttx = (nd.tx_msgs - tm) * FRAME_OH + (nd.tx_bytes - tb) * 8 / PHY_RATE
                trx = (nd.rx_msgs - rm) * FRAME_OH + (nd.rx_bytes - rb) * 8 / PHY_RATE
                tot_e += VOLT * (TX_I * ttx + RX_I * trx)
        gw = max(1, sum(1 for nd in nodes if nd.kind == "G"))
        first = [m.first_leader.get(g, float("nan")) for g in groups]
        mal_t, mal_inc, mal_wins = 0.0, 0, 0
        if self.scn == "S6":
            evs = sorted(e for e in m.serve_events if e[2] == 1)
            start = None
            for t, _, _, sgn in evs:
                mal_wins += sgn > 0 and t >= 60.0
                if sgn > 0:
                    start = t
                elif start is not None:
                    mal_inc += start < 60.0 < t
                    mal_t += max(0.0, t - max(start, 60.0))
                    start = None
            if start is not None:
                mal_inc += start < 60.0
                mal_t += t1 - max(start, 60.0)
        # tenures that end in the window, and those still running at the end, which would otherwise be lost
        tenures = [d for e, d in m.tenures if e >= t0] + [t1 - s for s in m.serving.values()]
        tenure = statistics.mean(tenures) if tenures else float("nan")
        episodes, closed = m.vacancies(t0, groups, VACANCY_LIMIT)
        return {
            "scn": self.scn, "variant": self.variant, "seed": self.seed, "n": self.n,
            "first_election_s": max(first) if first else float("nan"),
            "reelect_s": statistics.mean(m.reelect) if m.reelect else float("nan"),
            "recovery_s": statistics.mean(m.recover) if m.recover else float("nan"),
            "alert_mean_s": statistics.mean(lat) if lat else float("nan"),
            "alert_p50_s": pct(0.5), "alert_p95_s": pct(0.95), "alert_p99_s": pct(0.99),
            "alert_max_s": lat[-1] if lat else float("nan"),
            "alert_vac_mean_s": statistics.mean(vac) if vac else float("nan"),
            "alert_delivery": (len(lat) / n_alert) if n_alert else float("nan"),
            "avail": avail, "avail_window": avail_win, "dual_leader_s": dual,
            "leader_changes": m.changes, "tenure_s": tenure,
            "elec_msgs": m.elec_msgs, "ctrl_msgs": m.ctrl_msgs, "ctrl_bytes": m.ctrl_bytes,
            "msgs_per_node_s": m.ctrl_msgs / (self.n * (t1 - t0)),
            "kbytes_per_node_s": m.ctrl_bytes / 1e3 / (self.n * (t1 - t0)),
            "energy_mW_per_gw": 1e3 * tot_e / (gw * (t1 - t0)),
            "elec_episodes": episodes, "elec_success": (closed / episodes) if episodes else float("nan"),
            "req_total": m.req_total, "req_hit": m.req_hit, "hit_ratio": m.req_hit / max(1, m.req_total),
            "hit_ratio_em": (m.req_em_hit / m.req_em_total) if m.req_em_total else float("nan"),
            "req_lat_p50": statistics.median(m.req_lat) if m.req_lat else float("nan"),
            "mal_lead_s": mal_t, "mal_incumbent": mal_inc, "mal_wins": mal_wins, "skipped": self.skipped, "wall_s": wall, "events": next(sim.c),
        }


def run_one(scn, variant, seed, table, n=20, wards=4, overrides=None, fast_hb=None):
    return Harness(scn, variant, seed, table, n, wards, overrides, fast_hb).run()
