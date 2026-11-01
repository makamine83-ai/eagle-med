"""Election protocols compared in the study.

QuorumNode implements Raft's term, pre-vote, vote and check-quorum rules. Eagle, ResRaft and
TrustRaft extend it with candidate-selection policies, so all of them inherit the same safety
mechanism. Bully and Ring are the classical non-quorum algorithms on the same failure detector.
"""
import math
import warnings

import numpy as np

from .core import AC_BE, AC_VO, SIZE
from .params import EagleParams, Profile, margin

warnings.filterwarnings("ignore", message="All-NaN slice")
INF = float("inf")
HB_BYTES = SIZE["HB"]


def reported_median(rec, seen, count):
    """Median of the reports on each target, over the observers that have reported it; zero if none has."""
    a = np.where(seen, rec, INF)
    a.sort(axis=1)
    pick = lambda k: np.take_along_axis(a, np.broadcast_to(k[None, None, :], (a.shape[0], 1, a.shape[2])), axis=1)[:, 0, :]
    with np.errstate(invalid="ignore"):
        med = 0.5 * (pick(np.maximum(count - 1, 0) // 2) + pick(count // 2))
    return np.where(count > 0, med, 0.0)


class Group:
    """One election domain: members, quorum size and per-node protocol instances."""

    def __init__(self, sim, gid, members, cls, p, prio=None, **kw):
        self.sim, self.gid, self.mem, self.p = sim, gid, list(members), p
        self.ix = {m: i for i, m in enumerate(self.mem)}
        self.n = len(self.mem)
        self.Q = self.n // 2 + 1
        self.prio = prio or {m: m for m in self.mem}
        self.inst = {m: cls(self, m, **kw) for m in self.mem}
        self.u_true = 0.0

    def start(self):
        for m in self.mem:
            self.inst[m].start()

    def serving(self):
        return [m for m in self.mem if self.inst[m].is_serving()]


class QuorumNode:
    fixed_hb = None
    adaptive = False

    def __init__(self, g, nid):
        self.g, self.s, self.id, self.p = g, g.sim, nid, g.p
        self.node = self.s.nodes[nid]
        self.mem, self.Q, self.gid = g.mem, g.Q, g.gid
        self.role, self.term, self.voted, self.leader = "F", 0, -1, -1
        self.last_hb = -INF         # reference of the election timer: heartbeats, votes granted, own campaigns
        self.hb_rx = -INF           # last heartbeat actually received
        self.to_cur = 0.0
        self.timer_on = False
        self.ack_t = {}
        self.serve = False
        self.pv_term, self.pv = -1, set()
        self.pv_reason = "timeout"
        self.hold = 0.0
        self.vote_recent = False    # a voter of this campaign still saw a live leader
        self.votes = set()
        self.tick_gen = 0
        self.fail_camp = 0

    # helpers --------------------------------------------------------------------------------
    def is_serving(self):
        return self.serve and self.node.up

    def peers(self):
        return [m for m in self.mem if m != self.id]

    def psi(self):
        return 0.0

    def Thb(self):
        if self.fixed_hb is not None:
            return self.fixed_hb
        return self.p.profile.at("Thb", self.psi())

    def Tmin_to(self):
        return self.p.K * self.Thb()

    def timeout(self):
        To = self.Tmin_to()
        return To + self.s.rng_prot.random() * To

    def broadcast(self, kind, msg, size=None, ac=AC_BE):
        for m in self.peers():
            self.s.send(self.id, m, kind, msg, size, ac, self.gid)

    # lifecycle ------------------------------------------------------------------------------
    def start(self):
        self.to_cur = self.timeout()
        self.last_hb = self.s.now
        self.ensure_timer()

    def ensure_timer(self):
        if not self.timer_on:
            self.timer_on = True
            self.s.at(self.last_hb + self.to_cur, self.on_timer)

    def on_timer(self):
        self.timer_on = False
        if not self.node.up or self.role == "L":
            return
        due = self.last_hb + self.to_cur
        if self.s.now + 1e-9 < due:
            self.ensure_timer()
            return
        self.campaign("timeout")
        self.last_hb = self.s.now
        self.to_cur = self.timeout()
        self.ensure_timer()

    def crash(self):
        if self.serve:
            self.s.m.leader_down(self.gid, self.id)
        self.serve = False
        self.role = "F"
        self.tick_gen += 1

    def recover(self):
        self.role, self.leader, self.serve = "F", -1, False
        self.last_hb = self.s.now
        self.to_cur = self.timeout()
        self.ensure_timer()

    # election -------------------------------------------------------------------------------
    def campaign(self, reason):
        self.pv_term, self.pv = self.term + 1, {self.id}
        self.pv_reason = reason
        self.broadcast("PV", (self.pv_term, reason))

    def recent_hb(self):
        return self.s.now - self.hb_rx < self.Tmin_to() and self.leader >= 0

    def grant_pv(self, cand, term, reason):
        return term > self.term and not self.recent_hb()

    def vote_ok(self, cand, reason):
        return True

    def evidence_against_leader(self, cand):
        """True when this voter's own view justifies replacing a leader it still hears."""
        return False

    def on_pv(self, src, term, reason):
        self.s.send(self.id, src, "PVR", (term, self.grant_pv(src, term, reason) and term > self.term), None, AC_BE, self.gid)

    def on_pvr(self, src, term, grant):
        if term != self.pv_term or not grant or self.role == "L":
            return
        self.pv.add(src)
        if len(self.pv) >= self.Q and self.term < term:
            self.term, self.role, self.voted = term, "C", self.id
            self.votes = {self.id}
            self.vote_recent = False
            self.s.m.attempts += 1
            self.broadcast("RV", (term, self.pv_reason))

    def on_rv(self, src, term, reason):
        recent = self.role != "L" and self.recent_hb()
        if term > self.term and recent and not self.evidence_against_leader(src):
            # the pre-vote rule holds at the vote too: a heartbeat may have arrived since, or the pre-vote was skipped
            self.s.send(self.id, src, "RVR", (self.term, False, False), None, AC_BE, self.gid)
            return
        if term > self.term:
            if self.role == "L":
                self.s.send(self.id, src, "SD", (term,), None, AC_BE, self.gid)
            self.step_down(term)
        grant = term == self.term and self.voted in (-1, src) and self.vote_ok(src, reason)
        if grant:
            self.voted = src
            self.last_hb = self.s.now
        self.s.send(self.id, src, "RVR", (self.term, grant, recent), None, AC_BE, self.gid)

    def on_rvr(self, src, term, grant, recent):
        if term > self.term:
            self.step_down(term)
            return
        if self.role != "C" or term != self.term or not grant:
            return
        self.vote_recent = self.vote_recent or recent
        self.votes.add(src)
        if len(self.votes) >= self.Q:
            self.become_leader()

    def become_leader(self):
        self.role, self.leader, self.serve = "L", self.id, False
        self.fail_camp = 0
        self.s.m.wins += 1
        self.hold = 0.0
        self.ack_t = {}
        self.start_acks = set()
        self.tick_gen += 1
        self.lead_t = self.s.now
        # a voter that still heard the old leader means it may hold a lease: wait out the longest one, or its step-down
        self.hold = self.s.now + self.p.K * self.Thb() + self.p.drift_margin if self.vote_recent else 0.0
        self.tick(self.tick_gen)

    def step_down(self, term):
        if self.serve:
            self.s.m.leader_down(self.gid, self.id)
        was_leader = self.role == "L"
        self.serve = False
        if term > self.term:
            self.voted = -1  # one durable vote per term
        self.role, self.term = "F", term
        self.tick_gen += 1
        if was_leader:
            self.last_hb = self.s.now
            self.to_cur = self.timeout()
            self.ensure_timer()

    def hb_size(self):
        return HB_BYTES

    def lease(self):
        """A leader stops serving before any follower that acknowledged it may vote again."""
        return (self.p.K - 1) * self.Thb() - self.p.drift_margin

    def tick(self, gen):
        if gen != self.tick_gen or self.role != "L" or not self.node.up:
            return
        now = self.s.now
        lease = self.lease()
        acks = sorted(self.ack_t.values(), reverse=True)
        tq = acks[self.Q - 2] if len(acks) >= self.Q - 1 else self.lead_t
        if now - max(tq, self.lead_t) > lease:
            self.step_down(self.term)
            self.leader = -1
            return
        self.broadcast("HB", (self.term, now), self.hb_size())
        self.s.at(now + self.Thb(), self.tick, gen)

    def on_hb(self, src, term, t_send):
        if term < self.term:
            self.s.send(self.id, src, "ACK", (self.term, t_send), None, AC_BE, self.gid)
            return
        if term > self.term or self.role != "F":
            self.step_down(term)
        self.accept_leader(src)
        self.last_hb = self.hb_rx = self.s.now
        self.s.send(self.id, src, "ACK", (term, t_send), None, AC_BE, self.gid)

    def accept_leader(self, src):
        self.leader = src

    def on_ack(self, src, term, t_send):
        if term > self.term:
            self.step_down(term)
            self.leader = -1
            return
        if self.role != "L":
            return
        self.ack_t[src] = max(self.ack_t.get(src, 0.0), t_send)
        if not self.serve:
            self.start_acks.add(src)
            self.try_serve()

    def try_serve(self):
        if self.role == "L" and not self.serve and len(self.start_acks) + 1 >= self.Q and self.s.now >= self.hold:
            self.serve = True
            self.node.busy = max(self.node.busy, self.s.now + self.p.Tsync)
            self.s.m.leader_up(self.gid, self.id)

    def on_msg(self, src, kind, msg):
        if kind == "HB":
            self.on_hb(src, *msg)
        elif kind == "ACK":
            self.on_ack(src, *msg)
        elif kind == "PV":
            self.on_pv(src, *msg)
        elif kind == "PVR":
            self.on_pvr(src, *msg)
        elif kind == "RV":
            self.on_rv(src, *msg)
        elif kind == "RVR":
            self.on_rvr(src, *msg)
        elif kind == "SD":
            if self.role == "L" and msg[0] == self.term:
                self.hold = 0.0
                self.try_serve()
        else:
            self.on_extra(src, kind, msg)

    def on_extra(self, src, kind, msg):
        pass

    # outcome feedback from the coordination workload
    def req_outcome(self, leader, ok):
        pass

    def set_urgency(self, u):
        pass


class Raft(QuorumNode):
    """Raft leader election with PreVote and CheckQuorum [Ongaro and Ousterhout]."""


class EvidenceNode(QuorumNode):
    """Gossip of liveness counters, capacity claims and probes, shared by evidence-based policies."""

    digest_entries = 10

    def __init__(self, g, nid):
        super().__init__(g, nid)
        n = g.n
        self.n = n
        self.me = g.ix[nid]
        self.ctr = np.zeros(n)
        self.heard = np.full(n, -INF)
        self.claim = np.zeros(n)
        self.batt = np.ones(n)
        self.ew_w = np.zeros(n)     # EWMA of probe service time
        self.minrtt = np.full(n, INF)
        self.dh = np.full(n, 0.01)  # delay estimate
        self.pend = {}
        self.seq = 0

    def start(self):
        super().start()
        self.t_eval = self.s.now
        self.gossip_on = True
        self.s.at(self.s.now + self.s.rng_prot.random() * self.p.Tg, self.gossip_tick)

    def recover(self):
        super().recover()
        self.t_eval = self.s.now  # nothing was observed while down
        self.pend.clear()
        if not self.gossip_on:
            self.gossip_on = True
            self.s.at(self.s.now + self.s.rng_prot.random() * self.p.Tg, self.gossip_tick)

    def digest_size(self):
        return 96 + 2 * self.n + 40 * min(self.n, self.digest_entries)

    def own_claim(self):
        """Capacity offered to this group: what other leaderships held by this node leave free."""
        nd = self.node
        others = nd.leads - (1 if self.serve else 0)
        return max(1.0, nd.mu() - others * self.p.profile.lam[0]) * nd.claim_x

    def gossip_tick(self):
        if not self.node.up:
            self.gossip_on = False
            return
        now = self.s.now
        self.ctr[self.me] += 1
        self.heard[self.me] = now
        self.claim[self.me] = self.own_claim()
        self.batt[self.me] = self.node.battery
        # probes unanswered for a full period count as communication failures
        for k, (t0, _) in list(self.pend.items()):
            if now - t0 > 0.9 * self.p.Tg:
                self.note_probe(k[0], False, 0.0)
                del self.pend[k]
        self.evaluate()
        peers = self.mem
        rng = self.s.rng_prot
        pick = set()
        while len(pick) < min(self.p.gossip_fanout, self.n - 1):
            m = peers[int(rng.random() * self.n)]
            if m != self.id:
                pick.add(m)
        dig = self.make_digest()
        for m in pick:
            self.seq += 1
            self.pend[(m, self.seq)] = (now, self.seq)
            self.s.send(self.id, m, "PING", (self.seq, dig), self.digest_size(), AC_BE, self.gid)
        self.s.at(now + self.p.Tg, self.gossip_tick)

    def make_digest(self):
        return (self.ctr.copy(), self.claim.copy(), self.batt.copy())

    def merge_digest(self, src, dig):
        ctr, claim, batt = dig[:3]
        newer = ctr > self.ctr
        if newer.any():
            self.ctr[newer] = ctr[newer]
            self.heard[newer] = self.s.now
            self.claim[newer] = claim[newer]
            self.batt[newer] = batt[newer]

    def on_extra(self, src, kind, msg):
        if kind == "PING":
            seq, dig = msg
            self.merge_digest(src, dig)
            wait = self.s.rng_svc.expovariate(max(1e-3, self.node.mu()))
            self.s.at(self.s.now + wait, self.s.send, self.id, src, "PONG", (seq,), None, AC_BE, self.gid)
        elif kind == "PONG":
            key = (src, msg[0])
            rec = self.pend.pop(key, None)
            if rec is not None:
                rtt = self.s.now - rec[0]
                x = self.g.ix[src]
                self.minrtt[x] = min(self.minrtt[x], rtt)
                w = max(1e-4, rtt - self.minrtt[x])
                a = self.p.alpha
                self.ew_w[x] = w if self.ew_w[x] == 0 else (1 - a) * self.ew_w[x] + a * w
                self.dh[x] = (1 - a) * self.dh[x] + a * min(rtt / 2, self.minrtt[x])
                self.note_probe(src, True, w)
        elif kind == "XFER":
            if self.role == "F":
                self.campaign("risk")

    def note_probe(self, peer, ok, w):
        pass

    def evaluate(self):
        pass


class Eagle(EvidenceNode):
    """EAGLE-Med: Stages 1 to 6. The alert plane of Stage 7 lives in the scenario harness."""

    def __init__(self, g, nid):
        super().__init__(g, nid)
        n = self.n
        self.P: Profile = self.p.profile
        # evidence per class (communication, availability, compliance): good and bad masses
        self.Rg = np.zeros((3, n))
        self.Bd = np.zeros((3, n))
        self.recR = np.zeros((3, n, n))  # reports of each observer about each target
        self.recB = np.zeros((3, n, n))
        self.recS = np.zeros((n, n), bool)  # which observer has reported which target
        self.recM = np.zeros((n, n))     # measured service rates reported by observers
        self.recD = np.full((n, n), np.nan)
        self.A = np.full(n, 0.5)
        self.u_ext, self.u_ver, self.psi_v, self.psi_t = 0.0, -INF, 0.0, 0.0
        self.measured = np.zeros(n, bool)
        self.t_sw = -INF
        self.t_risk = -INF
        self.top = np.arange(min(n, self.digest_entries))
        self.Elig = np.ones(n, bool)
        self.rho = np.zeros(n)
        self.S = np.ones(n)
        self.score_ready = False
        self.quar = np.zeros(n)
        self.strikes = np.zeros(n)

    # Stage 1 --------------------------------------------------------------------------------
    def set_urgency(self, u, ver=None):
        """The newest label wins, so that a delayed digest cannot revive an emergency already cleared."""
        now = self.s.now
        self.psi_update(now)
        self.u_ext, self.u_ver = u, now if ver is None else ver
        self.psi_v = max(self.psi_v, u / 2.0)

    def psi_update(self, now):
        dt = now - self.psi_t
        if dt > 0:
            self.psi_v = max(self.u_ext / 2.0, self.psi_v * math.exp(-dt / self.p.tau_u))
            self.psi_t = now

    def psi(self):
        if not self.p.use_urgency:
            return 0.0
        self.psi_update(self.s.now)
        return self.psi_v

    # evidence -------------------------------------------------------------------------------
    def note_probe(self, peer, ok, w):
        x = self.g.ix[peer]
        if ok:
            self.Rg[0, x] += 1.0
        else:
            self.Bd[0, x] += 1.0

    def req_outcome(self, leader, ok):
        if leader not in self.g.ix:
            return
        x = self.g.ix[leader]
        if ok:
            self.Rg[2, x] += 1.0
        elif leader == self.leader and self.s.now - self.hb_rx < self.Tmin_to():
            # selective silence: requests go unanswered while the same leader's heartbeats still arrive
            self.Bd[2, x] += self.p.neg
        else:
            self.Bd[1, x] += 1.0  # silent on every channel: a failure or outage, not misbehaviour

    def digest_size(self):
        return super().digest_size() if self.p.use_trust else 96 + 2 * self.n

    def make_digest(self):
        idx = self.top
        return (self.ctr.copy(), self.claim.copy(), self.batt.copy(), idx, self.Rg[:, idx].copy(), self.Bd[:, idx].copy(),
                self.meas_vec()[idx], self.dh[idx].copy(), self.u_ext, self.u_ver)

    def meas_vec(self):
        with np.errstate(divide="ignore"):
            return np.where(self.ew_w > 0, 1.0 / self.ew_w, 0.0)

    def merge_digest(self, src, dig):
        super().merge_digest(src, dig)
        ctr, claim, batt, idx, r, b, m, d, u, ver = dig
        x = self.g.ix[src]
        self.recR[:, x, idx] = r
        self.recB[:, x, idx] = b
        self.recS[x, idx] = True
        self.recM[x, idx] = m
        self.recD[x, idx] = d
        if ver > self.u_ver:
            self.set_urgency(u, ver)

    # Stages 2 to 5 --------------------------------------------------------------------------
    def assess(self, psi):
        """Trust lower bound, availability, capacity, deadline-hit probability and eligibility."""
        p, P = self.p, self.P
        n = self.n
        if p.use_trust:
            ob = np.ones(n, bool)
            ob[self.me] = False
            obs = np.nonzero(ob)[0]
            seen = self.recS[obs]
            count = seen.sum(axis=0)
            mR = reported_median(self.recR[:, obs, :], seen, count)
            mB = reported_median(self.recB[:, obs, :], seen, count)
            # each observer is one independent sample of a candidate: the median bounds what a minority
            # of liars can do, and the observer count restores the confidence of the pooled estimate
            scale = p.omega * count
            Rp = self.Rg + scale * mR
            Bp = self.Bd + scale * mB
            N = Rp + Bp
            tau = (Rp + 1.0) / (N + 2.0)
            sig = np.sqrt(tau * (1.0 - tau) / (N + 3.0))
            Tk = tau - P.at("kappa", psi) * sig
            # communication and availability rank candidates; compliance is a gate that applies
            # once a candidate has produced such evidence, so incumbents are not ranked on exposure
            T = np.clip(Tk[:2].min(axis=0), 1e-3, 1.0)
            Tg = np.where(N[2] >= p.Nmin, np.minimum(T, Tk[2]), T)
            self.trip = (N[2] >= p.Nmin) & (Tk[2] < P.at("Tmin", psi))
            mrep = self.recM.copy()
            mrep[self.me] = self.meas_vec()
            with np.errstate(invalid="ignore", all="ignore"):
                mm = np.nanmedian(np.where(mrep > 0, mrep, np.nan), axis=0)
            self.measured = ~np.isnan(mm)
            # a claim nobody has measured yet counts at half weight, and is no evidence of anything
            mu = np.where(self.measured, np.minimum(self.claim, (1 + p.xi) * mm), self.claim * 0.5)
            mu[self.me] = self.own_claim() / self.node.claim_x
        else:
            T = np.ones(n)
            Tg = T
            self.trip = np.zeros(n, bool)
            mu = self.claim.copy()
        drep = self.recD.copy()
        drep[self.me] = self.dh
        d = np.nanmedian(drep, axis=0)
        d = np.where(np.isnan(d), self.dh, d)
        lam, D = P.at("lam", psi), P.at("D", psi)
        gap = np.maximum(mu - lam, 0.0)
        slack = np.maximum(D - d, 0.0)
        Phi = 1.0 - np.exp(-gap * slack)
        sec = np.ones(n, bool)
        Elig = (Tg >= P.at("Tmin", psi)) & (self.A >= P.at("Amin", psi)) & (d <= D / 2) & (mu >= lam) & sec
        return T, mu, Phi, Elig

    def evaluate(self):
        p, P = self.p, self.P
        now = self.s.now
        dt = now - self.t_eval
        self.t_eval = now
        self.s.m.eval_ops += self.n
        # forgetting and availability evidence
        if p.use_trust:
            f0, f2 = p.gamma ** dt, p.gamma_c ** dt
            self.Rg[:2] *= f0
            self.Bd[:2] *= f0
            self.Rg[2] *= f2
            self.Bd[2] *= f2
            self.recR[:2] *= f0
            self.recB[:2] *= f0
            self.recR[2] *= f2
            self.recB[2] *= f2
        fresh = (now - self.heard) < 4.0 * p.Tg
        fresh[self.me] = True
        self.A += p.alpha * (fresh.astype(float) - self.A)
        if p.use_trust:
            self.Rg[1] += fresh * dt
            self.Bd[1] += (~fresh) * dt
        psi = self.psi()
        T, mu, Phi, Elig = self.assess(psi)
        new = self.trip & (now >= self.quar)
        if new.any():
            # a failed compliance gate suspends the candidate, doubling on repeated offences
            self.quar[new] = now + p.Tq * 2.0 ** self.strikes[new]
            self.strikes[new] += 1
        held = now < self.quar
        Elig = Elig & ~held
        # graded relaxation: if nothing qualifies, step back towards the normal profile
        lev = psi
        while not Elig.any() and lev > 0:
            lev = max(0.0, lev - 0.5)
            T, mu, Phi, Elig = self.assess(lev)
            Elig = Elig & ~held
        if not Elig.any():
            # last resort: any known node that is not suspended; suspensions are never relaxed
            Elig = ((self.claim > 0) | (np.arange(self.n) == self.me)) & ~held
            Phi = np.maximum(Phi, 0.01)
        if p.use_trust:
            self.anomaly(mu, dt)
        w = np.array(P.wE) * psi + np.array(P.wN) * (1 - psi) if p.use_adaptive_w else np.array(P.wN)
        batt = np.clip(self.batt, 0.01, 1.0)
        x = np.vstack([T, np.clip(self.A, 1e-3, 1), np.clip(Phi, 1e-6, 1), batt])
        rho = -(w[:, None] * np.log(x)).sum(axis=0)
        self.rho = rho
        self.S = np.exp(-rho)
        self.Elig = Elig
        self.score_ready = True
        order = np.lexsort((np.arange(self.n), np.where(Elig, rho, rho + 1e3)))
        k = min(self.n, self.digest_entries)
        head = order[: k - k // 2]
        rest = order[k - k // 2:]
        if len(rest):
            rot = int(self.s.rng_prot.random() * len(rest))
            head = np.concatenate([head, np.roll(rest, rot)[: k // 2]])
        self.top = np.sort(head)
        best = int(order[0])
        self.rank = int(np.nonzero(order == self.me)[0][0])
        self.best = best
        if self.role != "L":
            self.to_cur = self.timeout()
        # Stage 6 triggers
        delta = margin(p, psi) if p.use_hysteresis else 0.0
        if self.role == "L":
            x_me = self.me
            bad = not Elig[x_me] and self.probation_ok()
            better = rho[x_me] - rho[best] > delta
            if best != x_me and self.serve and (bad or (better and self.dwell_ok())):
                self.s.send(self.id, self.mem[best], "XFER", (self.term,), None, AC_BE, self.gid)
        elif self.leader >= 0 and self.leader in self.g.ix and best == self.me:
            lx = self.g.ix[self.leader]
            bad = not Elig[lx] and self.probation_ok()
            better = rho[lx] - rho[best] > delta
            if bad or (better and self.dwell_ok()):
                self.campaign("risk")

    def anomaly(self, mu, dt):
        """Claim gap: a claimed capacity beyond what probes support is negative compliance evidence."""
        p = self.p
        a = np.zeros(self.n)
        valid = (self.claim > 0) & self.measured
        a[valid] = np.clip(1.0 - mu[valid] / self.claim[valid], 0.0, 1.0)  # mu already allows tolerance xi
        a[self.me] = 0.0
        self.Bd[2] += p.eta * a * dt

    def dwell_ok(self):
        if not self.p.use_hysteresis:
            return True
        return self.s.now - self.t_sw >= self.p.Tdw

    def probation_ok(self):
        """A new leader cannot be judged on evidence it has not had time to produce."""
        if not self.p.use_hysteresis:
            return True
        return self.s.now - self.t_sw >= self.p.Tpb

    def accept_leader(self, src):
        if src != self.leader:
            self.t_sw = self.s.now
        self.leader = src
        self.fail_camp = 0                       # a leader is back: the vacancy that relaxed the filter is over
        self.heard[self.g.ix[src]] = self.s.now  # a heartbeat is direct proof of life

    def become_leader(self):
        self.t_sw = self.s.now  # dwell and probation also bind the leader's own hand-off
        super().become_leader()

    def campaign(self, reason):
        # a quorum that just refused a preemptive attempt is not asked again at once
        if reason == "risk" and self.p.use_hysteresis:
            if self.s.now - self.t_risk < self.p.Trt:
                return
            self.t_risk = self.s.now
        super().campaign(reason)

    def on_timer(self):
        # a lost lease is availability evidence against that leader; the median keeps it group-wide
        if self.node.up and self.role != "L" and self.leader in self.g.ix and self.s.now + 1e-9 >= self.last_hb + self.to_cur:
            self.Bd[1, self.g.ix[self.leader]] += self.p.neg
        super().on_timer()

    def timeout(self):
        To = self.Tmin_to()
        rank = getattr(self, "rank", self.n // 2)
        if not self.score_ready:
            # bootstrap: order candidates by claimed capacity only
            return To + self.p.Tr * self.n / 2 * self.s.rng_prot.random() + self.p.Tj * self.s.rng_prot.random()
        if not self.Elig[self.me]:
            rank = max(rank, self.n // 2)
        return To + self.p.Tr * rank + self.p.Tj * self.s.rng_prot.random()

    def cand_ok(self, cand):
        if not self.score_ready:
            return True
        x = self.g.ix[cand]
        if not self.Elig[x]:
            return False
        # after failed campaigns voters relax the score filter, never eligibility
        return self.fail_camp >= self.p.relax_after or bool(self.rho[x] <= self.rho[self.Elig].min() + self.p.eps_v + 1e-3)

    def leader_rejected(self, cand):
        """True when this voter's own evidence supports replacing the incumbent."""
        if self.leader not in self.g.ix or not self.score_ready:
            return False
        lx, cx = self.g.ix[self.leader], self.g.ix[cand]
        if not self.Elig[lx] and self.probation_ok():
            return True
        delta = margin(self.p, self.psi()) if self.p.use_hysteresis else 0.0
        return self.rho[lx] - self.rho[cx] > delta and self.dwell_ok()

    def evidence_against_leader(self, cand):
        return self.leader_rejected(cand)

    def grant_pv(self, cand, term, reason):
        if self.recent_hb() and not self.leader_rejected(cand):
            return False
        if not self.cand_ok(cand):
            self.fail_camp += 1
            return False
        return True

    def vote_ok(self, cand, reason):
        return self.cand_ok(cand)


class ResRaft(EvidenceNode):
    """Resource-aware Raft: timeouts shrink with a static weighted resource score; claims are trusted."""
    digest_entries = 0

    def __init__(self, g, nid):
        super().__init__(g, nid)
        self.score = np.zeros(self.n)

    def digest_size(self):
        return 96 + 10 * self.n

    def evaluate(self):
        c = self.claim
        cmax = max(c.max(), 1e-6)
        d = self.dh
        dmax = max(d.max(), 1e-6)
        self.score = 0.5 * c / cmax + 0.3 * (1 - d / dmax) + 0.2 * self.batt
        self.score[(self.claim <= 0) | (self.s.now - self.heard >= 4.0 * self.p.Tg)] = 0.0
        if self.role != "L":
            self.to_cur = self.timeout()

    def timeout(self):
        To = self.Tmin_to()
        sc = float(self.score[self.me]) if hasattr(self, "score") else 0.5
        rnk = (self.score > sc).sum() / max(1, self.n) if hasattr(self, "score") else 0.5
        return To + To * rnk + self.p.Tj * self.s.rng_prot.random()

    def vote_ok(self, cand, reason):
        # the reference is the best node heard recently: a silent top node must not block every vote
        x = self.g.ix[cand]
        fresh = self.s.now - self.heard < 1.5 * self.p.Tg
        top = self.score[fresh].max() if fresh.any() else 0.0
        return self.score[x] == 0.0 or self.score[x] >= top - 0.1


class TrustRaft(EvidenceNode):
    """Raft with a beta-reputation filter: binary threshold on the posterior mean, direct evidence only."""
    theta = 0.5

    def __init__(self, g, nid):
        super().__init__(g, nid)
        self.Rg = np.zeros(self.n)
        self.Bd = np.zeros(self.n)
        self.block = False

    def digest_size(self):
        return 96 + 8 * self.n

    def note_probe(self, peer, ok, w):
        x = self.g.ix[peer]
        if ok:
            self.Rg[x] += 1.0
        else:
            self.Bd[x] += 1.0

    def req_outcome(self, leader, ok):
        if leader in self.g.ix:
            x = self.g.ix[leader]
            if ok:
                self.Rg[x] += 1.0
            else:
                self.Bd[x] += 1.0

    def tau(self, x):
        return (self.Rg[x] + 1.0) / (self.Rg[x] + self.Bd[x] + 2.0)

    def evaluate(self):
        dt = self.s.now - self.t_eval
        self.t_eval = self.s.now
        self.Rg *= self.p.gamma ** dt
        self.Bd *= self.p.gamma ** dt

    def on_hb(self, src, term, t_send):
        # a voter that distrusts the leader stops refreshing its timer, which leads to re-election
        x = self.g.ix[src]
        if self.tau(x) < self.theta and term >= self.term:
            self.leader = src
            self.s.send(self.id, src, "ACK", (term, t_send), None, AC_BE, self.gid)
            return
        super().on_hb(src, term, t_send)

    def evidence_against_leader(self, cand):
        return self.leader in self.g.ix and self.tau(self.g.ix[self.leader]) < self.theta

    def grant_pv(self, cand, term, reason):
        if self.evidence_against_leader(cand):
            return self.vote_ok(cand, reason)
        return super().grant_pv(cand, term, reason) and self.vote_ok(cand, reason)

    def vote_ok(self, cand, reason):
        return self.tau(self.g.ix[cand]) >= self.theta


class ClassicNode:
    """Shared heartbeat failure detector for the non-quorum algorithms."""
    fixed_hb = None

    def __init__(self, g, nid):
        self.g, self.s, self.id, self.p = g, g.sim, nid, g.p
        self.node = self.s.nodes[nid]
        self.gid, self.mem = g.gid, g.mem
        self.prio = g.prio
        self.leader = -1
        self.serve = False
        self.last_hb = -INF
        self.tick_gen = 0
        self.timer_on = False

    def is_serving(self):
        return self.serve and self.node.up

    def Thb(self):
        return self.fixed_hb if self.fixed_hb else self.p.profile.Thb[0]

    def to(self):
        return self.p.K * self.Thb()

    def start(self):
        self.last_hb = self.s.now
        self.s.at(self.s.now + self.s.rng_prot.random() * 0.05, self.begin)

    def begin(self):
        self.detect_arm()
        self.start_election()

    def detect_arm(self):
        if not self.timer_on:
            self.timer_on = True
            due = self.s.now + self.to() if self.leader == self.id else max(self.s.now, self.last_hb + self.to())
            self.s.at(due, self.detect)

    def detect(self):
        self.timer_on = False
        if not self.node.up:
            return
        if self.leader != self.id and self.s.now - self.last_hb >= self.to() - 1e-9:
            self.leader = -1
            self.last_hb = self.s.now
            self.start_election()
        self.detect_arm()

    def lead(self):
        self.leader = self.id
        if not self.serve:
            self.serve = True
            self.node.busy = max(self.node.busy, self.s.now + self.p.Tsync)
            self.s.m.leader_up(self.gid, self.id)
        self.tick_gen += 1
        self.tick(self.tick_gen)

    def tick(self, gen):
        if gen != self.tick_gen or not self.serve or not self.node.up:
            return
        for m in self.mem:
            if m != self.id:
                self.s.send(self.id, m, "HB", (self.id,), None, AC_BE, self.gid)
        self.s.at(self.s.now + self.Thb(), self.tick, gen)

    def resign(self):
        if self.serve:
            self.serve = False
            self.s.m.leader_down(self.gid, self.id)
            self.tick_gen += 1

    def crash(self):
        self.resign()
        self.tick_gen += 1

    def recover(self):
        self.leader = -1
        self.last_hb = self.s.now
        self.start_election()
        self.detect_arm()

    def req_outcome(self, leader, ok):
        pass

    def set_urgency(self, u):
        pass

    def on_hb(self, src):
        if self.prio[src] > self.prio[self.id] or not self.serve:
            self.resign_if_lower(src)
            self.leader = src
            self.last_hb = self.s.now
        else:
            self.start_election()

    def resign_if_lower(self, src):
        if self.serve:
            self.resign()


class Bully(ClassicNode):
    """Garcia-Molina Bully algorithm with static priorities."""
    Tans = 0.1
    Tcoord = 0.3

    def __init__(self, g, nid):
        super().__init__(g, nid)
        self.in_el = False
        self.answered = False
        self.el_gen = 0

    def start_election(self):
        if self.in_el or not self.node.up:
            return
        self.in_el, self.answered = True, False
        self.el_gen += 1
        high = [m for m in self.mem if self.prio[m] > self.prio[self.id]]
        for m in high:
            self.s.send(self.id, m, "ELEC", (), None, AC_BE, self.gid)
        self.s.at(self.s.now + self.Tans, self.after_ans, self.el_gen)

    def after_ans(self, gen):
        if gen != self.el_gen or not self.in_el or not self.node.up:
            return
        if not self.answered:
            self.win()
        else:
            self.s.at(self.s.now + self.Tcoord, self.after_coord, gen)

    def after_coord(self, gen):
        if gen != self.el_gen or not self.in_el or not self.node.up:
            return
        self.in_el = False
        self.start_election()

    def win(self):
        self.in_el = False
        for m in self.mem:
            if m != self.id:
                self.s.send(self.id, m, "COORD", (), None, AC_BE, self.gid)
        self.lead()

    def on_msg(self, src, kind, msg):
        if kind == "HB":
            self.on_hb(src)
        elif kind == "ELEC":
            self.s.send(self.id, src, "ANS", (), None, AC_BE, self.gid)
            if self.serve:
                self.s.send(self.id, src, "COORD", (), None, AC_BE, self.gid)
            else:
                self.start_election()
        elif kind == "ANS":
            self.answered = True
        elif kind == "COORD":
            if self.prio[src] >= self.prio[self.id] or not self.serve:
                self.resign_if_lower(src)
                self.leader, self.in_el = src, False
                self.el_gen += 1
                self.last_hb = self.s.now
            else:
                self.start_election()


class Ring(ClassicNode):
    """Chang-Roberts election on a logical ring, with per-hop acknowledgement to skip dead successors."""
    Tack = 0.1

    def __init__(self, g, nid):
        super().__init__(g, nid)
        order = sorted(self.mem, key=lambda m: self.prio[m])
        self.ring = order
        self.pos = order.index(nid)
        self.part = False
        self.seq = 0
        self.wait = {}
        self.seen = {}   # (tag, value) -> time last relayed; retransmission must not duplicate a token

    def next_of(self, skip):
        return self.ring[(self.pos + 1 + skip) % len(self.ring)]

    def forward(self, tag, val, skip=0):
        if skip >= len(self.ring) - 1:
            return
        self.seq += 1
        self.wait[self.seq] = (tag, val, skip)
        self.s.send(self.id, self.next_of(skip), "RING", (tag, val, self.seq), None, AC_BE, self.gid)
        self.s.at(self.s.now + self.Tack, self.ack_check, self.seq)

    def ack_check(self, seq):
        rec = self.wait.pop(seq, None)
        if rec is not None and self.node.up:
            self.forward(rec[0], rec[1], rec[2] + 1)

    def start_election(self):
        if self.part or not self.node.up:
            return
        self.part = True
        self.s.at(self.s.now + 3.0, self.reset_part)
        self.forward("E", self.id)

    def reset_part(self):
        self.part = False

    def on_msg(self, src, kind, msg):
        if kind == "HB":
            self.on_hb(src)
        elif kind == "RACK":
            self.wait.pop(msg[0], None)
        elif kind == "RING":
            tag, val, seq = msg
            self.s.send(self.id, src, "RACK", (seq,), None, AC_BE, self.gid)
            now = self.s.now
            if now - self.seen.get((tag, val), -INF) < 1.0:
                return
            self.seen[(tag, val)] = now
            if tag == "E":
                if self.prio[val] > self.prio[self.id]:
                    self.part = True
                    self.forward("E", val)
                elif val == self.id:
                    self.part = False
                    self.lead()
                    self.forward("C", self.id)
                elif not self.part:
                    self.part = True
                    self.forward("E", self.id)
            else:
                if val != self.id:
                    self.resign_if_lower(val)
                    self.leader, self.part = val, False
                    self.last_hb = self.s.now
                    self.forward("C", val)


CLASSES = {"eagle": Eagle, "raft": Raft, "resraft": ResRaft, "trustraft": TrustRaft, "bully": Bully, "ring": Ring}
