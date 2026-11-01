"""Discrete-event kernel, nodes, link model and metrics for the EAGLE-Med evaluation.

Time is in seconds. All randomness comes from named streams derived from one integer seed, so
that competing protocols see identical scenario draws (common random numbers).
"""
import bisect
import heapq
import itertools
import json
import math
import os
import random

AC_BE, AC_VO = 0, 1  # 802.11 access categories used by the control and alert planes

# wire sizes in bytes (payload plus UDP/IP headers); evidence digests are size-dependent
SIZE = {"HB": 96, "ACK": 72, "PV": 96, "PVR": 72, "RV": 96, "RVR": 72, "XFER": 72, "REQ": 256, "RESP": 256,
        "ALERT": 192, "AACK": 72, "URG": 96, "ELEC": 72, "ANS": 72, "COORD": 72, "RING": 96, "RACK": 64,
        "PING": 96, "PONG": 96, "SD": 72}
ELECTION_KINDS = {"PV", "PVR", "RV", "RVR", "XFER", "SD", "ELEC", "ANS", "COORD", "RING", "RACK"}
WORKLOAD_KINDS = {"REQ", "RESP", "ALERT", "AACK"}


class Node:
    __slots__ = ("id", "kind", "ward", "cap", "load", "battery", "mains", "up", "claim_x", "mal", "drop_q",
                 "tx_msgs", "rx_msgs", "tx_bytes", "rx_bytes", "busy", "leads", "home")

    def __init__(self, nid, kind, ward, cap, load, battery, mains):
        self.id, self.kind, self.ward = nid, kind, ward
        self.cap, self.load, self.battery, self.mains = cap, load, battery, mains
        self.up = True
        self.claim_x = 1.0  # claim inflation applied by a compromised node
        self.mal = False    # compromised and misbehaving when leading
        self.drop_q = 0.0
        self.tx_msgs = self.rx_msgs = self.tx_bytes = self.rx_bytes = 0
        self.busy = 0.0     # time at which the coordination queue drains
        self.leads = 0      # leaderships currently held
        self.home = -1

    def mu(self):
        """Service rate left by background load, Eq. (2). Leaderships share one queue at this rate."""
        return self.cap * max(0.02, 1.0 - self.load)


class LinkModel:
    """Trace-driven link model: one-way delay and loss resampled from ns-3 quantile tables."""

    def __init__(self, table, rng, base_load):
        self.t = table
        self.loads = sorted(float(k) for k in next(iter(table.values())).keys())
        self.rng = rng
        self.base = base_load
        self.ew_frames = 0.0  # frames per second per ward, smoothed
        self.win_count = 0
        self.win_t = 0.0
        self.bad = {}         # node id -> (loss, jitter factor) during instability

    def equivalent_load(self):
        # frames occupy airtime: 250 us per small frame on a BSS of about 110 Mb/s of usable goodput
        return self.base + 110.0 * self.ew_frames * 250e-6

    def note_frame(self, now, wards):
        self.win_count += 1
        if now - self.win_t >= 1.0:
            rate = self.win_count / max(1e-6, now - self.win_t) / max(1, wards)
            self.ew_frames = 0.7 * self.ew_frames + 0.3 * rate
            self.win_t, self.win_count = now, 0

    @staticmethod
    def _at(rows, load):
        return rows[str(load)] if str(load) in rows else rows[f"{load:.1f}"]

    def _rows(self, cls, ac, load):
        """Calibration rows that bracket the load, and the interpolation weight of the upper one."""
        rows = self.t[f"{cls}|{ac}"]
        ls = self.loads
        i = min(bisect.bisect_right(ls, load), len(ls) - 1)
        lo, hi = ls[max(i - 1, 0)], ls[i]
        w = 0.0 if hi == lo else min(1.0, (load - lo) / (hi - lo))
        return self._at(rows, lo), self._at(rows, hi), w

    def sample(self, cls, ac, s, d):
        """Return (delivered, delay)."""
        lo, hi, w = self._rows(cls, ac, self.equivalent_load())
        r = self.rng.random
        loss = (1.0 - w) * lo["loss"] + w * hi["loss"]
        bs, bd = self.bad.get(s), self.bad.get(d)
        jit = 1.0
        for b in (bs, bd):
            if b is not None:
                loss = 1.0 - (1.0 - loss) * (1.0 - b[0])
                jit = max(jit, b[1])
        if r() < loss:
            return False, 0.0
        u = r() * 100.0
        i = min(int(u), 99)
        f = u - i
        qa, qb = lo["q"], hi["q"]
        return True, ((1.0 - w) * (qa[i] + f * (qa[i + 1] - qa[i])) + w * (qb[i] + f * (qb[i + 1] - qb[i]))) * jit


def synthetic_table():
    """Fallback table used only by unit tests when the ns-3 calibration file is absent."""
    tab = {}
    for cls, base in (("WW_same", 2.2e-3), ("WW_cross", 2.2e-3), ("WS", 1.2e-3), ("SW", 1.2e-3), ("SS", 1.5e-4)):
        for ac in (0, 1):
            rows = {}
            for load in (0.0, 5.0, 20.0, 50.0, 90.0, 140.0):
                scale = 1.0 + (load / 40.0) ** 2 * (0.3 if ac else 1.0)
                sig = 0.6
                q = []
                for p in range(101):
                    z = math.sqrt(2) * _erfinv(2 * max(min(p / 100.0, 0.999), 0.001) - 1)
                    q.append(base * scale * math.exp(sig * z))
                rows[str(load)] = {"loss": 0.002 + (load / 400.0) ** 2, "q": q}
            tab[f"{cls}|{ac}"] = rows
    return tab


def _erfinv(x):
    a = 0.147
    l = math.log(1 - x * x)
    t = 2 / (math.pi * a) + l / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - l / a) - t), x)


def load_table(path):
    if path and os.path.exists(path):
        return json.load(open(path))
    return synthetic_table()


class Metrics:
    def __init__(self, sim):
        self.sim = sim
        self.t0 = 0.0              # start of the measurement window; traffic and changes before it are not counted
        self.serving = {}          # (group, node) -> start time of current serving period
        self.serve_events = []     # (t, group, node, +1/-1)
        self.first_leader = {}     # group -> t
        self.prev_leader = {}      # group -> node
        self.changes = 0
        self.tenures = []          # (end time, duration) of finished serving periods
        self.crash_t = []          # (t, group, node)
        self.reelect = []          # re-election latencies after leader loss
        self.recover = []          # service recovery after leader crash
        self.pending_recover = {}
        self.pending_reelect = {}
        self.handover = []         # planned hand-over gaps
        self.attempts = 0
        self.wins = 0
        self.req_total = self.req_hit = self.req_lost = 0
        self.req_em_total = self.req_em_hit = 0
        self.req_lat = []
        self.alerts = {}           # alert id -> birth, first arrival, generated in emergency
        self.elec_msgs = 0
        self.ctrl_msgs = 0
        self.ctrl_bytes = 0
        self.alert_msgs = 0
        self.mal_serving = 0.0
        self.eval_ops = 0

    # leadership timeline -------------------------------------------------------------------
    def leader_up(self, g, n):
        now = self.sim.now
        if (g, n) in self.serving:
            return
        self.serving[(g, n)] = now
        self.serve_events.append((now, g, n, +1))
        self.sim.nodes[n].leads += 1
        if g not in self.first_leader:
            self.first_leader[g] = now
        if g in self.prev_leader and self.prev_leader[g] != n:
            self.changes += now >= self.t0
        self.prev_leader[g] = n
        if g in self.pending_reelect:
            self.reelect.append(now - self.pending_reelect.pop(g))

    def leader_down(self, g, n):
        now = self.sim.now
        t0 = self.serving.pop((g, n), None)
        if t0 is None:
            return
        self.serve_events.append((now, g, n, -1))
        self.sim.nodes[n].leads = max(0, self.sim.nodes[n].leads - 1)
        self.tenures.append((now, now - t0))

    def note_leader_lost(self, g, n):
        """A serving leader has become unusable (crash); start the failover clocks."""
        now = self.sim.now
        self.crash_t.append((now, g, n))
        self.pending_reelect.setdefault(g, now)
        self.pending_recover.setdefault(g, now)

    def note_req_ok(self, g, t_issue):
        """The first request issued after the crash and answered in time stops the recovery clock."""
        t = self.pending_recover.get(g)
        if t is not None and t_issue >= t:
            self.recover.append(self.sim.now - self.pending_recover.pop(g))

    # summary -------------------------------------------------------------------------------
    def availability(self, t0, t1, groups):
        """Fraction of [t0, t1] during which a group has a serving, live leader; and dual-leader time."""
        res, dual = [], 0.0
        for g in groups:
            ev = sorted(e for e in self.serve_events if e[1] == g)
            cnt, last, cov, dl = 0, t0, 0.0, 0.0
            for t, _, _, s in ev:
                t = min(max(t, t0), t1)
                if cnt >= 1:
                    cov += t - last
                if cnt >= 2:
                    dl += t - last
                last, cnt = t, cnt + s
            if cnt >= 1:
                cov += t1 - last
            if cnt >= 2:
                dl += t1 - last
            res.append(cov / max(1e-9, t1 - t0))
            dual += dl
        return sum(res) / max(1, len(res)), dual

    def vacancies(self, t0, groups, limit):
        """Leaderless intervals that open at or after t0: their number and how many close within limit."""
        total = ok = 0
        for g in groups:
            cnt, opened = 0, None
            for t, _, _, s in sorted(e for e in self.serve_events if e[1] == g):
                cnt += s
                if cnt == 0 and t >= t0:
                    total, opened = total + 1, t
                elif cnt > 0 and opened is not None:
                    ok += t - opened <= limit
                    opened = None
        return total, ok


class Sim:
    def __init__(self, seed, table, base_load):
        self.now = 0.0
        self.q = []
        self.c = itertools.count()
        self.seed = seed
        root = random.Random(seed)
        # named streams: scenario draws never depend on protocol behaviour
        self.rng_scn = random.Random(root.getrandbits(64))
        self.rng_load = random.Random(root.getrandbits(64))
        self.rng_net = random.Random(root.getrandbits(64))
        self.rng_prot = random.Random(root.getrandbits(64))
        self.rng_svc = random.Random(root.getrandbits(64))
        self.link = LinkModel(table, self.rng_net, base_load)
        self.nodes = []
        self.groups = []
        self.m = Metrics(self)
        self.partition = None      # dict node -> side, while a partition is active
        self.cut = {}              # node -> disconnected-until time
        self.wards = 4
        self.deliver_hook = None

    def at(self, t, fn, *a):
        heapq.heappush(self.q, (t, next(self.c), fn, a))

    def run(self, t_end):
        q, pop = self.q, heapq.heappop
        while q:
            t, _, fn, a = pop(q)
            if t > t_end:
                break
            self.now = t
            fn(*a)
        self.now = t_end

    def link_class(self, s, d):
        a, b = self.nodes[s], self.nodes[d]
        if a.kind == "G":
            if b.kind == "G":
                return "WW_same" if a.ward == b.ward else "WW_cross"
            return "WS"
        return "SW" if b.kind == "G" else "SS"

    def send(self, s, d, kind, msg, size=None, ac=AC_BE, grp=None):
        a = self.nodes[s]
        if not a.up:
            return
        sz = size if size is not None else SIZE[kind]
        a.tx_msgs += 1
        a.tx_bytes += sz
        m = self.m
        if self.now >= m.t0:
            if kind in ELECTION_KINDS:
                m.elec_msgs += 1
            if kind in ("ALERT", "AACK"):
                m.alert_msgs += 1
            elif kind not in WORKLOAD_KINDS:
                m.ctrl_msgs += 1
                m.ctrl_bytes += sz
        self.link.note_frame(self.now, self.wards)
        if self.partition is not None and self.partition.get(s, 0) != self.partition.get(d, 0):
            return
        if self.cut:
            now = self.now
            if self.cut.get(s, -1) > now or self.cut.get(d, -1) > now:
                return
        ok, dl = self.link.sample(self.link_class(s, d), ac, s, d)
        if ok:
            self.at(self.now + dl, self.deliver, s, d, kind, msg, sz, grp)

    def deliver(self, s, d, kind, msg, sz, grp):
        b = self.nodes[d]
        if not b.up:
            return
        b.rx_msgs += 1
        b.rx_bytes += sz
        if grp is None:
            self.deliver_hook(s, d, kind, msg)
        else:
            self.groups[grp].inst[d].on_msg(s, kind, msg)
