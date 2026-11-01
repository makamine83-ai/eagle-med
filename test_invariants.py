"""Invariant checks on running protocols. These test the implementation, they do not prove the protocol.

I1  at most one leader per term in each group (Proposition 1)
I2  no two serving leaders at once for quorum protocols (Proposition 2, under the simulator's perfect clocks)
I3  alert send times do not depend on the leader: the same seed with and without a leader crash emits the
    alerts of the plane at identical times (Proposition 4)
I4  voluntary switches respect the dwell bound of Proposition 3
"""
import sys
from collections import defaultdict

from eaglesim.core import load_table
from eaglesim.protocols import Eagle, QuorumNode
from eaglesim.scenarios import Harness

TABLE = load_table(sys.argv[1] if len(sys.argv) > 1 else None)


def check_terms(scn, var, seed):
    h = Harness(scn, var, seed, TABLE)
    won = defaultdict(set)
    orig = QuorumNode.become_leader

    def wrapped(self):
        won[(self.gid, self.term)].add(self.id)
        orig(self)

    QuorumNode.become_leader = wrapped
    try:
        r = h.run()
    finally:
        QuorumNode.become_leader = orig
    bad = {k: v for k, v in won.items() if len(v) > 1}
    assert not bad, (scn, var, seed, bad)
    assert r["dual_leader_s"] == 0.0, (scn, var, seed, r["dual_leader_s"])
    return len(won)


def alert_sends(scenario, seed):
    h = Harness(scenario, "EAGLE", seed, TABLE)
    log = []
    orig = h.alert_try

    def tap(aid, src, g, att):
        if att == 0:
            log.append((round(h.sim.now, 9), src))
        orig(aid, src, g, att)

    h.alert_try = tap
    h.run()
    return log


def check_alert_independence(seed):
    # S2 and S3 share every draw except the crash, which S3 adds after the emergency starts
    a, b = alert_sends("S2", seed), alert_sends("S3", seed)
    # the crash removes one alert source if the leader is a gateway; every other send time must be identical
    sa, sb = set(a), set(b)
    assert len({src for _, src in sa ^ sb}) <= 1, (len(sa), len(sb), len(sa & sb))
    return len(sa & sb), len(sa)


def check_dwell(scn, seed):
    """A live leader is displaced only after T_pb, and before T_dw only if a quorum finds it ineligible (Proposition 3)."""
    h = Harness(scn, "EAGLE", seed, TABLE)
    p = h.sim.groups[0].p
    last = {}
    early = []
    gaps = []
    orig = Eagle.become_leader

    def wrapped(self):
        now, prev = self.s.now, last.get(self.gid)
        if prev and prev[0] != self.id and self.pv_reason == "risk" and self.s.nodes[prev[0]].up:
            gap = now - prev[1]
            gaps.append(gap)
            assert gap >= p.Tpb, (scn, seed, gap)
            if gap < p.Tdw:
                lx = self.g.ix[prev[0]]
                views = sum(not self.g.inst[m].Elig[lx] for m in self.mem if self.s.nodes[m].up)
                early.append((round(gap, 2), views))
                assert views >= self.Q, (scn, seed, gap, views, self.Q)
        last[self.gid] = (self.id, now)
        orig(self)

    Eagle.become_leader = wrapped
    try:
        h.run()
    finally:
        Eagle.become_leader = orig
    return len(gaps), round(min(gaps), 2) if gaps else None, early


if __name__ == "__main__":
    for scn in ("S3", "S5", "S6"):
        for var in ("EAGLE", "Raft", "TrustRaft"):
            for seed in (11, 12):
                print("terms", scn, var, seed, check_terms(scn, var, seed), flush=True)
    for seed in (11, 12):
        print("alert independence", seed, check_alert_independence(seed), flush=True)
    for scn in ("S1", "S2", "S5"):
        for seed in (11, 12):
            print("dwell", scn, seed, check_dwell(scn, seed), flush=True)
    print("ALL INVARIANTS HELD")
