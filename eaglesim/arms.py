"""Control arms of the campaign. Each one changes a single assumption of a main scenario, so that a result can be
checked against that assumption. Tagged supp:* in the raw runs and kept out of the main statistics."""
import math

from .params import VARIANTS, Profile
from .protocols import CLASSES, ResRaft
from .scenarios import Harness


class ResRaftSeeded(ResRaft):
    """ResRaft with nominal capacities configured before the first election."""

    def start(self):
        super().start()
        for m, x in self.g.ix.items():
            self.claim[x] = self.s.nodes[m].cap
        self.heard[:] = self.s.now
        self.evaluate()


class FalseLabel(Harness):
    """Nodes receive u=2 for 60 s, as in S2, while the workload, the servers and the gateways stay as in S1."""

    def schedule(self):
        super().schedule()
        te = 90.0 + self.sim.rng_scn.uniform(0, 5)
        g = self.sim.groups[0]
        self.sim.at(te, self.label, g, 2)
        self.sim.at(te + 60.0, self.label, g, 0)

    def label(self, g, u):
        sim = self.sim
        for m in g.mem:
            if sim.nodes[m].up:
                sim.at(sim.now + 0.002 + 0.008 * sim.rng_scn.random(), g.inst[m].set_urgency, u)


class WrongRate(Harness):
    """The true emergency rate differs from the anchor that the protocol ranks with."""
    factor = 1.0

    def rate_now(self, g):
        r = super().rate_now(g)
        return r * self.factor if self.u[g.gid] > 0 else r


class Rate13(WrongRate):
    factor = 1.3


class Rate07(WrongRate):
    factor = 0.7


class Lognormal(Harness):
    """Service times keep their mean but have a coefficient of variation of 2, so Eq. (13) is no longer exact."""
    SIGMA2 = math.log(5.0)

    def at_leader(self, src, l, rid, g):
        sim = self.sim
        nd = sim.nodes[l]
        inst = g.inst[l]
        if not inst.is_serving():
            return
        if nd.mal and sim.rng_svc.random() < nd.drop_q:
            return
        start = max(sim.now, nd.busy)
        mean = 1.0 / max(1e-3, nd.mu())
        nd.busy = start + sim.rng_svc.lognormvariate(math.log(mean) - self.SIGMA2 / 2, math.sqrt(self.SIGMA2))
        if src == l:
            sim.at(nd.busy, self.req_done, rid)
        else:
            sim.at(nd.busy, sim.send, l, src, "RESP", (rid,), None, 0, None)


class SwapTop(Harness):
    """Static priority with the second server on top."""

    def priorities(self):
        prio = super().priorities()
        prio[0], prio[1] = prio[1], prio[0]
        return prio


class Kappa0(Harness):
    """No pessimism in the trust bound, at any urgency."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.p.profile = Profile(kappa=(0.0, 0.0))


ARMS = {"plain": Harness, "falselabel": FalseLabel, "rate1.3": Rate13, "rate0.7": Rate07, "lognormal": Lognormal, "swaptop": SwapTop, "kappa0": Kappa0}


def register():
    CLASSES["resraft_seeded"] = ResRaftSeeded
    VARIANTS["ResRaft-seeded"] = {"family": "resraft_seeded"}
