"""Control arms added after campaign v3 was read. Tagged post:* and kept in post_runs.csv, apart from the frozen
campaign. Each arm is a subclass, so the frozen sources stay byte-identical."""
import numpy as np

from .params import VARIANTS, Profile
from .protocols import CLASSES, Eagle
from .scenarios import T0, Harness


class Kappa0Live(Harness):
    """kappa = 0 at any urgency. The nodes hold their own reference to the profile, so it is replaced on them as well;
    the arm of the same name in arms.py changed only the harness copy and had no effect."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        prof = Profile(kappa=(0.0, 0.0))
        self.p.profile = prof
        for g in self.sim.groups:
            for inst in g.inst.values():
                inst.P = prof


class EagleFloor(Eagle):
    """A new suspension may not leave fewer than n - Q + 1 unsuspended nodes, so one of them is up whenever a quorum
    is. Candidates are spared in order of direct evidence, which keeps the rule local to each node."""

    def assess(self, psi):
        T, mu, Phi, Elig = super().assess(psi)
        now = self.s.now
        free = (now >= self.quar) & ~self.trip
        need = self.n - self.Q + 1 - int(free.sum())
        if need > 0:
            due = np.nonzero(self.trip & (now >= self.quar))[0]
            spared = due[np.argsort(-T[due], kind="stable")][:need]
            self.trip = self.trip.copy()
            self.trip[spared] = False
        return T, mu, Phi, Elig


class HoldHb(Harness):
    """The heartbeat stays at the normal period during an emergency, so the lease stays at 0.38 s."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        prof = Profile(Thb=(0.2, 0.2))
        self.p.profile = prof
        for g in self.sim.groups:
            for inst in g.inst.values():
                inst.P = prof


class HeldTrace(Harness):
    """Records the largest median number of nodes suspended in a node's view. Draws no random number."""

    def snap(self):
        g = self.sim.groups[0]
        held = [int((self.sim.now < i.quar).sum()) for m, i in g.inst.items() if self.sim.nodes[m].up]
        self.held_peak = max(self.held_peak, int(np.median(held)))
        if self.sim.now < self.t_end - 2.0:
            self.sim.at(self.sim.now + 2.0, self.snap)

    def run(self):
        self.held_peak = 0
        self.sim.at(T0 + 2.0, self.snap)
        r = super().run()
        r["held_peak"] = self.held_peak
        return r


class EagleNoSilence(Eagle):
    """An unanswered request is never selective silence: it counts as availability evidence, as for a silent leader."""

    def req_outcome(self, leader, ok):
        if ok or leader not in self.g.ix:
            super().req_outcome(leader, ok)
        else:
            self.Bd[1, self.g.ix[leader]] += 1.0


ARMS_POST = {"kappa0live": Kappa0Live, "holdhb": HoldHb, "heldtrace": HeldTrace}


def register_post():
    CLASSES["eagle_floor"] = EagleFloor
    VARIANTS["EAGLE-floor"] = {"family": "eagle_floor", "alert_plane": True}
    CLASSES["eagle_nosil"] = EagleNoSilence
    VARIANTS["EAGLE-nosil"] = {"family": "eagle_nosil", "alert_plane": True}
