"""Protocol profile P(psi) and protocol variants.

Every urgency-dependent parameter is interpolated between a normal anchor (psi = 0) and an
emergency anchor (psi = 1), Eq. (6). These anchors are protocol operating points chosen for the
simulated deployment; they are not clinical thresholds.
"""
import math
from dataclasses import dataclass, field


@dataclass
class Profile:
    # anchors: (normal, emergency)
    D: tuple = (0.50, 0.15)        # coordination deadline, s
    lam: tuple = (20.0, 45.0)      # design request rate of a group, req/s
    H: tuple = (120.0, 60.0)       # planning horizon, s
    kappa: tuple = (1.0, 2.0)      # pessimism of the trust lower bound
    Tmin: tuple = (0.30, 0.50)     # minimum trust lower bound
    Amin: tuple = (0.70, 0.85)     # minimum availability estimate
    Thb: tuple = (0.20, 0.10)      # heartbeat period, s
    # loss weights of (trust, availability, deadline-hit probability, energy reserve) in the risk of Eq. (17)
    wN: tuple = (3.0, 1.0, 1.0, 0.5)
    wE: tuple = (4.0, 1.0, 2.0, 0.2)

    def at(self, name, psi):
        a, b = getattr(self, name)
        return a + psi * (b - a)


@dataclass
class EagleParams:
    profile: Profile = field(default_factory=Profile)
    K: int = 3                # missed heartbeats before suspicion
    Tr: float = 0.10          # rank spacing of campaign delays, s
    Tj: float = 0.05          # campaign jitter, s
    Tg: float = 1.0           # evidence gossip period, s
    gossip_fanout: int = 3
    gamma: float = 0.98       # evidence forgetting per second (communication, availability)
    gamma_c: float = 0.92     # evidence forgetting per second (compliance)
    neg: float = 3.0          # weight of a failed coordination outcome relative to a success
    Tq: float = 120.0         # base suspension after a failed compliance gate, s
    Nmin: float = 3.0         # compliance evidence mass needed before that class counts
    eta: float = 3.0          # anomaly penalty, evidence units per second
    omega: float = 0.5        # weight of pooled second-hand evidence
    xi: float = 1.0           # tolerated gap between claimed and measured capacity (noisy probes)
    Tresp: float = 1.0        # a request unanswered after this time is a compliance failure, s
    Tpb: float = 3.0          # probation of a new leader before ineligibility can remove it, s
    Trt: float = 5.0          # least interval between preemptive attempts of one node, s
    tau_u: float = 30.0       # urgency release time constant, s
    alpha: float = 0.2        # EWMA gain of delay and availability estimates
    eps: float = 0.10         # risk estimation error bound, the p99 of the stable-network deviation rounded up
    drift_margin: float = 0.02  # lease safety margin for clock drift and processing, s
    beta: float = 1.0         # extra weight of disruption during emergencies
    Th: float = 0.25          # expected hand-over disruption, s
    cm: float = 2.0           # message cost expressed in requests
    Tdw: float = 10.0         # minimum dwell between voluntary switches, s
    Tsync: float = 0.05       # state synchronisation before a new leader answers requests, s
    eps_v: float = 0.20       # vote tolerance on stale scores, in risk units
    f_trim: int = 2           # trim count of the pooled-evidence mean
    relax_after: int = 3      # failed campaigns before the vote filter is relaxed
    Tr_alert: float = 0.20    # alert retransmission period, s
    alert_ttl: float = 10.0   # alert lifetime, s
    # variant switches
    use_urgency: bool = True
    use_trust: bool = True
    use_hysteresis: bool = True
    use_alert_plane: bool = True
    use_adaptive_w: bool = True


def margin(p: EagleParams, psi):
    """Stability margin, Eq. (15): the noise guard plus the transition cost per planning horizon."""
    lam = p.profile.at("lam", psi)
    H = p.profile.at("H", psi)
    return 2 * p.eps + ((1 + p.beta * psi) * p.Th + p.cm / lam) / H


# Protocol registry ------------------------------------------------------------------------------
# family: quorum (Raft mechanics), bully or ring. All variants share the failure detector
# parameters K and heartbeat period so that detection is not a source of artificial differences.
VARIANTS = {
    "EAGLE":     dict(family="eagle"),
    "EAGLE-noU": dict(family="eagle", use_urgency=False),
    "EAGLE-noT": dict(family="eagle", use_trust=False),
    "EAGLE-noS": dict(family="eagle", use_hysteresis=False),
    "EAGLE-noA": dict(family="eagle", use_alert_plane=False),
    "EAGLE-noW": dict(family="eagle", use_adaptive_w=False),
    "Raft":      dict(family="raft"),
    "Raft+AP":   dict(family="raft", alert_plane=True),
    "Raft-100":  dict(family="raft", fast_hb=0.10),
    "ResRaft":   dict(family="resraft"),
    "TrustRaft": dict(family="trustraft"),
    "Bully":     dict(family="bully"),
    "Ring":      dict(family="ring"),
}
BASELINES = ["Raft", "ResRaft", "TrustRaft", "Bully", "Ring"]
ABLATIONS = ["EAGLE", "EAGLE-noU", "EAGLE-noT", "EAGLE-noS", "EAGLE-noA", "EAGLE-noW"]
