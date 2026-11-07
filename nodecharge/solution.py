"""Solution container, evaluation and feasibility checker."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .instance import Instance, NodeCharge


@dataclass
class Solution:
    """Rebalancing decision for one interval.

    Attributes
    ----------
    routes : for each vehicle (same order as ``Instance.vehicles``), the
        sequence of node-charges it traverses, from its origin to its target.
        A route of length 1 means the vehicle stays put.
    assignment : ``(i, g) -> (j, h)``: demand node-charge (i, g) is covered by
        the vehicle(s) at node-charge (j, h) (the X_igjh variables).
    access_cost : sum of lambda_ig * tau_ij * X_igjh (first term of Eq. 3).
    rebalancing_cost : sum of c * W over arcs (second term of Eq. 3, before theta).
    objective : access_cost + theta * rebalancing_cost.
    """

    routes: List[List[NodeCharge]]
    assignment: Dict[NodeCharge, NodeCharge]
    access_cost: float
    rebalancing_cost: float
    objective: float
    method: str = ""
    status: str = ""
    runtime: float = 0.0
    queueing_enforced: bool = True
    info: dict = field(default_factory=dict)

    @property
    def destinations(self) -> List[NodeCharge]:
        return [r[-1] for r in self.routes]

    def server_counts(self, inst: Instance) -> np.ndarray:
        n = np.zeros((inst.n_zones, inst.n_levels), dtype=int)
        for j, h in self.destinations:
            n[j, h] += 1
        return n

    def charging_station(self, f: int) -> Optional[int]:
        r = self.routes[f]
        for a, c in zip(r[:-1], r[1:]):
            if a[0] == c[0] and c[1] == a[1] + 1:
                return a[0]
        return None

    def arc_flows(self) -> Counter:
        """W: number of vehicles on each arc ``(tail, head)``."""
        w: Counter = Counter()
        for r in self.routes:
            for a, c in zip(r[:-1], r[1:]):
                w[(a, c)] += 1
        return w

    def summary(self, inst: Instance) -> str:
        lines = [
            f"[{self.method}] status={self.status}  objective={self.objective:.4f}  "
            f"(access={self.access_cost:.4f}, rebalancing={self.rebalancing_cost:.3f} "
            f"x theta={inst.theta})  runtime={self.runtime:.3f}s"
        ]
        for f, r in enumerate(self.routes):
            st = self.charging_station(f)
            o, d = r[0], r[-1]
            move = "stays" if len(r) == 1 else f"-> zone {d[0]}, level {d[1]}"
            chg = f" (charges at station {st}: level {o[1]} -> {d[1]})" if st is not None else ""
            lines.append(f"  vehicle {f}: zone {o[0]}, level {o[1]} {move}{chg}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
def station_paths_needed(inst: Instance, sol: Solution) -> Dict[int, int]:
    """Minimum number of charging paths through each station implied by W.

    Constraint (13)-(14) decompose the charging-arc flows of a station into
    paths (enter at g, exit at h > g). The minimum number of such paths for
    segment flows w_0, w_1, ... is sum(max(0, w_g - w_{g-1})).
    """
    w = sol.arc_flows()
    need = {}
    for j in inst.station_list:
        seg = [w.get(((j, g), (j, g + 1)), 0) for g in range(inst.n_levels - 1)]
        prev, cnt = 0, 0
        for s in seg:
            cnt += max(0, s - prev)
            prev = s
        need[j] = cnt
    return need


def evaluate(inst: Instance, routes, assignment) -> Tuple[float, float, float]:
    """Return (access_cost, rebalancing_cost, objective) of a decision."""
    tau, lam = inst.tau, inst.lam
    access = sum(lam[i, g] * tau[i, j] for (i, g), (j, _h) in assignment.items())
    cost = {(a, c): w for a, c, w, _ in inst.arcs}
    reb = 0.0
    for r in routes:
        for a, c in zip(r[:-1], r[1:]):
            reb += cost[(a, c)]
    return float(access), float(reb), float(access + inst.theta * reb)


def check_solution(
    inst: Instance, sol: Solution, queueing: Optional[bool] = None, tol: float = 1e-6
) -> List[str]:
    """Verify a solution against constraints (4)-(18). Returns a list of violations.

    ``queueing`` defaults to ``sol.queueing_enforced``; set False to skip
    Constraint (7) (myopic policy).
    """
    if queueing is None:
        queueing = sol.queueing_enforced
    errs: List[str] = []
    N, H = inst.n_zones, inst.n_levels
    arcset = {(a, c) for a, c, _, _ in inst.arcs}

    # Routes: start at the vehicle origin, follow arcs of the node-charge graph.
    if len(sol.routes) != inst.B:
        errs.append(f"expected {inst.B} routes, got {len(sol.routes)}")
    for f, (r, o) in enumerate(zip(sol.routes, inst.vehicles)):
        if tuple(r[0]) != tuple(o):
            errs.append(f"vehicle {f}: route starts at {r[0]} but vehicle is at {o}")
        for a, c in zip(r[:-1], r[1:]):
            if (tuple(a), tuple(c)) not in arcset:
                errs.append(f"vehicle {f}: {a}->{c} is not an arc of the node-charge graph")

    # (8), (16): B servers in total, at most C per node-charge.
    n = sol.server_counts(inst)
    if n.sum() != inst.B:
        errs.append("(8) number of servers differs from number of idle vehicles")
    if n.max(initial=0) > inst.C:
        errs.append(f"(16) more than C={inst.C} vehicles at a node-charge")

    # (4), (5), (9): every demand node-charge covered by a server at a level >= g.
    load = np.zeros((N, H))
    for i in range(N):
        for g in range(H):
            if inst.lam[i, g] <= 0:
                continue
            s = sol.assignment.get((i, g))
            if s is None:
                errs.append(f"(4) demand {(i, g)} is not assigned")
                continue
            j, h = s
            if h < g:
                errs.append(f"(5) demand {(i, g)} assigned to lower charge level {s}")
            if n[j, h] == 0:
                errs.append(f"(9) demand {(i, g)} assigned to {s} which has no vehicle")
            load[j, h] += inst.lam[i, g]

    # (7): queueing intensity.
    if queueing:
        cap = inst.mu * inst.rho[np.minimum(n, len(inst.rho) - 1)]
        bad = np.argwhere(load > cap + tol)
        for j, h in bad:
            errs.append(
                f"(7) node-charge {(int(j), int(h))}: load {load[j, h]:.4f} > "
                f"mu*rho_{n[j, h]} = {cap[j, h]:.4f}"
            )

    # (13)-(14): charging-station path capacity.
    for j, need in station_paths_needed(inst, sol).items():
        if need > inst.stations[j]:
            errs.append(f"(14) station {j}: {need} charging paths > capacity {inst.stations[j]}")

    # Objective consistency.
    acc, reb, obj = evaluate(inst, sol.routes, sol.assignment)
    if abs(obj - sol.objective) > 1e-6 * max(1.0, abs(obj)):
        errs.append(f"reported objective {sol.objective} != recomputed {obj}")
    return errs
