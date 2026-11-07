"""Exact route-capacitated minimum-cost-flow relocation model (Eqs. 3-18).

Solved with SciPy's interface to the open-source HiGHS MILP solver, so no
commercial solver is needed (the paper used MATLAB ``intlinprog``).

Notes on the formulation
------------------------
* Coverage constraints (4) are written for every demand node-charge with
  ``lambda_ig > 0``; a node-charge with no demand contributes nothing to the
  objective or to (7), so it is not forced to be covered.
* Constraints (10)-(11) are implied by flow conservation (12) together with
  (6) and (16), so they are not added explicitly.
* Arc flows W are declared integer (the paper relies on total unimodularity;
  declaring them integer guarantees integral routes even with the side
  constraints (13)-(14)).
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .instance import Instance, NodeCharge
from .solution import Solution, evaluate


class _Rows:
    """Tiny helper to assemble a sparse constraint matrix row by row."""

    def __init__(self) -> None:
        self.r: List[int] = []
        self.c: List[int] = []
        self.v: List[float] = []
        self.lb: List[float] = []
        self.ub: List[float] = []

    def add(self, cols, vals, lb, ub) -> None:
        k = len(self.lb)
        self.r.extend([k] * len(cols))
        self.c.extend(cols)
        self.v.extend(vals)
        self.lb.append(lb)
        self.ub.append(ub)

    def build(self, n_vars: int) -> LinearConstraint:
        A = coo_matrix((self.v, (self.r, self.c)), shape=(len(self.lb), n_vars)).tocsr()
        return LinearConstraint(A, np.array(self.lb), np.array(self.ub))


def solve_milp(
    inst: Instance,
    queueing: bool = True,
    time_limit: Optional[float] = None,
    mip_rel_gap: float = 1e-6,
    relax_Y: bool = False,
    verbose: bool = False,
) -> Solution:
    """Solve the cost function approximation exactly.

    Parameters
    ----------
    queueing : enforce the queueing constraint (7). ``False`` gives the myopic
        model used as a benchmark in Section 5.
    time_limit : seconds passed to HiGHS.
    relax_Y : treat the server variables Y as continuous in [0, 1]
        (Sayarshad and Chow, 2017), which is usually much faster.
    """
    t0 = time.perf_counter()
    N, H, C = inst.n_zones, inst.n_levels, inst.C
    lam, mu, tau, rho = inst.lam, inst.mu, inst.tau, inst.rho
    y0 = inst.initial_counts

    # ---------------------------------------------------------------- indices
    nc = [(i, h) for i in range(N) for h in range(H)]
    demands = [(i, g) for (i, g) in nc if lam[i, g] > 0]
    X_idx: Dict[tuple, int] = {}
    k = 0
    for (i, g) in demands:
        for j in range(N):
            for h in range(g, H):  # (5): only levels h >= g can serve g
                X_idx[(i, g, j, h)] = k
                k += 1
    Y_idx = {}
    for (j, h) in nc:
        for m in range(1, C + 1):
            Y_idx[(j, h, m)] = k
            k += 1
    arcs = inst.arcs
    W0 = k
    k += len(arcs)
    P_idx = {}
    for j in inst.station_list:
        if inst.stations[j] <= 0:
            continue
        for g in range(H):
            for h in range(g + 1, H):
                P_idx[(j, g, h)] = k
                k += 1
    n_var = k

    # -------------------------------------------------------------- objective
    cvec = np.zeros(n_var)
    for (i, g, j, h), col in X_idx.items():
        cvec[col] = lam[i, g] * tau[i, j]
    for a, (_, _, cost, _) in enumerate(arcs):
        cvec[W0 + a] = inst.theta * cost

    # ------------------------------------------------------------- bounds/int
    lb = np.zeros(n_var)
    ub = np.full(n_var, np.inf)
    integ = np.zeros(n_var)
    for col in X_idx.values():
        ub[col] = 1
        integ[col] = 1
    for col in Y_idx.values():
        ub[col] = 1
        integ[col] = 0 if relax_Y else 1
    for a in range(len(arcs)):
        ub[W0 + a] = inst.B
        integ[W0 + a] = 1

    rows = _Rows()
    # (4) each demand node-charge is assigned exactly once
    by_demand = defaultdict(list)
    by_server = defaultdict(list)
    for key, col in X_idx.items():
        by_demand[key[:2]].append(col)
        by_server[key[2:]].append((key, col))
    for d in demands:
        cols = by_demand[d]
        rows.add(cols, [1.0] * len(cols), 1.0, 1.0)
    # (6) Y_jhm <= Y_jh,m-1
    for (j, h) in nc:
        for m in range(2, C + 1):
            rows.add([Y_idx[(j, h, m)], Y_idx[(j, h, m - 1)]], [1.0, -1.0], -np.inf, 0.0)
    # (7) queueing intensity (piecewise linear in the number of servers)
    if queueing:
        for (j, h) in nc:
            cols = [col for _, col in by_server[(j, h)]]
            vals = [lam[key[0], key[1]] for key, _ in by_server[(j, h)]]
            for m in range(1, C + 1):
                cols.append(Y_idx[(j, h, m)])
                vals.append(-mu[j, h] * (rho[m] - rho[m - 1]))
            rows.add(cols, vals, -np.inf, 0.0)
    # (8) number of servers = number of idle vehicles
    cols = list(Y_idx.values())
    rows.add(cols, [1.0] * len(cols), inst.B, inst.B)
    # (9) only node-charges with a server can cover demand
    for key, col in X_idx.items():
        rows.add([col, Y_idx[(key[2], key[3], 1)]], [1.0, -1.0], -np.inf, 0.0)
    # (12) flow conservation: in - out - sum_m Y = -y
    inc = defaultdict(list)
    out = defaultdict(list)
    for a, (tail, head, _, _) in enumerate(arcs):
        out[tail].append(W0 + a)
        inc[head].append(W0 + a)
    for v in nc:
        cols = inc[v] + out[v] + [Y_idx[(v[0], v[1], m)] for m in range(1, C + 1)]
        vals = [1.0] * len(inc[v]) + [-1.0] * len(out[v]) + [-1.0] * C
        rows.add(cols, vals, -float(y0[v]), -float(y0[v]))
    # (13) charging-arc flow = sum of path flows covering that segment
    charge_arc = {(tail[0], tail[1]): W0 + a for a, (tail, _, _, ch) in enumerate(arcs) if ch}
    for (j, g), wcol in charge_arc.items():
        cols = [P_idx[(j, g1, h1)] for g1 in range(g + 1) for h1 in range(g + 1, H)]
        rows.add(cols + [wcol], [1.0] * len(cols) + [-1.0], 0.0, 0.0)
    # (14) path-based station capacity
    for j in inst.station_list:
        cols = [col for (jj, _, _), col in P_idx.items() if jj == j]
        if cols:
            rows.add(cols, [1.0] * len(cols), -np.inf, float(inst.stations[j]))

    options = {"disp": verbose, "mip_rel_gap": mip_rel_gap}
    if time_limit is not None:
        options["time_limit"] = time_limit
    res = milp(
        cvec,
        integrality=integ,
        bounds=Bounds(lb, ub),
        constraints=rows.build(n_var),
        options=options,
    )
    runtime = time.perf_counter() - t0
    if res.x is None:
        return Solution(
            routes=[[v] for v in inst.vehicles],
            assignment={},
            access_cost=np.nan,
            rebalancing_cost=np.nan,
            objective=np.nan,
            method="MILP" + ("" if queueing else " (myopic)"),
            status=f"no solution: {res.message}",
            runtime=runtime,
            queueing_enforced=queueing,
            info={"n_vars": n_var, "n_cons": len(rows.lb)},
        )
    x = res.x

    # ------------------------------------------------------------ extraction
    assignment = {}
    for (i, g, j, h), col in X_idx.items():
        if x[col] > 0.5:
            assignment[(i, g)] = (j, h)
    servers = np.zeros((N, H))
    for (j, h, m), col in Y_idx.items():
        servers[j, h] += x[col]
    servers = np.rint(servers).astype(int)  # integral by (12) even if Y is relaxed
    flows = {a: int(round(x[W0 + a])) for a in range(len(arcs)) if x[W0 + a] > 0.5}
    routes = _decompose_flows(inst, arcs, flows, servers)
    acc, reb, obj = evaluate(inst, routes, assignment)
    return Solution(
        routes=routes,
        assignment=assignment,
        access_cost=acc,
        rebalancing_cost=reb,
        objective=obj,
        method="MILP" + ("" if queueing else " (myopic)"),
        status=res.message,
        runtime=runtime,
        queueing_enforced=queueing,
        info={
            "n_vars": n_var,
            "n_cons": len(rows.lb),
            "solver_objective": float(res.fun),
            "mip_gap": getattr(res, "mip_gap", None),
            "relax_Y": relax_Y,
        },
    )


def _decompose_flows(inst: Instance, arcs, flows: Dict[int, int], servers: np.ndarray):
    """Split integral arc flows W into one route per vehicle (path decomposition)."""
    remaining = dict(flows)
    out = defaultdict(list)
    for a in flows:
        out[arcs[a][0]].append(a)
    deficit = servers.copy()
    routes = []
    for origin in inst.vehicles:
        v: NodeCharge = origin
        route = [v]
        for _ in range(10 * len(arcs) + 1):
            if deficit[v] > 0:
                break
            nxt = next((a for a in out[v] if remaining[a] > 0), None)
            if nxt is None:  # cannot happen for a conservation-feasible flow
                raise RuntimeError(f"flow decomposition stuck at {v}")
            remaining[nxt] -= 1
            v = arcs[nxt][1]
            route.append(v)
        deficit[v] -= 1
        routes.append(route)
    return routes
