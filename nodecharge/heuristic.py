"""Greedy heuristic for the node-charge relocation problem (Algorithm 1).

The heuristic is a Teitz and Bart (1968)-style greedy for the p-median
problem, modified to keep every intermediate solution feasible with respect to
(a) the queueing constraint (7), (b) charging-station capacity (14),
(c) minimum-cost-flow relocation in the node-charge graph, and (d) coverage of
demand at every charge level (4)-(5).

Outline (step numbers refer to Algorithm 1 in the paper)
--------------------------------------------------------
1. (Steps 1, 14-15) All B idle vehicles are first placed *virtually* at the
   single node-charge that minimises total access cost: the 1-median zone at
   the highest charge level with demand. Because one location pools all B servers, this is
   the configuration with the most queueing capacity; if even this cannot
   satisfy (7) the MILP is infeasible and the myopic model is used instead.
2. (Steps 2-13) In each iteration k, a savings value is computed for every
   candidate node-charge (i, g):

       gain_ig = sum_{j, h <= g} lambda_jh * max(0, a_jh - tau_ji)

   where a_jh is the current access cost of demand (j, h) (matrix A^{k-1}).
   This is the demand-weighted access saving of opening a server at (i, g).
   Following Steps 8-11, the saving is scaled by the share of the attracted
   demand d1 that the extra server can actually absorb,
   d2 = mu_ig * (rho_{n+1} - rho_n), where n is the number of servers already
   at (i, g):  s_ig = gain_ig * min(1, d2 / d1). The rebalancing cost of the
   vehicle, theta * c(L_f -> (i, g)), is subtracted.
3. (Steps 16-18) The (vehicle, node-charge) pair with the largest savings is
   chosen, skipping candidates that would violate (7) or coverage (the set
   N^M), that exceed C servers, or that need a charging station without spare
   capacity (G' restricted to the vehicle's own level when no charger is free).
4. (Steps 20-21) The vehicle is moved, station capacity is updated, and the
   demand-to-vehicle assignment (and thus A^k) is recomputed.

Implementation choices (the pseudo-code leaves these open)
----------------------------------------------------------
* Charging capacity is tracked per station (a vehicle that charges uses one
  path at the station it passes through), which is a conservative version of
  Step 20's fleet-wide count and always satisfies (13)-(14).
* Relocation cost to (i, g), g > g0, uses the cheapest station with a free
  charger: c = t(i0, j) + (g - g0) * charge_time_j + t(j, i).
* By default the vehicle relocated at each iteration is the one with the best
  savings (``vehicle_order="best"``); ``"index"`` processes vehicles in the
  given order, as in the literal reading of Step 18.
* Demand is assigned to vehicles with a capacity-aware greedy (single
  sourcing, nearest feasible vehicle first).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .instance import Instance, NodeCharge
from .queueing import rho_table
from .solution import Solution, evaluate


class _Assigner:
    """Assign demand node-charges to vehicles subject to (4), (5), (7), (9)."""

    def __init__(self, inst: Instance, rho: np.ndarray, queueing: bool):
        self.inst = inst
        self.rho = rho
        self.queueing = queueing
        dem = np.argwhere(inst.lam > 0)
        self.di, self.dg = dem[:, 0], dem[:, 1]
        self.dl = inst.lam[self.di, self.dg]
        self.nd = len(self.di)

    def __call__(self, counts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Return (server zone, server level, access cost) per demand, or None."""
        inst = self.inst
        srv = np.argwhere(counts > 0)
        sj, sh = srv[:, 0], srv[:, 1]
        T = inst.tau[self.di][:, sj]
        ok = sh[None, :] >= self.dg[:, None]
        T = np.where(ok, T, np.inf)
        if self.nd and not np.all(ok.any(axis=1)):
            return None  # some charge level is not covered at all
        if not self.queueing:
            k = np.argmin(T, axis=1)
            return sj[k], sh[k], float(np.sum(self.dl * T[np.arange(self.nd), k]))
        cap = inst.mu[sj, sh] * self.rho[counts[sj, sh]]
        best = None
        for k in (self._by_distance(T, cap), self._by_level(T, cap)):
            if k is None:
                continue
            cost = float(np.sum(self.dl * T[np.arange(self.nd), k]))
            if best is None or cost < best[1]:
                best = (k, cost)
        if best is None:
            return None
        k = best[0]
        return sj[k], sh[k], best[1]

    def _by_distance(self, T, cap):
        """Assign pairs in increasing access cost if the vehicle has room."""
        res = cap.copy()
        k = np.full(self.nd, -1)
        flat = np.argsort(T, axis=None, kind="stable")
        S = T.shape[1]
        left = self.nd
        for p in flat:
            d, s = divmod(int(p), S)
            if k[d] >= 0:
                continue
            if not np.isfinite(T[d, s]):
                break
            if self.dl[d] <= res[s] + 1e-12:
                k[d] = s
                res[s] -= self.dl[d]
                left -= 1
                if left == 0:
                    break
        return k if left == 0 else None

    def _by_level(self, T, cap):
        """High charge levels first (fewest options), then large demand first."""
        res = cap.copy()
        k = np.full(self.nd, -1)
        order = np.lexsort((-self.dl, -self.dg))
        for d in order:
            row = np.where(res >= self.dl[d] - 1e-12, T[d], np.inf)
            s = int(np.argmin(row))
            if not np.isfinite(row[s]):
                return None
            k[d] = s
            res[s] -= self.dl[d]
        return k


def _relocation_costs(inst: Instance, origin: NodeCharge, resid: Dict[int, int]):
    """Cheapest cost from ``origin`` to every node-charge, and station used."""
    N, H = inst.n_zones, inst.n_levels
    tau = inst.tau
    i0, g0 = origin
    R = np.full((N, H), np.inf)
    ST = np.full((N, H), -1)
    R[:, g0] = tau[i0, :]
    S = np.array([j for j in inst.station_list if resid.get(j, 0) > 0], dtype=int)
    if len(S) and g0 < H - 1:
        cc = np.array([inst.charge_cost(j) for j in S])
        base = tau[i0, S][:, None] + tau[S, :]  # (|S|, N)
        for g in range(g0 + 1, H):
            tot = base + (cc * (g - g0))[:, None]
            k = np.argmin(tot, axis=0)
            R[:, g] = tot[k, np.arange(N)]
            ST[:, g] = S[k]
    return R, ST


def _route(inst: Instance, origin: NodeCharge, dest: NodeCharge, station: int) -> List[NodeCharge]:
    i0, g0 = origin
    i, g = dest
    if g == g0:
        return [(z, g0) for z in inst.zone_path(i0, i)]
    route = [(z, g0) for z in inst.zone_path(i0, station)]
    route += [(station, h) for h in range(g0 + 1, g + 1)]
    route += [(z, g) for z in inst.zone_path(station, i)[1:]]
    return route


def _greedy(inst: Instance, queueing: bool, vehicle_order: str, max_tries: int, n_eval: int):
    N, H, B, C = inst.n_zones, inst.n_levels, inst.B, inst.C
    mu, tau = inst.mu, inst.tau
    rho = rho_table(max(C, B) + 1, inst.eta, inst.b)
    assigner = _Assigner(inst, rho, queueing)
    di, dg, dl = assigner.di, assigner.dg, assigner.dl
    level_mask = (dg[:, None] <= np.arange(H)[None, :]).astype(float)  # (Nd, H)

    # Steps 1, 14-15: all vehicles virtually at the 1-median zone, at the
    # highest charge level that has demand (so every demand level is covered).
    top = int(dg.max()) if len(dg) else 0
    seed = int(np.argmin(dl @ tau[di, :]))
    real = np.zeros((N, H), dtype=int)
    virt = np.zeros((N, H), dtype=int)
    virt[seed, top] = B
    cur = assigner(real + virt)
    if cur is None:
        return None
    resid = dict(inst.stations)
    routes: List[Optional[List[NodeCharge]]] = [None] * B
    unplaced = list(range(B))
    iters = []

    while unplaced:  # Step 2
        aj, ah, _ = cur
        a = tau[di, aj]  # A^{k-1}: current access cost of each demand
        # Steps 3-13: savings matrix S^k
        attract = np.maximum(0.0, a[None, :] - tau[:, di])  # (N, Nd)
        gain = (attract * dl[None, :]) @ level_mask  # (N, H)
        d1 = ((attract > 0) * dl[None, :]) @ level_mask
        n_now = real + virt
        d2 = mu * (rho[n_now + 1] - rho[n_now])
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(d1 > 0, np.minimum(1.0, d2 / d1), 0.0)
        s = gain * frac
        full = real >= C

        pool = unplaced if vehicle_order == "best" else unplaced[:1]
        scores, stations = [], []
        for f in pool:
            R, ST = _relocation_costs(inst, inst.vehicles[f], resid)
            sc = s - inst.theta * R
            sc[full | ~np.isfinite(R)] = -np.inf
            scores.append(sc)
            stations.append(ST)
        scores = np.stack(scores)  # (len(pool), N, H)

        # Steps 16-18: best feasible candidate (skipping the set N^M)
        # With n_eval > 1, the n_eval best feasible candidates by savings are
        # re-scored with their exact objective change (access cost from the
        # updated assignment, Step 21, plus theta * relocation cost).
        order = np.argsort(-scores, axis=None, kind="stable")
        chosen = None
        tries = n_ok = 0
        for p in order:
            pf, i, g = np.unravel_index(p, scores.shape)
            if not np.isfinite(scores[pf, i, g]) or tries >= max_tries or n_ok >= n_eval:
                break
            tries += 1
            trial = real + virt
            trial[i, g] += 1
            trial[seed, top] -= 1
            res = assigner(trial)
            if res is None:
                continue  # (i, g) belongs to N^M in this iteration
            n_ok += 1
            obj = res[2] + (s[i, g] - scores[pf, i, g])  # access + theta * relocation
            if chosen is None or obj < chosen[4] - 1e-12:
                chosen = (int(pf), int(i), int(g), res, obj)
        if chosen is None:
            return None
        pf, i, g, cur, _ = chosen
        f = pool[pf]
        st = int(stations[pf][i, g])
        origin = inst.vehicles[f]
        routes[f] = _route(inst, origin, (i, g), st)
        if g > origin[1]:
            resid[st] -= 1  # Step 20: one charging path used at station st
        real[i, g] += 1
        virt[seed, top] -= 1
        unplaced.remove(f)
        iters.append({"vehicle": f, "target": (i, g), "savings": float(scores[pf, i, g]), "tries": tries})

    aj, ah, _ = cur
    assignment = {(int(i), int(g)): (int(j), int(h)) for i, g, j, h in zip(di, dg, aj, ah)}
    return routes, assignment, iters


def solve_heuristic(
    inst: Instance,
    queueing: bool = True,
    vehicle_order: str = "best",
    max_tries: int = 500,
    n_eval: int = 10,
    fallback_myopic: bool = True,
) -> Solution:
    """Run the greedy heuristic (Algorithm 1).

    Parameters
    ----------
    queueing : enforce the queueing constraint (7) (non-myopic policy).
    vehicle_order : ``"best"`` (relocate the vehicle with the largest savings
        each iteration) or ``"index"`` (vehicles in input order).
    max_tries : maximum candidates tested for feasibility per iteration.
    fallback_myopic : if no queueing-feasible solution is found, re-run with
        (7) relaxed, as recommended in Section 3.3.
    """
    if vehicle_order not in ("best", "index"):
        raise ValueError("vehicle_order must be 'best' or 'index'")
    t0 = time.perf_counter()
    out = _greedy(inst, queueing, vehicle_order, max_tries, n_eval)
    status = "feasible"
    enforced = queueing
    if out is None and queueing and fallback_myopic:
        out = _greedy(inst, False, vehicle_order, max_tries, n_eval)
        status = "myopic fallback: queueing constraint (7) relaxed"
        enforced = False
    runtime = time.perf_counter() - t0
    method = "Heuristic" + ("" if queueing else " (myopic)")
    if out is None:
        return Solution(
            routes=[[v] for v in inst.vehicles],
            assignment={},
            access_cost=np.nan,
            rebalancing_cost=np.nan,
            objective=np.nan,
            method=method,
            status="infeasible",
            runtime=runtime,
            queueing_enforced=enforced,
        )
    routes, assignment, iters = out
    acc, reb, obj = evaluate(inst, routes, assignment)
    return Solution(
        routes=routes,
        assignment=assignment,
        access_cost=acc,
        rebalancing_cost=reb,
        objective=obj,
        method=method,
        status=status,
        runtime=runtime,
        queueing_enforced=enforced,
        info={"iterations": iters},
    )
