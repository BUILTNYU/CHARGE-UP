"""Problem instance and node-charge graph (Section 3.2 of Pantelidis et al., 2022).

Conventions
-----------
* Zones are indexed ``0 .. N-1``; charge levels are indexed ``0 .. H-1`` where
  level ``h`` is the band "at least the (h+1)-th charge level" (e.g. with H = 4,
  level 0 = 20%+, level 3 = 80%+). A vehicle at level ``h`` can serve customers
  requesting any level ``g <= h``.
* A node-charge is a pair ``(i, h)``. The node-charge graph has
    - relocation arcs ``(i, h) -> (j, h)`` for every road link {i, j} and level h
      (cost = link travel time), and
    - charging arcs ``(j, h) -> (j, h+1)`` at every charging station j
      (cost = charging time for one level).
* Access cost ``tau[i, j]`` (customer at zone i walking/travelling to a vehicle
  at zone j) and relocation travel costs both come from the shortest paths in
  the zone network, as in the paper's test instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from .queueing import rho_table

NodeCharge = Tuple[int, int]


@dataclass
class Instance:
    """A single rebalancing-interval instance of the node-charge relocation problem.

    Parameters
    ----------
    n_zones : |N|, number of zone centroids.
    n_levels : |H|, number of discrete charge levels.
    edges : undirected road links ``(i, j, travel_time)``.
    lam : array (N, H). ``lam[i, g]`` = arrival rate of customers at zone i who
        need charge level g or higher (lambda_ig).
    mu : array (N, H) or scalar. Service rate per vehicle at node-charge (mu_jh).
    stations : mapping ``zone -> u_j`` (number of chargers at that zone).
    vehicles : initial node-charge ``(zone, level)`` of each idle vehicle (|F| = B).
    charge_time : cost of one charging arc (one level up). Scalar, or mapping
        ``zone -> cost`` for station-specific charging speeds.
    theta : weight of rebalancing cost in the objective (Eq. 3).
    C : maximum number of servers (vehicles) per node-charge (Eq. 16).
    eta, b : queueing reliability parameters used to compute rho (Eq. 19).
    coords : optional (N, 2) zone coordinates, only used for plotting.
    """

    n_zones: int
    n_levels: int
    edges: Sequence[Tuple[int, int, float]]
    lam: np.ndarray
    mu: np.ndarray | float
    stations: Dict[int, int]
    vehicles: Sequence[NodeCharge]
    charge_time: float | Dict[int, float] = 25.0
    theta: float = 0.02
    C: int = 3
    eta: float = 0.95
    b: int = 0
    coords: Optional[np.ndarray] = None
    name: str = "instance"
    _extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        N, H = self.n_zones, self.n_levels
        self.lam = np.asarray(self.lam, dtype=float)
        if self.lam.shape != (N, H):
            raise ValueError(f"lam must have shape {(N, H)}, got {self.lam.shape}")
        if np.any(self.lam < 0):
            raise ValueError("arrival rates must be non-negative")
        self.mu = np.broadcast_to(np.asarray(self.mu, dtype=float), (N, H)).copy()
        self.vehicles = [tuple(map(int, v)) for v in self.vehicles]
        for i, h in self.vehicles:
            if not (0 <= i < N and 0 <= h < H):
                raise ValueError(f"vehicle location {(i, h)} outside the node-charge graph")
        for j, u in self.stations.items():
            if not 0 <= j < N:
                raise ValueError(f"station zone {j} out of range")
            if u < 0:
                raise ValueError("station capacity must be >= 0")
        if self.C < 1:
            raise ValueError("C must be >= 1")
        if len(self.vehicles) > N * H * self.C:
            raise ValueError("more vehicles than node-charge server slots (N*H*C)")
        _, labels = connected_components(self._adjacency, directed=False)
        if len(set(labels)) > 1:
            raise ValueError("zone network must be connected")

    # ------------------------------------------------------------------ basics
    @property
    def B(self) -> int:
        """Number of idle vehicles (Eq. 8)."""
        return len(self.vehicles)

    @property
    def station_list(self) -> List[int]:
        return sorted(self.stations)

    def charge_cost(self, j: int) -> float:
        if isinstance(self.charge_time, dict):
            return float(self.charge_time[j])
        return float(self.charge_time)

    @cached_property
    def _adjacency(self) -> csr_matrix:
        N = self.n_zones
        rows, cols, w = [], [], []
        for i, j, t in self.edges:
            if i == j:
                continue
            rows += [i, j]
            cols += [j, i]
            w += [t, t]
        return csr_matrix((w, (rows, cols)), shape=(N, N))

    @cached_property
    def _sp(self):
        return shortest_path(self._adjacency, directed=False, return_predecessors=True)

    @property
    def tau(self) -> np.ndarray:
        """All-pairs shortest travel time between zones (access cost tau_ij)."""
        return self._sp[0]

    @property
    def predecessors(self) -> np.ndarray:
        return self._sp[1]

    def zone_path(self, i: int, j: int) -> List[int]:
        """Shortest zone sequence from i to j (inclusive)."""
        path = [j]
        pred = self.predecessors
        while path[-1] != i:
            path.append(int(pred[i, path[-1]]))
        return path[::-1]

    @cached_property
    def rho(self) -> np.ndarray:
        """``rho[m]`` for m = 0..max(C, B); rho[0] = 0 (Eq. 19)."""
        return rho_table(max(self.C, self.B), self.eta, self.b)

    @property
    def initial_counts(self) -> np.ndarray:
        """y_ig: number of idle vehicles at each node-charge before rebalancing."""
        y = np.zeros((self.n_zones, self.n_levels), dtype=int)
        for i, h in self.vehicles:
            y[i, h] += 1
        return y

    # ------------------------------------------------------- node-charge graph
    @cached_property
    def arcs(self) -> List[Tuple[NodeCharge, NodeCharge, float, bool]]:
        """Arcs of the node-charge graph: ``(tail, head, cost, is_charging)``."""
        out = []
        for h in range(self.n_levels):
            for i, j, t in self.edges:
                if i == j:
                    continue
                out.append(((i, h), (j, h), float(t), False))
                out.append(((j, h), (i, h), float(t), False))
        for j in self.station_list:
            if self.stations[j] <= 0:
                continue
            for h in range(self.n_levels - 1):
                out.append(((j, h), (j, h + 1), self.charge_cost(j), True))
        return out

    def summary(self) -> str:
        return (
            f"{self.name}: |N|={self.n_zones}, |H|={self.n_levels}, "
            f"node-charges={self.n_zones * self.n_levels}, B={self.B}, "
            f"|J|={len(self.stations)} (capacity {sum(self.stations.values())}), "
            f"C={self.C}, theta={self.theta}, eta={self.eta}, b={self.b}, "
            f"sum(lambda)={self.lam.sum():.3f}"
        )


# --------------------------------------------------------------------------
def random_instance(
    n_zones: int = 10,
    n_levels: int = 5,
    seed: int | None = 0,
    *,
    fleet_ratio: float = 0.4,
    station_ratio: float = 0.1,
    theta: float = 0.02,
    C: int = 3,
    eta: float = 0.95,
    b: int = 0,
    mu_factor: float = 1.5,
    area_km: float = 10.0,
    speed_kmh: float = 30.0,
    charge_minutes_per_level: float = 6.0,
) -> Instance:
    """Random instance following the settings in Table 1 of the paper.

    * |H| = 5, B = 0.4|N| idle vehicles, |J| = 0.1|N| charging stations,
      u_j = B / |J|, lambda_ig ~ U[0, 1], theta = 0.02, C = 3,
      rho from eta = 0.95, b = 0 (gives rho = 0.2236, 0.6416, 1.1576, ...),
      mu = 1.5 * sum(lambda).
    * The paper does not specify the network geometry, so zones are placed
      uniformly in an ``area_km`` square and connected by a Delaunay
      triangulation; link travel times are in minutes at ``speed_kmh``.
    * Charging one level takes ``charge_minutes_per_level`` (default: a 30-min
      DC fast charge split into 5 levels, as in the Brooklyn case study).
    * Idle vehicles start at uniformly random node-charges.
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    N, H = n_zones, n_levels
    coords = rng.uniform(0.0, area_km, size=(N, 2))
    if N == 2:
        pairs = {(0, 1)}
    else:
        tri = Delaunay(coords)
        pairs = set()
        for s in tri.simplices:
            for a in range(3):
                for c in range(a + 1, 3):
                    i, j = sorted((int(s[a]), int(s[c])))
                    pairs.add((i, j))
    minutes_per_km = 60.0 / speed_kmh
    edges = [
        (i, j, round(float(np.linalg.norm(coords[i] - coords[j])) * minutes_per_km, 3))
        for i, j in sorted(pairs)
    ]
    lam = rng.uniform(0.0, 1.0, size=(N, H))
    B = max(1, round(fleet_ratio * N))
    n_st = max(1, round(station_ratio * N))
    st_zones = rng.choice(N, size=n_st, replace=False)
    u = int(np.ceil(B / n_st))
    stations = {int(j): u for j in st_zones}
    vehicles = [(int(rng.integers(N)), int(rng.integers(H))) for _ in range(B)]
    return Instance(
        n_zones=N,
        n_levels=H,
        edges=edges,
        lam=lam,
        mu=mu_factor * lam.sum(),
        stations=stations,
        vehicles=vehicles,
        charge_time=charge_minutes_per_level,
        theta=theta,
        C=C,
        eta=eta,
        b=b,
        coords=coords,
        name=f"random_N{N}_H{H}_seed{seed}",
    )
