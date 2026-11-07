import numpy as np
import pytest

from nodecharge import (
    Instance,
    check_solution,
    random_instance,
    rho_table,
    solve_heuristic,
    solve_milp,
)
from nodecharge.solution import Solution, station_paths_needed


def line_instance(u=1, vehicles=((2, 0),), lam=None, mu=100.0, theta=0.1):
    """Three zones on a line 0 - 1 - 2 (10 min links), two charge levels,
    a charging station at zone 0 with u chargers."""
    if lam is None:
        lam = np.zeros((3, 2))
        lam[0, 1] = 1.0  # demand only at zone 0 for the high charge level
    return Instance(
        n_zones=3,
        n_levels=2,
        edges=[(0, 1, 10.0), (1, 2, 10.0)],
        lam=lam,
        mu=mu,
        stations={0: u},
        vehicles=list(vehicles),
        charge_time=25.0,
        theta=theta,
        C=3,
    )


def test_rho_matches_table1():
    # Table 1 of the paper lists rho = (0.2236, 0.6416, 1.1576, 1.7345)
    rho = rho_table(4, eta=0.95, b=0)
    assert np.allclose(rho[1:], [0.2236, 0.6416, 1.1576, 1.7345], atol=1e-4)


def test_single_vehicle_must_charge():
    inst = line_instance()
    for solver in (solve_milp, solve_heuristic):
        sol = solver(inst)
        assert check_solution(inst, sol) == []
        assert sol.destinations == [(0, 1)]
        assert sol.charging_station(0) == 0
        # route 2 -> 1 -> 0 at level 0, then charge: theta * (10 + 10 + 25)
        assert sol.objective == pytest.approx(0.1 * 45.0)


def test_station_capacity_binds():
    lam = np.zeros((3, 2))
    lam[0, 1] = 1.0
    lam[2, 1] = 1.0
    two = [(1, 0), (1, 0)]
    # With 2 chargers both vehicles charge and serve one zone each.
    roomy = line_instance(u=2, vehicles=two, lam=lam)
    s2 = solve_milp(roomy)
    assert sorted(s2.destinations) == [(0, 1), (2, 1)]
    # With 1 charger only one vehicle can reach level 1; it must cover both zones.
    tight = line_instance(u=1, vehicles=two, lam=lam)
    for solver in (solve_milp, solve_heuristic):
        s1 = solver(tight)
        assert check_solution(tight, s1) == []
        assert sum(h == 1 for _, h in s1.destinations) == 1
        assert s1.objective > s2.objective


def test_queueing_forces_extra_server():
    # Demand 1.0 at a single node-charge; one server handles mu * rho_1.
    lam = np.zeros((3, 2))
    lam[1, 0] = 1.0
    inst = line_instance(u=1, vehicles=[(0, 0), (2, 0)], lam=lam, mu=1.0 / 0.5)
    # mu * rho_1 = 0.447 < 1 and mu * rho_2 = 1.28 >= 1: two servers needed at zone 1
    sol = solve_milp(inst)
    assert check_solution(inst, sol) == []
    assert sol.destinations == [(1, 0), (1, 0)]
    myopic = solve_milp(inst, queueing=False)
    assert myopic.objective < sol.objective
    heur = solve_heuristic(inst)
    assert check_solution(inst, heur) == []
    assert heur.objective == pytest.approx(sol.objective)


def test_heuristic_falls_back_to_myopic_when_infeasible():
    lam = np.zeros((3, 2))
    lam[1, 0] = 1.0
    inst = line_instance(vehicles=[(0, 0)], lam=lam, mu=1.0)  # mu * rho_1 < 1
    assert "no solution" in solve_milp(inst).status
    sol = solve_heuristic(inst)
    assert sol.status.startswith("myopic fallback")
    assert check_solution(inst, sol) == []


def test_path_counting_matches_constraint_13_14():
    # Hand-built routes for a 3-level station: segment flows (1, 1) need one path.
    inst3 = Instance(n_zones=1, n_levels=3, edges=[], lam=np.ones((1, 3)), mu=10.0,
                     stations={0: 1}, vehicles=[(0, 0), (0, 1)])
    sol = Solution(routes=[[(0, 0), (0, 1)], [(0, 1), (0, 2)]], assignment={},
                   access_cost=0, rebalancing_cost=0, objective=0)
    assert station_paths_needed(inst3, sol) == {0: 1}
    sol2 = Solution(routes=[[(0, 0), (0, 1)], [(0, 0), (0, 1), (0, 2)]], assignment={},
                    access_cost=0, rebalancing_cost=0, objective=0)
    assert station_paths_needed(inst3, sol2) == {0: 2}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n", [10, 20])
def test_random_instances_feasible_and_bounded(n, seed):
    inst = random_instance(n, 5, seed=seed)
    exact = solve_milp(inst)
    heur = solve_heuristic(inst)
    assert check_solution(inst, exact) == []
    assert check_solution(inst, heur) == []
    assert heur.objective >= exact.objective - 1e-6
    myopic = solve_milp(inst, queueing=False)
    assert myopic.objective <= exact.objective + 1e-6
