"""Small test case: one rebalancing interval on a random 10-zone instance.

Uses the parameter settings of Table 1 in Pantelidis et al. (2022):
|N| = 10 zones, |H| = 5 charge levels, B = 4 idle EVs, 1 charging station,
lambda ~ U[0, 1], theta = 0.02, C = 3, eta = 0.95, b = 0, mu = 1.5 * sum(lambda).

Run from the repository root:

    python examples/random_instance.py            # text output
    python examples/random_instance.py --plot     # also saves a figure
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nodecharge import check_solution, random_instance, solve_heuristic, solve_milp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", type=int, default=10)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true", help="save examples/random_instance.png")
    args = ap.parse_args()

    inst = random_instance(args.zones, args.levels, seed=args.seed)
    print(inst.summary())
    print("rho_m (m = 1..C):", [round(float(r), 4) for r in inst.rho[1 : inst.C + 1]])
    print("Initial idle vehicles (zone, level):", inst.vehicles)
    print("Charging stations {zone: chargers}:", inst.stations)
    print()

    results = [
        solve_milp(inst, queueing=True),
        solve_heuristic(inst, queueing=True),
        solve_milp(inst, queueing=False),
    ]
    for sol in results:
        print(sol.summary(inst))
        errs = check_solution(inst, sol)
        print("  feasibility check:", "OK" if not errs else errs)
        print()

    exact, heur = results[0], results[1]
    gap = 100 * (heur.objective - exact.objective) / exact.objective
    print(f"Heuristic optimality gap: {gap:.2f}%   "
          f"(MILP {exact.runtime:.3f}s vs heuristic {heur.runtime:.3f}s)")

    if args.plot:
        import matplotlib.pyplot as plt

        from nodecharge.plot import plot_solution

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        plot_solution(inst, exact, ax=axes[0])
        plot_solution(inst, heur, ax=axes[1])
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "random_instance.png")
        fig.savefig(out, dpi=150)
        print("Saved", out)


if __name__ == "__main__":
    main()
