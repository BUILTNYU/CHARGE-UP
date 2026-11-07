"""Scalability test in the spirit of Table 2 of the paper.

For each network size, random instances (Table 1 settings) are solved with the
exact MILP (HiGHS) and with the greedy heuristic; the table reports mean
runtimes and the heuristic's optimality gap.

    python examples/benchmark.py --sizes 10 20 50 --reps 5
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nodecharge import check_solution, random_instance, solve_heuristic, solve_milp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=600.0)
    args = ap.parse_args()

    print(f"{'N':>5} {'H':>3} {'N x H':>7} {'MILP (s)':>10} {'gap (%)':>9} {'max gap':>8} {'heuristic (s)':>14}")
    for N in args.sizes:
        tm, th, gaps = [], [], []
        for rep in range(args.reps):
            inst = random_instance(N, args.levels, seed=rep)
            m = solve_milp(inst, time_limit=args.time_limit)
            h = solve_heuristic(inst, n_eval=args.n_eval)
            assert not check_solution(inst, m), check_solution(inst, m)
            assert not check_solution(inst, h), check_solution(inst, h)
            tm.append(m.runtime)
            th.append(h.runtime)
            gaps.append(100 * (h.objective - m.objective) / m.objective)
        print(f"{N:>5} {args.levels:>3} {N * args.levels:>7} {np.mean(tm):>10.2f} "
              f"{np.mean(gaps):>9.2f} {np.max(gaps):>8.2f} {np.mean(th):>14.3f}")


if __name__ == "__main__":
    main()
