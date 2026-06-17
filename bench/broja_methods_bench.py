"""
Benchmark BROJA bivariate solvers: scipy, admUI, and exponential cone (ECOS).

Run with::

    uv run python bench/broja_methods_bench.py
    uv run python bench/broja_methods_bench.py --repeat 5 --json results.json

Developer tool; not imported by ``dit``.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from math import prod

import numpy as np

import dit
from dit.algorithms.broja_method import (
    ADMUI_MIN_JOINT,
    SCIPY_MAX_JOINT,
    broja_solve_bivariate,
    ecos_available,
)
from dit.algorithms.pid_broja import prepare_dist as broja_prepare_dist
from dit.pid.distributions import bivariates

SEED = 0
METHODS = ["scipy", "admui"] + (["cone"] if ecos_available() else [])


@contextmanager
def _seeded(seed=SEED):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _ui_sum(d, method):
    uniques, meta = broja_solve_bivariate(d, ((0,), (1,)), (2,), method=method)
    return float(sum(uniques.values())), uniques, meta


def _cases():
    cases = []

    for k in (2, 3, 4, 5, 6, 8):
        n = k**3
        pmf = np.ones(n) / n
        outcomes = ["".join(map(str, o)) for o in np.ndindex(k, k, k)]
        cases.append((f"uniform_{k}^3", dit.Distribution(outcomes, pmf)))

    for b in (1, 2, 3, 4, 6):
        cases.append((f"summed_dice_b{b}", dit.example_dists.summed_dice(0.5, b)))

    for name in ("and", "diff", "redundant"):
        cases.append((f"bivariate_{name}", bivariates[name]))

    return cases


def run(repeat):
    ref_method = "scipy"
    results = {}

    for case_name, dist in _cases():
        prepared = broja_prepare_dist(dist, [[0], [1]], [2])
        alphabet_sizes = [len(a) for a in prepared.alphabet]
        n_joint = prod(alphabet_sizes)

        with _seeded():
            ref_val, ref_uniques, _ = _ui_sum(dist, ref_method)

        row = {"n_joint": n_joint, "alphabet_sizes": alphabet_sizes, "methods": {}}

        for method in METHODS:
            times = []
            value = None
            uniques = None
            meta = None
            for _ in range(repeat):
                with _seeded():
                    t0 = time.perf_counter()
                    uniques, meta = broja_solve_bivariate(dist, ((0,), (1,)), (2,), method=method)
                    times.append(time.perf_counter() - t0)
                    value = float(sum(uniques.values()))

            delta = max(abs(uniques[s] - ref_uniques[s]) for s in ref_uniques)
            row["methods"][method] = {
                "best_s": min(times),
                "mean_s": sum(times) / len(times),
                "ui_sum": value,
                "delta_ui": delta,
                "meta": {k: v for k, v in (meta or {}).items() if k != "Solver Object"},
            }

        results[case_name] = row

    return results


def _summarize(results):
    lines = []
    lines.append(f"Thresholds: SCIPY_MAX_JOINT={SCIPY_MAX_JOINT}, ADMUI_MIN_JOINT={ADMUI_MIN_JOINT}")
    if "admui" not in METHODS:
        lines.append("admui not run")
        return lines

    for case, row in results.items():
        scipy_t = row["methods"]["scipy"]["best_s"]
        admui_t = row["methods"]["admui"]["best_s"]
        delta = row["methods"]["admui"]["delta_ui"]
        speedup = scipy_t / admui_t if admui_t > 0 else float("inf")
        if speedup >= 1.5 and delta < 1e-4:
            lines.append(f"admui wins: {case} (|XYZ|={row['n_joint']}, {speedup:.2f}x, Δ={delta:.2e})")

    return lines


def _print_table(results):
    width = max(len(k) for k in results)
    print(f"{'case':<{width}}  {'|XYZ|':>6}  " + "  ".join(f"{m + '_s':>10}" for m in METHODS) + "  admui_Δ")
    print("-" * (width + 30 + 12 * len(METHODS)))
    for case, row in results.items():
        parts = [f"{case:<{width}}", f"{row['n_joint']:>6}"]
        for m in METHODS:
            parts.append(f"{row['methods'][m]['best_s']:>10.4f}")
        parts.append(f"{row['methods'].get('admui', {}).get('delta_ui', float('nan')):>10.2e}")
        print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    results = run(args.repeat)
    _print_table(results)
    print()
    for line in _summarize(results):
        print(line)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
