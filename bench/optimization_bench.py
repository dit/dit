"""
Benchmark harness for dit's optimization layer.

Times a few representative optimizers across backends and reports both
wall-clock and the achieved objective value, so speedup work can be verified
against an accuracy baseline (objective values must not regress).

Run with::

    uv run python bench/optimization_bench.py
    uv run python bench/optimization_bench.py --backends numpy jax
    uv run python bench/optimization_bench.py --repeat 5 --json baseline.json

This file is a developer tool; it is not imported by the ``dit`` package.
"""

import argparse
import json
import time
from contextlib import contextmanager

import numpy as np

import dit
from dit.multivariate import (
    deweese_total_correlation,
    exact_common_information,
    stochastic_gk_common_information,
    wyner_common_information,
)
from dit.multivariate.secret_key_agreement import (
    intrinsic_dual_total_correlation,
    intrinsic_total_correlation,
)

SEED = 0


@contextmanager
def _seeded(seed=SEED):
    """Run with a fixed numpy RNG state so random restarts are reproducible."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _skar_dist():
    """A small tripartite distribution for the intrinsic-TC (SKAR) optimizer."""
    # Correlated X, Y with an eavesdropper Z; small alphabets keep it tractable.
    return dit.Distribution(
        ["000", "011", "101", "110", "001", "010", "100", "111"],
        [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
    )


# Each case: (name, zero-arg thunk returning the objective value).
def _build_cases(backend):
    cases = []

    # --- convex: maximum entropy via the scipy convex optimizer ---
    def maxent():
        from dit.algorithms.distribution_optimizers import MaxEntOptimizer

        d = dit.example_dists.n_mod_m(4, 2)
        meo = MaxEntOptimizer(d, [[0, 1], [1, 2], [2, 3], [0, 3]])
        meo.optimize()
        return float(meo.objective(meo._optima))

    cases.append(("maxent_scipy", maxent))

    # --- non-convex distribution optimizer with an analytic gradient ---
    def minent():
        from dit.algorithms.distribution_optimizers import MinEntOptimizer

        d = dit.example_dists.n_mod_m(4, 2)
        meo = MinEntOptimizer(d, [[0, 1], [1, 2], [2, 3]])
        meo.optimize(niter=5)
        return float(meo.objective(meo._optima))

    cases.append(("minent_scipy", minent))

    # --- non-convex: Wyner common information (Markov-var optimizer) ---
    # And() has dtc != H, so the optimizer (and W-minimization post-process)
    # actually runs rather than hitting the dtc==H shortcut.
    def wyner():
        return float(wyner_common_information(dit.example_dists.And(), backend=backend))

    cases.append(("wyner_common_information", wyner))

    # --- non-convex: exact common information ---
    def exact():
        return float(exact_common_information(dit.example_dists.And(), backend=backend))

    cases.append(("exact_common_information", exact))

    # --- non-convex: intrinsic total correlation (SKAR, aux-var optimizer) ---
    def itc():
        return float(intrinsic_total_correlation(_skar_dist(), [[0], [1]], [2]))

    cases.append(("intrinsic_total_correlation", itc))

    # --- aux-var optimizers newly given analytic gradients ---
    def idtc():
        return float(intrinsic_dual_total_correlation(_skar_dist(), [[0], [1]], [2]))

    cases.append(("intrinsic_dual_total_correlation", idtc))

    def stochastic_gk():
        return float(stochastic_gk_common_information(dit.example_dists.Xor(), [[0], [1]], [2]))

    cases.append(("stochastic_gk_common_information", stochastic_gk))

    def deweese_tc():
        return float(deweese_total_correlation(dit.example_dists.Xor(), [[0], [1], [2]]))

    cases.append(("deweese_total_correlation", deweese_tc))

    # --- one-way SKAR (parallelized bound sweep + role swaps) ---
    def owskar():
        from dit.multivariate.secret_key_agreement import one_way_skar

        return float(one_way_skar(_skar_dist(), [0], [1], [2], backend=backend))

    cases.append(("one_way_skar", owskar))

    # --- PID I_BROJA with 3 sources (parallelized per-source loop) ---
    # Time the per-source unique-information sweep directly: that is the loop
    # wrapped in parallel_sweep. (The full PID_BROJA lattice for 3 sources spends
    # its time in unrelated Mobius machinery and yields nan on its higher nodes,
    # so it is not a meaningful gauge of this sweep.)
    def ibroja3():
        from dit.pid.measures.ibroja import PID_BROJA

        d = dit.Distribution(
            ["0000", "0111", "1011", "1101", "1110", "0001", "0010", "0100"],
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.0833, 0.0833, 0.0834],
        )
        uniques = PID_BROJA._measure(d, [(0,), (1,), (2,)], (3,))
        return float(sum(uniques.values()))

    cases.append(("pid_broja_3source", ibroja3))

    # --- rate-distortion (residual entropy, analytic gradient) ---
    def rd():
        from dit.rate_distortion import RDCurve

        d = dit.Distribution(["00", "01", "10", "11"], [0.1, 0.2, 0.3, 0.4])
        curve = RDCurve(d, beta_num=8)
        return float(np.trapezoid(curve.rates, curve.distortions))

    cases.append(("rate_distortion_curve", rd))

    # --- information bottleneck (analytic gradient) ---
    def ib():
        from dit.rate_distortion.information_bottleneck import InformationBottleneck

        d = dit.Distribution(["00", "01", "10", "11"], [0.1, 0.2, 0.3, 0.4])
        opt = InformationBottleneck(d, beta=3.0, rvs=[[0], [1]])
        opt.optimize()
        pmf = opt.construct_joint(opt._optima)
        return float(opt.complexity(pmf))

    cases.append(("information_bottleneck", ib))

    # --- Gray-Wyner rate point (parallelized region/curve sweeps) ---
    def gray_wyner():
        from dit.rate_distortion.gray_wyner import GrayWynerNetwork

        d = dit.Distribution(["00", "11"], [0.5, 0.5])
        net = GrayWynerNetwork(d)
        point = net.rate_point([1.0, 1.0, 1.0])
        return float(point.common)

    cases.append(("gray_wyner_point", gray_wyner))

    return cases


def run(backends, repeat):
    """Run all benchmark cases for each backend, returning a results dict."""
    results = {}
    for backend in backends:
        for name, thunk in _build_cases(backend):
            times = []
            value = None
            for _ in range(repeat):
                with _seeded():
                    t0 = time.perf_counter()
                    value = thunk()
                    times.append(time.perf_counter() - t0)
            key = f"{backend}:{name}"
            results[key] = {
                "backend": backend,
                "case": name,
                "best_s": min(times),
                "mean_s": sum(times) / len(times),
                "value": value,
            }
    return results


def _print_table(results):
    width = max(len(k) for k in results)
    header = f"{'case':<{width}}  {'best (s)':>10}  {'mean (s)':>10}  {'objective':>14}"
    print(header)
    print("-" * len(header))
    for key, r in results.items():
        print(f"{key:<{width}}  {r['best_s']:>10.4f}  {r['mean_s']:>10.4f}  {r['value']:>14.8f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["numpy"], help="Backends to benchmark.")
    parser.add_argument("--repeat", type=int, default=3, help="Repetitions per case (best is reported).")
    parser.add_argument("--json", default=None, help="Optional path to dump results as JSON.")
    args = parser.parse_args()

    results = run(args.backends, args.repeat)
    _print_table(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
