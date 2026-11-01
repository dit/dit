"""
Benchmark the partial information decompositions: time to compute every node of each
measure's lattice on distributions with two, three and four sources.

Run with::

    uv run python bench/pid_bench.py
    uv run python bench/pid_bench.py --sources 3 --measures PID_WB PID_SX
    uv run python bench/pid_bench.py --json before.json
    uv run python bench/pid_bench.py --compare before.json

``--compare`` reports the speedup over a saved run and the largest difference between
their values (exact measures should agree to rounding). Each case runs in a fresh
process, so that the lattices cached by one case do not speed up the next.

Developer tool; not imported by ``dit``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

# measure -> options, keeping the optimization-based measures fast and reproducible
MEASURES = {
    "PID_WB": {},
    "PID_MMI": {},
    "PID_PM": {},
    "PID_SX": {},
    "PID_CCS": {},
    "PID_GK": {},
    "PID_RR": {},
    "PID_CT": {},
    "PID_Do": {},
    "PID_MES": {},
    "PID_IG": {},
    "PID_RAV": {},
    "PID_BROJA": {},
    "PID_dep": {},
    "PID_RA": {},
    "PID_Proj": {},
    "PID_Deg": {},
    "PID_Prec": {},
    "PID_GH": {},
    "PID_CI": {},
    "PID_RDR": {},
    "PID_Delta": {},
    "PID_MC": {"niter": 3},
    "PID_LN": {"niter": 3},
    "PID_IPID": {"niter": 2},
    "PID_DeltaLambda": {"niter": 2},
}

# measures too slow, or undefined, for some numbers of sources
SKIP = {
    3: {"PID_RAV", "PID_Proj", "PID_IPID", "PID_DeltaLambda", "PID_MC"},
    4: {
        "PID_RAV",
        "PID_Proj",
        "PID_IPID",
        "PID_DeltaLambda",
        "PID_MC",
        "PID_LN",
        "PID_GH",
        "PID_Delta",
        "PID_dep",
        "PID_RA",
        "PID_CCS",
        "PID_BROJA",
        "PID_Deg",
        "PID_Prec",
        "PID_PM",
        "PID_SX",
        "PID_CI",
        "PID_RDR",
    },
}


# I_RAV searches the partitions of the outcomes, too many for three ternary variables
SKIP_CASES = {("PID_RAV", "random 3x3x3")}


def _cases(n_sources):
    """
    The distributions with the given number of sources, by name.
    """
    import numpy as np

    from dit import Distribution
    from dit.pid.distributions import bivariates, trivariates

    def random(seed, shape):
        rng = np.random.default_rng(seed)
        pmf = rng.dirichlet(np.ones(int(np.prod(shape))))
        outcomes = ["".join(map(str, o)) for o in np.ndindex(*shape)]
        return Distribution(outcomes, pmf)

    if n_sources == 2:
        return {
            "and": bivariates["and"],
            "reduced or": bivariates["reduced or"],
            "sum": bivariates["sum"],
            "random 3x3x3": random(0, (3, 3, 3)),
        }
    if n_sources == 3:
        return {
            "anddup": trivariates["anddup"],
            "max min": trivariates["max min"],
            "random 2x2x2x2": random(1, (2, 2, 2, 2)),
        }
    if n_sources == 4:
        outcomes = ["".join(map(str, o)) + str(sum(o) % 2) for o in np.ndindex(2, 2, 2, 2)]
        return {"parity": Distribution(outcomes, [1 / 16] * 16)}
    raise ValueError(n_sources)


def _value(x):
    x = float(x)
    return None if math.isnan(x) else x


def run_case(measure, n_sources, case, repeat):
    """
    Time a measure on one case; return the best time and the decomposition's values.
    """
    import numpy as np

    import dit.pid

    cls = getattr(dit.pid, measure)
    d = _cases(n_sources)[case]
    options = MEASURES[measure]
    times = []
    for _ in range(repeat):
        np.random.seed(0)
        start = time.perf_counter()
        pid = cls(d, **options)
        reds = {str(node): _value(pid.get_red(node)) for node in pid._lattice}
        pis = {str(node): _value(pid.get_pi(node)) for node in pid._lattice}
        times.append(time.perf_counter() - start)
    return {"time": min(times), "reds": reds, "pis": pis}


def _run_isolated(measure, n_sources, case, repeat, timeout, pythonpath):
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    here = os.path.dirname(os.path.abspath(__file__))
    code = (
        f"import json, sys; sys.path.insert(0, {here!r}); import pid_bench; "
        f"print(json.dumps(pid_bench.run_case({measure!r}, {n_sources}, {case!r}, {repeat})))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, env=env, check=False
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    if proc.returncode:
        return {"error": (proc.stderr.strip().splitlines() or ["?"])[-1]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _max_difference(a, b):
    worst = 0.0
    for key in ("reds", "pis"):
        for node, x in a[key].items():
            y = b[key].get(node)
            if (x is None) != (y is None):
                return math.inf
            if x is not None:
                worst = max(worst, abs(x - y))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--measures", nargs="+", default=list(MEASURES))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--json", help="save the results to this file")
    parser.add_argument("--compare", help="compare with the results saved in this file")
    parser.add_argument("--pythonpath", help="run another version of dit, e.g. a checkout")
    args = parser.parse_args()

    baseline = {}
    if args.compare:
        with open(args.compare) as f:
            baseline = {tuple(json.loads(k)): v for k, v in json.load(f).items()}

    results = {}
    for n_sources in args.sources:
        for case in _cases(n_sources):
            for measure in args.measures:
                if measure in SKIP.get(n_sources, ()) or (measure, case) in SKIP_CASES:
                    continue
                result = _run_isolated(measure, n_sources, case, args.repeat, args.timeout, args.pythonpath)
                results[(measure, n_sources, case)] = result
                line = f"{measure:16s} {n_sources} {case:16s} "
                if "error" in result:
                    line += result["error"]
                else:
                    line += f"{result['time']:9.4f}s"
                    old = baseline.get((measure, n_sources, case))
                    if old and "error" not in old:
                        line += f"  x{old['time'] / result['time']:6.2f}  max|diff| {_max_difference(result, old):.1e}"
                    elif old:
                        line += f"  (baseline: {old['error']})"
                print(line, flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({json.dumps(list(k)): v for k, v in results.items()}, f, indent=1)


if __name__ == "__main__":
    main()
