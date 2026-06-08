"""
Tests for the ``parallel_sweep`` helper and the parallel/serial parity of the
sweeps that use it. Parallelism is opt-in via ``DIT_OPT_JOBS`` and must never
change the value an optimizer returns: the serial and parallel paths consume
the same deterministic per-task generators, so a sweep is reproducible with
respect to a seeded ``np.random`` regardless of the worker count.
"""

import os
from contextlib import contextmanager

import numpy as np
import pytest

from dit import Distribution
from dit.algorithms.optimization import _resolve_n_jobs, parallel_sweep


@contextmanager
def _opt_jobs(value):
    """Temporarily set ``DIT_OPT_JOBS`` (restoring the prior value on exit)."""
    prev = os.environ.get("DIT_OPT_JOBS")
    if value is None:
        os.environ.pop("DIT_OPT_JOBS", None)
    else:
        os.environ["DIT_OPT_JOBS"] = str(value)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("DIT_OPT_JOBS", None)
        else:
            os.environ["DIT_OPT_JOBS"] = prev


def _sweep(seed, jobs):
    """Run a fixed stochastic sweep under a given ``DIT_OPT_JOBS`` setting."""
    np.random.seed(seed)
    with _opt_jobs(jobs):
        return parallel_sweep(lambda item, rng: (item, float(rng.random())), list(range(8)))


def test_resolve_n_jobs_serial():
    with _opt_jobs(1):
        assert _resolve_n_jobs(8) == 1


def test_resolve_n_jobs_explicit_cap():
    with _opt_jobs(4):
        assert _resolve_n_jobs(8) == 4
        assert _resolve_n_jobs(2) == 2  # capped at n_tasks


def test_parallel_sweep_preserves_order():
    with _opt_jobs(4):
        out = parallel_sweep(lambda item, rng: item * item, list(range(8)))
    assert out == [i * i for i in range(8)]


def test_parallel_sweep_empty():
    assert parallel_sweep(lambda item, rng: item, []) == []


def test_parallel_sweep_serial_parallel_parity():
    """Same seed ⇒ identical per-task draws whether serial or parallel."""
    serial = _sweep(1234, 1)
    parallel = _sweep(1234, 4)
    assert serial == parallel


def test_parallel_sweep_nesting_guard():
    """A sweep launched inside a sweep collapses to serial (no oversubscribe)."""

    def outer(item, rng):
        # Inside a worker, _resolve_n_jobs must report serial regardless of jobs.
        return _resolve_n_jobs(8)

    with _opt_jobs(4):
        assert parallel_sweep(outer, list(range(4))) == [1, 1, 1, 1]


@pytest.mark.parametrize("jobs", [4])
def test_ibroja_3source_parallel_parity(jobs):
    """
    PID_BROJA's parallelized per-source unique-information sweep (the >2-source
    branch in ``_measure``) must return the same values whether run serially or
    across worker threads.
    """
    from dit.pid.measures.ibroja import PID_BROJA

    d = Distribution(
        ["0000", "0111", "1011", "1101", "1110", "0001", "0010", "0100"],
        [0.15, 0.15, 0.15, 0.15, 0.15, 0.0833, 0.0833, 0.0834],
    )
    sources, target = [(0,), (1,), (2,)], (3,)

    np.random.seed(0)
    with _opt_jobs(1):
        ref = PID_BROJA._measure(d, sources, target)
    np.random.seed(0)
    with _opt_jobs(jobs):
        par = PID_BROJA._measure(d, sources, target)

    for source in sources:
        assert np.isclose(par[source], ref[source], atol=1e-6)
