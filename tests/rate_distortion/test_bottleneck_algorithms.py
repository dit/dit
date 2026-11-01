"""
Tests for finite-alphabet information bottleneck algorithms.
"""

from itertools import pairwise

import numpy as np
import pytest

from dit.exceptions import ditException
from dit.rate_distortion.bottleneck_algorithms import (
    _compact_assignments,
    _encoder_from_assignments,
    agglomerative_ib,
    bottleneck_result,
    greedy_privacy_funnel,
    sequential_ib,
)


def test_bottleneck_result_evaluates_encoder():
    """
    Test bottleneck result bookkeeping.
    """
    p_xy = np.asarray([[1 / 2, 0], [0, 1 / 2]])
    p_t_given_x = np.eye(2)
    result = bottleneck_result(p_xy, p_t_given_x, beta=1.0)

    assert result.p_xyt.shape == (2, 2, 2)
    assert result.active == 2
    assert result.complexity == pytest.approx(1.0)
    assert result.relevance == pytest.approx(1.0)
    assert result.distortion == pytest.approx(0.0)


def test_sequential_ib_returns_hard_encoder():
    """
    Test that sequential DIB returns a deterministic encoder.
    """
    p_xy = np.asarray([[1 / 2, 0], [0, 1 / 2]])
    result = sequential_ib(p_xy, beta=10.0, n_clusters=2, restarts=1)

    assert result.active == 2
    assert np.allclose(result.p_t_given_x.sum(axis=1), 1.0)
    assert np.all((result.p_t_given_x == 0.0) | (result.p_t_given_x == 1.0))
    assert result.complexity == pytest.approx(1.0)
    assert result.relevance == pytest.approx(1.0)


def test_sequential_ib_objective_history_is_nonincreasing():
    """
    Test that accepted coordinate moves do not increase the objective.
    """
    p_xy = np.asarray(
        [
            [1 / 3, 0, 0],
            [0, 1 / 3, 0],
            [0, 0, 1 / 3],
        ]
    )
    result = sequential_ib(p_xy, beta=4.0, n_clusters=3, restarts=1)

    assert all(a >= b for a, b in pairwise(result.history))


def test_agglomerative_ib_records_decreasing_active_alphabet():
    """
    Test that agglomerative DIB builds a merge path.
    """
    p_xy = np.asarray(
        [
            [1 / 3, 0, 0],
            [0, 1 / 3, 0],
            [0, 0, 1 / 3],
        ]
    )
    result = agglomerative_ib(p_xy, beta=1.0)

    assert result.active_history == (3, 2, 1)
    assert len(result.history) == 3


def test_sequential_ib_rejects_invalid_clusters():
    """
    Test validation.
    """
    p_xy = np.asarray([[1 / 2, 0], [0, 1 / 2]])

    with pytest.raises(ditException):
        sequential_ib(p_xy, beta=1.0, n_clusters=0)


# ── Input validation, hard-assignment helpers, and solver options ────────────


def _h2(p):
    """
    Binary entropy in bits.
    """
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


# X is uniform on three values and Y = X, so a hard T has I(T : Y) = H(T).
_COPY3 = np.eye(3) / 3

# p(y | x) is (2/3, 1/3) for both x = 0 and x = 1, and (1/2, 1/2) for x = 2.
_TWIN_ROWS = np.asarray([[0.2, 0.1], [0.4, 0.2], [0.05, 0.05]])

_ENTRY_POINTS = {
    "bottleneck_result": lambda p_xy: bottleneck_result(p_xy, np.eye(2), beta=1.0),
    "sequential_ib": lambda p_xy: sequential_ib(p_xy, beta=1.0),
    "agglomerative_ib": lambda p_xy: agglomerative_ib(p_xy, beta=1.0),
    "greedy_privacy_funnel": lambda p_xy: greedy_privacy_funnel(p_xy),
}


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
@pytest.mark.parametrize(
    ("p_xy", "match"),
    [
        (np.asarray([1 / 2, 1 / 2]), "p_xy must be a two-dimensional joint distribution"),
        (np.asarray([[np.inf, 1 / 2], [1 / 4, 1 / 4]]), "p_xy must contain only finite values"),
        (np.asarray([[-0.1, 0.6], [1 / 4, 1 / 4]]), "p_xy must be non-negative"),
        (np.zeros((2, 2)), "p_xy must have positive total mass"),
    ],
    ids=["one-dimensional", "infinite", "negative", "massless"],
)
def test_entry_points_reject_malformed_joints(entry, p_xy, match):
    """
    Test that every public solver validates p(x, y).
    """
    with pytest.raises(ditException, match=match):
        _ENTRY_POINTS[entry](p_xy)


@pytest.mark.parametrize(
    ("p_t_given_x", "match"),
    [
        (np.ones(2), r"p_t_given_x must have shape \(\|X\|, \|T\|\)"),
        (np.ones((3, 2)), r"p_t_given_x must have shape \(\|X\|, \|T\|\)"),
        (np.asarray([[1.5, -0.5], [0.0, 1.0]]), "p_t_given_x must contain finite non-negative values"),
        (np.asarray([[np.nan, 1.0], [0.0, 1.0]]), "p_t_given_x must contain finite non-negative values"),
        (np.asarray([[0.0, 0.0], [0.0, 1.0]]), "each row of p_t_given_x must have positive mass"),
    ],
    ids=["one-dimensional", "wrong-rows", "negative", "nan", "empty-row"],
)
def test_bottleneck_result_rejects_malformed_encoders(p_t_given_x, match):
    """
    Test encoder validation.
    """
    p_xy = np.asarray([[1 / 2, 0], [0, 1 / 2]])
    with pytest.raises(ditException, match=match):
        bottleneck_result(p_xy, p_t_given_x, beta=1.0)


def test_bottleneck_result_normalizes_inputs():
    """
    Test that an unnormalized joint and unnormalized encoder rows evaluate like their normalizations.
    """
    p_xy = np.asarray([[0.3, 0.1], [0.2, 0.4]])
    encoder = np.asarray([[0.9, 0.1], [0.25, 0.75]])
    reference = bottleneck_result(p_xy, encoder, beta=2.0)
    scaled = bottleneck_result(4 * p_xy, encoder * np.asarray([[3.0], [0.5]]), beta=2.0)

    np.testing.assert_allclose(scaled.p_t_given_x, encoder)
    np.testing.assert_allclose(scaled.p_xyt, reference.p_xyt)
    assert scaled.p_xyt.sum() == pytest.approx(1.0)
    assert scaled.objective == pytest.approx(reference.objective)


def test_compact_assignments_relabels_positive_mass_clusters():
    """
    Test that live clusters are relabeled contiguously, in label order, and massless ones fold into cluster 0.
    """
    p_x = np.asarray([0.5, 0.25, 0.25, 0.0])
    np.testing.assert_array_equal(_compact_assignments([3, 1, 1, 7], p_x), [1, 0, 0, 0])
    np.testing.assert_array_equal(_compact_assignments([2, 5, 5], np.zeros(3)), [0, 0, 0])


def test_encoder_from_assignments():
    """
    Test building deterministic encoders from hard assignments, and their validation.
    """
    np.testing.assert_array_equal(_encoder_from_assignments([0, 2, 0]), [[1, 0, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(_encoder_from_assignments([1, 0], n_clusters=3), [[0, 1, 0], [1, 0, 0]])

    with pytest.raises(ditException, match="assignments must be non-negative integers"):
        _encoder_from_assignments([0, -1])
    with pytest.raises(ditException, match="n_clusters must be positive"):
        _encoder_from_assignments([])
    with pytest.raises(ditException, match="n_clusters must be positive"):
        _encoder_from_assignments([0, 0], n_clusters=0)
    with pytest.raises(ditException, match="assignments contain a cluster outside n_clusters"):
        _encoder_from_assignments([0, 2], n_clusters=2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"beta": -1.0}, "beta must be non-negative"),
        ({"n_clusters": 3}, r"n_clusters must be between 1 and \|X\|"),
        ({"maxiter": -1}, "maxiter must be non-negative"),
        ({"restarts": -1}, "restarts must be non-negative"),
        ({"initial_assignments": [0, 1, 0]}, r"initial_assignments must have shape \(\|X\|,\)"),
        ({"initial_assignments": [0, -1]}, "initial_assignments must be non-negative"),
    ],
)
def test_sequential_ib_rejects_invalid_options(kwargs, match):
    """
    Test sequential DIB option validation.
    """
    p_xy = np.asarray([[1 / 2, 0], [0, 1 / 2]])
    with pytest.raises(ditException, match=match):
        sequential_ib(p_xy, **{"beta": 1.0, **kwargs})


def test_sequential_ib_defaults_to_one_cluster_per_symbol():
    """
    Test that n_clusters defaults to |X|: at large beta a copy of a uniform ternary X keeps all three states.
    """
    result = sequential_ib(_COPY3, beta=10.0, restarts=2, rng=0)

    assert result.active == 3
    assert result.complexity == pytest.approx(np.log2(3))
    assert result.distortion == pytest.approx(0.0, abs=1e-12)
    assert result.objective == pytest.approx(np.log2(3))


def test_sequential_ib_without_sweeps_returns_its_start():
    """
    Test that with no restarts the deterministic start x mod n_clusters is used, and maxiter=0 keeps it.
    """
    beta = 10.0
    result = sequential_ib(_COPY3, beta=beta, n_clusters=2, maxiter=0, restarts=0)

    np.testing.assert_array_equal(result.assignments, [0, 1, 0])
    assert result.iterations == 0
    assert not result.converged
    assert result.history == (result.objective,)
    assert result.objective == pytest.approx(_h2(1 / 3) + beta * (np.log2(3) - _h2(1 / 3)))


def test_sequential_ib_initial_assignments():
    """
    Test that initial assignments are reduced modulo n_clusters and seed the coordinate descent.
    """
    start = sequential_ib(_COPY3, beta=10.0, n_clusters=2, initial_assignments=[3, 4, 5], maxiter=0, restarts=0)
    np.testing.assert_array_equal(start.assignments, [1, 0, 1])

    # From a single cluster, one sweep separates every symbol, and a second sweep confirms the fixed point.
    one_sweep = sequential_ib(_COPY3, beta=10.0, initial_assignments=[0, 0, 0], maxiter=1, restarts=0)
    assert one_sweep.iterations == 1
    assert not one_sweep.converged
    assert one_sweep.active == 3
    assert one_sweep.history[0] == pytest.approx(10.0 * np.log2(3))
    assert one_sweep.objective == pytest.approx(np.log2(3))

    converged = sequential_ib(_COPY3, beta=10.0, initial_assignments=[0, 0, 0], maxiter=5, restarts=0)
    assert converged.iterations == 2
    assert converged.converged
    assert converged.history == pytest.approx([10.0 * np.log2(3), np.log2(3), np.log2(3)])


def test_hard_solvers_ignore_massless_symbols():
    """
    Test that a zero-probability x never occupies a bottleneck state of its own.
    """
    p_xy = np.asarray([[1 / 2, 0], [0, 1 / 2], [0, 0]])

    sequential = sequential_ib(p_xy, beta=10.0, restarts=1)
    np.testing.assert_array_equal(sequential.assignments, [0, 1, 0])
    assert sequential.p_t_given_x.shape == (3, 2)
    assert sequential.active == 2
    assert sequential.complexity == pytest.approx(1.0)
    assert sequential.relevance == pytest.approx(1.0)

    agglomerative = agglomerative_ib(p_xy, beta=10.0)
    assert agglomerative.active_history[0] == 2
    assert agglomerative.active == 2
    assert agglomerative.complexity == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"beta": -0.5}, "beta must be non-negative"),
        ({"n_clusters": 0}, r"n_clusters must be between 1 and \|X\|"),
        ({"n_clusters": 3}, r"n_clusters must be between 1 and \|X\|"),
    ],
)
def test_agglomerative_ib_rejects_invalid_options(kwargs, match):
    """
    Test agglomerative DIB option validation.
    """
    p_xy = np.asarray([[1 / 2, 0], [0, 1 / 2]])
    with pytest.raises(ditException, match=match):
        agglomerative_ib(p_xy, **{"beta": 1.0, **kwargs})


def test_agglomerative_ib_stops_at_target_clusters():
    """
    Test that a target n_clusters stops the merge path there, first merging the two x with equal p(y | x).
    """
    result = agglomerative_ib(_TWIN_ROWS, beta=1.0, n_clusters=2)

    assert result.merges == ((0, 1),)
    np.testing.assert_array_equal(result.assignments, [0, 0, 1])
    assert result.iterations == 1
    assert result.active_history == (3, 2)
    assert result.entropy == pytest.approx(_h2(0.1))
    assert result.distortion == pytest.approx(0.0, abs=1e-12)

    untouched = agglomerative_ib(_TWIN_ROWS, beta=1.0, n_clusters=3)
    assert untouched.merges == ()
    assert untouched.iterations == 0
    assert untouched.active == 3


def test_agglomerative_ib_target_overrides_best_on_path():
    """
    Test that a target n_clusters is honoured even where the best partition on the path is coarser.
    """
    best = agglomerative_ib(_TWIN_ROWS, beta=0.1)
    forced = agglomerative_ib(_TWIN_ROWS, beta=0.1, n_clusters=2)

    assert best.active == 1
    assert forced.active == 2
    assert forced.objective == pytest.approx(_h2(0.1))
    assert forced.objective > best.objective
    # Both follow the same greedy merges; the target only truncates the path.
    assert len(forced.history) == 2
    assert forced.history == pytest.approx(best.history[:2])
