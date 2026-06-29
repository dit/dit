"""
Tests for finite-alphabet information bottleneck algorithms.
"""

from itertools import pairwise

import numpy as np
import pytest

from dit.exceptions import ditException
from dit.rate_distortion.bottleneck_algorithms import agglomerative_ib, bottleneck_result, sequential_ib


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
