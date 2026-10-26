"""
Tests for dit.example_dists.nonsignalling_boxes.
"""

from itertools import product

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dit import Distribution
from dit.example_dists import (
    chsh_value,
    correlated_pr_box,
    is_nonsignalling,
    pr_box,
    pr_box_d,
    tripartite_pr_box,
)
from dit.exceptions import ditException

unit = st.floats(min_value=0, max_value=1)


def test_pr_1():
    """
    Test
    """
    d1 = Distribution(list(product([0, 1], repeat=4)), [1 / 16] * 16)
    d2 = pr_box(0.0)
    assert d1.is_approx_equal(d2)


def test_pr_2():
    """
    Test
    """
    d1 = Distribution(
        [
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 1, 0, 0),
            (0, 1, 1, 1),
            (1, 0, 0, 0),
            (1, 0, 1, 1),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
        ],
        [1 / 8] * 8,
    )
    d2 = pr_box(1.0)
    assert d1.is_approx_equal(d2)


@pytest.mark.parametrize(
    ("box", "names"),
    [
        (pr_box, ("X", "Y", "A", "B")),
        (lambda **kw: pr_box_d(3, **kw), ("X", "Y", "A", "B")),
        (tripartite_pr_box, ("X", "Y", "Z", "A", "B", "C")),
        (lambda **kw: correlated_pr_box(0.5, **kw), ("X", "Y", "A", "B")),
    ],
)
def test_names(box, names):
    """
    Boxes are named by default and unnamed with name=False.
    """
    assert box().get_rv_names() == names
    assert box(name=False).get_rv_names() is None


@given(eta=unit)
def test_pr_box_d_2(eta):
    """
    The 2-output box is the PR box.
    """
    assert pr_box_d(2, eta).is_approx_equal(pr_box(eta))


@pytest.mark.parametrize(
    ("box", "size"),
    [
        (pr_box(), 8),
        (pr_box_d(3), 12),
        (pr_box_d(5), 20),
        (tripartite_pr_box(), 32),
    ],
)
def test_support(box, size):
    """
    Size of the support of the ideal boxes.
    """
    assert sum(p > 0 for p in box.copy(base="linear").pmf) == size


@given(eta=unit)
def test_chsh_pr_box(eta):
    """
    CHSH(pr_box(eta)) = 4 eta.
    """
    assert chsh_value(pr_box(eta)) == pytest.approx(4 * eta)


@given(eps=unit)
def test_chsh_correlated(eps):
    """
    CHSH(correlated_pr_box(eps)) = 2 (1 + eps).
    """
    assert chsh_value(correlated_pr_box(eps), inputs="XY", outputs="AB") == pytest.approx(2 * (1 + eps))


def test_chsh_non_binary():
    """
    CHSH requires binary variables.
    """
    with pytest.raises(ditException):
        chsh_value(pr_box_d(3))


@pytest.mark.parametrize(
    ("box", "inputs", "outputs"),
    [
        (pr_box(0.3), "XY", "AB"),
        (pr_box_d(4, 0.6), "XY", "AB"),
        (tripartite_pr_box(0.8), "XYZ", "ABC"),
        (correlated_pr_box(0.4), (0, 1), (2, 3)),
    ],
)
def test_nonsignalling(box, inputs, outputs):
    """
    All of the boxes are non-signalling.
    """
    assert is_nonsignalling(box, inputs, outputs)


def test_signalling():
    """
    A box where Alice outputs Bob's input signals.
    """
    d = Distribution([(x, y, y, 0) for x, y in product([0, 1], repeat=2)], [1 / 4] * 4)
    assert not is_nonsignalling(d, (0, 1), (2, 3))
