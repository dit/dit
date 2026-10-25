"""
Tests for dit.pid.gid and dit.pid.helpers.representational_complexity.
"""

from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.distconst import product_distribution
from dit.divergences import kullback_leibler_divergence
from dit.example_dists import Xor
from dit.multivariate import total_correlation
from dit.pid import GID, PED_SX, PID_MMI, PID_SX
from dit.pid.distributions import bivariates
from dit.pid.helpers import representational_complexity
from dit.utils.testing import distributions


def _pair():
    """
    The posterior/prior pair used to generate the Syntropy references.
    """
    rng = np.random.default_rng(21)
    outcomes = ["".join(map(str, s)) for s in product([0, 1], repeat=3)]
    w, w2 = rng.random(8), rng.random(8)
    return D(outcomes, w / w.sum()), D(outcomes, w2 / w2.sum())


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("sx", {((0,), (1,), (2,)): -0.02757588556891038, ((0, 1, 2),): 0.3484310690104354}),
        ("mos", {((0,), (1,), (2,)): 0.23234162365734734, ((0, 1, 2),): 0.5838758754778376}),
    ],
)
def test_matches_syntropy(measure, expected):
    """
    References from Syntropy's ``generalized_information_decomposition(post, prior, ...)``.
    """
    gid = GID(*_pair(), measure=measure)
    for node, value in expected.items():
        assert gid[node] == pytest.approx(value, abs=1e-9)


def test_atoms_sum_to_divergence():
    post, prior = _pair()
    gid = GID(post, prior)
    assert sum(gid[n] for n in gid._lattice) == pytest.approx(kullback_leibler_divergence(post, prior))
    top = ((0, 1, 2),)
    assert gid.get_red(top) == pytest.approx(kullback_leibler_divergence(post, prior))
    assert "GID (sx)" in str(gid)


def test_total_correlation_prior():
    d = Xor()
    gid = GID(d, product_distribution(d))
    assert sum(gid[n] for n in gid._lattice) == pytest.approx(total_correlation(d))


def test_local_atoms_average():
    post, prior = _pair()
    gid = GID(post, prior)
    for node in gid._lattice:
        assert sum(post[o] * v for o, v in gid.get_pw_pi(node).items()) == pytest.approx(gid[node])


def test_errors():
    post, prior = _pair()
    with pytest.raises(ValueError):
        GID(post, prior, measure="nope")
    with pytest.raises(ValueError):
        GID(post, D(["000", "111"], [1 / 2] * 2))


def test_representational_complexity():
    assert representational_complexity(PID_SX(bivariates["redundant"])) == pytest.approx(1.0)
    ped = PED_SX(Xor())
    # Syntropy's representational_complexity on the same h_sx PED
    assert representational_complexity(ped) == pytest.approx(1.1225562489182657)
    assert representational_complexity(ped, max) == pytest.approx(1.3774437510817343)
    with pytest.raises(ValueError):
        representational_complexity(ped, sum)


def test_complexity_counts_sources():
    """
    A source made of two variables has degree 1, not 2.
    """
    outcomes = [f"{a}{b}{c}{a ^ b}" for a, b, c in product([0, 1], repeat=3)]
    d = D(outcomes, [1 / 8] * 8)
    pid = PID_MMI(d, sources=[[0, 1], [2]], target=[3])
    assert representational_complexity(pid) == pytest.approx(1.0)


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_gid_of_itself_vanishes(dist):
    gid = GID(dist, dist)
    assert all(gid[n] == pytest.approx(0, abs=1e-9) for n in gid._lattice)
