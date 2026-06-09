"""
Fast tests for dit.algorithms.maxentropy.

These exercise the (non-optimizing) constraint-matrix builders, the moment
machinery, the rank helpers, and the construction of MomentMaximumEntropy.
The full ``moment_maxent_dists`` / ``marginal_maxent_dists`` optimizations are
slow and covered elsewhere; here we only confirm the supporting code paths run
and are syntactically correct.
"""

import numpy as np
import pytest

import dit
from dit.algorithms.maxentropy import (
    MomentMaximumEntropy,
    ising_constraint_rank,
    marginal_constraint_rank,
    marginal_constraints,
    marginal_constraints_generic,
    moment,
    moment_constraint_rank,
    moment_constraints,
    negentropy,
)
from dit.algorithms.optutil import prepare_dist


@pytest.fixture
def d3():
    """A prepared (dense, linear) uniform 3-bit distribution."""
    return prepare_dist(dit.uniform_distribution(3, 2))


# ── marginal constraint builders ─────────────────────────────────────────


def test_marginal_constraints_with_normalization(d3):
    """With normalization, the first row is the all-ones constraint."""
    A, b = marginal_constraints_generic(d3, [[0, 1], [2]], with_normalization=True)
    assert np.allclose(A[0], np.ones(8))
    assert b[0] == 1


def test_marginal_constraints_without_normalization(d3):
    """Without normalization, the all-ones row is omitted (line 165->170)."""
    A_norm, _ = marginal_constraints_generic(d3, [[0, 1], [2]], with_normalization=True)
    A_no, _ = marginal_constraints_generic(d3, [[0, 1], [2]], with_normalization=False)
    assert A_no.shape[0] == A_norm.shape[0] - 1
    assert not np.allclose(A_no[0], np.ones(8))


def test_marginal_constraints_too_many_ways(d3):
    """Constraining more-way marginals than variables raises ValueError."""
    with pytest.raises(ValueError, match="Cannot constrain"):
        marginal_constraints(d3, 4)


def test_marginal_constraints_uses_rv_names(d3):
    """When the distribution has named rvs, they are used to build marginals."""
    d3.set_rv_names("XYZ")
    A, b = marginal_constraints(d3, 1)
    # normalization + 3 variables * 2 symbols = 7 rows
    assert A.shape == (7, 8)


def test_marginal_constraint_rank(d3):
    """The 1-way marginal model on 3 bits has rank 4 (incl. normalization)."""
    assert marginal_constraint_rank(d3, 1) == 4


# ── moments ──────────────────────────────────────────────────────────────


def test_moment_first():
    """The first moment of a uniform distribution over {0,1,2,3}."""
    f = np.array([0.0, 1.0, 2.0, 3.0])
    pmf = np.array([0.25] * 4)
    assert moment(f, pmf, n=1) == pytest.approx(1.5)


def test_moment_centered():
    """A centered second moment is the variance for a uniform pmf."""
    f = np.array([0.0, 1.0, 2.0, 3.0])
    pmf = np.array([0.25] * 4)
    assert moment(f, pmf, center=1.5, n=2) == pytest.approx(1.25)


def test_moment_constraints_scalar_m(d3):
    """Moment constraints for a scalar m build a finite matrix."""
    A, b = moment_constraints(d3.pmf, 3, 2, [-1, 1])
    assert A.shape[1] == 8
    assert np.isfinite(A).all()
    assert np.isfinite(b).all()


def test_moment_constraints_list_m(d3):
    """A list of m values is accepted (the ``mvals = m`` branch)."""
    A, b = moment_constraints(d3.pmf, 3, [1, 2], [-1, 1])
    assert A.shape[1] == 8


def test_moment_constraints_without_replacement(d3):
    """Selecting moments without replacement omits self-terms like <xx>."""
    A_with, _ = moment_constraints(d3.pmf, 3, 2, [-1, 1], with_replacement=True)
    A_without, _ = moment_constraints(d3.pmf, 3, 2, [-1, 1], with_replacement=False)
    assert A_without.shape[0] < A_with.shape[0]


def test_moment_constraints_bad_length():
    """A pmf whose length disagrees with the sample space raises ValueError."""
    with pytest.raises(ValueError, match="Length of"):
        moment_constraints(np.array([0.5, 0.5]), 3, 2, [-1, 1])


def test_moment_constraint_rank(d3):
    """The cumulative moment constraint rank is a positive integer."""
    assert int(moment_constraint_rank(d3, 2)) > 0


def test_ising_constraint_rank(d3):
    """Ising (no-replacement) constraint rank is computed."""
    assert int(ising_constraint_rank(d3, 2)) > 0


# ── negentropy ───────────────────────────────────────────────────────────


def test_negentropy_uniform():
    """Negentropy of a fair coin is -1 bit."""
    assert negentropy(np.array([0.5, 0.5])) == pytest.approx(-1.0)


def test_negentropy_deterministic():
    """Negentropy of a point mass is 0 (the ``nansum`` ignores the 0*log0)."""
    assert negentropy(np.array([1.0, 0.0])) == pytest.approx(0.0)


# ── MomentMaximumEntropy construction (no optimize) ──────────────────────


def test_moment_maximum_entropy_construction():
    """Constructing the optimizer builds its equality constraints (A, b)."""
    d = prepare_dist(dit.uniform_distribution(2, 2))
    opt = MomentMaximumEntropy(d, 2, [-1, 1])
    assert opt.A is not None
    assert opt.b is not None
