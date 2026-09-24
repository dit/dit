"""
Tests for dit.algorithms.support_spectrum
"""

import numpy as np
import pytest

import dit
from dit.algorithms import (
    spectral_entanglement_bound,
    support_biadjacency,
    support_singular_values,
)
from dit.divergences import maximum_correlation
from dit.multivariate import entropy, total_correlation


def giant_bit():
    """A perfect matching on two vertices per side."""
    return dit.Distribution(["00", "11"], [0.5, 0.5])


def cycle(n=4):
    """The uniform distribution on a 2n-cycle; biregular of degree two."""
    outcomes = [f"{i}{j}" for i in range(n) for j in (i, (i + 1) % n)]
    return dit.Distribution(outcomes, [1 / (2 * n)] * (2 * n))


def fano():
    """Uniform on the Fano plane's incidences; biregular of degree three."""
    lines = ["012", "034", "056", "136", "145", "235", "246"]
    return dit.Distribution([p + str(i) for i, line in enumerate(lines) for p in line], [1 / 21] * 21)


def complete():
    """The uniform distribution on a complete bipartite support."""
    return dit.Distribution(["00", "01", "10", "11"], [0.25] * 4)


def test_biadjacency_is_zero_one():
    """The biadjacency matrix marks the support."""
    matrix = support_biadjacency(giant_bit())
    np.testing.assert_array_equal(matrix, np.eye(2))


def test_fano_is_biregular_of_degree_three():
    """The Fano incidence graph has top singular value three."""
    assert support_singular_values(fano())[0] == pytest.approx(3.0)


@pytest.mark.parametrize("dist", [giant_bit(), cycle(), cycle(5), fano(), complete()])
def test_top_singular_value_is_the_half_profile(dist):
    """For uniform biregular supports, log lambda_1 = H[XY] - (H[X] + H[Y]) / 2."""
    sigmas = support_singular_values(dist)
    expected = float(entropy(dist)) - (float(entropy(dist, [0])) + float(entropy(dist, [1]))) / 2

    assert np.log2(sigmas[0]) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("dist", [giant_bit(), cycle(), cycle(5), fano(), complete()])
def test_spectral_gap_is_the_maximal_correlation(dist):
    """For uniform biregular supports, lambda_2 / lambda_1 = rho_m."""
    sigmas = support_singular_values(dist)

    assert sigmas[1] / sigmas[0] == pytest.approx(float(maximum_correlation(dist)), abs=1e-9)


def test_identities_fail_off_biregular_supports():
    """Both identities are specific to biregular supports."""
    d = dit.Distribution(["00", "01", "10"], [1 / 3] * 3)
    sigmas = support_singular_values(d)

    half_profile = float(entropy(d)) - (float(entropy(d, [0])) + float(entropy(d, [1]))) / 2
    assert np.log2(sigmas[0]) != pytest.approx(half_profile, abs=1e-3)
    assert sigmas[1] / sigmas[0] != pytest.approx(float(maximum_correlation(d)), abs=1e-3)


def test_bound_is_capped_by_the_mutual_information():
    """The nominal bound never exceeds I, and a full gap returns I."""
    for dist in (giant_bit(), cycle(), fano(), complete()):
        assert spectral_entanglement_bound(dist) <= float(total_correlation(dist)) + 1e-9

    # A complete bipartite support has a single nonzero singular value.
    assert spectral_entanglement_bound(complete()) == pytest.approx(float(total_correlation(complete())))


def test_bound_is_not_a_pointwise_certificate():
    """The suppressed slop dominates at small alphabets.

    The 8-cycle has an entanglement of roughly 0.67 bits against a nominal
    bound of 1.00, so the bound as stated is asymptotic only. This test pins
    that fact so nobody promotes the nominal term to a guarantee.
    """
    assert spectral_entanglement_bound(cycle()) == pytest.approx(1.0, abs=1e-9)


def test_requires_a_pair():
    """The support spectrum is a property of a bipartite graph."""
    with pytest.raises(dit.exceptions.ditException, match="2 variables"):
        support_singular_values(dit.example_dists.Xor())
