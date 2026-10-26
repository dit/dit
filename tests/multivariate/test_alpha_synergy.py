"""
Tests for dit.multivariate.alpha_synergy.
"""

from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.example_dists import Xor
from dit.multivariate import (
    alpha_synergistic_entropy,
    coinformation,
    entropy,
    partial_entropy_spectrum,
    partial_information_spectrum,
    partial_kullback_leibler_spectrum,
    partial_total_correlation_spectrum,
    total_correlation,
)
from dit.utils.testing import distributions


def _system():
    """
    The distributions used to generate the Syntropy references.
    """
    rng = np.random.default_rng(9)
    outcomes = ["".join(map(str, s)) for s in product([0, 1], [0, 1, 2], [0, 1])]
    w, w2 = rng.random(12), rng.random(12)
    return D(outcomes, w / w.sum()), D(outcomes, w2 / w2.sum())


def test_xor():
    """
    Removing any one variable of XOR loses nothing; removing two loses a bit.
    """
    d = Xor()
    assert partial_entropy_spectrum(d) == pytest.approx([0, 1, 1])
    assert alpha_synergistic_entropy(d, 1) == pytest.approx(0)
    assert alpha_synergistic_entropy(d, 3) == pytest.approx(2)


def test_redundant_copy():
    d = D(["000", "111"], [1 / 2] * 2)
    assert partial_entropy_spectrum(d) == pytest.approx([0, 0, 1])


def test_matches_syntropy():
    """
    References from Syntropy's ``partial_*_spectra`` (averaged where Syntropy is local).
    """
    d, prior = _system()
    assert partial_entropy_spectrum(d) == pytest.approx([0.4988370424417624, 1.2415328479673184, 1.576040022840798])
    assert partial_kullback_leibler_spectrum(d, prior) == pytest.approx(
        [0.7467452162358458, 0.30549642699343676, 0.10319179051516587]
    )
    assert partial_total_correlation_spectrum(d) == pytest.approx(
        [0.34565447880899713, -0.13322693699478866, 0.0], abs=1e-12
    )
    assert partial_information_spectrum(d, [[0], [1]], [2]) == pytest.approx(
        [0.11131963047517986, 0.008301783974340837]
    )


@pytest.mark.parametrize("definition", ["min", "max", "avg"])
def test_definitions_sum_to_entropy(definition):
    d, _ = _system()
    assert partial_entropy_spectrum(d, definition=definition).sum() == pytest.approx(entropy(d))


def test_errors():
    d = Xor()
    with pytest.raises(ValueError):
        alpha_synergistic_entropy(d, 0)
    with pytest.raises(ValueError):
        partial_entropy_spectrum(d, definition="median")
    with pytest.raises(ValueError):
        partial_kullback_leibler_spectrum(d, D(["000", "111"], [1 / 2] * 2))


@pytest.mark.parametrize("tiny", [2.5e-226, 3.6e-308, 4.9e-309])
def test_spectra_tiny_probabilities(tiny):
    """
    Near-underflow probabilities neither overflow the local spectra nor drop
    out of the product of the marginals.
    """
    d = D(["010", "111", "100"], [1 - 2 * tiny, tiny, tiny])
    assert partial_entropy_spectrum(d).sum() == pytest.approx(entropy(d), abs=1e-6)
    assert partial_total_correlation_spectrum(d).sum() == pytest.approx(total_correlation(d), abs=1e-6)


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_spectra_sum_to_parent(dist):
    assert partial_entropy_spectrum(dist).sum() == pytest.approx(entropy(dist), abs=1e-6)
    assert partial_total_correlation_spectrum(dist).sum() == pytest.approx(total_correlation(dist), abs=1e-6)
    mi = coinformation(dist, [[0, 1], [2]])
    assert partial_information_spectrum(dist, [[0], [1]], [2]).sum() == pytest.approx(mi, abs=1e-6)
