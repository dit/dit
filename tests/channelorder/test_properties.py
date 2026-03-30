"""
Hypothesis property tests for channel orderings and deficiencies.

These test mathematical invariants that must hold for *all* channels,
not just specific examples.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume

from dit.channelorder.deficiency import (
    le_cam_deficiency,
    le_cam_distance,
    weighted_le_cam_deficiency,
    weighted_output_kl_deficiency,
)
from dit.channelorder.orderings import (
    is_less_noisy,
    is_more_capable,
    is_output_degraded,
    is_shannon_included,
)
from dit.utils.testing import channel_pairs, degraded_channel_pairs, markov_chains

epsilon = 1e-4


# ── Order hierarchy on degraded pairs ──────────────────────────────────────


@given(pair=degraded_channel_pairs(input_size=(2, 3), output_z_size=(2, 3), output_y_size=(2, 3)))
@settings(max_examples=30)
def test_degraded_implies_output_degraded(pair):
    """Constructed degraded pairs should be detected as output-degraded."""
    mu, kappa = pair
    assert is_output_degraded(mu, kappa)


@given(pair=degraded_channel_pairs(input_size=(2, 3), output_z_size=(2, 3), output_y_size=(2, 3)))
@settings(max_examples=20)
def test_degraded_implies_more_capable(pair):
    """Output-degraded => more capable (Proposition 3)."""
    mu, kappa = pair
    assert is_more_capable(mu, kappa)


@given(pair=degraded_channel_pairs(input_size=(2, 3), output_z_size=(2, 3), output_y_size=(2, 3)))
@settings(max_examples=20)
def test_degraded_implies_less_noisy(pair):
    """Output-degraded => less noisy (Proposition 3)."""
    mu, kappa = pair
    assert is_less_noisy(mu, kappa)


@given(pair=degraded_channel_pairs(input_size=2, output_z_size=2, output_y_size=2))
@settings(max_examples=20)
def test_degraded_implies_shannon(pair):
    """Output-degraded => Shannon inclusion (same input alphabet)."""
    mu, kappa = pair
    assert is_shannon_included(mu, kappa)


# ── Order hierarchy implication chain on random pairs ──────────────────────


@given(pair=channel_pairs(input_size=(2, 3), output_y_size=(2, 3), output_z_size=(2, 3)))
@settings(max_examples=30)
def test_output_degraded_implies_less_noisy(pair):
    """If output-degraded holds, less noisy must also hold."""
    mu, kappa = pair
    if is_output_degraded(mu, kappa):
        assert is_less_noisy(mu, kappa)


@given(pair=channel_pairs(input_size=(2, 3), output_y_size=(2, 3), output_z_size=(2, 3)))
@settings(max_examples=30)
def test_less_noisy_implies_more_capable(pair):
    """If less noisy holds, more capable must also hold."""
    mu, kappa = pair
    if is_less_noisy(mu, kappa):
        assert is_more_capable(mu, kappa)


# ── Deficiency non-negativity ──────────────────────────────────────────────


@given(pair=channel_pairs(input_size=(2, 3), output_y_size=(2, 3), output_z_size=(2, 3)))
@settings(max_examples=30)
def test_le_cam_deficiency_nonneg(pair):
    """Le Cam deficiency is always >= 0."""
    mu, kappa = pair
    d = le_cam_deficiency(mu, kappa)
    assert d >= -epsilon


@given(pair=channel_pairs(input_size=(2, 3), output_y_size=(2, 3), output_z_size=(2, 3)))
@settings(max_examples=30)
def test_le_cam_distance_nonneg(pair):
    """Le Cam distance is always >= 0."""
    mu, kappa = pair
    d = le_cam_distance(mu, kappa)
    assert d >= -epsilon


# ── Le Cam deficiency zero iff degraded (forward direction) ────────────────


@given(pair=degraded_channel_pairs(input_size=(2, 3), output_z_size=(2, 3), output_y_size=(2, 3)))
@settings(max_examples=30)
def test_deficiency_zero_when_degraded(pair):
    """If mu output-degrades to kappa, Le Cam deficiency is 0."""
    mu, kappa = pair
    d = le_cam_deficiency(mu, kappa)
    assert d < epsilon


# ── Le Cam distance symmetry ──────────────────────────────────────────────


@given(pair=channel_pairs(input_size=(2, 3), output_y_size=(2, 3), output_z_size=(2, 3)))
@settings(max_examples=20)
def test_le_cam_distance_symmetric(pair):
    """Le Cam distance is symmetric: Delta(mu, kappa) = Delta(kappa, mu)."""
    mu, kappa = pair
    # Only test when same output sizes (since distance needs same input)
    assume(mu.shape == kappa.shape)
    d1 = le_cam_distance(mu, kappa)
    d2 = le_cam_distance(kappa, mu)
    assert d1 == pytest.approx(d2, abs=1e-6)


# ── Le Cam distance triangle inequality ───────────────────────────────────


@given(
    pair1=channel_pairs(input_size=2, output_y_size=2, output_z_size=2),
    pair2=channel_pairs(input_size=2, output_y_size=2, output_z_size=2),
)
@settings(max_examples=20)
def test_le_cam_distance_triangle(pair1, pair2):
    """
    Triangle inequality: Delta(a, c) <= Delta(a, b) + Delta(b, c).
    """
    a = pair1[0]
    b = pair1[1]
    c = pair2[1]
    d_ab = le_cam_distance(a, b)
    d_bc = le_cam_distance(b, c)
    d_ac = le_cam_distance(a, c)
    assert d_ac <= d_ab + d_bc + epsilon


# ── Pinsker bound ─────────────────────────────────────────────────────────


@given(pair=channel_pairs(input_size=(2, 3), output_y_size=(2, 3), output_z_size=(2, 3)))
@settings(max_examples=20)
def test_pinsker_bound(pair):
    """
    Eq. 25: weighted_le_cam(mu,kappa,pi) <= sqrt(ln(2)/2 * weighted_output_kl(mu,kappa,pi)).
    """
    mu, kappa = pair
    n_s = mu.shape[0]
    pi = np.ones(n_s) / n_s  # uniform prior

    tv = weighted_le_cam_deficiency(mu, kappa, pi)
    kl = weighted_output_kl_deficiency(mu, kappa, pi)

    assert tv <= np.sqrt(np.log(2) / 2 * kl) + epsilon


# ── Markov chain => zero deficiency ───────────────────────────────────────


@given(dist=markov_chains(alphabets=((2, 3), (2, 3), (2, 3))))
@settings(max_examples=20)
def test_markov_chain_zero_deficiency(dist):
    """
    If S-Z-Y is a Markov chain, then P(Z|S) output-degrades to P(Y|S),
    so le_cam_deficiency(P(Z|S), P(Y|S)) = 0.
    """
    from dit.channelorder._utils import channels_from_joint

    dims = list(dist.dims)
    kappa, mu, pi_s = channels_from_joint(dist, [dims[0]], [dims[2]], [dims[1]])
    d = le_cam_deficiency(mu, kappa)
    assert d < epsilon


@given(dist=markov_chains(alphabets=((2, 3), (2, 3), (2, 3))))
@settings(max_examples=20)
def test_markov_chain_output_degraded(dist):
    """
    If S-Z-Y Markov, then Z is Blackwell-sufficient for Y w.r.t. S:
    is_output_degraded(P(Z|S), P(Y|S)) is True.
    """
    from dit.channelorder._utils import channels_from_joint

    dims = list(dist.dims)
    kappa, mu, pi_s = channels_from_joint(dist, [dims[0]], [dims[2]], [dims[1]])
    assert is_output_degraded(mu, kappa)
