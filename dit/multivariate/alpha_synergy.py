"""
The alpha-synergy ("synergy-first backbone") decomposition of Varley.

The :math:`\\alpha`-synergistic entropy of an outcome is the least information
that is lost when any :math:`\\alpha` of the variables are removed:

.. math::

    h^{syn}_{\\alpha}(x) = \\min_{|a| = \\alpha} h(x_a \\mid x_{-a}).

It grows with :math:`\\alpha` from 0 to :math:`h(x)` at :math:`\\alpha = n`, and
its increments :math:`h^{syn}_{\\alpha} - h^{syn}_{\\alpha - 1}` form the
*partial entropy spectrum*. Unlike lattice decompositions it needs only
:math:`n` terms, so it scales to many variables :cite:`varley2024scalable`.
"""

from itertools import combinations

import numpy as np

from ..helpers import normalize_rvs

__all__ = (
    "alpha_synergistic_entropy",
    "partial_entropy_spectrum",
    "partial_kullback_leibler_spectrum",
    "partial_total_correlation_spectrum",
    "partial_information_spectrum",
)

_COMBINE = {"min": np.min, "max": np.max, "avg": np.mean}


def _codes(dist, rvs):
    """
    The outcomes of positive probability, their probabilities, and an integer
    code per outcome for each variable in `rvs`.
    """
    d = dist.coalesce(rvs)
    outcomes = [o for o in d.outcomes if d[o] > 0]
    pmf = np.array([d[o] for o in outcomes])
    codes = np.empty((len(outcomes), len(rvs)), dtype=np.int64)
    for j in range(len(rvs)):
        index = {}
        codes[:, j] = [index.setdefault(o[j], len(index)) for o in outcomes]
    return outcomes, pmf, codes


def _marginal(pmf, codes, subset):
    """
    The probability of each outcome's restriction to the variables in `subset`.
    """
    if not subset:
        return np.ones(len(pmf))
    _, inverse = np.unique(codes[:, list(subset)], axis=0, return_inverse=True)
    inverse = inverse.ravel()
    return np.bincount(inverse, weights=pmf)[inverse]


def _spectra(log_marginal, n, definition):
    """
    The partial entropy spectrum of each outcome, shape ``(outcomes, n)``, in
    bits, given ``log_marginal(subset)``: the log2-probability of each
    outcome's restriction to `subset`.
    """
    combine = _COMBINE[definition]
    log_joint = log_marginal(tuple(range(n)))
    synergies = np.zeros((len(log_joint), n + 1))
    for alpha in range(1, n + 1):
        losses = [
            log_marginal(tuple(i for i in range(n) if i not in a)) - log_joint for a in combinations(range(n), alpha)
        ]
        synergies[:, alpha] = combine(losses, axis=0)
    return np.diff(synergies, axis=1)


def _local_spectra(pmf, codes, definition):
    """
    The partial entropy spectrum of each outcome, shape ``(outcomes, n)``, in bits.
    """
    return _spectra(lambda subset: np.log2(_marginal(pmf, codes, subset)), codes.shape[1], definition)


def _product_local_spectra(pmf, codes, definition):
    """
    The local spectra of the product of the marginals, at each outcome of the
    joint. Built from single-variable log-probabilities, since the product's
    probabilities can underflow to zero.
    """
    n = codes.shape[1]
    log_marginals = np.column_stack([np.log2(_marginal(pmf, codes, (i,))) for i in range(n)])
    return _spectra(lambda subset: log_marginals[:, list(subset)].sum(axis=1), n, definition)


def _check_definition(definition):
    if definition not in _COMBINE:
        raise ValueError(f"`definition` must be one of {sorted(_COMBINE)}.")


def alpha_synergistic_entropy(dist, alpha, rvs=None, definition="min"):
    """
    The :math:`\\alpha`-synergistic entropy :cite:`varley2024scalable`.

    .. math::

        \\operatorname{H}^{syn}_{\\alpha} = \\mathbb{E}\\left[\\min_{|a| = \\alpha} h(x_a \\mid x_{-a})\\right]

    Parameters
    ----------
    dist : Distribution
        The distribution.
    alpha : int
        The number of variables removed, from 1 to the number of variables.
    rvs : list, None
        The variables. If None, all of them.
    definition : {'min', 'max', 'avg'}
        How the losses of the :math:`\\binom{n}{\\alpha}` subsets are combined;
        ``'min'`` is Varley's definition.

    Returns
    -------
    h : float
        The expected :math:`\\alpha`-synergistic entropy, in bits.
    """
    _check_definition(definition)
    rvs, _ = normalize_rvs(dist, rvs, None)
    if not 1 <= alpha <= len(rvs):
        raise ValueError("`alpha` must be between 1 and the number of variables.")
    _, pmf, codes = _codes(dist, rvs)
    return float(pmf @ _local_spectra(pmf, codes, definition)[:, :alpha].sum(axis=1))


def partial_entropy_spectrum(dist, rvs=None, definition="min"):
    """
    The partial entropy spectrum :cite:`varley2024scalable`: the expected
    increments :math:`h^{syn}_{\\alpha} - h^{syn}_{\\alpha - 1}` for
    :math:`\\alpha = 1, \\ldots, n`. They sum to the joint entropy.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    rvs : list, None
        The variables. If None, all of them.
    definition : {'min', 'max', 'avg'}
        See :func:`alpha_synergistic_entropy`.

    Returns
    -------
    spectrum : ndarray
        The spectrum, in bits; entry ``k`` is :math:`\\alpha = k + 1`.
    """
    _check_definition(definition)
    rvs, _ = normalize_rvs(dist, rvs, None)
    _, pmf, codes = _codes(dist, rvs)
    return pmf @ _local_spectra(pmf, codes, definition)


def partial_kullback_leibler_spectrum(posterior, prior, rvs=None, definition="min"):
    """
    The partial spectrum of :math:`D_{KL}[\\text{posterior} \\| \\text{prior}]`
    :cite:`varley2024scalable`: the posterior expectation of the prior's local
    spectrum minus the posterior's. It sums to the divergence.

    Parameters
    ----------
    posterior, prior : Distribution
        The distributions; the posterior's support must be within the prior's.
    rvs : list, None
        The variables. If None, all of them.
    definition : {'min', 'max', 'avg'}
        See :func:`alpha_synergistic_entropy`.

    Returns
    -------
    spectrum : ndarray
        The spectrum, in bits.
    """
    _check_definition(definition)
    rvs, _ = normalize_rvs(posterior, rvs, None)
    post_outcomes, post_pmf, post_codes = _codes(posterior, rvs)
    prior_outcomes, prior_pmf, prior_codes = _codes(prior, rvs)
    position = {o: i for i, o in enumerate(prior_outcomes)}
    try:
        rows = [position[o] for o in post_outcomes]
    except KeyError:
        raise ValueError("The posterior's support must be contained in the prior's.") from None
    prior_spectra = _local_spectra(prior_pmf, prior_codes, definition)[rows]
    post_spectra = _local_spectra(post_pmf, post_codes, definition)
    return post_pmf @ (prior_spectra - post_spectra)


def partial_total_correlation_spectrum(dist, rvs=None, definition="min"):
    """
    The partial spectrum of the total correlation :cite:`varley2024scalable`,
    i.e. :func:`partial_kullback_leibler_spectrum` against the product of the
    marginals.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    rvs : list, None
        The variables. If None, all of them.
    definition : {'min', 'max', 'avg'}
        See :func:`alpha_synergistic_entropy`.

    Returns
    -------
    spectrum : ndarray
        The spectrum, in bits.
    """
    _check_definition(definition)
    rvs, _ = normalize_rvs(dist, rvs, None)
    _, pmf, codes = _codes(dist, rvs)
    return pmf @ (_product_local_spectra(pmf, codes, definition) - _local_spectra(pmf, codes, definition))


def partial_information_spectrum(dist, rvs, target, definition="min"):
    """
    The partial spectrum of :math:`I[X : Y]` for inputs :math:`X` = `rvs` and
    target :math:`Y` :cite:`varley2024scalable`: the expectation over :math:`y`
    of the input spectrum minus the spectrum of :math:`X \\mid Y = y`. It sums
    to the mutual information.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    rvs : list
        The input variables.
    target : list
        The target variable(s).
    definition : {'min', 'max', 'avg'}
        See :func:`alpha_synergistic_entropy`.

    Returns
    -------
    spectrum : ndarray
        The spectrum, in bits; entry ``k`` is :math:`\\alpha = k + 1`.
    """
    _check_definition(definition)
    rvs = [list(rv) for rv in rvs]
    d = dist.coalesce([*rvs, list(target)])
    _, pmf, codes = _codes(d, [[i] for i in range(len(rvs) + 1)])
    inputs, y = codes[:, :-1], codes[:, -1]
    # the local spectra of the input marginal, evaluated at each joint outcome
    _, x_index = np.unique(inputs, axis=0, return_inverse=True)
    x_index = x_index.ravel()
    p_x = np.bincount(x_index, weights=pmf)
    first = np.unique(x_index, return_index=True)[1]
    marginal_spectra = _local_spectra(p_x, inputs[first], definition)[x_index]
    spectrum = np.zeros(len(rvs))
    for value in np.unique(y):
        rows = y == value
        p_y = pmf[rows].sum()
        conditional = _local_spectra(pmf[rows] / p_y, inputs[rows], definition)
        spectrum += pmf[rows] @ (marginal_spectra[rows] - conditional)
    return spectrum
