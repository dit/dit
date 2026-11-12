r"""
A likelihood-ratio test for an interaction among all of several discrete variables.
"""

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import chi2

from ..algorithms.distribution_optimizers import maxent_dist
from ..distribution import Distribution
from ._symbols import UndersamplingWarning, as_generator

__all__ = (
    "ConnectedInformationTest",
    "connected_information_test",
)


@dataclass(frozen=True)
class ConnectedInformationTest:
    """
    The result of :func:`connected_information_test`.

    Attributes
    ----------
    order : int
        The number of variables, :math:`k`: the test is of their :math:`k`-th order
        connected information.
    statistic : str
        The name of the test statistic, ``'G2'``.
    value : float
        The observed :math:`G^2` statistic.
    pvalue : float
        The p-value of the observed statistic.
    dof : int
        Degrees of freedom of the asymptotic chi-squared distribution,
        :math:`\\prod_i (L_i - 1)` for the observed numbers of levels :math:`L_i`.
    connected_information : float
        The plug-in connected information :math:`\\hat{I}^{(k)}`, in bits;
        ``value = 2 * n_samples * ln(2) * connected_information``.
    n_samples : int
        The number of observations.
    min_expected : float
        The smallest positive expected count under the fitted null.
    null : np.ndarray or None
        The statistic of each bootstrap sample (bootstrap tests only).
    """

    order: int
    statistic: str
    value: float
    pvalue: float
    dof: int
    connected_information: float
    n_samples: int
    min_expected: float
    null: np.ndarray | None = None


def _fit(codes):
    """
    The observed cells of the rows of `codes` with their counts, and the maximum entropy
    distribution with every marginal of all but one of the columns.
    """
    cells, counts = np.unique(codes, axis=0, return_counts=True)
    k = codes.shape[1]
    empirical = Distribution([tuple(int(v) for v in cell) for cell in cells], counts / counts.sum())
    null = maxent_dist(empirical, [list(subset) for subset in combinations(range(k), k - 1)])
    return cells, counts, null


def _statistic(codes):
    """
    The likelihood-ratio statistic :math:`G^2 = 2 \\sum O \\ln(O / E)` and the fitted null.
    """
    cells, counts, null = _fit(codes)
    probabilities = dict(zip(null.outcomes, null.pmf, strict=True))
    expected = counts.sum() * np.array([probabilities[tuple(int(v) for v in cell)] for cell in cells])
    return max(0.0, 2 * float(np.sum(counts * np.log(counts / expected)))), null


def connected_information_test(data, rvs=None, method="asymptotic", n_surrogates=1000, rng=None):
    r"""
    Test whether the variables of `data` interact as a whole: whether their highest-order
    connected information vanishes.

    The :math:`k`-th order connected information of :math:`k` variables is the entropy
    that the maximum entropy distribution :math:`P^{(k-1)}` with every
    :math:`(k-1)`-variable marginal of :math:`P` has beyond :math:`P`,
    :math:`I^{(k)} = H(P^{(k-1)}) - H(P) = D(P \| P^{(k-1)})`
    :cite:`Amari2001,Schneidman2003`. It is the part of the dependence that requires a
    :math:`k`-way interaction. For three variables the co-information splits as
    :math:`I(X_0 : X_1 : X_2) = C^{(2)} - I^{(3)}`, with :math:`C^{(2)}` the co-information
    of :math:`P^{(2)}`, so a positive :math:`I^{(3)}` is synergy that pairwise structure
    cannot produce :cite:`vansanten2026between`.

    :math:`I^{(k)} = 0` is the log-linear model without a :math:`k`-way interaction, and its
    likelihood-ratio statistic is
    :math:`G^2 = 2 \sum O \ln(O / E) = 2 N \ln 2 \, \hat{I}^{(k)}`, with expected counts
    :math:`E` from the maximum entropy fit to the observed frequencies
    :cite:`bishop1975discrete`. The plug-in :math:`\hat{I}^{(k)}` is positive in every finite
    sample; under the null its mean is about :math:`\prod_i (L_i - 1) / (2 N \ln 2)` bits.

    Parameters
    ----------
    data : array_like
        Observations in rows and variables in columns.
    rvs : list of lists, None
        The groups of columns that form the variables, at least two and disjoint. If None,
        each column is a variable.
    method : {'asymptotic', 'bootstrap'}
        ``'asymptotic'`` compares :math:`G^2` with its chi-squared limit, which needs the
        expected counts not to be small. ``'bootstrap'`` draws samples of the same size
        from the fitted null, refits each, and compares with their statistics; it is
        valid for sparse tables.
    n_surrogates : int
        The number of bootstrap samples.
    rng : None, int, Generator, RandomState
        Source of randomness for the bootstrap.

    Returns
    -------
    result : ConnectedInformationTest

    Raises
    ------
    ValueError
        If there are fewer than two variables or the groups overlap, or `method` is unknown.

    Notes
    -----
    The degrees of freedom count every cell of the full table of observed levels. Cells
    that the fitted null makes impossible, structural zeros of the marginals, reduce them,
    and the asymptotic p-value is then conservative; the bootstrap does not depend on
    them. An :class:`~dit.inference.UndersamplingWarning` is raised for the asymptotic
    test when the smallest expected count is below 1 or more than a fifth of the expected
    counts are below 5. The bootstrap p-value is
    ``(1 + #{bootstrap statistics at least as large}) / (1 + n_surrogates)``.

    Examples
    --------
    The parity of two independent bits is a pure three-way interaction:

    >>> import numpy as np
    >>> from dit.inference import connected_information_test
    >>> rng = np.random.default_rng(0)
    >>> x = rng.integers(0, 2, size=(500, 2))
    >>> data = np.column_stack([x, x[:, 0] ^ x[:, 1]])
    >>> result = connected_information_test(data)
    >>> round(result.connected_information, 2), result.pvalue < 1e-10
    (0.99, True)
    """
    if method not in ("asymptotic", "bootstrap"):
        msg = f"Unknown method {method!r}; expected 'asymptotic' or 'bootstrap'."
        raise ValueError(msg)
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("`data` must have observations in rows and variables in columns.")
    if rvs is None:
        rvs = [[column] for column in range(data.shape[1])]
    columns = [column for group in rvs for column in group]
    if len(rvs) < 2 or len(set(columns)) != len(columns):
        raise ValueError("The test needs at least two disjoint groups of columns.")

    codes = np.column_stack([np.unique(data[:, list(group)], axis=0, return_inverse=True)[1].ravel() for group in rvs])
    n = len(codes)
    value, null_dist = _statistic(codes)
    dof = int(np.prod(codes.max(axis=0)))
    expected = n * null_dist.pmf[null_dist.pmf > 0]
    min_expected = float(expected.min())

    if method == "asymptotic":
        if min_expected < 1 or np.mean(expected < 5) > 0.2:
            warnings.warn(
                f"Undersampled: the smallest expected count is {min_expected:.3g}, and "
                f"{np.mean(expected < 5):.0%} of the expected counts are below 5. The chi-squared "
                "approximation may be poor; consider method='bootstrap'.",
                UndersamplingWarning,
                stacklevel=2,
            )
        pvalue = float(chi2.sf(value, dof)) if dof > 0 else 1.0
        null = None
    else:
        rng = as_generator(rng)
        support = np.array(null_dist.outcomes)
        null = np.empty(n_surrogates)
        for i in range(n_surrogates):
            sample = support[rng.choice(len(support), size=n, p=null_dist.pmf / null_dist.pmf.sum())]
            null[i] = _statistic(sample)[0]
        pvalue = float((1 + np.sum(null >= value - 1e-9 * max(1.0, value))) / (1 + n_surrogates))

    return ConnectedInformationTest(
        order=len(rvs),
        statistic="G2",
        value=value,
        pvalue=pvalue,
        dof=dof,
        connected_information=float(value / (2 * n * np.log(2))),
        n_samples=n,
        min_expected=min_expected,
        null=null,
    )
