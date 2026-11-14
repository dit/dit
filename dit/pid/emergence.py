"""
Practical criteria for causal emergence :cite:`rosas2020reconciling`.

A feature :math:`V` of a system :math:`X = (X^1, \\ldots, X^n)` is causally
emergent when it predicts the system's future beyond what its parts do.
Detecting this exactly needs an integrated information decomposition (see
:meth:`PhiID.emergence_capacity <dit.pid.PhiID.emergence_capacity>`). The
criteria here give sufficient conditions from mutual informations alone, given
the joint distribution of the system and the feature at two times :math:`t`
and :math:`t'`. For order :math:`k`, :math:`\\alpha` and :math:`\\beta` range
over the groups of :math:`k` parts.
"""

from itertools import combinations

from ..multivariate import coinformation
from ..utils import flatten, unitful

__all__ = (
    "emergence_delta",
    "emergence_gamma",
    "emergence_psi",
)


def _mi(dist, a, b):
    return coinformation(dist, [list(a), list(b)])


def _groups(parts, k):
    """
    The variables of each group of `k` of the `parts`.
    """
    parts = [list(part) for part in parts]
    if not 1 <= k <= len(parts):
        raise ValueError(f"The order k={k} must be between 1 and the number of parts, {len(parts)}.")
    return [list(flatten(group)) for group in combinations(parts, k)]


@unitful
def emergence_psi(dist, micro_past, macro_past, macro_future, k=1):
    """
    The causal emergence criterion :cite:`rosas2020reconciling`

    .. math::

        \\Psi^{(k)}_{t,t'}(V) = I[V_t : V_{t'}] - \\sum_{|\\alpha| = k} I[X^\\alpha_t : V_{t'}] .

    If it is positive, :math:`V` is causally emergent: it predicts its own
    future beyond what the groups of `k` parts do. The converse fails, and the
    size of :math:`\\Psi` is not a degree of emergence. A positive
    :math:`\\Psi` can also come from a statistically autonomous coarse-graining
    of parts that do not interact at all, such as the sum of independent Markov
    chains :cite:`liardi2025simple`.

    The guarantee assumes that :math:`V_t - X_t - X_{t'} - V_{t'}` is a Markov
    chain, as when :math:`V` is a function of :math:`X`. This is not checked.

    Parameters
    ----------
    dist : Distribution
        The joint distribution of the system and the feature at times t and t'.
    micro_past : iter of iters
        The parts :math:`X^1_t, \\ldots, X^n_t` of the system at time t, each a
        list of variables.
    macro_past, macro_future : iter
        The variables of the feature at times t and t'.
    k : int
        The order of emergence: the size of the groups of parts.

    Returns
    -------
    psi : float
        The criterion.

    Raises
    ------
    ValueError
        If `k` is not between 1 and the number of parts.
    """
    groups = _groups(micro_past, k)
    return _mi(dist, macro_past, macro_future) - sum(_mi(dist, group, macro_future) for group in groups)


@unitful
def emergence_delta(dist, micro_past, micro_future, macro_past, k=1):
    """
    The downward causation criterion :cite:`rosas2020reconciling`

    .. math::

        \\Delta^{(k)}_{t,t'}(V) = \\max_{|\\alpha| = k} \\left( I[V_t : X^\\alpha_{t'}]
            - \\sum_{|\\beta| = k} I[X^\\beta_t : X^\\alpha_{t'}] \\right) .

    If it is positive, :math:`V` exerts downward causation: it predicts some
    group of `k` future parts beyond what the groups of `k` past parts do. As
    with :func:`emergence_psi`, this is a sufficient condition that assumes
    :math:`V_t - X_t - X_{t'}` is a Markov chain.

    Parameters
    ----------
    dist : Distribution
        The joint distribution of the system and the feature at times t and t'.
    micro_past, micro_future : iter of iters
        The parts of the system at times t and t', each a list of variables.
    macro_past : iter
        The variables of the feature at time t.
    k : int
        The order of emergence: the size of the groups of parts.

    Returns
    -------
    delta : float
        The criterion.

    Raises
    ------
    ValueError
        If `k` is not between 1 and the number of parts.
    """
    pasts, futures = _groups(micro_past, k), _groups(micro_future, k)
    return max(_mi(dist, macro_past, future) - sum(_mi(dist, past, future) for past in pasts) for future in futures)


@unitful
def emergence_gamma(dist, micro_future, macro_past, k=1):
    """
    The causal decoupling criterion :cite:`rosas2020reconciling`

    .. math::

        \\Gamma^{(k)}_{t,t'}(V) = \\max_{|\\alpha| = k} I[V_t : X^\\alpha_{t'}] .

    If :math:`\\Psi^{(k)} > 0` (see :func:`emergence_psi`) and
    :math:`\\Gamma^{(k)} = 0`, :math:`V` is causally decoupled: it predicts its
    own future but no group of `k` future parts.

    Parameters
    ----------
    dist : Distribution
        The joint distribution of the system and the feature at times t and t'.
    micro_future : iter of iters
        The parts of the system at time t', each a list of variables.
    macro_past : iter
        The variables of the feature at time t.
    k : int
        The order of emergence: the size of the groups of parts.

    Returns
    -------
    gamma : float
        The criterion.

    Raises
    ------
    ValueError
        If `k` is not between 1 and the number of parts.
    """
    return max(_mi(dist, macro_past, future) for future in _groups(micro_future, k))
