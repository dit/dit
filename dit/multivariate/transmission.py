"""
The transmission, a reconstructability-analysis measure of the information lost
when a distribution is decomposed into (reconstructed from) a set of marginals.
"""

from ..divergences import kullback_leibler_divergence

__all__ = ("transmission",)


def transmission(dist, structure=None):
    """
    Compute the transmission of `dist` relative to a marginal model: the
    Kullback-Leibler divergence from the data to the maximum entropy
    distribution reconstructed from the marginals in `structure`.

    This is the decomposition error of reconstructability analysis. With
    Shannon entropy denoted by U, it equals ``U(model) - U(data)``: the
    constraint the model fails to capture. For the independence structure it
    reduces to the total correlation, and for the saturated structure (all
    variables together) it is zero.

    Parameters
    ----------
    dist : Distribution
        The distribution whose decomposition error is computed.
    structure : list of lists, None
        The marginal model: a list of marginals (each a set of random
        variables, or "projection") to hold fixed, e.g. ``[[0, 1], [1, 2]]``
        for the structure ``AB:BC``. If None, the independence model (each
        variable on its own) is used, recovering the total correlation.

    Returns
    -------
    T : float
        The transmission (decomposition error).

    Examples
    --------
    >>> d = dit.example_dists.Xor()
    >>> dit.multivariate.transmission(d)
    1.0
    >>> dit.multivariate.transmission(d, [[0, 1], [1, 2]])
    1.0
    >>> dit.multivariate.transmission(d, [[0, 1, 2]])
    0.0
    """
    # Imported lazily to avoid a circular import: dit.algorithms imports from
    # dit.multivariate at load time.
    from ..algorithms import maxent_dist

    if structure is None:
        structure = [[v] for v in range(dist.outcome_length())]

    me = maxent_dist(dist, structure)
    return kullback_leibler_divergence(dist, me)
