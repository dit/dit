"""
The functional common information.
"""

from ..distconst import insert_rvf
from ..helpers import flatten, normalize_rvs
from ..math import close
from ..utils import partitions

from .entropy import entropy
from .binding_information import dual_total_correlation

def add_partition(dist, part):
    invert_part = {e: str(i) for i, es in enumerate(part) for e in es}
    dist = insert_rvf(dist, lambda j: invert_part[j])
    return dist

def functional_markov_chain(dist, rvs=None, crvs=None, rv_mode=None):
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)
    outcomes = dist.outcomes
    f = [len(dist.rvs)]
    parts = partitions(outcomes, tuples=True)
    dists = [ add_partition(dist, part) for part in parts ]

    B = lambda d: dual_total_correlation(d, rvs, crvs+f, rv_mode)

    dists = [ d for d in dists if close(B(d), 0) ]
    return min(dists, key=lambda d: entropy(d, rvs=f, rv_mode=rv_mode))

def functional_common_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Compute the functional common information, F, of `dist`. It is the entropy
    of the smallest random variable W such that all the variables in `rvs` are
    rendered independent conditioned on W, and W is a function of `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the functional common information is
        computed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    F : float
        The functional common information.
    """
    d = functional_markov_chain(dist, rvs, crvs, rv_mode)
    return H(d.marginalize(list(flatten(dist.rvs))))
