# -*- coding: utf-8 -*-
"""
"""

from collections import defaultdict

from .lattice import dist_from_induced_sigalg, insert_rv
from ..helpers import parse_rvs
from ..math import sigma_algebra

__all__ = ['mss_sigalg',
           'insert_mss',
           'mss',
          ]

def partial_match(first, second, places):
    """
    Returns whether `second` is a marginal outcome at `places` of `first`.
    """
    return tuple([first[i] for i in places]) == tuple(second)

def mss_sigalg(dist, rvs, about=None, rv_mode=None):
    """
    Construct the sigma algebra for the minimal sufficient statistic of `rvs`
    about `about`.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    """
    mapping = parse_rvs(dist, about)[1]

    partition = defaultdict(list)

    md, cds = dist.condition_on(rvs=rvs, crvs=about, rv_mode=rv_mode)

    for outcome, cd in zip(md.outcomes, cds):
        matches = [ o for o in dist.outcomes if partial_match(o, outcome, mapping) ]
        for c in partition.keys():
            if c.is_approx_equal(cd):
                eq[c].extend(matches)
                break
        else:
            eq[cd].extend(matches)

    mss_sa = sigma_algebra(map(frozenset, eq.values()))

    return mss_sa

def insert_mss(dist, idx=-1, rvs, about=None, rv_mode=None):
    """
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    """
    mss_sa = mss_sigalg(dist, rvs, about, rv_mode)
    new_dist = insert_rv(dist, idx, mss_sa)
    return new_dist

def mss(dist, rvs, about=None, rv_mode=None, int_outcomes=True):
    """
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    """
    mss_sa = mss_sigalg
    d = dist_from_induced_sigalg(dist, mss_sa, int_outcomes)
    return d
