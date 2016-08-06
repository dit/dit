"""
Helper functions related to conditional distributions.

"""

import numpy as np

from .exceptions import ditException
from .helpers import copypmf

import dit

__all__ = [
    'joint_from_factors',
]

def cdist_array(cdists, base='linear', mode='asis'):
    """
    Returns a 2D array for P(Y|X). Rows are X, columns are Y.

    """
    dists = [copypmf(d, base=base, mode=mode) for d in cdists]
    return np.vstack(dists)

def mask_is_complementary(mask1, mask2):
    """
    Returns ``True`` if the masks for d1 and d2 are complementary.

    """
    mask1_comp = tuple(not b for b in mask1)
    return mask1_comp == mask2

def outcome_iter(outcome_X, outcome_Y, mask_Y):
    it_X = iter(outcome_X)
    it_Y = iter(outcome_Y)
    for mask in mask_Y:
        if mask:
            yield next(it_X)
        else:
            yield next(it_Y)

def joint_from_factors(mdist, cdists, strict=True):
    """
    Returns a joint distribution P(X,Y) built from P(X) and P(Y|X).

    Parameters
    ----------
    mdist : Distribution
        The marginal distribution P(X).
    cdists : list
        The list of conditional distributions P(Y|X=x).
    strict : bool
        If ``True``, then the ordering of the random variables is inferred
        from the masks on each distribution. If the masks are not compatible,
        meaning that one is not the elementwise complement of the other, then
        an exception is raised. If ``False``, then the distributions are
        combined such that all of X appears before all of Y. Effectively, the
        existing masks are ignored in that situation.

    Returns
    -------
    d : Distribution
        The joint distribution P(X,Y).

    Raises
    ------
    ditException
        When ``strict=True`` and the masks for ``mdist`` and ``cdists`` are
        not compatible with each other.

    Examples
    --------
    >>> d = dit.random_distribution(5, 2)
    >>> d.set_rv_names('ABCDE')
    >>> pBD, pACEgBD = d.condition_on('BD')
    >>> pABCDE = dit.joint_from_factors(pBD, pACEgBD)
    >>> pABCDE.is_approx_equal(d)
    True

    """
    # We assume that the mask is the same each dist in cdists.
    cdist_mask = cdists[0]._mask

    # Raise exception if mdist and cdists are not compatible.
    compatible = mask_is_complementary(mdist._mask, cdists[0]._mask)
    if strict and not compatible:
        msg = 'Incompatible masks for ``mdist`` and ``cdists``.'
        raise ditException(msg)

    if not compatible:
        cdist_mask = [True] * mdist.outcome_length()
        cdist_mask.extend([False] * cdists[0].outcome_length())

    # Make sure mdist has the proper number of outcomes.
    YgX_pmf = cdist_array(cdists)
    if len(mdist) != YgX_pmf.shape[0]:
        # Maybe it is not trim.
        mdist = mdist.copy(base='linear')
        mdist.make_sparse()
        if len(mdist) != YgX_pmf.shape[0]:
            msg = 'len(mdist) != len(cdists)'
            raise ditException(msg)
        else:
            X_outcomes = mdist.outcomes
            X_pmf = mdist.pmf
    else:
        # Note this could be non-trim and sparse. If so, then an entire row
        # of carr might be turned to zeros. Nothing wrong with that, except
        # it means that the marginal did not come from the distribution that
        # cdists was obtained from, since d.condition_on() only returns
        # conditional distriutions whose condition has positive probability.
        X_outcomes = mdist.outcomes
        X_pmf = copypmf(mdist, base='linear', mode='asis')

    # The joint probabilities
    XY_pmf = YgX_pmf * X_pmf[:, np.newaxis]

    ctor = cdists[0]._outcome_ctor
    # We can't use NumPy for the outcomes, since an array of tuples is
    # automatically turned into a 2D array. We could initialize as a 1D
    # object array, but we'd still have to populate through for loops.
    # So we might as well avoid NumPy here.
    outcomes = []
    for i, X in enumerate(X_outcomes):
        tmp = [ctor(outcome_iter(X, Y, cdist_mask)) for Y in cdists[i].outcomes]
        outcomes.extend(tmp)

    d = dit.Distribution(outcomes, list(XY_pmf.flat),
                         sparse=True, trim=False)

    X_rv_names = mdist.get_rv_names()
    Y_rv_names = cdists[0].get_rv_names()
    if X_rv_names and Y_rv_names:
        rv_names = outcome_iter(X_rv_names, Y_rv_names, cdist_mask)
        d.set_rv_names(list(rv_names))

    return d
