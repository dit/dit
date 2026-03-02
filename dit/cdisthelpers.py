"""
Helper functions related to conditional distributions.
"""

import numpy as np

from .exceptions import ditException
from .helpers import copypmf

__all__ = ("joint_from_factors",)


def cdist_array(cdists, base="linear", mode="asis"):
    """
    Returns a 2D array for P(Y|X). Rows are X, columns are Y.
    """
    dists = [copypmf(d, base=base, mode=mode) for d in cdists]
    return np.vstack(dists)


def joint_from_factors(mdist, cdists, strict=True):
    """
    Returns a joint distribution P(X,Y) built from P(X) and P(Y|X).

    Parameters
    ----------
    mdist : Distribution
        The marginal distribution P(X).
    cdists : list of Distribution
        The list of conditional distributions P(Y|X=x).
    strict : bool
        If True, require that the marginal and conditional masks are
        complementary.  If False, concatenate X before Y.

    Returns
    -------
    d : Distribution
        The joint distribution P(X,Y).
    """
    from .distribution import Distribution

    mdist_lin = mdist.copy(base="linear")
    mdist_lin.make_sparse()
    X_outcomes = mdist_lin.outcomes
    X_pmf = mdist_lin.pmf

    if len(X_outcomes) != len(cdists):
        raise ditException("len(mdist) != len(cdists)")

    # Build joint outcome/probability pairs by combining X outcomes with Y|X outcomes
    outcomes = []
    pmf = []
    for i, x_outcome in enumerate(X_outcomes):
        cd = cdists[i]
        cd_lin = cd.copy(base="linear")
        cd_lin.make_dense()
        for y_outcome, y_prob in zip(cd_lin.outcomes, cd_lin.pmf, strict=True):
            joint_outcome = tuple(x_outcome) + tuple(y_outcome)
            joint_prob = float(X_pmf[i]) * float(y_prob)
            outcomes.append(joint_outcome)
            pmf.append(joint_prob)

    d = Distribution(outcomes, pmf)

    # Assign rv_names if both marginal and conditionals have them
    x_names = list(mdist.get_rv_names()) if mdist.get_rv_names() else None
    y_names = list(cdists[0].get_rv_names()) if cdists[0].get_rv_names() else None
    if x_names and y_names:
        # Deduplicate names: if Y names overlap with X names, prefix them
        new_y_names = []
        for yn in y_names:
            if yn in x_names:
                yn = f"{yn}_Y"
            new_y_names.append(yn)
        d.set_rv_names(x_names + new_y_names)

    return d
