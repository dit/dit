"""
The Kamath-Anantharam dual common information.

A new dual to the Gács-Körner common information, defined via the Gray-Wyner
system. See Kamath & Anantharam (2010).
"""

from ...algorithms.minimal_sufficient_statistic import insert_mss
from ...helpers import normalize_rvs, parse_rvs
from ...shannon import conditional_entropy as H
from ...utils import flatten, unitful

__all__ = (
    "directed_kamath_common_information",
    "kamath_common_information",
)


def _directed_value(dist, rvs, about, crvs):
    """
    Compute H(Phi^{about}_{rvs} | crvs) as a plain float.

    Phi^{about}_{rvs} is the minimal sufficient statistic of `rvs` about
    `about` (Kamath & Anantharam 2010, Lemma 3.5(5)).
    """
    rvs_idx = list(parse_rvs(dist, rvs)[1])
    about_idx = list(parse_rvs(dist, about)[1])
    crvs_idx = list(parse_rvs(dist, crvs)[1]) if crvs else []

    d = insert_mss(dist, -1, rvs=rvs_idx, about=about_idx)
    new_idx = d.outcome_length() - 1
    return H(d, [new_idx], crvs_idx)


@unitful
def directed_kamath_common_information(dist, rvs, about, crvs=None):
    """
    Calculates the directed Kamath-Anantharam common information
    G(rvs -> about) = H(Phi^{about}_{rvs}), where Phi^{about}_{rvs} is the
    minimal sufficient statistic of `rvs` about `about`.

    In the bivariate notation of Kamath & Anantharam (2010), this computes
    G(Y -> X) when called with `rvs=Y, about=X`. The variable on the tail
    of the arrow (`rvs`) is compressed via the minimal sufficient statistic;
    the variable on the head (`about`) is what that statistic preserves
    information about.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the directed common information is
        calculated.
    rvs : list
        The indexes (or names) of the random variables on the tail of the
        arrow. These are compressed to their minimal sufficient statistic
        about `about`.
    about : list
        The indexes (or names) of the random variables on the head of the
        arrow. The minimal sufficient statistic preserves all information
        about these variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then
        no conditioning is performed.

    Returns
    -------
    G : float
        The directed Kamath-Anantharam common information.

    Raises
    ------
    ditException
        Raised if `rvs`, `about`, or `crvs` contain non-existant random
        variables.

    """
    rvs = list(flatten([rvs]))
    about = list(flatten([about]))
    crvs = [] if crvs is None else list(flatten([crvs]))

    return _directed_value(dist, rvs, about, crvs)


@unitful
def kamath_common_information(dist, rvs=None, crvs=None):
    """
    Calculates the Kamath-Anantharam dual common information U[X1:X2...] of
    the random variables in `rvs`.

    For two variables this is

        U(X; Y) = max{ G(Y -> X), G(X -> Y) }
                = max{ H(Phi^X_Y),  H(Phi^Y_X) },

    the symmetric "dual" to the Gács-Körner common information introduced
    by Kamath & Anantharam (2010). It is generalized to n variables as

        U(X_{0:n}) = max_i H(Phi^{X_{!=i}}_{X_i}),

    the maximum entropy of the minimal sufficient statistic of one variable
    about all the others.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the common information is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of a random
        variable to compress via its minimal sufficient statistic about the
        rest. If None, then each single random variable is used.
    crvs : list, None
        A single list of indexes specifying the random variables to
        condition on. If None, then no conditioning is performed.

    Returns
    -------
    U : float
        The Kamath-Anantharam dual common information.

    Raises
    ------
    ditException
        Raised if `rvs` or `crvs` contain non-existant random variables.

    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    values = []
    for i, rv in enumerate(rvs):
        rv_flat = list(flatten([rv]))
        rest = list(flatten(rvs[:i] + rvs[i + 1 :]))
        values.append(_directed_value(dist, rv_flat, rest, crvs))

    return max(values)
