"""
The I_rdr measure (reachable decision regions), by Mages & Rohner

https://arxiv.org/abs/2407.04415
"""

from functools import reduce

import numpy as np
from scipy.spatial import ConvexHull

from ..pid import BasePID

__all__ = ("PID_RDR",)


PRECISION = 14


def r_prec(x, prec=PRECISION):
    """
    Rounding a float to precision

    Parameters
    ----------
    x : Float
        input value
    pred: Int
        rounding precision

    Returns
    -------
    y : float
        rounded result
    """
    return np.round(x, prec)


def condition_pw(p_Tt, p_STt):
    """
    Compute set of vectors corresponding to the conditional pointwise channel (zonotope vectors)

    Parameters
    ----------
    p_Tt : Float
        Probability of P(T=t)
    p_STt : dict
        dictionary corresponding to the joint distribution of P(S,Tt)

    Returns
    -------
    p_S_g_Tt : list of tuples
        set of vectors specifying P(S|Tt)
    """
    s_alphabet = {x[0] for x in p_STt}
    # return list of tuples with target order (True,False)
    return [(p_STt.get((s, True), 0) / p_Tt, p_STt.get((s, False), 0) / (1 - p_Tt)) for s in s_alphabet]


def cv_hull(p_S1_g_Tt, p_S2_g_Tt):
    """
    Compute convex hull of two pointwise channels (blackwell joint of binary input channels)

    Parameters
    ----------
    p_S1_g_Tt : list of tuples
        set of vectors specifying P(S1|Tt)
    p_S2_g_Tt : list of tuples
        set of vectors specifying P(S2|Tt)

    Returns
    -------
    p_S2vS2_g_Tt : list of tuples
        set of vectors specifying their convex hull
    """
    # sort by likelihood ratio (construct zonotope)
    channel1 = sorted(p_S1_g_Tt, key=lambda x: x[0] / x[1] if x[1] != 0 else np.inf, reverse=True)
    channel2 = sorted(p_S2_g_Tt, key=lambda x: x[0] / x[1] if x[1] != 0 else np.inf, reverse=True)
    # check dimensionality > 1 (causes issues in ConvexHull function)
    if len({x[0] / x[1] if x[1] != 0 else np.inf for x in (channel1 + channel2)}) > 1:
        # generate zonotopes and their convex hull
        points1 = reduce(lambda p, v: p + [(p[-1][0] + v[0], p[-1][1] + v[1])], channel1[1:], [channel1[0]])
        points2 = reduce(lambda p, v: p + [(p[-1][0] + v[0], p[-1][1] + v[1])], channel2[1:], [channel2[0]])
        hull = ConvexHull([(b, a) for a, b in ([(0.0, 0.0)] + points1 + points2)])
        hull_points = [(a, b) for b, a in hull.points[hull.vertices].tolist()]
        # generate resulting channel from vertices
        hull_points = sorted([x for x in hull_points if x not in ((0, 0), (0.0, 0.0))])
        diff_list = zip([(0.0, 0.0)] + hull_points[:-1], hull_points, strict=True)
        return [(r_prec(n[0] - m[0]), r_prec(n[1] - m[1])) for m, n in diff_list]
    return [(1, 1)]


def i_pw(p_Tt, p_Sx_g_Tt):
    """
    Compute pointwise valuation function (Equation 28a and 28b)

    Parameters
    ----------
    p_Tt : Float
        Probability of P(T=t)
    p_Sx_g_Tt : list of list of tuples
        conditional probabilities of the pointwise channels

    Returns
    -------
    i : float
        pointwise valuation
    """
    if len(p_Sx_g_Tt) == 0:
        return 0
    if len(p_Sx_g_Tt) == 1:
        # Equation 28a
        return np.nansum([x * np.log2(x / (p_Tt * x + (1 - p_Tt) * y)) for x, y in p_Sx_g_Tt[0] if x != 0.0])
    # Equation 28b
    pw_joint = [cv_hull(p_Sx_g_Tt[0], x) for x in p_Sx_g_Tt[1:]]
    return i_pw(p_Tt, [p_Sx_g_Tt[0]]) + i_pw(p_Tt, p_Sx_g_Tt[1:]) - i_pw(p_Tt, pw_joint)


class PID_RDR(BasePID):
    """
    The Mages & Rohner partial information decomposition.
    """

    _name = "I_rdr"

    @staticmethod
    def _measure(dist, sources, target):
        """
        Compute I_rdr(sources : target) = I(meet(sources) : target)

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_rdr for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        irdr : float
            The value of I_rdr.
        """
        # marginals with target as second variable
        p_SxT = [dist.coalesce((source, target)) for source in sources]
        p_t = dist.marginal(target)

        # construct pointwise marginal channels
        p_Sx_g_Tt = []  # p_Sx_g_Tt structure: [(p_t[t1], [p_S1_g_Tt1, p_S2_g_Tt1, ...]), ...]
        for t in p_t.outcomes:
            p_SxTt = []  # pointwise joint distributions
            for p_ST in p_SxT:
                # convert target to binary variable
                p_STt = [((key[0], key[1] == t), val) for key, val in p_ST.to_dict().items()]
                # aggregate identical states
                p_STt = reduce(lambda d, x: d.update({x[0]: d.get(x[0], 0) + x[1]}) or d, p_STt, {})
                p_SxTt.append(p_STt)
            # convert maginals to conditionals (pointwise channels)
            p_Sx_g_Tt.append((p_t[t], [condition_pw(p_t[t], x) for x in p_SxTt]))

        # compute result (Equation 28c)
        return sum(p_Tt * i_pw(p_Tt, p_S_g_Tt) for p_Tt, p_S_g_Tt in p_Sx_g_Tt if 0.0 < p_Tt < 1.0)
