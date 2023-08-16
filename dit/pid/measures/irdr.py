"""
The I_rdr measure (reachable decision regions), by Mages & Rohner
"""

from ..pid import BasePID
import numpy as np
from functools import reduce
from scipy.spatial import ConvexHull

__all__ = (
    'PID_RDR',
)

precision = 14

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
    s_alphabet = set([x[0] for x in p_STt.keys()])
    return [(p_STt.get((s,True),0)/p_Tt, p_STt.get((s,False),0)/(1-p_Tt))  for s in s_alphabet] # list of tuples with target order (True,False)


def cv_hull(p_S1_g_Tt,p_S2_g_Tt):
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
    channel1 = sorted(p_S1_g_Tt, key=lambda x: x[0]/x[1] if x[1] != 0 else np.inf, reverse=True)
    channel2 = sorted(p_S2_g_Tt, key=lambda x: x[0]/x[1] if x[1] != 0 else np.inf, reverse=True)
    # check dimensionality > 1 (causes issues in ConvexHull function)
    if len(set([x[0]/x[1] if x[1] != 0 else np.inf for x in (channel1 + channel2)])) > 1:
        # generate zonotopes and their convex hull
        points1  = [(0.0,0.0)] + reduce(lambda p,v: p+[(p[-1][0]+v[0], p[-1][1]+v[1])], channel1[1:], [channel1[0]])
        points2  = [(0.0,0.0)] + reduce(lambda p,v: p+[(p[-1][0]+v[0], p[-1][1]+v[1])], channel2[1:], [channel2[0]])
        hull = ConvexHull([(b,a) for a,b in (points1 + points2)])
        hull_points = [(a,b) for b,a in hull.points[hull.vertices].tolist()]
        # generate resulting channel
        hull_points = sorted([x for x in hull_points if x != (0,0) and x != (0.0,0.0)])
        ch_channel = [hull_points[0]] + [(np.round(n[0]-l[0],precision),np.round(n[1]-l[1],precision)) for l,n in zip(hull_points[:-1],hull_points[1:])]  # rounding since float imprecision can create small negative vectors
        return ch_channel
    else:
        return [(1,1)]

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
    elif len(p_Sx_g_Tt) == 1:
        return np.nansum([x*np.log2(x/(p_Tt*x + (1-p_Tt)*y)) for x,y in p_Sx_g_Tt[0]])  # Equation 28a
    else:
        return i_pw(p_Tt, [p_Sx_g_Tt[0]]) + i_pw(p_Tt, p_Sx_g_Tt[1:]) - i_pw(p_Tt, [cv_hull(p_Sx_g_Tt[0], x) for x in p_Sx_g_Tt[1:]])  # Equation 28b

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
        p_SxT = [dist.coalesce((source,target)) for source in sources]   # marginals with target as second variable
        p_t   =  dist.marginal(target)
        # construct pointwise marginal channels
        f = lambda x,t: ((x[0][0],x[0][1] == t),x[1])                    # convert to binary target
        acc = lambda d, x: d.update({x[0]: d.get(x[0],0)+x[1]}) or d     # aggregate identical states
        p_SxTt = [(p_t[t], [reduce(acc,map(lambda x: f(x,t), dist.to_dict().items()), {}) for dist in p_SxT]) for t in p_t.outcomes] # p_SxTt : [(p_t[t1], [p_S1Tt1, p_S2Tt1, ...]), ...]
        # compute result
        return np.nansum([p_Tt * i_pw(p_Tt, [condition_pw(p_Tt,x) for x in p_STt]) for p_Tt, p_STt in p_SxTt if 0.0 < p_Tt and p_Tt < 1.0]) # Equation 28c