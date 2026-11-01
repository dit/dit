"""
A PID inspired by BROJA's * assumption
"""

from itertools import product

import numpy as np

from ...distribution import Distribution
from ...multivariate import coinformation
from ..pid import BaseBivariatePID

__all__ = ("PID_MES",)


def _conditionally_independent(d, sources, target):
    """
    The maximum entropy distribution with the source-target marginals of `d`,
    q = p(t) prod_i p(x_i | t), over integer-coded sources and target.

    Parameters
    ----------
    d : Distribution
        The distribution.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.

    Returns
    -------
    q : Distribution
        The distribution of the sources (variables 0..n-1) and the target (variable n).
    """
    d = d.coalesce(list(sources) + [target])
    d.make_dense()
    shape = [len(alphabet) for alphabet in d.alphabet]
    p = np.asarray(d.pmf, dtype=float).reshape(shape)
    n = len(shape) - 1
    p_t = p.sum(axis=tuple(range(n)))
    q = p_t.reshape([1] * n + [shape[-1]])
    for i in range(n):
        p_it = p.sum(axis=tuple(j for j in range(n) if j != i))
        conditional = np.divide(p_it, p_t, out=np.zeros_like(p_it), where=p_t > 0)
        q = q * conditional.reshape([shape[i] if j == i else 1 for j in range(n)] + [shape[-1]])
    outcomes = list(product(*(range(size) for size in shape)))
    return Distribution(outcomes, q.ravel())


class PID_MES(BaseBivariatePID):
    """
    The maximum entropy PID inspired by BROJA's * assumption.

    Redundancy is the coinformation of the sources and the target under the maximum
    entropy distribution with the same source-target marginals,
    q = p(t) prod_i p(x_i | t). The sources are independent given the target under q, so
    for two sources this is I_q(X_0 : X_1), which is also the redundancy of
    :class:`~dit.pid.PID_Do`.
    """

    _name = "I_ME*"

    @staticmethod
    def _measure(d, sources, target):
        """
        I_ME*, the coinformation under the maximum entropy distribution satisfying
        the * assumption of BROJA.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_mes for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        i_mes : float
            The value of I_ME*.
        """
        return coinformation(_conditionally_independent(d, sources, target))
