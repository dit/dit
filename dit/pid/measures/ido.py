"""
The do-calculus PID measure from Lyu, Clark & Raviv (2024).

For two sources X_i, X_j and target Y, the "intervened" source X'_i is
defined via Pearl-style do-calculus by

    p(X'_i = x_i, Y = y | X_j = x_j) = p(X_i = x_i | Y = y) p(Y = y | X_j = x_j)

so that p(X'_i, X_j, Y) = p(X_j) p(Y|X_j) p(X_i|Y).  The do-calculus
redundancy is then

    I_do(X_1, X_2 ; Y) = I(X'_1 ; X_2) = I(X'_2 ; X_1)

and the unique-information atoms are I_partial(X_i ; Y) = I(X'_i ; Y | X_j).
The two forms of the redundancy are equal in exact arithmetic; we average
them for numerical symmetry.

Only defined for two sources.

References
----------
.. [1] A. Lyu, A. Clark, N. Raviv, "Explicit Formula for Partial
       Information Decomposition", 2024 IEEE International Symposium on
       Information Theory (ISIT), pp. 2329-2334, 2024.
       arXiv:2402.03554.
"""

import numpy as np

from ...exceptions import ditException
from ..pid import BaseBivariatePID

__all__ = ("PID_Do",)


def _intervened_joint(p_xyz):
    """
    Build the intervened joint q(x'_i, x_j, y) = p(x_j) p(y|x_j) p(x_i|y).

    Parameters
    ----------
    p_xyz : np.ndarray
        The joint p(X_i, X_j, Y) as a 3D array indexed by (x_i, x_j, y).

    Returns
    -------
    q : np.ndarray
        The intervened joint, indexed by (x'_i, x_j, y), same shape as p_xyz.
    """
    p_xi_y = p_xyz.sum(axis=1)
    p_xj_y = p_xyz.sum(axis=0)
    p_xj = p_xj_y.sum(axis=1)
    p_y = p_xi_y.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_xi_given_y = np.where(p_y[None, :] > 0, p_xi_y / p_y[None, :], 0.0)
        p_y_given_xj = np.where(p_xj[:, None] > 0, p_xj_y / p_xj[:, None], 0.0)

    return (
        p_xj[None, :, None]
        * p_y_given_xj[None, :, :]
        * p_xi_given_y[:, None, :]
    )


def _mi_from_joint(q_ab):
    """
    Mutual information I(A; B) from a 2D joint pmf, in bits.
    """
    p_a = q_ab.sum(axis=1)
    p_b = q_ab.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = p_a[:, None] * p_b[None, :]
        ratio = np.where((q_ab > 0) & (denom > 0), q_ab / denom, 1.0)
        log_ratio = np.log2(ratio)
    return float(np.nansum(q_ab * log_ratio))


def i_do(d, source_0, source_1, target):
    """
    Compute the bivariate do-calculus redundancy I_do(source_0, source_1 ; target).

    Returns the average of I(X'_0 ; X_1) and I(X'_1 ; X_0) computed on the
    two intervened joints.  These are equal in exact arithmetic.
    """
    d = d.coalesce([source_0, source_1, target])
    d.make_dense()
    p_xyz = d.pmf.reshape([len(a) for a in d.alphabet])

    q_for_0 = _intervened_joint(p_xyz)
    q_xprime0_x1 = q_for_0.sum(axis=2)
    mi_a = _mi_from_joint(q_xprime0_x1)

    q_for_1 = _intervened_joint(p_xyz.transpose(1, 0, 2))
    q_xprime1_x0 = q_for_1.sum(axis=2)
    mi_b = _mi_from_joint(q_xprime1_x0)

    return 0.5 * (mi_a + mi_b)


class PID_Do(BaseBivariatePID):
    """
    The do-calculus partial information decomposition of Lyu, Clark & Raviv.

    Notes
    -----
    Bivariate only.  The paper explicitly notes that I_do only fully
    determines the PID for n = 2 sources.
    """

    _name = "I_do"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute I_do(sources : target) for a pair of sources.

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_do for.
        sources : iterable of iterables
            The source variables (exactly two).
        target : iterable
            The target variable.

        Returns
        -------
        ri : float
            The do-calculus redundancy.
        """
        if len(sources) != 2:  # pragma: no cover
            msg = f"I_do is only defined for two sources, {len(sources)} given."
            raise ditException(msg)

        return i_do(d, sources[0], sources[1], target)
