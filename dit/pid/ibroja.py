"""
The I_BROJA unique measure, as proposed by the BROJA team.
"""

from __future__ import division

import numpy as np

from .pid import BaseUniquePID

from ..algorithms.scipy_optimizers import BaseConvexOptimizer, BROJABivariateOptimizer
from ..multivariate import coinformation


class BROJAOptimizer(BaseConvexOptimizer):
    """
    Optimizer for computing the max mutual information between
    inputs and outputs. In the bivariate case, this corresponds to
    maximizing the coinformation.
    """

    def __init__(self, dist, input, others, output, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to base the optimization on.
        input : iterable
            Variables to treat as inputs.
        others : iterable of iterables
            The other input variables.
        output : iterable
            The output variable.
        rv_mode : bool
            Unused, provided for compatibility with parent class.
        """
        dist = dist.coalesce((input,) + (sum(others, ()),) + (output,))
        constraints = [[0, 2], [1, 2]]
        super(BROJAOptimizer, self).__init__(dist, constraints)


    def objective(self, x):
        """
        Minimize I(input:output|others).

        Parameters
        ----------
        x : np.ndarray
            Optimization vector.

        Returns
        -------
        b : float
            The objective.
        """
        pmf = self._expand(x).reshape(self._shape)
        h_total = -np.nansum(pmf * np.log2(pmf))

        p_output = pmf.sum(axis=(0, 1))
        h_output = -np.nansum(p_output * np.log2(p_output))

        input_pmf = pmf.sum(axis=2)
        h_input = -np.nansum(input_pmf * np.log2(input_pmf))

        mi = h_input + h_output - h_total

        reduced_pmf = pmf.sum(axis=0)
        h_reduced = -np.nansum(reduced_pmf * np.log2(reduced_pmf))

        others_pmf = pmf.sum(axis=(0, 2))
        h_others = -np.nansum(others_pmf * np.log2(others_pmf))

        omi = h_others + h_output - h_reduced

        cmi = mi - omi

        return cmi


def i_broja(d, inputs, output, maxiters=1000):
    """
    This computes unique information as min{I(input : output | other_inputs)} over the space of distributions
    which matches input-output marginals.

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_broja for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    ibroja : dict
        The value of I_broja for each individual input.
    """
    uniques = {}
    if len(inputs) == 2:
        broja = BROJABivariateOptimizer(d, list(inputs), output)
        broja.optimize(maxiters=maxiters)
        opt_dist = broja.construct_dist()
        uniques[inputs[0]] = coinformation(opt_dist, [[0], [2]], [1])
        uniques[inputs[1]] = coinformation(opt_dist, [[1], [2]], [0])
    else:
        for input_ in inputs:
            others = sum([i for i in inputs if i != input_], ())
            dm = d.coalesce([input_, others, output])
            broja = BROJAOptimizer(dm, (0,), ((1,),), (2,))
            broja.optimize(maxiters=maxiters)
            d_opt = broja.construct_dist()
            uniques[input_] = coinformation(d_opt, [[0], [2]], [1])

    return uniques


class PID_BROJA(BaseUniquePID):
    """
    The BROJA partial information decomposition.

    Notes
    -----
    This partial information decomposition, at least in the bivariate input case,
    was independently suggested by Griffith.
    """
    _name = "I_broja"
    _measure = staticmethod(i_broja)
