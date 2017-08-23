"""
The I_BROJA unique measure, as proposed by the BROJA team.
"""

from __future__ import division

import numpy as np

from .pid import BaseUniquePID

from ..algorithms.scipy_optimizers import BaseConvexOptimizer
from ..multivariate import coinformation


class BROJAOptimizer(BaseConvexOptimizer):
    """
    Optimizer for computing the max mutual information between
    inputs and outputs. In the bivariate case, this corresponds to
    maximizing the coinformation.
    """

    def __init__(self, dist, inputs, output, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to base the optimization on.
        inputs : iterable of iterables
            Variables to treat as inputs.
        output : iterable
            The output variable.
        rv_mode : bool
            Unused, provided for compatibility with parent class.
        """
        self._inputs = inputs
        self._output = output
        self._var_map = {var: i for i, var in enumerate(inputs + (output,))}
        dist = dist.coalesce(inputs + (output,))
        constraints = [i + dist.rvs[-1] for i in dist.rvs[:-1]]
        super(BROJAOptimizer, self).__init__(dist, constraints)


    def objective(self, x):
        """
        Minimize the portion of I(inputs:output) that is not I(input:output|others).

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

        inputs = tuple(range(len(self._shape) - 1))
        p_output = pmf.sum(axis=inputs)
        h_output = -np.nansum(p_output * np.log2(p_output))

        input_pmf = pmf.sum(axis=-1)
        h_input = -np.nansum(input_pmf * np.log2(input_pmf))

        mi = h_input + h_output - h_total

        return mi


def i_broja(d, inputs, output):
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
    for input_ in inputs: # fix this to do simpler, and independent optimizations
        others = sum([i for i in inputs if i != input_], ())
        dm = d.coalesce([input_, others, output])
        broja = BROJAOptimizer(dm, ((0,), (1,)), (2,))
        broja.optimize()
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
