"""
The I_BROJA unique measure, as proposed by the BROJA team.
"""

import numpy as np

from .pid import BaseUniquePID

from ..algorithms.scipy_optimizers import BaseNonConvexOptimizer
from ..multivariate import coinformation


class BROJAOptimizer(BaseNonConvexOptimizer):
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

    # objective 1: maximize B2
    def objective_1(self, x):
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
        inputs = tuple(range(len(self._shape) - 1))
        pmf = self._expand(x).reshape(self._shape)
        h = -np.nansum(pmf * np.log2(pmf))

        p_output = pmf.sum(axis=inputs)
        marg_h = -np.nansum(p_output * np.log2(p_output))

        input_pmf = pmf.sum(axis=-1)
        input_h = -np.nansum(input_pmf * np.log2(input_pmf))

        cond_h = h - input_h

        mi = marg_h - cond_h

        cmis = []
        for input_ in inputs:
            p = pmf.sum(axis=input_)
            h = -np.nansum(p * np.log2(p))
            ip = p.sum(axis=-1)
            ih = -np.nansum(ip * np.log2(ip))
            cmi = (h - ih) - cond_h
            cmis.append(cmi)

        obj = -(mi - sum(cmis))

        return obj

    # objective 2: minimize sum of CMIs
    def objective_2(self, x):
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
        inputs = tuple(range(len(self._shape) - 1))
        pmf = self._expand(x).reshape(self._shape)
        h = -np.nansum(pmf * np.log2(pmf))

        input_pmf = pmf.sum(axis=-1)
        input_h = -np.nansum(input_pmf * np.log2(input_pmf))

        cond_h = h - input_h

        cmis = []
        for input_ in inputs:
            p = pmf.sum(axis=input_)
            h = -np.nansum(p * np.log2(p))
            ip = p.sum(axis=-1)
            ih = -np.nansum(ip * np.log2(ip))
            cmi = (h - ih) - cond_h
            cmis.append(cmi)

        obj = sum(cmis)

        return obj

    def objective_bad(self, x):
        """
        Compute the mutual information between inputs and output.

        Paramters
        ---------
        x : np.ndarray
            Optimization vector.

        Returns
        -------
        i : float
            The mutual information between inputs and output.
        """
        inputs = tuple(range(len(self._shape) - 1))
        pmf = self._expand(x).reshape(self._shape)
        h = -np.nansum(pmf * np.log2(pmf))

        p_output = pmf.sum(axis=inputs)
        marg_h = -np.nansum(p_output * np.log2(p_output))

        input_pmf = pmf.sum(axis=-1)
        input_h = -np.nansum(input_pmf * np.log2(input_pmf))

        cond_h = h - input_h

        return marg_h - cond_h

    objective = objective_2

def i_broja(d, inputs, output):
    """
    This computes unique information as I(input : output | other_inputs) in a distribution
    which matches input-output marginals, but otherwise maximizes I(inputs : output).

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
    broja = BROJAOptimizer(d.copy(), inputs, output)
    broja.optimize(nhops=15)
    d_opt = broja.construct_dist()
    uniques = {}
    for input_ in inputs:
        invar = [broja._var_map[input_]]
        outvar = [broja._var_map[output]]
        others = [i for i, var in enumerate(inputs) if var != input_]
        uniques[input_] = coinformation(d_opt, [invar, outvar], others)
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
