"""
The I_BROJA unique measure, as proposed by the BROJA team.
"""

from __future__ import division

from .pid import BaseUniquePID

from ..algorithms import BaseConvexOptimizer
from ..algorithms.distribution_optimizers import BaseDistOptimizer, BROJABivariateOptimizer
from ..multivariate import coinformation


class BROJAOptimizer(BaseDistOptimizer, BaseConvexOptimizer):
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
        super(BROJAOptimizer, self).__init__(dist, marginals=constraints, rv_mode=rv_mode)
        self._input = {0}
        self._others = {1}
        self._output = {2}

    def _objective(self):
        """
        Minimize I(input:output|others).

        Parameters
        ----------
        x : np.ndarray
            Optimization vector.

        Returns
        -------
        obj : func
            The objective.
        """
        cmi = self._conditional_mutual_information(self._input, self._output, self._others)

        def objective(self, x):
            """
            Compute I[input : output | others]

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            pmf = self.construct_joint(x)
            return cmi(pmf)

        return objective


def i_broja(d, inputs, output, maxiter=1000):
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
        broja.optimize(niter=1, maxiter=maxiter)
        opt_dist = broja.construct_dist()
        uniques[inputs[0]] = coinformation(opt_dist, [[0], [2]], [1])
        uniques[inputs[1]] = coinformation(opt_dist, [[1], [2]], [0])
    else:
        for input_ in inputs:
            others = sum([i for i in inputs if i != input_], ())
            dm = d.coalesce([input_, others, output])
            broja = BROJAOptimizer(dm, (0,), ((1,),), (2,))
            broja.optimize(niter=1, maxiter=maxiter)
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
