"""
One-way secret key agreement rate. This is the rate at which Alice and Bob can
agree upon a secret key with Eve eavesdropping, if only Alice is permitted
to publicly communicate.
"""

from __future__ import division

from abc import abstractmethod

from ...algorithms import BaseAuxVarOptimizer
from ...utils import unitful


__all__ = [
    'one_way_skar',
]


class BaseOneWaySKAR(BaseAuxVarOptimizer):
    """
    Compute lower bounds on the secret key agreement rate of the form:

        max_{V - U - X - YZ} objective()
    """

    def __init__(self, dist, rv_x=None, rv_y=None, rv_z=None, rv_mode=None, bound_u=None, bound_v=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rv_x : iterable
            The variables to consider `X`.
        rv_y : iterable
            The variables to consider `Y`.
        rv_z : iterable
            The variables to consider `Z`.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        bound_u : int, None
            Specifies a bound on the size of the auxiliary random variable. If
            None, then the theoretical bound is used.
        bound_v : int, None
            Specifies a bound on the size of the auxiliary random variable. If
            None, then the theoretical bound is used.
        """
        super(BaseOneWaySKAR, self).__init__(dist, [rv_x, rv_y], rv_z, rv_mode=rv_mode)

        theoretical_bound_u = self._get_u_bound()
        bound_u = min(bound_u, theoretical_bound_u) if bound_u else theoretical_bound_u

        theoretical_bound_v = self._get_v_bound()
        bound_v = min(bound_v, theoretical_bound_v) if bound_v else theoretical_bound_v

        self._construct_auxvars([({0}, bound_u), ({3}, bound_v)])
        self._x = {0}
        self._y = {1}
        self._z = {2}
        self._u = {3}
        self._v = {4}
        self._default_hops *= 2

    @abstractmethod
    def _get_u_bound(self):
        """
        Bound of |U|

        Returns
        -------
        bound : int
            The bound
        """
        pass

    @abstractmethod
    def _get_v_bound(self):
        """
        Bound of |V|

        Returns
        -------
        bound : int
            The bound
        """
        pass


class OneWaySKAR(BaseOneWaySKAR):
    """
    Compute the one-way secret key agreement rate:
        max_{V - U - X - YZ} I[U:Y|V] - I[U:Z|V]
    """

    def _get_u_bound(self):
        """
        |U| <= |X|
        """
        return self._shape[0]

    def _get_v_bound(self):
        """
        |U| <= |X|^2
        """
        return self._shape[0]**2

    def _objective(self):
        """
        The multivariate mutual information to minimize.

        Returns
        -------
        obj : func
            The objective function.
        """
        # I[U:Y|V]
        cmi_a = self._conditional_mutual_information(self._u, self._y, self._v)
        # I[U:Z|V]
        cmi_b = self._conditional_mutual_information(self._u, self._z, self._v)

        def objective(self, x):
            """
            Compute I[U:Y]/I[U:X]

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

            a = cmi_a(pmf)
            b = cmi_b(pmf)

            return -(a - b)

        return objective


@unitful
def one_way_skar(dist, X, Y, Z, rv_mode=None, niter=None, bound_u=None, bound_v=None):
    """
    Compute the secret key agreement rate constrained to one-way communication.

    Parameters
    -----------
    dist : Distribution
        The distribution of interest.
    X : iterable
        The indices to consider as the X variable, Alice.
    Y : iterable
        The indices to consider as the Y variable, Bob.
    Z : iterable
        The indices to consider as the Z variable, Eve.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.
    niter : int, None
        The number of hops to perform during optimization.
    bound_u : int, None
        The bound to use on the size of the variable U. If none, use the
        theoretical bound of |X|.
    bound_v : int, None
        The bound to use on the size of the variable V. If none, use the
        theoretical bound of |X|^2.

    Returns
    -------
    owskar : float
        The necessary intrinsic mutual information.
    """
    values = []
    for bound in {1, 2, 3, bound_v}:
        nimi = OneWaySKAR(dist, X, Y, Z, rv_mode=rv_mode, bound_u=bound_u, bound_v=bound)
        nimi.optimize(niter=niter)
        values.append(-nimi.objective(nimi._optima))

    return max(values)
