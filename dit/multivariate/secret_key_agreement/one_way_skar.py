"""
One-way secret key agreement rate. This is the rate at which Alice and Bob can
agree upon a secret key with Eve eavesdropping, if only Alice is permitted
to publicly communicate.
"""

from ...utils import unitful
from .._backend import _make_backend_subclass
from .base_skar_optimizers import BaseOneWaySKAR

__all__ = (
    'one_way_skar',
)


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
        |V| <= |X|^2
        """
        return self._shape[0]**2

    def _objective(self):
        """
        Maximize I[U:Y|V] - I[U:Z|V]

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
            Compute I[U:Y] - I[U:X]

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
def one_way_skar(dist, X, Y, Z, rv_mode=None, niter=None, bound_u=None,
                 bound_v=None, backend='numpy'):
    """
    Compute the secret key agreement rate constrained to one-way communication.

    Parameters
    ----------
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
    backend : str
        The optimization backend. One of ``'numpy'`` (default),
        ``'jax'``, or ``'torch'``.

    Returns
    -------
    owskar : float
        The necessary intrinsic mutual information.
    """
    actual_cls = _make_backend_subclass(OneWaySKAR, backend)
    values = []
    for bound in {1, 2, 3, bound_v}:
        nimi = actual_cls(dist, X, Y, Z, rv_mode=rv_mode, bound_u=bound_u, bound_v=bound)
        nimi.optimize(niter=niter)
        values.append(-nimi.objective(nimi._optima))

    return max(values)
