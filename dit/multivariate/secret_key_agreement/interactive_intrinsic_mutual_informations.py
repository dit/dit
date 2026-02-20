"""
A lower bound on the two-way secret key agreement rate.
"""

from itertools import chain, zip_longest

from ...algorithms import BaseAuxVarOptimizer
from ...utils import unitful
from .._backend import _make_backend_subclass

__all__ = ("interactive_intrinsic_mutual_information",)


# ── Mixin (backend-agnostic logic) ───────────────────────────────────────


class InteractiveSKARMixin:
    """
    Mixin containing interactive SKAR optimizer logic.

    Must be composed with a ``BaseAuxVarOptimizer``-compatible base class.
    """

    def __init__(self, dist, rv_x=None, rv_y=None, rv_z=None, rounds=2, bound_func=None, rv_mode=None):
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
        rounds : int
            The number of communication rounds to utilize. Defaults to 2.
        bound_func : func
            A function over i, |X|, |Y| which returns the bound on variable U_i.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        super().__init__(dist, [rv_x, rv_y], rv_z, rv_mode=rv_mode)

        self._rounds = rounds
        if bound_func is None:
            bound_func = self.bound

        auxvars = []
        for i in range(rounds):
            markov = {i % 2} | set(range(len(self._shape), len(self._shape) + i))
            bound = bound_func(i, self._shape[0], self._shape[1])
            auxvars.append((markov, bound))

        self._construct_auxvars(auxvars)
        self._x, self._y = [{rv} for rv in sorted(self._rvs)]
        self._z = self._crvs

    @staticmethod
    def bound(i, x, y):
        """ """
        i += 1
        return x ** (i // 2 + i % 2) * y ** (i // 2)

    def _objective(self):
        """
        Maximize:
            \\sum I[U_i : Y | U_(0..i)] - I[U_i : Z | U_(0..i)] + (i even)
            \\sum I[U_i : X | U_(0..i)] - I[U_i : Z | U_(0..i)]   (i odd)

        Returns
        -------
        obj : func
            The objective function.
        """
        arvs = sorted(self._arvs)

        evens = [
            (
                self._conditional_mutual_information(self._y, {arvs[i]}, set(arvs[:i])),
                self._conditional_mutual_information(self._crvs, {arvs[i]}, set(arvs[:i])),
            )
            for i in range(0, self._rounds, 2)
        ]
        odds = [
            (
                self._conditional_mutual_information(self._x, {arvs[i]}, set(arvs[:i])),
                self._conditional_mutual_information(self._crvs, {arvs[i]}, set(arvs[:i])),
            )
            for i in range(1, self._rounds, 2)
        ]
        terms = [term for term in chain.from_iterable(zip_longest(evens, odds)) if term]

        def objective(self, x):
            """
            Compute:
                \\sum I[U_i : Y | U_(0..i)] - I[U_i : Z | U_(0..i)] + (i even)
                \\sum I[U_i : X | U_(0..i)] - I[U_i : Z | U_(0..i)]   (i odd)

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

            values = [a(pmf) - b(pmf) for a, b in terms]

            return -max(sum(values[i:]) for i in range(self._rounds))

        return objective


# ── Backward-compatible composed class (numpy backend) ────────────────────


class InteractiveSKAR(InteractiveSKARMixin, BaseAuxVarOptimizer):
    """
    Compute a lower bound on the secret key agreement rate based on interactive
    communication between Alice and Bob.

    Uses the default NumPy / SciPy optimization backend.
    """

    pass


@unitful
def interactive_intrinsic_mutual_information(
    dist, rvs=None, crvs=None, rounds=2, bound_func=None, niter=None, rv_mode=None, backend="numpy"
):
    """
    Compute a lower bound on the secret key agreement rate based on
    interactive communication.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable
        The indices to consider as X (Alice) and Y (Bob).
    crvs : iterable
        The indices to consider as Z (Eve).
    rounds : int
        The number of rounds of communication to allow. Defaults to 2.
    bound_func : func
        A function over i, |X|, |Y| which returns the bound on variable U_i.
    niter : int, None
        The number of hops to perform during optimization.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.
    backend : str
        The optimization backend. One of ``'numpy'`` (default),
        ``'jax'``, or ``'torch'``.

    Returns
    -------
    iskar : float
        The lower bound.
    """
    actual_cls = _make_backend_subclass(InteractiveSKAR, backend)
    iskar = actual_cls(dist, rv_x=rvs[0], rv_y=rvs[1], rv_z=crvs, rounds=rounds, bound_func=bound_func, rv_mode=rv_mode)
    iskar.optimize(niter=niter)
    val = iskar.objective(iskar._optima)
    val = float(val.detach().cpu().item()) if hasattr(val, "detach") else float(val)

    return -val
