"""
The reduced intrinsic mutual information.
"""

from ...distribution import Distribution
from .base_skar_optimizers import BaseReducedIntrinsicMutualInformation

__all__ = (
    "reduced_intrinsic_total_correlation",
    "reduced_intrinsic_dual_total_correlation",
    "reduced_intrinsic_CAEKL_mutual_information",
)


class ReducedIntrinsicTotalCorrelation(BaseReducedIntrinsicMutualInformation):
    """
    Compute the reduced intrinsic total correlation.
    """

    name = "total correlation"

    def measure(self, rvs, crvs):
        """
        The total correlation.

        Parameters
        ----------
        rvs : iterable of iterables
            The random variables.
        crvs : iterable
            The variables to condition on.

        Returns
        -------
        tc : func
            The total correlation.
        """
        return self._total_correlation(rvs, crvs)

    def _objective_gradient(self):
        """Gradient of ``T[X:Y:...|Zbar] + H[U]`` w.r.t. the joint."""
        tc_grad = self._total_correlation_grad(self._rvs, self._zbar)
        h_grad = self._entropy_grad(self._u)

        def grad(pmf):
            return tc_grad(pmf) + h_grad(pmf)

        return grad


reduced_intrinsic_total_correlation = ReducedIntrinsicTotalCorrelation.functional()


class ReducedIntrinsicDualTotalCorrelation(BaseReducedIntrinsicMutualInformation):
    """
    Compute the reduced intrinsic dual total correlation.
    """

    name = "dual total correlation"

    def measure(self, rvs, crvs):
        """
        The dual total correlation, also known as the binding information.

        Parameters
        ----------
        rvs : iterable of iterables
            The random variables.
        crvs : iterable
            The variables to condition on.

        Returns
        -------
        dtc : func
            The dual total correlation.
        """
        return self._dual_total_correlation(rvs, crvs)

    def _objective_gradient(self):
        """Gradient of ``B[X:Y:...|Zbar] + H[U]`` w.r.t. the joint."""
        dtc_grad = self._dual_total_correlation_grad(self._rvs, self._zbar)
        h_grad = self._entropy_grad(self._u)

        def grad(pmf):
            return dtc_grad(pmf) + h_grad(pmf)

        return grad


reduced_intrinsic_dual_total_correlation = ReducedIntrinsicDualTotalCorrelation.functional()


class ReducedIntrinsicCAEKLMutualInformation(BaseReducedIntrinsicMutualInformation):
    """
    Compute the reduced intrinsic CAEKL mutual information.
    """

    name = "CAEKL mutual information"

    def measure(self, rvs, crvs):
        """
        The CAEKL mutual information.

        Parameters
        ----------
        rvs : iterable of iterables
            The random variables.
        crvs : iterable
            The variables to condition on.

        Returns
        -------
        caekl : func
            The CAEKL mutual information.
        """
        return self._caekl_mutual_information(rvs, crvs)

    def _objective_gradient(self):
        """Gradient of ``J[X:Y:...|Zbar] + H[U]`` w.r.t. the joint."""
        caekl_grad = self._caekl_mutual_information_grad(self._rvs, self._zbar)
        h_grad = self._entropy_grad(self._u)

        def grad(pmf):
            return caekl_grad(pmf) + h_grad(pmf)

        return grad


reduced_intrinsic_CAEKL_mutual_information = ReducedIntrinsicCAEKLMutualInformation.functional()


def reduced_intrinsic_mutual_information_constructor(func):  # pragma: no cover
    """
    Given a measure of shared information, construct an optimizer which computes
    its ``reduced intrinsic'' form.

    Parameters
    ----------
    func : function
        A function which computes the information shared by a set of variables.
        It must accept the arguments `rvs' and `crvs'.

    Returns
    -------
    RIMI : BaseReducedIntrinsicMutualInformation
        An reduced intrinsic mutual information optimizer using `func` as the
        measure of multivariate mutual information.

    Notes
    -----
    Due to the casting to a Distribution for processing, optimizers constructed
    using this function will be significantly slower than if the objective were
    written directly using the joint probability ndarray.
    """

    class ReducedIntrinsicMutualInformation(BaseReducedIntrinsicMutualInformation):
        name = func.__name__

        def measure(self, rvs, crvs):
            """
            Dummy method.
            """
            pass

        def objective(self, x):
            pmf = self.construct_joint(x)
            d = Distribution.from_ndarray(pmf)
            mi = func(d, rvs=[[rv] for rv in self._rvs], crvs=self._zbar)
            h = self._entropy(self._u)(pmf)
            return mi + h

    ReducedIntrinsicMutualInformation.__doc__ = f"""
    Compute the reduced intrinsic {func.__name__}.
    """

    docstring = f"""
    Compute the {func.__name__}.

    Parameters
    ----------
    x : np.ndarray
        An optimization vector.

    Returns
    -------
    obj : float
        The {func.__name__}-based objective function.
    """
    try:
        # python 2
        ReducedIntrinsicMutualInformation.objective.__func__.__doc__ = docstring
    except AttributeError:
        # python 3
        ReducedIntrinsicMutualInformation.objective.__doc__ = docstring

    return ReducedIntrinsicMutualInformation
