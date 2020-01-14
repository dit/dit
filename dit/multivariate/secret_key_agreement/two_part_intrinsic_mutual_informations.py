"""
The tightest known upper bound on two-way secret key agreement rate.
"""

from .base_skar_optimizers import BaseTwoPartIntrinsicMutualInformation
from ... import Distribution


__all__ = [
    'two_part_intrinsic_total_correlation',
    'two_part_intrinsic_dual_total_correlation',
    'two_part_intrinsic_CAEKL_mutual_information',
]


class TwoPartIntrinsicTotalCorrelation(BaseTwoPartIntrinsicMutualInformation):
    """
    Compute the two part intrinsic total correlation.
    """
    name = 'total correlation'

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


two_part_intrinsic_total_correlation = TwoPartIntrinsicTotalCorrelation.functional()


class TwoPartIntrinsicDualTotalCorrelation(BaseTwoPartIntrinsicMutualInformation):
    """
    Compute the two part intrinsic dual total correlation.
    """
    name = 'dual total correlation'

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
        dtc : float
            The dual total correlation.
        """
        return self._dual_total_correlation(rvs, crvs)


two_part_intrinsic_dual_total_correlation = TwoPartIntrinsicDualTotalCorrelation.functional()


class TwoPartIntrinsicCAEKLMutualInformation(BaseTwoPartIntrinsicMutualInformation):
    """
    Compute the two part intrinsic CAEKL mutual information.
    """
    name = 'CAEKL mutual information'

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
        caekl : float
            The CAEKL mutual information.
        """
        return self._caekl_mutual_information(rvs, crvs)


two_part_intrinsic_CAEKL_mutual_information = TwoPartIntrinsicCAEKLMutualInformation.functional()


def two_part_intrinsic_mutual_information_constructor(func):  # pragma: no cover
    """
    Given a measure of shared information, construct an optimizer which computes
    its ``two part intrinsic'' form.

    Parameters
    ----------
    func : func
        A function which computes the information shared by a set of variables.
        It must accept the arguments `rvs' and `crvs'.

    Returns
    -------
    TPIMI : BaseTwoPartIntrinsicMutualInformation
        An two part intrinsic mutual information optimizer using `func` as the
        measure of multivariate mutual information.

    Notes
    -----
    Due to the casting to a Distribution for processing, optimizers constructed
    using this function will be significantly slower than if the objective were
    written directly using the joint probability ndarray.
    """
    class TwoPartIntrinsicMutualInformation(BaseTwoPartIntrinsicMutualInformation):
        name = func.__name__

        def measure(self, rvs, crvs):  # pragma: no cover
            """
            Dummy method.
            """
            pass

        def objective(self, x):
            pmf = self.construct_joint(x)
            d = Distribution.from_ndarray(pmf)
            mi = func(d, rvs=[[rv] for rv in self._rvs], crvs=self._j)
            cmi1 = self._conditional_mutual_information(self._u, self._j, self._v)(pmf)
            cmi1 = self._conditional_mutual_information(self._u, self._crvs, self._v)(pmf)
            return mi + cmi1 - cmi2

    TwoPartIntrinsicMutualInformation.__doc__ = \
    """
    Compute the two part intrinsic {name}.
    """.format(name=func.__name__)

    docstring = \
    """
    Compute the {name}.

    Parameters
    ----------
    x : np.ndarray
        An optimization vector.

    Returns
    -------
    obj : float
        The {name}-based objective function.
    """.format(name=func.__name__)
    try:
        # python 2
        TwoPartIntrinsicMutualInformation.objective.__func__.__doc__ = docstring
    except AttributeError:
        # python 3
        TwoPartIntrinsicMutualInformation.objective.__doc__ = docstring

    return TwoPartIntrinsicMutualInformation
