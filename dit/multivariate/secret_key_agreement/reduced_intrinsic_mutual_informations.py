"""
The reduced intrinsic mutual information.

Note: this code is nowhere near efficient enough to actually run. Don't try it.
"""

from .base_skar_optimizers import BaseReducedIntrinsicMutualInformation
from .intrinsic_mutual_informations import (intrinsic_total_correlation,
                                            intrinsic_dual_total_correlation,
                                            intrinsic_caekl_mutual_information,
                                            )

__all__ = (
    'reduced_intrinsic_total_correlation',
    'reduced_intrinsic_dual_total_correlation',
    'reduced_intrinsic_CAEKL_mutual_information',
)


class ReducedIntrinsicTotalCorrelation(BaseReducedIntrinsicMutualInformation):
    """
    Compute the reduced intrinsic total correlation.
    """

    name = 'total correlation'
    measure = staticmethod(intrinsic_total_correlation)


reduced_intrinsic_total_correlation = ReducedIntrinsicTotalCorrelation.functional()


class ReducedIntrinsicDualTotalCorrelation(BaseReducedIntrinsicMutualInformation):
    """
    Compute the reduced intrinsic dual total correlation.
    """

    name = 'dual total correlation'
    measure = staticmethod(intrinsic_dual_total_correlation)


reduced_intrinsic_dual_total_correlation = ReducedIntrinsicDualTotalCorrelation.functional()


class ReducedIntrinsicCAEKLMutualInformation(BaseReducedIntrinsicMutualInformation):
    """
    Compute the reduced intrinsic CAEKL mutual information.
    """

    name = 'CAEKL mutual information'
    measure = staticmethod(intrinsic_caekl_mutual_information)


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
        measure = staticmethod(func)

    ReducedIntrinsicMutualInformation.__doc__ = \
    """
    Compute the reduced intrinsic {name}.
    """.format(name=func.__name__)

    docstring = \
    """
    Compute the {name}.

    Parameters
    ----------
    d : Distribution
        The distribution to compute {name} of.

    Returns
    -------
    imi : float
        The {name}.
    """.format(name=func.__name__)
    try:
        # python 2
        ReducedIntrinsicMutualInformation.objective.__func__.__doc__ = docstring
    except AttributeError:
        # python 3
        ReducedIntrinsicMutualInformation.objective.__doc__ = docstring

    return ReducedIntrinsicMutualInformation
