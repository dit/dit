"""
Intrinsic Mutual Informations
"""

from __future__ import division

from .base_intrinsic_information import BaseIntrinsicMutualInformation

__all__ = [
    'intrinsic_total_correlation',
    'intrinsic_dual_total_correlation',
    'intrinsic_caekl_mutual_information',
]


class IntrinsicTotalCorrelation(BaseIntrinsicMutualInformation):
    """
    Compute the intrinsic total correlation.
    """
    name = 'total correlation'

    def _objective(self):
        """
        The total correlation.

        Returns
        -------
        obj : func
            The objective function.
        """
        total_correlation = self._total_correlation(self._rvs, self._arvs)

        def objective(self, x):
            """
            Compute T[X:Y:...|Z]

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
            return total_correlation(pmf)

        return objective


intrinsic_total_correlation = IntrinsicTotalCorrelation.functional()


class IntrinsicDualTotalCorrelation(BaseIntrinsicMutualInformation):
    """
    Compute the intrinsic dual total correlation.
    """
    name = 'dual total correlation'

    def _objective(self):
        """
        The dual total correlation, also known as the binding information.]

        Returns
        -------
        obj : func
            The objective function.
        """
        dual_total_correlation = self._dual_total_correlation(self._rvs, self._arvs)

        def objective(self, x):
            """
            Compute B[X:Y:...|Z]

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
            return dual_total_correlation(pmf)

        return objective


intrinsic_dual_total_correlation = IntrinsicDualTotalCorrelation.functional()


class IntrinsicCAEKLMutualInformation(BaseIntrinsicMutualInformation):
    """
    Compute the intrinsic CAEKL mutual information.
    """
    name = 'CAEKL mutual information'

    def _objective(self):
        """
        The CAEKL mutual information.

        Returns
        -------
        obj : func
            The objective function.
        """
        caekl_mutual_information = self._caekl_mutual_information(self._rvs, self._arvs)

        def objective(self, x):
            """
            Compute B[X:Y:...|Z]

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
            return caekl_mutual_information(pmf)

        return objective


intrinsic_caekl_mutual_information = IntrinsicCAEKLMutualInformation.functional()


def intrinsic_mutual_information_constructor(func):
    """
    Given a measure of shared information, construct an optimizer which computes
    its ``intrinsic'' form.

    Parameters
    ----------
    func : function
        A function which computes the information shared by a set of variables.
        It must accept the arguments `rvs' and `crvs'.

    Returns
    -------
    IMI : BaseIntrinsicMutualInformation
        An intrinsic mutual information optimizer using `func` as the measure of
        multivariate mutual information.

    Notes
    -----
    Due to the casting to a Distribution for processing, optimizers constructed
    using this function will be significantly slower than if the objective were
    written directly using the joint probability ndarray.
    """
    class IntrinsicMutualInformation(BaseIntrinsicMutualInformation):
        name = func.__name__

        def _objective(self):
            """

            Returns
            -------
            obj : func
                The objective function.
            """
            def objective(self, x):
                d = self.construct_distribution(x)
                mi = func(d, rvs=self._true_rvs, crvs=d.rvs[-1])
                return mi

            docstring = \
            """
            Compute the {name}.
        
            Parameters
            ----------
            x : ndarray
                An optimization vector.
        
            Returns
            -------
            mi : float
                The {name}.
            """.format(name=func.__name__)
            try:
                # python 2
                objective.__func__.__doc__ = docstring
            except AttributeError:
                # python 3
                objective.__doc__ = docstring

            return objective

    IntrinsicMutualInformation.__doc__ = \
    """
    Compute the intrinsic {name}.
    """.format(name=func.__name__)

    return IntrinsicMutualInformation
