"""

"""

from __future__ import division

from .base_intrinsic_information import BaseMoreIntrinsicMutualInformation
from ... import Distribution

__all__ = ['minimal_intrinsic_total_correlation',
           'minimal_intrinsic_dual_total_correlation',
           'minimal_intrinsic_CAEKL_mutual_information',
          ]


class BaseMinimalIntrinsicMutualInformation(BaseMoreIntrinsicMutualInformation):
    """
    Compute the minimal intrinsic mutual information, a lower bound on the secret
    key agreement rate:

        I[X : Y \downarrow\downarrow\downarrow Z] = min_U I[X:Y|U] + I[XY:U|Z]
    """

    type = "minimal"

    def _objective(self):
        """
        Compute I[X:Y|U] + I[XY:U|Z], or its multivariate analog.

        Returns
        -------
        obj : func
            The objective function.
        """
        mmi = self.measure(self._rvs, self._arvs)
        cmi = self._conditional_mutual_information(self._rvs, self._arvs, self._crvs)

        def objective(self, x):
            """
            Compute I[X:Y|U] + I[XY:U|Z]

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

            # I[X:Y|U]
            a = mmi(pmf)

            # I[XY:U|Z]
            b = cmi(pmf)

            return a + b

        return objective


class MinimalIntrinsicTotalCorrelation(BaseMinimalIntrinsicMutualInformation):
    """
    Compute the minimal intrinsic total correlation.
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


minimal_intrinsic_total_correlation = MinimalIntrinsicTotalCorrelation.functional()


class MinimalIntrinsicDualTotalCorrelation(BaseMinimalIntrinsicMutualInformation):
    """
    Compute the minimal intrinsic dual total correlation.
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


minimal_intrinsic_dual_total_correlation = MinimalIntrinsicDualTotalCorrelation.functional()


class MinimalIntrinsicCAEKLMutualInformation(BaseMinimalIntrinsicMutualInformation):
    """
    Compute the minimal intrinsic CAEKL mutual information.
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


minimal_intrinsic_CAEKL_mutual_information = MinimalIntrinsicCAEKLMutualInformation.functional()


def minimal_intrinsic_mutual_information_constructor(func):
    """
    Given a measure of shared information, construct an optimizer which computes
    its ``minimal intrinsic'' form.

    Parameters
    ----------
    func : func
        A function which computes the information shared by a set of variables.
        It must accept the arguments `rvs' and `crvs'.

    Returns
    -------
    MIMI : BaseMinimalIntrinsicMutualInformation
        An minimal intrinsic mutual information optimizer using `func` as the
        measure of multivariate mutual information.

    Notes
    -----
    Due to the casting to a Distribution for processing, optimizers constructed
    using this function will be significantly slower than if the objective were
    written directly using the joint probability ndarray.
    """
    class MinimalIntrinsicMutualInformation(BaseMinimalIntrinsicMutualInformation):
        name = func.__name__

        def measure(self, rvs, crvs):  # pragma: no cover
            """
            Dummy method.
            """
            pass

        def objective(self, x):
            pmf = self.construct_joint(x)
            d = Distribution.from_ndarray(pmf)
            mi = func(d, rvs=[[rv] for rv in self._rvs], crvs=self._arvs)
            cmi = self._conditional_mutual_information(self._rvs, self._arvs, self._crvs)(pmf)
            return mi + cmi

    MinimalIntrinsicMutualInformation.__doc__ = \
    """
    Compute the minimal intrinsic {name}.
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
        MinimalIntrinsicMutualInformation.objective.__func__.__doc__ = docstring
    except AttributeError:
        # python 3
        MinimalIntrinsicMutualInformation.objective.__doc__ = docstring

    return MinimalIntrinsicMutualInformation
