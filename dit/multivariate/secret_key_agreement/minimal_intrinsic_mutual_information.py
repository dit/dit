"""

"""

from __future__ import division

from abc import abstractmethod

import numpy as np

from .base_intrinsic_information import BaseMoreIntrinsicMutualInformation
from ... import Distribution
from ...shannon import entropy_pmf as h
from ...utils import partitions

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

    @abstractmethod
    def measure(self, pmf, rvs, crvs): # pragma: no cover
        """
        """
        pass

    def objective(self, x):
        """
        The multivariate mutual information to minimize.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        obj : float
            The value of the objective function.
        """
        pmf = self.construct_joint(x)

        # I[X:Y|U]
        a = self.measure(pmf, self._rvs, self._U)

        # I[XY:U|Z]
        b = self._conditional_mutual_information(pmf, self._rvs, self._U, self._crvs)

        return a + b


class MinimalIntrinsicTotalCorrelation(BaseMinimalIntrinsicMutualInformation):
    """
    Compute the minimal intrinsic total correlation.
    """
    name = 'total correlation'

    def measure(self, joint, rvs, crvs):
        """
        The total correlation.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        tc : float
            The total correlation.
        """
        others = self._all_vars - (rvs | crvs)
        joint = joint.sum(axis=tuple(others))

        margs = [joint.sum(axis=tuple(rvs - {rv})) for rv in rvs]
        crv = joint.sum(axis=tuple(rvs))

        a = sum(h(marg.ravel()) for marg in margs)
        b = h(joint.ravel())
        c = h(crv.ravel())

        tc = a - b - (len(rvs) - 1) * c

        return tc

minimal_intrinsic_total_correlation = MinimalIntrinsicTotalCorrelation.functional()


class MinimalIntrinsicDualTotalCorrelation(BaseMinimalIntrinsicMutualInformation):
    """
    Compute the minimal intrinsic dual total correlation.
    """
    name = 'dual total correlation'

    def measure(self, joint, rvs, crvs):
        """
        The dual total correlation, also known as the binding information.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        dtc : float
            The dual total correlation.
        """
        others = self._all_vars - (rvs | crvs)
        joint = joint.sum(axis=tuple(others))

        margs = [joint.sum(axis=(rv,)) for rv in rvs]
        crv = joint.sum(axis=tuple(rvs))

        a = sum(h(marg.ravel()) for marg in margs)
        b = h(joint.ravel())
        c = h(crv.ravel())

        dtc = a - (len(rvs) - 1) * b - c

        return dtc

minimal_intrinsic_dual_total_correlation = MinimalIntrinsicDualTotalCorrelation.functional()


class MinimalIntrinsicCAEKLMutualInformation(BaseMinimalIntrinsicMutualInformation):
    """
    Compute the minimal intrinsic CAEKL mutual information.
    """
    name = 'CAEKL mutual information'

    def measure(self, joint, rvs, crvs):
        """
        The CAEKL mutual information.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        caekl : float
            The CAEKL mutual information.
        """
        others = self._all_vars - (rvs|crvs)
        joint = joint.sum(axis=tuple(others))
        crv = joint.sum(axis=tuple(rvs))

        H_crv = h(crv.ravel())
        H = h(joint.ravel()) - H_crv

        def I_P(part):
            margs = [ joint.sum(axis=tuple(rvs - p)) for p in part ]
            a = sum(h(marg.ravel()) - H_crv for marg in margs)
            return (a - H)/(len(part) - 1)

        parts = [p for p in partitions(rvs) if len(p) > 1]

        caekl = min(I_P(p) for p in parts)

        return caekl

minimal_intrinsic_CAEKL_mutual_information = MinimalIntrinsicCAEKLMutualInformation.functional()


def minimal_intrinsic_mutual_information(func):
    """
    Given a measure of shared information, construct an optimizer which computes
    its ``minimal intrinsic'' form.

    Parameters
    ----------
    func : function
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

        def measure(self, joint, rvs, crvs):
            d = Distribution.from_ndarray(joint)
            mi = func(d, rvs=rvs, crvs=crvs)
            return mi

    MinimalIntrinsicMutualInformation.__doc__ = \
    """
    Compute the minimal intrinsic {name}.
    """.format(name=func.__name__)

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
        MinimalIntrinsicMutualInformation.objective.__func__.__doc__ = docstring
    except AttributeError:
        # python 3
        MinimalIntrinsicMutualInformation.objective.__doc__ = docstring

    return MinimalIntrinsicMutualInformation
