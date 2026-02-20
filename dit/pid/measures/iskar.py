"""
The I_downarrow unique measure, proposed by Griffith et al, and shown to be inconsistent.

The idea is to measure unique information as the intrinsic mutual information between
and source and the target, given the other sources. It turns out that these unique values
are inconsistent, in that they produce differing redundancy values.
"""

from ...multivariate.secret_key_agreement import (
    no_communication_skar,
    one_way_skar,
    two_way_skar,
)
from ...utils import flatten
from ..pid import BaseUniquePID

__all__ = (
    "PID_SKAR_nw",
    "PID_SKAR_owa",
    "PID_SKAR_owb",
    "PID_SKAR_tw",
)


class PID_SKAR_nw(BaseUniquePID):
    """
    The two-way secret key agreement rate partial information decomposition.

    Notes
    -----
    This method progressively utilizes better bounds on the SKAR, and if even when using
    the tightest bounds does not result in a singular SKAR, nan is returned.
    """

    _name = "I_>-<"

    @staticmethod
    def _measure(d, sources, target, niter=25, bound=None):
        """
        This computes unique information as S(X_0 >-< Y || X_1).

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_SKAR for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        i_skar_nw : dict
            The value of I_SKAR_nw for each individual source.
        """
        uniques = {}
        for source in sources:
            others = list(sources)
            others.remove(source)
            others = list(flatten(others))
            uniques[source] = no_communication_skar(d, source, target, others)
        return uniques


class PID_SKAR_owa(BaseUniquePID):
    """
    The one-way secret key agreement rate partial information decomposition,
    source to target.
    """

    _name = "I_>->"

    @staticmethod
    def _measure(d, sources, target, niter=25, bound=None):
        """
        This computes unique information as S(X_0 >-> Y || X_1).

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_SKAR for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        i_skar_owa : dict
            The value of I_SKAR_owa for each individual source.
        """
        uniques = {}
        for source in sources:
            others = list(sources)
            others.remove(source)
            others = list(flatten(others))
            uniques[source] = one_way_skar(d, source, target, others)
        return uniques


class PID_SKAR_owb(BaseUniquePID):
    """
    The one-way secret key agreement rate partial information decomposition,
    target to source.
    """

    _name = "I_<-<"

    @staticmethod
    def _measure(d, sources, target, niter=25, bound=None):
        """
        This computes unique information as S(X_0 <-< Y || X_1).

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_SKAR for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        i_skar_owb : dict
            The value of I_SKAR_owb for each individual source.
        """
        uniques = {}
        for source in sources:
            others = list(sources)
            others.remove(source)
            others = list(flatten(others))
            uniques[source] = one_way_skar(d, target, source, others)
        return uniques


class PID_SKAR_tw(BaseUniquePID):
    """
    The two-way secret key agreement rate partial information decomposition.

    Notes
    -----
    This method progressively utilizes better bounds on the SKAR, and if even when using
    the tightest bounds does not result in a singular SKAR, nan is returned.
    """

    _name = "I_<->"

    @staticmethod
    def _measure(d, sources, target, niter=25, bound=None):
        """
        This computes unique information as S(X_0 <-> Y || X_1), when possible.

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_SKAR for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        i_skar_tw : dict
            The value of I_SKAR_tw for each individual source.
        """
        uniques = {}
        for source in sources:
            others = list(sources)
            others.remove(source)
            others = list(flatten(others))
            uniques[source] = two_way_skar(d, [source, target], others)
        return uniques
