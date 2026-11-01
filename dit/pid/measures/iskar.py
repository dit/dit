"""
The I_downarrow unique measure, proposed by Griffith et al, and shown to be inconsistent.

The idea is to measure unique information as the intrinsic mutual information between
and source and the target, given the other sources. It turns out that these unique values
are inconsistent, in that they produce differing redundancy values.
"""

from itertools import chain

from ...algorithms.optimization import parallel_sweep
from ...multivariate.secret_key_agreement import (
    no_communication_skar,
    one_way_skar,
    two_way_skar,
)
from ..pid import BaseUniquePID

__all__ = (
    "PID_SKAR_nw",
    "PID_SKAR_owa",
    "PID_SKAR_owb",
    "PID_SKAR_tw",
)


def _with_rest(d, sources, source, target):
    """
    The joint distribution of a source, the other sources, and the target, as the
    variables 0, 1 and 2, so that sources sharing variables never repeat an index.

    Parameters
    ----------
    d : Distribution
        The distribution.
    sources : iterable of iterables
        The sources.
    source : tuple
        The source whose unique information is wanted.
    target : iterable
        The target.

    Returns
    -------
    dist : Distribution
        The distribution of (source, the other sources, target).
    """
    rest = tuple(dict.fromkeys(chain(*(other for other in sources if other != source))))
    return d.coalesce([source, rest, target])


class PID_SKAR_nw(BaseUniquePID):
    """
    The no-communication secret key agreement rate partial information decomposition.
    """

    _name = "I_>-<"

    @staticmethod
    def _measure(d, sources, target):
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

        def _run(source, rng):
            return no_communication_skar(_with_rest(d, sources, source, target), [0], [2], [1])

        return dict(zip(sources, parallel_sweep(_run, sources), strict=True))


class PID_SKAR_owa(BaseUniquePID):
    """
    The one-way secret key agreement rate partial information decomposition,
    source to target.
    """

    _name = "I_>->"

    @staticmethod
    def _measure(d, sources, target, niter=None, bound_u=None, bound_v=None):
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
        niter : int, None
            The number of hops of the optimization.
        bound_u, bound_v : int, None
            Bounds on the sizes of the auxiliary variables of the one-way rate.

        Returns
        -------
        i_skar_owa : dict
            The value of I_SKAR_owa for each individual source.
        """
        return {
            source: one_way_skar(
                _with_rest(d, sources, source, target), [0], [2], [1], niter=niter, bound_u=bound_u, bound_v=bound_v
            )
            for source in sources
        }


class PID_SKAR_owb(BaseUniquePID):
    """
    The one-way secret key agreement rate partial information decomposition,
    target to source.
    """

    _name = "I_<-<"

    @staticmethod
    def _measure(d, sources, target, niter=None, bound_u=None, bound_v=None):
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
        niter : int, None
            The number of hops of the optimization.
        bound_u, bound_v : int, None
            Bounds on the sizes of the auxiliary variables of the one-way rate.

        Returns
        -------
        i_skar_owb : dict
            The value of I_SKAR_owb for each individual source.
        """
        return {
            source: one_way_skar(
                _with_rest(d, sources, source, target), [2], [0], [1], niter=niter, bound_u=bound_u, bound_v=bound_v
            )
            for source in sources
        }


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
    def _measure(d, sources, target):
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
        return {source: two_way_skar(_with_rest(d, sources, source, target), [[0], [2]], [1]) for source in sources}
