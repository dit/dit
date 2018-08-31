"""
The I_downarrow unique measure, proposed by Griffith et al, and shown to be inconsistent.

The idea is to measure unique information as the intrinsic mutual information between
and input and the output, given the other inputs. It turns out that these unique values
are inconsistent, in that they produce differing redundancy values.
"""

from __future__ import division

import numpy as np

from .pid import BaseUniquePID

from ..exceptions import ditException
from ..multivariate.secret_key_agreement import (
    no_communication_skar,
    interactive_skar,
    intrinsic_mutual_information,
    minimal_intrinsic_mutual_information,
    necessary_intrinsic_mutual_information,
    one_way_skar,
    two_part_intrinsic_mutual_information,
)
from ..utils import flatten

__all__ = [
    'PID_SKAR_nw',
    'PID_SKAR_owa',
    'PID_SKAR_owb',
    'PID_SKAR_tw',
]


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
    def _measure(d, inputs, output, niter=25, bound=None):
        """
        This computes unique information as S(X_0 >-< Y || X_1).

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_SKAR for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        i_skar_nw : dict
            The value of I_SKAR_nw for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = no_communication_skar(d, input_, output, others)
        return uniques


class PID_SKAR_owa(BaseUniquePID):
    """
    The one-way secret key agreement rate partial information decomposition,
    source to target.
    """
    _name = "I_>->"

    @staticmethod
    def _measure(d, inputs, output, niter=25, bound=None):
        """
        This computes unique information as S(X_0 >-> Y || X_1).

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_SKAR for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        i_skar_owa : dict
            The value of I_SKAR_owa for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = one_way_skar(d, input_, output, others)
        return uniques


class PID_SKAR_owb(BaseUniquePID):
    """
    The one-way secret key agreement rate partial information decomposition,
    target to source.
    """
    _name = "I_<-<"

    @staticmethod
    def _measure(d, inputs, output, niter=25, bound=None):
        """
        This computes unique information as S(X_0 <-< Y || X_1).

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_SKAR for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        i_skar_owb : dict
            The value of I_SKAR_owb for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = one_way_skar(d, output, input_, others)
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
    def _measure(d, inputs, output, niter=25, bound=None):
        """
        This computes unique information as S(X_0 <-> Y || X_1), when possible.

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_SKAR for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        i_skar_tw : dict
            The value of I_SKAR_tw for each individual input.
        """
        def bounds(input_, output, others):
            lower = no_communication_skar(d, input_, output, others)
            upper = intrinsic_mutual_information(d, [input_, output], others)
            yield lower, upper
            lower = necessary_intrinsic_mutual_information(d, [input_, output], others, niter=niter, bound_u=bound, bound_v=bound)
            yield lower, upper
            upper = minimal_intrinsic_mutual_information(d, [input_, output], others, niter=niter, bounds=(bound,))
            yield lower, upper
            lower = interactive_skar(d, [input_, output], others, niter=niter)
            yield lower, upper
            lower = interactive_skar(d, [input_, output], others, niter=niter, rounds=3)
            yield lower, upper
            upper = two_part_intrinsic_mutual_information(d, [input_, output], others, niter=niter, bound_j=2, bound_u=2, bound_v=2)
            yield lower, upper

        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            for lower, upper in bounds(input_, output, others):
                if np.isclose(lower, upper):
                    uniques[input_] = lower
                    break
            else:
                print(lower)
                print(upper)
                uniques[input_] = np.nan
        return uniques
