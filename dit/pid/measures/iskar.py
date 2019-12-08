# -*- coding: utf-8 -*-

"""
The I_downarrow unique measure, proposed by Griffith et al, and shown to be inconsistent.

The idea is to measure unique information as the intrinsic mutual information between
and input and the output, given the other inputs. It turns out that these unique values
are inconsistent, in that they produce differing redundancy values.
"""

import numpy as np

from ..pid import BaseUniquePID

from ...exceptions import ditException
from ...multivariate.secret_key_agreement import (
    no_communication_skar,
    one_way_skar,
    two_way_skar,
)
from ...utils import flatten

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
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = two_way_skar(d, [input_, output], others)
        return uniques
