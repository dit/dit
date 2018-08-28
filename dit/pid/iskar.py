"""
The I_downarrow unique measure, proposed by Griffith et al, and shown to be inconsistent.

The idea is to measure unique information as the intrinsic mutual information between
and input and the output, given the other inputs. It turns out that these unique values
are inconsistent, in that they produce differing redundancy values.
"""

from __future__ import division

from .pid import BaseUniquePID

from ..exceptions import ditException
from ..multivariate import (intrinsic_mutual_information,
                            minimal_intrinsic_mutual_information,
                            reduced_intrinsic_mutual_information,
                            total_correlation,
                           )
from ..multivariate.secret_key_agreement.secrecy_capacity import (
    secrecy_capacity,
    )
from ..multivariate.secret_key_agreement.one_way_skar import (
    one_way_skar,
    )
from ..multivariate.secret_key_agreement.trivial_bounds import (
    lower_intrinsic_mutual_information_directed,
    )
from ..utils import flatten


__all__ = [
    'PID_uparrow',
    'PID_double_uparrow',
    'PID_triple_uparrow',
    'PID_downarrow',
    'PID_double_downarrow',
    'PID_triple_downarrow',
    'PID_rectified_downarrow',
]


class PID_uparrow(BaseUniquePID):
    """
    The lower intrinsic mutual information partial information decomposition.

    Notes
    -----
    This decomposition is known to be invalid; that is, the redundancy values
    computed using either unique value are not consistent.
    """
    _name = "I_ua"

    @staticmethod
    def _measure(d, inputs, output):
        """
        This computes unique information as I(input : output \\uparrow other_inputs).

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_downarrow for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        ida : dict
            The value of I_uparrow for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = lower_intrinsic_mutual_information_directed(d, input_, output, others)
        return uniques


class PID_double_uparrow(BaseUniquePID):
    """
    The secrecy capacity partial information decomposition.

    Notes
    -----
    This decomposition is known to be invalid; that is, the redundancy values
    computed using either unique value are not consistent.
    """
    _name = "I_dua"

    @staticmethod
    def _measure(d, inputs, output, niter=None, bound_u=None):
        """
        This computes unique information as I(input : output \\uparrow\\uparrow other_inputs).

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_double_uparrow for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        ida : dict
            The value of I_double_uparrow for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = secrecy_capacity(d, input_, output, others, niter=niter, bound_u=bound_u)
        return uniques


class PID_triple_uparrow(BaseUniquePID):
    """
    The necessary intrinsic mutual information partial information decomposition.

    Notes
    -----
    This decomposition is known to be invalid; that is, the redundancy values
    computed using either unique value are not consistent.
    """
    _name = "I_tua"

    @staticmethod
    def _measure(d, inputs, output, niter=5, bound_u=None, bound_v=None):
        """
        This computes unique information as I(input : output \\uparrow\\uparrow\\uparrow other_inputs).

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_triple_uparrow for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        ida : dict
            The value of I_triple_uparrow for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = one_way_skar(d, input_, output, others,  niter=niter, bound_u=bound_u, bound_v=bound_v)
        return uniques


class PID_downarrow(BaseUniquePID):
    """
    The intrinsic mutual information partial information decomposition.

    Notes
    -----
    This decomposition is known to be invalid; that is, the redundancy values
    computed using either unique value are not consistent.
    """
    _name = "I_da"

    @staticmethod
    def _measure(d, inputs, output, niter=25, bound=None):
        """
        This computes unique information as I(input : output \\downarrow other_inputs).

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_downarrow for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        ida : dict
            The value of I_downarrow for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = intrinsic_mutual_information(d, [input_, output], others, niter=niter, bound=bound)
        return uniques


class PID_double_downarrow(BaseUniquePID):
    """
    The reduced intrinsic mutual information partial information decomposition.
    """
    _name = "I_dda"

    @staticmethod
    def _measure(d, inputs, output, niter=5): # pragma: no cover
        """
        This computes unique information as I(input : output \\Downarrow other_inputs).

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_double_downarrow for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        idda : dict
            The value of I_double_downarrow for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = reduced_intrinsic_mutual_information(d, [input_, output], others,
                                                                niter=niter)
        return uniques


class PID_triple_downarrow(BaseUniquePID):
    """
    The minimal intrinsic mutual information partial information decomposition.
    """
    _name = "I_tda"

    @staticmethod
    def _measure(d, inputs, output, niter=5, bounds=None):
        """
        This computes unique information as I(input : output \\downarrow\\downarrow\\downarrow other_inputs).

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_triple_downarrow for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        itda : dict
            The value of I_triple_downarrow for each individual input.
        """
        uniques = {}
        for input_ in inputs:
            others = list(inputs)
            others.remove(input_)
            others = list(flatten(others))
            uniques[input_] = minimal_intrinsic_mutual_information(d, [input_, output], others,
                                                                niter=niter, bounds=bounds)
        return uniques


################################################################################
# Rectified versions to avoid inconsistent PIDs


class PID_rectified_downarrow(BaseUniquePID):
    """
    The intrinsic mutual information partial information decomposition.

    Notes
    -----
    This decomposition is known to be invalid; that is, the redundancy values
    computed using either unique value are not consistent.
    """
    _name = "I_rda"

    @staticmethod
    def _measure(d, inputs, output, niter=25, bound=None):
        """
        This computes unique information as I(input : output \\downarrow other_inputs).

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_downarrow for.
        inputs : iterable of iterables
            The input variables.
        output : iterable
            The output variable.

        Returns
        -------
        ida : dict
            The value of I_downarrow for each individual input.
        """
        if len(inputs) != 2:
            msg = ""
            raise ditException(msg)

        x0, x1 = inputs
        y = output

        mis = [total_correlation(d, [x0, y]), total_correlation(d, [x1, y])]

        imis = [intrinsic_mutual_information(d, [x0, y], x1, niter=niter, bound=bound),
                intrinsic_mutual_information(d, [x1, y], x0, niter=niter, bound=bound)]

        uniques = {}
        uniques[x0] = max([imis[0], imis[1] + mis[0] - mis[1]])
        uniques[x1] = max([imis[1], imis[0] + mis[1] - mis[0]])

        return uniques
