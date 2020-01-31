"""
The redundancy measure of Sigtermans based on causal tensors.
"""

import numpy as np

from ...exceptions import ditException
from ..pid import BaseBivariatePID


__all__ = (
    'PID_Triangle',
)


def path_information(d, X, Y, Z):
    """
    Compute the indirect path information from X to Z mediated by Y.
    """
    d = d.copy().coalesce([X, Y, Z])
    d.make_dense()
    p_xyz = d.pmf.reshape([len(a) for a in d.alphabet])
    p_xy = np.nansum(p_xyz, axis=2, keepdims=True)
    p_yz = np.nansum(p_xyz, axis=0, keepdims=True)
    p_x = np.nansum(p_xyz, axis=(1, 2), keepdims=True)
    p_y = np.nansum(p_xyz, axis=(0, 2), keepdims=True)
    p_z = np.nansum(p_xyz, axis=(0, 1), keepdims=True)
    A = p_xy / p_x
    B = p_yz / p_y
    return np.nansum(p_xyz * np.log2(np.nansum(A * B, axis=1, keepdims=True) / p_z))


class PID_Triangle(BaseBivariatePID):
    """
    The bivariate PID defined by Sigtermans using causal tensors.
    """

    _name = "I_△"

    @staticmethod
    def _measure(d, sources, target):
        """
        The PID measure of Sigtermans based on causal tensors.
        """
        if len(sources) != 2:  # pragma: no cover
            msg = "This method needs exact two sources, {} given.".format(len(sources))
            raise ditException(msg)

        path_0 = path_information(d, sources[0], sources[1], target)
        path_1 = path_information(d, sources[1], sources[0], target)
        return min(path_0, path_1)


class PID_Triangle2(BaseBivariatePID):
    """
    The bivariate PID defined by Sigtermans using causal tensors.
    """

    _name = "I_▽"

    @staticmethod
    def _measure(d, sources, target):
        """
        A PID measure inspired by that of Sigtermans based on causal tensors.
        """
        if len(sources) != 2:  # pragma: no cover
            msg = "This method needs exact two sources, {} given.".format(len(sources))
            raise ditException(msg)

        path_0 = path_information(d, target, sources[1], sources[0])
        path_1 = path_information(d, target, sources[0], sources[1])
        return min(path_0, path_1)
