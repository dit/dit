"""
The redundancy measure of Sigtermans based on causal tensors.
"""

import numpy as np

from ...exceptions import ditException
from ..pid import BaseBivariatePID

__all__ = ("PID_CT",)


def i_triangle(d, source_0, source_1, target):
    """
    Compute the path information, and if it is direct or not.

    Parameters
    ----------

    Returns
    -------
    direct_path : Bool
        Whether Y -> Z exists or not.
    """
    d = d.coalesce([source_0, source_1, target])
    d.make_dense()

    p_xyz = d.pmf.reshape([len(a) for a in d.alphabet])

    p_xy = np.nansum(p_xyz, axis=(2,), keepdims=True)
    p_xz = np.nansum(p_xyz, axis=(1,), keepdims=True)
    p_yz = np.nansum(p_xyz, axis=(0,), keepdims=True)
    p_x = np.nansum(p_xyz, axis=(1, 2), keepdims=True)
    p_y = np.nansum(p_xyz, axis=(0, 2), keepdims=True)
    p_z = np.nansum(p_xyz, axis=(0, 1), keepdims=True)

    A = p_xy / p_x
    B = p_yz / p_y
    C = p_xz / p_x
    Add = p_xy / p_y

    cascade_xyz = np.nansum(A * B, axis=(1,), keepdims=True)
    cascade_yxz = np.nansum(Add * C, axis=(0,), keepdims=True)

    direct_yz = abs(B - cascade_yxz).sum() > 1e-6
    direct_xz = abs(C - cascade_xyz).sum() > 1e-6

    path_info_xyz = np.nansum((p_x * cascade_xyz) * np.log2(cascade_xyz / p_z))
    path_info_yxz = np.nansum((p_y * cascade_yxz) * np.log2(cascade_yxz / p_z))

    if not (direct_xz ^ direct_yz):
        return min(path_info_xyz, path_info_yxz)
    elif direct_xz:
        return path_info_yxz
    elif direct_yz:
        return path_info_xyz
    else:
        raise ditException("Something went wrong...")


class PID_CT(BaseBivariatePID):
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
            msg = f"This method needs exact two sources, {len(sources)} given."
            raise ditException(msg)

        return i_triangle(d, sources[0], sources[1], target)
