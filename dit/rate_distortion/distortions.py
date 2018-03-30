"""
Distortion measures for use with the Blahut-Arimoto algorithms.
"""

from __future__ import division

import numpy as np


def hamming_distortion(p_xy):
    """
    """
    distortion = 1 - np.eye(*p_xy.shape)
    return distortion


def residual_entropy_distortion(p_xy):
    """
    """
    h_x_y = -np.log2(p_xy / p_xy.sum(axis=0, keepdims=True))
    h_y_x = -np.log2(p_xy / p_xy.sum(axis=1, keepdims=True))
    distortion = h_x_y + h_y_x
    return distortion
