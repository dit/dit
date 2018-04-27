"""
Distortion measures for use with rate distortion theory.
"""

from __future__ import division

from collections import namedtuple

import numpy as np

from .information_bottleneck import (InformationBottleneck,
                                     InformationBottleneckDivergence,
                                     )
from .rate_distortion import (RateDistortionHamming,
                              RateDistortionMaximumCorrelation,
                              RateDistortionResidualEntropy,
                              )


__all__ = (
    'hamming',
    'residual_entropy',
    'maximum_correlation',
)


Distortion = namedtuple('Distortion', ['name', 'matrix', 'optimizer'])


def hamming_distortion(p_x, p_y_x):
    """
    """
    distortion = 1 - np.eye(*p_y_x.shape)
    return distortion


def residual_entropy_distortion(p_x, p_y_x):
    """
    """
    p_xy = p_x[:, np.newaxis] * p_y_x
    h_x_y = -np.log2(p_xy / p_xy.sum(axis=0, keepdims=True))
    h_y_x = -np.log2(p_xy / p_xy.sum(axis=1, keepdims=True))
    distortion = h_x_y + h_y_x
    return distortion


hamming = Distortion('Hamming', hamming_distortion, RateDistortionHamming)
residual_entropy = Distortion('Residual Entropy', residual_entropy_distortion, RateDistortionResidualEntropy)
maximum_correlation = Distortion('Maximum Correlation', None, RateDistortionMaximumCorrelation)

###############################################################################
# Information Bottleneck-like distortions

IBDistortion = namedtuple('IBDistortion', ['name', 'divergence', 'optimizer'])

