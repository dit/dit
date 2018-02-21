"""
Distributions useful for illustrating the behavior of the various intrinsic measures.
"""

from __future__ import division

from .. import Distribution

__all__ = [
    'intrinsic_1',
    'intrinsic_2',
    'intrinsic_3',
]


# from the intrinsic information paper
intrinsic_1 = Distribution(['000', '011', '101', '110', '222', '333'], [1/8]*4 + [1/4]*2)
intrinsic_1.secret_rate = 0.0


# from the reduced intrinsic information paper
intrinsic_2 = Distribution(['000', '011', '101', '110', '220', '331'], [1/8]*4 + [1/4]*2)
intrinsic_2.secret_rate = 1.0


# from the minimal intrinsic information paper, with alpha_1 = 1/3 and alpha_2 = 1/2
intrinsic_3 = Distribution(['000', '001', '012', '013', '102', '103', '110', '111', '220', '221', '332', '333'],
                           [1/24,  1/12,  1/24,  1/12,  1/24,  1/12,  1/24,  1/12,  1/8,   1/8,   1/8,   1/8])
intrinsic_3.secret_rate = 0.97927916037609197
