"""
Miscellaneous distributions illustrating particular phenomena.
"""

from __future__ import division

from .. import Distribution

__all__ = [
    'gk_pos_i_neg',
]


# has K(d) > 0 while I(d) < 0
gk_pos_i_neg = Distribution(['000', '011', '101', '110', '222'], [7/32]*4 + [1/8])
