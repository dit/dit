"""
The two distributions studied in Multivariate Dependencies Beyond Shannon Information.
"""

from ..distconst import uniform

__all__ = ['dyadic', 'triadic']

dyadic = uniform(['000', '021', '102', '123', '210', '231', '312', '333'])

triadic = uniform(['000', '111', '022', '133', '202', '313', '220', '331'])
