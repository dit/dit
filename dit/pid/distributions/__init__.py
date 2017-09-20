"""
A variety of distributions exemplifying aspects of the partial information decomposition.
"""

from . import bivariate
from . import trivariate

__all__ = ['bivariates', 'trivariates']


bivariates = {'uniform': bivariate.uni,
              'null': bivariate.null,
              'redundant': bivariate.rdn,
              'simple': bivariate.simple,
              'unique 1': bivariate.unq1,
              'cat': bivariate.cat,
              'synergy': bivariate.syn,
              'diff': bivariate.diff,
              'and': bivariate.and_,
              'reduced or': bivariate.reduced_or,
              'sum': bivariate.sum_,
              'f1': bivariate.f1,
              'jeff': bivariate.jeff_2,
              'wb 1': bivariate.wb_1,
              'wb 2': bivariate.wb_2,
              'wb 3': bivariate.wb_3,
              'imp. rdn': bivariate.imperfect_rdn,
              'rdn xor': bivariate.rdn_xor,
              'prob 1': bivariate.prob_1,
              'prob 2': bivariate.prob_2,
              'gband': bivariate.gband,
              'boom': bivariate.boom,
              'not two': bivariate.not_two,
              'pnt. unq': bivariate.pwu,
              }

trivariates = {'uniform': trivariate.uni,
               'null': trivariate.null,
               'redundant': trivariate.rdn,
               'synergy': trivariate.syn,
               'cat': trivariate.cat,
               'sum': trivariate.sum_,
               'jeff': trivariate.jeff_3,
               'xor cat': trivariate.xor_cat,
               'anddup': trivariate.anddup,
               'shared xor': trivariate.shared_xor,
               'xor shared': trivariate.xor_shared,
               'giant xor': trivariate.giant_xor,
               'dbl xor': trivariate.dblxor,
               }
