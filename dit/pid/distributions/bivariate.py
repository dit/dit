"""
Example bivariate distributions. The last index is designed to be the output.
"""

from ... import Distribution
from ...distconst import erasure, uniform
from ...example_dists import jeff


__all__ = (
    'and_',
    'boom',
    'cat',
    'diff',
    'dup',
    'erase',
    'f1',
    'gband',
    'imperfect_rdn',
    'jeff_2',
    'not_two',
    'null',
    'prob_1',
    'prob_2',
    'pwu',
    'rand',
    'rdn',
    'rdn_xor',
    'reduced_or',
    'simple',
    'sum_',
    'syn',
    'uni',
    'unq1',
    'wb_1',
    'wb_2',
    'wb_3',
)


# three unrelated bit
uni = uniform(['000', '001', '010', '011', '100', '101', '110', '111'])

# correlated inputs independent of output
null = uniform(['000', '001', '110', '111'])

# inputs are fully redundant with output
rdn = uniform(['000', '111'])

# inputs are completely redundant, but don't always determine output
# problematic for i_dep_b
simple = Distribution(['000', '110', '111'], [1 / 2, 1 / 4, 1 / 4])

# unique information from X1
unq1 = uniform(['000', '011', '100', '111'])

# output is the concatenation of the inputs
cat = uniform(['000', '011', '102', '113'])

# output is the exclusive or of the inputs
syn = uniform(['000', '011', '101', '110'])

# output is the logical and of the inputs
and_ = uniform(['000', '010', '100', '111'])

# f1 from 'extractable information'
f1 = uniform(['002', '010', '102', '111'])

# jeff's generalization of rdn
jeff_2 = jeff(2)

# reduced or, from lizier by way of ince.
reduced_or = Distribution(['000', '011', '101'], [1 / 2, 1 / 4, 1 / 4])

# output is sum of inputs
sum_ = uniform(['000', '011', '101', '112'])

# output is a randomly chosen input
rand = Distribution(['000', '010', '011', '100', '101', '111'], [1 / 4, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4])

# example 1 from williams & beer
wb_1 = uniform(['000', '011', '102'])

# example 2 from williams & beer
wb_2 = uniform(['000', '011', '111', '102'])

# from ince, inspired by wb_2
wb_3 = uniform(['000', '110', '011', '111', '012', '102'])

# from griffith
imperfect_rdn = Distribution(['000', '010', '111'], [0.499, 0.001, 0.5])

# from griffith
rdn_xor = uniform(['000', '011', '101', '110', '222', '233', '323', '332'])

# distribution problematic for i_dep_b
prob_1 = uniform(['000', '001', '010', '011', '100'])

# counterexample for i_downarrow
prob_2 = uniform(['000', '010', '021', '101'])

# differentiates proj and broja
diff = uniform(['000', '001', '010', '101'])

# gband, some measures demonstrate subadditivity of redundancy with this distribution
gband = uniform(['000', '010', '100', '111', '222', '232', '322', '333'])

# min == proj, broja == ccs, dep stands alone
boom = uniform(['001', '002', '020', '121', '202', '212'])

# contains holistic synergy
not_two = uniform(['000', '001', '010', '100', '111'])

# the prototype for redundancy in and and sum
dup = uniform(['000', '001', '111', '112'])

# pointwise unique
pwu = uniform(['011', '101', '022', '202'])

# erase distribution. two-way secret key agreement rate is non-zero, even though
# both one-way secret key agreement rates are zero.
erase = erasure(rdn)
