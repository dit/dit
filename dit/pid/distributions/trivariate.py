"""
Example trivariate distributions. The last index is designed to be the output.
"""

from __future__ import division

from itertools import product

from ...distconst import uniform
from ...example_dists import jeff


# independent bits
uni = uniform(''.join(_) for _ in product('01', repeat=4))

# correlated inputs, independent of output
null = uniform(['0000', '0001', '1110', '1111'])

# targets {0}{1}{2}
rdn = uniform(['0000', '1111'])

# targets {012}, all events have even parity
syn = uniform(['0000', '0011', '0101', '0110', '1001', '1010', '1100', '1111'])

# output is concatenated inputs
cat = uniform(['0000', '0011', '0102', '0113', '1004', '1015', '1106', '1117'])

# output is the sum of the inputs
sum_ = uniform(['0000', '0011', '0101', '0112', '1001', '1012', '1102', '1113'])

# problematic for everyone
xor_cat = uniform(['0000', '0111', '1012', '1103'])

# targets {01}{12}
shared_xor = uniform(['0000', '0110', '0211', '0301', '1011', '1101', '1200', '1310'])

# targets {0}{12}
xor_shared = uniform(['0000', '1011', '2101', '3110'])

# targets {01}{02}{12}
giant_xor = uniform(['0000', '0120', '0231', '0311', '1031', '1111', '1200', '1320',
                     '2010', '2130', '2221', '2301', '3021', '3101', '3210', '3330'])

xor_giant = uniform(['0000', '0211', '1021', '1230', '2101', '2310', '3120', '3331'])

# difficult to interpret
jeff_3 = jeff(3)

# anddup, from griffith
anddup = uniform(['0000', '0100', '1010', '1111'])

dblxor = uniform(['0000', '0011', '0103', '0112', '1002', '1013', '1101', '1110'])
