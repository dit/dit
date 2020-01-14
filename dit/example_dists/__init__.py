"""
A selection of generic distributions which could be useful.
"""

from .circuits import Unq, Rdn, Xor, And, Or, RdnXor, ImperfectRdn, Subtle
from .dice import iid_sum, summed_dice
from .giant_bit import giant_bit, jeff
from .mdbsi import dyadic, triadic
from .n_mod_m import n_mod_m
from .numeric import bernoulli, binomial, hypergeometric, uniform
from .nonsignalling_boxes import pr_box
