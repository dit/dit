"""
A selection of generic distributions which could be useful.
"""

from .circuits import And, ImperfectRdn, Or, Rdn, RdnXor, Subtle, Unq, Xor
from .dice import iid_sum, summed_dice
from .giant_bit import giant_bit, jeff
from .mdbsi import dyadic, triadic
from .n_mod_m import n_mod_m
from .nonsignalling_boxes import pr_box
from .numeric import bernoulli, binomial, hypergeometric, uniform
