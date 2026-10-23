"""
A selection of generic distributions which could be useful.
"""

from .circuits import And, ImperfectRdn, Or, Rdn, RdnXor, Subtle, Unq, Xor
from .common_information import (
    erasure_channel,
    noisy_copies,
    reverse_erasure_channel,
    symmetric_common_information_witness,
    task_assignment,
)
from .dice import iid_sum, summed_dice
from .empirical import bach, blood_types, car, congress, corelli, penguins, student, titanic
from .genetic_code import genetic_code
from .giant_bit import giant_bit, jeff
from .mdbsi import dyadic, triadic
from .n_mod_m import n_mod_m
from .nonsignalling_boxes import pr_box
from .numeric import bernoulli, binomial, hypergeometric, multinomial, uniform
