"""
Various algorithms related to manipulating or measuring properties of
distributions.
"""

from .lattice import insert_join, insert_meet
from .stats import mean, median, mode, standard_deviation, central_moment, \
                   standard_moment
from .prune_expand import pruned_samplespace, expanded_samplespace
from .information_partitions import ShannonPartition
