"""
Various algorithms related to manipulating or measuring properties of
distributions.
"""

from .channelcapacity import channel_capacity
from .information_partitions import ShannonPartition
from .lattice import insert_join, insert_meet
from .maxentropy import *
from .minimal_sufficient_statistic import insert_mss
from .prune_expand import pruned_samplespace, expanded_samplespace
from .stats import mean, median, mode, standard_deviation, central_moment, \
                   standard_moment
