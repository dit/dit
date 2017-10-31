"""
Module for basic inference tools.

"""
from .binning import binned
from .counts import get_counts, distribution_from_data
from .estimators import entropy_0, entropy_1, entropy_2
from .knn_estimators import total_correlation_ksg
