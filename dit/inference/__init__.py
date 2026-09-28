"""
Module for basic inference tools.
"""

from ._symbols import Trials, UndersamplingWarning
from .binning import binned
from .counts import distribution_from_data, get_counts
from .estimators import (
    ENTROPY_ESTIMATORS,
    block_entropy,
    conditional_entropy_rate,
    entropy_0,
    entropy_1,
    entropy_2,
    entropy_from_counts,
    lz_entropy_rate,
)
from .knn_estimators import conditional_mutual_information_test_knn, differential_entropy_knn, total_correlation_ksg
from .markov_order import MarkovOrderTest, markov_order_test, select_markov_order
from .posterior import EntropyRatePosterior, entropy_rate_posterior
from .significance import (
    SurrogateTest,
    benjamini_hochberg,
    bootstrap_ci,
    conditional_mutual_information,
    conditional_mutual_information_test,
    stationary_bootstrap,
    transfer_entropy,
    transfer_entropy_ci,
    transfer_entropy_test,
)
from .surrogates import block_surrogates, shift_surrogates, whittle_count, whittle_surrogates
from .time_series import dist_from_timeseries
