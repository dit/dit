"""
Import several functions as shorthand.
"""

from dit import (Distribution as D,
                 ScalarDistribution as SD,
                )

from dit.algorithms import channel_capacity_joint as CC

from dit.divergences import (cross_entropy as xH,
                             kullback_leibler_divergence as DKL,
                             jensen_shannon_divergence as JSD,
                            )

from dit.other import (extropy as X,
                       perplexity as P,
                       cumulative_residual_entropy as CRE,
                       generalized_cumulative_residual_entropy as GCRE,
                      )

from dit.multivariate import (caekl_mutual_information as J,
                              coinformation as I,
                              dual_total_correlation as B,
                              entropy as H,
                              exact_common_information as G,
                              functional_common_information as F,
                              gk_common_information as K,
                              interaction_information as II,
                              mss_common_information as M,
                              residual_entropy as R,
                              total_correlation as T,
                              tse_complexity as TSE,
                              wyner_common_information as C,
                             )

# distribution types
_dists = [
    'D',    # a joint distribution
    'SD',   # a scalar distribution
]

# measures directly computed from i-diagrams
_entropies = [
    'H',    # the joint conditional entropy
    'R',    # the conditional residual entropy [erasure entropy]
]

# mutual informations
_mutual_informations = [
    'I',    # the multivariate conditional mututal information
    'T',    # the conditional total correlation [multi-information/integration]
    'B',    # the conditional dual total correlation [binding information]
    'J',    # the CAEKL common information
    'II',   # the interaction information
]

# common informations
_common_informations = [
    'K',    # the Gacs-Korner common information [meet entropy]
    'C',    # the wyner common information
    'G',    # the exact common information
    'F',    # the functional common information
    'M',    # the joint minimal sufficient statistic entropy
]

# measures of distance between distriutions
_divergences = [
    'xH',   # the cross entropy
    'DKL',  # the Kullback-Leibler divergence
    'JSD',  # the Jensen-Shannon divergence
]

# other measures
_others = [
    'CC',   # the channel capacity
    'P',    # the perplexity
    'X',    # the extropy
    'TSE',  # the TSE complexity
    'CRE',  # the cumulative residual entropy
    'GCRE', # the generalized cumulative residual entropy
]

__all__ = _dists + _entropies + _mutual_informations + _common_informations + _divergences + _others
