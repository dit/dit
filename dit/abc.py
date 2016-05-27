"""
Import several functions as shorthand.
"""

from dit import (Distribution as D,
                 ScalarDistribution as SD,
                )

from dit.divergences import (cross_entropy as xH,
                             kullback_leibler_divergence as DKL,
                             jensen_shannon_divergence as JSD,
                            )

from dit.other import (extropy as X,
                       perplexity as P,
                       cumulative_residual_entropy as CRE,
                       generalized_cumulative_residual_entropy as GCRE,
                      )

from dit.multivariate import (binding_information as B,
                              caekl_mutual_information as J,
                              coinformation as I,
                              entropy as H,
                              functional_common_information as F,
                              gk_common_information as K,
                              interaction_information as II,
                              mss_common_information as M,
                              residual_entropy as R,
                              total_correlation as T,
                              tse_complexity as TSE,
                             )

# distribution types
_dists = [
    'D',    # a joint distribution
    'SD',   # a scalar distribution
]

# measures directly computed from i-diagrams
_shannon = [
    'H',    # the joint conditional entropy
    'I',    # the multivariate conditional mututal information
    'T',    # the conditional total correlation [multi-information/integration]
    'B',    # the conditional dual total correlation [binding information]
    'R',    # the conditional residual entropy [erasure entropy]
    'TSE',  # the TSE complexity
]

# measures representable on i-diagrams
_shannon_ext = [
    'J',    # the CAEKL common information
    'K',    # the Gacs-Korner common information [meet entropy]
    'II',   # the interaction information
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
    'P',    # the perplexity
    'X',    # the extropy
    'CRE',  # the cumulative residual entropy
    'GCRE', # the generalized cumulative residual entropy
]

__all__ = _dists + _shannon + _shannon_ext + _divergences + _others
