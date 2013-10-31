"""
Import several functions as shorthand.
"""

from dit import (Distribution as D,
                 ScalarDistribution as SD,
                )

from dit.algorithms.entropy2 import entropy2 as H

from dit.algorithms import (coinformation as I,
                            common_information as K,
                            total_correlation as T,
                            perplexity as P,
                            binding_information as B,
                            residual_entropy as R,
                            jensen_shannon_divergence as JSD,
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
    'B',    # the conditional binding information [dual total correlation]
    'R',    # the conditional residual entropy [erasure entropy]
]

# measures representable on i-diagrams
_shannon_ext = [
    'K',    # the Gacs-Korner common information [meet entropy]
]

# measures of distance between distriutions
_divergences = [
    'JSD',  # the Jensen-Shannon divergence
]

# other measures
_others = [
    'P',    # the perplexity
]

__all__ = _dists + _shannon + _shannon_ext + _divergences + _others
