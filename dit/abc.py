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

__all__ = ['D', 'SD', 'H', 'I', 'K', 'T', 'P', 'B', 'R', 'JSD']

