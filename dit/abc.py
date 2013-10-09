"""
Import several functions as shorthand.
"""

from dit import (Distribution as D,
                 ScalarDistribution as SD,
                )

from dit.algorithms import (coinformation as I,
                            common_information as K,
                            total_correlation as T,
                            perplexity as P,
                            jensen_shannon_divergence as JSD,
                           )

from dit.algorithms.entropy2 import entropy2 as H