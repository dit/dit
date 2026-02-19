"""
Import several functions as shorthand.
"""

from dit import (
                 Distribution as D,
)
from dit import (
                 ScalarDistribution as SD,
)
from dit.algorithms import channel_capacity_joint as CC
from dit.divergences import (
                 cross_entropy as xH,
)
from dit.divergences import (
                 jensen_shannon_divergence as JSD,
)
from dit.divergences import (
                 kullback_leibler_divergence as DKL,
)
from dit.multivariate import (
                 caekl_mutual_information as J,
)
from dit.multivariate import (
                 coinformation as I,
)
from dit.multivariate import (
                 dual_total_correlation as B,
)
from dit.multivariate import (
                 entropy as H,
)
from dit.multivariate import (
                 exact_common_information as G,
)
from dit.multivariate import (
                 functional_common_information as F,
)
from dit.multivariate import (
                 gk_common_information as K,
)
from dit.multivariate import (
                 interaction_information as II,
)
from dit.multivariate import (
                 mss_common_information as M,
)
from dit.multivariate import (
                 residual_entropy as R,
)
from dit.multivariate import (
                 total_correlation as T,
)
from dit.multivariate import (
                 tse_complexity as TSE,
)
from dit.multivariate import (
                 wyner_common_information as C,
)
from dit.other import (
                 cumulative_residual_entropy as CRE,
)
from dit.other import (
                 extropy as X,
)
from dit.other import (
                 generalized_cumulative_residual_entropy as GCRE,
)
from dit.other import (
                 perplexity as P,
)

# distribution types
_dists = (
    'D',     # a joint distribution
    'SD',    # a scalar distribution
)

# measures directly computed from i-diagrams
_entropies = (
    'H',     # the joint conditional entropy
    'R',     # the conditional residual entropy [erasure entropy]
)

# mutual informations
_mutual_informations = (
    'I',     # the multivariate conditional mutual information
    'T',     # the conditional total correlation [multi-information/integration]
    'B',     # the conditional dual total correlation [binding information]
    'J',     # the CAEKL mutual information
    'II',    # the interaction information
)

# common informations
_common_informations = (
    'K',     # the Gacs-Korner common information [meet entropy]
    'C',     # the wyner common information
    'G',     # the exact common information
    'F',     # the functional common information
    'M',     # the joint minimal sufficient statistic entropy
)

# measures of distance between distriutions
_divergences = (
    'xH',    # the cross entropy
    'DKL',   # the Kullback-Leibler divergence
    'JSD',   # the Jensen-Shannon divergence
)

# other measures
_others = (
    'CC',    # the channel capacity
    'P',     # the perplexity
    'X',     # the extropy
    'TSE',   # the TSE complexity
    'CRE',   # the cumulative residual entropy
    'GCRE',  # the generalized cumulative residual entropy
)

__all__ = _dists + _entropies + _mutual_informations + _common_informations + _divergences + _others
