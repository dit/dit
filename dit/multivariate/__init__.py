"""
Multivariate measures of information. Some are direct extensions of Shannon's
measures and others are more distantly related.
"""

from .binding_information import binding_information, dual_total_correlation, residual_entropy, variation_of_information
from .caekl_mutual_information import caekl_mutual_information
from .coinformation import coinformation
from .entropy import entropy
from .functional_common_information import functional_common_information
from .gk_common_information import gk_common_information
from .interaction_information import interaction_information
from .joint_mss import joint_mss_entropy
from .total_correlation import total_correlation
from .tse_complexity import tse_complexity
