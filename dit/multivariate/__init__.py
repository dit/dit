"""
Multivariate measures of information. Some are direct extensions of Shannon's
measures and others are more distantly related.
"""

from .caekl_mutual_information import caekl_mutual_information
from .coinformation import coinformation
from .dual_total_correlation import (binding_information,
                                     dual_total_correlation,
                                     independent_information,
                                     residual_entropy,
                                     variation_of_information)
from .entropy import entropy
from .exact_common_information import exact_common_information
from .functional_common_information import functional_common_information
from .gk_common_information import gk_common_information
from .interaction_information import interaction_information
from .intrinsic_mutual_information import (intrinsic_total_correlation,
                                           intrinsic_dual_total_correlation,
                                           intrinsic_caekl_mutual_information)
from .lautum_information import lautum_information
from .mss_common_information import mss_common_information
from .total_correlation import total_correlation
from .tse_complexity import tse_complexity
from .wyner_common_information import wyner_common_information
