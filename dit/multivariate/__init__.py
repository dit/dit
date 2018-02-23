"""
Multivariate measures of information. Some are direct extensions of Shannon's
measures and others are more distantly related.
"""

from .caekl_mutual_information import caekl_mutual_information
from .coinformation import coinformation
from .common_informations import *
from .deweese import *
from .dual_total_correlation import (binding_information,
                                     dual_total_correlation,
                                     independent_information,
                                     residual_entropy,
                                     variation_of_information)
from .entropy import entropy
from .interaction_information import interaction_information
from .secret_key_agreement import *
from .total_correlation import total_correlation
from .tse_complexity import tse_complexity
