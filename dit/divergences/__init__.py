"""
Divergences, measuring the distance between distributions. They are not
necessarily true metrics, but some are.
"""

from . import pmf
from .copy_mutual_information import (
    copy_mutual_information,
)

# from .coupling_metrics import (
#     coupling_metric,
# )
from .cross_entropy import (
    cross_entropy,
)
from .earth_movers_distance import (
    earth_movers_distance,
)
from .generalized_divergences import (
    alpha_divergence,
    f_divergence,
    hellinger_divergence,
    hellinger_sum,
    renyi_divergence,
    tsallis_divergence,
)
from .hypercontractivity_coefficient import (
    hypercontractivity_coefficient,
)
from .jensen_shannon_divergence import (
    jensen_shannon_divergence,
)
from .kullback_leibler_divergence import (
    kullback_leibler_divergence,
    relative_entropy,
)
from .maximum_correlation import (
    maximum_correlation,
)
from .variational_distance import (
    bhattacharyya_coefficient,
    chernoff_information,
    hellinger_distance,
    variational_distance,
)
