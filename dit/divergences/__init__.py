"""
Divergences, measuring the distance between distributions. They are not
necessarily true metrics, but some are.
"""

from .jensen_shannon_divergence import (
    jensen_shannon_divergence,
)

from .cross_entropy import (
    cross_entropy,
)

from .kullback_leibler_divergence import (
    kullback_leibler_divergence,
    relative_entropy,
)

from .generalized_divergences import (
    alpha_divergence,
    hellinger_divergence,
    renyi_divergence,
    tsallis_divergence,
    f_divergence,
    hellinger_sum,
)

from .variational_distance import (
    bhattacharyya_coefficient,
    hellinger_distance,
    variational_distance,
)

from . import pmf
