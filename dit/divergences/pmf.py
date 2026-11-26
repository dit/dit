"""
Provide a common place to access pmf-based divergences.
"""

from ._kl_nonmerge import (
    cross_entropy_pmf as cross_entropy,
)
from ._kl_nonmerge import (
    relative_entropy_pmf as relative_entropy,
)
from .channel_simulation_divergence import (
    channel_simulation_divergence_pmf as channel_simulation_divergence,
)
from .earth_movers_distance import (
    earth_movers_distance_pmf as earth_movers_distance,
)
from .jensen_shannon_divergence import (
    jensen_shannon_divergence_pmf as jensen_shannon_divergence,
)
from .maximum_correlation import (
    conditional_maximum_correlation_pmf as conditional_maximum_correlation,
)
from .maximum_correlation import (
    maximum_correlation_pmf as maximum_correlation,
)
from .variational_distance import (
    bhattacharyya_angle_pmf as bhattacharyya_angle,
)
from .variational_distance import (
    bhattacharyya_coefficient_pmf as bhattacharyya_coefficient,
)
from .variational_distance import (
    bhattacharyya_distance_pmf as bhattacharyya_distance,
)
from .variational_distance import (
    chernoff_information_pmf as chernoff_information,
)
from .variational_distance import (
    fisher_rao_distance_pmf as fisher_rao_distance,
)
from .variational_distance import (
    hellinger_distance_pmf as hellinger_distance,
)
from .variational_distance import (
    matusita_distance_pmf as matusita_distance,
)
from .variational_distance import (
    variational_distance_pmf as variational_distance,
)

__all__ = (
    "bhattacharyya_angle",
    "bhattacharyya_coefficient",
    "bhattacharyya_distance",
    "channel_simulation_divergence",
    "chernoff_information",
    "conditional_maximum_correlation",
    "cross_entropy",
    "earth_movers_distance",
    "fisher_rao_distance",
    "hellinger_distance",
    "matusita_distance",
    "jensen_shannon_divergence",
    "jensen_shannon_divergence2",
    "maximum_correlation",
    "relative_entropy",
    "variational_distance",
)


def jensen_shannon_divergence2(p, q):
    """
    Compute the Jensen-Shannon divergence between two pmfs.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    jsd : float
        The Jensen-Shannon divergence.
    """
    return jensen_shannon_divergence([p, q])
