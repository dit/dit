"""
Provide a common place to access pmf-based divergences.

"""

from .earth_movers_distance import (
    earth_movers_distance_pmf as earth_movers_distance,
)

from .jensen_shannon_divergence import (
    jensen_shannon_divergence_pmf as jensen_shannon_divergence,
)

from ._kl_nonmerge import (
    cross_entropy_pmf as cross_entropy,
    relative_entropy_pmf as relative_entropy,
)

from .maximum_correlation import (
    maximum_correlation_pmf as maximum_correlation,
    conditional_maximum_correlation_pmf as conditional_maximum_correlation,
)

from .variational_distance import (
    bhattacharyya_coefficient_pmf as bhattacharyya_coefficient,
    chernoff_information_pmf as chernoff_information,
    hellinger_distance_pmf as hellinger_distance,
    variational_distance_pmf as variational_distance,
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