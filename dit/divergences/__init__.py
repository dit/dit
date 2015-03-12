"""
Divergences, measuring the distance between distributions. They are not
necessarily true metrics, but some are.
"""

from .jensen_shannon_divergence import (
	jensen_shannon_divergence,
	jensen_shannon_divergence_pmf,
)

from .cross_entropy import (
    cross_entropy,
)

from .kullback_leibler_divergence import (
    kullback_leibler_divergence,
	relative_entropy,
)

from .kl import (
#	relative_entropy,
#	cross_entropy,
#	DKL,
	cross_entropy_pmf,
	relative_entropy_pmf,
	DKL_pmf
)
