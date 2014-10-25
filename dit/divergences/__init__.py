"""
Divergences, measuring the distance between distributions. They are not
necessarily true metrics, but some are.
"""

from .jensen_shannon_divergence import (
	jensen_shannon_divergence,
	jensen_shannon_divergence_pmf,
)
from .kl import (
	relative_entropy,
	cross_entropy,
	DKL,
	cross_entropy_pmf,
	relative_entropy_pmf,
	DKL_pmf
)
