"""
The generalized (lossy, n-source) Gray-Wyner network.

Exposes the achievable rate region (`GrayWynerNetwork`), the common-rate vs
private-rate trade-off curve (`GrayWynerCurve`) and its plotter
(`GrayWynerPlotter`), the underlying optimizer (`GrayWynerOptimizer`), and the
lossy Wyner common information (`lossy_wyner_common_information`).
"""

from .curve import GrayWynerCurve
from .network import GrayWynerNetwork, lossy_wyner_common_information
from .optimizer import GrayWynerOptimizer, GrayWynerPoint, hamming_matrix

__all__ = (
    "GrayWynerCurve",
    "GrayWynerNetwork",
    "GrayWynerOptimizer",
    "GrayWynerPoint",
    "hamming_matrix",
    "lossy_wyner_common_information",
)


def __getattr__(name):
    """Lazily expose the (matplotlib-dependent) plotter."""
    if name == "GrayWynerPlotter":
        from .plotting import GrayWynerPlotter

        return GrayWynerPlotter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
