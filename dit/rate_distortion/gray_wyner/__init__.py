"""
The generalized (lossy, n-source) Gray-Wyner network.

Exposes the achievable rate region (`GrayWynerNetwork`), the common-rate vs
private-rate trade-off curve (`GrayWynerCurve`) and its plotter
(`GrayWynerPlotter`), the underlying optimizer (`GrayWynerOptimizer`), and the
lossy Wyner common information (`lossy_wyner_common_information`).

The same region viewed in other coordinates gives the region of tension, the
extension profile, and the mutual information region; `region` holds those
affine maps and `shape` the associated shape function.
"""

from .curve import GrayWynerCurve
from .network import GrayWynerNetwork, lossy_wyner_common_information
from .optimizer import GrayWynerOptimizer, GrayWynerPoint, hamming_matrix
from .region import (
    ExtensionPoint,
    MutualInformationPoint,
    TensionPoint,
    asymmetric_private_interaction_information,
    extension_to_rates,
    maximal_interaction_information,
    mutual_information_to_rates,
    rates_to_extension,
    rates_to_mutual_information,
    rates_to_tension,
    symmetric_private_interaction_information,
    tension_to_rates,
)
from .shape import ShapeFunction

__all__ = (
    "ExtensionPoint",
    "GrayWynerCurve",
    "GrayWynerNetwork",
    "GrayWynerOptimizer",
    "GrayWynerPoint",
    "MutualInformationPoint",
    "ShapeFunction",
    "TensionPoint",
    "asymmetric_private_interaction_information",
    "extension_to_rates",
    "hamming_matrix",
    "lossy_wyner_common_information",
    "maximal_interaction_information",
    "mutual_information_to_rates",
    "rates_to_extension",
    "rates_to_mutual_information",
    "rates_to_tension",
    "symmetric_private_interaction_information",
    "tension_to_rates",
)


def __getattr__(name):
    """Lazily expose the (matplotlib-dependent) plotter."""
    if name == "GrayWynerPlotter":
        from .plotting import GrayWynerPlotter

        return GrayWynerPlotter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
