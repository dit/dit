"""
The delta-PID measure from Banerjee, Olbrich, Jost & Rauh (2018).

Defines unique information via weighted output KL deficiency:
    delta(M : X \\ Y) := inf_{P(X'|Y)} E_M[D_KL(P(X|M) || P(X'|Y) ∘ P(Y|M))]

Redundancy is then min-symmetrized:
    RI_delta(M : X; Y) = min{ I(M;X) - delta(M:X\\Y), I(M;Y) - delta(M:Y\\X) }

References
----------
.. [1] P. K. Banerjee, E. Olbrich, J. Jost, J. Rauh,
       "Unique informations and deficiencies", Allerton Conference, 2018.
.. [2] P. Venkatesh, K. Gurushankar, G. Schamberg,
       "Capturing and Interpreting Unique Information", arXiv:2302.11873, 2023.
"""

import numpy as np

from ...channelorder.deficiency import weighted_output_kl_deficiency
from ...channelorder._utils import channels_from_joint
from ...multivariate import coinformation
from ..pid import BaseBivariatePID

__all__ = ("PID_Delta",)


def _kl_deficiency(d, source, other, target, nstarts=15):
    """
    Compute the weighted output KL deficiency delta(M : source \\ other).

    This is the cost of approximating P(source|M) from P(other|M) via
    output randomization, weighted by P(M).

    Parameters
    ----------
    d : Distribution
        The joint distribution.
    source : iterable
        The source whose channel we want to approximate.
    other : iterable
        The other source used for approximation.
    target : iterable
        The target variable (M).
    nstarts : int
        Number of optimization restarts.

    Returns
    -------
    delta : float
        The deficiency (non-negative).
    """
    d = d.coalesce([list(source), list(other), list(target)])
    src, oth, tgt = d.dims
    kappa, mu, pi_s = channels_from_joint(d, [tgt], [src], [oth])
    # weighted_output_kl_deficiency returns nats; convert to bits
    return weighted_output_kl_deficiency(mu, kappa, pi_s) / np.log(2)


class PID_Delta(BaseBivariatePID):
    """
    The delta-PID based on weighted output KL deficiency.

    The deficiency delta(M : X \\ Y) measures how well Y can simulate X
    for making decisions about M, via the expected KL divergence between
    the true channel P(X|M) and the best approximation P(X'|Y) composed
    with P(Y|M).

    Redundancy is symmetrized via the minimum:
        RI = min{ I(M;X) - delta(M:X\\Y), I(M;Y) - delta(M:Y\\X) }

    References
    ----------
    .. [1] P. K. Banerjee et al., "Unique informations and deficiencies",
           Allerton Conference, 2018.
    """

    _name = "I_\u03b4"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute the delta-PID redundancy for a pair of sources.

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_delta for.
        sources : iterable of iterables
            The source variables (exactly two).
        target : iterable
            The target variable.

        Returns
        -------
        ri : float
            The delta-PID redundancy.
        """
        source_a, source_b = sources

        delta_ab = _kl_deficiency(d, source_a, source_b, target)
        delta_ba = _kl_deficiency(d, source_b, source_a, target)

        mi_a = coinformation(d, [source_a, target])
        mi_b = coinformation(d, [source_b, target])

        return min(mi_a - delta_ab, mi_b - delta_ba)
