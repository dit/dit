"""
The entanglement and the tension common information.

The *entanglement* of a pair of random variables,

.. math::

    E(X, Y) = \\inf_W \\Big\\{ I[X{:}Y|W] + I[X{:}W|Y] + I[W{:}Y|X] \\Big\\},

measures how far the mutual information of the pair is from being
*extractable*. It is the quantity appearing on the right of the
Makarychev-Makarychev-Romashchenko-Vereshchagin non-Shannon inequality
:cite:`makarychev2002class`, and was studied by Zhang :cite:`zhang2003new`
(who wrote it ``W(X, Y)``) in connection with the approximate
representability of mutual information. The name is due to Matveev &
Romashchenko :cite:`matveev2026beyond`; it has nothing to do with quantum
entanglement.

Geometrically it is the smallest coordinate-sum over the region of tension of
Prabhakaran & Prabhakaran :cite:`prabhakaran2014assisted`, since the three
summands are exactly the tension coordinates. This is how it is computed
here: one weighted-rate query on the Gray-Wyner region.

Extremes
--------
``E = 0`` exactly when the mutual information is Gacs-Korner extractable,
i.e. when ``K[X:Y] = I[X:Y]``, in which case the pair splits as
``(X, Y) = (X', Y') + (W, W)`` with ``X'`` independent of ``Y'``. At the
other extreme ``E`` attains ``min{H[X|Y], H[Y|X], I[X:Y]}``, and the pair is
*maximally non-extractable*.

The tension common information
------------------------------
Since ``E`` is a deficiency, the derived quantity

.. math::

    \\Theta(X_{0:n}) = T[X_{0:n}] - E(X_{0:n})

runs the other way: it is a continuous relaxation of the Gacs-Korner common
information, satisfying

.. math::

    (n - 1) K[X_{0:n}] \\leq \\Theta(X_{0:n}) \\leq T[X_{0:n}].

The lower bound follows by using the Gacs-Korner meet as the probe: it makes
every source tension vanish and leaves a residual of
``T - (n-1) K``. Unlike ``K``, which is determined by the connected
components of the support graph and therefore collapses to zero under an
arbitrarily small perturbation, ``Theta`` varies continuously.

Several variables
-----------------
For ``n`` sources the tension coordinates are
``tau_i = I[X_{-i} : W | X_i]`` and ``tau_res = T[X_{0:n} | W]``, so

.. math::

    E(X_{0:n}) = \\inf_W \\Big\\{ T[X_{0:n}|W]
        + \\sum_i I[X_{-i}{:}W|X_i] \\Big\\},

which reduces to the pairwise definition at ``n = 2``. Replacing the residual
total correlation with the dual total correlation gives a different
generalization that agrees at ``n = 2``; only the total-correlation form is
affine in the Gray-Wyner rates, and it is the one implemented here.
"""

from ...utils import flatten, unitful
from ..total_correlation import total_correlation
from .gk_common_information import gk_common_information

__all__ = (
    "entanglement",
    "tension_common_information",
)


def _network(dist, rvs, crvs, bound):
    """
    Build the Gray-Wyner network backing these measures.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list of lists, None
        The source groups.
    crvs : list, None
        Variables to condition on.
    bound : int, None
        Optional cap on the cardinality of the probe.

    Returns
    -------
    network : GrayWynerNetwork
        The network.
    sources : list of lists
        The normalized source groups.
    """
    from ...exceptions import ditException
    from ...rate_distortion import GrayWynerNetwork

    sources = [[i] for i in flatten(dist.rvs)] if rvs is None else rvs
    if len(sources) < 2:
        msg = "The entanglement requires at least two sources."
        raise ditException(msg)

    return GrayWynerNetwork(dist, rvs=sources, crvs=crvs, bound=bound), sources


@unitful
def entanglement(dist, rvs=None, crvs=None, niter=None, maxiter=1000, bound=None):
    """
    Compute the entanglement.

    The entanglement is the least total tension attainable by any probe ``W``,

    .. math::

        E(X_{0:n}) = \\inf_W \\Big\\{ T[X_{0:n}|W]
            + \\sum_i I[X_{-i}{:}W|X_i] \\Big\\},

    which for a pair is
    ``inf_W I[X:Y|W] + I[X:W|Y] + I[W:Y|X]``
    :cite:`makarychev2002class,zhang2003new,matveev2026beyond`.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the entanglement is computed.
    rvs : list of lists, None
        The source groups. If None, each variable of `dist` is its own
        source.
    crvs : list, None
        A single list of indexes specifying variables to condition on. If
        None, none are conditioned on.
    niter : int, None
        Number of basin hops to perform during the optimization.
    maxiter : int
        The number of iterations of the optimization subroutine to perform.
    bound : int, None
        Bound the size of the probe variable.

    Returns
    -------
    E : float
        The entanglement.

    Examples
    --------
    A pair whose mutual information is fully extractable has zero
    entanglement:

    >>> from dit import Distribution
    >>> from dit.multivariate import entanglement
    >>> giant_bit = Distribution(['00', '11'], [1/2, 1/2])
    >>> float(entanglement(giant_bit))
    0.0
    """
    network, sources = _network(dist, rvs, crvs, bound)
    n = len(sources)

    # The tension coordinates sum to (n + 1) R_0 + 2 sum_i R_i minus a
    # constant, so minimizing that weighted rate minimizes the total tension.
    lambdas = [float(n + 1)] + [2.0] * n
    tension = network.tension_point(lambdas, niter=niter, maxiter=maxiter)
    value = sum(tension.tensions) + tension.residual

    # Two provable, cheap envelopes. The trivial probe W = . gives a total
    # tension of T, and the Gacs-Korner meet gives T - (n-1) K; the meet is
    # not among the optimizer's deterministic seeds, so imposing it here is
    # what makes the Theta sandwich exact.
    tc = float(total_correlation(dist, sources, crvs))
    gk = float(gk_common_information(dist, sources, crvs))

    return min(max(value, 0.0), tc - (n - 1) * gk, tc)


@unitful
def tension_common_information(dist, rvs=None, crvs=None, niter=None, maxiter=1000, bound=None):
    """
    Compute the tension common information.

    This is the portion of the total correlation that is not obstructed by
    tension,

    .. math::

        \\Theta(X_{0:n}) = T[X_{0:n}] - E(X_{0:n}),

    a continuous relaxation of the Gacs-Korner common information obeying
    ``(n - 1) K <= Theta <= T`` :cite:`matveev2026beyond`.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the tension common information is
        computed.
    rvs : list of lists, None
        The source groups. If None, each variable of `dist` is its own
        source.
    crvs : list, None
        A single list of indexes specifying variables to condition on. If
        None, none are conditioned on.
    niter : int, None
        Number of basin hops to perform during the optimization.
    maxiter : int
        The number of iterations of the optimization subroutine to perform.
    bound : int, None
        Bound the size of the probe variable.

    Returns
    -------
    Theta : float
        The tension common information.

    Examples
    --------
    Unlike the Gacs-Korner common information, this does not collapse when
    the block structure of the support is perturbed:

    >>> from dit import Distribution
    >>> from dit.multivariate import gk_common_information, tension_common_information
    >>> blocks = Distribution(['00', '01', '10', '11', '22'], [0.2] * 5)
    >>> float(gk_common_information(blocks))
    0.7219280948873623
    """
    _, sources = _network(dist, rvs, crvs, bound)

    tc = float(total_correlation(dist, sources, crvs))
    e = entanglement(dist, rvs=sources, crvs=crvs, niter=niter, maxiter=maxiter, bound=bound)

    return max(tc - float(e), 0.0)
