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
summands are exactly the tension coordinates. For a pair it is computed as
one weighted-rate query on the Gray-Wyner region.

Extremes
--------
``E = 0`` exactly when the mutual information is Gacs-Korner extractable,
i.e. when ``K[X:Y] = I[X:Y]``, in which case the pair splits as
``(X, Y) = (X', Y') + (W, W)`` with ``X'`` independent of ``Y'``. At the
other extreme ``E`` attains ``min{H[X|Y], H[Y|X], I[X:Y]}``, and the pair is
*maximally non-extractable*.

Several variables
-----------------
For ``n`` sources the leftover dependence is measured by the *dual* total
correlation and the source tensions are ``tau_i = I[X_{-i} : W | X_i]``:

.. math::

    E(X_{0:n}) = \\inf_W \\Big\\{ B[X_{0:n}|W]
        + \\sum_i I[X_{-i}{:}W|X_i] \\Big\\},

which reduces to the pairwise definition at ``n = 2``. The dual total
correlation keeps the measure on the scale of the common informations:
conditioning on the Gacs-Korner meet lowers ``B`` by exactly ``K``, whereas
it lowers the total correlation by ``(n - 1) K``. The residual
``B[X_{0:n}|W]`` is not affine in the Gray-Wyner rates, so for ``n >= 3`` the
probe is optimized directly.

The tension common information
------------------------------
Since ``E`` is a deficiency, the derived quantity

.. math::

    \\Theta(X_{0:n}) = B[X_{0:n}] - E(X_{0:n})

runs the other way: it is a continuous relaxation of the Gacs-Korner common
information, satisfying

.. math::

    K[X_{0:n}] \\leq \\Theta(X_{0:n}) \\leq B[X_{0:n}].

The lower bound follows by using the Gacs-Korner meet as the probe: it makes
every source tension vanish and leaves a residual of ``B - K``. Unlike
``K``, which is determined by the connected components of the support graph
and therefore collapses to zero under an arbitrarily small perturbation,
``Theta`` varies continuously.
"""

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...utils import flatten, unitful
from ..dual_total_correlation import dual_total_correlation
from .gk_common_information import gk_common_information

__all__ = (
    "entanglement",
    "tension_common_information",
)


def _sources(dist, rvs):
    """
    Normalize the source groups.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list of lists, None
        The source groups.

    Returns
    -------
    sources : list of lists
        The normalized source groups.
    """
    from ...exceptions import ditException

    sources = [[i] for i in flatten(dist.rvs)] if rvs is None else rvs
    if len(sources) < 2:
        msg = "The entanglement requires at least two sources."
        raise ditException(msg)

    return sources


class EntanglementOptimizer(BaseAuxVarOptimizer):
    """
    Minimize the total tension ``B[X_{0:n}|W] + sum_i I[X_{-i} : W | X_i]``
    over probes ``W``.
    """

    name = "entanglement"
    description = "The least total tension over probes W."

    def __init__(self, dist, rvs=None, crvs=None, bound=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rvs : list of lists
            The source groups.
        crvs : list, None
            Variables to condition on.
        bound : int, None
            Cap on the cardinality of the probe ``W``.
        """
        super().__init__(dist, rvs=rvs, crvs=crvs)

        theoretical_bound = self.compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([(self._rvs | self._crvs, bound)])

    def compute_bound(self):
        """
        The support-lemma bound on the probe: preserve the joint of the
        sources and the ``n + 1`` tension coordinates.

        Returns
        -------
        bound : int
            The bound on the size of W.
        """
        return int(np.prod(self._shape)) + len(self._rvs)

    def _objective(self):
        """
        The total tension.

        Returns
        -------
        obj : func
            The objective function.
        """
        residual = self._dual_total_correlation(self._rvs, self._crvs | self._arvs)
        tensions = [
            self._conditional_mutual_information(self._rvs - {rv}, self._arvs, {rv} | self._crvs) for rv in self._rvs
        ]

        def objective(self, x):
            """
            Compute the total tension.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            pmf = self.construct_joint(x)
            return residual(pmf) + sum(tension(pmf) for tension in tensions)

        return objective


def _pairwise_entanglement(dist, sources, crvs, niter, maxiter, bound):
    """
    The pairwise entanglement as a weighted-rate query on the Gray-Wyner
    region, where the three tension coordinates sum to ``3 R_0 + 2 R_1 + 2 R_2``
    minus a constant.
    """
    from ...rate_distortion import GrayWynerNetwork

    network = GrayWynerNetwork(dist, rvs=sources, crvs=crvs, bound=bound)
    tension = network.tension_point([3.0, 2.0, 2.0], niter=niter, maxiter=maxiter)
    return sum(tension.tensions) + tension.residual


@unitful
def entanglement(dist, rvs=None, crvs=None, niter=None, maxiter=1000, bound=None):
    """
    Compute the entanglement.

    The entanglement is the least total tension attainable by any probe ``W``,

    .. math::

        E(X_{0:n}) = \\inf_W \\Big\\{ B[X_{0:n}|W]
            + \\sum_i I[X_{-i}{:}W|X_i] \\Big\\},

    which for a pair is
    ``inf_W I[X:Y|W] + I[X:W|Y] + I[W:Y|X]``
    :cite:`makarychev2002class,zhang2003new,matveev2026beyond`. The
    multivariate residual is the dual total correlation, which puts ``E`` on
    the scale of the common informations.

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
    sources = _sources(dist, rvs)

    if len(sources) == 2:
        value = _pairwise_entanglement(dist, sources, crvs, niter, maxiter, bound)
    else:
        opt = EntanglementOptimizer(dist, rvs=sources, crvs=crvs, bound=bound)
        opt.optimize(niter=niter, maxiter=maxiter)
        value = float(opt.objective(opt._optima))

    # Two provable, cheap envelopes. The trivial probe W = . gives a total
    # tension of B, and the Gacs-Korner meet gives B - K; the meet is not
    # among the optimizer's seeds, so imposing it here is what makes the
    # Theta sandwich exact.
    dtc = float(dual_total_correlation(dist, sources, crvs))
    gk = float(gk_common_information(dist, sources, crvs))

    return min(max(value, 0.0), dtc - gk, dtc)


@unitful
def tension_common_information(dist, rvs=None, crvs=None, niter=None, maxiter=1000, bound=None):
    """
    Compute the tension common information.

    This is the portion of the dual total correlation that is not obstructed
    by tension,

    .. math::

        \\Theta(X_{0:n}) = B[X_{0:n}] - E(X_{0:n}),

    a continuous relaxation of the Gacs-Korner common information obeying
    ``K <= Theta <= B`` :cite:`matveev2026beyond`. For a pair ``B`` is the
    mutual information.

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
    sources = _sources(dist, rvs)

    dtc = float(dual_total_correlation(dist, sources, crvs))
    e = entanglement(dist, rvs=sources, crvs=crvs, niter=niter, maxiter=maxiter, bound=bound)

    return max(dtc - float(e), 0.0)
