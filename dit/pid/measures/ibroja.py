"""
The I_BROJA unique measure, as proposed by the BROJA team.
"""

from itertools import chain

from ...algorithms.broja_method import broja_solve_bivariate
from ..pid import BaseUniquePID

__all__ = ("PID_BROJA",)


class PID_BROJA(BaseUniquePID):
    """
    The BROJA partial information decomposition.

    Notes
    -----
    This partial information decomposition, at least in the bivariate source
    case, was independently suggested by Griffith.

    For two sources, ``method`` selects the bivariate solver:

    - ``'scipy'`` — SLSQP on the marginal-matching polytope, which can stop short of the minimum
    - ``'admui'`` — alternating divergence minimization (:cite:`banerjee2017computing`)
    - ``'cone'`` — exponential cone program via ECOS (:cite:`makkeh2018broja`)
    - ``'auto'`` — ``'cone'`` if ECOS is installed and ``'admui'`` otherwise, with fallbacks

    With more than two sources, the unique information of each source is its unique
    information in the bivariate decomposition of it and the union of the other sources,
    computed by the same solver.
    """

    _name = "I_broja"

    @staticmethod
    def _measure(d, sources, target, maxiter=1000, method="auto", rng=None, **ecos_kwargs):
        """
        This computes unique information as min{I(source : target | other_sources)}
        over the space of distributions which matches source-target marginals.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_broja for.
        sources : iterable of iterables
            The target variables.
        target : iterable
            The target variable.
        maxiter : int
            Maximum iterations for scipy/admui solvers.
        method : str
            Bivariate solver: ``'scipy'``, ``'admui'``, ``'cone'``, or ``'auto'``.
        rng : np.random.Generator, optional
            RNG for scipy restarts.
        **ecos_kwargs
            Passed to ECOS when ``method='cone'``.

        Returns
        -------
        ibroja : dict
            The value of I_broja for each individual source.
        """
        options = {"maxiter": maxiter, "method": method, "rng": rng, **ecos_kwargs}
        if len(sources) == 2:
            uniques, _meta = broja_solve_bivariate(d, sources, target, **options)
            return uniques

        uniques = {}
        for source in sources:
            rest = tuple(dict.fromkeys(chain(*(other for other in sources if other != source))))
            pair, _meta = broja_solve_bivariate(d, (source, rest), target, **options)
            uniques[source] = pair[source]
        return uniques
