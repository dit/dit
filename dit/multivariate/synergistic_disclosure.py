"""
Scalar functions for synergistic disclosure.

Rosas, Mediano, Rassouli & Barrett (2020), "An operational information
decomposition via synergistic disclosure", J. Phys. A: Math. Theor. 53 485001.
https://doi.org/10.1088/1751-8121/abb723

Provides standalone functions for computing:
- S_alpha(X -> Y) for a single constraint set alpha
- The backbone decomposition {B^m_partial for m=1..n}
- Self-synergy S_alpha(X -> X)
"""

from ..utils import flatten, unitful

__all__ = (
    "backbone_disclosure",
    "modified_synergistic_disclosure",
    "self_synergy",
    "synergistic_disclosure",
)


@unitful
def synergistic_disclosure(dist, sources, target, alpha, niter=None, bound=None):
    """
    Compute S_alpha(X -> Y) for a single constraint set alpha.

    This is the maximum I(V; Y) over all alpha-synergistic channels,
    i.e. channels p_{V|X} satisfying I(V; X_{alpha_i}) = 0 for each
    subgroup alpha_i in alpha.

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources and target.
    sources : list of lists
        Each inner list gives the indices of one source variable group.
    target : list
        The indices of the target variable.
    alpha : list of lists
        Each inner list gives source-list indices (0-based) specifying
        which sources form each constraint subgroup.
    niter : int, None
        Number of basin-hopping restarts.
    bound : int, None
        Cardinality bound on V.

    Returns
    -------
    s_alpha : float
        The synergistic disclosure, in bits.
    """
    from ..pid.syndisc import SyndiscOptimizer

    alpha_tuples = tuple(tuple(a) for a in alpha)

    if not alpha_tuples:
        from .coinformation import coinformation

        return coinformation(dist, [list(flatten(sources)), list(target)])

    flat_sources = [list(s) for s in sources]
    try:
        opt = SyndiscOptimizer(
            dist,
            flat_sources,
            list(target),
            alpha_tuples,
            bound=bound,
        )
        opt.optimize(niter=niter)
        val = opt.synergistic_disclosure(opt._optima)
        return max(val, 0.0)
    except Exception:
        return 0.0


def backbone_disclosure(dist, sources=None, target=None, niter=None, bound=None):
    """
    Compute the full backbone decomposition of I(X; Y).

    The backbone decomposes I(X;Y) = sum_{m=1}^{n} B^m_partial, where each
    B^m_partial >= 0 measures the marginal gain from relaxing constraints on
    groups of m variables to groups of m-1.

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources and target.
    sources : iter of iters, None
        The source variable groups. If None, all but last are used.
    target : iter, None
        The target variable. If None, the last variable is used.
    niter : int, None
        Number of basin-hopping restarts per lattice node.
    bound : int, None
        Cardinality bound on V.

    Returns
    -------
    backbone : dict
        Keys are integers m=1..n, values are B^m_partial (non-negative
        backbone atoms).
    """
    from ..pid.syndisc import SynDisc

    sd = SynDisc(dist, sources=sources, target=target, niter=niter, bound=bound)
    n = len(sd._sources)
    return {m: sd.get_backbone_atom(m) for m in range(1, n + 1)}


@unitful
def modified_synergistic_disclosure(dist, sources, target, alpha, niter=None, bound=None):
    """
    Compute the modified synergistic disclosure for a single constraint alpha.

    For singleton alpha (|alpha| = 1), returns I(T : a^C | a) instead of
    the optimization-based S_alpha.  For |alpha| >= 2, falls back to the
    standard synergistic disclosure.

    Gutknecht, Makkeh & Wibral (2023), "From Babel to Boole: The Logical
    Organization of Information Decompositions", arXiv:2306.00734v2, eq. 52.

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources and target.
    sources : list of lists
        Each inner list gives the indices of one source variable group.
    target : list
        The indices of the target variable.
    alpha : list of lists
        Each inner list gives source-list indices (0-based) specifying
        which sources form each constraint subgroup.
    niter : int, None
        Number of basin-hopping restarts (used only for |alpha| >= 2).
    bound : int, None
        Cardinality bound on V (used only for |alpha| >= 2).

    Returns
    -------
    s_msd : float
        The modified synergistic disclosure, in bits.
    """
    alpha_tuples = tuple(tuple(a) for a in alpha)

    if not alpha_tuples:
        from .coinformation import coinformation

        return coinformation(dist, [list(flatten(sources)), list(target)])

    if len(alpha_tuples) == 1:
        from .coinformation import coinformation

        constrained_vars = list(flatten(sources[i] for i in alpha_tuples[0]))
        all_source_vars = list(flatten(sources))
        total = coinformation(dist, [all_source_vars, list(target)])
        source_mi = coinformation(dist, [constrained_vars, list(target)])
        return total - source_mi

    return synergistic_disclosure(
        dist,
        sources,
        target,
        alpha,
        niter=niter,
        bound=bound,
    )


@unitful
def self_synergy(dist, sources=None, alpha=None, niter=None, bound=None):
    """
    Compute S_alpha(X -> X), the self-synergy of sources X.

    When alpha is None, uses the full individual-source constraints
    alpha = {{0}, {1}, ..., {n-1}}.

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources.
    sources : list of lists, None
        The source variable groups. If None, each variable is its own group.
    alpha : list of lists, None
        Constraint subgroups (source-list indices). If None, each individual
        source is constrained.
    niter : int, None
        Number of basin-hopping restarts.
    bound : int, None
        Cardinality bound on V.

    Returns
    -------
    s_self : float
        The self-synergy, in bits.
    """
    if sources is None:
        sources = dist.rvs

    flat_sources = [list(s) for s in sources]
    target = list(flatten(flat_sources))

    from ..pid.syndisc import SyndiscOptimizer

    alpha = tuple((i,) for i in range(len(flat_sources))) if alpha is None else tuple(tuple(a) for a in alpha)

    opt = SyndiscOptimizer(
        dist,
        flat_sources,
        target,
        alpha,
        bound=bound,
    )
    opt.optimize(niter=niter)
    val = opt.synergistic_disclosure(opt._optima)
    return max(val, 0.0)
