"""
The dependency-decomposition based unique measure partial information decomposition.
"""

from ...algorithms.pid_references import reference_coinformations
from ...multivariate import coinformation
from ...profiles import DependencyDecomposition
from ..pid import BaseUniquePID

__all__ = (
    "PID_RA",
    "PID_dep",
    "PID_dep_a",
    "PID_dep_b",
    "PID_dep_c",
)


def _closed_form_uniques(d, sources, target, redundancy, maxiter=None):
    """
    Unique informations from the closed-form redundancy of a source and the other sources.

    Each of these measures takes the unique information of a source from the dependency
    lattice of three variables: that source, the other sources together, and the target.
    On that lattice, the measure's redundancy is a function of the co-informations of the
    reference distributions :cite:`vansanten2026between`, so no lattice is built.

    Parameters
    ----------
    d : Distribution
        The distribution.
    sources : tuple of tuples
        The source variables.
    target : tuple
        The target variable.
    redundancy : callable
        Maps a :class:`~dit.algorithms.pid_references.ReferenceCoinformations` to the
        redundancy of the triplet.
    maxiter : int, None
        The iteration limit of the maximum entropy fit.

    Returns
    -------
    uniques : dict
        The unique information of each source.
    """
    if len(sources) == 2:
        references = reference_coinformations(d, sources, target, maxiter=maxiter)
        red = redundancy(references)
        return {source: mi - red for source, mi in zip(sources, references.mutual_informations, strict=True)}

    uniques = {}
    for source in sources:
        others = sum((i for i in sources if i != source), ())
        references = reference_coinformations(d, (source, others), target, maxiter=maxiter)
        uniques[source] = references.mutual_informations[0] - redundancy(references)
    return uniques


class PID_dep(BaseUniquePID):
    r"""
    The dependency PID of James et al. :cite:`james2017unique`.

    The unique information of a source is the least change in I(sources : target) along
    the edges of the dependency lattice that add the constraint of the source and the
    target. For two sources this four-edge minimum is equivalently
    :cite:`vansanten2026between`, Appendix A,

    .. math::

        R_{\mathrm{dep}} = \max\{C(P^{(2)}), I_{Q_{\mathrm{CI}}}(X_0;X_1)\},

    where :math:`P^{(2)}` matches all three pairwise marginals and
    :math:`Q_{\mathrm{CI}}(x_0,x_1,y)=p(y)p(x_0|y)p(x_1|y)`. The two terms are the
    redundancies of :class:`PID_RA` and :class:`PID_MES`, so this one is the larger of
    them. Only one pairwise maximum-entropy fit is needed. With more sources, each source
    is paired with the union of the other sources, and the same formula gives its unique
    information.
    """

    _name = "I_dep"

    @staticmethod
    def _measure(d, sources, target, maxiter=None):
        """
        This computes unique information as min(delta(I(sources : target))) where delta
        is taken over the dependency decomposition.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_dep for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        maxiter : int, None
            The iteration limit forwarded to the maximum-entropy optimizer.

        Returns
        -------
        idep : dict
            The value of I_dep for each individual source.
        """
        return _closed_form_uniques(d, sources, target, lambda r: max(r.pairwise, r.independent), maxiter=maxiter)


class PID_RA(BaseUniquePID):
    """
    The "reconstructability analysis" partial information decomposition, derived
    from the work of Zwick :cite:`zwick2004overview`.

    The unique information of a source is the change in I(sources : target) when the
    constraint of the source and the target is added to the other two pairwise
    constraints. With two sources, the redundancy is therefore the co-information of the
    pairwise maximum-entropy model, and the synergy is the third-order connected
    information :cite:`vansanten2026between`. Redundancy can be negative;
    :class:`PID_dep_c` is the nonnegative, clipped version for two sources. With more
    sources, each source is paired with the union of the other sources.
    """

    _name = "I_RA"

    @staticmethod
    def _measure(d, sources, target, maxiter=None):
        """
        This computes unique information as the change in I[sources : target]
        when adding the source-target constraint.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_RA for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        maxiter : int, None
            The iteration limit forwarded to the maximum-entropy optimizer.

        Returns
        -------
        ira : dict
            The value of I_RA for each individual source.
        """
        return _closed_form_uniques(d, sources, target, lambda r: r.pairwise, maxiter=maxiter)


def _full_dependency_decomposition(d, sources, target, maxiter=None):
    """
    The dependency decomposition of I(sources : target) over the lattice of all the
    sources and the target, each a single variable: the sources are 0..n-1 and the
    target is n.
    """
    n = len(sources)
    dm = d.coalesce([*sources, target])
    measure = {"I": lambda d: coinformation(d, [list(range(n)), [n]])}
    return DependencyDecomposition(dm, list(range(n + 1)), measures=measure, maxiter=maxiter), n


class PID_dep_a(BaseUniquePID):
    """
    An experimental variant of the dependency decomposition PID of James et al.

    The unique information of a source is the least change in I(sources : target) among
    the edges of the dependency lattice of all the sources and the target that add the
    constraint of the source and the target. With two sources this is :class:`PID_dep`;
    with more, :class:`PID_dep` coalesces the other sources into one variable.

    Notes
    -----
    Experimental: this alternative method behaves oddly with three or more sources, and
    its lattice grows quickly with the number of sources.
    """

    _name = "I_dep_a"

    @staticmethod
    def _measure(d, sources, target, maxiter=None):
        """
        Compute the unique information of each source.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_dep_a for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        maxiter : int, None
            The maximum number of iterations of each maximum entropy optimization.

        Returns
        -------
        idepa : dict
            The value of I_dep_a for each individual source.
        """
        dd, n = _full_dependency_decomposition(d, sources, target, maxiter=maxiter)
        return {
            source: min(dd.delta(edge, "I") for edge in dd.edges(frozenset([frozenset((i, n))])))
            for i, source in enumerate(sources)
        }


class PID_dep_b(BaseUniquePID):
    """
    An experimental, reduced variant of the dependency decomposition PID of James et al.

    As :class:`PID_dep_a`, but only over the edges whose constraints among several
    variables all involve the target, so that dependencies among the sources alone are
    never constrained.

    Notes
    -----
    Experimental: this decomposition is known to be inconsistent; for instance, its unique
    informations for two copies of a source are inconsistent with self-redundancy.
    """

    _name = "I_dep_b"

    @staticmethod
    def _measure(d, sources, target, maxiter=None):
        """
        Compute the unique information of each source.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_dep_b for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        maxiter : int, None
            The maximum number of iterations of each maximum entropy optimization.

        Returns
        -------
        idepb : dict
            The value of I_dep_b for each individual source.
        """
        dd, n = _full_dependency_decomposition(d, sources, target, maxiter=maxiter)

        def broja_style(edge):
            return all(n in constraint and len(constraint) > 1 for constraint in edge[0] if len(constraint) > 1)

        return {
            source: min(dd.delta(edge, "I") for edge in dd.edges(frozenset([frozenset((i, n))])) if broja_style(edge))
            for i, source in enumerate(sources)
        }


class PID_dep_c(BaseUniquePID):
    r"""
    A reduced dependency-lattice PID; pairwise-reference PID for two sources.

    As :class:`PID_dep_a`, but only over the edges whose constraints include the joint
    distribution of all the sources, so that the dependencies among the sources are always
    kept.

    For two sources this is the pairwise-reference PID of
    :cite:`vansanten2026between`, Definition 2. If :math:`P^{(2)}` is the
    maximum-entropy distribution matching all three pairwise marginals,
    :math:`C^{(2)}` its co-information, and
    :math:`I^{(3)}=H(P^{(2)})-H(P)`, then

    .. math::

        R = [C^{(2)}]_+, \qquad
        S = I^{(3)} + [-C^{(2)}]_+, \qquad
        U_i = I(X_i;Y) - R.

    Thus the bivariate decomposition is nonnegative and is the positive-part
    repair of :class:`PID_RA`, not a distinct maximum-entropy optimization; it is
    computed from one pairwise maximum-entropy fit.

    Notes
    -----
    With more than two sources this is an experimental full dependency-lattice
    extension. The bivariate equivalence and nonnegativity guarantee do not
    apply; the extension can give inconsistent decompositions.
    """

    _name = "I_dep_c"

    @staticmethod
    def _measure(d, sources, target, maxiter=None):
        """
        Compute the unique information of each source.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_dep_c for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        maxiter : int, None
            The maximum number of iterations of each maximum entropy optimization.

        Returns
        -------
        idepc : dict
            The value of I_dep_c for each individual source.
        """
        if len(sources) == 2:
            return _closed_form_uniques(d, sources, target, lambda r: max(r.pairwise, 0.0), maxiter=maxiter)
        dd, n = _full_dependency_decomposition(d, sources, target, maxiter=maxiter)
        sources_joint = frozenset(range(n))
        return {
            source: min(
                dd.delta(edge, "I")
                for edge in dd.edges(frozenset([frozenset((i, n))]))
                if any(sources_joint <= constraint for constraint in edge[0])
            )
            for i, source in enumerate(sources)
        }
