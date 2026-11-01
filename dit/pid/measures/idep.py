"""
The dependency-decomposition based unique measure partial information decomposition.
"""

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


class PID_dep(BaseUniquePID):
    """
    The dependency partial information decomposition, as defined by James at al.
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

        Returns
        -------
        idep : dict
            The value of I_dep for each individual source.
        """
        uniques = {}
        measure = {"I": lambda d: coinformation(d, [[0, 1], [2]])}
        source_0_target = frozenset((frozenset((0, 2)),))
        source_1_target = frozenset((frozenset((1, 2)),))
        if len(sources) == 2:
            dm = d.coalesce(sources + (target,))  # put it into [0, 1], [2] order
            dd = DependencyDecomposition(dm, measures=measure, maxiter=maxiter)
            u_0 = min(dd.delta(edge, "I") for edge in dd.edges(source_0_target))
            u_1 = min(dd.delta(edge, "I") for edge in dd.edges(source_1_target))
            uniques[sources[0]] = u_0
            uniques[sources[1]] = u_1
        else:
            for source in sources:
                others = sum((i for i in sources if i != source), ())
                dm = d.coalesce([source, others, target])
                dd = DependencyDecomposition(dm, measures=measure, maxiter=maxiter)
                u = min(dd.delta(edge, "I") for edge in dd.edges(source_0_target))
                uniques[source] = u

        return uniques


class PID_RA(BaseUniquePID):
    """
    The "reproducibility analysis" partial information decomposition, derived
    from the work of Zwick.
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

        Returns
        -------
        ira : dict
            The value of I_RA for each individual source.
        """
        uniques = {}
        measure = {"I": lambda d: coinformation(d, [[0, 1], [2]])}
        source_0_target = frozenset([frozenset((0, 2))])
        source_1_target = frozenset([frozenset((1, 2))])
        all_pairs = frozenset([frozenset((0, 1))]) | source_0_target | source_1_target
        if len(sources) == 2:
            dm = d.coalesce(sources + (target,))
            dd = DependencyDecomposition(dm, measures=measure, maxiter=maxiter)
            u_0 = dd.delta((all_pairs, all_pairs - source_0_target), "I")
            u_1 = dd.delta((all_pairs, all_pairs - source_1_target), "I")
            uniques[sources[0]] = u_0
            uniques[sources[1]] = u_1
        else:
            for source in sources:
                others = sum((i for i in sources if i != source), ())
                dm = d.coalesce([source, others, target])
                dd = DependencyDecomposition(dm, measures=measure, maxiter=maxiter)
                u = dd.delta((all_pairs, all_pairs - source_0_target), "I")
                uniques[source] = u

        return uniques


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
    """
    An experimental, reduced variant of the dependency decomposition PID of James et al.

    As :class:`PID_dep_a`, but only over the edges whose constraints include the joint
    distribution of all the sources, so that the dependencies among the sources are always
    kept.

    Notes
    -----
    Experimental: this decomposition can result in subadditive redundancy.
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
