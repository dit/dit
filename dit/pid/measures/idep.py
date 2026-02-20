"""
The dependency-decomposition based unique measure partial information decomposition.
"""

from ...multivariate import coinformation
from ...profiles import DependencyDecomposition
from ..pid import BaseUniquePID

__all__ = (
    "PID_dep",
    "PID_RA",
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


class PID_dep_a(BaseUniquePID):
    """
    The dependency partial information decomposition, as defined by James at al.

    Notes
    -----
    This alternative method behaves oddly with three or more sources.
    """

    _name = "I_dep_a"

    @staticmethod
    def _measure(d, sources, target):  # pragma: no cover
        """
        This computes unique information as min(delta(I(sources : target))) where delta
        is taken over the dependency decomposition.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_dep_a for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        idepa : dict
            The value of I_dep_a for each individual source.
        """
        var_to_index = {var: i for i, var in enumerate(sources + (target,))}
        d = d.coalesce(sorted(var_to_index.keys(), key=lambda k: var_to_index[k]))
        invars = [var_to_index[var] for var in sources]
        outvar = [var_to_index[(var,)] for var in target]
        measure = {"I": lambda d: coinformation(d, [invars, outvar])}
        dd = DependencyDecomposition(d, list(var_to_index.values()), measures=measure)
        uniques = {}
        for source in sources:
            constraint = frozenset((frozenset((var_to_index[source], var_to_index[target])),))
            u = min(dd.delta(edge, "I") for edge in dd.edges(constraint))
            uniques[source] = u
        return uniques


class PID_dep_b(BaseUniquePID):
    """
    The reduced dependency partial information decomposition, as defined by James at al.

    Notes
    -----
    This decomposition is known to be inconsistent.
    """

    _name = "I_dep_b"

    @staticmethod
    def _measure(d, sources, target):  # pragma: no cover
        """
        This computes unique information as min(delta(I(sources : target))) where delta
        is taken over a restricted dependency decomposition which never constrains dependencies
        among the sources.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_dep_b for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        idepb : dict
            The value of I_dep_b for each individual source.
        """
        var_to_index = {var: i for i, var in enumerate(sources + (target,))}
        target_index = var_to_index[target]
        d = d.coalesce(sorted(var_to_index.keys(), key=lambda k: var_to_index[k]))
        invars = [var_to_index[var] for var in sources]
        outvar = [var_to_index[(var,)] for var in target]
        measure = {"I": lambda d: coinformation(d, [invars, outvar])}
        dd = DependencyDecomposition(d, list(var_to_index.values()), measures=measure)
        uniques = {}
        for source in sources:
            constraint = frozenset((frozenset((var_to_index[source], target_index)),))
            broja_style = lambda edge: all({target_index} < set(_) for _ in edge[0] if len(_) > 1)
            edge_set = (edge for edge in dd.edges(constraint) if broja_style(edge))
            u = min(dd.delta(edge, "I") for edge in edge_set)
            uniques[source] = u
        return uniques


class PID_dep_c(BaseUniquePID):
    """
    The reduced dependency partial information decomposition, as defined by James at al.

    Notes
    -----
    This decomposition can result in subadditive redundancy.
    """

    _name = "I_dep_c"

    @staticmethod
    def _measure(d, sources, target):  # pragma: no cover
        """
        This computes unique information as min(delta(I(sources : target))) where delta
        is taken over a restricted dependency decomposition which never constrains dependencies
        among the sources.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_dep_c for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        idepc : dict
            The value of I_dep_c for each individual source.
        """
        var_to_index = {var: i for i, var in enumerate(sources + (target,))}
        d = d.coalesce(sorted(var_to_index.keys(), key=lambda k: var_to_index[k]))
        invars = [var_to_index[var] for var in sources]
        outvar = [var_to_index[(var,)] for var in target]
        measure = {"I": lambda d: coinformation(d, [invars, outvar])}
        dd = DependencyDecomposition(d, list(var_to_index.values()), measures=measure)
        uniques = {}
        for source in sources:
            constraint = frozenset((frozenset((var_to_index[source], var_to_index[target])),))
            edge_set = (edge for edge in dd.edges(constraint) if tuple(invars) in edge[0])
            u = min(dd.delta(edge, "I") for edge in edge_set)
            uniques[source] = u
        return uniques
