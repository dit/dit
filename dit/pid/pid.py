"""
Classes implementing the partial information decomposition.
"""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from itertools import product

from lattices.lattices import free_distributive_lattice
import networkx as nx
import numpy as np
import prettytable

from .. import ditParams
from ..multivariate import coinformation
from ..utils import flatten, powerset


def sort_key(lattice):
    """
    A key for sorting the nodes of a PID lattice.

    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice to sort.

    Returns
    -------
    key : function
        A function on nodes which returns the properties from which the lattice should be ordered.
    """
    pls = nx.shortest_path_length(lattice._lattice, source=lattice.top)

    def key(node):
        depth = pls[node]
        size = len(node)
        return depth, -size, node

    return key


def _transform(lattice):
    """
    Transform a free distributive lattice from being frozensets of frozensets
    of tuples of integers to being tuples of tuples of integers.

    Parameters
    ----------
    lattice : Lattice
        The lattice to transform.

    Returns
    -------
    tupled_lattice : Lattice
        The lattice, but with tuples in place of frozensets.
    """
    def tuplefy(n):
        return tuple(sorted((tuple(sorted(sum(_, ()))) for _ in n), key=lambda tup: (len(tup), tup)))

    def freeze(n):
        return frozenset(frozenset((__,) for __ in _) for _ in n)

    tuple_lattice = deepcopy(lattice)

    tuple_edges = [(tuplefy(e[0]), tuplefy(e[1])) for e in lattice._lattice.edges]
    tuple_lattice._lattice = nx.DiGraph(tuple_edges)

    tuple_lattice._relationship = lambda a, b: lattice._relationship(freeze(a), freeze(b))
    tuple_lattice.top = tuplefy(lattice.top)
    tuple_lattice.bottom = tuplefy(lattice.bottom)
    tuple_lattice._ts = [tuplefy(n) for n in lattice._ts]

    return tuple_lattice


class BasePID(metaclass=ABCMeta):
    """
    This implements the basic Williams & Beer Partial Information Decomposition.
    """

    _red_string = "I_r"
    _pi_string = "pi"

    def __init__(self, dist, inputs=None, output=None, reds=None, pis=None, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        inputs : iter of iters, None
            The set of input variables. If None, `dist.rvs` less indices
            in `output` is used.
        output : iter, None
            The output variable. If None, `dist.rvs[-1]` is used.
        reds : dict, None
            Redundancy values pre-assessed.
        pis : dict, None
            Partial information values pre-assessed.
        """
        self._dist = dist

        if output is None:
            output = dist.rvs[-1]
        if inputs is None:
            inputs = [var for var in dist.rvs if var[0] not in output]

        self._inputs = tuple(map(tuple, inputs))
        self._output = tuple(output)
        self._kwargs = kwargs

        self._lattice = _transform(free_distributive_lattice(self._inputs))
        self._inverse_lattice = self._lattice.inverse()
        self._total = coinformation(self._dist, [list(flatten(self._inputs)), self._output])

        self._reds = {} if reds is None else reds
        self._pis = {} if pis is None else pis

        self._compute()

    @abstractmethod
    def _measure(self, node, output):
        """
        Compute a redundancy value for `node`.

        Parameters
        ----------
        node : tuple(tuples)
            The lattice node to compute the redundancy of.
        output : iterable
            The indices to consider the target/output of the PID.

        Returns
        -------
        red : float
            The redundancy value.
        """
        pass

    @property
    @classmethod
    @abstractmethod
    def _name(cls):
        """
        The name of the PID.

        Returns
        -------
        name : str
            The name.
        """
        pass

    def __eq__(self, other):
        """
        Test if this and `other` are equal partial information decompositions.

        Parameters
        ----------
        other : BasePID

        Returns
        -------
        eq : bool
            If `self` and `other` are the same partial information decomposition.
        """
        return all(np.isclose(self[node], other[node], atol=1e-5, rtol=1e-5) for node in self._lattice)

    def __ne__(self, other):
        """
        Test if this and `other` are not equal.

        Parameters
        ----------
        other : BasePID

        Returns
        -------
        eq : bool
            If `self` and `other` are different partial information decomposition.
        """
        return not (self == other)

    def __getitem__(self, key):
        """
        Get the partial information value associated with `key`.

        Parameters
        ----------
        key : iterable of iterables
            The node to get the partial information of.

        Returns
        -------
        pi : float
            The partial information associated with `key`.
        """
        return float(self._pis[key])

    def __repr__(self):  # pragma: no cover
        """
        Returns a representation of the PID.

        Returns
        -------
        repr : str
            A representation of this object.
        """
        if ditParams['repr.print']:
            return self.to_string()
        else:
            return super().__repr__()

    def __str__(self):
        """
        Return a string representation of the PID.

        Returns
        -------
        pid : str
            The PID as a string.
        """
        return self.to_string()

    def _compute(self):
        """
        Use the redundancy measure to populate the lattice.
        """
        for node in self._lattice:
            if node not in self._reds:  # pragma: no branch
                self._reds[node] = self._measure(self._dist, node, self._output, **self._kwargs)

        self._compute_mobius_inversion()

    def _compute_mobius_inversion(self):
        """
        Perform as much of a Mobius inversion as possible.
        """
        for node in reversed(list(self._lattice)):
            if node not in self._pis:
                try:
                    self._pis[node] = self._reds[node] - sum(self._pis[n] for n in self._lattice.descendants(node))
                except KeyError:
                    pass

    def to_string(self, digits=4):
        """
        Create a table representing the redundancy and PI lattices.

        Parameters
        ----------
        digits : int
            The number of digits of precision to display.

        Returns
        -------
        table : str
            The table of values.
        """
        red_string = self._red_string
        pi_string = self._pi_string

        table = prettytable.PrettyTable([self.name, red_string, pi_string], title=getattr(self._dist, 'name', ''))

        if ditParams['text.font'] == 'linechar':  # pragma: no cover
            try:
                table.set_style(prettytable.UNICODE_LINES)
            except AttributeError:
                pass

        table.float_format[red_string] = '{}.{}'.format(digits + 2, digits)
        table.float_format[pi_string] = '{}.{}'.format(digits + 2, digits)

        for node in sorted(self._lattice, key=sort_key(self._lattice)):
            node_label = ''.join('{{{}}}'.format(':'.join(map(str, n))) for n in node)
            red_value = self._reds[node]
            pi_value = self._pis[node]
            if np.isclose(0, red_value, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):  # pragma: no cover
                red_value = 0.0
            if np.isclose(0, pi_value, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):  # pragma: no cover
                pi_value = 0.0
            table.add_row([node_label, red_value, pi_value])

        return table.get_string()

    @property
    def name(self):  # pragma: no cover
        """
        Get the name of the decomposition. If colorama is available, the name will be styled
        according to its properties.

        Returns
        -------
        name : str
            The name of the decomposition.
        """
        try:
            from colorama import Fore, Style
            inconsistent_style = lambda x: Fore.RED + x + Style.RESET_ALL
            negative_style = lambda x: Fore.GREEN + x + Style.RESET_ALL
            incomplete_style = lambda x: Fore.BLUE + x + Style.RESET_ALL
        except:
            inconsistent_style = lambda x: x
            negative_style = lambda x: x
            incomplete_style = lambda x: x

        if not self.consistent:
            return inconsistent_style(self._name)
        elif not self.nonnegative:
            return negative_style(self._name)
        elif not self.complete:
            return incomplete_style(self._name)
        else:
            return self._name

    @property
    def consistent(self):
        """
        Determine if the assignment of values to the lattice is self-consistent.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistent, False otherwise.
        """
        return True

    @property
    def nonnegative(self):
        """
        Determine if the partial information values are all non-negative.

        Returns
        -------
        nonnegative : bool
            True if all pi values are non-negative, False otherwise.
        """
        nonnegative = all(np.round(pi, 4) >= 0 for pi in self._pis.values() if not np.isnan(pi))
        return nonnegative

    @property
    def complete(self):
        """
        Determine if all partial information values are assigned.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistant, False otherwise.
        """
        return True


class BaseIncompletePID(BasePID):
    """
    A special PID class for measures which do not compute the redundancy of an arbitrary antichain.

    Properties
    ----------
    LATTICE_MONOTONICITY : bool
    REDUCED_PID : bool
    SELF_REDUNDANCY : bool
    """

    LATTICE_MONOTONICITY = True
    REDUCED_PID = True
    SELF_REDUNDANCY = True

    def __eq__(self, other):
        """
        Test if this and `other` are equal partial information decompositions.

        Parameters
        ----------
        other : BasePID

        Returns
        -------
        eq : bool
            If `self` and `other` are the same partial information decomposition.
        """
        equal_pi = super().__eq__(other)
        equal_red = (np.isclose(self._reds[node], other._reds[node], atol=1e-5, rtol=1e-5) for node in self._lattice)
        return equal_pi and all(equal_red)

    def _compute_lattice_monotonicity(self):
        """
        Infer the redundancy and partial information of lattice elements via lattice monotonicity.
        """
        # everything below a redundancy of 0 is a redundancy of 0
        nodes = list(self._lattice)
        while nodes:
            node = nodes.pop(0)
            if node in self._reds and np.isclose(0, self._reds[node]):
                for n in self._lattice.descendants(node):
                    if n not in self._reds:
                        self._reds[n] = 0
                        nodes.remove(n)

        # everything above a redundancy of I(inputs, output) is I(inputs, output)
        nodes = list(reversed(list(self._lattice)))
        while nodes:
            node = nodes.pop(0)
            if node in self._reds and np.isclose(self._reds[node], self._total):
                for n in self._lattice.ascendants(node):
                    if n not in self._reds:
                        self._reds[n] = self._total
                        nodes.remove(n)

        # if redundancy of A == redundancy of B, then for all A -> C -> B, redundancy of C = redundancy of A, B
        tops = [node for node in self._lattice if node in self._reds and any((n not in self._reds) for n in self._lattice.covers(node))]
        bottoms = [node for node in self._lattice if
                   node in self._reds and any((n not in self._reds) for n in self._inverse_lattice.covers(node))]
        for top, bottom in product(tops, bottoms):
            if np.isclose(self._reds[top], self._reds[bottom], atol=1e-5, rtol=1e-5):
                for path in nx.all_simple_paths(self._lattice._lattice, top, bottom):
                    for node in path[1:-1]:
                        if node not in self._reds:
                            self._reds[node] = self._reds[top]

        # if redundancy of A is equal to the redundancy any of A's children, then pi(A) = 0
        for node in self._lattice:
            if node not in self._pis and node in self._reds:
                if any(np.isclose(self._reds[n], self._reds[node], atol=1e-5, rtol=1e-5) for n in self._lattice.covers(node) if n in self._reds):
                    self._pis[node] = 0

    def _compute_attempt_linsolve(self):
        """
        Infer a linear constraint matrix from missing PI values and the mobius inversion.
        """
        missing_rvs = [node for node in self._lattice if node not in self._pis]
        if not missing_rvs:
            return

        def predicate(nodes):
            def inner(node):
                a = node in self._reds
                b = all((n in self._pis or n in nodes) for n in self._lattice.descendants(node, include=True))
                return a and b
            return inner

        for rvs in reversed(list(powerset(missing_rvs))[1:]):

            try:
                lub = self._lattice.join(*rvs, predicate=predicate(rvs))
            except ValueError:
                continue

            row = lambda node: [1 if (c in self._lattice.descendants(node, include=True)) else 0 for c in rvs]

            A = np.array([row(node) for node in rvs if node in self._reds] + [[1] * len(rvs)])
            b = np.array([self._reds[node] for node in rvs if node in self._reds] + \
                         [self._reds[lub] - sum(self._pis[node] for node in self._lattice.descendants(lub, include=True) if node in self._pis)])
            try:
                new_pis = np.linalg.solve(A, b)
                if np.all(new_pis > -1e-6):
                    for node, pi in zip(rvs, new_pis):
                        self._pis[node] = pi

                    for node in self._lattice:
                        if node not in self._reds:
                            try:
                                self._reds[node] = sum(self._pis[n] for n in self._lattice.descendants(node, include=True))
                            except KeyError:  # pragma: no cover
                                pass

                    break

            except:
                pass

    def _compute_single_child(self):
        """
        If a node has a single child, and both redundancies are known, then the PI of the node
        is the difference in the redundancies.
        """
        # if a node has only a single child, and you know both its redundancy
        # and its partial then you know the redundancy of the child
        for node in self._lattice:
            if node in self._reds and node in self._pis and len(self._lattice.covers(node)) == 1:
                n = next(iter(self._lattice.covers(node)))
                if n not in self._reds:
                    self._reds[n] = self._reds[node] - self._pis[node]

    def _compute(self):
        """
        Use a variety of methods to fill out as much of the lattice as possible.
        """
        # set redundancies of single input sets to I(input, output) and
        # plug in computed unique values
        if self.SELF_REDUNDANCY:  # pragma: no branch
            for node in self._lattice:
                if len(node) == 1:
                    self._reds[node] = coinformation(self._dist, [node[0], self._output])

        if self.LATTICE_MONOTONICITY:  # pragma: no branch
            self._compute_lattice_monotonicity()

        # if a node exists in a smaller PID, use that to compute redundancy (if possible)
        if self.REDUCED_PID:  # pragma: no branch
            for node in self._lattice:
                if node not in self._reds and len(node) < len(self._inputs):
                    sub_pid = self.__class__(self._dist.copy(), node, self._output)
                    self._reds[node] = sub_pid._reds[node]

        while True:
            num_reds = len(self._reds)
            num_pis = len(self._pis)

            # if a node has a single child, their redundancies determine the node's partial information
            self._compute_single_child()

            # if the lattice is monotonic, then everything below a zero is zero, and everything above a max is max
            if self.LATTICE_MONOTONICITY:  # pragma: no branch
                self._compute_lattice_monotonicity()

            # do as much of the mobius inversion as possible
            self._compute_mobius_inversion()

            # see if the remaining pis can be solved with linear constraints
            self._compute_attempt_linsolve()

            if len(self._reds) == num_reds and len(self._pis) == num_pis:
                break

        # if we know all but one partial, we know the last
        # note: this might be subsumed by _compute_attempt_linsolve
        diff = set(self._lattice) - set(self._pis)
        if len(diff) == 1:  # pragma: no cover
            self._pis[diff.pop()] = self._total - sum(self._pis.values())

        # if the sum of known PIs is I(inputs, output), all other PIs are zero
        # note: this might be subsumed by _compute_attempt_linsolve
        if np.isclose(sum(self._pis.values()), self._total):
            for node in self._lattice:
                if node not in self._pis or np.isnan(self._pis[node]):  # pragma: no cover
                    self._pis[node] = 0

        # plug in nan for all unknown values
        for node in self._lattice:
            if node not in self._reds:
                self._reds[node] = np.nan
            if node not in self._pis:
                self._pis[node] = np.nan

    @BasePID.consistent.getter
    def consistent(self):
        """
        Determine if the assignment of values to the lattice is self-consistant.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistent, False otherwise.
        """
        if self.SELF_REDUNDANCY:  # pragma: no branch
            for node in self._lattice:
                if len(node) == 1:
                    red = self._reds[node]
                    mi = coinformation(self._dist, [node[0], self._output])
                    if not np.isclose(red, mi, atol=1e-5, rtol=1e-5):  # pragma: no cover
                        return False

        # ensure that the mobius inversion holds
        for node in self._lattice:
            red = self._reds[node]
            parts = sum(self._pis[n] for n in self._lattice.descendants(node, include=True))
            if not np.isnan(red) and not np.isnan(parts):
                if not np.isclose(red, parts, atol=1e-5, rtol=1e-5):
                    return False

        return True

    @BasePID.complete.getter
    def complete(self):
        """
        Determine if all partial information values are assigned.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistant, False otherwise.
        """
        return not any(np.isnan(pi) for pi in self._pis.values())


class BaseUniquePID(BaseIncompletePID):
    """
    PID class for measures which define only unique informations.
    """

    def _compute(self):
        """
        """
        uniques = self._measure(self._dist, self._inputs, self._output, **self._kwargs)

        for node in self._lattice:
            if len(node) == 1 and node[0] in uniques and node not in self._pis:
                self._pis[node] = uniques[node[0]]

        super()._compute()


class BaseBivariatePID(BaseIncompletePID):
    """
    PID class for measures which define only a bivariate measure of redundancy.
    """

    def _compute(self):
        """
        """
        for node in self._lattice:
            if len(node) == 2 and node not in self._reds:
                self._reds[node] = self._measure(self._dist, node, self._output, **self._kwargs)

        super()._compute()
