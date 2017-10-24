"""
Classes implementing the partial information decomposition.
"""

from __future__ import division

from sys import version_info

from itertools import product

import networkx as nx
import numpy as np

import prettytable

from .lattice import ascendants, descendants, least_upper_bound, pid_lattice, sort_key
from .. import ditParams
from ..multivariate import coinformation
from ..utils import flatten, powerset


class BasePID(object):
    """
    This implements the basic Williams & Beer Partial Information Decomposition.
    """

    def __init__(self, dist, inputs=None, output=None, reds=None, pis=None):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        inputs : iter of iters, None
            The set of input variables. If None, `dist.rvs[:-1]` is used.
        output : iter, None
            The output variable. If None, `dist.rvs[-1]` is used.
        """
        self._dist = dist

        if inputs is None:
            inputs = dist.rvs[:-1]
        if output is None:
            output = dist.rvs[-1]

        self._red_string = "I_r"
        self._pi_string = "pi"
        self._inputs = tuple(map(tuple, inputs))
        self._output = tuple(output)
        self._lattice = pid_lattice(self._inputs)
        self._total = coinformation(self._dist, [list(flatten(self._inputs)), self._output])
        self._compute(reds, pis)

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
        return self.get_partial(key)

    def __repr__(self): # pragma: no cover
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
            return super(BasePID, self).__repr__()

    def __str__(self):
        """
        Return a string representation of the PID.

        Returns
        -------
        pid : str
            The PID as a string.
        """
        return self.to_string()

    def _compute(self, reds=None, pis=None):
        """
        Use the redundancy measure to populate the lattice.
        """
        if reds is None: # pragma: no cover
            reds = {}
        if pis is None: # pragma: no cover
            pis = {}

        for node in self._lattice:
            if node not in reds:
                reds[node] = self._measure(self._dist, node, self._output)

        reds, pis = self._compute_mobius_inversion(reds=reds, pis=pis)

        nx.set_node_attributes(self._lattice, name='red', values=reds)
        nx.set_node_attributes(self._lattice, name='pi', values=pis)

    def _compute_mobius_inversion(self, reds=None, pis=None):
        """
        Perform as much of a Mobius inversion as possible.

        Parameters
        ----------
        reds : dict
            Currently known redundancy values.
        pis : dict
            Currently known partial information values.

        Returns
        -------
        reds : dict
            Updated redundancy values.
        pis : dict
            Updated partial information values.
        """
        if reds is None: # pragma: no cover
            reds = {}
        if pis is None:
            pis = {}

        for node in reversed(list(nx.topological_sort(self._lattice))):
            if node not in pis:
                try:
                    pis[node] = reds[node] - sum(pis[n] for n in descendants(self._lattice, node))
                except KeyError:
                    pass

        return reds, pis

    def get_redundancy(self, node):
        """
        Return the redundancy associated with `node`.

        Parameters
        ----------
        node : tuple of tuples
            The node to get the redundancy for.

        Returns
        -------
        red : float
            The redundancy associated with `node`.
        """
        return self._lattice.node[node]['red']

    def get_partial(self, node):
        """
        Return the partial information associated with `node`.

        Parameters
        ----------
        node : tuple of tuples
            The node to get the partial information for.

        Returns
        -------
        pi : float
            The partial information associated with `node`.
        """
        return self._lattice.node[node]['pi']

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

        if ditParams['text.font'] == 'linechar': # pragma: no cover
            try:
                table.set_style(prettytable.BOX_CHARS)
            except AttributeError:
                pass

        table.float_format[red_string] = '{}.{}'.format(digits + 2, digits)
        table.float_format[pi_string] = '{}.{}'.format(digits + 2, digits)

        for node in sorted(self._lattice, key=sort_key(self._lattice)):
            node_label = ''.join('{{{}}}'.format(':'.join(map(str, n))) for n in node)
            red_value = self.get_redundancy(node)
            pi_value = self.get_partial(node)
            if np.isclose(0, red_value, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):
                red_value = 0.0
            if np.isclose(0, pi_value, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):
                pi_value = 0.0
            table.add_row([node_label, red_value, pi_value])

        return table.get_string()

    @property
    def name(self):
        """
        Get the name of the decomposition. If colorama is available, the name will be styled
        according to its properties.

        Returns
        -------
        name : str
            The name of the decomposition.
        """
        try: # pragma: no cover
            from colorama import Fore, Back, Style
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
        pis = nx.get_node_attributes(self._lattice, 'pi')
        return all(pi >= -1e-6 for pi in pis.values() if not np.isnan(pi))

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
        equal_pi = super(BaseIncompletePID, self).__eq__(other)
        equal_red = (np.isclose(self.get_redundancy(node), other.get_redundancy(node), atol=1e-5, rtol=1e-5) for node in self._lattice)
        return equal_pi and all(equal_red)

    def _compute_lattice_monotonicity(self, reds, pis):
        """
        Infer the redundancy and partial information of lattice elements via lattice monotonicity.

        Parameters
        ----------
        reds : dict
            Currently known redundancy values.
        pis : dict
            Currently known partial information values.

        Returns
        -------
        reds : dict
            Updated redundancy values.
        pis : dict
            Updated partial information values.
        """
        # everything below a redundancy of 0 is a redundany of 0
        nodes = list(nx.topological_sort(self._lattice))
        while nodes:
            node = nodes.pop(0)
            if node in reds and np.isclose(0, reds[node]):
                for n in descendants(self._lattice, node):
                    if n not in reds:
                        reds[n] = 0
                        nodes.remove(n)

        # everything above a redundancy of I(inputs, output) is I(inputs, output)
        nodes = list(reversed(list(nx.topological_sort(self._lattice))))
        while nodes:
            node = nodes.pop(0)
            if node in reds and np.isclose(reds[node], self._total):
                for n in ascendants(self._lattice, node):
                    if n not in reds:
                        reds[n] = self._total
                        nodes.remove(n)

        # if redundancy of A == redundancy of B, then for all A -> C -> B, redundancy of C = redundancy of A, B
        tops = [node for node in self._lattice if node in reds and any((n not in reds) for n in self._lattice[node])]
        bottoms = [node for node in self._lattice if
                   node in reds and any((n not in reds) for n in self._lattice.reverse()[node])]
        for top, bottom in product(tops, bottoms):
            if np.isclose(reds[top], reds[bottom], atol=1e-5, rtol=1e-5):
                for path in nx.all_simple_paths(self._lattice, top, bottom):
                    for node in path[1:-1]:
                        if node not in reds:
                            reds[node] = reds[top]

        # if redundancy of A is equal to the reundancy any of A's children, then pi(A) = 0
        for node in self._lattice:
            if node not in pis:
                if node in reds and all(n in reds for n in self._lattice[node]) and self._lattice[node]:
                    if any(np.isclose(reds[n], reds[node], atol=1e-5, rtol=1e-5) for n in self._lattice[node]):
                        pis[node] = 0

        return reds, pis

    def _compute_attempt_linsolve(self, reds, pis):
        """
        Infer a linear constraint matrix from missing PI values and the mobius inversion.

        Parameters
        ----------
        reds : dict
            Currently known redundancy values.
        pis : dict
            Currently known partial information values.

        Returns
        -------
        reds : dict
            Updated redundancy values.
        pis : dict
            Updated partial information values.
        """
        missing_vars = [node for node in self._lattice if node not in pis]
        if not missing_vars:
            return reds, pis

        def predicate(node, nodes):
            a = node in reds
            b = all((n in pis or n in nodes) for n in descendants(self._lattice, node, self=True))
            return a and b

        for vars in reversed(list(powerset(missing_vars))[1:]):

            lub = least_upper_bound(self._lattice, vars, predicate)


            if lub is None:
                continue

            row = lambda node: [1 if (c in descendants(self._lattice, node, self=True)) else 0 for c in vars]

            A = np.array([row(node) for node in vars if node in reds] + [[1] * len(vars)])
            if version_info >= (3, 0, 0): # not sure why this is needed...
                A = A.T
            b = np.array([reds[node] for node in vars if node in reds] + [reds[lub] - sum(pis[node] for node in descendants(self._lattice, lub, True) if node in pis)])
            try:
                new_pis = np.linalg.solve(A, b)
                if np.all(new_pis > -1e-6):
                    for node, pi in zip(vars, new_pis):
                        pis[node] = pi

                    for node in self._lattice:
                        if node not in reds:
                            try:
                                reds[node] = sum(pis[n] for n in descendants(self._lattice, node, self=True))
                            except KeyError:
                                pass

                    break

            except:
                pass

        return reds, pis

    def _compute_single_child(self, reds, pis):
        """
        If a node has a single child, and both redundancies are known, then the PI of the node
        is the difference in the redundancies.

        Parameters
        ----------
        reds : dict
            Currently known redundancy values.
        pis : dict
            Currently known partial information values.

        Returns
        -------
        reds : dict
            Updated redundancy values.
        pis : dict
            Updated partial information values.
        """
        # if a node has only a single child, and you know both its redundancy and its partial
        # then you know the redundancy of the child
        for node in self._lattice:
            if node in reds and node in pis and len(self._lattice[node]) == 1:
                n = next(iter(self._lattice[node]))
                if n not in reds:
                    reds[n] = reds[node] - pis[node]

        return reds, pis

    def _compute(self, reds=None, pis=None):
        """
        Use a variety of methods to fill out as much of the lattice as possible.

        Parameters
        ----------
        reds : dict, None
            Currently known redundancy values.
        pis : dict, None
            Currently known partial information values.
        """
        if reds is None:
            reds = {}
        if pis is None:
            pis = {}

        # set redundancies of single input sets to I(input, output) and
        # plug in computed unique values
        if self.SELF_REDUNDANCY:
            for node in self._lattice:
                if len(node) == 1:
                    reds[node] = coinformation(self._dist, [node[0], self._output])

        if self.LATTICE_MONOTONICITY:
            reds, pis = self._compute_lattice_monotonicity(reds, pis)

        # if a node exists in a smaller PID, use that to compute redundancy (if possible)
        if self.REDUCED_PID:
            for node in self._lattice:
                if node not in reds and len(node) < len(self._inputs):
                    sub_pid = self.__class__(self._dist.copy(), node, self._output)
                    reds[node] = sub_pid.get_redundancy(node)

        while True:
            num_reds = len(reds)
            num_pis = len(pis)

            # if a node has a single child, their redundancies determine the node's partial information
            reds, pis = self._compute_single_child(reds=reds, pis=pis)

            # if the lattice is monotonic, then everything below a zero is zero, and everything above a max is max
            if self.LATTICE_MONOTONICITY:
                reds, pis = self._compute_lattice_monotonicity(reds=reds, pis=pis)

            # do as much of the mobius inversion as possible
            reds, pis = self._compute_mobius_inversion(reds=reds, pis=pis)

            # see if the remaining pis can be solved with linear constraints
            reds, pis = self._compute_attempt_linsolve(reds=reds, pis=pis)

            if len(reds) == num_reds and len(pis) == num_pis:
                break

        # if we know all but one partial, we know the last
        diff = set(self._lattice) - set(pis)
        if len(diff) == 1:
            pis[diff.pop()] = self._total - sum(pis.values())

        # if the sum of known PIs is I(inputs, output), all other PIs are zero
        if np.isclose(sum(pis.values()), self._total):
            for node in self._lattice:
                if node not in pis or np.isnan(pis[node]):
                    pis[node] = 0

        # plug in nan for all unknown values
        for node in self._lattice:
            if node not in reds:
                reds[node] = np.nan
            if node not in pis:
                pis[node] = np.nan

        nx.set_node_attributes(self._lattice, name='red', values=reds)
        nx.set_node_attributes(self._lattice, name='pi', values=pis)

    @BasePID.consistent.getter
    def consistent(self):
        """
        Determine if the assignment of values to the lattice is self-consistant.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistent, False otherwise.
        """
        reds = nx.get_node_attributes(self._lattice, 'red')
        pis = nx.get_node_attributes(self._lattice, 'pi')

        if self.SELF_REDUNDANCY: # pragma: no cover
            for node in self._lattice:
                if len(node) == 1:
                    red = reds[node]
                    mi = coinformation(self._dist, [node[0], self._output])
                    if not np.isclose(red, mi, atol=1e-5, rtol=1e-5):
                        return False

        # ensure that the mobius inversion holds
        for node in self._lattice:
            red = reds[node]
            parts = sum(pis[n] for n in descendants(self._lattice, node, self=True))
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
        pis = nx.get_node_attributes(self._lattice, 'pi')
        return not any(np.isnan(pi) for pi in pis.values())


class BaseUniquePID(BaseIncompletePID):
    """
    PID class for measures which define only unique informations.
    """

    def _compute(self, reds=None, pis=None):
        """
        """
        uniques = self._measure(self._dist, self._inputs, self._output)
        if pis is None:
            pis = {}

        for node in self._lattice:
            if len(node) == 1 and node[0] in uniques and node not in pis:
                pis[node] = uniques[node[0]]

        super(BaseUniquePID, self)._compute(reds=reds, pis=pis)


class BaseBivariatePID(BaseIncompletePID):
    """
    PID class for measures which define only a bivariate measure of redundancy.
    """

    def _compute(self, reds=None, pis=None):
        """
        """
        if reds is None:
            reds = {}
        for node in self._lattice:
            if len(node) == 2 and node not in reds:
                reds[node] = self._measure(self._dist, node, self._output)

        super(BaseBivariatePID, self)._compute(reds=reds, pis=pis)
