"""
The Integrated Information Decomposition (PhiID).

PhiID decomposes the mutual information between a set of source variables
(e.g. the past of a system) and a set of target variables (e.g. its future)
over the product of the two redundancy lattices :cite:`mediano2019beyond`.
Each atom :math:`\\alpha \\to \\beta` is the information that the sources
carry in the "mode" :math:`\\alpha` and the targets in the mode :math:`\\beta`;
for two sources and two targets there are 16 atoms, among them persistent
redundancy :math:`\\{1\\}\\{2\\} \\to \\{1\\}\\{2\\}` and downward causation
:math:`\\{12\\} \\to \\{1\\}`.

The atoms are obtained by Mobius inversion of a *double-redundancy* function
:math:`I_\\partial(\\alpha \\to \\beta)` on the product lattice. Three are
provided:

* ``'mmi'`` — :math:`\\min_{i,j} I[A_i : B_j]` :cite:`mediano2019beyond`.
* ``'pm'`` — the expectation of
  :math:`\\min_i h(a_i) + \\min_j h(b_j) - \\min_{i,j} h(a_i, b_j)`, after
  Finn & Lizier :cite:`finn2020generalised`.
* ``'sx'`` — the expectation of
  :math:`h_{sx}(\\alpha) + h_{sx}(\\beta) - h_{sx}(\\{A_i \\cup B_j\\})`, using
  shared probability mass exclusions :cite:`varley2023decomposing`.
"""

from itertools import product

import numpy as np
from lattices.lattices import free_distributive_lattice

from ..helpers import as_linear
from ..multivariate import coinformation
from ..utils import build_table, flatten
from .hmos import h_mos_pw
from .hsx import h_sx_pw
from .pid import _transform, _unit_scale, sort_key

__all__ = ("PhiID",)


def _label(node):
    return "".join("{{{}}}".format(":".join(map(str, group))) for group in node)


class PhiID:
    """
    The Integrated Information Decomposition of :math:`I[X : Y]` for sources
    :math:`X` and targets :math:`Y`.
    """

    _red_string = "I_dd"
    _pi_string = "phi"

    def __init__(self, dist, sources=None, targets=None, measure="mmi", pointwise=False):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to decompose.
        sources, targets : iter of iters, None
            The source (past) and target (future) variables. If both are None,
            the first half of `dist.rvs` are the sources and the second half
            the targets.
        measure : {'mmi', 'pm', 'sx'}
            The double-redundancy function.
        pointwise : bool
            For the local measures (``'pm'`` and ``'sx'``), also keep the
            per-outcome double redundancies and atoms.
        """
        if measure not in {"mmi", "pm", "sx"}:
            raise ValueError(f"Unknown double-redundancy {measure!r}; choose from 'mmi', 'pm', 'sx'.")
        if pointwise and measure == "mmi":
            raise ValueError("The 'mmi' double redundancy has no pointwise form.")
        self._unit_scale = _unit_scale(dist)
        self._dist = dist = as_linear(dist)
        if sources is None and targets is None:
            rvs = dist.rvs
            if len(rvs) % 2:
                raise ValueError("Specify `sources` and `targets` for an odd number of variables.")
            sources, targets = rvs[: len(rvs) // 2], rvs[len(rvs) // 2 :]
        elif sources is None or targets is None:
            raise ValueError("Specify both `sources` and `targets`, or neither.")
        self._sources = tuple(map(tuple, sources))
        self._targets = tuple(map(tuple, targets))
        self._measure = measure
        self._source_lattice = _transform(free_distributive_lattice(self._sources))
        self._target_lattice = _transform(free_distributive_lattice(self._targets))
        self._nodes = list(product(self._source_lattice, self._target_lattice))
        self._total = self._scaled(
            coinformation(self._dist, [list(flatten(self._sources)), list(flatten(self._targets))])
        )
        self._mi_cache = {}
        self._reds = {}
        self._pis = {}
        self._pw_reds = {}
        self._pw_pis = {}
        if pointwise:
            self._compute_pointwise()
        for node in self._nodes:
            self.get_pi(node)

    def _scaled(self, value):
        return value * self._unit_scale

    def _mi(self, a, b):
        key = (a, b)
        if key not in self._mi_cache:
            self._mi_cache[key] = coinformation(self._dist, [list(a), list(b)])
        return self._mi_cache[key]

    def _pw_double_red(self, node):
        alpha, beta = node
        joint = tuple(tuple(sorted(set(a) | set(b))) for a in alpha for b in beta)
        pw = h_mos_pw if self._measure == "pm" else h_sx_pw
        ha, hb, hab = (pw(self._dist, n) for n in (alpha, beta, joint))
        return {o: ha[o] + hb[o] - hab[o] for o in ha}

    def _double_red(self, node):
        if self._measure == "mmi":
            alpha, beta = node
            return min(self._mi(a, b) for a in alpha for b in beta)
        pw = self._pw_reds.get(node) or self._pw_double_red(node)
        return sum(self._dist[o] * v for o, v in pw.items())

    def _descendants(self, node):
        """
        The nodes strictly below `node` in the product order.
        """
        alpha, beta = node
        below_a = [alpha, *self._source_lattice.descendants(alpha)]
        below_b = [beta, *self._target_lattice.descendants(beta)]
        return [n for n in product(below_a, below_b) if n != node]

    def _compute_pointwise(self):
        for node in self._nodes:
            self._pw_reds[node] = self._pw_double_red(node)
        outcomes = list(next(iter(self._pw_reds.values())))
        for node in sorted(self._nodes, key=lambda n: len(self._descendants(n))):
            below = self._descendants(node)
            self._pw_pis[node] = {o: self._pw_reds[node][o] - sum(self._pw_pis[n][o] for n in below) for o in outcomes}

    def get_red(self, node):
        """
        The double redundancy :math:`I_\\partial(\\alpha \\to \\beta)` of `node`.

        Parameters
        ----------
        node : (tuple of tuples, tuple of tuples)
            The pair :math:`(\\alpha, \\beta)` of source and target antichains.

        Returns
        -------
        red : float
        """
        if node not in self._reds:
            if node not in self._nodes:
                raise ValueError(f"{node!r} is not a node of the product lattice.")
            self._reds[node] = self._scaled(self._double_red(node))
        return self._reds[node]

    def get_pi(self, node):
        """
        The PhiID atom :math:`\\alpha \\to \\beta`.

        Parameters
        ----------
        node : (tuple of tuples, tuple of tuples)
            The pair :math:`(\\alpha, \\beta)` of source and target antichains.

        Returns
        -------
        pi : float
        """
        if node not in self._pis:
            self._pis[node] = self.get_red(node) - sum(self.get_pi(n) for n in self._descendants(node))
        return self._pis[node]

    def __getitem__(self, node):
        return self.get_pi(node)

    def get_pw_red(self, node):
        """
        Pointwise double redundancies of `node`, ``{outcome: float}`` (in bits).
        """
        return self._pw_reds[node]

    def get_pw_pi(self, node):
        """
        Pointwise atoms of `node`, ``{outcome: float}`` (in bits).
        """
        return self._pw_pis[node]

    @property
    def nodes(self):
        """
        The nodes :math:`(\\alpha, \\beta)` of the product lattice.
        """
        return list(self._nodes)

    def to_string(self, digits=4):
        """
        A table of the double redundancies and atoms.

        Parameters
        ----------
        digits : int
            Precision for display.

        Returns
        -------
        table : str
        """
        name = f"PhiID ({self._measure})"
        columns = [name, self._red_string, self._pi_string]
        dist_name = getattr(self._dist, "name", "")
        table = build_table(columns, title=f"{name} | {dist_name}" if dist_name else name)
        table.float_format[self._red_string] = f"{digits + 2}.{digits}"
        table.float_format[self._pi_string] = f"{digits + 2}.{digits}"
        src_key, tgt_key = sort_key(self._source_lattice), sort_key(self._target_lattice)
        for alpha, beta in sorted(self._nodes, key=lambda n: (src_key(n[0]), tgt_key(n[1]))):
            node = (alpha, beta)
            red, pi = self.get_red(node), self.get_pi(node)
            table.add_row([f"{_label(alpha)} -> {_label(beta)}", red, 0.0 if np.isclose(pi, 0) else pi])
        return table.get_string()

    def __str__(self):
        return self.to_string()

    def __repr__(self):  # pragma: no cover
        return self.to_string()
