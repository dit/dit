"""
The Generalized Information Decomposition (GID) of Varley (2024).

The GID decomposes the Kullback-Leibler divergence of a posterior :math:`q`
from a prior :math:`p` over the redundancy lattice of the variables. The local
divergence :math:`\\log q(x) / p(x) = h_p(x) - h_q(x)` is the difference of two
local entropies, so each atom is the posterior expectation of the difference
of the two pointwise partial entropy atoms :cite:`varley2024generalized`.
Mutual information, total correlation and negentropy are all divergences, so
choosing the prior recovers decompositions of each.
"""

import numpy as np

from ..divergences import kullback_leibler_divergence
from ..helpers import as_linear
from ..utils import build_table
from ._utils import _label
from .hmos import PED_MOS
from .hsx import PED_SX
from .pid import _descendants, _unit_scale, sort_key

__all__ = ("GID",)

_PEDS = {"sx": PED_SX, "mos": PED_MOS}


class GID:
    """
    The generalized information decomposition of
    :math:`D_{KL}[\\text{posterior} \\| \\text{prior}]`.
    """

    _red_string = "D_r"
    _pi_string = "D_d"

    def __init__(self, posterior, prior, sources=None, measure="sx"):
        """
        Parameters
        ----------
        posterior, prior : Distribution
            The distributions, over the same variables; the posterior's
            support must be within the prior's.
        sources : iter of iters, None
            The variables of the lattice. If None, `posterior.rvs` is used.
        measure : {'sx', 'mos'}
            The local partial entropy decomposition: :class:`PED_SX` or
            :class:`PED_MOS` (:math:`h_{\\min}`).
        """
        if measure not in _PEDS:
            raise ValueError(f"Unknown measure {measure!r}; choose from {sorted(_PEDS)}.")
        self._unit_scale = _unit_scale(posterior)
        self._dist = posterior = as_linear(posterior)
        prior = as_linear(prior)
        if not {o for o in posterior.outcomes if posterior[o] > 0} <= set(prior.outcomes):
            raise ValueError("The posterior's support must be contained in the prior's.")
        self._measure = measure
        ped = _PEDS[measure]
        self._posterior_ped = ped(posterior, sources=sources, pointwise=True)
        self._prior_ped = ped(prior, sources=sources, pointwise=True)
        self._lattice = self._posterior_ped._lattice
        self._sources = self._posterior_ped._sources
        self._total = kullback_leibler_divergence(posterior, prior) * self._unit_scale
        self._pw_pis = {}
        self._pis = {}
        for node in self._lattice:
            post, pri = self._posterior_ped.get_pw_pi(node), self._prior_ped.get_pw_pi(node)
            self._pw_pis[node] = {o: pri[o] - v for o, v in post.items()}
            self._pis[node] = self._unit_scale * sum(posterior[o] * v for o, v in self._pw_pis[node].items())

    def get_pi(self, node):
        """
        The atom of `node`.

        Parameters
        ----------
        node : tuple of tuples
            A node of the redundancy lattice.

        Returns
        -------
        pi : float
        """
        return self._pis[node]

    def __getitem__(self, node):
        return self.get_pi(node)

    def get_pw_pi(self, node):
        """
        The local atoms of `node`, ``{outcome: float}`` (in bits).
        """
        return self._pw_pis[node]

    def get_red(self, node):
        """
        The redundant divergence of `node`: the sum of the atoms at and below it.
        """
        return self._pis[node] + sum(self._pis[n] for n in _descendants(self._lattice, node))

    def to_string(self, digits=4):
        """
        A table of the redundant divergences and atoms.

        Parameters
        ----------
        digits : int
            Precision for display.

        Returns
        -------
        table : str
        """
        name = f"GID ({self._measure})"
        columns = [name, self._red_string, self._pi_string]
        dist_name = getattr(self._dist, "name", "")
        table = build_table(columns, title=f"{name} | {dist_name}" if dist_name else name)
        table.float_format[self._red_string] = f"{digits + 2}.{digits}"
        table.float_format[self._pi_string] = f"{digits + 2}.{digits}"
        for node in sorted(self._lattice, key=sort_key(self._lattice)):
            label = _label(node)
            pi = self.get_pi(node)
            table.add_row([label, self.get_red(node), 0.0 if np.isclose(pi, 0) else pi])
        return table.get_string()

    def __str__(self):
        return self.to_string()

    def __repr__(self):  # pragma: no cover
        return self.to_string()
