"""
Base class for Partial Entropy Decompositions (PEDs).

PEDs decompose the joint entropy of a set of source variables (no target)
over the free distributive lattice via Mobius inversion.  This contrasts with
PIDs, which decompose the mutual information between sources and a target.

Concrete PED measures subclass ``BasePED`` and provide ``_name`` and
``_measure``.
"""

from ..helpers import as_linear
from ..multivariate import entropy
from ..utils import build_table
from ._utils import _label
from .pid import BasePID, _descendants, _lattice_of, _unit_scale, _validate_sources, sort_key

__all__ = (
    "BasePED",
    "PointwisePED",
)


class BasePED(BasePID):
    """
    Base class for Partial Entropy Decompositions.

    PEDs decompose the joint entropy of a set of source variables
    (no target) over the free distributive lattice via Mobius inversion.

    Subclasses must provide:
      * ``_name``    -- display name for the decomposition
      * ``_measure`` -- static method computing the redundancy functional
    """

    _red_string = "H_r"
    _pi_string = "H_d"

    def __init__(self, dist, sources=None, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        sources : iter of iters, None
            The set of variables to include. If None, ``dist.rvs`` is used.
        """
        self._unit_scale = _unit_scale(dist)
        self._dist = dist = as_linear(dist)

        if sources is None:
            sources = dist.rvs

        self._kwargs = kwargs
        self._sources = tuple(map(tuple, sources))
        _validate_sources(dist, self._sources)
        self._target = None
        self._lattice = _lattice_of(self._sources)
        self._total = self._scaled(entropy(self._dist, rvs=self._sources))
        self._reds = {}
        self._pis = {}
        self._computed = False
        self._ensure_computed()


class PointwisePED(BasePED):
    """
    A partial entropy decomposition whose redundancy is the expectation of a
    pointwise (per-outcome) redundancy.

    Subclasses provide ``_pw_measure(dist, node, target)``, returning
    ``{outcome: value}``, in addition to ``_name`` and ``_measure``. When
    ``pointwise=True`` the pointwise redundancies and atoms (by per-outcome
    Mobius inversion) are kept as well.
    """

    def __init__(self, dist, sources=None, pointwise=False, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        sources : iter of iters, None
            The set of variables to include. If None, `dist.rvs` is used.
        pointwise : bool
            If True, also compute per-outcome redundancies and PI atoms.
        """
        self._pointwise = pointwise
        self._pw_reds = {}
        self._pw_pis = {}
        super().__init__(dist, sources=sources, **kwargs)

    def _compute(self):
        super()._compute()
        if self._pointwise:
            self._compute_pointwise()

    def _compute_pointwise(self):
        """Compute pointwise redundancies and PI atoms for all lattice nodes."""
        for node in self._lattice:
            self._pw_reds[node] = self._pw_measure(self._dist, node, self._target, **self._kwargs)

        any_node = next(iter(self._pw_reds))
        outcomes = list(self._pw_reds[any_node].keys())

        for node in reversed(list(self._lattice)):
            pw_pi = {}
            for o in outcomes:
                desc_sum = sum(self._pw_pis[n][o] for n in _descendants(self._lattice, node))
                pw_pi[o] = self._pw_reds[node].get(o, 0.0) - desc_sum
            self._pw_pis[node] = pw_pi

    def get_pw_red(self, node):
        """
        Pointwise redundancy values for *node*.

        Returns
        -------
        pw_red : dict
            ``{outcome: float}``
        """
        return self._pw_reds[node]

    def get_pw_pi(self, node):
        """
        Pointwise PI atom values for *node*.

        Returns
        -------
        pw_pi : dict
            ``{outcome: float}``
        """
        return self._pw_pis[node]

    def pw_to_string(self, outcome=None, digits=4):
        """
        Create a table of pointwise PED values.

        Parameters
        ----------
        outcome : tuple, optional
            If given, show values for this single outcome. Otherwise show all.
        digits : int
            Precision for display.

        Returns
        -------
        table : str
        """
        if not self._pw_pis:
            return "(pointwise PED not computed -- pass pointwise=True)"

        any_node = next(iter(self._pw_pis))
        all_outcomes = list(self._pw_pis[any_node].keys())
        outcomes = [outcome] if outcome is not None else all_outcomes
        red_string = "h_r"
        pi_string = "h_d"

        columns = [self._name, "outcome", red_string, pi_string]
        dist_name = getattr(self._dist, "name", "")
        base_title = f"{self._name} (pointwise)"
        title = f"{base_title} | {dist_name}" if dist_name else base_title
        table = build_table(columns, title=title)
        table.float_format[red_string] = f"{digits + 2}.{digits}"
        table.float_format[pi_string] = f"{digits + 2}.{digits}"

        for node in sorted(self._lattice, key=sort_key(self._lattice)):
            node_label = _label(node)
            for o in outcomes:
                red_val = self._pw_reds[node].get(o, 0.0)
                pi_val = self._pw_pis[node].get(o, 0.0)
                table.add_row([node_label, str(o), red_val, pi_val])

        return table.get_string()
