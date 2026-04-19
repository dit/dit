"""
Base class for Partial Entropy Decompositions (PEDs).

PEDs decompose the joint entropy of a set of source variables (no target)
over the free distributive lattice via Mobius inversion.  This contrasts with
PIDs, which decompose the mutual information between sources and a target.

Concrete PED measures subclass ``BasePED`` and provide ``_name`` and
``_measure``.
"""

from lattices.lattices import free_distributive_lattice

from ..multivariate import entropy
from .pid import BasePID, _transform

__all__ = ("BasePED",)


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
        self._dist = dist

        if sources is None:
            sources = dist.rvs

        self._kwargs = kwargs
        self._sources = tuple(map(tuple, sources))
        self._target = None
        self._lattice = _transform(free_distributive_lattice(self._sources))
        self._total = entropy(self._dist, rvs=self._sources)
        self._reds = {}
        self._pis = {}
        self._compute()
