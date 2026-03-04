"""
An xarray-backed distribution class for discrete random variables.

This module provides ``Distribution``, a distribution class built on top of
xarray DataArrays. Each dimension in the DataArray corresponds to a random
variable, coordinates along each dimension are that variable's alphabet, and
the array values are probabilities.

The class tracks which dimensions are **free** (being described) vs **given**
(conditioned on), enabling natural algebraic operations:

- ``p(X,Y) * p(Z|X,Y)`` yields ``p(X,Y,Z)`` (chain rule)
- ``p(Z|X,Y) * p(X)`` yields ``p(X,Z|Y)`` (partial application)
- ``p(X,Y) / p(X)`` yields ``p(Y|X)`` (conditioning by division)

These work because xarray automatically aligns arrays by dimension name
during arithmetic, and the free/given metadata tracks which variables are
being described vs conditioned on.

Examples
--------
>>> import dit
>>> from dit.distribution import Distribution
>>>
>>> d = dit.example_dists.Xor()
>>> d.set_rv_names("XYZ")
>>>
>>> p_xy = d.marginal('X', 'Y')       # p(X,Y)
>>> p_z_given_xy = d.condition_on('X', 'Y')  # p(Z|X,Y)
>>> p_xyz_rebuilt = p_xy * p_z_given_xy    # p(X,Y) * p(Z|X,Y) = p(X,Y,Z)
"""

import itertools

import numpy as np

try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

from .math import prng as _default_prng
from .math.ops import get_ops

__all__ = ("Distribution",)


def _check_xarray():
    """Raise an error if xarray is not available."""
    if not XARRAY_AVAILABLE:
        raise ImportError("xarray is required for Distribution. Install with: pip install xarray")


class Distribution:
    """
    A distribution backed by an xarray DataArray.

    The distribution tracks which dimensions are "free" (joint) variables
    and which are "given" (conditioned on). This allows natural algebraic
    operations:

    - Multiplying ``p(X,Y) * p(Z|X,Y)`` yields ``p(X,Y,Z)``
    - Summing over a free variable marginalizes it out
    - Dividing by a marginal conditions on it

    Attributes
    ----------
    data : xr.DataArray
        The underlying probability array.
    free_vars : frozenset of str
        The names of the free (joint) variables. For ``p(X,Y|Z)`` this
        is ``{'X','Y'}``.
    given_vars : frozenset of str
        The names of the conditioned variables. For ``p(X,Y|Z)`` this
        is ``{'Z'}``.
    ops : Operations
        The operations instance for the current probability base.

    Notes
    -----
    Normalization convention: for a distribution ``p(X,Y|Z)``, summing over
    all free variables (X and Y) for each fixed value of Z should yield 1.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        data,
        pmf=None,
        rv_names=None,
        free_vars=None,
        given_vars=None,
        base="linear",
        sample_space=None,
        sparse=True,
        trim=True,
        sort=True,
        validate=True,
        prng=None,
    ):
        """
        Initialize an Distribution.

        There are three construction modes:

        1. **DataArray** -- pass an ``xr.DataArray`` directly (original API).
        2. **Outcomes + pmf** -- pass a sequence of outcomes and a sequence
           of probabilities, matching the ``dit.Distribution`` signature.
        3. **Dict** -- pass a dict mapping outcomes to probabilities.

        Parameters
        ----------
        data : xr.DataArray, sequence, or dict
            If an ``xr.DataArray``, used directly as the probability data.
            If a dict, keys are outcomes and values are probabilities.
            Otherwise, treated as a sequence of outcomes (each outcome is
            an indexable container whose length equals the number of
            random variables).
        pmf : sequence of float, optional
            Probability values corresponding to *data* when *data* is a
            sequence of outcomes. Ignored when *data* is a DataArray or dict.
        rv_names : list of str, optional
            Names for each random variable. Only used when *data* is
            outcomes or a dict. Defaults to ``'X0'``, ``'X1'``, ...
        free_vars : set-like of str, optional
            Names of the free (joint) variables. If *both* ``free_vars``
            and ``given_vars`` are None, all dimensions are treated as free.
        given_vars : set-like of str, optional
            Names of the conditioned variables.
        base : str, float, or None
            The probability base. ``'linear'`` (default) for raw
            probabilities, ``2``, ``'e'``, or any positive float for log
            probabilities. If ``None``, auto-detected (linear if the
            pmf sums to ~1, else ``ditParams['base']``).
        sample_space : sequence or CartesianProduct, optional
            Explicit sample space. If provided, used to determine the
            full set of possible outcomes.
        sparse : bool
            If True, ``outcomes`` and ``pmf`` only report non-zero entries.
        trim : bool
            Ignored (kept for API compatibility).
        sort : bool
            Ignored (alphabets are always sorted).
        validate : bool
            If True, validate normalisation after construction.
        prng : random state, optional
            Pseudo-random number generator. Defaults to ``dit.math.prng``.

        Examples
        --------
        From outcomes and pmf (like ``dit.Distribution``):

        >>> xrd = Distribution(['00','01','10','11'],
        ...                      [.25, .25, .25, .25],
        ...                      rv_names=['X', 'Y'])

        From a dict:

        >>> xrd = Distribution({'00': .5, '11': .5}, rv_names=['X', 'Y'])

        From a DataArray (original API):

        >>> xrd = Distribution(my_dataarray, free_vars={'X', 'Y'})
        """
        _check_xarray()

        self.prng = _default_prng if prng is None else prng
        _rv_names_explicit = rv_names is not None

        if not trim:
            sparse = False

        # -- Dispatch: build a DataArray if outcomes were provided ----------
        if isinstance(data, xr.DataArray):
            # Original path: DataArray passed directly
            da = data
        else:
            # Outcomes path: data is outcomes (sequence or dict)
            if isinstance(data, dict):
                outcomes = list(data.keys())
                pmf = list(data.values())
            else:
                outcomes = list(data)
                if pmf is None:
                    # Distribution compat: a bare list of numbers is
                    # treated as a pmf with auto-generated integer outcomes.
                    if outcomes and all(isinstance(o, (int, float, np.integer, np.floating)) for o in outcomes):
                        pmf = outcomes
                        outcomes = list(range(len(pmf)))
                    else:
                        raise ValueError("pmf is required when data is a sequence of outcomes")
                pmf = list(pmf)

            if len(outcomes) == 0:
                raise ValueError("outcomes must be non-empty")
            if len(outcomes) != len(pmf):
                raise ValueError(f"outcomes and pmf must have the same length, got {len(outcomes)} and {len(pmf)}")

            # Auto-detect base when None
            if base is None:
                from .math import LinearOperations
                from .validate import is_pmf

                base = (
                    "linear"
                    if is_pmf(np.asarray(pmf, dtype=float), LinearOperations())
                    else __import__("dit").ditParams["base"]
                )

            # Detect scalar outcomes (int, float, etc.) and wrap in 1-tuples
            # so they fit into xarray's coordinate system.
            try:
                n = len(outcomes[0])
            except TypeError:
                outcomes = [(o,) for o in outcomes]
                n = 1
            if rv_names is None:
                rv_names = [f"X{i}" for i in range(n)]
            if len(rv_names) != n:
                raise ValueError(f"Expected {n} rv_names, got {len(rv_names)}")

            # Build alphabet from sample_space if provided, else from outcomes
            if sample_space is not None:
                from .samplespace import CartesianProduct

                if isinstance(sample_space, CartesianProduct):
                    alphabets = [sorted(a) for a in sample_space.alphabets]
                else:
                    ss_list = list(sample_space)
                    alphabets = [sorted({o[i] for o in ss_list}) for i in range(n)]
            else:
                alphabets = [sorted({o[i] for o in outcomes}) for i in range(n)]

            coords = {name: alpha for name, alpha in zip(rv_names, alphabets, strict=True)}

            shape = tuple(len(a) for a in alphabets)
            arr = np.zeros(shape)
            for outcome, p in zip(outcomes, pmf, strict=True):
                idx = tuple(alphabets[i].index(outcome[i]) for i in range(n))
                arr[idx] = p

            da = xr.DataArray(arr, dims=rv_names, coords=coords)

            # Default: all variables are free when constructing from outcomes
            if free_vars is None and given_vars is None:
                free_vars = set(rv_names)

        if base is None:
            base = "linear"

        # -- Common initialisation ------------------------------------------
        self.data = da
        self.ops = get_ops(base)

        all_dims = frozenset(da.dims)

        if free_vars is None and given_vars is None:
            self.free_vars = all_dims
            self.given_vars = frozenset()
        elif free_vars is not None and given_vars is not None:
            self.free_vars = frozenset(free_vars)
            self.given_vars = frozenset(given_vars)
        elif free_vars is not None:
            self.free_vars = frozenset(free_vars)
            self.given_vars = all_dims - self.free_vars
        else:  # given_vars only
            self.given_vars = frozenset(given_vars)
            self.free_vars = all_dims - self.given_vars

        if self.free_vars | self.given_vars != all_dims:
            raise ValueError(
                f"free_vars and given_vars must cover all dimensions. "
                f"Dims: {all_dims}, free: {self.free_vars}, "
                f"given: {self.given_vars}"
            )
        if self.free_vars & self.given_vars:
            raise ValueError(f"free_vars and given_vars must be disjoint. Overlap: {self.free_vars & self.given_vars}")

        self._outcome_class = tuple
        self._outcome_ctor = tuple
        self._sparse = sparse
        self._meta = {"is_joint": True, "is_numerical": True, "is_sparse": sparse}
        self._rv_names_set = _rv_names_explicit

    @classmethod
    def from_distribution(cls, dist, rv_names=None):
        """
        Create an Distribution from an existing distribution, optionally
        renaming its random variables.

        Parameters
        ----------
        dist : Distribution
            The source distribution.
        rv_names : list of str, optional
            Names for each random variable. If None, uses the
            distribution's existing rv_names, or defaults to ``'X0'``, ``'X1'``, etc.

        Returns
        -------
        xrd : Distribution
        """
        result = dist.copy(base="linear")
        result.make_dense()

        if rv_names is not None:
            n = result.outcome_length()
            if len(rv_names) != n:
                raise ValueError(f"Expected {n} variable names, got {len(rv_names)}")
            result.set_rv_names(rv_names)

        return result

    @classmethod
    def from_array(cls, arr, dim_names, alphabets, free_vars=None, given_vars=None, base="linear"):
        """
        Create an Distribution from a numpy array.

        Parameters
        ----------
        arr : np.ndarray
            The probability array.
        dim_names : list of str
            Names for each dimension.
        alphabets : list of list
            The alphabet (coordinate values) for each dimension.
        free_vars : set-like of str, optional
            Names of the free variables.
        given_vars : set-like of str, optional
            Names of the conditioned variables.
        base : str or float
            Probability base (``'linear'``, ``2``, ``'e'``, ...).

        Returns
        -------
        xrd : Distribution
        """
        _check_xarray()

        coords = {n: list(a) for n, a in zip(dim_names, alphabets, strict=True)}
        data = xr.DataArray(arr, dims=dim_names, coords=coords)
        result = cls(data, free_vars=free_vars, given_vars=given_vars, base=base)
        result._rv_names_set = True
        return result

    @classmethod
    def from_factors(cls, marginal, conditional):
        """
        Build a joint distribution from a marginal and a conditional.

        ``p(X,Y) = p(X) * p(Y|X)``

        Parameters
        ----------
        marginal : Distribution
            The marginal distribution, e.g. ``p(X)``.
        conditional : Distribution
            The conditional distribution, e.g. ``p(Y|X)``.

        Returns
        -------
        joint : Distribution
            The resulting joint distribution.
        """
        return marginal * conditional

    # ─────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def dims(self):
        """All dimension (variable) names as a tuple, in array order."""
        return tuple(self.data.dims)

    @property
    def shape(self):
        """Shape of the underlying array."""
        return self.data.shape

    @property
    def all_vars(self):
        """All variable names as a frozenset."""
        return self.free_vars | self.given_vars

    # ── Compatibility with dit.Distribution API ──────────────────────────

    @property
    def alphabet(self):
        """
        Tuple of alphabets, one per dimension (in array-dimension order).

        This mirrors ``dit.Distribution.alphabet``.
        """

        def _native(v):
            return v.item() if hasattr(v, "item") else v

        return tuple(tuple(_native(v) for v in self.data.coords[d].values) for d in self.data.dims)

    @property
    def outcomes(self):
        """
        Tuple of outcomes in lexicographic order.

        When sparse (the default), only non-zero probability outcomes are
        included.  After :meth:`make_dense`, all outcomes are included.

        For 1-D numerical distributions, outcomes are the coordinate values
        directly (e.g. ``(0, 1, 2)``).  For multi-variable distributions
        or distributions with non-numeric coordinates, each outcome is a
        tuple whose elements correspond to the dimensions in :attr:`dims`
        order.
        """
        dims = list(self.data.dims)
        coord_vals = [self.data.coords[d].values for d in dims]

        def _native(v):
            """Convert numpy scalar to Python native type."""
            return v.item() if hasattr(v, "item") else v

        def _wrap(combo):
            if self._unwrap_scalar:
                return _native(combo[0])
            return tuple(_native(v) for v in combo)

        if not self._sparse:
            return tuple(_wrap(combo) for combo in itertools.product(*coord_vals))
        arr = self._linear_data()
        outs = []
        for combo in itertools.product(*coord_vals):
            sel = {d: v for d, v in zip(dims, combo, strict=True)}
            p = float(arr.sel(sel))
            if p > 0:
                outs.append(_wrap(combo))
        return tuple(outs)

    @property
    def pmf(self):
        """
        1-D numpy array of probabilities corresponding to :attr:`outcomes`.

        Returns values in the current base (log if the distribution is in
        log space, linear otherwise), matching ``dit.Distribution.pmf``.
        """
        dims = list(self.data.dims)
        probs = []
        for o in self.outcomes:
            key = (o,) if self._unwrap_scalar else o
            sel = {d: v for d, v in zip(dims, key, strict=True)}
            probs.append(float(self.data.sel(sel)))
        return np.array(probs)

    @pmf.setter
    def pmf(self, value):
        """
        Set probabilities from a 1-D array.

        If the array length matches the current (sparse) outcomes, sets those.
        If it matches the full sample space size, sets all outcomes densely.
        """
        value = np.asarray(value, dtype=float)
        cur_outcomes = self.outcomes
        if len(value) == len(cur_outcomes):
            for o, p in zip(cur_outcomes, value, strict=True):
                self[o] = float(p)
        else:
            was_sparse = self._sparse
            self.make_dense()
            all_outcomes = self.outcomes
            if len(value) != len(all_outcomes):
                raise ValueError(
                    f"pmf length {len(value)} doesn't match outcomes "
                    f"(sparse={len(cur_outcomes)}, dense={len(all_outcomes)})"
                )
            for o, p in zip(all_outcomes, value, strict=True):
                self[o] = float(p)
            if was_sparse:
                self.make_sparse()

    def to_dict(self):
        """
        Return a dictionary mapping outcomes to probabilities.

        Returns
        -------
        d : dict
            ``{outcome_tuple: float}``
        """
        return dict(zip(self.outcomes, self.pmf.tolist(), strict=True))

    @property
    def _outcomes_index(self):
        """
        A dict mapping each outcome to its position in :attr:`outcomes`.

        Mirrors ``dit.Distribution._outcomes_index``.
        """
        return {o: i for i, o in enumerate(self.outcomes)}

    def _linear_data(self):
        """Return a DataArray guaranteed to be in linear probability space."""
        if self.is_log():
            return xr.DataArray(
                self.ops.exp(self.data.values),
                dims=self.data.dims,
                coords=self.data.coords,
            )
        return self.data

    def outcome_length(self):
        """
        Number of random variables (dimensions).

        Returns
        -------
        n : int
        """
        return len(self.data.dims)

    def get_rv_names(self):
        """
        Return the variable names as a tuple, or None if not explicitly set.

        Returns
        -------
        names : tuple of str or None
        """
        if not self._rv_names_set:
            return None
        return tuple(self.data.dims)

    def set_rv_names(self, rv_names):
        """
        Rename the dimensions (random variables).

        Parameters
        ----------
        rv_names : list of str
            New names, one per dimension.
        """
        rv_names = list(rv_names)
        if len(rv_names) != len(self.data.dims):
            raise ValueError(f"Expected {len(self.data.dims)} names, got {len(rv_names)}")

        old_dims = list(self.data.dims)
        if old_dims == rv_names:
            return

        # Two-pass rename via temporary names to avoid conflicts
        target_set = set(rv_names)
        tmp = {}
        for d in old_dims:
            if d in target_set and d != rv_names[old_dims.index(d)]:
                tmp[d] = f"__tmp_{d}_{id(self)}"
        if tmp:
            self.data = self.data.rename(tmp)

        current_dims = list(self.data.dims)
        final_map = {c: n for c, n in zip(current_dims, rv_names, strict=True) if c != n}
        if final_map:
            self.data = self.data.rename(final_map)

        full_map = dict(zip(old_dims, rv_names, strict=True))
        self.free_vars = frozenset(full_map.get(v, v) for v in self.free_vars)
        self.given_vars = frozenset(full_map.get(v, v) for v in self.given_vars)
        self._rvs = {name: i for i, name in enumerate(self.dims)}
        self._rv_names_set = True

    def __len__(self):
        """Number of outcomes currently represented (respects sparse/dense)."""
        if not self._sparse:
            return int(np.prod(self.data.shape))
        return int(np.count_nonzero(self._linear_data().values > 0))

    def __iter__(self):
        """Iterate over outcomes."""
        return iter(self.outcomes)

    def __reversed__(self):
        """Reverse-iterate over outcomes."""
        return reversed(self.outcomes)

    def __contains__(self, outcome):
        """
        Check if *outcome* is in the sample space.

        Parameters
        ----------
        outcome : tuple or dict
            If a tuple, must have length equal to ``len(self.dims)``.
            If a dict, keys must be dimension names.
        """
        from .exceptions import InvalidOutcome

        try:
            self[outcome]
            return True
        except (KeyError, IndexError, ValueError, InvalidOutcome):
            return False

    def is_joint(self):
        """
        True if this distribution describes more than one random variable.

        Returns False for 1-D distributions (single RV) and conditional
        distributions.
        """
        if self.is_conditional():
            return False
        return self.outcome_length() > 1

    def _is_unconditional(self):
        """True if there are no conditioned (given) variables."""
        return len(self.given_vars) == 0

    def is_conditional(self):
        """True if this is a conditional distribution."""
        return len(self.given_vars) > 0

    @property
    def _unwrap_scalar(self):
        """True when outcomes should be presented as bare values, not 1-tuples.

        This applies to 1-D distributions whose coordinates are numeric
        (int/float), e.g. ``binomial(10, 0.5)`` returns outcomes ``(0, 1, …, 10)``
        rather than ``((0,), (1,), …, (10,))``.
        """
        return self.outcome_length() == 1 and self.is_numerical()

    def is_log(self):
        """True if the distribution stores log probabilities."""
        return self.ops.base != "linear"

    def is_dense(self):
        """True when the distribution reports all outcomes (including zero-probability)."""
        return not self._sparse

    def is_sparse(self):
        """True when the distribution reports only non-zero outcomes."""
        return self._sparse

    def is_numerical(self):
        """True if all coordinate values across all dimensions are numeric.

        When True, operations like :meth:`mean`, :meth:`std`, and
        :meth:`variance` are well-defined.
        """
        import numbers

        for dim in self.dims:
            for v in self.data.coords[dim].values:
                val = v.item() if hasattr(v, "item") else v
                if not isinstance(val, numbers.Number):
                    return False
        return True

    def is_homogeneous(self):
        """True if the alphabet for each random variable is the same."""
        if len(self.alphabet) == 0:
            return True
        a1 = self.alphabet[0]
        return all(a == a1 for a in self.alphabet[1:])

    def has_outcome(self, outcome, null=True):
        """
        Check if *outcome* exists in the sample space.

        Parameters
        ----------
        outcome : tuple or str
            The outcome to check.
        null : bool
            If True, accept zero-probability outcomes in the sample space.
            If False, only accept outcomes with positive probability.
        """
        if isinstance(outcome, str) and len(self.data.dims) > 1:
            outcome = tuple(outcome)
        try:
            p = self[outcome]
        except Exception:
            return False
        if null:
            return True
        return p > 0

    def atoms(self, patoms=False):
        """
        Yield atoms of the probability space.

        Parameters
        ----------
        patoms : bool
            If True, yield only positive-probability atoms.
        """
        mode = "patoms" if patoms else "atoms"
        for outcome, _ in self.zipped(mode):
            yield outcome

    def event_space(self):
        """Return a generator over the event space (powerset of sample space)."""
        from dit.utils import powerset

        return powerset(list(self.sample_space()))

    def rand(self, size=None, rand=None, prng=None):
        """
        Return a random sample from the distribution.

        Parameters
        ----------
        size : int or None
            Number of samples. None for a single sample.
        rand : float, array, or None
            Pre-generated random numbers. None to generate internally.
        prng : random state, optional
            Random number generator. Defaults to ``self.prng``.
        """
        import dit.math

        return dit.math.sample(self, size, rand, prng)

    @property
    def rvs(self):
        """
        List of RV groupings, one per free variable.

        Each element is a single-element list containing a sequential
        integer index (0-based among the free dims), matching the
        ``dit.Distribution.rvs`` convention. Integer indices are used
        (rather than names) so that helpers like ``flatten`` do not
        recursively split multi-character strings.
        """
        free = [d for d in self.dims if d in self.free_vars]
        return [[i] for i in range(len(free))]

    @property
    def _mask(self):
        """
        Tuple of bools indicating which dims are given (conditioned on).

        Mirrors ``dit.Distribution._mask``.
        """
        return tuple(d in self.given_vars for d in self.dims)

    @property
    def _sample_space(self):
        """
        A ``CartesianProduct`` over the alphabets of all dimensions.

        Mirrors ``dit.Distribution._sample_space``.
        """
        from .samplespace import CartesianProduct

        if hasattr(self, "_sample_space_override"):
            return self._sample_space_override
        alphabets = [list(self.data.coords[d].values) for d in self.dims]
        return CartesianProduct(alphabets)

    @_sample_space.setter
    def _sample_space(self, value):
        self._sample_space_override = value

    @property
    def _product(self):
        """Product function for generating outcomes (itertools.product)."""
        return itertools.product

    def make_dense(self):
        """
        Switch to dense mode so that :attr:`outcomes` and :attr:`pmf`
        include all outcomes (including zero-probability ones).

        Returns
        -------
        int
            Always returns 0 (the DataArray is inherently dense).
        """
        self._sparse = False
        return 0

    def make_sparse(self, trim=True):
        """
        Switch to sparse mode so that :attr:`outcomes` and :attr:`pmf`
        include only non-zero probability outcomes (the default).

        Parameters
        ----------
        trim : bool, optional
            Ignored. Kept for API compatibility.

        Returns
        -------
        int
            Always returns 0.
        """
        self._sparse = True
        return 0

    def zipped(self, mode="pmf"):
        """
        Iterator over ``(outcome, probability)`` tuples.

        Parameters
        ----------
        mode : str
            ``'pmf'`` to iterate over non-zero outcomes (default),
            ``'atoms'`` to iterate over the full sample space,
            ``'patoms'`` is treated identically to ``'pmf'`` (provided
            for ``dit.Distribution`` compatibility).

        Yields
        ------
        outcome : scalar or tuple
        probability : float
        """
        dims = list(self.data.dims)
        coord_vals = [self.data.coords[d].values for d in dims]
        arr = self._linear_data()

        def _native(v):
            return v.item() if hasattr(v, "item") else v

        for combo in itertools.product(*coord_vals):
            sel = {d: v for d, v in zip(dims, combo, strict=True)}
            p = float(arr.sel(sel))
            if mode == "atoms" or p > 0:
                o = _native(combo[0]) if self._unwrap_scalar else tuple(_native(v) for v in combo)
                yield o, p

    # ── Base / ops ───────────────────────────────────────────────────────

    def get_base(self, numerical=False):
        """
        Return the current probability base.

        Parameters
        ----------
        numerical : bool
            If True and the base is ``'e'``, return its float value.

        Returns
        -------
        base : str or float
        """
        return self.ops.get_base(numerical=numerical)

    def set_base(self, base):
        """
        Change the probability base in-place.

        Parameters
        ----------
        base : str or float
            ``'linear'``, ``2``, ``'e'``, or any positive float.
        """
        self._set_base_inplace(base)

    def _set_base_inplace(self, base):
        """Change probability base in-place."""
        new_ops = get_ops(base)
        old_ops = self.ops

        if old_ops.base == new_ops.base:
            return

        values = self.data.values.copy()
        old_base = old_ops.base
        new_base = new_ops.base

        if old_base == "linear" and new_base != "linear":
            with np.errstate(divide="ignore"):
                values = new_ops.log(values)
        elif old_base != "linear" and new_base == "linear":
            values = old_ops.exp(values)
        elif old_base != "linear" and new_base != "linear":
            values = old_ops.exp(values)
            with np.errstate(divide="ignore"):
                values = new_ops.log(values)

        self.data = xr.DataArray(values, dims=self.data.dims, coords=self.data.coords)
        self.ops = new_ops

    # ─────────────────────────────────────────────────────────────────────
    # Representation
    # ─────────────────────────────────────────────────────────────────────

    def _notation(self):
        """
        Build a string like ``'p(X,Y|Z)'`` describing this distribution.
        """
        # Use dims order for stable output
        ordered_free = [d for d in self.dims if d in self.free_vars]
        ordered_given = [d for d in self.dims if d in self.given_vars]
        free = ",".join(ordered_free)
        if ordered_given:
            given = ",".join(ordered_given)
            return f"p({free}|{given})"
        return f"p({free})"

    def __repr__(self):
        from .params import ditParams

        if ditParams["repr.print"]:
            return self.to_string()

        free = ",".join(d for d in self.dims if d in self.free_vars)
        given = ",".join(d for d in self.dims if d in self.given_vars)
        if given:
            label = f"p({free}|{given})"
        elif free:
            label = f"p({free})"
        else:
            label = "p()"
        return f"<Distribution {label}>"

    def __str__(self):
        return self.to_string()

    def to_string(self, digits=None, exact=None, tol=1e-9, show_mask=False, str_outcomes=False):
        """
        Return a string representation compatible with dit.Distribution format.

        Parameters
        ----------
        digits : int or None
            Round probabilities. None for no rounding.
        exact : bool or None
            If True, display as fractions. None uses ditParams.
        tol : float
            Fraction tolerance when exact=True.
        show_mask : bool
            Ignored (kept for API compatibility).
        str_outcomes : bool
            If True, attempt to join tuple outcomes into strings.
        """
        from io import StringIO

        from .math import approximate_fraction
        from .params import ditParams

        s = StringIO()

        if exact is None:
            exact = ditParams["print.exact"]

        d = self.copy(base="linear") if exact else self
        pmf = d.pmf.round(digits) if digits is not None and digits is not False else d.pmf
        if exact:
            pmf = [approximate_fraction(x, tol) for x in pmf]

        outcomes = list(d.outcomes)
        if str_outcomes and self.is_joint():
            try:
                outcomes = ["".join(str(v) for v in o) for o in outcomes]
            except Exception:
                outcomes = [str(o) for o in outcomes]
        else:
            outcomes = [str(o) for o in outcomes]

        max_length = max(map(len, outcomes)) if outcomes else 0
        free = ",".join(dim for dim in self.dims if dim in self.free_vars)
        given = ",".join(dim for dim in self.dims if dim in self.given_vars)
        if given:
            plabel = f"{free}|{given}"
        elif free:
            plabel = free
        else:
            plabel = "x"
        pstr = f"log p({plabel})" if d.is_log() else f"p({plabel})"
        base = d.get_base()

        headers = ["Class: ", "Alphabet: ", "Base: "]
        vals = [self.__class__.__name__, self._native_alphabet(self.alphabet), base]
        L = max(map(len, headers))
        for head, val in zip(headers, vals, strict=True):
            s.write(f"{head.ljust(L)}{val}\n")
        s.write("\n")

        s.write("".join(["x".ljust(max_length), "   ", pstr, "\n"]))
        for o, p in zip(outcomes, pmf, strict=True):
            s.write("".join([o.ljust(max_length), "   ", str(p), "\n"]))

        s.seek(0)
        result = s.read()
        return result[:-1] if result.endswith("\n") else result

    def to_html(self, digits=None, exact=None, tol=1e-9):  # pragma: no cover
        """
        Return an HTML representation compatible with dit.Distribution format.
        """
        from .math import approximate_fraction
        from .params import ditParams

        if exact is None:
            exact = ditParams["print.exact"]

        d = self.copy(base="linear") if exact else self
        pmf = d.pmf.round(digits) if digits is not None and digits is not False else d.pmf
        if exact:
            pmf = [approximate_fraction(x, tol) for x in pmf]

        outcomes = list(d.outcomes)
        if not self.is_joint():
            outcomes = [(o,) for o in outcomes]

        base = d.get_base()
        info = [
            ("Class", self.__class__.__name__),
            ("Alphabet", self._native_alphabet(self.alphabet)),
            ("Base", base),
        ]
        infos = "".join(f"<tr><th>{a}:</th><td>{b}</td></tr>" for a, b in info)
        header = f'<table border="1">{infos}</table>'

        rv_names = list(self.get_rv_names())
        pstr = "log p(x)" if d.is_log() else "p(x)"
        table_header = "<tr>" + "".join(f"<th>{a}</th>" for a in rv_names) + f"<th>{pstr}</th></tr>"
        table_rows = "".join(
            "<tr>" + "".join(f"<td>{_}</td>" for _ in o) + f"<td>{p}</td></tr>"
            for o, p in zip(outcomes, pmf, strict=True)
        )
        table = f"<table>{table_header}{table_rows}</table>"
        return f'<div><div style="float: left">{header}</div><div style="float: left">{table}</div></div>'

    def _repr_html_(self):
        """
        Rich HTML representation for Jupyter notebooks.

        Returns
        -------
        html : str
        """
        return self._to_html()

    # ── Display helpers ───────────────────────────────────────────────

    @staticmethod
    def _native_alphabet(alphabet):
        """
        Convert an alphabet tuple to native Python types for clean display.

        Parameters
        ----------
        alphabet : tuple of tuples
            Raw alphabet from ``self.alphabet``.

        Returns
        -------
        clean : tuple of tuples
        """

        def _native(v):
            """Convert numpy scalars to native Python types."""
            if hasattr(v, "item"):
                return v.item()
            return v

        return tuple(tuple(_native(v) for v in alpha) for alpha in alphabet)

    @staticmethod
    def _fmt_prob(p, digits=None):
        """
        Format a probability value for display.

        Parameters
        ----------
        p : float
            Probability value.
        digits : int or None
            Number of digits to round to. ``None`` for a default compact
            representation.

        Returns
        -------
        s : str
        """
        if digits is not None:
            return str(round(p, digits))
        # Compact default: up to 6 significant figures, strip trailing zeros
        return f"{p:.6g}"

    def _to_string(self, digits=None):
        """
        Build a plain-text representation of the distribution.

        Parameters
        ----------
        digits : int or None
            Round probabilities to this many digits. ``None`` for a compact
            default format.

        Returns
        -------
        s : str
        """
        from io import StringIO

        s = StringIO()

        notation = self._notation()
        base = self.get_base()
        alphabet = self._native_alphabet(self.alphabet)
        free_str = ", ".join(sorted(self.free_vars))
        given_str = ", ".join(sorted(self.given_vars)) if self.given_vars else "(none)"

        s.write("Class:     Distribution\n")
        s.write(f"Notation:  {notation}\n")
        s.write(f"Alphabet:  {alphabet}\n")
        s.write(f"Base:      {base}\n")
        s.write(f"Free vars: {{{free_str}}}\n")
        s.write(f"Given:     {given_str}\n")
        s.write("\n")

        dims = list(self.data.dims)
        arr = self._linear_data()

        # Gather all rows (non-zero for joint, all for conditional)
        rows = []
        coord_vals = [self.data.coords[d].values for d in dims]
        for combo in itertools.product(*coord_vals):
            sel = {d: v for d, v in zip(dims, combo, strict=True)}
            p = float(arr.sel(sel))
            if p > 0 or not self._is_unconditional():
                rows.append((combo, p))

        if not rows:
            s.write("(empty distribution)\n")
            s.seek(0)
            return s.read().rstrip()

        # Format probabilities and outcome values
        str_vals = [
            (tuple(str(v.item() if hasattr(v, "item") else v) for v in combo), self._fmt_prob(p, digits))
            for combo, p in rows
        ]

        col_sep = "   "
        col_widths = [max(len(str(d)), max(len(sv[0][i]) for sv in str_vals)) for i, d in enumerate(dims)]
        prob_header = "p" if self._is_unconditional() else "p(·|·)"
        prob_width = max(len(prob_header), max(len(sv[1]) for sv in str_vals))

        header = col_sep.join(str(d).ljust(w) for d, w in zip(dims, col_widths, strict=True))
        header += col_sep + prob_header.rjust(prob_width)
        s.write(header + "\n")

        for combo_strs, p_str in str_vals:
            line = col_sep.join(v.ljust(w) for v, w in zip(combo_strs, col_widths, strict=True))
            line += col_sep + p_str.rjust(prob_width)
            s.write(line + "\n")

        s.seek(0)
        return s.read().rstrip()

    def _to_html(self, digits=None):
        """
        Build an HTML representation of the distribution for notebooks.

        Parameters
        ----------
        digits : int or None
            Round probabilities to this many digits. ``None`` for a compact
            default format.

        Returns
        -------
        html : str
        """
        notation = self._notation()
        base = self.get_base()
        alphabet = self._native_alphabet(self.alphabet)
        free_str = ", ".join(sorted(self.free_vars))
        given_str = ", ".join(sorted(self.given_vars)) if self.given_vars else "—"

        # Info table
        info_rows = [
            ("Class", "Distribution"),
            ("Notation", f"<code>{notation}</code>"),
            ("Alphabet", str(alphabet)),
            ("Base", str(base)),
            ("Free vars", f"{{{free_str}}}"),
            ("Given vars", f"{{{given_str}}}"),
        ]
        info_html = "".join(
            f'<tr><th style="text-align:left; padding:2px 8px;">{k}:</th><td style="padding:2px 8px;">{v}</td></tr>'
            for k, v in info_rows
        )

        dims = list(self.data.dims)
        arr = self._linear_data()

        # Gather rows
        rows = []
        coord_vals = [self.data.coords[d].values for d in dims]
        for combo in itertools.product(*coord_vals):
            sel = {d: v for d, v in zip(dims, combo, strict=True)}
            p = float(arr.sel(sel))
            if p > 0 or not self._is_unconditional():
                rows.append((combo, p))

        prob_header = "p" if self._is_unconditional() else "p(·|·)"

        # Probability table
        th_style = 'style="text-align:center; padding:2px 8px; border-bottom:2px solid #333;"'
        td_style = 'style="text-align:center; padding:2px 8px;"'
        td_prob_style = 'style="text-align:right; padding:2px 8px; font-family:monospace;"'

        thead = "<tr>" + "".join(f"<th {th_style}>{d}</th>" for d in dims) + f"<th {th_style}>{prob_header}</th></tr>"

        tbody_rows = []
        for combo, p in rows:
            val_str = self._fmt_prob(p, digits)
            native = (v.item() if hasattr(v, "item") else v for v in combo)
            cells = "".join(f"<td {td_style}>{v}</td>" for v in native)
            cells += f"<td {td_prob_style}>{val_str}</td>"
            tbody_rows.append(f"<tr>{cells}</tr>")
        tbody = "".join(tbody_rows)

        if not rows:
            ncols = len(dims) + 1
            tbody = f'<tr><td colspan="{ncols}" style="text-align:center; padding:8px; color:#888;">(empty)</td></tr>'

        html = (
            '<div style="display:flex; gap:24px; align-items:flex-start; '
            'flex-wrap:wrap;">'
            f'<table style="border-collapse:collapse;">{info_html}</table>'
            f'<table style="border-collapse:collapse;">'
            f"<thead>{thead}</thead><tbody>{tbody}</tbody></table>"
            "</div>"
        )
        return html

    # ─────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────

    def validate(self, atol=1e-9):
        """
        Validate normalisation.

        For a joint distribution ``p(X,Y)``, the total sum should be 1.
        For a conditional ``p(X|Y)``, the sum over X for each Y should be 1.

        Parameters
        ----------
        atol : float
            Absolute tolerance.

        Returns
        -------
        valid : bool

        Raises
        ------
        ValueError
            If the distribution is not properly normalised.
        """
        arr = self._linear_data()

        if self._is_unconditional():
            total = float(arr.sum())
            if not np.isclose(total, 1.0, atol=atol):
                raise ValueError(f"Distribution sums to {total}, expected 1.0")
        else:
            sums = arr.sum(dim=list(self.free_vars))
            vals = sums.values.ravel()
            nonzero = vals[vals > atol]
            if len(nonzero) > 0 and not np.allclose(nonzero, 1.0, atol=atol):
                raise ValueError(f"Conditional distribution does not normalise properly. Sums over free vars:\n{sums}")
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Core probability operations
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_rv_names(self, rvs):
        """
        Resolve a list of RV specs (indices or names) to dimension names.

        Integers are treated as positional indices into ``self.dims``.
        Strings are treated as dimension names.

        Parameters
        ----------
        rvs : list
            Random variable identifiers -- integers (indices) or strings
            (dimension names).

        Returns
        -------
        names : list of str
        """
        if rvs and all(isinstance(r, (int, np.integer)) for r in rvs):
            try:
                return [self.dims[i] for i in rvs]
            except IndexError as err:
                from .exceptions import ditException

                raise ditException(f"RV index out of range: {rvs} for {len(self.dims)} dims") from err

        return list(rvs)

    def marginal(self, *args):
        """
        Marginalise to keep only the specified free variables.

        Given (conditioned) variables are always kept.

        Supports two call signatures:

        - ``marginal('X', 'Y')`` -- positional variable names
        - ``marginal(['X', 'Y'])`` -- list of names (or integer indices)

        Parameters
        ----------
        *args : str, or a single list/tuple
            The free variable names to keep. Integer indices are
            auto-resolved to dimension names.

        Returns
        -------
        result : Distribution
        """

        if len(args) == 1 and isinstance(args[0], (list, tuple, frozenset, set, range)):
            keep_vars = self._resolve_rv_names(list(args[0]))
        else:
            keep_vars = list(args)

        keep = frozenset(keep_vars)
        invalid = keep - self.free_vars
        if invalid:
            from .exceptions import ditException

            raise ditException(
                f"Cannot keep {invalid}: not free variables. Free: {self.free_vars}, given: {self.given_vars}"
            )

        sum_over = list(self.free_vars - keep)
        if not sum_over:
            return self.copy()

        if self.is_log():
            lin = self._linear_data()
            new_data = lin.sum(dim=sum_over)
            new_ops = self.ops
            new_data = xr.DataArray(
                new_ops.log(new_data.values),
                dims=new_data.dims,
                coords=new_data.coords,
            )
            result = Distribution(new_data, free_vars=keep, given_vars=self.given_vars, base=self.ops.base)
        else:
            new_data = self.data.sum(dim=sum_over)
            result = Distribution(new_data, free_vars=keep, given_vars=self.given_vars)
        result._rv_names_set = self._rv_names_set
        return result

    def marginalize(self, *args):
        """
        Marginalise out (remove) the specified free variables.

        Supports two call signatures:

        - ``marginalize('X')`` -- positional variable names
        - ``marginalize(['X'])`` -- list of names (or integer indices)

        Parameters
        ----------
        *args : str, or a single list/tuple
            The free variable names to remove.

        Returns
        -------
        result : Distribution
        """

        if len(args) == 1 and isinstance(args[0], (list, tuple, frozenset, set)):
            drop_vars = self._resolve_rv_names(list(args[0]))
        else:
            drop_vars = list(args)

        drop = frozenset(drop_vars)
        invalid = drop - self.free_vars
        if invalid:
            raise ValueError(f"Cannot drop {invalid}: not free variables. Free: {self.free_vars}")
        keep = self.free_vars - drop
        return self.marginal(*keep)

    def coalesce(self, rvs, extract=False):
        """
        Return a new distribution after coalescing random variables.

        Each inner sequence in *rvs* defines one new random variable as a
        combination of original variables.  The result is a joint
        ``Distribution`` over ``len(rvs)`` new random variables whose
        outcomes are tuples (or the inner values when ``extract=True``
        with a single group).

        Parameters
        ----------
        rvs : sequence of sequences
            Each inner sequence contains variable names (or integer indices).
        extract : bool
            If ``True`` and ``len(rvs) == 1``, the single group's values
            are used directly as outcomes instead of being wrapped in
            1-tuples.

        Returns
        -------
        d : Distribution
        """
        from collections import defaultdict

        groups = [self._resolve_rv_names(list(rv)) for rv in rvs]

        if len(groups) > 1 and extract:
            raise ValueError("Cannot extract with more than one rv group")

        lin = self._linear_data()
        dims = list(self.data.dims)
        coord_vals = [self.data.coords[d].values for d in dims]

        accum = defaultdict(float)
        for combo in itertools.product(*coord_vals):
            dim_val = {d: v for d, v in zip(dims, combo, strict=True)}
            p = float(lin.sel(dim_val))
            if p == 0:
                continue
            inner = [tuple(dim_val[name] for name in grp) for grp in groups]
            key = inner[0] if len(groups) == 1 and extract else tuple(inner)
            accum[key] += p

        if not accum:
            raise ValueError("Distribution has no non-zero outcomes to coalesce")

        outcomes = sorted(accum.keys())
        pmf_vals = [accum[o] for o in outcomes]

        if len(groups) == 1 and extract:
            # Outcomes are flat tuples like ('0','1') -- each element
            # is one original variable. Build a standard Distribution.
            n_vars = len(groups[0])
            rv_names = [f"X{i}" for i in range(n_vars)]
            return Distribution(outcomes, pmf_vals, rv_names=rv_names)

        # Outcomes are tuples of tuples, e.g. (('0','0'), ('1',)).
        # Each position is a coalesced variable whose alphabet entries are
        # themselves tuples.  xarray can't use tuples as coordinates, so
        # we serialise them to strings for the coordinate labels.
        n_vars = len(groups)
        rv_names = [f"X{i}" for i in range(n_vars)]

        def _label(t):
            """Convert a tuple to a compact string label."""
            return ",".join(str(v.item() if hasattr(v, "item") else v) for v in t)

        alphabets_raw = [sorted({o[i] for o in outcomes}) for i in range(n_vars)]
        alphabets_str = [[_label(t) for t in alpha] for alpha in alphabets_raw]
        coords = {name: alpha for name, alpha in zip(rv_names, alphabets_str, strict=True)}

        shape = tuple(len(a) for a in alphabets_raw)
        arr = np.zeros(shape)
        for outcome, p in zip(outcomes, pmf_vals, strict=True):
            idx = tuple(alphabets_raw[i].index(outcome[i]) for i in range(n_vars))
            arr[idx] = p

        data = xr.DataArray(arr, dims=rv_names, coords=coords)
        return Distribution(data, free_vars=set(rv_names), given_vars=set())

    def condition_on(self, *cond_vars, rvs=None, crvs=None):
        """
        Condition on the specified free variables.

        Supports two call signatures:

        - **Native:** ``condition_on('X', 'Y')`` -- positional var names.
          Returns a single conditional ``Distribution``.
        - **dit-compat:** any of these forms triggers the dit-compatible
          return format ``(marginal, list_of_conditionals)``:

          - ``condition_on(crvs=['X'], rvs=['Y'])``
          - ``condition_on(['X'], rvs=['Y'])``  (positional crvs)
          - ``condition_on(crvs=['X'])``

          The returned list contains one ``Distribution`` per outcome
          of the conditioning variable.

        Parameters
        ----------
        *cond_vars : str, or a single list/tuple
            Variable names to condition on.  If a single list/tuple is
            passed *and* ``rvs`` is provided, it is interpreted as
            ``crvs`` (dit-compat positional form).
        rvs : list, optional
            Variables to keep in the conditional (dit-compat API).
        crvs : list, optional
            Variables to condition on (dit-compat API).

        Returns
        -------
        result : Distribution or tuple
            A single conditional distribution (native), or a
            ``(marginal, list_of_Distributions)`` tuple (dit-compat).

        Examples
        --------
        >>> p_xyz.condition_on('Z')   # native: returns p(X,Y|Z)
        >>> p_xyz.condition_on('X', 'Y')  # native: returns p(Z|X,Y)
        >>> marg, cdists = p_xyz.condition_on(crvs=['Z'])  # dit-compat
        >>> marg, cdists = p_xyz.condition_on(['Z'], rvs=['X'])  # dit-compat
        """
        # Detect dit-compat form: list/tuple positional arg, or keywords.
        # Single-string positional args go to the NATIVE path.
        _dit_compat = False
        if crvs is not None or rvs is not None:
            _dit_compat = True
        elif len(cond_vars) == 1 and isinstance(cond_vars[0], (list, tuple)):
            crvs = cond_vars[0]
            cond_vars = ()
            _dit_compat = True

        if _dit_compat:
            if crvs is None:
                # Unwrap a single list/tuple positional arg
                if len(cond_vars) == 1 and isinstance(cond_vars[0], (list, tuple)):
                    crvs = list(cond_vars[0])
                else:
                    crvs = list(cond_vars)
                cond_vars = ()
            cond_names = self._resolve_rv_names(list(crvs))
            if rvs is not None:
                keep_names = set(self._resolve_rv_names(list(rvs)))
            else:
                keep_names = self.free_vars - frozenset(cond_names)

            all_needed = set(cond_names) | keep_names
            to_drop = self.free_vars - all_needed
            sub = self.marginal(*all_needed) if to_drop else self.copy()

            marginal_dist = sub.marginal(*cond_names)
            cond_slices = sub._condition_on_slices(cond_names, list(keep_names))
            return marginal_dist, cond_slices

        # Native path: positional variable names
        cond = frozenset(cond_vars)
        return self._condition_on_names(cond)

    def _condition_on_names(self, cond):
        """
        Internal: condition on a frozenset of variable names.

        Returns a single conditional Distribution.
        """
        invalid = cond - self.free_vars
        if invalid:
            raise ValueError(f"Cannot condition on {invalid}: not free variables. Free: {self.free_vars}")

        new_free = self.free_vars - cond
        if not new_free:
            raise ValueError("Cannot condition on all free variables")

        lin = self._linear_data()
        marginal_data = lin.sum(dim=list(new_free))
        conditional_data = xr.where(marginal_data > 0, lin / marginal_data, 0.0)

        new_given = self.given_vars | cond

        if self.is_log():
            conditional_data = xr.DataArray(
                self.ops.log(conditional_data.values),
                dims=conditional_data.dims,
                coords=conditional_data.coords,
            )
            result = Distribution(conditional_data, free_vars=new_free, given_vars=new_given, base=self.ops.base)
        else:
            result = Distribution(conditional_data, free_vars=new_free, given_vars=new_given)
        result._rv_names_set = self._rv_names_set
        return result

    def _condition_on_slices(self, cond_names, keep_names):
        """
        Produce a list of conditional Distribution slices, one per
        *non-zero* outcome of the conditioning variables.

        Matches the ``dit.Distribution.condition_on`` return format
        where the second element is a list of distributions (one per
        outcome in the marginal's ``.outcomes``).

        Parameters
        ----------
        cond_names : list of str
            Variable names to condition on.
        keep_names : list of str
            Variable names to keep in each conditional slice.

        Returns
        -------
        slices : list of Distribution
        """
        lin = self._linear_data()
        marginal_data = lin.sum(dim=keep_names)
        conditional_data = xr.where(marginal_data > 0, lin / marginal_data, 0.0)

        cond_coords = [self.data.coords[d].values for d in cond_names]
        slices = []
        for combo in itertools.product(*cond_coords):
            sel = dict(zip(cond_names, combo, strict=True))
            marg_p = float(marginal_data.sel(sel))
            if marg_p <= 0:
                continue
            sliced = conditional_data.sel(sel)
            if sliced.ndim == 0:
                sliced = sliced.expand_dims(keep_names)
            free = frozenset(keep_names)
            cd = Distribution(sliced, free_vars=free, given_vars=frozenset())
            cd._rv_names_set = self._rv_names_set
            slices.append(cd)

        return slices

    # ─────────────────────────────────────────────────────────────────────
    # Arithmetic
    # ─────────────────────────────────────────────────────────────────────

    def __mul__(self, other):
        """
        Multiply two distributions.

        Core operation enabling the chain rule and partial application:

        - ``p(X,Y) * p(Z|X,Y)  = p(X,Y,Z)``
        - ``p(X) * p(Y|X)      = p(X,Y)``
        - ``p(X) * p(Z|X,Y)    = p(X,Z|Y)``
        - ``scalar * p(X)       = scaled p(X)``

        Parameters
        ----------
        other : Distribution or float
            Another distribution to multiply, or a scalar for scaling.

        Returns
        -------
        result : Distribution
            The product distribution. For scalar multiplication, a scaled
            copy with the same free/given structure.

        Notes
        -----
        Distribution multiplication rules:

        - ``free_A`` and ``free_B`` must be disjoint
        - ``given_B`` must be a subset of ``all_A``
        - ``result_free = free_A | free_B``
        - ``result_given = (given_A | given_B) - result_free``
        """
        if isinstance(other, (int, float, np.number)):
            return Distribution(
                self.data * other,
                free_vars=self.free_vars,
                given_vars=self.given_vars,
                base=self.ops.base,
            )

        if not isinstance(other, Distribution):
            return NotImplemented

        # Validate
        free_overlap = self.free_vars & other.free_vars
        if free_overlap:
            raise ValueError(
                f"Cannot multiply: both have free variables {free_overlap}. "
                f"Did you mean to condition one on the other first?"
            )

        # Note: given vars of 'other' that are not provided by 'self'
        # simply remain as given vars in the result (partial application).
        # For example, p(X) * p(Z|X,Y) = p(X,Z|Y): Y stays given.

        # Work in linear space for the multiplication
        lin_self = self._linear_data()
        lin_other = other._linear_data()
        product_data = lin_self * lin_other

        result_free = self.free_vars | other.free_vars
        result_given = (self.given_vars | other.given_vars) - result_free

        # If either operand was log-based, convert result back
        base = self.ops.base
        if base != "linear":
            product_data = xr.DataArray(
                get_ops(base).log(product_data.values),
                dims=product_data.dims,
                coords=product_data.coords,
            )
            return Distribution(product_data, free_vars=result_free, given_vars=result_given, base=base)

        return Distribution(product_data, free_vars=result_free, given_vars=result_given)

    def __rmul__(self, other):
        """
        Right multiplication (for scalars).

        Parameters
        ----------
        other : int or float
            Scalar to multiply.

        Returns
        -------
        result : Distribution
            Scaled distribution, or NotImplemented if other is not a scalar.
        """
        if isinstance(other, (int, float, np.number)):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, other):
        """
        Divide two distributions.

        ``p(X,Y) / p(X)`` yields ``p(Y|X)`` -- division creates a
        conditional distribution.

        Parameters
        ----------
        other : Distribution or float
            Denominator distribution, or scalar for scaling.

        Returns
        -------
        result : Distribution
            The quotient distribution. The denominator's free variables
            become given variables in the result.
        """
        if isinstance(other, (int, float, np.number)):
            return Distribution(
                self.data / other,
                free_vars=self.free_vars,
                given_vars=self.given_vars,
                base=self.ops.base,
            )

        if not isinstance(other, Distribution):
            return NotImplemented

        if not other.free_vars <= self.free_vars:
            raise ValueError(
                f"Cannot divide: denominator has free vars {other.free_vars - self.free_vars} not in numerator"
            )

        lin_self = self._linear_data()
        lin_other = other._linear_data()
        quotient_data = xr.where(lin_other > 0, lin_self / lin_other, 0.0)

        new_free = self.free_vars - other.free_vars
        new_given = (self.given_vars | other.free_vars | other.given_vars) - new_free

        base = self.ops.base
        if base != "linear":
            quotient_data = xr.DataArray(
                get_ops(base).log(quotient_data.values),
                dims=quotient_data.dims,
                coords=quotient_data.coords,
            )
            return Distribution(quotient_data, free_vars=new_free, given_vars=new_given, base=base)

        return Distribution(quotient_data, free_vars=new_free, given_vars=new_given)

    def __add__(self, other):
        """
        Element-wise addition of distributions (for convex combinations).

        Both distributions must share the same sample space. The result
        is **not** automatically normalised.
        """
        if isinstance(other, (int, float, np.number)):
            if other == 0:
                return self.copy()
            return NotImplemented
        if not isinstance(other, Distribution):
            return NotImplemented
        new_data = self.data + other.data
        return Distribution(
            new_data,
            free_vars=self.free_vars,
            given_vars=self.given_vars,
            base=self.ops.base,
        )

    def __radd__(self, other):
        """Right-addition (supports ``sum(dists)`` starting from 0)."""
        if isinstance(other, (int, float, np.number)) and other == 0:
            return self.copy()
        return self.__add__(other)

    def __sub__(self, other):
        """Element-wise subtraction of distributions."""
        if not isinstance(other, Distribution):
            return NotImplemented
        new_data = self.data - other.data
        return Distribution(
            new_data,
            free_vars=self.free_vars,
            given_vars=self.given_vars,
            base=self.ops.base,
        )

    def __matmul__(self, other):
        """
        Cartesian product of two distributions (treated as independent).

        Combines outcomes via tuple concatenation and multiplies probabilities.
        """
        if not isinstance(other, Distribution):
            return NotImplemented
        from collections import defaultdict
        from itertools import product as iprod

        d2 = other.copy(base=self.get_base())
        dist = defaultdict(float)
        for (o1, p1), (o2, p2) in iprod(self.zipped(), d2.zipped()):
            combined = tuple(o1) + tuple(o2)
            dist[combined] += self.ops.mult(p1, p2)

        outcomes = sorted(dist.keys())
        pmf = [dist[o] for o in outcomes]
        return Distribution(outcomes, pmf, base=self.get_base())

    # ── Classmethods ──────────────────────────────────────────────────

    @classmethod
    def from_ndarray(cls, ndarray, base=None, prng=None):
        """
        Construct from a multi-dimensional numpy ndarray interpreted as a pmf.

        Each axis represents a random variable, and the index along that axis
        is the variable's value. For example, a (2, 3) array has two variables
        with alphabet sizes 2 and 3 respectively.

        Parameters
        ----------
        ndarray : np.ndarray
        base : str or float, optional
        prng : random state, optional
        """
        outcomes, pmf = zip(*np.ndenumerate(ndarray), strict=True)
        return cls(list(outcomes), list(pmf), base=base or "linear", prng=prng)

    @classmethod
    def from_rv_discrete(cls, ssrv, base=None, prng=None):
        """
        Construct from a ``scipy.stats.rv_discrete`` instance.

        Parameters
        ----------
        ssrv : scipy.stats.rv_discrete
            A frozen discrete random variable with ``.xk`` and ``.pk``
            attributes (as produced by ``rv_discrete(values=...)``).
        base : str or float, optional
            Probability base. Defaults to ``'linear'``.
        prng : random state, optional
        """
        outcomes = [(int(x),) for x in ssrv.xk]
        pmf = list(ssrv.pk)
        return cls(outcomes, pmf, base=base or "linear", prng=prng)

    # ─────────────────────────────────────────────────────────────────────
    # Information-theoretic convenience methods
    # ─────────────────────────────────────────────────────────────────────

    def entropy(self, base=2):
        """
        Compute the (conditional) entropy.

        For ``p(X,Y)`` returns ``H(X,Y)``.
        For ``p(X|Y)`` returns ``H(X|Y)`` computed as the average
        per-slice entropy: ``(1/|Y|) * sum_y H(X|Y=y)``.

        Note: the true conditional entropy ``H(X|Y) = sum_y p(y) H(X|Y=y)``
        requires knowledge of the marginal ``p(Y)``, which is not stored.
        Use a joint distribution and ``H(X,Y) - H(Y)`` for the exact value.

        Parameters
        ----------
        base : float, optional
            Logarithm base for the result (default: 2).

        Returns
        -------
        h : float
            The (conditional) entropy.
        """
        lin = self._linear_data()
        log_b = np.log(base)

        if self._is_unconditional():
            p = lin.values.ravel()
            p = p[p > 0]
            return float(-np.sum(p * np.log(p)) / log_b)
        else:
            # Average per-slice entropy over given variable assignments
            given_dims = list(self.given_vars)
            coord_vals = [lin.coords[d].values for d in given_dims]
            total_h = 0.0
            n_slices = 0
            for combo in itertools.product(*coord_vals):
                sel = {d: v for d, v in zip(given_dims, combo, strict=True)}
                slc = lin.sel(sel).values.ravel()
                slc = slc[slc > 0]
                if len(slc) > 0:
                    total_h += float(-np.sum(slc * np.log(slc)) / log_b)
                    n_slices += 1
            if n_slices == 0:
                return 0.0
            return total_h / n_slices

    def mutual_information(self, var_x, var_y, base=2):
        """
        Compute the mutual information ``I(X;Y)``.

        Only valid for joint distributions.

        Parameters
        ----------
        var_x : str or set of str
            Variable(s) for the first argument of I(X;Y).
        var_y : str or set of str
            Variable(s) for the second argument of I(X;Y).
        base : float, optional
            Logarithm base for the result (default: 2).

        Returns
        -------
        mi : float
            The mutual information I(X;Y).
        """
        if self.is_conditional():
            raise ValueError("Mutual information requires an unconditional distribution")

        var_x = {var_x} if isinstance(var_x, str) else set(var_x)
        var_y = {var_y} if isinstance(var_y, str) else set(var_y)

        h_x = self.marginal(*var_x).entropy(base)
        h_y = self.marginal(*var_y).entropy(base)
        h_xy = self.marginal(*(var_x | var_y)).entropy(base)

        return h_x + h_y - h_xy

    # ─────────────────────────────────────────────────────────────────────
    # Selection / indexing
    # ─────────────────────────────────────────────────────────────────────

    def sel(self, **kwargs):
        """
        Fix variables to specific values (label-based selection).

        Parameters
        ----------
        **kwargs
            Variable-name to value mappings.

        Returns
        -------
        result : Distribution or float
            If all dimensions are selected, returns a float (probability or
            log probability, depending on the distribution's base). Otherwise
            returns a reduced Distribution.

        Examples
        --------
        >>> p_xyz.sel(Y='0')        # p(X,Z) at Y=0 (un-normalised slice)
        >>> p_xyz.sel(X='0', Y='1') # p(Z) at X=0,Y=1
        """
        new_data = self.data.sel(kwargs)
        if new_data.ndim == 0:
            return float(new_data)
        dropped = frozenset(kwargs.keys())
        new_free = self.free_vars - dropped
        new_given = self.given_vars - dropped
        return Distribution(new_data, free_vars=new_free, given_vars=new_given, base=self.ops.base)

    def __getitem__(self, key):
        """
        Index by dict, outcome tuple, or string.

        Parameters
        ----------
        key : dict, tuple, or str
            If a dict, performs label-based selection via :meth:`sel`.
            If a string with length matching dims, each character is one
            variable's value.
            If a tuple with the same length as :attr:`dims`, looks up the
            probability of that outcome.

        Returns
        -------
        result : float or Distribution

        Raises
        ------
        InvalidOutcome
            If the outcome is not in the sample space.
        """
        from .exceptions import InvalidOutcome

        if isinstance(key, dict):
            return self.sel(**key)
        # String keys: treat each character as a coordinate value
        if isinstance(key, str):
            if len(key) == len(self.data.dims):
                key = tuple(key)
            elif len(self.data.dims) == 1:
                try:
                    return float(self.data.sel({self.data.dims[0]: key}))
                except KeyError as exc:
                    raise InvalidOutcome(msg=f"Outcome {key!r} is not in the sample space.") from exc
            else:
                raise InvalidOutcome(msg=f"Outcome {key!r} has wrong length for {len(self.data.dims)} dims.")
        # Scalar key for 1-D distributions
        if not isinstance(key, tuple) and len(self.data.dims) == 1:
            try:
                return float(self.data.sel({self.data.dims[0]: key}))
            except KeyError as exc:
                raise InvalidOutcome(msg=f"Outcome {key!r} is not in the sample space.") from exc
        if isinstance(key, tuple) and len(key) == len(self.data.dims):
            sel = dict(zip(self.data.dims, key, strict=True))
            try:
                return float(self.data.sel(sel))
            except KeyError:
                pass

            # Coalesced distributions have string coords; key elements may
            # be tuples from Distribution-style outcome construction.
            def _serialize(v):
                if isinstance(v, tuple):
                    return ",".join(str(x) for x in v)
                return v

            sel2 = {d: _serialize(v) for d, v in zip(self.data.dims, key, strict=True)}
            try:
                return float(self.data.sel(sel2))
            except KeyError as exc:
                raise InvalidOutcome(msg=f"Outcome {key!r} is not in the sample space.") from exc
        raise InvalidOutcome(msg=f"Invalid outcome: {key!r}")

    def __delitem__(self, outcome):
        """Set the probability of *outcome* to zero."""
        self[outcome] = 0.0

    def __setitem__(self, key, value):
        """
        Set the probability of an outcome.

        Parameters
        ----------
        key : dict, tuple, or str
            If a dict, keys are dimension names mapped to coordinate values.
            If a string with length matching dims, each character is one
            variable's value.
            If a tuple with length equal to ``len(self.dims)``, elements
            correspond to dimensions in order.
        value : float
            The probability value to set.
        """
        if isinstance(key, dict):
            self.data.loc[key] = value
        elif isinstance(key, str) and len(key) == len(self.data.dims):
            sel = dict(zip(self.data.dims, tuple(key), strict=True))
            self.data.loc[sel] = value
        elif isinstance(key, str) and len(self.data.dims) == 1:
            self.data.loc[{self.data.dims[0]: key}] = value
        elif isinstance(key, tuple) and len(key) == len(self.data.dims):
            sel = dict(zip(self.data.dims, key, strict=True))
            self.data.loc[sel] = value
        elif not isinstance(key, (tuple, dict, str)) and len(self.data.dims) == 1:
            self.data.loc[{self.data.dims[0]: key}] = value
        else:
            raise KeyError(f"Invalid key: {key!r}")

    # ─────────────────────────────────────────────────────────────────────
    # Copy and conversion
    # ─────────────────────────────────────────────────────────────────────

    def copy(self, base=None):
        """
        Return a deep copy of this distribution.

        Parameters
        ----------
        base : str or float, optional
            If given, the copy will be converted to this base.

        Returns
        -------
        c : Distribution
        """
        c = Distribution(
            self.data.copy(deep=True),
            free_vars=self.free_vars,
            given_vars=self.given_vars,
            base=self.ops.base,
        )
        c._sparse = self._sparse
        c._rv_names_set = self._rv_names_set
        c._meta = dict(self._meta)
        c.prng = self.prng
        if base is not None:
            c._set_base_inplace(base)
        return c

    def to_distribution(self):
        """
        Deprecated. Returns self since Distribution is now the sole distribution class.
        """
        return self.copy()

    def to_numpy(self):
        """
        Return the underlying data as a numpy array.

        Returns
        -------
        np.ndarray
            Copy of the probability values (linear or log, depending on base).
        """
        return self.data.values.copy()

    @property
    def DataArray(self):
        """
        Return the underlying xr.DataArray (read-only view).

        Returns
        -------
        xr.DataArray
            The probability array. Modifying it may affect this distribution.
        """
        return self.data

    # ─────────────────────────────────────────────────────────────────────
    # Comparison
    # ─────────────────────────────────────────────────────────────────────

    def __eq__(self, other):
        if not isinstance(other, Distribution):
            return NotImplemented
        return (
            self.free_vars == other.free_vars and self.given_vars == other.given_vars and self.data.equals(other.data)
        )

    def __hash__(self):
        return hash((self.free_vars, self.given_vars, tuple(self.outcomes), tuple(float(p) for p in self.pmf)))

    def is_approx_equal(self, other, atol=1e-9, rtol=None):
        """
        Check approximate equality of two distributions.

        Compares by sample space and per-outcome probabilities, ignoring
        dimension names.  This matches the old ``dit.Distribution`` behavior.

        Parameters
        ----------
        other : Distribution
            Distribution to compare against.
        atol : float, optional
            Absolute tolerance for value comparison (default: 1e-9).
        rtol : float, optional
            Ignored (kept for signature compatibility).

        Returns
        -------
        eq : bool
        """
        if not isinstance(other, Distribution):
            return False

        # If both have named dimensions, align by name (order-independent)
        if self._rv_names_set and other._rv_names_set:
            if set(self.dims) != set(other.dims):
                return False
            for dim in self.dims:
                s_alpha = tuple(self.data.coords[dim].values)
                o_alpha = tuple(other.data.coords[dim].values)
                if len(s_alpha) != len(o_alpha):
                    return False
            s_lin = self._linear_data()
            o_lin = other._linear_data()
            o_aligned = o_lin.transpose(*self.dims)
            return bool(np.allclose(s_lin.values, o_aligned.values, atol=atol))

        if self.alphabet != other.alphabet:
            return False
        return all(np.isclose(self[outcome], other[outcome], atol=atol) for outcome in self.outcomes)

    def normalize(self):
        """
        Normalise the distribution in-place.

        For a joint distribution, divides by the total sum.
        For a conditional, normalises each conditional slice.

        Returns
        -------
        None
        """
        lin = self._linear_data()

        if self._is_unconditional():
            total = float(lin.sum())
            if total > 0:
                lin = lin / total
        else:
            sums = lin.sum(dim=list(self.free_vars))
            lin = xr.where(sums > 0, lin / sums, 0.0)

        if self.is_log():
            self.data = xr.DataArray(
                self.ops.log(lin.values),
                dims=self.data.dims,
                coords=self.data.coords,
            )
        else:
            self.data = lin

    def sample_space(self):
        """
        Iterator over all outcomes in the sample space.

        Yields
        ------
        outcome : tuple
        """
        dims = list(self.data.dims)
        coord_vals = [self.data.coords[d].values for d in dims]
        for combo in itertools.product(*coord_vals):
            yield tuple(combo)

    def event_probability(self, event):
        """
        Compute the probability of an event (subset of outcomes).

        Parameters
        ----------
        event : iterable of tuples
            Outcomes in the event.

        Returns
        -------
        p : float
        """
        dims = list(self.data.dims)
        lin = self._linear_data()
        total = 0.0
        for outcome in event:
            if not isinstance(outcome, tuple):
                outcome = (outcome,)
            sel = {d: v for d, v in zip(dims, outcome, strict=True)}
            total += float(lin.sel(sel))
        return total
