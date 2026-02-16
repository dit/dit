"""
An xarray-backed distribution class for discrete random variables.

This module provides ``XRDistribution``, a distribution class built on top of
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
>>> from dit.xrdist import XRDistribution
>>>
>>> d = dit.example_dists.Xor()
>>> p_xyz = XRDistribution.from_distribution(d, ['X', 'Y', 'Z'])
>>>
>>> p_xy = p_xyz.marginal('X', 'Y')       # p(X,Y)
>>> p_z_given_xy = p_xyz.condition_on('X', 'Y')  # p(Z|X,Y)
>>> p_xyz_rebuilt = p_xy * p_z_given_xy    # p(X,Y) * p(Z|X,Y) = p(X,Y,Z)
"""

import itertools

import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

from .math.ops import get_ops, LinearOperations


__all__ = (
    'XRDistribution',
)


def _check_xarray():
    """Raise an error if xarray is not available."""
    if not XARRAY_AVAILABLE:
        raise ImportError(
            "xarray is required for XRDistribution. "
            "Install with: pip install xarray"
        )


class XRDistribution:
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

    def __init__(self, data, pmf=None, rv_names=None, free_vars=None,
                 given_vars=None, base='linear'):
        """
        Initialize an XRDistribution.

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
        base : str or float
            The probability base. ``'linear'`` (default) for raw
            probabilities, ``2``, ``'e'``, or any positive float for log
            probabilities.

        Examples
        --------
        From outcomes and pmf (like ``dit.Distribution``):

        >>> xrd = XRDistribution(['00','01','10','11'],
        ...                      [.25, .25, .25, .25],
        ...                      rv_names=['X', 'Y'])

        From a dict:

        >>> xrd = XRDistribution({'00': .5, '11': .5}, rv_names=['X', 'Y'])

        From a DataArray (original API):

        >>> xrd = XRDistribution(my_dataarray, free_vars={'X', 'Y'})
        """
        _check_xarray()

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
                    raise ValueError(
                        "pmf is required when data is a sequence of outcomes"
                    )
                pmf = list(pmf)

            if len(outcomes) == 0:
                raise ValueError("outcomes must be non-empty")
            if len(outcomes) != len(pmf):
                raise ValueError(
                    f"outcomes and pmf must have the same length, "
                    f"got {len(outcomes)} and {len(pmf)}"
                )

            n = len(outcomes[0])
            if rv_names is None:
                rv_names = [f'X{i}' for i in range(n)]
            if len(rv_names) != n:
                raise ValueError(
                    f"Expected {n} rv_names, got {len(rv_names)}"
                )

            # Infer sorted alphabet per variable from the outcomes
            alphabets = [sorted(set(o[i] for o in outcomes)) for i in range(n)]
            coords = {name: alpha for name, alpha in zip(rv_names, alphabets)}

            shape = tuple(len(a) for a in alphabets)
            arr = np.zeros(shape)
            for outcome, p in zip(outcomes, pmf):
                idx = tuple(alphabets[i].index(outcome[i]) for i in range(n))
                arr[idx] = p

            da = xr.DataArray(arr, dims=rv_names, coords=coords)

            # Default: all variables are free when constructing from outcomes
            if free_vars is None and given_vars is None:
                free_vars = set(rv_names)

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
            raise ValueError(
                f"free_vars and given_vars must be disjoint. "
                f"Overlap: {self.free_vars & self.given_vars}"
            )

    @classmethod
    def from_distribution(cls, dist, rv_names=None):
        """
        Create an XRDistribution from an existing dit Distribution.

        Parameters
        ----------
        dist : dit.Distribution
            The source distribution.
        rv_names : list of str, optional
            Names for each random variable. If None, uses the
            distribution's rv_names, or defaults to ``'X0'``, ``'X1'``, etc.

        Returns
        -------
        xrd : XRDistribution
        """
        _check_xarray()

        dist = dist.copy(base='linear')
        dist.make_dense()

        if rv_names is None:
            rv_names = dist.get_rv_names()
            if rv_names is None:
                rv_names = [f'X{i}' for i in range(dist.outcome_length())]
            else:
                rv_names = list(rv_names)

        n = dist.outcome_length()
        if len(rv_names) != n:
            raise ValueError(f"Expected {n} variable names, got {len(rv_names)}")

        coords = {}
        for i, name in enumerate(rv_names):
            coords[name] = list(dist.alphabet[i])

        shape = tuple(len(v) for v in coords.values())
        arr = np.zeros(shape)

        for outcome, prob in zip(dist.outcomes, dist.pmf):
            idx = tuple(
                coords[name].index(outcome[i])
                for i, name in enumerate(rv_names)
            )
            arr[idx] = prob

        data = xr.DataArray(arr, dims=rv_names, coords=coords)
        return cls(data, free_vars=set(rv_names), given_vars=set())

    @classmethod
    def from_array(cls, arr, dim_names, alphabets,
                   free_vars=None, given_vars=None, base='linear'):
        """
        Create an XRDistribution from a numpy array.

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
        xrd : XRDistribution
        """
        _check_xarray()

        coords = {n: list(a) for n, a in zip(dim_names, alphabets)}
        data = xr.DataArray(arr, dims=dim_names, coords=coords)
        return cls(data, free_vars=free_vars, given_vars=given_vars, base=base)

    @classmethod
    def from_factors(cls, marginal, conditional):
        """
        Build a joint distribution from a marginal and a conditional.

        ``p(X,Y) = p(X) * p(Y|X)``

        Parameters
        ----------
        marginal : XRDistribution
            The marginal distribution, e.g. ``p(X)``.
        conditional : XRDistribution
            The conditional distribution, e.g. ``p(Y|X)``.

        Returns
        -------
        joint : XRDistribution
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
        return tuple(
            tuple(self.data.coords[d].values) for d in self.data.dims
        )

    @property
    def outcomes(self):
        """
        Tuple of outcomes with non-zero probability, in lexicographic order.

        Each outcome is a tuple whose elements correspond to the
        dimensions in :attr:`dims` order. This mirrors
        ``dit.Distribution.outcomes``.
        """
        dims = list(self.data.dims)
        coord_vals = [self.data.coords[d].values for d in dims]
        arr = self._linear_data()
        outs = []
        for combo in itertools.product(*coord_vals):
            sel = {d: v for d, v in zip(dims, combo)}
            p = float(arr.sel(sel))
            if p > 0:
                outs.append(tuple(combo))
        return tuple(outs)

    @property
    def pmf(self):
        """
        1-D numpy array of probabilities corresponding to :attr:`outcomes`.

        This mirrors ``dit.Distribution.pmf``.
        """
        dims = list(self.data.dims)
        arr = self._linear_data()
        probs = []
        for o in self.outcomes:
            sel = {d: v for d, v in zip(dims, o)}
            probs.append(float(arr.sel(sel)))
        return np.array(probs)

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
        Return the variable names as a tuple.

        Returns
        -------
        names : tuple of str
        """
        return tuple(self.data.dims)

    def set_rv_names(self, rv_names):
        """
        Rename the dimensions (random variables).

        Parameters
        ----------
        rv_names : list of str
            New names, one per dimension.
        """
        if len(rv_names) != len(self.data.dims):
            raise ValueError(
                f"Expected {len(self.data.dims)} names, got {len(rv_names)}"
            )
        rename_map = dict(zip(self.data.dims, rv_names))
        self.data = self.data.rename(rename_map)

        old_free = self.free_vars
        old_given = self.given_vars
        self.free_vars = frozenset(rename_map.get(v, v) for v in old_free)
        self.given_vars = frozenset(rename_map.get(v, v) for v in old_given)

    def is_joint(self):
        """True if this is a joint distribution (no conditioned variables)."""
        return len(self.given_vars) == 0

    def is_conditional(self):
        """True if this is a conditional distribution."""
        return len(self.given_vars) > 0

    def is_log(self):
        """True if the distribution stores log probabilities."""
        return self.ops.base != 'linear'

    def is_numerical(self):
        """True (always numerical)."""
        return True

    def make_dense(self):
        """
        No-op -- DataArray is always dense.

        Provided for API compatibility with :class:`dit.Distribution`.

        Returns
        -------
        int
            Always returns 0.
        """
        return 0

    def make_sparse(self, trim=True):
        """
        No-op -- DataArray is always dense.

        Provided for API compatibility with :class:`dit.Distribution`.
        Sparse representation is not supported.

        Parameters
        ----------
        trim : bool, optional
            Ignored. Kept for API compatibility.

        Returns
        -------
        int
            Always returns 0.
        """
        return 0

    def zipped(self, mode='pmf'):
        """
        Iterator over ``(outcome, probability)`` tuples.

        Parameters
        ----------
        mode : str
            ``'pmf'`` to iterate over non-zero outcomes (default),
            ``'atoms'`` to iterate over the full sample space.

        Yields
        ------
        outcome : tuple
        probability : float
        """
        dims = list(self.data.dims)
        coord_vals = [self.data.coords[d].values for d in dims]
        arr = self._linear_data()
        for combo in itertools.product(*coord_vals):
            sel = {d: v for d, v in zip(dims, combo)}
            p = float(arr.sel(sel))
            if mode == 'atoms' or p > 0:
                yield tuple(combo), p

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
        old_ops = self.ops
        new_ops = get_ops(base)

        if old_ops.base == new_ops.base:
            return

        values = self.data.values.copy()

        old_base = old_ops.base
        new_base = new_ops.base

        if old_base == 'linear' and new_base != 'linear':
            values = new_ops.log(values)
        elif old_base != 'linear' and new_base == 'linear':
            values = old_ops.exp(values)
        elif old_base != 'linear' and new_base != 'linear':
            # log_a -> linear -> log_b
            values = old_ops.exp(values)
            values = new_ops.log(values)

        self.data = xr.DataArray(values, dims=self.data.dims,
                                 coords=self.data.coords)
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
        free = ','.join(ordered_free)
        if ordered_given:
            given = ','.join(ordered_given)
            return f'p({free}|{given})'
        return f'p({free})'

    def __repr__(self):
        return f'XRDistribution {self._notation()}\n{self.data}'

    def __str__(self):
        return self.__repr__()

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

        if self.is_joint():
            total = float(arr.sum())
            if not np.isclose(total, 1.0, atol=atol):
                raise ValueError(
                    f"Joint distribution sums to {total}, expected 1.0"
                )
        else:
            sums = arr.sum(dim=list(self.free_vars))
            # Only check values where the given vars have positive marginal
            vals = sums.values.ravel()
            nonzero = vals[vals > atol]
            if len(nonzero) > 0 and not np.allclose(nonzero, 1.0, atol=atol):
                raise ValueError(
                    f"Conditional distribution does not normalise properly. "
                    f"Sums over free vars:\n{sums}"
                )
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Core probability operations
    # ─────────────────────────────────────────────────────────────────────

    def marginal(self, *keep_vars):
        """
        Marginalise to keep only the specified free variables.

        Given (conditioned) variables are always kept.

        Parameters
        ----------
        *keep_vars : str
            The free variable names to keep.

        Returns
        -------
        result : XRDistribution

        Examples
        --------
        >>> p_xy = XRDistribution(...)  # p(X, Y)
        >>> p_x = p_xy.marginal('X')   # p(X)
        """
        keep = frozenset(keep_vars)
        invalid = keep - self.free_vars
        if invalid:
            raise ValueError(
                f"Cannot keep {invalid}: not free variables. "
                f"Free: {self.free_vars}, given: {self.given_vars}"
            )

        sum_over = list(self.free_vars - keep)
        if not sum_over:
            return self.copy()

        if self.is_log():
            # Convert to linear, sum, convert back
            lin = self._linear_data()
            new_data = lin.sum(dim=sum_over)
            new_ops = self.ops
            new_data = xr.DataArray(
                new_ops.log(new_data.values),
                dims=new_data.dims, coords=new_data.coords,
            )
            return XRDistribution(new_data, free_vars=keep,
                                  given_vars=self.given_vars,
                                  base=self.ops.base)
        else:
            new_data = self.data.sum(dim=sum_over)
            return XRDistribution(new_data, free_vars=keep,
                                  given_vars=self.given_vars)

    def marginalize(self, *drop_vars):
        """
        Marginalise out (remove) the specified free variables.

        Parameters
        ----------
        *drop_vars : str
            The free variable names to remove.

        Returns
        -------
        result : XRDistribution
        """
        drop = frozenset(drop_vars)
        invalid = drop - self.free_vars
        if invalid:
            raise ValueError(
                f"Cannot drop {invalid}: not free variables. "
                f"Free: {self.free_vars}"
            )
        keep = self.free_vars - drop
        return self.marginal(*keep)

    def condition_on(self, *cond_vars):
        """
        Condition on the specified free variables.

        Given ``p(X,Y,Z)``, conditioning on Z produces ``p(X,Y|Z)``.

        Parameters
        ----------
        *cond_vars : str
            The free variable names to condition on.

        Returns
        -------
        result : XRDistribution
            A single conditional distribution object.

        Examples
        --------
        >>> p_xyz.condition_on('Z')   # p(X,Y|Z)
        >>> p_xyz.condition_on('X', 'Y')  # p(Z|X,Y)
        """
        cond = frozenset(cond_vars)
        invalid = cond - self.free_vars
        if invalid:
            raise ValueError(
                f"Cannot condition on {invalid}: not free variables. "
                f"Free: {self.free_vars}"
            )

        new_free = self.free_vars - cond
        if not new_free:
            raise ValueError("Cannot condition on all free variables")

        # Work in linear space
        lin = self._linear_data()
        marginal_data = lin.sum(dim=list(new_free))
        conditional_data = xr.where(marginal_data > 0,
                                    lin / marginal_data,
                                    0.0)

        new_given = self.given_vars | cond

        if self.is_log():
            conditional_data = xr.DataArray(
                self.ops.log(conditional_data.values),
                dims=conditional_data.dims,
                coords=conditional_data.coords,
            )
            return XRDistribution(conditional_data, free_vars=new_free,
                                  given_vars=new_given, base=self.ops.base)

        return XRDistribution(conditional_data, free_vars=new_free,
                              given_vars=new_given)

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
        other : XRDistribution or float
            Another distribution to multiply, or a scalar for scaling.

        Returns
        -------
        result : XRDistribution
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
            return XRDistribution(
                self.data * other,
                free_vars=self.free_vars,
                given_vars=self.given_vars,
                base=self.ops.base,
            )

        if not isinstance(other, XRDistribution):
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
        if base != 'linear':
            product_data = xr.DataArray(
                get_ops(base).log(product_data.values),
                dims=product_data.dims, coords=product_data.coords,
            )
            return XRDistribution(product_data, free_vars=result_free,
                                  given_vars=result_given, base=base)

        return XRDistribution(product_data, free_vars=result_free,
                              given_vars=result_given)

    def __rmul__(self, other):
        """
        Right multiplication (for scalars).

        Parameters
        ----------
        other : int or float
            Scalar to multiply.

        Returns
        -------
        result : XRDistribution
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
        other : XRDistribution or float
            Denominator distribution, or scalar for scaling.

        Returns
        -------
        result : XRDistribution
            The quotient distribution. The denominator's free variables
            become given variables in the result.
        """
        if isinstance(other, (int, float, np.number)):
            return XRDistribution(
                self.data / other,
                free_vars=self.free_vars,
                given_vars=self.given_vars,
                base=self.ops.base,
            )

        if not isinstance(other, XRDistribution):
            return NotImplemented

        if not other.free_vars <= self.free_vars:
            raise ValueError(
                f"Cannot divide: denominator has free vars "
                f"{other.free_vars - self.free_vars} not in numerator"
            )

        lin_self = self._linear_data()
        lin_other = other._linear_data()
        quotient_data = xr.where(lin_other > 0,
                                 lin_self / lin_other,
                                 0.0)

        new_free = self.free_vars - other.free_vars
        new_given = (self.given_vars | other.free_vars | other.given_vars) - new_free

        base = self.ops.base
        if base != 'linear':
            quotient_data = xr.DataArray(
                get_ops(base).log(quotient_data.values),
                dims=quotient_data.dims, coords=quotient_data.coords,
            )
            return XRDistribution(quotient_data, free_vars=new_free,
                                  given_vars=new_given, base=base)

        return XRDistribution(quotient_data, free_vars=new_free,
                              given_vars=new_given)

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

        if self.is_joint():
            p = lin.values.ravel()
            p = p[p > 0]
            return float(-np.sum(p * np.log(p)) / log_b)
        else:
            # Average per-slice entropy over given variable assignments
            given_dims = list(self.given_vars)
            free_dims = list(self.free_vars)
            coord_vals = [lin.coords[d].values for d in given_dims]
            total_h = 0.0
            n_slices = 0
            for combo in itertools.product(*coord_vals):
                sel = {d: v for d, v in zip(given_dims, combo)}
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
        if not self.is_joint():
            raise ValueError("Mutual information requires a joint distribution")

        if isinstance(var_x, str):
            var_x = {var_x}
        else:
            var_x = set(var_x)
        if isinstance(var_y, str):
            var_y = {var_y}
        else:
            var_y = set(var_y)

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
        result : XRDistribution or float
            If all dimensions are selected, returns a float (probability or
            log probability, depending on the distribution's base). Otherwise
            returns a reduced XRDistribution.

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
        return XRDistribution(new_data, free_vars=new_free,
                              given_vars=new_given, base=self.ops.base)

    def __getitem__(self, key):
        """
        Index by dict or by outcome tuple.

        Parameters
        ----------
        key : dict or tuple
            If a dict, performs label-based selection via :meth:`sel`.
            If a tuple with the same length as :attr:`dims`, looks up the
            value (probability or log probability) of that outcome.

        Returns
        -------
        result : float or XRDistribution
            The value at the outcome, or a reduced distribution for dict
            selection. For tuple keys, returns probability or log probability
            depending on the distribution's base.
        """
        if isinstance(key, dict):
            return self.sel(**key)
        if isinstance(key, tuple) and len(key) == len(self.data.dims):
            sel = dict(zip(self.data.dims, key))
            return float(self.data.sel(sel))
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
        c : XRDistribution
        """
        c = XRDistribution(
            self.data.copy(deep=True),
            free_vars=self.free_vars,
            given_vars=self.given_vars,
            base=self.ops.base,
        )
        if base is not None:
            c.set_base(base)
        return c

    def to_distribution(self):
        """
        Convert to a ``dit.Distribution``.

        Only works for joint distributions (no conditioned variables).

        Returns
        -------
        dist : dit.Distribution

        Raises
        ------
        ValueError
            If the distribution is conditional.
        """
        from .npdist import Distribution

        if not self.is_joint():
            raise ValueError(
                "Can only convert joint distributions to dit.Distribution. "
                f"This is {self._notation()}"
            )

        lin = self._linear_data()
        dims = list(lin.dims)

        outcomes = []
        pmf = []
        coord_values = [lin.coords[d].values for d in dims]
        for combo in itertools.product(*coord_values):
            sel = {d: v for d, v in zip(dims, combo)}
            p = float(lin.sel(sel))
            if p > 0:
                outcomes.append(tuple(combo))
                pmf.append(p)

        d = Distribution(outcomes, pmf)
        d.set_rv_names(dims)
        return d

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
        if not isinstance(other, XRDistribution):
            return NotImplemented
        return (self.free_vars == other.free_vars
                and self.given_vars == other.given_vars
                and self.data.equals(other.data))

    def is_approx_equal(self, other, atol=1e-9):
        """
        Check approximate equality of two distributions.

        Parameters
        ----------
        other : XRDistribution
            Distribution to compare against.
        atol : float, optional
            Absolute tolerance for value comparison (default: 1e-9).

        Returns
        -------
        eq : bool
            True if free_vars, given_vars match and values are close.
        """
        if not isinstance(other, XRDistribution):
            return False
        if self.free_vars != other.free_vars:
            return False
        if self.given_vars != other.given_vars:
            return False
        return bool(np.allclose(self.data.values, other.data.values, atol=atol))

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

        if self.is_joint():
            total = float(lin.sum())
            if total > 0:
                lin = lin / total
        else:
            sums = lin.sum(dim=list(self.free_vars))
            lin = xr.where(sums > 0, lin / sums, 0.0)

        if self.is_log():
            self.data = xr.DataArray(
                self.ops.log(lin.values),
                dims=self.data.dims, coords=self.data.coords,
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
            sel = {d: v for d, v in zip(dims, outcome)}
            total += float(lin.sel(sel))
        return total
