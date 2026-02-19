"""
Shared backend resolution utilities for optimization.

Provides :func:`_get_base_class` and :func:`_make_backend_subclass` which
allow any ``BaseAuxVarOptimizer`` subclass hierarchy to transparently switch
between the NumPy/SciPy, JAX, and PyTorch optimization backends.

The recommended pattern is a **Mixin + composed class** approach:

1. Extract all problem-specific logic (``__init__``, ``optimize``,
   ``_objective``, …) into a **Mixin** class that inherits from nothing
   (or ``object``).  The mixin's ``__init__`` should call
   ``super().__init__(…)`` so that the call chain reaches the backend base.
2. Create a backward-compatible composed class::

       class MyOptimizer(MyMixin, BaseAuxVarOptimizer): ...

3. :func:`_make_backend_subclass` detects the mixins in the MRO and
   creates ``type(name, (MyMixin, BackendBase), leaf_attrs)``.  Because
   ``super()`` inside the mixin now resolves to ``BackendBase``, the
   backend's ``__init__`` / ``optimize`` is called, setting up arrays
   in the correct format from the start.
"""

from ..algorithms import BaseAuxVarOptimizer


__all__ = (
    '_get_base_class',
    '_make_backend_subclass',
)


# ── Backend resolution ───────────────────────────────────────────────────

def _get_base_class(backend='numpy'):
    """
    Return the appropriate ``BaseAuxVarOptimizer`` class for *backend*.

    Parameters
    ----------
    backend : str
        One of ``'numpy'``, ``'jax'``, ``'torch'``.

    Returns
    -------
    cls : type
        The base auxiliary-variable optimizer class.

    Raises
    ------
    ValueError
        If *backend* is not recognised.
    """
    if backend == 'numpy':
        return BaseAuxVarOptimizer
    elif backend == 'jax':
        from ..algorithms.optimization_jax import BaseAuxVarJaxOptimizer
        return BaseAuxVarJaxOptimizer
    elif backend == 'torch':
        from ..algorithms.optimization_torch import BaseAuxVarTorchOptimizer
        return BaseAuxVarTorchOptimizer
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Choose from 'numpy', 'jax', 'torch'."
        )


_backend_class_cache = {}

_EXCLUDE_ATTRS = frozenset({'__dict__', '__weakref__', '_abc_impl'})


def _is_mixin(klass):
    """Return True if *klass* looks like a backend-agnostic mixin class."""
    return klass.__name__.endswith('Mixin')


def _make_backend_subclass(cls, backend):
    """
    Return a version of *cls* whose optimizer base uses *backend*.

    For ``backend='numpy'`` this is a no-op (returns *cls* unchanged).

    When *cls* follows the Mixin pattern (one or more ``*Mixin`` classes
    in its MRO), this function collects the mixins and the concrete
    (non-abstract) attributes from the leaf class hierarchy, then creates::

        type(cls.__name__, (Mixin1, Mixin2, …, BackendBase), leaf_attrs)

    ``super()`` calls inside the mixins resolve to *BackendBase*, so
    the backend's ``__init__`` / ``optimize`` is used transparently.

    For classes *without* mixins a legacy strategy is used: the backend's
    methods are injected into a subclass of *cls*.

    Parameters
    ----------
    cls : type
        A concrete subclass of ``BaseAuxVarOptimizer``.
    backend : str
        One of ``'numpy'``, ``'jax'``, ``'torch'``.

    Returns
    -------
    new_cls : type
        The optimizer class with the requested backend.
    """
    if backend == 'numpy':
        return cls

    cache_key = (cls, backend)
    if cache_key in _backend_class_cache:
        return _backend_class_cache[cache_key]

    Base = _get_base_class(backend)

    # Partition cls.__mro__ into mixins and non-mixin classes,
    # stopping at BaseAuxVarOptimizer (which the backend Base replaces).
    mixins = []
    non_mixin_classes = []
    for klass in cls.__mro__:
        if klass is BaseAuxVarOptimizer or klass is object:
            break
        if _is_mixin(klass):
            mixins.append(klass)
        else:
            non_mixin_classes.append(klass)

    if mixins:
        new_cls = _build_mixin_class(cls, mixins, non_mixin_classes, Base)
    else:
        new_cls = _build_legacy_class(cls, Base)

    _backend_class_cache[cache_key] = new_cls
    return new_cls


def _build_mixin_class(cls, mixins, non_mixin_classes, Base):
    """
    Build a backend-switched class using the mixin pattern.

    Collects concrete (non-abstract) attributes from every non-mixin class
    in *non_mixin_classes* (which are the leaf class and its composed
    intermediaries) and creates a new class inheriting from the mixins and
    *Base*.  The mixins provide ``__init__``/``optimize``/etc. whose
    ``super()`` calls resolve to *Base*.

    Parameters
    ----------
    cls : type
        The original concrete class.
    mixins : list[type]
        Mixin classes found in *cls.__mro__*, in MRO order.
    non_mixin_classes : list[type]
        Non-mixin classes from *cls.__mro__* up to ``BaseAuxVarOptimizer``,
        in MRO order (leaf first).
    Base : type
        The backend optimizer base class.
    """
    # Iterate base-to-leaf so that leaf attrs override base attrs.
    attrs = {}
    for klass in reversed(non_mixin_classes):
        for name, value in klass.__dict__.items():
            if name in _EXCLUDE_ATTRS:
                continue
            if getattr(value, '__isabstractmethod__', False):
                continue
            attrs[name] = value

    # The mixin provides __init__, so don't carry over the composed
    # class's __init__ (which would contain a super() referencing the
    # old class and fail with TypeError).
    attrs.pop('__init__', None)

    new_cls = type(cls.__name__, tuple(mixins) + (Base,), attrs)
    return new_cls


def _build_legacy_class(cls, Base):
    """
    Build a backend-switched class for classes *without* the mixin pattern.

    Injects the backend's methods into a subclass of *cls*.  ``super()``
    calls inside *cls* resolve correctly because the new class inherits
    from *cls*.  This is the pre-mixin approach kept as a fallback.

    Parameters
    ----------
    cls : type
        The original concrete class.
    Base : type
        The backend optimizer base class.
    """
    override_attrs = {}
    for klass in reversed(Base.__mro__):
        if klass is object:
            continue
        for name, value in klass.__dict__.items():
            if name in _EXCLUDE_ATTRS:
                continue
            if getattr(value, '__isabstractmethod__', False):
                continue
            override_attrs[name] = value

    override_attrs.pop('__init__', None)

    new_cls = type(cls.__name__, (cls,), override_attrs)
    return new_cls
