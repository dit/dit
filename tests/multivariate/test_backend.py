"""
Tests for dit.multivariate._backend: backend resolution and the construction of
backend-specific optimizer classes.

The optional jax / torch / pytensor backends are never used: backend names are
resolved to small dummy base classes instead, so the class-construction logic is
exercised on every platform without those packages.
"""

import sys
from abc import ABCMeta, abstractmethod

import pytest

from dit.algorithms import BaseAuxVarOptimizer
from dit.multivariate import _backend
from dit.multivariate._backend import _get_base_class, _is_mixin, _make_backend_subclass
from dit.multivariate.common_informations.base_markov_optimizer import MarkovVarMixin, MarkovVarOptimizer

# ── Dummy backend bases (stand-ins for e.g. BaseAuxVarJaxOptimizer) ──────


class DummyBackendRoot(metaclass=ABCMeta):
    """
    Root of a dummy backend hierarchy, analogous to ``BaseJaxOptimizer``.
    """

    backend_tag = "root"

    def __init__(self, *args, **kwargs):
        self.backend_init_args = (args, kwargs)

    def optimize(self):
        return "backend optimize"

    def describe(self):
        return "root"

    @abstractmethod
    def _objective(self):
        """
        Abstract hook implemented by concrete optimizers.
        """


class DummyBackendBase(DummyBackendRoot):
    """
    Dummy auxiliary-variable backend base, analogous to ``BaseAuxVarJaxOptimizer``.
    """

    backend_tag = "aux"

    def describe(self):
        return "aux"


class OtherBackendBase(DummyBackendBase):
    """
    A second dummy backend base.
    """


# ── Optimizers following the Mixin + composed class pattern ──────────────


class RecordingMixin:
    """
    Backend-agnostic logic whose ``super()`` calls must reach the backend base.
    """

    def __init__(self, value):
        super().__init__(value, flag=True)
        self.value = value

    def compute_bound(self):
        return 2 * self.value


class ComposedOptimizer(RecordingMixin, BaseAuxVarOptimizer):
    """
    Composed numpy class, analogous to ``MarkovVarOptimizer``.
    """

    name = "composed"
    construct_initial = BaseAuxVarOptimizer.construct_copy_initial

    def __init__(self, *args, **kwargs):
        self.composed_init_ran = True
        super().__init__(*args, **kwargs)

    @abstractmethod
    def describe(self):
        """
        Abstract placeholder that must not shadow the backend's implementation.
        """


class LeafOptimizer(ComposedOptimizer):
    """
    Concrete optimizer, analogous to ``ExactCommonInformation``.
    """

    name = "leaf"

    def _objective(self):
        return "leaf objective"


class OuterMixin:
    """
    A second mixin layered on top, analogous to ``MinimizingMarkovVarMixin``.
    """

    def __init__(self, value):
        super().__init__(value)
        self.outer_ran = True


class TwoMixinOptimizer(OuterMixin, LeafOptimizer):
    """
    Concrete optimizer with two mixins in its MRO.
    """


# ── An optimizer without any mixin (legacy strategy) ─────────────────────


class LegacyOptimizer(BaseAuxVarOptimizer):
    """
    Optimizer predating the mixin pattern: no ``*Mixin`` class in its MRO.
    """

    name = "legacy"

    def __init__(self, value):
        self.value = value

    def _objective(self):
        return "legacy objective"

    def optimize(self):
        return "legacy optimize"


@pytest.fixture
def dummy_backends(monkeypatch):
    """
    Resolve the backend names 'dummy' and 'other' to the dummy bases (deferring
    to the real resolver for any other name), start from an empty class cache,
    and record every backend resolution.
    """
    resolved = []
    real_get_base_class = _backend._get_base_class
    fakes = {"dummy": DummyBackendBase, "other": OtherBackendBase}

    def fake_get_base_class(backend="numpy"):
        resolved.append(backend)
        return fakes[backend] if backend in fakes else real_get_base_class(backend)

    monkeypatch.setattr(_backend, "_get_base_class", fake_get_base_class)
    monkeypatch.setattr(_backend, "_backend_class_cache", {})
    return resolved


# ── _get_base_class / _is_mixin ──────────────────────────────────────────


def test_get_base_class_defaults_to_numpy():
    """
    Without an argument the NumPy/SciPy auxiliary-variable optimizer is used.
    """
    assert _get_base_class() is BaseAuxVarOptimizer


@pytest.mark.parametrize("backend", ["jax", "torch", "pytensor"])
def test_get_base_class_imports_backends_lazily(monkeypatch, backend):
    """
    Optional backends are imported only on request: an unavailable backend fails
    with an ImportError (no silent numpy fallback) while numpy keeps working.
    """
    monkeypatch.setitem(sys.modules, f"dit.algorithms.optimization_{backend}", None)
    assert _get_base_class("numpy") is BaseAuxVarOptimizer
    with pytest.raises(ImportError, match=f"optimization_{backend}"):
        _get_base_class(backend)


def test_is_mixin_uses_class_name_suffix():
    """
    Mixins are recognised by the ``Mixin`` suffix of their class name.
    """
    assert _is_mixin(MarkovVarMixin)
    assert _is_mixin(RecordingMixin)
    assert not _is_mixin(MarkovVarOptimizer)
    assert not _is_mixin(BaseAuxVarOptimizer)
    assert not _is_mixin(type("MixinHelper", (), {}))


# ── _make_backend_subclass: resolution and caching ───────────────────────


def test_numpy_backend_is_a_no_op(dummy_backends):
    """
    The numpy backend returns the class itself without resolving or caching anything.
    """
    assert _make_backend_subclass(LeafOptimizer, "numpy") is LeafOptimizer
    assert _make_backend_subclass(LegacyOptimizer, "numpy") is LegacyOptimizer
    assert dummy_backends == []
    assert _backend._backend_class_cache == {}


def test_failed_backend_resolution_is_not_cached(dummy_backends, monkeypatch):
    """
    Unknown and unavailable backends raise, and leave nothing in the class cache.
    """
    with pytest.raises(ValueError, match="Unknown backend: 'tensorflow'. Choose from 'numpy', 'jax'"):
        _make_backend_subclass(LeafOptimizer, "tensorflow")
    monkeypatch.setitem(sys.modules, "dit.algorithms.optimization_jax", None)
    with pytest.raises(ImportError, match="optimization_jax"):
        _make_backend_subclass(LeafOptimizer, "jax")
    assert _backend._backend_class_cache == {}


def test_backend_classes_are_cached_per_class_and_backend(dummy_backends):
    """
    Each (class, backend) pair is built once; other pairs get distinct classes.
    """
    dummy_leaf = _make_backend_subclass(LeafOptimizer, "dummy")
    assert _make_backend_subclass(LeafOptimizer, "dummy") is dummy_leaf
    other_leaf = _make_backend_subclass(LeafOptimizer, "other")
    dummy_legacy = _make_backend_subclass(LegacyOptimizer, "dummy")

    assert other_leaf is not dummy_leaf
    assert other_leaf.__bases__ == (RecordingMixin, OtherBackendBase)
    assert dummy_legacy is not dummy_leaf
    # the cache hit did not resolve the backend a second time
    assert dummy_backends == ["dummy", "other", "dummy"]
    assert set(_backend._backend_class_cache) == {
        (LeafOptimizer, "dummy"),
        (LeafOptimizer, "other"),
        (LegacyOptimizer, "dummy"),
    }


# ── _make_backend_subclass: Mixin + composed class strategy ──────────────


def test_mixin_class_swaps_numpy_base_for_backend_base(dummy_backends):
    """
    The mixin is recombined with the backend base, which replaces BaseAuxVarOptimizer.
    """
    new_cls = _make_backend_subclass(LeafOptimizer, "dummy")
    assert new_cls.__name__ == "LeafOptimizer"
    assert new_cls.__bases__ == (RecordingMixin, DummyBackendBase)
    assert BaseAuxVarOptimizer not in new_cls.__mro__
    assert not issubclass(new_cls, LeafOptimizer)


def test_mixin_class_carries_concrete_leaf_attributes(dummy_backends):
    """
    Concrete attributes of the composed hierarchy are copied, the leaf overriding its parents.
    """
    new_cls = _make_backend_subclass(LeafOptimizer, "dummy")
    assert new_cls.name == "leaf"
    assert new_cls._objective is LeafOptimizer._objective
    assert new_cls.construct_initial is BaseAuxVarOptimizer.construct_copy_initial
    # methods the hierarchy does not define come from the backend
    assert new_cls.optimize is DummyBackendRoot.optimize


def test_mixin_class_skips_abstract_placeholders(dummy_backends):
    """
    An abstract placeholder in the composed hierarchy does not shadow the backend's
    implementation, so the generated class is concrete.
    """
    new_cls = _make_backend_subclass(LeafOptimizer, "dummy")
    assert "describe" not in new_cls.__dict__
    assert new_cls.describe is DummyBackendBase.describe
    assert new_cls.__abstractmethods__ == frozenset()
    assert new_cls(1).describe() == "aux"


def test_mixin_class_init_chain_reaches_backend(dummy_backends):
    """
    The composed class's __init__ is dropped: the mixin's __init__ runs and its
    super() call reaches the backend base.
    """
    new_cls = _make_backend_subclass(LeafOptimizer, "dummy")
    assert "__init__" not in new_cls.__dict__
    obj = new_cls(3)
    assert obj.backend_init_args == ((3,), {"flag": True})
    assert obj.value == 3
    assert obj.compute_bound() == 6
    assert obj._objective() == "leaf objective"
    assert not hasattr(obj, "composed_init_ran")


def test_multiple_mixins_keep_mro_order(dummy_backends):
    """
    Every mixin is kept, in MRO order, ahead of the backend base.
    """
    new_cls = _make_backend_subclass(TwoMixinOptimizer, "dummy")
    assert new_cls.__name__ == "TwoMixinOptimizer"
    assert new_cls.__bases__ == (OuterMixin, RecordingMixin, DummyBackendBase)
    obj = new_cls(4)
    assert obj.outer_ran
    assert obj.value == 4
    assert obj.backend_init_args == ((4,), {"flag": True})
    assert new_cls.name == "leaf"


# ── _make_backend_subclass: legacy strategy (no mixin) ───────────────────


def test_legacy_class_injects_backend_methods(dummy_backends):
    """
    Without a mixin, the backend's concrete methods are injected into a subclass of
    the class, the most-derived backend class taking precedence.
    """
    new_cls = _make_backend_subclass(LegacyOptimizer, "dummy")
    assert new_cls.__name__ == "LegacyOptimizer"
    assert new_cls.__bases__ == (LegacyOptimizer,)
    assert new_cls.optimize is DummyBackendRoot.optimize
    assert new_cls.describe is DummyBackendBase.describe
    assert new_cls.backend_tag == "aux"
    # attributes the backend does not define are inherited unchanged
    assert new_cls.name == "legacy"


def test_legacy_class_keeps_init_and_concrete_methods(dummy_backends):
    """
    The legacy class keeps its own __init__, and the backend's abstract placeholders
    do not override its concrete methods.
    """
    new_cls = _make_backend_subclass(LegacyOptimizer, "dummy")
    assert "__init__" not in new_cls.__dict__
    assert "_objective" not in new_cls.__dict__
    obj = new_cls(5)
    # Only LegacyOptimizer.__init__ ran, and the backend root's ``__dict__``
    # descriptor was not copied (it would make ``vars`` raise a TypeError).
    assert vars(obj) == {"value": 5}
    assert obj._objective() == "legacy objective"
    assert obj.optimize() == "backend optimize"
