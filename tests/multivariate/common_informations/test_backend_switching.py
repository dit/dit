"""
Tests for the backend-switching functionality in Markov variable optimizers.
"""

import pytest

from dit import Distribution as D
from dit.algorithms.optimization import BaseAuxVarOptimizer
from dit.multivariate._backend import (
    _get_base_class,
    _make_backend_subclass,
)
from dit.multivariate.common_informations.base_markov_optimizer import (
    MarkovVarMixin,
    MarkovVarOptimizer,
    MinimizingMarkovVarMixin,
    MinimizingMarkovVarOptimizer,
    make_markov_var_optimizer,
)
from dit.multivariate.common_informations.exact_common_information import (
    ExactCommonInformation,
    exact_common_information,
)
from dit.multivariate.common_informations.wyner_common_information import (
    WynerCommonInformation,
    wyner_common_information,
)

# ── _get_base_class ──────────────────────────────────────────────────────


class TestGetBaseClass:
    def test_numpy(self):
        cls = _get_base_class("numpy")
        assert cls is BaseAuxVarOptimizer

    def test_jax(self):
        from dit.algorithms.optimization_jax import BaseAuxVarJaxOptimizer

        cls = _get_base_class("jax")
        assert cls is BaseAuxVarJaxOptimizer

    def test_torch(self):
        from dit.algorithms.optimization_torch import BaseAuxVarTorchOptimizer

        cls = _get_base_class("torch")
        assert cls is BaseAuxVarTorchOptimizer

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            _get_base_class("tensorflow")


# ── MRO structure ────────────────────────────────────────────────────────


class TestMRO:
    def test_markov_var_optimizer_has_mixin(self):
        assert MarkovVarMixin in MarkovVarOptimizer.__mro__

    def test_markov_var_optimizer_has_base(self):
        assert BaseAuxVarOptimizer in MarkovVarOptimizer.__mro__

    def test_minimizing_has_both_mixins(self):
        assert MinimizingMarkovVarMixin in MinimizingMarkovVarOptimizer.__mro__
        assert MarkovVarMixin in MinimizingMarkovVarOptimizer.__mro__

    def test_exact_inherits_markov(self):
        assert MarkovVarOptimizer in ExactCommonInformation.__mro__
        assert MarkovVarMixin in ExactCommonInformation.__mro__

    def test_wyner_inherits_markov(self):
        assert MarkovVarOptimizer in WynerCommonInformation.__mro__
        assert MarkovVarMixin in WynerCommonInformation.__mro__


# ── _make_backend_subclass ───────────────────────────────────────────────


class TestMakeBackendSubclass:
    def test_numpy_is_identity(self):
        result = _make_backend_subclass(ExactCommonInformation, "numpy")
        assert result is ExactCommonInformation

    def test_jax_has_correct_methods(self):
        from dit.algorithms.optimization_jax import BaseAuxVarJaxOptimizer, BaseJaxOptimizer

        JaxExact = _make_backend_subclass(ExactCommonInformation, "jax")
        # The new class has the mixin and JAX base
        assert MarkovVarMixin in JaxExact.__mro__
        assert BaseAuxVarJaxOptimizer in JaxExact.__mro__
        # Verify JAX-specific methods are available
        assert hasattr(JaxExact, "_use_jit")
        assert JaxExact.optimize is BaseJaxOptimizer.optimize

    def test_torch_has_correct_methods(self):
        from dit.algorithms.optimization_torch import BaseAuxVarTorchOptimizer, BaseTorchOptimizer

        TorchExact = _make_backend_subclass(ExactCommonInformation, "torch")
        # The new class has the mixin and torch base
        assert MarkovVarMixin in TorchExact.__mro__
        assert BaseAuxVarTorchOptimizer in TorchExact.__mro__
        # Verify torch-specific methods are available
        assert hasattr(TorchExact, "_use_autodiff")
        assert TorchExact.optimize is BaseTorchOptimizer.optimize

    def test_preserves_name(self):
        JaxExact = _make_backend_subclass(ExactCommonInformation, "jax")
        assert JaxExact.__name__ == "ExactCommonInformation"

    def test_preserves_class_attrs(self):
        JaxExact = _make_backend_subclass(ExactCommonInformation, "jax")
        assert JaxExact.name == "exact"
        assert hasattr(JaxExact, "compute_bound")
        assert hasattr(JaxExact, "_objective")

    def test_preserves_wyner_attrs(self):
        JaxWyner = _make_backend_subclass(WynerCommonInformation, "jax")
        assert JaxWyner.name == "wyner"
        assert hasattr(JaxWyner, "compute_bound")
        assert hasattr(JaxWyner, "_objective")

    def test_has_mixin(self):
        JaxExact = _make_backend_subclass(ExactCommonInformation, "jax")
        assert MarkovVarMixin in JaxExact.__mro__

    def test_caching(self):
        cls1 = _make_backend_subclass(ExactCommonInformation, "jax")
        cls2 = _make_backend_subclass(ExactCommonInformation, "jax")
        assert cls1 is cls2

    def test_different_backends_different_classes(self):
        jax_cls = _make_backend_subclass(ExactCommonInformation, "jax")
        torch_cls = _make_backend_subclass(ExactCommonInformation, "torch")
        assert jax_cls is not torch_cls


# ── make_markov_var_optimizer ────────────────────────────────────────────


class TestMakeMarkovVarOptimizer:
    def test_numpy_returns_original(self):
        cls = make_markov_var_optimizer("numpy")
        assert cls is MarkovVarOptimizer

    def test_jax_returns_jax_based(self):
        from dit.algorithms.optimization_jax import BaseAuxVarJaxOptimizer, BaseJaxOptimizer

        cls = make_markov_var_optimizer("jax")
        assert MarkovVarMixin in cls.__mro__
        assert BaseAuxVarJaxOptimizer in cls.__mro__
        assert cls.optimize is BaseJaxOptimizer.optimize

    def test_torch_returns_torch_based(self):
        from dit.algorithms.optimization_torch import BaseAuxVarTorchOptimizer, BaseTorchOptimizer

        cls = make_markov_var_optimizer("torch")
        assert MarkovVarMixin in cls.__mro__
        assert BaseAuxVarTorchOptimizer in cls.__mro__
        assert cls.optimize is BaseTorchOptimizer.optimize


# ── functional() backend parameter ──────────────────────────────────────


class TestFunctionalBackendParam:
    def test_exact_has_backend_param(self):
        import inspect

        sig = inspect.signature(exact_common_information)
        assert "backend" in sig.parameters
        assert sig.parameters["backend"].default == "numpy"

    def test_wyner_has_backend_param(self):
        import inspect

        sig = inspect.signature(wyner_common_information)
        assert "backend" in sig.parameters
        assert sig.parameters["backend"].default == "numpy"


# ── End-to-end numpy (backward compat) ──────────────────────────────────


@pytest.mark.flaky(reruns=5)
def test_exact_numpy_backward_compat():
    """ExactCommonInformation with default numpy backend produces correct results."""
    d = D([(0, 0), (1, 1)], [0.5, 0.5])
    eci = ExactCommonInformation(d)
    eci.optimize()
    val = eci.objective(eci._optima)
    assert val == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_exact_functional_numpy_backward_compat():
    """exact_common_information(d) works unchanged."""
    d = D([(0, 0), (1, 1)], [0.5, 0.5])
    val = exact_common_information(d)
    assert float(val) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_exact_functional_explicit_numpy():
    """Explicitly passing backend='numpy' is equivalent to the default."""
    d = D([(0, 0), (1, 1)], [0.5, 0.5])
    val = exact_common_information(d, backend="numpy")
    assert float(val) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_wyner_numpy_backward_compat():
    """WynerCommonInformation with default numpy backend produces correct results."""
    d = D([(0, 0), (1, 1)], [0.5, 0.5])
    wci = WynerCommonInformation(d, bound=2)
    wci.optimize()
    val = wci.objective(wci._optima)
    assert float(val) == pytest.approx(1.0, abs=1e-3)
