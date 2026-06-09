"""
Tests for dit.algorithms.optimizers
"""

import warnings
from itertools import product
from types import MethodType

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from dit.algorithms import maxent_dist, pid_broja
from dit.algorithms.distribution_optimizers import (
    MaxCoInfoOptimizer,
    MaxDualTotalCorrelationOptimizer,
    MaxEntOptimizer,
    MinCoInfoOptimizer,
    MinDualTotalCorrelationOptimizer,
    MinEntOptimizer,
)
from dit.distconst import uniform
from dit.example_dists import Rdn, Unq, Xor
from dit.multivariate import coinformation as I
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import entropy as H


@pytest.mark.parametrize(
    "rvs",
    [
        [[0], [1], [2]],
        [[0, 1], [2]],
        [[0, 2], [1]],
        [[0], [1, 2]],
        [[0, 1], [1, 2]],
        [[0, 1], [0, 2]],
        [[0, 2], [1, 2]],
        [[0, 1], [0, 2], [1, 2]],
    ],
)
def test_maxent_1(rvs):
    """
    Test xor only fixing individual marginals.
    """
    d1 = uniform(["000", "011", "101", "110"])
    d2 = uniform(["000", "001", "010", "011", "100", "101", "110", "111"])
    d1_maxent = maxent_dist(d1, rvs)
    assert d2.is_approx_equal(d1_maxent, rtol=1e-3, atol=1e-3)


def test_maxent_2():
    """
    Text a distribution with differing alphabets.
    """
    d1 = uniform(["00", "10", "21", "31"])
    d2 = uniform(["00", "01", "10", "11", "20", "21", "30", "31"])
    d1_maxent = maxent_dist(d1, [[0], [1]])
    assert d2.is_approx_equal(d1_maxent, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
def test_maxent_3():
    """
    Test the RdnUnqXor distribution.
    """
    X00, X01, X02, Y01, Y02 = "rR", "aA", [0, 1], "bB", [0, 1]
    inputs = product(X00, X01, X02, Y01, Y02)
    events = [
        (x00 + x01 + str(x02), x00 + y01 + str(y02), x00 + x01 + y01 + str(x02 ^ y02))
        for x00, x01, x02, y01, y02 in inputs
    ]
    RdnUnqXor = uniform(events)
    d = maxent_dist(RdnUnqXor, [[0, 1], [0, 2], [1, 2]])
    assert H(d) == pytest.approx(6)


def test_minent_1():
    """
    Test minent
    """
    d = uniform(["000", "001", "010", "011", "100", "101", "110", "111"])
    meo = MinEntOptimizer(d, [[0], [1], [2]])
    meo.optimize()
    dp = meo.construct_dist()
    assert H(dp) == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize(
    "Optimizer",
    [
        MaxEntOptimizer,
        MinEntOptimizer,
        MaxCoInfoOptimizer,
        MinCoInfoOptimizer,
        MaxDualTotalCorrelationOptimizer,
        MinDualTotalCorrelationOptimizer,
    ],
)
def test_analytic_gradients_match_fd(Optimizer):
    """
    The analytic objective gradient (``_jacobian``) and the analytic constraint
    jacobian (``constraint_match_marginals_jac``) of every linear distribution
    optimizer must agree with SciPy's finite-difference gradients to within the
    finite-difference step error.
    """
    np.random.seed(0)
    d = uniform([f"{i:04b}" for i in range(16)])
    opt = Optimizer(d, [[0, 1], [1, 2], [2, 3]])
    opt.objective = MethodType(opt._objective(), opt)

    for _ in range(5):
        # Blend the random simplex point toward uniform so every free
        # probability stays well away from the 0 boundary (where the analytic
        # gradient intentionally floors and finite differences are unreliable).
        x = np.asarray(opt.construct_random_initial()).ravel()
        x = 0.5 * x + 0.5 / x.size

        obj_an = opt._jacobian(x)
        obj_fd = approx_fprime(x, lambda v: float(opt.objective(v)), 1e-7)
        assert obj_an == pytest.approx(obj_fd, abs=1e-3)

        con_an = opt.constraint_match_marginals_jac(x)
        con_fd = approx_fprime(x, lambda v: float(opt.constraint_match_marginals(v)), 1e-7)
        assert con_an == pytest.approx(con_fd, abs=1e-5)


def _auxvar_gradient_cases():
    """Build (label, optimizer) pairs for every analytic-gradient aux-var optimizer."""
    from dit.multivariate.common_informations.exact_common_information import ExactCommonInformation
    from dit.multivariate.common_informations.stochastic_gk_common_information import (
        StochasticGKCommonInformation,
    )
    from dit.multivariate.deweese import (
        DeWeeseCoInformation,
        DeWeeseDualTotalCorrelation,
        DeWeeseTotalCorrelation,
    )
    from dit.multivariate.secret_key_agreement.base_skar_optimizers import (
        InnerTwoPartIntrinsicMutualInformation,
    )
    from dit.multivariate.secret_key_agreement.interactive_intrinsic_mutual_informations import InteractiveSKAR
    from dit.multivariate.secret_key_agreement.intrinsic_mutual_informations import (
        IntrinsicDualTotalCorrelation,
        IntrinsicTotalCorrelation,
    )
    from dit.multivariate.secret_key_agreement.minimal_intrinsic_mutual_informations import (
        MinimalIntrinsicCAEKLMutualInformation,
        MinimalIntrinsicDualTotalCorrelation,
        MinimalIntrinsicTotalCorrelation,
    )
    from dit.multivariate.secret_key_agreement.one_way_skar import OneWaySKAR
    from dit.pid.measures.ideltalambda import DeltaLambdaOptimizer
    from dit.rate_distortion.information_bottleneck import InformationBottleneck
    from dit.rate_distortion.rate_distortion import (
        RateDistortionHamming,
        RateDistortionResidualEntropy,
    )

    xor = Xor()
    skar_dist = uniform(["000", "011", "101", "110"])
    inner_dist = uniform(["0000", "0011", "0101", "0110", "1001", "1010", "1100", "1111"])
    rd_dist = uniform(["000", "011", "101", "110", "001", "010"])
    return [
        ("exact", ExactCommonInformation(xor, [[0], [1]], [2])),
        ("stochastic_gk", StochasticGKCommonInformation(xor, [[0], [1]], [2])),
        ("intrinsic_tc", IntrinsicTotalCorrelation(xor, [[0], [1]], [2], bound=2)),
        ("intrinsic_dtc", IntrinsicDualTotalCorrelation(xor, [[0], [1]], [2], bound=2)),
        ("minimal_tc", MinimalIntrinsicTotalCorrelation(skar_dist, [[0], [1]], [2], bound=3)),
        ("minimal_dtc", MinimalIntrinsicDualTotalCorrelation(skar_dist, [[0], [1]], [2], bound=3)),
        ("minimal_caekl", MinimalIntrinsicCAEKLMutualInformation(skar_dist, [[0], [1]], [2], bound=3)),
        ("interactive_skar", InteractiveSKAR(skar_dist, rv_x=[0], rv_y=[1], rv_z=[2], rounds=3)),
        (
            "inner_two_part",
            InnerTwoPartIntrinsicMutualInformation(inner_dist, rvs=[[0], [1]], crvs=[2], j=[3], bound_u=2, bound_v=2),
        ),
        ("deweese_tc", DeWeeseTotalCorrelation(xor, [[0], [1], [2]], [])),
        ("deweese_coi", DeWeeseCoInformation(xor, [[0], [1], [2]], [])),
        ("deweese_dtc", DeWeeseDualTotalCorrelation(xor, [[0], [1], [2]], [])),
        ("one_way_skar", OneWaySKAR(skar_dist, [0], [1], [2], bound_u=2, bound_v=2)),
        ("rd_residual", RateDistortionResidualEntropy(rd_dist, beta=1.5, rv=[0], crvs=[1])),
        ("rd_hamming", RateDistortionHamming(rd_dist, beta=1.5, rv=[0], crvs=[1])),
        ("information_bottleneck", InformationBottleneck(rd_dist, beta=2.0, rvs=[[0], [1]], crvs=[2])),
        ("delta_lambda", DeltaLambdaOptimizer(xor, [0], [1], [2], lam=1.0)),
    ]


@pytest.mark.parametrize("label,opt", _auxvar_gradient_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_auxvar_analytic_gradients_match_fd(label, opt):
    """
    The analytic objective gradient of every wired auxiliary-variable optimizer
    (the ``construct_joint`` VJP composed with the measure pmf-gradient) must
    agree with SciPy's finite-difference gradient at interior points.
    """
    np.random.seed(0)
    opt.objective = MethodType(opt._objective(), opt)

    worst = 0.0
    for _ in range(6):
        # Blend two random channel parametrizations to stay in the interior.
        x = np.asarray(opt.construct_random_initial()).ravel()
        x = 0.7 * x + 0.3 * np.asarray(opt.construct_random_initial()).ravel()

        an = opt._jacobian(x)
        fd = approx_fprime(x, lambda v: float(opt.objective(v)), 1e-7)
        worst = max(worst, float(np.max(np.abs(an - fd))))

    assert worst < 1e-3, f"{label}: analytic vs FD gradient mismatch {worst:.2e}"


def test_mincoinfo_1():
    """
    Test mincoinfo
    """
    d = uniform(["000", "111"])
    mcio = MinCoInfoOptimizer(d, [[0], [1], [2]])
    mcio.optimize()
    dp = mcio.construct_dist()
    assert I(dp) == pytest.approx(-1, abs=1e-4)


@pytest.mark.skip(reason="This method if deprecated.")
@pytest.mark.parametrize(
    ("dist", "vals"),
    [
        (Rdn(), (1, 0, 0, 0)),
        (Unq(), (0, 1, 1, 0)),
        (Xor(), (0, 0, 0, 1)),
    ],
)
def test_broja_1(dist, vals):
    """
    Test broja.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pid = pid_broja(dist, [[0], [1]], [2])
    assert pid == pytest.approx(vals, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_dtc_1():
    """
    test max dtc
    """
    d = uniform(["000", "111"])
    max_dtc = MaxDualTotalCorrelationOptimizer(d, [[0], [1], [2]])
    max_dtc.optimize()
    dp = max_dtc.construct_dist()
    assert B(dp) == pytest.approx(2.0, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_dtc_2():
    """
    test min dtc
    """
    d = uniform(["000", "111"])
    max_dtc = MinDualTotalCorrelationOptimizer(d, [[0], [1], [2]])
    max_dtc.optimize()
    dp = max_dtc.construct_dist()
    assert B(dp) == pytest.approx(0.0, abs=1e-4)
