"""
Tests for dit.algorithms.optimizers
"""

from itertools import product
from types import MethodType

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, approx_fprime

from dit import Distribution
from dit.algorithms import maxent_dist
from dit.algorithms.distribution_optimizers import (
    BROJAConEOptimizer,
    MaxCAEKLMutualInformationOptimizer,
    MaxCoInfoOptimizer,
    MaxDualTotalCorrelationOptimizer,
    MaxEntOptimizer,
    MinCoInfoOptimizer,
    MinDualTotalCorrelationOptimizer,
    MinEntOptimizer,
    MinResidualEntropyOptimizer,
    marginal_maxent_dists,
)
from dit.distconst import uniform
from dit.example_dists import Xor
from dit.multivariate import coinformation as I
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import entropy as H
from dit.pid.distributions import bivariates


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


@pytest.mark.parametrize("n_groups", [2, 3])
def test_minent_no_worse_than_greedy(n_groups):
    """
    The minimum-entropy coupling is never worse than the greedy coupling it is seeded with.
    """
    rng = np.random.default_rng(0)
    margs = [rng.dirichlet(np.ones(4)) for _ in range(n_groups)]
    pmf = margs[0]
    for m in margs[1:]:
        pmf = np.multiply.outer(pmf, m)
    outcomes = ["".join(map(str, idx)) for idx in np.ndindex(*pmf.shape)]
    d = Distribution(outcomes, pmf.ravel())
    meo = MinEntOptimizer(d, [[i] for i in range(n_groups)])
    greedy = meo._greedy_coupling()
    assert greedy.reshape(pmf.shape).sum(axis=tuple(range(1, n_groups))) == pytest.approx(margs[0])
    meo.optimize()
    nz = greedy[greedy > 0]
    assert H(meo.construct_dist()) <= -(nz * np.log2(nz)).sum() + 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_minent_matches_marginals(seed):
    """
    The marginal constraints are linear equalities of full rank, so the optimum matches
    the marginals to machine precision. A single squared deviation used to count as
    satisfied below SLSQP's tolerance of 1e-7, leaving deviations of about 3e-4 that
    could make the optimum look better than the greedy coupling. An integer seed also
    used to crash the vertex seeds.
    """
    rng = np.random.default_rng(0)
    margs = [rng.dirichlet(np.ones(4)) for _ in range(2)]
    pmf = np.multiply.outer(*margs)
    d = Distribution(["".join(map(str, idx)) for idx in np.ndindex(*pmf.shape)], pmf.ravel())
    meo = MinEntOptimizer(d, [[0], [1]])
    meo.optimize(rng=seed)
    joint = meo.construct_joint(meo._optima)
    assert joint.sum(axis=1) == pytest.approx(margs[0], abs=1e-10)
    assert joint.sum(axis=0) == pytest.approx(margs[1], abs=1e-10)


@pytest.mark.parametrize(
    "Optimizer",
    [
        MaxEntOptimizer,
        MinEntOptimizer,
        MaxCoInfoOptimizer,
        MinCoInfoOptimizer,
        MaxDualTotalCorrelationOptimizer,
        MinDualTotalCorrelationOptimizer,
        MinResidualEntropyOptimizer,
        MaxCAEKLMutualInformationOptimizer,
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
        con_fd = approx_fprime(x, opt.constraint_match_marginals, 1e-7)
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
        # Blend two random channel parametrizations to stay in the interior, then
        # rescale entries: the optimizer searches unnormalized rows of ``x``.
        x = np.asarray(opt.construct_random_initial()).ravel()
        x = 0.7 * x + 0.3 * np.asarray(opt.construct_random_initial()).ravel()
        x = x * np.random.uniform(0.5, 1.5, size=x.size)

        an = opt._jacobian(x)
        fd = approx_fprime(x, lambda v: float(opt.objective(v)), 1e-7)
        worst = max(worst, float(np.max(np.abs(an - fd))))

    assert worst < 1e-3, f"{label}: analytic vs FD gradient mismatch {worst:.2e}"


def test_igh_analytic_gradients_match_fd():
    """
    The GHOptimizer objective gradient (``_jacobian``) and the analytic
    Markov-chain equality-constraint jacobian must both agree with SciPy's
    finite-difference gradients at interior points.
    """
    from dit.pid.distributions import bivariates
    from dit.pid.measures.igh import GHOptimizer

    np.random.seed(0)
    opt = GHOptimizer(bivariates["and"], [[0], [1]], [2])
    opt.objective = MethodType(opt._objective(), opt)
    con_fun = opt.constraint_markov_chains()
    con_jac = opt.constraint_markov_chains_jac()

    for _ in range(6):
        x = np.asarray(opt.construct_random_initial()).ravel()
        x = 0.7 * x + 0.3 * np.asarray(opt.construct_random_initial()).ravel()

        obj_an = opt._jacobian(x)
        obj_fd = approx_fprime(x, lambda v: float(opt.objective(v)), 1e-7)
        assert obj_an == pytest.approx(obj_fd, abs=1e-3)

        con_an = con_jac(x)
        con_fd = approx_fprime(x, lambda v: float(con_fun(v)), 1e-7)
        assert con_an == pytest.approx(con_fd, abs=1e-4)


def test_imc_analytic_gradients_match_fd():
    """
    The ``imc`` objective/constraint jacobian building blocks (``_mi_grad_wrt_K``
    composed with ``_softmax_vjp``) must agree with the finite-difference gradient
    of ``params -> _mi_bits(pi, softmax(params))`` for both interior and
    boundary (vertex/edge) input distributions.
    """
    from dit.pid.measures.imc import _mi_bits, _mi_grad_wrt_K, _params_to_stochastic, _softmax_vjp

    rng = np.random.default_rng(0)
    n_t, n_q = 3, 3
    pis = [
        np.array([0.5, 0.3, 0.2]),  # interior
        np.array([1.0, 0.0, 0.0]),  # vertex (zeros in pi)
        np.array([0.6, 0.4, 0.0]),  # edge (zero in pi)
    ]
    for pi in pis:
        for _ in range(5):
            params = rng.standard_normal(n_t * n_q)
            k_q = _params_to_stochastic(params, n_t, n_q)
            an = _softmax_vjp(k_q, _mi_grad_wrt_K(pi, k_q))
            fd = approx_fprime(
                params,
                lambda v, _pi=pi: _mi_bits(_pi, _params_to_stochastic(v, n_t, n_q)),
                1e-7,
            )
            assert an == pytest.approx(fd, abs=1e-5)


def test_mincoinfo_1():
    """
    Test mincoinfo
    """
    d = uniform(["000", "111"])
    mcio = MinCoInfoOptimizer(d, [[0], [1], [2]])
    mcio.optimize()
    dp = mcio.construct_dist()
    assert I(dp) == pytest.approx(-1, abs=1e-4)


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


# ── construct_dist, minimum-entropy seeds, the cone optimizer, marginal_maxent_dists ──


def test_construct_dist_explicit_vector_keeps_rv_names():
    """
    construct_dist builds the distribution of an explicit optimization vector: entries
    below the cutoff are dropped, the rest renormalized, and the variable names kept.
    """
    d = Distribution(["000", "011", "101", "110"], [0.25] * 4)
    d.set_rv_names("XYZ")
    meo = MaxEntOptimizer(d, [["X"], ["Y"], ["Z"]])
    x = 2 * np.array([0.25, 1e-9, 1e-9, 0.25, 1e-9, 0.25, 0.25, 1e-9])

    dp = meo.construct_dist(x=x)
    assert dp.get_rv_names() == ("X", "Y", "Z")
    assert dp.is_approx_equal(Distribution(["000", "011", "101", "110"], [0.25] * 4))

    dense = meo.construct_dist(x=x, sparse=False)
    assert dense.get_rv_names() == ("X", "Y", "Z")
    np.testing.assert_allclose(dense.pmf, [0.25, 0, 0, 0.25, 0, 0.25, 0.25, 0])


@pytest.mark.parametrize("marginals", [[[0, 1, 2]], [[0, 1], [1, 2]], [[0], [1]]])
def test_minent_greedy_coupling_needs_disjoint_cover(marginals):
    """
    The greedy coupling needs two or more disjoint groups covering every variable;
    otherwise only linear-programming vertices of the marginal polytope seed the search.
    """
    d = Distribution(["000", "011", "101", "110"], [0.1, 0.2, 0.3, 0.4])
    meo = MinEntOptimizer(d, marginals)
    assert meo._greedy_coupling() is None

    meo._n_vertex_seeds = 3
    seeds = meo._seed_pmfs(rng=0)
    assert len(seeds) == 3
    rank = np.linalg.matrix_rank(meo._A)
    for seed in seeds:
        np.testing.assert_allclose(meo._A @ seed, meo._b, atol=1e-9)
        assert seed.min() >= 0
        assert np.count_nonzero(seed > 1e-12) <= rank


def test_minent_seeds_skip_failed_linear_programs(monkeypatch):
    """
    Vertex seeds whose linear program returns no solution are skipped. The greedy coupling
    that remains is already optimal here, as the only couplings of (1/2, 1/2) and
    (3/5, 2/5) that are vertices have entropy H(1/2, 2/5, 1/10).
    """
    failed = OptimizeResult(x=None, success=False, status=2, message="forced failure")
    monkeypatch.setattr("dit.algorithms.distribution_optimizers.linprog", lambda *args, **kwargs: failed)
    d = Distribution(["00", "01", "10", "11"], [0.3, 0.2, 0.3, 0.2])
    meo = MinEntOptimizer(d, [[0], [1]])

    seeds = meo._seed_pmfs(rng=0)
    assert len(seeds) == 1
    np.testing.assert_allclose(seeds[0], [0.5, 0.0, 0.1, 0.4])

    meo.optimize(niter=2, rng=0)
    expected = -(0.5 * np.log2(0.5) + 0.4 * np.log2(0.4) + 0.1 * np.log2(0.1))
    assert H(meo.construct_dist()) == pytest.approx(expected, abs=1e-6)


def test_minent_fully_constrained_returns_input():
    """When the marginals determine the joint there is nothing to optimize, and the input comes back."""
    d = Distribution(["000", "011", "101", "110"], [0.1, 0.2, 0.3, 0.4])
    meo = MinEntOptimizer(d, [[0, 1, 2]])
    assert meo._free == []
    assert meo.optimize() is None
    assert meo.construct_dist().is_approx_equal(Distribution(["000", "011", "101", "110"], [0.1, 0.2, 0.3, 0.4]))


def test_broja_cone_optimizer_and_gate():
    """
    The exponential-cone BROJA optimizer keeps each (source, target) marginal and gives the
    AND gate's known decomposition: no unique information, so the redundancy, the
    co-information of the optimum, equals I(X0 : Y).
    """
    pytest.importorskip("ecos")
    d = bivariates["and"]
    opt = BROJAConEOptimizer(d, [[0], [1]], [2])
    assert opt.optimize() is None
    assert opt._cone_meta["converged"]
    assert opt._cone_meta["ui_source0"] == pytest.approx(0.0, abs=1e-6)
    assert opt._cone_meta["ui_source1"] == pytest.approx(0.0, abs=1e-6)

    p = np.zeros((2, 2, 2))
    for x0, x1, y in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]:
        p[x0, x1, y] = 0.25
    q = opt.construct_joint(opt._optima)
    np.testing.assert_allclose(q.sum(axis=1), p.sum(axis=1), atol=1e-6)
    np.testing.assert_allclose(q.sum(axis=0), p.sum(axis=0), atol=1e-6)
    assert I(opt.construct_dist()) == pytest.approx(I(d, [[0], [2]]), abs=1e-6)


def test_marginal_maxent_dists_truncated_order():
    """
    With ``k_max`` below the number of variables the sequence stops at order ``k_max``
    and omits the distribution itself. For the giant bit the pairwise marginals already
    pin down the joint, so the entropies are 3, 3 and 1 bit.
    """
    dists = marginal_maxent_dists(Distribution(["000", "111"], [0.5, 0.5]), k_max=2)
    assert [H(dist) for dist in dists] == pytest.approx([3, 3, 1])
    assert dists[2].is_approx_equal(Distribution(["000", "111"], [0.5, 0.5]), atol=1e-6)

    dists = marginal_maxent_dists(Distribution(["000", "111"], [0.5, 0.5]), k_max=1)
    assert len(dists) == 2
    np.testing.assert_allclose(dists[1].pmf, 1 / 8)
