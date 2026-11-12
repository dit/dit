"""Reference-family formulas for the bivariate dependency PIDs."""

import importlib

import numpy as np
import pytest

from dit import Distribution
from dit.algorithms import maxent_dist
from dit.example_dists import dyadic, triadic
from dit.multivariate import coinformation
from dit.pid import PID_BROJA, PID_MMI, PID_RA, PID_WB, PID_dep, PID_dep_a, PID_dep_c
from dit.pid.distributions import bivariates, trivariates
from dit.profiles import DependencyDecomposition
from dit.shannon import entropy

idep_module = importlib.import_module("dit.pid.measures.idep")
references_module = importlib.import_module("dit.algorithms.pid_references")
RED = ((0,), (1,))
SYN = ((0, 1),)


@pytest.mark.parametrize("name", sorted(bivariates))
def test_direct_matches_dependency_lattice(name):
    """PID_dep_a retains the original four-edge minimum for two sources."""
    direct = PID_dep(bivariates[name])
    reference = PID_dep_a(bivariates[name])
    for node in reference._lattice:
        assert direct.get_pi(node) == pytest.approx(reference.get_pi(node), abs=1e-5)


@pytest.mark.parametrize("shape,alpha", [((2, 2, 2), 1.0), ((2, 2, 2), 0.2), ((3, 3, 3), 1.0)])
def test_direct_on_dense_joints(shape, alpha):
    rng = np.random.default_rng(261108131)
    for _ in range(8):
        p = rng.dirichlet(np.full(np.prod(shape), alpha)).reshape(shape)
        d = Distribution.from_ndarray(p)
        direct, reference = PID_dep(d), PID_dep_a(d)
        for node in reference._lattice:
            assert direct.get_pi(node) == pytest.approx(reference.get_pi(node), abs=1e-5)


def test_direct_on_sparse_joints():
    """Sparse joints, including induced structural zeros, use the same max-ent fallback."""
    rng = np.random.default_rng(261108132)
    for _ in range(12):
        p = rng.random((2, 2, 2))
        p[rng.random(p.shape) < 0.4] = 0.0
        d = Distribution.from_ndarray(p / p.sum())
        direct, reference = PID_dep(d), PID_dep_a(d)
        for node in reference._lattice:
            assert direct.get_pi(node) == pytest.approx(reference.get_pi(node), abs=1e-5)


@pytest.mark.parametrize(
    "sources,target",
    [
        ([[0, 1], [2]], [3]),
        ([[0, 1], [1, 2]], [3]),
        ([[0], [1]], [2, 3]),
        ([[2], [0]], [3]),
    ],
)
def test_grouping_and_names(sources, target):
    """Coalescing precedes fitting, including overlapping sources and multivariate targets."""
    rng = np.random.default_rng(7)
    d = Distribution.from_ndarray(rng.dirichlet(np.ones(16)).reshape((2,) * 4))
    direct, reference = PID_dep(d, sources, target), PID_dep_a(d, sources, target)
    for node in reference._lattice:
        assert direct.get_pi(node) == pytest.approx(reference.get_pi(node), abs=1e-5)
    d.set_rv_names(["A", "B", "C", "T"])
    names = ["A", "B", "C", "T"]
    named_sources = [[names[i] for i in s] for s in sources]
    named_target = [names[i] for i in target]
    named = PID_dep(d, named_sources, named_target)
    for node in direct._lattice:
        named_node = tuple(tuple(names[i] for i in group) for group in node)
        assert named.get_pi(named_node) == pytest.approx(direct.get_pi(node), abs=1e-5)


@pytest.mark.parametrize("base", [np.e, 10])
def test_dense_and_log_base(base):
    reference = PID_dep_a(bivariates["and"])
    d = bivariates["and"].copy(base=base)
    d.make_dense()
    actual = PID_dep(d)
    for node in reference._lattice:
        assert actual.get_pi(node) == pytest.approx(reference.get_pi(node) / np.log2(base), abs=1e-5)


def test_one_fit_without_dependency_lattice(monkeypatch):
    """The fast path uses one pairwise fit, forwarding the existing maxiter keyword."""
    fits = []

    def fit(d, rvs, **kwargs):
        fits.append((rvs, kwargs))
        return maxent_dist(d, rvs, **kwargs)

    def no_lattice(*args, **kwargs):
        raise AssertionError("the bivariate fast path must not construct the dependency lattice")

    monkeypatch.setattr(references_module, "maxent_dist", fit)
    monkeypatch.setattr(idep_module, "DependencyDecomposition", no_lattice)
    pid = PID_dep(bivariates["and"], maxiter=17)
    assert pid.consistent
    assert pid.get_red(RED) == pytest.approx(0.08170416594551, abs=1e-6)
    assert len(fits) == 1
    assert set(map(tuple, fits[0][0])) == {(0, 1), (0, 2), (1, 2)}
    assert fits[0][1]["maxiter"] == 17


def test_more_sources_keep_existing_definition():
    """The higher-source unique values are still the original focal-source-versus-rest minima."""
    d = trivariates["anddup"]
    sources = ((0,), (1,), (2,))
    actual = PID_dep._measure(d, sources, (3,))
    for source in sources:
        rest = tuple(i for other in sources if other != source for i in other)
        reference = PID_dep_a(d, (source, rest), (3,))
        assert actual[source] == pytest.approx(reference.get_pi((source,)), abs=1e-5)


@pytest.mark.parametrize("d", [*bivariates.values(), dyadic, triadic])
def test_pairwise_reference_equivalence(d):
    """Definition 2 is the clipped RA decomposition, not a distinct two-source measure."""
    p2 = maxent_dist(d, [[0, 1], [0, 2], [1, 2]])
    c2 = coinformation(p2)
    i3 = entropy(p2) - entropy(d)
    reference = PID_dep_c(d)
    ra = PID_RA(d)
    assert ra.get_red(RED) == pytest.approx(c2, abs=1e-5)
    assert reference.get_red(RED) == pytest.approx(max(c2, 0.0), abs=1e-5)
    assert reference.get_pi(SYN) == pytest.approx(i3 + max(-c2, 0.0), abs=1e-5)
    assert reference.nonnegative
    assert reference.consistent


@pytest.mark.parametrize("name", sorted(bivariates))
def test_reference_family_ordering(name):
    """The nested reference families order these nonnegative bivariate PIDs."""
    d = bivariates[name]
    low = max(0.0, coinformation(d))
    pairwise = PID_dep_c(d).get_red(RED)
    dep = PID_dep(d).get_red(RED)
    broja = PID_BROJA(d).get_red(RED)
    mmi = PID_MMI(d).get_red(RED)
    wb = PID_WB(d).get_red(RED)
    for lhs, rhs in zip([low, pairwise, dep, broja], [pairwise, dep, broja, mmi], strict=True):
        assert lhs <= rhs + 1e-5
    assert broja <= wb + 1e-5
    assert wb <= mmi + 1e-5


def _lattice_uniques(d, sources, target, kind):
    """
    Each source's unique information computed as originally defined, on the dependency
    lattice of the source, the other sources together, and the target.
    """
    measure = {"I": lambda dist: coinformation(dist, [[0, 1], [2]])}
    added = frozenset([frozenset((0, 2))])
    all_pairs = frozenset([frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 2))])
    sources_joint = frozenset((0, 1))
    uniques = {}
    for source in sources:
        others = sum((other for other in sources if other != source), ())
        dd = DependencyDecomposition(d.coalesce([source, others, target]), measures=measure)
        if kind == "dep":
            uniques[source] = min(dd.delta(edge, "I") for edge in dd.edges(added))
        elif kind == "ra":
            uniques[source] = dd.delta((all_pairs, all_pairs - added), "I")
        else:  # dep_c keeps only the edges whose upper node constrains the sources jointly
            uniques[source] = min(
                dd.delta(edge, "I") for edge in dd.edges(added) if any(sources_joint <= c for c in edge[0])
            )
    return uniques


def _random_joints(n_sources, count, seed, sparse=False):
    rng = np.random.default_rng(seed)
    joints = []
    for _ in range(count):
        p = rng.dirichlet(np.ones(2 ** (n_sources + 1)))
        if sparse:
            p[rng.random(p.shape) < 0.3] = 0.0
            p /= p.sum()
        joints.append(Distribution.from_ndarray(p.reshape((2,) * (n_sources + 1))))
    return joints


TWO_SOURCE = {
    **{f"bivariates[{name!r}]": d for name, d in bivariates.items()},
    "dyadic": dyadic,
    "triadic": triadic,
    **{f"dense {i}": d for i, d in enumerate(_random_joints(2, 6, 2611081))},
    **{f"sparse {i}": d for i, d in enumerate(_random_joints(2, 6, 2611082, sparse=True))},
}
THREE_SOURCE = {
    **{f"trivariates[{name!r}]": d for name, d in trivariates.items()},
    **{f"dense {i}": d for i, d in enumerate(_random_joints(3, 6, 2611083))},
}


@pytest.mark.parametrize("measure,kind", [(PID_dep, "dep"), (PID_RA, "ra"), (PID_dep_c, "dep_c")])
@pytest.mark.parametrize("name", sorted(TWO_SOURCE))
def test_closed_form_matches_lattice_two_sources(measure, kind, name):
    """The closed forms reproduce the original dependency-lattice definitions."""
    d = TWO_SOURCE[name]
    sources = ((0,), (1,))
    actual = measure._measure(d, sources, (2,))
    expected = _lattice_uniques(d, sources, (2,), kind)
    for source in sources:
        assert actual[source] == pytest.approx(expected[source], abs=1e-5)


@pytest.mark.parametrize("measure,kind", [(PID_dep, "dep"), (PID_RA, "ra")])
@pytest.mark.parametrize("name", sorted(THREE_SOURCE))
def test_closed_form_matches_lattice_three_sources(measure, kind, name):
    """With more sources, each source is compared with the union of the others."""
    d = THREE_SOURCE[name]
    sources = ((0,), (1,), (2,))
    actual = measure._measure(d, sources, (3,))
    expected = _lattice_uniques(d, sources, (3,), kind)
    for source in sources:
        assert actual[source] == pytest.approx(expected[source], abs=1e-5)


def test_three_sources_without_dependency_lattice(monkeypatch):
    """The source-versus-rest triplets use one pairwise fit each, and no lattice."""
    fits = []

    def fit(d, rvs, **kwargs):
        fits.append(rvs)
        return maxent_dist(d, rvs, **kwargs)

    def no_lattice(*args, **kwargs):
        raise AssertionError("the closed forms must not construct the dependency lattice")

    monkeypatch.setattr(references_module, "maxent_dist", fit)
    monkeypatch.setattr(idep_module, "DependencyDecomposition", no_lattice)
    for measure in (PID_dep, PID_RA):
        fits.clear()
        measure._measure(trivariates["anddup"], ((0,), (1,), (2,)), (3,))
        assert len(fits) == 3
