"""
Tests for dit.pid.irav.
"""

import pytest

from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.irav import PID_RAV


def test_pid_rav1():
    """
    Test irav on a generic distribution.
    """
    d = bivariates["diff"]
    pid = PID_RAV(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.12255624891826589)
    assert pid[((0,),)] == pytest.approx(0.18872187554086706)
    assert pid[((1,),)] == pytest.approx(0.18872187554086706)
    assert pid[((0, 1),)] == pytest.approx(0)


def test_pid_rav2():
    """
    Test irav on a trivariate distribution based on a bivariate one.
    """
    d = trivariates["anddup"]
    pid = PID_RAV(d, [[0], [1], [2]], [3])
    d_and = bivariates["and"]
    pid_and = PID_RAV(d_and, [[0], [1]], [2])
    assert pid[((0, 1), (1, 2))] == pytest.approx(pid_and[((0, 1),)])
    assert pid[((0,), (2,))] == pytest.approx(pid_and[((0,),)])
    assert pid[((1,),)] == pytest.approx(pid_and[((1,),)])
    assert pid[((0,), (1,), (2,))] == pytest.approx(pid_and[((0,), (1,))])
    for atom in pid._lattice:
        if atom not in [((0, 1), (1, 2)), ((0,), (2,)), ((1,),), ((0,), (1,), (2,))]:
            assert pid[atom] == pytest.approx(0.0)


def test_pid_rav3():
    """
    Test irav on a generic trivariate distribution.
    """
    d = trivariates["xor shared"]
    pid = PID_RAV(d)
    for atom in pid._lattice:
        if atom == ((0,), (1, 2)):
            assert pid[atom] == pytest.approx(1.0)
        else:
            assert pid[atom] == pytest.approx(0.0)


def _rav_brute_force(d, sources, target):
    """
    The former implementation: one extended distribution, and its coinformation, per partition.
    """
    from dit.distconst import RVFunctions, insert_rvf
    from dit.multivariate import coinformation
    from dit.utils import extended_partition, partitions

    d = d.coalesce(tuple(sources) + (tuple(target),))
    source_parts = partitions(d.marginal(sum(d.rvs[:-1], [])).outcomes)
    idxs = list(range(len(sources)))
    parts = [extended_partition(d.outcomes, idxs, part, d._outcome_ctor) for part in source_parts]
    bf = RVFunctions(d)
    return max(coinformation(insert_rvf(d, bf.from_partition(part))) for part in parts)


@pytest.mark.parametrize("n", range(1, 9))
def test_partition_batches(n):
    """
    The batches hold every partition of n items exactly once (the Bell numbers).
    """
    import numpy as np

    from dit.pid.measures.irav import _partition_batches

    rows = np.vstack(list(_partition_batches(n, batch=7)))
    bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140]
    assert len(rows) == bell[n]
    canonical = {tuple(np.unique(r, return_inverse=True)[1]) for r in rows}
    assert len(canonical) == bell[n]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("sources", [[[0], [1]], [[0], [1], [2]], [[0, 1], [1, 2]]])
def test_pid_rav_brute_force(seed, sources):
    """
    The batched computation agrees with one coinformation per partition.
    """
    import numpy as np

    from dit import Distribution

    n_in = max(max(s) for s in sources) + 1
    rng = np.random.default_rng(seed)
    shape = (2,) * n_in + (3,)
    p = rng.dirichlet(np.full(int(np.prod(shape)), 0.7)).reshape(shape)
    # keep at most 5 joint input values (52 partitions), so that the brute force stays quick
    cells = list(np.ndindex(*shape[:-1]))
    for i in rng.permutation(len(cells))[5:]:
        p[cells[i]] = 0
    p[p < np.quantile(p[p > 0], 0.2)] = 0
    p /= p.sum()
    outcomes = [o for o in np.ndindex(*shape) if p[o] > 0]
    d = Distribution(outcomes, [p[o] for o in outcomes])
    pid = PID_RAV(d, sources, [n_in])
    red = tuple(tuple(s) for s in sources)
    assert pid.get_red(red) == pytest.approx(_rav_brute_force(d, sources, [n_in]), abs=1e-10)
