"""
Tests for dit.pid.iig.
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.iig import PID_IG, ig_synergy


def test_pid_ig1():
    """
    Test iproj on a generic distribution.
    """
    d = bivariates["and"]
    pid = PID_IG(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.08283, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.22845, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.22845, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.27155, abs=1e-4)


def test_pid_proj2():
    """
    Test iproj on another generic distribution.
    """
    d = bivariates["reduced or"]
    pid = PID_IG(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(-0.03122, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.34250, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.34250, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.34622, abs=1e-4)


def test_ig_synergy_requires_two_sources():
    """ig_synergy is bivariate-only and rejects a single source."""
    d = bivariates["and"]
    with pytest.raises(ValueError, match="exactly 2 sources"):
        ig_synergy(d, [[0]], [2])


def test_ig_synergy_without_fuzz():
    """ig_synergy runs with fuzz disabled on a fully-supported distribution.

    fuzz exists to perturb structural zeros, so disabling it requires a
    distribution with full support to keep the optimization well-defined.
    """
    import numpy as np

    from dit import Distribution

    weights = np.arange(1, 9, dtype=float)
    weights /= weights.sum()
    outcomes = [f"{a}{b}{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    d = Distribution(outcomes, list(weights))

    syn_no_fuzz = ig_synergy(d, [[0], [1]], [2], fuzz=0)
    syn_fuzz = ig_synergy(d, [[0], [1]], [2])
    assert syn_no_fuzz == pytest.approx(syn_fuzz, abs=1e-4)
