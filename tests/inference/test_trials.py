"""
Tests for multi-trial (Trials) input across dit.inference.
"""

from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dit.inference import (
    Trials,
    block_entropy,
    block_surrogates,
    dist_from_timeseries,
    distribution_from_data,
    entropy_from_counts,
    markov_order_test,
    select_markov_order,
    shift_surrogates,
    stationary_bootstrap,
    transfer_entropy,
    transfer_entropy_test,
    whittle_count,
    whittle_surrogates,
)

trials_strategy = st.lists(st.lists(st.integers(0, 2), min_size=1, max_size=30), min_size=1, max_size=4)


def _pooled_words(trials, L):
    counts = Counter()
    for t in trials:
        counts.update(tuple(t[i : i + L]) for i in range(len(t) - L + 1))
    return counts


@settings(max_examples=40, deadline=None)
@given(trials=trials_strategy, L=st.integers(1, 3))
def test_pooled_counts_are_sum_of_trials(trials, L):
    """
    Pooled word counts equal the sum of per-trial counts; no word crosses a boundary.
    """
    counts = _pooled_words(trials, L)
    expected = entropy_from_counts(list(counts.values())) if counts else 0.0
    assert block_entropy(Trials(trials), L) == pytest.approx(expected)
    if counts:
        d = distribution_from_data(Trials(trials), L, base="linear")
        total = sum(counts.values())
        for word, c in counts.items():
            outcome = word[0] if L == 1 else word
            assert d[outcome] == pytest.approx(c / total)


def test_boundary_words_excluded():
    d = distribution_from_data(Trials([[0, 0], [1, 1]]), 2)
    assert set(d.outcomes) == {(0, 0), (1, 1)}
    assert (0, 1) in distribution_from_data([0, 0, 1, 1], 2).outcomes


def test_single_trial_matches_sequence():
    x = np.random.default_rng(0).integers(0, 3, 200)
    assert block_entropy(Trials([x]), 3) == pytest.approx(block_entropy(x, 3))
    assert markov_order_test(Trials([x]), 1, method="asymptotic").pvalue == pytest.approx(
        markov_order_test(x, 1, method="asymptotic").pvalue
    )


def test_whittle_trials_preserve_each_trial():
    trials = Trials([[0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0]])
    assert whittle_count(trials, 1) == pytest.approx(whittle_count(trials[0], 1) + whittle_count(trials[1], 1))
    for draw in whittle_surrogates(trials, 1, n=10, prng=0):
        assert isinstance(draw, Trials)
        for s, t in zip(draw, trials, strict=True):
            assert s[0] == t[0]
            assert _pooled_words([list(s)], 2) == _pooled_words([t], 2)


def test_shift_and_block_trials():
    trials = Trials([np.arange(6), np.arange(10, 14)])
    for draw in shift_surrogates(trials, n=5, prng=0) + block_surrogates(trials, 2, n=5, prng=0):
        assert [sorted(s) for s in draw] == [sorted(t) for t in trials]


def test_markov_order_many_short_trials():
    """
    Many short trials of a first-order chain: pooled evidence identifies order 1.
    """
    rng = np.random.default_rng(1)
    trials = []
    for _ in range(40):
        x = [int(rng.integers(2))]
        for _ in range(29):
            x.append(x[-1] if rng.random() < 0.85 else 1 - x[-1])
        trials.append(x)
    trials = Trials(trials)
    assert markov_order_test(trials, 0, n_surrogates=99, prng=0).pvalue < 0.05
    assert select_markov_order(trials, 3, method="bic") == 1
    assert select_markov_order(trials, 3, n_surrogates=99, prng=0) == 1


def test_transfer_entropy_trials():
    rng = np.random.default_rng(2)
    xs, ys = [], []
    for _ in range(20):
        x = rng.integers(0, 2, 50)
        y = np.concatenate([[0], x[:-1]])
        xs.append(x)
        ys.append(y)
    te = transfer_entropy(Trials(xs), Trials(ys))
    assert te == pytest.approx(1.0, abs=0.05)
    assert transfer_entropy_test(Trials(xs), Trials(ys), null="whittle", n_surrogates=19, prng=0).pvalue <= 0.05
    with pytest.raises(ValueError):
        transfer_entropy(Trials(xs), ys[0])


def test_stationary_bootstrap_trials():
    trials = Trials([[0], [1], [2]])
    draws = stationary_bootstrap(trials, n=20, prng=0)
    assert all(isinstance(d, Trials) and len(d) == 3 for d in draws)
    assert {tuple(t) for d in draws for t in d} <= {(0,), (1,), (2,)}


def test_dist_from_timeseries_trials():
    trials = Trials([[0, 1, 0, 1], [1, 1, 1]])
    d = dist_from_timeseries(trials, history_length=1, base="linear")
    assert set(d.outcomes) == {(0, 1), (1, 0), (1, 1)}
    assert d[(1, 1)] == pytest.approx(2 / 5)


def test_undersampling_warning():
    from dit.inference import UndersamplingWarning, conditional_entropy_rate

    rng = np.random.default_rng(3)
    x = rng.integers(0, 4, 100)
    with pytest.warns(UndersamplingWarning):
        block_entropy(x, 4)
    with pytest.warns(UndersamplingWarning):
        conditional_entropy_rate(x, 3)
    with pytest.warns(UndersamplingWarning):
        entropy_from_counts(np.ones(50))
    y = rng.integers(0, 2, 5000)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UndersamplingWarning)
        block_entropy(y, 3)
        markov_order_test(y, 1, method="asymptotic")


def test_effective_sample_sizes():
    x = np.random.default_rng(4).integers(0, 2, 300)
    assert markov_order_test(x, 2, method="asymptotic").n_windows == 300 - 3
    assert markov_order_test(Trials([x[:100], x[100:]]), 2, method="asymptotic").n_windows == 300 - 6
    result = transfer_entropy_test(x, np.roll(x, 1), history_length=2, n_surrogates=5, prng=0)
    assert result.n_samples == 300 - 2


def test_distribution_from_data_observed_words_only():
    """
    With trim=True only observed words are built; the result matches trim=False.
    """
    x = np.random.default_rng(5).integers(0, 20, 2000)
    d = distribution_from_data(x, 3, base="linear")
    assert len(d.outcomes) == len(set(map(tuple, np.lib.stride_tricks.sliding_window_view(x, 3))))
    full = distribution_from_data(x, 3, trim=False, base="linear")
    assert d.is_approx_equal(full)


def test_count_level_estimators_handle_huge_word_spaces():
    x = np.random.default_rng(6).integers(0, 50, 2000)
    with pytest.warns(UserWarning):
        assert block_entropy(x, 12) == pytest.approx(np.log2(2000 - 11))
