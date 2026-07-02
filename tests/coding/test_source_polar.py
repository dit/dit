"""
Tests for dit.coding source polarization and the polar source code.
"""

import itertools

import numpy as np
import pytest

from dit import Distribution as D
from dit.coding import (
    PolarSourceCode,
    polar_source,
    source_bhattacharyya,
    source_high_entropy_set,
    source_polarization_profile,
)
from dit.coding._util import polar_transform
from dit.exceptions import ditException
from dit.shannon import conditional_entropy, entropy


def dsbs(a):
    """A doubly-symmetric binary source (X, Y) with crossover probability ``a``."""
    return D(["00", "01", "10", "11"], [(1 - a) / 2, a / 2, a / 2, (1 - a) / 2])


# ── shared transform ────────────────────────────────────────────────────────


def test_polar_transform_is_involution():
    """The Arikan transform over GF(2) is its own inverse."""
    prng = np.random.default_rng(0)
    for n in (1, 2, 4, 8):
        for _ in range(20):
            v = [int(b) for b in prng.integers(0, 2, n)]
            assert polar_transform(polar_transform(v)) == v


# ── source Bhattacharyya ────────────────────────────────────────────────────


def test_source_bhattacharyya_unconditional():
    """Z(X) = 2 sqrt(p0 p1) for a plain Bernoulli source."""
    p = 0.2
    Z = source_bhattacharyya(D(["0", "1"], [p, 1 - p]))
    assert pytest.approx(2 * np.sqrt(p * (1 - p))) == Z


def test_source_bhattacharyya_uniform_is_one():
    """A uniform bit has Z = 1."""
    assert source_bhattacharyya(D(["0", "1"], [0.5, 0.5])) == pytest.approx(1.0)


def test_source_bhattacharyya_deterministic_is_zero():
    """X determined by Y has Z(X | Y) = 0."""
    assert source_bhattacharyya(D(["00", "11"], [0.5, 0.5]), rv=0, crvs=[1]) == pytest.approx(0.0)


def test_source_bhattacharyya_conditioning_helps():
    """Side information cannot raise the source Bhattacharyya parameter."""
    d = dsbs(0.1)
    assert source_bhattacharyya(d, rv=0, crvs=[1]) < source_bhattacharyya(d, rv=0)


def test_source_bhattacharyya_requires_binary():
    """A non-binary source variable raises."""
    with pytest.raises(ditException):
        source_bhattacharyya(D(["0", "1", "2"], [1 / 3, 1 / 3, 1 / 3]))


# ── polarization profile ────────────────────────────────────────────────────


def test_profile_entropy_conservation():
    """Conditional entropies sum to N * H(X | Y)."""
    d = dsbs(0.1)
    N = 4
    profile = source_polarization_profile(d, N, rv=0, crvs=[1])
    total = sum(row["entropy"] for row in profile)
    assert total == pytest.approx(N * conditional_entropy(d, [0], [1]))


def test_profile_entropy_conservation_no_side_info():
    """Without side information the entropies sum to N * H(X)."""
    d = D(["0", "1"], [0.11, 0.89])
    N = 8
    profile = source_polarization_profile(d, N)
    assert sum(row["entropy"] for row in profile) == pytest.approx(N * entropy(d))


def test_profile_length_two_inequalities():
    """H(U1 | Y) >= H(X | Y) >= H(U2 | U1, Y) for a single polarization step."""
    d = dsbs(0.1)
    profile = source_polarization_profile(d, 2, rv=0, crvs=[1])
    HXgY = conditional_entropy(d, [0], [1])
    assert profile[0]["entropy"] >= HXgY - 1e-12
    assert HXgY >= profile[1]["entropy"] - 1e-12


def test_profile_entropies_in_unit_interval():
    """Every conditional entropy lies in [0, 1]."""
    profile = source_polarization_profile(dsbs(0.1), 4, rv=0, crvs=[1])
    assert all(0.0 <= row["entropy"] <= 1.0 + 1e-12 for row in profile)


def test_profile_polarizes_with_block_length():
    """Longer blocks push more mass toward the extremes."""

    def extremes(N):
        profile = source_polarization_profile(D(["0", "1"], [0.11, 0.89]), N)
        return sum(1 for r in profile if r["entropy"] < 0.1 or r["entropy"] > 0.9) / N

    assert extremes(8) >= extremes(2)


def test_profile_max_correlation_diagnostic():
    """The Goela max-correlation diagnostic is present, in [0, 1], zero at i=0."""
    profile = source_polarization_profile(
        dsbs(0.1), 4, rv=0, crvs=[1], metrics=("max_correlation_with_past",)
    )
    assert profile[0]["max_correlation_with_past"] == pytest.approx(0.0)
    assert all(-1e-9 <= row["max_correlation_with_past"] <= 1.0 + 1e-9 for row in profile)


def test_profile_requires_power_of_two():
    """A non-power-of-two block length raises."""
    with pytest.raises(ditException):
        source_polarization_profile(dsbs(0.1), 3, rv=0, crvs=[1])


def test_profile_unknown_metric_raises():
    """An unknown metric name raises."""
    with pytest.raises(ditException):
        source_polarization_profile(dsbs(0.1), 2, metrics=("nonsense",))


# ── high-entropy set ────────────────────────────────────────────────────────


def test_high_entropy_set_lossless_default():
    """The default set keeps exactly the non-deterministic coordinates."""
    d = dsbs(0.05)
    N = 8
    profile = source_polarization_profile(d, N, rv=0, crvs=[1])
    expected = sorted(i for i in range(N) if profile[i]["entropy"] > 1e-9)
    assert source_high_entropy_set(d, N, rv=0, crvs=[1]) == expected


def test_high_entropy_set_size():
    """An explicit size selects exactly that many indices."""
    indices = source_high_entropy_set(dsbs(0.1), 8, size=3, rv=0, crvs=[1])
    assert len(indices) == 3
    assert indices == sorted(indices)


def test_high_entropy_set_rate():
    """A rate selects round(rate * N) indices."""
    assert len(source_high_entropy_set(dsbs(0.1), 8, rate=0.5, rv=0, crvs=[1])) == 4


def test_high_entropy_set_rejects_both_rate_and_size():
    """Giving both rate and size raises."""
    with pytest.raises(ditException):
        source_high_entropy_set(dsbs(0.1), 4, rate=0.5, size=2)


def test_high_entropy_set_ranks_by_metric():
    """The selected size-1 index is the highest-entropy coordinate."""
    d = dsbs(0.1)
    N = 4
    profile = source_polarization_profile(d, N, rv=0, crvs=[1])
    top = max(range(N), key=lambda i: profile[i]["entropy"])
    assert source_high_entropy_set(d, N, size=1, rv=0, crvs=[1]) == [top]


# ── PolarSourceCode ─────────────────────────────────────────────────────────


def test_code_is_source_coding():
    """The polar source code is a SourceCoding instance."""
    from dit.coding import SourceCoding

    code = polar_source(D(["0", "1"], [0.3, 0.7]), 4)
    assert isinstance(code, SourceCoding)
    assert isinstance(code, PolarSourceCode)


def test_code_lossless_roundtrip_no_side_info():
    """Every block round-trips exactly under the lossless default."""
    code = polar_source(D(["0", "1"], [0.11, 0.89]), 4)
    for block in itertools.product((0, 1), repeat=4):
        assert code.decode(code.encode(block)) == list(block)


def test_code_rate_zero_when_determined_by_side_info():
    """X = Y gives a rate-zero code recovered from side information alone."""
    code = polar_source(D(["00", "11"], [0.5, 0.5]), 4, rv=0, crvs=[1])
    assert code.rate() == 0.0
    for block in itertools.product((0, 1), repeat=4):
        side = [str(b) for b in block]
        assert code.decode(code.encode(block), side_information=side) == list(block)


def test_code_lossless_roundtrip_with_side_info():
    """A correlated source round-trips exactly with matched side information."""
    code = polar_source(dsbs(0.05), 8, rv=0, crvs=[1])
    prng = np.random.default_rng(0)
    for _ in range(16):
        block = [int(b) for b in prng.integers(0, 2, 8)]
        side = [str(b) for b in block]
        assert code.decode(code.encode(block), side_information=side) == block


def test_code_rate_matches_high_entropy_set():
    """rate == |high_entropy_set| / block_length."""
    code = polar_source(dsbs(0.05), 8, rv=0, crvs=[1])
    assert code.rate() == pytest.approx(len(code.high_entropy_set) / 8)


def test_code_rate_below_one_when_side_info_helps():
    """Strong side information yields a rate below one."""
    code = polar_source(dsbs(0.001), 8, rv=0, crvs=[1])
    assert code.rate() < 1.0


def test_code_accepts_alphabet_symbols():
    """Encoding accepts the source's own binary alphabet, not just 0/1 ints."""
    code = polar_source(D(["a", "b"], [0.3, 0.7]), 4)
    d = code.dist
    zero = d.alphabet[0][0]
    one = d.alphabet[0][1]
    block = [zero, one, one, zero]
    # Encoding by symbol matches encoding by the corresponding bits.
    assert code.encode(block) == code.encode([0, 1, 1, 0])


def test_code_requires_power_of_two():
    """A non-power-of-two block length raises."""
    with pytest.raises(ditException):
        polar_source(D(["0", "1"], [0.3, 0.7]), 3)


def test_code_max_states_guard():
    """The state guard trips before enumerating a large joint table."""
    with pytest.raises(ditException):
        polar_source(D(["0", "1"], [0.3, 0.7]), 8, max_states=10)


def test_code_encode_wrong_length_raises():
    """Encoding a wrong-length block raises."""
    code = polar_source(D(["0", "1"], [0.3, 0.7]), 4)
    with pytest.raises(ditException):
        code.encode([0, 1])


def test_code_decode_missing_side_info_raises():
    """Decoding without required side information raises."""
    code = polar_source(dsbs(0.1), 4, rv=0, crvs=[1])
    encoded = code.encode([0, 0, 0, 0])
    with pytest.raises(ditException):
        code.decode(encoded)


def test_code_decode_wrong_side_info_length_raises():
    """Wrong-length side information raises."""
    code = polar_source(dsbs(0.1), 4, rv=0, crvs=[1])
    encoded = code.encode([0, 0, 0, 0])
    with pytest.raises(ditException):
        code.decode(encoded, side_information=["0"])


def test_code_decode_extra_side_info_raises():
    """Passing side information to a plain code raises."""
    code = polar_source(D(["0", "1"], [0.3, 0.7]), 4)
    encoded = code.encode([0, 0, 0, 0])
    with pytest.raises(ditException):
        code.decode(encoded, side_information=["0", "0", "0", "0"])


def test_code_message_length_matches_encoding():
    """message_length equals the number of encoded bits."""
    code = polar_source(dsbs(0.05), 8, rv=0, crvs=[1])
    assert code.message_length == len(code.encode([0] * 8))
