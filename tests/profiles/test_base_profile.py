"""
Tests for dit.profiles.base_profile.BaseProfile (via the concrete
ComplexityProfile subclass).
"""

from dit import Distribution
from dit.profiles import ComplexityProfile


def test_to_string():
    """to_string renders a table with the profile's name and values."""
    # The uniform distribution's profile has near-zero entries, exercising the
    # -0.0 cleanup branch.
    d = Distribution(["000", "001", "010", "011", "100", "101", "110", "111"], [1 / 8] * 8)
    cp = ComplexityProfile(d)
    out = cp.to_string()
    assert "Complexity Profile" in out
    assert "bits" in out


def test_str_matches_to_string():
    """__str__ delegates to to_string."""
    d = Distribution(["000", "111"], [1 / 2] * 2)
    cp = ComplexityProfile(d)
    assert str(cp) == cp.to_string()


def test_nonlinear_base_unit():
    """A log-base distribution sets the profile unit from its base."""
    d = Distribution(["00", "11"], [1 / 2] * 2)
    d.set_base("e")
    cp = ComplexityProfile(d)
    assert cp.unit == "nats"
    assert cp.ylabel == "information [nats]"
