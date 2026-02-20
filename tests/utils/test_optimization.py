"""
Tests for dit.utils.optimization.
"""

from dit.utils.optimization import Uniquifier


def test_unq1():
    """
    Test uniquifier.
    """
    unq = Uniquifier()
    x = [unq(i) for i in (0, 0, 0, 2, 1)]
    assert x == [0, 0, 0, 1, 2]


def test_unq2():
    """
    Test uniquifier with strings.
    """
    unq = Uniquifier()
    x = [unq(i, string=True) for i in (0, 0, 0, "pants", 1)]
    assert x == ["0", "0", "0", "1", "2"]
