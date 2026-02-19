"""
Shared backend parametrize list for optimization backend tests.

Usage in test files::

    from tests._backends import backends

    @pytest.mark.parametrize('backend', backends)
    def test_something(backend):
        result = some_function(..., backend=backend)
"""

import pytest


def _can_import(module_name):
    """Check whether *module_name* is importable."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


backends = [
    'numpy',
    pytest.param('jax', marks=[
        pytest.mark.skipif(not _can_import('jax'), reason="jax not installed"),
        pytest.mark.very_slow,
    ]),
    pytest.param('torch', marks=[
        pytest.mark.skipif(not _can_import('torch'), reason="torch not installed"),
        pytest.mark.very_slow,
    ]),
]
