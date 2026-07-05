"""
Empirical distributions derived from real-world datasets.

Unlike the parametric and hand-constructed example distributions, these are
estimated from published data. Each constructor fetches its source data at call
time and returns the empirical joint distribution.
"""

from .titanic import titanic
