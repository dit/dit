"""
Utilities related to testing.
"""

from hypothesis import assume
from hypothesis.strategies import composite, floats, integers, lists, tuples

from .. import Distribution

@composite
def distributions(draw, size=integers(3, 4), alphabet=integers(2, 4), uniform=False):
    """
    A hypothesis strategy for generating distributions.

    Parameters
    ----------
    draw : function
        A sampling function passed in by hypothesis.
    size : int
        The size of outcomes desired. Defaults to a 3 or 4, randomly.
    alphabet : int
        The alphabet size for each variable. Defaults to 2, 3, or 4, randomly.
    uniform : bool
        Whether the probabilities should be uniform or random. Defaults to random.

    Returns
    -------
    dist : Distribution
        A random distribution.
    """
    try:
        size_ = draw(size)
    except:
        size_ = size
    try:
        alphabet_ = draw(alphabet)
    except:
        alphabet_ = alphabet

    events = draw(lists(tuples(*[integers(0, alphabet_ - 1)] * size_), min_size=1, unique=True))

    if uniform:
        probs = [1 / len(events)] * len(events)
    else:
        probs = draw(tuples(*[floats(0, 1)] * len(events)))
        assume(sum(probs) > 0)
        total = sum(probs)
        probs = [p / total for p in probs]

    return Distribution(events, probs)
