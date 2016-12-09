"""
Canonical non-signalling boxes.
"""

from __future__ import division

from dit import Distribution

from itertools import product

__all__ = ['pr_box']

def pr_box(eta=1, name=False):
    """
    The Popescu-Rohrlich box, or PR box, is the canonical non-signalling, non-local probability
    distribution used in the study of superquantum correlations. It has two space-like seperated
    inputs, X and Y, and two associated outputs, A and B.

    `eta` is the noise level of this correlation. For 0 <= eta <= 1/2 the box can be realized
    classically. For 1/2 < eta <= 1/sqrt(2) the box can be realized quantum-mechanically.

    Parameters
    ----------
    eta : float, 0 <= eta <= 1
        The noise level of the box. Defaults to 1.

    name : bool
        Whether to set rv names or not. Defaults to False.

    Returns
    -------
    pr : Distribution
        The PR box distribution.
    """
    outcomes = list(product([0, 1], repeat=4))
    pmf = [ ((1+eta)/16 if (x*y == a^b) else (1-eta)/16) for x, y, a, b in outcomes ]
    pr = Distribution(outcomes, pmf)

    if name:
        pr.set_rv_names("XYAB")

    return pr
