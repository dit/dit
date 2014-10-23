# coding: utf-8

from __future__ import division
from __future__ import print_function

import dit
import numpy as np

def perturb_pmf(pmf, eps=.1, prng=None):
    """
    Returns a new distribution with all probabilities perturbed.

    Probabilities which are zero in the pmf cannot be perturbed by this method.
    All other probabilities are perturbed via the following process:

    0. Initial pmf ``p`` lives on the ``n``-simplex.
    1. Transform ``p`` via ilr (inverse logarithmic ratio) transform.
    2. Uniformly draw ``n`` random numbers between ``[0,1]``.
    3. Construct new transformed pmf: `p2_ilr = p1_ilr + eps * rand`
    4. Apply inverse ilr transformation.

    Practically, a large value of `eps` means that there is a better chance
    the perturbation will take the distribution closer to the simplex boundary.
    Large distributions (with more than 60 elements) fail, due to some
    underflow issue with the ilr transformation.

    Parameters
    ----------
    pmf : NumPy array
        The distribution to be perturbed.
    eps : float
        The scaling factor used for perturbing. Values of `10` correspond
        to large perturbations for the ``1``-simplex.
    prng : NumPy RandomState
        A random number generator.

    Returns
    -------
    out : NumPy array
        The perturbed distribution.

    """
    if prng is None:
        prng = dit.math.prng

    idx = pmf > 0
    p1 = pmf[idx]

    p1_ilr = dit.math.aitchison.ilr(p1)
    delta = eps * (prng.rand(len(p1_ilr)) - .5)
    p2_ilr = p1_ilr + delta
    p2 = dit.math.aitchison.ilr_inv(p2_ilr)

    out = np.zeros(len(pmf))
    out[idx] = p2

    return out

