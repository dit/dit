#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

# Global random number generator
import numpy as np
prng = np.random.RandomState()
# Set the error level to ignore...for example: log2(0).
np.seterr(all='ignore')
del np

from .equal import close, allclose
from .sampling import sample, _sample, _samples
from .ops import *
from .fraction import approximate_fraction
