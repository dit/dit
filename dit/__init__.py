#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dit is a Python package for information theory on discrete random variables.

d = discrete
i = information
t = theory

However, the more precise statement (at this point) is that `dit` is a
Python package for sigma-algebras defined on finite sets. Presently,
a few assumptions are made which make `dit` unsuitable as a general
sigma algebra package (on finite sets).  Some of these assumptions
deal with how the sample space and sigma algebras are formed from
the probability mass function (and its outcomes).

"""

# Order is important!
from .params import ditParams
from .npscalardist import ScalarDistribution
from .npdist import Distribution

from .distconst import *

import dit.algorithms
import dit.divergences
import dit.example_dists
import dit.inference
import dit.esoteric
import dit.multivariate
import dit.shannon
