#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dit is a Python package for information theory on discrete random variables.

d = discrete
i = information
t = theory

However, the more precise statement (at this point) is that `dit` is a
Python package for sigma-algebras defined on finite sets.

"""

__version__ = "1.0.0.dev27"

# Order is important!
from .params import ditParams
from .npscalardist import ScalarDistribution
from .npdist import Distribution

# Order does not matter for these
from .samplespace import ScalarSampleSpace, SampleSpace, CartesianProduct
from .distconst import *
from .bgm import *
from .helpers import copypmf
from .cdisthelpers import joint_from_factors
from .algorithms import pruned_samplespace, expanded_samplespace

import dit.algorithms
import dit.divergences
import dit.example_dists
import dit.inference
import dit.other
import dit.pid
import dit.profiles
import dit.multivariate
import dit.shannon
