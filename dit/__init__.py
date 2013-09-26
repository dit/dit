#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dit is a Python project for discrete-valued information theory.

"""

# Order is important!
from .params import ditParams
from .npscalardist import ScalarDistribution
from .npdist import Distribution

from .distconst import *

import dit.algorithms
