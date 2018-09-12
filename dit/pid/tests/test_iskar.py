"""
Tests for dit.pid.iskar.
"""

import pytest

from dit.pid.iskar import (PID_SKAR_nw,
                           PID_SKAR_owa,
                           PID_SKAR_owb,
                           PID_SKAR_tw,
                           )
from dit.pid.distributions import bivariates, trivariates
