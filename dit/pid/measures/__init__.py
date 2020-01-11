# -*- coding: utf-8 -*-

"""
All the measures.
"""

from .imin import PID_WB
from .iproj import PID_Proj
from .iwedge import PID_GK
from .immi import PID_MMI
from .ibroja import PID_BROJA
from .iccs import PID_CCS
from .idep import PID_dep, PID_RA
from .iskar import PID_SKAR_owb
from .ipm import PID_PM
from .irav import PID_RAV
from .irr import PID_RR


__all_pids = [
    PID_MMI,
    PID_GK,
    PID_WB,
    PID_RR,
    PID_CCS,
    PID_PM,
    PID_Proj,
    PID_BROJA,
    PID_dep,
    PID_RA,
    PID_RAV,
    PID_SKAR_owb,
]
