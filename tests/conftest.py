"""
Configuration for tests.
"""

import os

# xdist already uses one process per core; nested optimizer thread pools and
# OpenBLAS threads inside each worker oversubscribe the CPU. Must run before
# numpy is imported.
for var in ("DIT_OPT_JOBS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

from hypothesis import settings  # noqa: E402

settings.register_profile("dit", deadline=None)
settings.load_profile("dit")
