"""
Uniform access for ExitStack context manager.

"""
import sys

if sys.version_info[:2] >= (3,3): # pragma: no cover
    from contextlib import ExitStack
else: # pragma: no cover
    from contextlib2 import ExitStack
