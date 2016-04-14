"""
Useful context managers.
"""

import contextlib
import os
import shutil
import tempfile

from .bindargs import bindcallargs

@contextlib.contextmanager
def cd(newpath):
    """
    Change the current working directory to `newpath`, temporarily.

    If the old current working directory no longer exists, do not return back.
    """
    oldpath = os.getcwd()
    os.chdir(newpath)
    try:
        yield
    finally:
        try:
            os.chdir(oldpath)
        except OSError:
            # If oldpath no longer exists, stay where we are.
            pass

@contextlib.contextmanager
def named_tempfile(*args, **kwargs):
    """
    Calls tempfile.NamedTemporaryFile(*kwargs) with delete=False.

    The file is deleted after the context.

    """
    args, kwargs = bindcallargs(tempfile.NamedTemporaryFile, *args, **kwargs)
    # Override any specification for delete=True.
    # Note: delete is the last argument.
    args = list(args)
    args[-1] = False

    #mode = args[0]
    ntf = tempfile.NamedTemporaryFile(*args, **kwargs)
    try:
        yield ntf
    finally:
        ntf.close()
        os.remove(ntf.name)

@contextlib.contextmanager
def tempdir(*args, **kwargs):
    """
    Calls tempfile.mkdtemp() and deletes the directory after the context.

    """
    tmpdir = tempfile.mkdtemp(*args, **kwargs)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)
