"""
Tests for dit.utils.context.
"""

from __future__ import unicode_literals

import sys

from nose.tools import assert_equal, assert_false, assert_true
import os
from dit.utils import cd, named_tempfile, tempdir

import tempfile

root = 'C:\\' if sys.platform in ('win32', 'cygwin') else '/'

def test_cd():
    with cd(root):
        assert_equal(os.getcwd(), root)

def test_cd_bad_oldcwd():
    # Test attempting to go back to a directory that no longer exists.
    name = tempfile.mkdtemp()
    with cd(name):
        assert_equal(os.getcwd(), os.path.realpath(name))
        with cd(root):
            assert_equal(os.getcwd(), root)
            os.rmdir(name)
        assert_equal(os.getcwd(), root)

def test_named_tempfile1():
    name = None
    with named_tempfile() as tempfile:
        name = tempfile.name
        assert_true(os.path.isfile(name))
        tempfile.write('hello'.encode('utf8'))
        tempfile.close()
        assert_true(os.path.isfile(name))
    assert_false(os.path.isfile(name))

def test_named_tempfile2():
    name = None
    # The specification of delete=True should be ignored.
    with named_tempfile(delete=True) as tempfile:
        name = tempfile.name
        assert_true(os.path.isfile(name))
        tempfile.write('hello'.encode('utf8'))
        tempfile.close()
        assert_true(os.path.isfile(name))
    assert_false(os.path.isfile(name))


def test_tempdir():
    with tempdir() as tmpdir:
        assert_true(os.path.isdir(tmpdir))
    assert_false(os.path.isdir(tmpdir))
