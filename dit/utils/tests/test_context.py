from __future__ import unicode_literals

from nose.tools import *
import os
import time
from dit.utils import cd, named_tempfile, tempdir

def test_cd():
    with cd('/'):
        assert_equal(os.getcwd(), '/')

def test_named_tempfile():
    name = None
    with named_tempfile() as tempfile:
        name = tempfile.name
        assert_true(os.path.isfile(name))
        tempfile.write('hello'.encode('ascii'))
        tempfile.close()
        assert_true(os.path.isfile(name))
    assert_false(os.path.isfile(name))

def test_tempdir():
    name = None
    with tempdir() as tmpdir:
        assert_true(os.path.isdir(tmpdir))
    assert_false(os.path.isdir(tmpdir))
