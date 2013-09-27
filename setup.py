#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Installation script for Dit.

"""

import os
import sys
import warnings
from glob import glob

import distutils
from distutils.core import setup, Extension
from distutils.errors import DistutilsPlatformError
from distutils.command import install_data
from distutils.command.build_ext import build_ext

import numpy as np

class my_install_data(install_data.install_data):
    # A custom install_data command, which will install it's files
    # into the standard directories (normally lib/site-packages).
    def finalize_options(self):
        if self.install_dir is None:
            installobj = self.distribution.get_command_obj('install')
            self.install_dir = installobj.install_lib
        print 'Installing data files to %s' % self.install_dir
        install_data.install_data.finalize_options(self)

def has_cython():
    """Returns True if Cython is found on the system."""
    try:
        import Cython
        return True
    except ImportError:
        return False

def write_version():
    """Creates a file containing version information."""
    target = os.path.join(base, 'dit', 'version.py')
    fh = open(target, 'w')
    text = '''"""
Version information for Dit, created during installation.
"""

__version__ = '%s'

'''
    fh.write(text % release.version)
    fh.close()

def check_opt(name):
    exec('x = has_%s()' % name.lower())
    msg = "%(name)s not found. %(name)s extensions will not be built."
    if not x:
        warnings.warn(msg % {'name':name})
    return x

def hack_distutils(debug=False, fast_link=True):
    # hack distutils.sysconfig to eliminate debug flags
    # stolen from mpi4py

    def remove_prefixes(optlist, bad_prefixes):
        for bad_prefix in bad_prefixes:
            for i, flag in enumerate(optlist):
                if flag.startswith(bad_prefix):
                    optlist.pop(i)
                    break
        return optlist

    import sys
    if not sys.platform.lower().startswith("win"):
        from distutils import sysconfig

        cvars = sysconfig.get_config_vars()
        cflags = cvars.get('OPT')
        if cflags:
            cflags = remove_prefixes(cflags.split(),
                    ['-g', '-O', '-Wstrict-prototypes', '-DNDEBUG'])
            if debug:
                cflags.append("-g")
            else:
                cflags.append("-O3")
                cflags.append("-DNDEBUG")
            cvars['OPT'] = str.join(' ', cflags)
            cvars["CFLAGS"] = cvars["BASECFLAGS"] + " " + cvars["OPT"]

        if fast_link:
            for varname in ["LDSHARED", "BLDSHARED"]:
                ldsharedflags = cvars.get(varname)
                if ldsharedflags:
                    ldsharedflags = remove_prefixes(ldsharedflags.split(),
                            ['-Wl,-O'])
                    cvars[varname] = str.join(' ', ldsharedflags)

def main():
    hack_distutils()
    #write_version()

    opt = {}
    for name in ['Cython']:
        lname = name.lower()
        try:
            idx = sys.argv.index("--{0}".format(lname))
        except ValueError:
            opt[lname] = False
        else:
            opt[lname] = check_opt(name)
            del sys.argv[idx]

    cmdclass = {'install_data': my_install_data}

    cython_modules = []
    if opt['cython']:
        import Cython.Distutils
        cmdclass['build_ext'] = Cython.Distutils.build_ext

        cython_modules = []

        close = Extension(
            "dit.math._close",
            ["dit/math/_close.pyx"]
        )

        samplediscrete = Extension(
            "dit.math._samplediscrete",
            ["dit/math/_samplediscrete.pyx"],
            include_dirs=[np.get_include()]
        )

        # Active Cython modules
        cython_modules = [
            close,
            samplediscrete,
        ]

    other_modules = []

    ext_modules = cython_modules + \
                  other_modules

    data_files = ()

    requires = [
        'numpy(>1.6)',
        'networkx(>1.6)',
    ]

    packages = [
        'dit',
        'dit.algorithms',
        'dit.math',
        'dit.utils',
    ]

    setup(
          name             = "Dit",
          provides         = ['dit'],
          version          = "0.0.1dev", # need to use 'git describe'
          author           = "The World",
          author_email     = "",
          description      = "Discrete Information Theory",
          keywords         = "",
          long_description = open('README.md').read(),
          license          = "LICENSE.txt",
          platforms        = "",
          url              = "",
          download_url     = "",
          packages         = packages,
          ext_modules      = ext_modules,
          cmdclass         = cmdclass,
          data_files       = data_files,
          classifiers      = ""
         )



if __name__ == '__main__':
    if sys.argv[-1] == 'setup.py':
        print "To install, run 'python setup.py install'\n"

    v = sys.version_info[:2]
    if v < (2, 6) and v >= (3,0):
        msg = "Dit requires Python version >2.6 and <3.0 (%d.%d detected)."
        print msg % v
        sys.exit(-1)

    main()
