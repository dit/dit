#!/usr/bin/env python

"""
This script can be used to build the website.

It is also run on each commit to github.

Example:  ./build public_html
"""

import os
import shutil
import subprocess
import sys
import time

BUILD_DIR = 'build'


def get_build_dir():
    try:
        build_dir = sys.argv[1]
    except IndexError:
        build_dir = BUILD_DIR

    basedir = os.path.abspath(os.path.curdir)
    build_dir = os.path.join(basedir, build_dir)
    return build_dir


def build(dest):
    source = os.path.split(os.path.abspath(__file__))[0]
    source = os.path.join(source, 'src')

    # We aren't doing anything fancy yet.
    shutil.copytree(source, dest)


def update_gitrepo():
    source = os.path.split(os.path.abspath(__file__))[0]
    initial = os.getcwd()
    try:
        os.chdir(source)
        subprocess.call(['git', 'pull'])
    finally:
        os.chdir(initial)


def main():
    try:
        min_delay = int(sys.argv[2]) * 60
    except IndexError:
        min_delay = 0

    # Build only if enough time has passed.
    build_dir = get_build_dir()
    if os.path.exists(build_dir):
        elapsed = time.time() - os.path.getmtime(build_dir)
        if elapsed < min_delay:
            print("Not enough time has elapsed since last build.")
            sys.exit(0)
        else:
            # Delete it all!
            if os.path.islink(build_dir):
                os.unlink(build_dir)
            else:
                shutil.rmtree(build_dir)
    elif os.path.islink(build_dir):
        # Then its a bad symlink.
        os.unlink(build_dir)

    # update_gitrepo()
    build(build_dir)
    subprocess.call(['touch', build_dir])
    print("Done.\n")


if __name__ == '__main__':
    main()
