#!/usr/bin/env python
from os.path import exists
from setuptools import setup

DISTNAME = 'qg_ssh'
PACKAGES = ['qg']
TESTS = [p + '.tests' for p in PACKAGES]

TESTS_REQUIRE = ['pytest >= 2.7.1']

AUTHOR = 'Marine Roge'
LICENSE = 'MIT'
DESCRIPTION = 'Advection of SSH using a 1.5 QG model'

VERSION = '0.1'

setup(name=DISTNAME,
      version=VERSION,
      scripts=['bin/run_submission.py', 'bin/run_submission_diff.py'],
      description=DESCRIPTION,
      author=AUTHOR,
      license=LICENSE,
      packages=PACKAGES + TESTS,
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      zip_safe=False)
