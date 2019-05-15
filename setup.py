#!/usr/bin/env python

import os
from os import path as op

from setuptools import setup

def package_tree(pkgroot):
    """Get the submodule list."""
    # adapted from mne-python
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)

setup(name='jumeg',
      version='0.18',
      description='MEG data analysis at FZJ',
      url='http://github.com/jdammers/jumeg',
      download_url='http://github.com/jdammers/jumeg',
      author='Praveen Sripad',
      author_email='pravsripad@gmail.com',
      license='BSD (3-clause)',
      packages=package_tree('jumeg'),
      package_data={'jumeg': [op.join('data', '*')]},
      zip_safe=False)
