#!/usr/bin/env python

import os
import os.path as op

from setuptools import setup

def package_tree(pkgroot):
    """Get the submodule list."""
    # adapted from mne-python
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
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
