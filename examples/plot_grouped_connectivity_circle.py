#!/usr/bin/env python

'''
Read grouped aparc labels from yaml file.
Plot grouped connectivity circle with these grouped labels.
'''

import os.path as op
import sys
import numpy as np
import yaml
import matplotlib.pyplot as pl
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_grouped_connectivity_circle

yaml_fname = get_jumeg_path() + '/examples/aparc_cortex_based_grouping.yaml'

# make a random matrix with 68 nodes
con = np.random.random((68, 68))

plot_grouped_connectivity_circle(yaml_fname, con)
